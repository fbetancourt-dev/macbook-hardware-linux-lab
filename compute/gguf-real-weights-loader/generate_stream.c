#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <omp.h>
#include <CL/cl.h>

#define N_LAYERS 28
#define D_MODEL 1536
#define HEAD_DIM 128
#define N_HEADS_Q 12
#define N_HEADS_KV 2
#define GQA_GROUP_SIZE 6
#define D_QKV 2048
#define D_FFN 8960
#define T_MAX 4096
#define ROPE_BASE 1000000.0f
#define EPSILON 1e-6f
#define PV_SEGMENT_SIZE 256
#define MAX_SEGMENTS (T_MAX / PV_SEGMENT_SIZE)

#define VOCAB_SIZE 151936
#define QK_K 256
#define BLOCKS_PER_ROW (D_MODEL / QK_K) // 6
#define HEAD_BYTES_PER_ROW (BLOCKS_PER_ROW * 210) // 1260

typedef struct {
    uint16_t d;       // IEEE 754 half
    uint8_t qs[16];   // 32 4-bit nibbles
} block_q4_0;

typedef struct __attribute__((packed)) {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d; // IEEE 754 half
} block_q6_k;

#define CHECK_CL(err, msg) do { \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FATAL OpenCL Error at %s:%d: %s (code %d)\n", __FILE__, __LINE__, msg, err); \
        exit(1); \
    } \
} while (0)

static inline double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static inline float fp16_to_float(uint16_t h) {
    return _cvtsh_ss(h);
}

static char* load_kernel_source(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char*)malloc(sz + 1);
    size_t rb = fread(src, 1, sz, f);
    (void)rb;
    src[sz] = '\0';
    fclose(f);
    return src;
}

static void read_exact_at(int fd, uint64_t offset, void *buf, size_t size) {
    uint8_t *p = (uint8_t*)buf;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = pread(fd, p, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("pread");
            exit(1);
        }
        if (n == 0) {
            fprintf(stderr, "Unexpected EOF at offset %lu (need %zu bytes)\n", offset, remaining);
            exit(1);
        }
        remaining -= n;
        p += n;
        offset += n;
    }
}

typedef struct {
    char name[64];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
    uint64_t abs_offset;
} GGUFTensor;

static int parse_gguf_tensors(int fd, GGUFTensor *tensors, int max_tensors, uint64_t *out_data_start) {
    (void)max_tensors;
    lseek(fd, 0, SEEK_SET);
    uint32_t magic, version;
    if (read(fd, &magic, 4) != 4 || read(fd, &version, 4) != 4) return -1;
    uint64_t tensor_count, metadata_kv_count;
    if (read(fd, &tensor_count, 8) != 8 || read(fd, &metadata_kv_count, 8) != 8) return -1;

    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        uint64_t key_len;
        if (read(fd, &key_len, 8) != 8) return -1;
        lseek(fd, key_len, SEEK_CUR);
        uint32_t val_type;
        if (read(fd, &val_type, 4) != 4) return -1;
        if (val_type == 0 || val_type == 1 || val_type == 7) lseek(fd, 1, SEEK_CUR);
        else if (val_type == 2 || val_type == 3) lseek(fd, 2, SEEK_CUR);
        else if (val_type == 4 || val_type == 5 || val_type == 6) lseek(fd, 4, SEEK_CUR);
        else if (val_type == 10 || val_type == 11 || val_type == 12) lseek(fd, 8, SEEK_CUR);
        else if (val_type == 8) {
            uint64_t slen;
            if (read(fd, &slen, 8) != 8) return -1;
            lseek(fd, slen, SEEK_CUR);
        } else if (val_type == 9) {
            uint32_t arr_type;
            uint64_t arr_len;
            if (read(fd, &arr_type, 4) != 4 || read(fd, &arr_len, 8) != 8) return -1;
            if (arr_type == 8) {
                for (uint64_t a = 0; a < arr_len; a++) {
                    uint64_t slen;
                    if (read(fd, &slen, 8) != 8) return -1;
                    lseek(fd, slen, SEEK_CUR);
                }
            } else if (arr_type == 0 || arr_type == 1 || arr_type == 7) lseek(fd, arr_len * 1, SEEK_CUR);
            else if (arr_type == 2 || arr_type == 3) lseek(fd, arr_len * 2, SEEK_CUR);
            else if (arr_type == 4 || arr_type == 5 || arr_type == 6) lseek(fd, arr_len * 4, SEEK_CUR);
            else if (arr_type == 10 || arr_type == 11 || arr_type == 12) lseek(fd, arr_len * 8, SEEK_CUR);
        }
    }

    for (uint64_t i = 0; i < tensor_count; i++) {
        uint64_t nlen;
        if (read(fd, &nlen, 8) != 8) return -1;
        if (nlen >= sizeof(tensors[i].name)) nlen = sizeof(tensors[i].name) - 1;
        if (read(fd, tensors[i].name, nlen) != (ssize_t)nlen) return -1;
        tensors[i].name[nlen] = '\0';

        uint32_t ndims;
        if (read(fd, &ndims, 4) != 4) return -1;
        tensors[i].n_dims = ndims;
        for (uint32_t d = 0; d < ndims; d++) {
            if (read(fd, &tensors[i].ne[d], 8) != 8) return -1;
        }
        if (read(fd, &tensors[i].type, 4) != 4) return -1;
        if (read(fd, &tensors[i].offset, 8) != 8) return -1;
    }

    off_t cur = lseek(fd, 0, SEEK_CUR);
    uint64_t alignment = 32;
    uint64_t data_start = ((uint64_t)cur + alignment - 1) & ~(alignment - 1);
    *out_data_start = data_start;

    for (uint64_t i = 0; i < tensor_count; i++) {
        tensors[i].abs_offset = data_start + tensors[i].offset;
    }
    return (int)tensor_count;
}

static uint64_t get_tensor_offset(const GGUFTensor *tensors, int count, const char *name) {
    for (int i = 0; i < count; i++) {
        if (strcmp(tensors[i].name, name) == 0) return tensors[i].abs_offset;
    }
    fprintf(stderr, "FATAL: Tensor '%s' not found!\n", name);
    exit(1);
}

static void get_token_embedding(int fd, uint64_t embd_base_offset, int token_id, float *x_out) {
    uint64_t token_offset = embd_base_offset + (uint64_t)token_id * 48 * sizeof(block_q4_0);
    block_q4_0 blks[48];
    read_exact_at(fd, token_offset, blks, sizeof(block_q4_0) * 48);

    for (int b = 0; b < 48; b++) {
        float d = fp16_to_float(blks[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t q = blks[b].qs[j];
            x_out[b * 32 + j]      = (float)((int)(q & 0x0F) - 8) * d;
            x_out[b * 32 + j + 16] = (float)((int)(q >>   4) - 8) * d;
        }
    }
}

static char **load_tokenizer_tokens(int fd, int *out_count) {
    off_t orig = lseek(fd, 0, SEEK_CUR);
    lseek(fd, 0, SEEK_SET);

    uint32_t magic, version;
    if (read(fd, &magic, 4) != 4 || read(fd, &version, 4) != 4) { lseek(fd, orig, SEEK_SET); return NULL; }
    uint64_t n_tensors, n_kv;
    if (read(fd, &n_tensors, 8) != 8 || read(fd, &n_kv, 8) != 8) { lseek(fd, orig, SEEK_SET); return NULL; }

    char **tokens = NULL;
    for (uint64_t k = 0; k < n_kv; k++) {
        uint64_t key_len;
        if (read(fd, &key_len, 8) != 8) break;
        char key[256];
        uint64_t to_read = (key_len < 255) ? key_len : 255;
        if (read(fd, key, to_read) != (ssize_t)to_read) break;
        key[to_read] = '\0';
        if (key_len > 255) lseek(fd, key_len - 255, SEEK_CUR);

        uint32_t val_type;
        if (read(fd, &val_type, 4) != 4) break;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            uint32_t arr_type;
            uint64_t arr_len;
            if (read(fd, &arr_type, 4) != 4 || read(fd, &arr_len, 8) != 8) break;
            tokens = (char**)malloc(sizeof(char*) * arr_len);
            for (uint64_t i = 0; i < arr_len; i++) {
                uint64_t slen;
                if (read(fd, &slen, 8) != 8) break;
                tokens[i] = (char*)malloc(slen + 1);
                if (read(fd, tokens[i], slen) != (ssize_t)slen) break;
                tokens[i][slen] = '\0';
            }
            *out_count = (int)arr_len;
            break;
        } else {
            if (val_type == 0 || val_type == 1 || val_type == 7) lseek(fd, 1, SEEK_CUR);
            else if (val_type == 2 || val_type == 3) lseek(fd, 2, SEEK_CUR);
            else if (val_type == 4 || val_type == 5 || val_type == 6) lseek(fd, 4, SEEK_CUR);
            else if (val_type == 10 || val_type == 11 || val_type == 12) lseek(fd, 8, SEEK_CUR);
            else if (val_type == 8) {
                uint64_t slen;
                if (read(fd, &slen, 8) != 8) break;
                lseek(fd, slen, SEEK_CUR);
            } else if (val_type == 9) {
                uint32_t atype; uint64_t alen;
                if (read(fd, &atype, 4) != 4 || read(fd, &alen, 8) != 8) break;
                if (atype == 8) {
                    for (uint64_t a = 0; a < alen; a++) {
                        uint64_t slen;
                        if (read(fd, &slen, 8) != 8) break;
                        lseek(fd, slen, SEEK_CUR);
                    }
                } else if (atype == 0 || atype == 1 || atype == 7) lseek(fd, alen * 1, SEEK_CUR);
                else if (atype == 2 || atype == 3) lseek(fd, alen * 2, SEEK_CUR);
                else if (atype == 4 || atype == 5 || atype == 6) lseek(fd, alen * 4, SEEK_CUR);
                else if (atype == 10 || atype == 11 || atype == 12) lseek(fd, alen * 8, SEEK_CUR);
            }
        }
    }
    lseek(fd, orig, SEEK_SET);
    return tokens;
}

static void print_token_piece(const char *raw_piece) {
    if (!raw_piece) return;
    for (int i = 0; raw_piece[i] != '\0'; ) {
        if ((unsigned char)raw_piece[i] == 0xC4 && (unsigned char)raw_piece[i+1] == 0xA0) {
            putchar(' ');
            i += 2;
        } else if ((unsigned char)raw_piece[i] == 0xC4 && (unsigned char)raw_piece[i+1] == 0x8A) {
            putchar('\n');
            i += 2;
        } else if ((unsigned char)raw_piece[i] == 0xC4 && (unsigned char)raw_piece[i+1] == 0x89) {
            putchar('\t');
            i += 2;
        } else {
            putchar(raw_piece[i]);
            i++;
        }
    }
    fflush(stdout);
}

typedef struct {
    float *gamma_attn;
    float *gamma_ffn;
    float *b_qkv;
    block_q4_0 *W_qkv;
    block_q4_0 *W_o;
    block_q4_0 *W_gate;
    block_q4_0 *W_up;
    block_q4_0 *W_down;
} LayerWeightsHost;

typedef struct {
    cl_kernel k_rmsnorm_attn;
    cl_kernel k_qkv_gemv;
    cl_kernel k_rope_kv;
    cl_kernel k_scores;
    cl_kernel k_softmax;
    cl_kernel k_pv_combine;
    cl_kernel k_pv_reduce;
    cl_kernel k_wo_residual;
    cl_kernel k_rmsnorm_ffn;
    cl_kernel k_swiglu_fused;
    cl_kernel k_down_res;
} LayerKernels;

typedef struct {
    cl_mem W_qkv;
    cl_mem b_qkv;
    cl_mem W_o;
    cl_mem W_gate;
    cl_mem W_up;
    cl_mem W_down;
    cl_mem gamma_attn;
    cl_mem gamma_ffn;
} LayerWeightsDevice;

typedef struct {
    cl_mem k_cache;
    cl_mem v_cache;
} LayerKVDevice;

typedef struct {
    cl_mem state;
    cl_mem norm;
    cl_mem qkv;
    cl_mem scores;
    cl_mem partial;
    cl_mem h;
    cl_mem d_k_sub;
    cl_mem d_v_sub;
} DecoderWorkspaceDevice;

static void gpu_decoder_layer_step(
    cl_command_queue queue,
    LayerKernels *k,
    LayerWeightsDevice *w,
    LayerKVDevice *kv,
    DecoderWorkspaceDevice *ws,
    int pos,
    int seq_len
) {
    int d_model = D_MODEL;
    int d_ffn = D_FFN;
    int param_d_qkv = D_QKV;
    float eps = EPSILON;
    int t_max = T_MAX;
    float rope_base = ROPE_BASE;
    float scale_factor = 1.0f / sqrtf((float)HEAD_DIM);
    int num_segs = (seq_len + PV_SEGMENT_SIZE - 1) / PV_SEGMENT_SIZE;

    clSetKernelArg(k->k_rmsnorm_attn, 0, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_rmsnorm_attn, 1, sizeof(cl_mem), &w->gamma_attn);
    clSetKernelArg(k->k_rmsnorm_attn, 2, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_rmsnorm_attn, 3, sizeof(int), &d_model);
    clSetKernelArg(k->k_rmsnorm_attn, 4, sizeof(float), &eps);

    clSetKernelArg(k->k_qkv_gemv, 0, sizeof(cl_mem), &w->W_qkv);
    clSetKernelArg(k->k_qkv_gemv, 1, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_qkv_gemv, 2, sizeof(cl_mem), &w->b_qkv);
    clSetKernelArg(k->k_qkv_gemv, 3, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_qkv_gemv, 4, sizeof(int), &param_d_qkv);
    clSetKernelArg(k->k_qkv_gemv, 5, sizeof(int), &d_model);

    clSetKernelArg(k->k_rope_kv, 0, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_rope_kv, 1, sizeof(cl_mem), &ws->d_k_sub);
    clSetKernelArg(k->k_rope_kv, 2, sizeof(cl_mem), &ws->d_v_sub);
    clSetKernelArg(k->k_rope_kv, 3, sizeof(cl_mem), &kv->k_cache);
    clSetKernelArg(k->k_rope_kv, 4, sizeof(cl_mem), &kv->v_cache);
    clSetKernelArg(k->k_rope_kv, 5, sizeof(int), &pos);
    clSetKernelArg(k->k_rope_kv, 6, sizeof(int), &t_max);
    clSetKernelArg(k->k_rope_kv, 7, sizeof(float), &rope_base);

    clSetKernelArg(k->k_scores, 0, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_scores, 1, sizeof(cl_mem), &kv->k_cache);
    clSetKernelArg(k->k_scores, 2, sizeof(cl_mem), &ws->scores);
    clSetKernelArg(k->k_scores, 3, sizeof(int), &seq_len);
    clSetKernelArg(k->k_scores, 4, sizeof(int), &t_max);
    clSetKernelArg(k->k_scores, 5, sizeof(float), &scale_factor);

    clSetKernelArg(k->k_softmax, 0, sizeof(cl_mem), &ws->scores);
    clSetKernelArg(k->k_softmax, 1, sizeof(int), &seq_len);
    clSetKernelArg(k->k_softmax, 2, sizeof(int), &t_max);

    clSetKernelArg(k->k_pv_combine, 0, sizeof(cl_mem), &ws->scores);
    clSetKernelArg(k->k_pv_combine, 1, sizeof(cl_mem), &kv->v_cache);
    clSetKernelArg(k->k_pv_combine, 2, sizeof(cl_mem), &ws->partial);
    clSetKernelArg(k->k_pv_combine, 3, sizeof(int), &seq_len);
    clSetKernelArg(k->k_pv_combine, 4, sizeof(int), &t_max);
    clSetKernelArg(k->k_pv_combine, 5, sizeof(int), &num_segs);

    clSetKernelArg(k->k_pv_reduce, 0, sizeof(cl_mem), &ws->partial);
    clSetKernelArg(k->k_pv_reduce, 1, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_pv_reduce, 2, sizeof(int), &num_segs);

    clSetKernelArg(k->k_wo_residual, 0, sizeof(cl_mem), &w->W_o);
    clSetKernelArg(k->k_wo_residual, 1, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_wo_residual, 2, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_wo_residual, 3, sizeof(int), &d_model);

    clSetKernelArg(k->k_rmsnorm_ffn, 0, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_rmsnorm_ffn, 1, sizeof(cl_mem), &w->gamma_ffn);
    clSetKernelArg(k->k_rmsnorm_ffn, 2, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_rmsnorm_ffn, 3, sizeof(int), &d_model);
    clSetKernelArg(k->k_rmsnorm_ffn, 4, sizeof(float), &eps);

    clSetKernelArg(k->k_swiglu_fused, 0, sizeof(cl_mem), &w->W_gate);
    clSetKernelArg(k->k_swiglu_fused, 1, sizeof(cl_mem), &w->W_up);
    clSetKernelArg(k->k_swiglu_fused, 2, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_swiglu_fused, 3, sizeof(cl_mem), &ws->h);
    clSetKernelArg(k->k_swiglu_fused, 4, sizeof(int), &d_ffn);
    clSetKernelArg(k->k_swiglu_fused, 5, sizeof(int), &d_model);

    clSetKernelArg(k->k_down_res, 0, sizeof(cl_mem), &w->W_down);
    clSetKernelArg(k->k_down_res, 1, sizeof(cl_mem), &ws->h);
    clSetKernelArg(k->k_down_res, 2, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_down_res, 3, sizeof(int), &d_model);
    clSetKernelArg(k->k_down_res, 4, sizeof(int), &d_ffn);

    size_t l_rmsnorm = 128, g_rmsnorm = 128;
    size_t l_qkv = 128, g_qkv = ((D_QKV + 3) / 4) * 128;
    size_t l_rope = 64, g_rope = 64;
    size_t l_scores[2] = { 32, 1 };
    size_t g_scores[2] = { (size_t)seq_len * 32, (size_t)N_HEADS_Q };
    size_t l_soft = 128, g_soft = 12 * 128;
    size_t l_pv[2] = { 128, 1 };
    size_t g_pv[2] = { (size_t)num_segs * 128, (size_t)N_HEADS_Q };
    size_t l_red = 128, g_red = 12 * 128;
    size_t l_wo = 128, g_wo = ((D_MODEL + 3) / 4) * 128;
    size_t l_swiglu = 128, g_swiglu = ((D_FFN + 3) / 4) * 128;
    size_t l_down = 128, g_down = ((D_MODEL + 3) / 4) * 128;

    cl_int err;
    err = clEnqueueNDRangeKernel(queue, k->k_rmsnorm_attn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL); CHECK_CL(err, "k_rmsnorm_attn");
    err = clEnqueueNDRangeKernel(queue, k->k_qkv_gemv,     1, NULL, &g_qkv,     &l_qkv,     0, NULL, NULL); CHECK_CL(err, "k_qkv_gemv");
    err = clEnqueueNDRangeKernel(queue, k->k_rope_kv,      1, NULL, &g_rope,    &l_rope,    0, NULL, NULL); CHECK_CL(err, "k_rope_kv");
    err = clEnqueueNDRangeKernel(queue, k->k_scores,       2, NULL, g_scores,   l_scores,   0, NULL, NULL); CHECK_CL(err, "k_scores");
    err = clEnqueueNDRangeKernel(queue, k->k_softmax,      1, NULL, &g_soft,    &l_soft,    0, NULL, NULL); CHECK_CL(err, "k_softmax");
    err = clEnqueueNDRangeKernel(queue, k->k_pv_combine,   2, NULL, g_pv,       l_pv,       0, NULL, NULL); CHECK_CL(err, "k_pv_combine");
    err = clEnqueueNDRangeKernel(queue, k->k_pv_reduce,    1, NULL, &g_red,     &l_red,     0, NULL, NULL); CHECK_CL(err, "k_pv_reduce");
    err = clEnqueueNDRangeKernel(queue, k->k_wo_residual,  1, NULL, &g_wo,      &l_wo,      0, NULL, NULL); CHECK_CL(err, "k_wo_residual");
    err = clEnqueueNDRangeKernel(queue, k->k_rmsnorm_ffn,  1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL); CHECK_CL(err, "k_rmsnorm_ffn");
    err = clEnqueueNDRangeKernel(queue, k->k_swiglu_fused, 1, NULL, &g_swiglu,  &l_swiglu,  0, NULL, NULL); CHECK_CL(err, "k_swiglu_fused");
    err = clEnqueueNDRangeKernel(queue, k->k_down_res,     1, NULL, &g_down,    &l_down,    0, NULL, NULL); CHECK_CL(err, "k_down_res");
}

int main(int argc, char **argv) {
    int max_new_tokens = 24;
    bool quiet = false;
    bool no_echo_prompt = false;
    bool test_nan = false;
    int *prompt_tokens = NULL;
    int n_prompt = 0;

    int default_prompt[] = { 750, 912, 2877, 11, 293, 982, 262, 470, 220 };
    int n_default = sizeof(default_prompt) / sizeof(default_prompt[0]);

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
            if (max_new_tokens < 0) max_new_tokens = 0;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            char *token_str = strdup(argv[++i]);
            int count = 0;
            char *tmp = strdup(token_str);
            char *tok = strtok(tmp, ", \t\n");
            while (tok) { count++; tok = strtok(NULL, ", \t\n"); }
            free(tmp);
            if (count > 0) {
                prompt_tokens = (int*)malloc(sizeof(int) * count);
                n_prompt = 0;
                tok = strtok(token_str, ", \t\n");
                while (tok) {
                    prompt_tokens[n_prompt++] = atoi(tok);
                    tok = strtok(NULL, ", \t\n");
                }
            }
            free(token_str);
        } else if (strcmp(argv[i], "--tokens-file") == 0 && i + 1 < argc) {
            FILE *tf = fopen(argv[++i], "r");
            if (tf) {
                int cap = 1024;
                prompt_tokens = (int*)malloc(sizeof(int) * cap);
                n_prompt = 0;
                int val;
                while (fscanf(tf, "%d", &val) == 1 || fscanf(tf, ",%d", &val) == 1) {
                    if (n_prompt >= cap) {
                        cap *= 2;
                        prompt_tokens = (int*)realloc(prompt_tokens, sizeof(int) * cap);
                    }
                    prompt_tokens[n_prompt++] = val;
                }
                fclose(tf);
            }
        } else if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) {
            quiet = true;
        } else if (strcmp(argv[i], "--no-echo-prompt") == 0) {
            no_echo_prompt = true;
        } else if (strcmp(argv[i], "--test-nan") == 0) {
            test_nan = true;
        } else if (argv[i][0] != '-') {
            max_new_tokens = atoi(argv[i]);
            if (max_new_tokens < 0) max_new_tokens = 0;
        }
    }

    if (!prompt_tokens || n_prompt == 0) {
        prompt_tokens = (int*)malloc(sizeof(default_prompt));
        memcpy(prompt_tokens, default_prompt, sizeof(default_prompt));
        n_prompt = n_default;
    }

    if (n_prompt + max_new_tokens > T_MAX) {
        if (!quiet) fprintf(stderr, "[Warning: Clamping max_new_tokens from %d to %d to fit T_MAX=%d]\n",
                            max_new_tokens, T_MAX - n_prompt, T_MAX);
        max_new_tokens = T_MAX - n_prompt;
        if (max_new_tokens < 0) max_new_tokens = 0;
    }

    if (!quiet) {
        fprintf(stderr, "========================================================================================\n");
        fprintf(stderr, " Phase C: Real-Time Streaming Autoregressive Generator (Qwen2.5-Coder-1.5B)            \n");
        fprintf(stderr, " Direct Hardware Execution on GT 750M (Kepler OpenCL 3.0 via Mesa Rusticl)             \n");
        fprintf(stderr, "========================================================================================\n");
    }

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("open GGUF"); exit(1); }

    GGUFTensor tensors[512];
    uint64_t data_start = 0;
    int tensor_count = parse_gguf_tensors(fd, tensors, 512, &data_start);
    if (tensor_count <= 0) { fprintf(stderr, "Failed to parse GGUF\n"); exit(1); }
    if (!quiet) fprintf(stderr, "GGUF Indexer: Indexed %d tensor descriptors.\n", tensor_count);

    int n_vocab_tokens = 0;
    char **vocab_tokens = load_tokenizer_tokens(fd, &n_vocab_tokens);
    if (!quiet) fprintf(stderr, "Vocabulary: Loaded %d token strings from GGUF.\n", n_vocab_tokens);

    uint64_t off_embd = get_tensor_offset(tensors, tensor_count, "token_embd.weight");
    uint64_t off_head = get_tensor_offset(tensors, tensor_count, "output.weight");
    uint64_t off_norm = get_tensor_offset(tensors, tensor_count, "output_norm.weight");

    int nb_qkv = D_MODEL / 32, nb_wo = D_MODEL / 32, nb_gate = D_MODEL / 32, nb_down = D_FFN / 32;
    size_t sz_W_qkv  = sizeof(block_q4_0) * D_QKV * nb_qkv;
    size_t sz_W_o    = sizeof(block_q4_0) * D_MODEL * nb_wo;
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * nb_down;
    size_t sz_kv     = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;

    LayerWeightsHost w_host[N_LAYERS];

    if (!quiet) fprintf(stderr, "Reading weights for all %d layers from GGUF...\n", N_LAYERS);
    double t_load_start = get_time_us();

    for (int l = 0; l < N_LAYERS; l++) {
        w_host[l].gamma_attn = (float*)malloc(sizeof(float) * D_MODEL);
        w_host[l].gamma_ffn  = (float*)malloc(sizeof(float) * D_MODEL);
        w_host[l].b_qkv      = (float*)malloc(sizeof(float) * D_QKV);
        w_host[l].W_qkv      = (block_q4_0*)malloc(sz_W_qkv);
        w_host[l].W_o        = (block_q4_0*)malloc(sz_W_o);
        w_host[l].W_gate     = (block_q4_0*)malloc(sz_W_gate);
        w_host[l].W_up       = (block_q4_0*)malloc(sz_W_gate);
        w_host[l].W_down     = (block_q4_0*)malloc(sz_W_down);

        char name[64];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].gamma_attn, sizeof(float) * D_MODEL);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].gamma_ffn, sizeof(float) * D_MODEL);

        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].b_qkv, sizeof(float) * 1536);

        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].b_qkv + 1536, sizeof(float) * 256);

        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].b_qkv + 1536 + 256, sizeof(float) * 256);

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_qkv, sizeof(block_q4_0) * 1536 * 48);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_qkv + 1536 * 48, sizeof(block_q4_0) * 256 * 48);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_qkv + (1536 + 256) * 48, sizeof(block_q4_0) * 256 * 48);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_o, sz_W_o);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_gate, sz_W_gate);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_up, sz_W_gate);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_down, sz_W_down);
    }

    float gamma_output_norm[D_MODEL];
    read_exact_at(fd, off_norm, gamma_output_norm, sizeof(float) * D_MODEL);

    size_t sz_head = (size_t)VOCAB_SIZE * HEAD_BYTES_PER_ROW;
    uint8_t *h_head = (uint8_t*)malloc(sz_head);
    read_exact_at(fd, off_head, h_head, sz_head);

    double t_load_end = get_time_us();
    if (!quiet) fprintf(stderr, "All weights read from disk in %.2f ms!\n\n", (t_load_end - t_load_start) / 1000.0);

    // OpenCL Setup
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms, num_devices;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, &num_platforms); CHECK_CL(err, "platform");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices); CHECK_CL(err, "device");

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    if (!quiet) fprintf(stderr, "Active Compute Engine: %s (OpenCL 3.0 via Rusticl)\n", dev_name);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_CL(err, "context");
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); CHECK_CL(err, "queue");

    char *src_dec = load_kernel_source("kernel_decoder_layer.cl");
    cl_program prog_dec = clCreateProgramWithSource(context, 1, (const char**)&src_dec, NULL, &err); CHECK_CL(err, "prog_dec");
    free(src_dec);
    err = clBuildProgram(prog_dec, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build dec");

    char *src_q6 = load_kernel_source("kernel_q6_k.cl");
    cl_program prog_q6 = clCreateProgramWithSource(context, 1, (const char**)&src_q6, NULL, &err); CHECK_CL(err, "prog_q6");
    free(src_q6);
    err = clBuildProgram(prog_q6, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build q6");

    LayerKernels kernels[N_LAYERS];
    for (int l = 0; l < N_LAYERS; l++) {
        kernels[l].k_rmsnorm_attn = clCreateKernel(prog_dec, "kernel_rmsnorm", &err);
        kernels[l].k_qkv_gemv     = clCreateKernel(prog_dec, "gemv_q4_0_bias", &err);
        kernels[l].k_rope_kv      = clCreateKernel(prog_dec, "kernel_rope_and_kv_append", &err);
        kernels[l].k_scores       = clCreateKernel(prog_dec, "kernel_gqa_scores", &err);
        kernels[l].k_softmax      = clCreateKernel(prog_dec, "kernel_softmax_gqa", &err);
        kernels[l].k_pv_combine   = clCreateKernel(prog_dec, "kernel_gqa_value_combine_segmented", &err);
        kernels[l].k_pv_reduce    = clCreateKernel(prog_dec, "kernel_gqa_reduce_segments", &err);
        kernels[l].k_wo_residual  = clCreateKernel(prog_dec, "gemv_q4_0_wo_residual", &err);
        kernels[l].k_rmsnorm_ffn  = clCreateKernel(prog_dec, "kernel_rmsnorm", &err);
        kernels[l].k_swiglu_fused = clCreateKernel(prog_dec, "gemv_swiglu_fused", &err);
        kernels[l].k_down_res     = clCreateKernel(prog_dec, "gemv_q4_0_down_residual", &err);
    }
    cl_kernel k_output_norm = clCreateKernel(prog_dec, "kernel_rmsnorm", &err); CHECK_CL(err, "k_output_norm");
    cl_kernel k_lm_head     = clCreateKernel(prog_q6, "gemv_q6_k", &err); CHECK_CL(err, "k_lm_head");

    if (!quiet) fprintf(stderr, "Transferring 28 layers + Output Norm + Full LM Head to GT 750M VRAM...\n");
    double t_vram_start = get_time_us();

    LayerWeightsDevice w_dev[N_LAYERS];
    LayerKVDevice kv_dev[N_LAYERS];

    for (int l = 0; l < N_LAYERS; l++) {
        w_dev[l].W_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_qkv, w_host[l].W_qkv, &err);
        w_dev[l].b_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*D_QKV, w_host[l].b_qkv, &err);
        w_dev[l].W_o        = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_o, w_host[l].W_o, &err);
        w_dev[l].W_gate     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_gate, &err);
        w_dev[l].W_up       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_up, &err);
        w_dev[l].W_down     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_down, w_host[l].W_down, &err);
        w_dev[l].gamma_attn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*D_MODEL, w_host[l].gamma_attn, &err);
        w_dev[l].gamma_ffn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*D_MODEL, w_host[l].gamma_ffn, &err);

        kv_dev[l].k_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err);
        kv_dev[l].v_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err);
    }

    cl_mem d_gamma_output_norm = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, gamma_output_norm, &err);
    cl_mem d_final_norm        = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err);
    cl_mem d_W_head = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_head, h_head, &err); CHECK_CL(err, "d_W_head");
    cl_mem d_logits = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * VOCAB_SIZE, NULL, &err); CHECK_CL(err, "d_logits");
    free(h_head);

    DecoderWorkspaceDevice ws;
    ws.state    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_MODEL, NULL, &err);
    ws.norm     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_MODEL, NULL, &err);
    ws.qkv      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_QKV, NULL, &err);
    ws.scores   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N_HEADS_Q*T_MAX, NULL, &err);
    ws.partial  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N_HEADS_Q*MAX_SEGMENTS*HEAD_DIM, NULL, &err);
    ws.h        = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_FFN, NULL, &err);

    size_t k_offset_bytes = 1536 * sizeof(float);
    size_t v_offset_bytes = (1536 + 256) * sizeof(float);
    cl_buffer_region reg_k = { k_offset_bytes, 256 * sizeof(float) };
    cl_buffer_region reg_v = { v_offset_bytes, 256 * sizeof(float) };
    ws.d_k_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_k, &err);
    ws.d_v_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_v, &err);

    float zero_val = 0.0f;
    for (int l = 0; l < N_LAYERS; l++) {
        clEnqueueFillBuffer(queue, kv_dev[l].k_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, kv_dev[l].v_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
    }
    clFinish(queue);

    double t_vram_end = get_time_us();
    if (!quiet) fprintf(stderr, "Entire 1.5B Parameter Model (1110 MB) Resident in GT 750M VRAM in %.2f ms!\n\n", (t_vram_end - t_vram_start) / 1000.0);

    // Free host weight buffers to reclaim RAM
    for (int l = 0; l < N_LAYERS; l++) {
        free(w_host[l].gamma_attn);
        free(w_host[l].gamma_ffn);
        free(w_host[l].b_qkv);
        free(w_host[l].W_qkv);
        free(w_host[l].W_o);
        free(w_host[l].W_gate);
        free(w_host[l].W_up);
        free(w_host[l].W_down);
    }

    if (!quiet) {
        fprintf(stderr, "========================================================================================\n");
        fprintf(stderr, " STAGE 1: MULTI-TOKEN PROMPT PREFILL (%d TOKENS)\n", n_prompt);
        fprintf(stderr, "========================================================================================\n");
    }

    // Print Prompt to stdout if not suppressed
    if (!no_echo_prompt) {
        for (int i = 0; i < n_prompt; i++) {
            print_token_piece(vocab_tokens[prompt_tokens[i]]);
        }
        fflush(stdout);
    }

    float *h_logits = (float*)malloc(sizeof(float) * VOCAB_SIZE);
    float h_embd[D_MODEL];

    // Prefill tokens 0 .. n_prompt - 1
    double t_prefill_start = get_time_us();
    for (int p = 0; p < n_prompt; p++) {
        int tok = prompt_tokens[p];
        get_token_embedding(fd, off_embd, tok, h_embd);

        double t_step_0 = get_time_us();
        clEnqueueWriteBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_embd, 0, NULL, NULL);

        for (int l = 0; l < N_LAYERS; l++) {
            gpu_decoder_layer_step(queue, &kernels[l], &w_dev[l], &kv_dev[l], &ws, p, p + 1);
        }

        bool is_last = (p == n_prompt - 1);
        if (is_last) {
            int D = D_MODEL;
            float eps = EPSILON;
            clSetKernelArg(k_output_norm, 0, sizeof(cl_mem), &ws.state);
            clSetKernelArg(k_output_norm, 1, sizeof(cl_mem), &d_gamma_output_norm);
            clSetKernelArg(k_output_norm, 2, sizeof(cl_mem), &d_final_norm);
            clSetKernelArg(k_output_norm, 3, sizeof(int), &D);
            clSetKernelArg(k_output_norm, 4, sizeof(float), &eps);
            size_t g_norm = 128, l_norm = 128;
            clEnqueueNDRangeKernel(queue, k_output_norm, 1, NULL, &g_norm, &l_norm, 0, NULL, NULL);

            int M_vocab = VOCAB_SIZE;
            int K_dim = D_MODEL;
            clSetKernelArg(k_lm_head, 0, sizeof(cl_mem), &d_W_head);
            clSetKernelArg(k_lm_head, 1, sizeof(cl_mem), &d_final_norm);
            clSetKernelArg(k_lm_head, 2, sizeof(cl_mem), &d_logits);
            clSetKernelArg(k_lm_head, 3, sizeof(int), &M_vocab);
            clSetKernelArg(k_lm_head, 4, sizeof(int), &K_dim);
            size_t l_head = 128;
            size_t g_head = ((VOCAB_SIZE + 3) / 4) * 128;
            clEnqueueNDRangeKernel(queue, k_lm_head, 1, NULL, &g_head, &l_head, 0, NULL, NULL);

            clEnqueueReadBuffer(queue, d_logits, CL_TRUE, 0, sizeof(float) * VOCAB_SIZE, h_logits, 0, NULL, NULL);
        } else {
            clFinish(queue);
        }
        double t_step_1 = get_time_us();
        if (!quiet) {
            fprintf(stderr, "[Prefill %d/%d | pos=%d | tok=%-5d '%s' | lat=%.2f ms]\n",
                    p + 1, n_prompt, p, tok, vocab_tokens[tok], (t_step_1 - t_step_0) / 1000.0);
        }
    }
    double t_prefill_end = get_time_us();
    if (!quiet) {
        fprintf(stderr, "\nPrefill Completed in %.2f ms (TTFT: %.2f ms)\n\n",
                (t_prefill_end - t_prefill_start) / 1000.0, (t_prefill_end - t_prefill_start) / 1000.0);

        fprintf(stderr, "========================================================================================\n");
        fprintf(stderr, " STAGE 2: AUTOREGRESSIVE STREAMING GENERATION (GREEDY ARGMAX)\n");
        fprintf(stderr, "========================================================================================\n");
    }

    int next_input_pos = n_prompt;
    double total_gen_time_ms = 0.0;
    int generated_count = 0;
    int decode_forward_count = 0;

    for (int gen = 0; gen < max_new_tokens; gen++) {
        if (test_nan && gen == 0) {
            h_logits[42] = 0.0f / 0.0f; // Test NaN injection
        }
        int best_tok = 0;
        float best_val = -1e30f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (!isfinite(h_logits[i])) {
                fprintf(stderr, "\nFATAL: Non-finite logit at gen=%d, pos=%d, token_id=%d (val=%f)\n",
                        gen, next_input_pos, i, h_logits[i]);
                exit(1);
            }
            if (h_logits[i] > best_val) {
                best_val = h_logits[i];
                best_tok = i;
            }
        }

        // Check EOG (151643: <|endoftext|>, 151645: <|im_end|>)
        if (best_tok == 151643 || best_tok == 151645) {
            fprintf(stderr, "\n[EOG Token %d reached: Stopping Generation]\n", best_tok);
            break;
        }

        print_token_piece(vocab_tokens[best_tok]);
        generated_count++;

        if (generated_count >= max_new_tokens) {
            break;
        }
        if (next_input_pos >= T_MAX) {
            fprintf(stderr, "\n[Notice: Reached max context length %d]\n", T_MAX);
            break;
        }

        double t_step_0 = get_time_us();
        get_token_embedding(fd, off_embd, best_tok, h_embd);
        err = clEnqueueWriteBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_embd, 0, NULL, NULL);
        CHECK_CL(err, "write ws.state");

        for (int l = 0; l < N_LAYERS; l++) {
            gpu_decoder_layer_step(queue, &kernels[l], &w_dev[l], &kv_dev[l], &ws, next_input_pos, next_input_pos + 1);
        }

        int D = D_MODEL;
        float eps = EPSILON;
        clSetKernelArg(k_output_norm, 0, sizeof(cl_mem), &ws.state);
        clSetKernelArg(k_output_norm, 1, sizeof(cl_mem), &d_gamma_output_norm);
        clSetKernelArg(k_output_norm, 2, sizeof(cl_mem), &d_final_norm);
        clSetKernelArg(k_output_norm, 3, sizeof(int), &D);
        clSetKernelArg(k_output_norm, 4, sizeof(float), &eps);
        size_t g_norm = 128, l_norm = 128;
        err = clEnqueueNDRangeKernel(queue, k_output_norm, 1, NULL, &g_norm, &l_norm, 0, NULL, NULL);
        CHECK_CL(err, "k_output_norm");

        int M_vocab = VOCAB_SIZE;
        int K_dim = D_MODEL;
        clSetKernelArg(k_lm_head, 0, sizeof(cl_mem), &d_W_head);
        clSetKernelArg(k_lm_head, 1, sizeof(cl_mem), &d_final_norm);
        clSetKernelArg(k_lm_head, 2, sizeof(cl_mem), &d_logits);
        clSetKernelArg(k_lm_head, 3, sizeof(int), &M_vocab);
        clSetKernelArg(k_lm_head, 4, sizeof(int), &K_dim);
        size_t l_head = 128;
        size_t g_head = ((VOCAB_SIZE + 3) / 4) * 128;
        err = clEnqueueNDRangeKernel(queue, k_lm_head, 1, NULL, &g_head, &l_head, 0, NULL, NULL);
        CHECK_CL(err, "k_lm_head");

        err = clEnqueueReadBuffer(queue, d_logits, CL_TRUE, 0, sizeof(float) * VOCAB_SIZE, h_logits, 0, NULL, NULL);
        CHECK_CL(err, "read d_logits");
        double t_step_1 = get_time_us();

        double step_ms = (t_step_1 - t_step_0) / 1000.0;
        total_gen_time_ms += step_ms;
        decode_forward_count++;
        if (!quiet) {
            fprintf(stderr, "[Gen %2d | pos=%2d | tok=%-5d '%-8s' | logit=%7.4f | lat=%6.2f ms]\n",
                    gen + 1, next_input_pos, best_tok, vocab_tokens[best_tok], best_val, step_ms);
        }
        next_input_pos++;
    }


    if (!quiet) {
        fprintf(stderr, "\n========================================================================================\n");
        fprintf(stderr, " GENERATION BENCHMARK SUMMARY (GT 750M Physical Hardware):\n");
        fprintf(stderr, "========================================================================================\n");
        fprintf(stderr, "  Prompt Tokens Processed:      %d tokens\n", n_prompt);
        fprintf(stderr, "  Generated Tokens:             %d tokens\n", generated_count);
        fprintf(stderr, "  Decode Forward Passes:        %d passes\n", decode_forward_count);
        if (decode_forward_count > 0) {
            fprintf(stderr, "  Mean Decode Forward Latency:  %.2f ms/pass\n", total_gen_time_ms / decode_forward_count);
            fprintf(stderr, "  Decode Forward Rate:          %.2f passes/sec (t/s)\n", (decode_forward_count * 1000.0) / total_gen_time_ms);
        } else {
            fprintf(stderr, "  Mean Decode Forward Latency:  N/A (no decode forward executed)\n");
            fprintf(stderr, "  Decode Forward Rate:          N/A\n");
        }
        fprintf(stderr, "========================================================================================\n\n");
    }

    printf("\n");
    fflush(stdout);

    free(prompt_tokens);
    close(fd);
    return 0;
}
