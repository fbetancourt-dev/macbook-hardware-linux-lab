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
#include <signal.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
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

#define SOCKET_PATH "/tmp/qwen.sock"

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

static void write_token_piece(int client_fd, const char *raw_piece) {
    if (!raw_piece || client_fd < 0) return;
    char buf[512];
    int out_len = 0;
    for (int i = 0; raw_piece[i] != '\0' && out_len < 500; ) {
        if ((unsigned char)raw_piece[i] == 0xC4 && (unsigned char)raw_piece[i+1] == 0xA0) {
            buf[out_len++] = ' ';
            i += 2;
        } else if ((unsigned char)raw_piece[i] == 0xC4 && (unsigned char)raw_piece[i+1] == 0x8A) {
            buf[out_len++] = '\n';
            i += 2;
        } else if ((unsigned char)raw_piece[i] == 0xC4 && (unsigned char)raw_piece[i+1] == 0x89) {
            buf[out_len++] = '\t';
            i += 2;
        } else {
            buf[out_len++] = raw_piece[i++];
        }
    }
    if (out_len > 0) {
        ssize_t w = write(client_fd, buf, out_len);
        (void)w;
    }
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
    cl_mem gamma_attn;
    cl_mem gamma_ffn;
    cl_mem b_qkv;
    cl_mem W_qkv;
    cl_mem W_o;
    cl_mem W_gate;
    cl_mem W_up;
    cl_mem W_down;
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
    int param_d_qkv = D_QKV;
    int d_ffn = D_FFN;
    int t_max = T_MAX;
    float eps = EPSILON;
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

static char g_active_sock[512] = SOCKET_PATH;
static bool g_created_symlink = false;
static volatile sig_atomic_t g_is_busy = 0;
static uint64_t g_base_hash = 0;
static int base_pos = 0;
static int base_generation = 0;
static const char *memory_state = "UNINITIALIZED";
static const char *g_model_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";

static void clean_exit_handler(int sig) {
    (void)sig;
    unlink(g_active_sock);
    if (g_created_symlink) {
        unlink("/tmp/qwen.sock");
    }
    _exit(0);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, clean_exit_handler);
    signal(SIGTERM, clean_exit_handler);

    const char *xdg = getenv("XDG_RUNTIME_DIR");
    if (xdg && access(xdg, W_OK) == 0) {
        snprintf(g_active_sock, sizeof(g_active_sock), "%s/qwen.sock", xdg);
    } else {
        strncpy(g_active_sock, SOCKET_PATH, sizeof(g_active_sock) - 1);
    }

    fprintf(stderr, "========================================================================================\n");
    fprintf(stderr, " qwen-server: Persistent Daemon on NVIDIA GT 750M (Kepler OpenCL 3.0 via Rusticl)       \n");
    fprintf(stderr, " Listening on Unix Domain Socket: %s\n", g_active_sock);
    fprintf(stderr, "========================================================================================\n");

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("open GGUF"); exit(1); }

    GGUFTensor tensors[512];
    uint64_t data_start = 0;
    int tensor_count = parse_gguf_tensors(fd, tensors, 512, &data_start);
    if (tensor_count <= 0) { fprintf(stderr, "Failed to parse GGUF\n"); exit(1); }

    int n_vocab_tokens = 0;
    char **vocab_tokens = load_tokenizer_tokens(fd, &n_vocab_tokens);
    fprintf(stderr, "Vocabulary: Loaded %d token strings from GGUF.\n", n_vocab_tokens);

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
    fprintf(stderr, "Reading weights for all %d layers from GGUF...\n", N_LAYERS);

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

    // OpenCL Setup
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms, num_devices;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, &num_platforms); CHECK_CL(err, "platform");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices); CHECK_CL(err, "device");

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    fprintf(stderr, "Active Compute Engine: %s (OpenCL 3.0 via Rusticl)\n", dev_name);

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
        kernels[l].k_rmsnorm_attn = clCreateKernel(prog_dec, "kernel_rmsnorm", &err); CHECK_CL(err, "k_rmsnorm_attn");
        kernels[l].k_qkv_gemv     = clCreateKernel(prog_dec, "gemv_q4_0_bias", &err); CHECK_CL(err, "k_qkv_gemv");
        kernels[l].k_rope_kv      = clCreateKernel(prog_dec, "kernel_rope_and_kv_append", &err); CHECK_CL(err, "k_rope_kv");
        kernels[l].k_scores       = clCreateKernel(prog_dec, "kernel_gqa_scores", &err); CHECK_CL(err, "k_scores");
        kernels[l].k_softmax      = clCreateKernel(prog_dec, "kernel_softmax_gqa", &err); CHECK_CL(err, "k_softmax");
        kernels[l].k_pv_combine   = clCreateKernel(prog_dec, "kernel_gqa_value_combine_segmented", &err); CHECK_CL(err, "k_pv_combine");
        kernels[l].k_pv_reduce    = clCreateKernel(prog_dec, "kernel_gqa_reduce_segments", &err); CHECK_CL(err, "k_pv_reduce");
        kernels[l].k_wo_residual  = clCreateKernel(prog_dec, "gemv_q4_0_wo_residual", &err); CHECK_CL(err, "k_wo_residual");
        kernels[l].k_rmsnorm_ffn  = clCreateKernel(prog_dec, "kernel_rmsnorm", &err); CHECK_CL(err, "k_rmsnorm_ffn");
        kernels[l].k_swiglu_fused = clCreateKernel(prog_dec, "gemv_swiglu_fused", &err); CHECK_CL(err, "k_swiglu_fused");
        kernels[l].k_down_res     = clCreateKernel(prog_dec, "gemv_q4_0_down_residual", &err); CHECK_CL(err, "k_down_res");
    }
    cl_kernel k_output_norm = clCreateKernel(prog_dec, "kernel_rmsnorm", &err); CHECK_CL(err, "k_output_norm");
    cl_kernel k_lm_head     = clCreateKernel(prog_q6, "gemv_q6_k", &err); CHECK_CL(err, "k_lm_head");

    fprintf(stderr, "Transferring 28 layers + Output Norm + Full LM Head to GT 750M VRAM...\n");
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

    // Free host weights
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

    fprintf(stderr, "Entire 1.5B Parameter Model Resident in GT 750M VRAM!\n");

    // Unix Domain Socket Server Setup
    unlink(g_active_sock);
    int server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock < 0) { perror("socket"); exit(1); }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_active_sock, sizeof(addr.sun_path) - 1);

    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); exit(1);
    }
    chmod(g_active_sock, 0600);

    if (strcmp(g_active_sock, "/tmp/qwen.sock") != 0) {
        unlink("/tmp/qwen.sock");
        if (symlink(g_active_sock, "/tmp/qwen.sock") == 0) {
            g_created_symlink = true;
        }
    }

    if (listen(server_sock, 5) < 0) {
        perror("listen"); exit(1);
    }

    fprintf(stderr, "Ready for queries on %s (symlink: /tmp/qwen.sock)\n\n", g_active_sock);

    float *h_logits = (float*)malloc(sizeof(float) * VOCAB_SIZE);
    float h_embd[D_MODEL];
    char req_buf[65536];

    while (1) {
        int client_sock = accept(server_sock, NULL, NULL);
        if (client_sock < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        if (g_is_busy) {
            const char *busy_resp = "ERR_BUSY\n";
            write(client_sock, busy_resp, strlen(busy_resp));
            close(client_sock);
            continue;
        }

        // Read request line until '\n'
        size_t total_read = 0;
        while (total_read < sizeof(req_buf) - 1) {
            ssize_t n = read(client_sock, req_buf + total_read, 1);
            if (n <= 0) break;
            if (req_buf[total_read] == '\n') {
                req_buf[total_read] = '\0';
                break;
            }
            total_read++;
        }
        if (total_read == 0) { close(client_sock); continue; }
        req_buf[total_read] = '\0';

        char cmd[32];
        if (sscanf(req_buf, "%31s", cmd) != 1) { close(client_sock); continue; }

        if (strcmp(cmd, "PING") == 0) {
            const char *resp = "PONG\n";
            write(client_sock, resp, strlen(resp));
            close(client_sock);
            continue;
        }

        if (strcmp(cmd, "STATUS") == 0) {
            char status_resp[512];
            snprintf(status_resp, sizeof(status_resp),
                     "STATUS OK model=%s t_max=%d base_pos=%d base_hash=0x%016lx base_generation=%d memory_state=%s vram_mb=1110 (weights_mb=850 kv_cache_mb=224 workspace_mb=36)\n",
                     g_model_path, T_MAX, base_pos, g_base_hash, base_generation, memory_state);
            write(client_sock, status_resp, strlen(status_resp));
            close(client_sock);
            continue;
        }

        if (strcmp(cmd, "SET_BASE") == 0) {
            char *ptr = req_buf + 8;
            int n_tokens = 0;
            int *tokens = (int*)malloc(sizeof(int) * 4096);
            char *tok = strtok(ptr, ", \t\n");
            while (tok && n_tokens < 4096) {
                tokens[n_tokens++] = atoi(tok);
                tok = strtok(NULL, ", \t\n");
            }

            // Validate capacity BEFORE changing state or modifying KV cache
            if (n_tokens + 1 >= T_MAX) {
                const char *err_resp = "ERR_CONTEXT_FULL\n";
                write(client_sock, err_resp, strlen(err_resp));
                close(client_sock);
                free(tokens);
                continue;
            }

            g_is_busy = 1;
            memory_state = "REBUILDING";

            fprintf(stderr, "[qwen-server] Freezing %d base system tokens into KV cache (generation %d)...\n", n_tokens, base_generation + 1);
            uint64_t hash = 14695981039346656037ULL;
            for (int p = 0; p < n_tokens; p++) {
                hash ^= (uint64_t)tokens[p];
                hash *= 1099511628211ULL;

                get_token_embedding(fd, off_embd, tokens[p], h_embd);
                clEnqueueWriteBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float)*D_MODEL, h_embd, 0, NULL, NULL);
                for (int l = 0; l < N_LAYERS; l++) {
                    gpu_decoder_layer_step(queue, &kernels[l], &w_dev[l], &kv_dev[l], &ws, p, p + 1);
                }
            }
            clFinish(queue);
            base_pos = n_tokens;
            g_base_hash = hash;
            base_generation++;
            memory_state = "READY";
            free(tokens);
            g_is_busy = 0;

            char resp[160];
            snprintf(resp, sizeof(resp), "OK base_pos=%d base_hash=0x%016lx base_generation=%d memory_state=READY\n",
                     base_pos, g_base_hash, base_generation);
            write(client_sock, resp, strlen(resp));
            close(client_sock);
            fprintf(stderr, "[qwen-server] Base prompt frozen at pos=%d (hash=0x%016lx, gen=%d). Subsequent queries will start from here!\n",
                    base_pos, g_base_hash, base_generation);
            continue;
        }

        if (strcmp(cmd, "QUERY") == 0 || strcmp(cmd, "RAW_QUERY") == 0) {
            bool is_raw = (strcmp(cmd, "RAW_QUERY") == 0);

            // Protect frozen base KV cache from RAW_QUERY
            if (is_raw && base_pos > 0 && strcmp(memory_state, "READY") == 0) {
                const char *err_raw = "ERR_RAW_NOT_ALLOWED_WHEN_BASE_ACTIVE\n";
                write(client_sock, err_raw, strlen(err_raw));
                close(client_sock);
                continue;
            }

            if (!is_raw && strcmp(memory_state, "READY") != 0) {
                const char *err_not_ready = "ERR_NOT_READY\n";
                write(client_sock, err_not_ready, strlen(err_not_ready));
                close(client_sock);
                continue;
            }
            int cur_pos = is_raw ? 0 : base_pos;
            int max_new_tokens = 32;

            char *line_ptr = req_buf + strlen(cmd);
            int tok_count = 0;
            int *q_tokens = (int*)malloc(sizeof(int) * 4096);

            // format: QUERY <max_new_tokens> <id1,id2,...>
            char *token_str = strtok(line_ptr, " \t\n");
            if (token_str) {
                max_new_tokens = atoi(token_str);
                token_str = strtok(NULL, " \t\n");
                if (token_str) {
                    char *sub = strtok(token_str, ",");
                    while (sub && tok_count < 4096) {
                        q_tokens[tok_count++] = atoi(sub);
                        sub = strtok(NULL, ",");
                    }
                }
            }

            // Boundary check: must have at least 1 token safety margin
            if (cur_pos + tok_count + max_new_tokens + 1 > T_MAX) {
                fprintf(stderr, "[qwen-server] ERR_CONTEXT_FULL: cur_pos(%d) + tok_count(%d) + max_new(%d) + 1 > T_MAX(%d)\n",
                        cur_pos, tok_count, max_new_tokens, T_MAX);
                const char *err_resp = "ERR_CONTEXT_FULL\n";
                write(client_sock, err_resp, strlen(err_resp));
                close(client_sock);
                free(q_tokens);
                continue;
            }

            g_is_busy = 1;
            fprintf(stderr, "[qwen-server] Query received: %d tokens (base_pos=%d, max_new=%d)\n",
                    tok_count, cur_pos, max_new_tokens);

            // Prefill user tokens
            for (int p = 0; p < tok_count; p++) {
                int tok = q_tokens[p];
                get_token_embedding(fd, off_embd, tok, h_embd);
                clEnqueueWriteBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_embd, 0, NULL, NULL);

                for (int l = 0; l < N_LAYERS; l++) {
                    gpu_decoder_layer_step(queue, &kernels[l], &w_dev[l], &kv_dev[l], &ws, cur_pos + p, cur_pos + p + 1);
                }

                if (p == tok_count - 1) {
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
                }
            }

            int next_input_pos = cur_pos + tok_count;

            // Autoregressive generation streaming directly into client_sock
            for (int gen = 0; gen < max_new_tokens; gen++) {
                int best_tok = 0;
                float best_val = -1e30f;
                for (int i = 0; i < VOCAB_SIZE; i++) {
                    if (h_logits[i] > best_val) {
                        best_val = h_logits[i];
                        best_tok = i;
                    }
                }

                // Check EOG (151643: <|endoftext|>, 151645: <|im_end|>)
                if (best_tok == 151643 || best_tok == 151645) {
                    break;
                }

                write_token_piece(client_sock, vocab_tokens[best_tok]);

                if (gen + 1 >= max_new_tokens || next_input_pos >= T_MAX) {
                    break;
                }

                get_token_embedding(fd, off_embd, best_tok, h_embd);
                clEnqueueWriteBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_embd, 0, NULL, NULL);

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
                next_input_pos++;
            }

            free(q_tokens);
            close(client_sock);
            g_is_busy = 0;
            fprintf(stderr, "[qwen-server] Query completed. KV cache reset to base_pos=%d.\n", base_pos);
        }
    }

    close(fd);
    unlink(g_active_sock);
    if (g_created_symlink) unlink("/tmp/qwen.sock");
    return 0;
}
