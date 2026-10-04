// Real-Time Streaming Autoregressive Generator for Qwen2.5-0.5B
// Direct Hardware Execution on NVIDIA GeForce GT 750M (Kepler GK107, 2 GB GDDR5)
// Via Mesa Rusticl OpenCL 3.0 over Nouveau
//
// 100% Isolated standalone implementation - Does not touch or modify 1.5B engine.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define N_LAYERS 24
#define D_MODEL 896
#define HEAD_DIM 64
#define N_HEADS_Q 14
#define N_HEADS_KV 2
#define GQA_GROUP_SIZE 7
#define D_QKV 1152
#define D_FFN 4864
#define T_MAX 4096
#define ROPE_BASE 1000000.0f
#define EPSILON 1e-6f
#define PV_SEGMENT_SIZE 128
#define MAX_SEGMENTS (T_MAX / PV_SEGMENT_SIZE)

#define VOCAB_SIZE 151936
#define BLOCKS_PER_ROW_Q8 (D_MODEL / 32) // 28
#define HEAD_BYTES_PER_ROW (BLOCKS_PER_ROW_Q8 * 34) // 952 bytes (Q8_0)

typedef struct {
    uint16_t d;       // IEEE 754 half
    uint8_t qs[16];   // 32 4-bit nibbles
} block_q4_0;

typedef struct {
    uint16_t d;       // IEEE 754 half
    int8_t  qs[32];   // 32 8-bit quantized weights
} block_q8_0;

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
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

static void read_exact_at(int fd, uint64_t offset, void *dest, size_t size) {
    uint8_t *ptr = (uint8_t*)dest;
    size_t total = 0;
    while (total < size) {
        ssize_t n = pread(fd, ptr + total, size - total, offset + total);
        if (n < 0) { perror("pread"); exit(1); }
        if (n == 0) { fprintf(stderr, "Unexpected EOF at offset %lu (wanted %zu, got %zu)\n", offset, size, total); exit(1); }
        total += n;
    }
}

typedef struct {
    char name[64];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
} GGUFTensor;

static int parse_gguf_tensors(int fd, GGUFTensor *tensors, int max_tensors, uint64_t *out_data_start) {
    off_t orig = lseek(fd, 0, SEEK_CUR);
    lseek(fd, 0, SEEK_SET);

    uint32_t magic, version;
    if (read(fd, &magic, 4) != 4 || read(fd, &version, 4) != 4) return -1;
    if (magic != 0x46554747) return -1; // "GGUF"

    uint64_t n_tensors, n_kv;
    if (read(fd, &n_tensors, 8) != 8 || read(fd, &n_kv, 8) != 8) return -1;

    uint32_t alignment = 32;

    for (uint64_t k = 0; k < n_kv; k++) {
        uint64_t key_len;
        if (read(fd, &key_len, 8) != 8) return -1;
        char key[256];
        uint64_t to_read = (key_len < 255) ? key_len : 255;
        if (read(fd, key, to_read) != (ssize_t)to_read) return -1;
        key[to_read] = '\0';
        if (key_len > 255) lseek(fd, key_len - 255, SEEK_CUR);

        uint32_t val_type;
        if (read(fd, &val_type, 4) != 4) return -1;

        if (strcmp(key, "general.alignment") == 0 && val_type == 4) {
            if (read(fd, &alignment, 4) != 4) return -1;
        } else {
            if (val_type == 0 || val_type == 1 || val_type == 7) lseek(fd, 1, SEEK_CUR);
            else if (val_type == 2 || val_type == 3) lseek(fd, 2, SEEK_CUR);
            else if (val_type == 4 || val_type == 5 || val_type == 6) lseek(fd, 4, SEEK_CUR);
            else if (val_type == 10 || val_type == 11 || val_type == 12) lseek(fd, 8, SEEK_CUR);
            else if (val_type == 8) {
                uint64_t slen;
                if (read(fd, &slen, 8) != 8) return -1;
                lseek(fd, slen, SEEK_CUR);
            } else if (val_type == 9) {
                uint32_t atype;
                uint64_t alen;
                if (read(fd, &atype, 4) != 4 || read(fd, &alen, 8) != 8) return -1;
                for (uint64_t i = 0; i < alen; i++) {
                    if (atype == 0 || atype == 1 || atype == 7) lseek(fd, 1, SEEK_CUR);
                    else if (atype == 2 || atype == 3) lseek(fd, 2, SEEK_CUR);
                    else if (atype == 4 || atype == 5 || atype == 6) lseek(fd, 4, SEEK_CUR);
                    else if (atype == 10 || atype == 11 || atype == 12) lseek(fd, 8, SEEK_CUR);
                    else if (atype == 8) {
                        uint64_t slen;
                        if (read(fd, &slen, 8) != 8) return -1;
                        lseek(fd, slen, SEEK_CUR);
                    }
                }
            }
        }
    }

    int count = (n_tensors < (uint64_t)max_tensors) ? (int)n_tensors : max_tensors;
    for (int i = 0; i < count; i++) {
        uint64_t name_len;
        if (read(fd, &name_len, 8) != 8) return -1;
        uint64_t to_read = (name_len < 63) ? name_len : 63;
        if (read(fd, tensors[i].name, to_read) != (ssize_t)to_read) return -1;
        tensors[i].name[to_read] = '\0';
        if (name_len > 63) lseek(fd, name_len - 63, SEEK_CUR);

        if (read(fd, &tensors[i].n_dims, 4) != 4) return -1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            if (read(fd, &tensors[i].ne[d], 8) != 8) return -1;
        }
        if (read(fd, &tensors[i].type, 4) != 4) return -1;
        if (read(fd, &tensors[i].offset, 8) != 8) return -1;
    }

    off_t header_end = lseek(fd, 0, SEEK_CUR);
    uint64_t data_start = (header_end + alignment - 1) & ~((uint64_t)alignment - 1);
    *out_data_start = data_start;

    for (int i = 0; i < count; i++) {
        tensors[i].offset += data_start;
    }

    lseek(fd, orig, SEEK_SET);
    return count;
}

static uint64_t get_tensor_offset(const GGUFTensor *tensors, int count, const char *name) {
    for (int i = 0; i < count; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            return tensors[i].offset;
        }
    }
    fprintf(stderr, "Missing tensor in GGUF: %s\n", name);
    exit(1);
}

static void get_token_embedding(int fd, uint64_t embd_base_offset, int token_id, float *x_out) {
    uint64_t token_offset = embd_base_offset + (uint64_t)token_id * 28 * sizeof(block_q4_0);
    block_q4_0 blks[28];
    read_exact_at(fd, token_offset, blks, sizeof(block_q4_0) * 28);

    for (int b = 0; b < 28; b++) {
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
            uint32_t atype;
            uint64_t alen;
            if (read(fd, &atype, 4) != 4 || read(fd, &alen, 8) != 8) break;
            *out_count = (int)alen;
            tokens = (char**)malloc(sizeof(char*) * alen);
            for (uint64_t i = 0; i < alen; i++) {
                uint64_t slen;
                if (read(fd, &slen, 8) != 8) break;
                tokens[i] = (char*)malloc(slen + 1);
                if (read(fd, tokens[i], slen) != (ssize_t)slen) break;
                tokens[i][slen] = '\0';
            }
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
                uint32_t atype;
                uint64_t alen;
                if (read(fd, &atype, 4) != 4 || read(fd, &alen, 8) != 8) break;
                for (uint64_t i = 0; i < alen; i++) {
                    if (atype == 0 || atype == 1 || atype == 7) lseek(fd, 1, SEEK_CUR);
                    else if (atype == 2 || atype == 3) lseek(fd, 2, SEEK_CUR);
                    else if (atype == 4 || atype == 5 || atype == 6) lseek(fd, 4, SEEK_CUR);
                    else if (atype == 10 || atype == 11 || atype == 12) lseek(fd, 8, SEEK_CUR);
                    else if (atype == 8) {
                        uint64_t slen;
                        if (read(fd, &slen, 8) != 8) break;
                        lseek(fd, slen, SEEK_CUR);
                    }
                }
            }
        }
    }
    lseek(fd, orig, SEEK_SET);
    return tokens;
}

static void print_token_piece(const char *piece) {
    if (!piece) return;
    const unsigned char *p = (const unsigned char*)piece;
    while (*p) {
        if (p[0] == 0xC4 && p[1] == 0xA0) {
            putchar(' ');
            p += 2;
        } else if (p[0] == 0xC4 && p[1] == 0x8A) {
            putchar('\n');
            p += 2;
        } else {
            putchar(*p);
            p++;
        }
    }
}

static char *load_kernel_source(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror(filename); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char*)malloc(sz + 1);
    fread(src, 1, sz, f);
    src[sz] = '\0';
    fclose(f);
    return src;
}

typedef struct {
    cl_mem gamma_attn;
    cl_mem gamma_ffn;
    cl_mem b_qkv;
    cl_mem W_qkv;
    cl_mem W_o;
    cl_mem W_gate;
    cl_mem W_up;
    cl_mem W_down;
} LayerWeightsDev;

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
    cl_mem k_cache;
    cl_mem v_cache;
} LayerKVCacheDev;

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
    cl_mem state;
    cl_mem norm;
    cl_mem qkv;
    cl_mem scores;
    cl_mem partial;
    cl_mem h;
    cl_mem d_k_sub;
    cl_mem d_v_sub;
} WorkSpaceDev;

static void gpu_decoder_layer_step(
    cl_command_queue queue,
    const LayerKernels *k,
    const LayerWeightsDev *w,
    const LayerKVCacheDev *kv,
    const WorkSpaceDev *ws,
    int pos,
    int seq_len
) {
    int d_model = D_MODEL;
    int param_d_qkv = D_QKV;
    int d_ffn = D_FFN;
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
    clSetKernelArg(k->k_qkv_gemv, 1, sizeof(cl_mem), &w->b_qkv);
    clSetKernelArg(k->k_qkv_gemv, 2, sizeof(cl_mem), &ws->norm);
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
    clSetKernelArg(k->k_wo_residual, 4, sizeof(int), &d_model);

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
    size_t l_rope = 32, g_rope = 32;
    size_t l_scores[2] = { 32, 1 };
    size_t g_scores[2] = { (size_t)seq_len * 32, (size_t)N_HEADS_Q };
    size_t l_soft = 128, g_soft = 14 * 128;
    size_t l_pv[2] = { 64, 1 };
    size_t g_pv[2] = { (size_t)num_segs * 64, (size_t)N_HEADS_Q };
    size_t l_red = 64, g_red = 14 * 64;
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

    int default_prompt[] = { 750, 912, 2877, 11, 293, 982, 262, 470, 220 }; // "def add(a, b):\n    return "
    int n_default = sizeof(default_prompt) / sizeof(default_prompt[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
            if (max_new_tokens < 0) max_new_tokens = 0;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            quiet = true;
        } else if (strcmp(argv[i], "--no-echo-prompt") == 0) {
            no_echo_prompt = true;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            char *tok_str = strdup(argv[++i]);
            char *saveptr = NULL;
            char *tok = strtok_r(tok_str, ",", &saveptr);
            int cap = 16;
            prompt_tokens = (int*)malloc(sizeof(int) * cap);
            n_prompt = 0;
            while (tok) {
                if (n_prompt >= cap) {
                    cap *= 2;
                    prompt_tokens = (int*)realloc(prompt_tokens, sizeof(int) * cap);
                }
                prompt_tokens[n_prompt++] = atoi(tok);
                tok = strtok_r(NULL, ",", &saveptr);
            }
            free(tok_str);
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
        max_new_tokens = T_MAX - n_prompt;
        if (max_new_tokens < 0) max_new_tokens = 0;
    }

    if (!quiet) {
        fprintf(stderr, "========================================================================================\n");
        fprintf(stderr, " Phase C: Real-Time Streaming Autoregressive Generator (Qwen2.5-0.5B)                   \n");
        fprintf(stderr, " Direct Hardware Execution on GT 750M (Kepler OpenCL 3.0 via Mesa Rusticl)             \n");
        fprintf(stderr, "========================================================================================\n");
    }

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-0.5b-instruct-q4_0.gguf";
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
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].b_qkv, sizeof(float) * 896);
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].b_qkv + 896, sizeof(float) * 128);
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].b_qkv + 896 + 128, sizeof(float) * 128);

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_qkv, sizeof(block_q4_0) * 896 * 28);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_qkv + 896 * 28, sizeof(block_q4_0) * 128 * 28);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        read_exact_at(fd, get_tensor_offset(tensors, tensor_count, name), w_host[l].W_qkv + (896 + 128) * 28, sizeof(block_q4_0) * 128 * 28);

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

    char *src_dec = load_kernel_source("kernel_decoder_layer_05b.cl");
    cl_program prog_dec = clCreateProgramWithSource(context, 1, (const char**)&src_dec, NULL, &err); CHECK_CL(err, "prog_dec");
    free(src_dec);
    err = clBuildProgram(prog_dec, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build dec");

    char *src_q8 = load_kernel_source("kernel_q8_0.cl");
    cl_program prog_q8 = clCreateProgramWithSource(context, 1, (const char**)&src_q8, NULL, &err); CHECK_CL(err, "prog_q8");
    free(src_q8);
    err = clBuildProgram(prog_q8, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build q8");

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
    cl_kernel k_lm_head = clCreateKernel(prog_q8, "gemv_q8_0", &err); CHECK_CL(err, "k_lm_head");

    if (!quiet) fprintf(stderr, "Transferring 24 layers + Output Norm + Full LM Head to GT 750M VRAM...\n");
    double t_vram_start = get_time_us();

    LayerWeightsDev w_dev[N_LAYERS];
    LayerKVCacheDev kv_dev[N_LAYERS];

    for (int l = 0; l < N_LAYERS; l++) {
        w_dev[l].gamma_attn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, w_host[l].gamma_attn, &err);
        w_dev[l].gamma_ffn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, w_host[l].gamma_ffn, &err);
        w_dev[l].b_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_QKV, w_host[l].b_qkv, &err);
        w_dev[l].W_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_qkv, w_host[l].W_qkv, &err);
        w_dev[l].W_o        = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_o, w_host[l].W_o, &err);
        w_dev[l].W_gate     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_gate, &err);
        w_dev[l].W_up       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_up, &err);
        w_dev[l].W_down     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_down, w_host[l].W_down, &err);

        kv_dev[l].k_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err);
        kv_dev[l].v_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err);
    }

    cl_mem d_gamma_output_norm = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, gamma_output_norm, &err);
    cl_mem d_W_head = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_head, h_head, &err);
    free(h_head);

    WorkSpaceDev ws;
    ws.state   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err);
    ws.norm    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err);
    ws.qkv     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_QKV, NULL, &err);
    ws.scores  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N_HEADS_Q * T_MAX, NULL, &err);
    ws.partial = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N_HEADS_Q * MAX_SEGMENTS * HEAD_DIM, NULL, &err);
    ws.h       = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_FFN, NULL, &err);

    cl_buffer_region r_k = { 896 * sizeof(float), 128 * sizeof(float) };
    cl_buffer_region r_v = { (896 + 128) * sizeof(float), 128 * sizeof(float) };
    ws.d_k_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &r_k, &err); CHECK_CL(err, "sub_k");
    ws.d_v_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &r_v, &err); CHECK_CL(err, "sub_v");

    cl_mem d_final_norm = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err);
    cl_mem d_logits     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * VOCAB_SIZE, NULL, &err);

    float zero_val = 0.0f;
    for (int l = 0; l < N_LAYERS; l++) {
        clEnqueueFillBuffer(queue, kv_dev[l].k_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, kv_dev[l].v_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
    }
    clFinish(queue);

    double t_vram_end = get_time_us();
    if (!quiet) fprintf(stderr, "Entire 0.5B Parameter Model (~420 MB) Resident in GT 750M VRAM in %.2f ms!\n\n", (t_vram_end - t_vram_start) / 1000.0);

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

    if (!no_echo_prompt) {
        for (int i = 0; i < n_prompt; i++) {
            print_token_piece(vocab_tokens[prompt_tokens[i]]);
        }
        fflush(stdout);
    }

    float *h_logits = (float*)malloc(sizeof(float) * VOCAB_SIZE);
    float h_embd[D_MODEL];

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
        }
        clFinish(queue);
        double t_step_1 = get_time_us();

        if (!quiet) {
            fprintf(stderr, "[Prefill %d/%d | pos=%d | tok=%-5d '%s' | lat=%.2f ms]\n",
                    p + 1, n_prompt, p, tok,
                    (tok < n_vocab_tokens && vocab_tokens[tok]) ? vocab_tokens[tok] : "<?>",
                    (t_step_1 - t_step_0) / 1000.0);
        }
    }

    double t_prefill_end = get_time_us();
    double ttft_ms = (t_prefill_end - t_prefill_start) / 1000.0;
    if (!quiet) {
        fprintf(stderr, "\nPrefill Completed in %.2f ms (TTFT: %.2f ms)\n", ttft_ms, ttft_ms);
        fprintf(stderr, "\n========================================================================================\n");
        fprintf(stderr, " STAGE 2: AUTOREGRESSIVE STREAMING GENERATION (GREEDY ARGMAX)\n");
        fprintf(stderr, "========================================================================================\n");
    }

    int next_input_pos = n_prompt;
    int best_tok = 0;
    float max_logit = -1e30f;
    for (int v = 0; v < VOCAB_SIZE; v++) {
        if (h_logits[v] > max_logit) {
            max_logit = h_logits[v];
            best_tok = v;
        }
    }

    print_token_piece(vocab_tokens[best_tok]);
    fflush(stdout);

    if (!quiet) {
        fprintf(stderr, "[Gen %2d | pos=%2d | tok=%-5d '%-7s' | logit=%.4f | lat=Prefill]\n",
                1, next_input_pos - 1, best_tok,
                (best_tok < n_vocab_tokens && vocab_tokens[best_tok]) ? vocab_tokens[best_tok] : "<?>",
                max_logit);
    }

    double total_decode_us = 0.0;
    int decode_steps = 0;

    for (int gen = 1; gen < max_new_tokens; gen++) {
        if (best_tok == 151643 || best_tok == 151645) { // <|endoftext|> / <|im_end|>
            if (!quiet) fprintf(stderr, "\n[Encountered EOS token %d]\n", best_tok);
            break;
        }

        if (next_input_pos >= T_MAX) {
            if (!quiet) fprintf(stderr, "\n[Context length reached T_MAX=%d]\n", T_MAX);
            break;
        }

        double t_gen_0 = get_time_us();

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
        clFinish(queue);

        double t_gen_1 = get_time_us();
        double lat_ms = (t_gen_1 - t_gen_0) / 1000.0;
        total_decode_us += (t_gen_1 - t_gen_0);
        decode_steps++;

        max_logit = -1e30f;
        int next_tok = 0;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            if (h_logits[v] > max_logit) {
                max_logit = h_logits[v];
                next_tok = v;
            }
        }

        print_token_piece(vocab_tokens[next_tok]);
        fflush(stdout);

        if (!quiet) {
            fprintf(stderr, "[Gen %2d | pos=%2d | tok=%-5d '%-7s' | logit=%.4f | lat=%.2f ms]\n",
                    gen + 1, next_input_pos, next_tok,
                    (next_tok < n_vocab_tokens && vocab_tokens[next_tok]) ? vocab_tokens[next_tok] : "<?>",
                    max_logit, lat_ms);
        }

        best_tok = next_tok;
        next_input_pos++;
    }

    putchar('\n');
    fflush(stdout);

    if (!quiet) {
        fprintf(stderr, "\n========================================================================================\n");
        fprintf(stderr, " GENERATION BENCHMARK SUMMARY (GT 750M Physical Hardware - 0.5B Model):\n");
        fprintf(stderr, "========================================================================================\n");
        fprintf(stderr, "  Prompt Tokens Processed:      %d tokens\n", n_prompt);
        fprintf(stderr, "  Generated Tokens:             %d tokens\n", decode_steps + 1);
        fprintf(stderr, "  Decode Forward Passes:        %d passes\n", decode_steps);
        if (decode_steps > 0) {
            double mean_ms = (total_decode_us / (double)decode_steps) / 1000.0;
            fprintf(stderr, "  Mean Decode Forward Latency:  %.2f ms/pass\n", mean_ms);
            fprintf(stderr, "  Decode Forward Rate:          %.2f passes/sec (t/s)\n", 1000.0 / mean_ms);
        }
        fprintf(stderr, "========================================================================================\n\n");
    }

    // Cleanup
    free(h_logits);
    free(prompt_tokens);
    for (int i = 0; i < n_vocab_tokens; i++) free(vocab_tokens[i]);
    free(vocab_tokens);
    close(fd);

    return 0;
}
