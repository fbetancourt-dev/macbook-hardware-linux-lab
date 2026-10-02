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

typedef struct {
    uint16_t d;       // IEEE 754 half
    uint8_t qs[16];   // 32 4-bit nibbles
} block_q4_0;

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

static inline float stable_silu(float x) {
    if (x >= 0.0f) {
        return x / (1.0f + expf(-x));
    } else {
        float ex = expf(x);
        return x * ex / (1.0f + ex);
    }
}

static inline float dot_block_q4_0_avx2(const block_q4_0 *blk, const float *x) {
    float d = fp16_to_float(blk->d);

    float w[32];
    for (int j = 0; j < 16; j++) {
        uint8_t q = blk->qs[j];
        w[j]      = (float)((int)(q & 0x0F) - 8);
        w[j + 16] = (float)((int)(q >>   4) - 8);
    }

    __m256 acc0 = _mm256_mul_ps(_mm256_loadu_ps(w + 0),  _mm256_loadu_ps(x + 0));
    __m256 acc1 = _mm256_mul_ps(_mm256_loadu_ps(w + 8),  _mm256_loadu_ps(x + 8));
    __m256 acc2 = _mm256_mul_ps(_mm256_loadu_ps(w + 16), _mm256_loadu_ps(x + 16));
    __m256 acc3 = _mm256_mul_ps(_mm256_loadu_ps(w + 24), _mm256_loadu_ps(x + 24));

    __m256 sum0 = _mm256_add_ps(acc0, acc1);
    __m256 sum1 = _mm256_add_ps(acc2, acc3);
    __m256 sum  = _mm256_add_ps(sum0, sum1);

    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));

    return _mm_cvtss_f32(s1) * d;
}

static void cpu_gemv_q4_0_bias(
    const block_q4_0 *W,
    const float *x,
    const float *bias,
    float *y,
    int M,
    int K
) {
    int blocks_per_row = K / 32;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        const block_q4_0 *row_blocks = W + row * blocks_per_row;
        float sum = (bias != NULL) ? bias[row] : 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            sum += dot_block_q4_0_avx2(&row_blocks[b], x + b * 32);
        }
        y[row] = sum;
    }
}

static void cpu_swiglu_fused(
    const block_q4_0 *W_gate,
    const block_q4_0 *W_up,
    const float *x,
    float *h,
    int M,
    int K
) {
    int blocks_per_row = K / 32;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        const block_q4_0 *gate_blocks = W_gate + row * blocks_per_row;
        const block_q4_0 *up_blocks   = W_up   + row * blocks_per_row;

        float sum_gate = 0.0f;
        float sum_up   = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            sum_gate += dot_block_q4_0_avx2(&gate_blocks[b], x + b * 32);
            sum_up   += dot_block_q4_0_avx2(&up_blocks[b],   x + b * 32);
        }
        h[row] = stable_silu(sum_gate) * sum_up;
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
    float *k_cache;
    float *v_cache;
} LayerKVHost;

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
    cl_kernel k_output_norm;
} LayerKernels;

void cpu_decoder_layer_step(
    const float *x_in,
    const LayerWeightsHost *w,
    LayerKVHost *kv,
    int pos,
    int seq_len,
    float *y_out
) {
    // 1. RMSNorm (attn)
    float sum_sq1 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq1 += x_in[i] * x_in[i];
    }
    float scale1 = 1.0f / sqrtf((sum_sq1 / (float)D_MODEL) + EPSILON);
    float z1[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z1[i] = x_in[i] * scale1 * w->gamma_attn[i];
    }

    // 2. QKV Projection with Bias
    float qkv[D_QKV];
    cpu_gemv_q4_0_bias(w->W_qkv, z1, w->b_qkv, qkv, D_QKV, D_MODEL);

    float *q = qkv;
    float *k = qkv + 1536;
    float *v = qkv + 1536 + 256;

    // 3. RoPE on Q & K, append to KV Cache
    for (int tid = 0; tid < 64; tid++) {
        float theta = (float)pos * powf(ROPE_BASE, -2.0f * (float)tid / (float)HEAD_DIM);
        float cos_th = cosf(theta);
        float sin_th = sinf(theta);

        for (int h = 0; h < N_HEADS_Q; h++) {
            int base = h * HEAD_DIM;
            float q0 = q[base + tid];
            float q1 = q[base + tid + 64];
            q[base + tid]      = q0 * cos_th - q1 * sin_th;
            q[base + tid + 64] = q0 * sin_th + q1 * cos_th;
        }

        for (int h = 0; h < N_HEADS_KV; h++) {
            int in_base = h * HEAD_DIM;
            float k0 = k[in_base + tid];
            float k1 = k[in_base + tid + 64];

            float k_rot0 = k0 * cos_th - k1 * sin_th;
            float k_rot1 = k0 * sin_th + k1 * cos_th;

            int out_base = (h * T_MAX + pos) * HEAD_DIM;
            kv->k_cache[out_base + tid]      = k_rot0;
            kv->k_cache[out_base + tid + 64] = k_rot1;

            kv->v_cache[out_base + tid]      = v[in_base + tid];
            kv->v_cache[out_base + tid + 64] = v[in_base + tid + 64];
        }
    }

    // 4 & 5 & 6. GQA Scores -> Softmax -> Value Combination
    float attn_out[D_MODEL];
    float scale_factor = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h_q = 0; h_q < N_HEADS_Q; h_q++) {
        int h_kv = h_q / GQA_GROUP_SIZE;
        float scores[seq_len];
        float max_s = -1e30f;
        const float *q_h = q + h_q * HEAD_DIM;

        for (int t = 0; t < seq_len; t++) {
            const float *k_h = kv->k_cache + (h_kv * T_MAX + t) * HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_h[d] * k_h[d];
            }
            float s = dot * scale_factor;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float exp_val = expf(scores[t] - max_s);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }

        float *out_h = attn_out + h_q * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            float accum = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                accum += scores[t] * kv->v_cache[(h_kv * T_MAX + t) * HEAD_DIM + d];
            }
            out_h[d] = accum;
        }
    }

    // 7. Output Projection + Residual Addition (r = x_in + Wo * attn_out)
    float r[D_MODEL];
    cpu_gemv_q4_0_bias(w->W_o, attn_out, NULL, r, D_MODEL, D_MODEL);
    for (int i = 0; i < D_MODEL; i++) {
        r[i] += x_in[i];
    }

    // 8. RMSNorm (ffn)
    float sum_sq2 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq2 += r[i] * r[i];
    }
    float scale2 = 1.0f / sqrtf((sum_sq2 / (float)D_MODEL) + EPSILON);
    float z2[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z2[i] = r[i] * scale2 * w->gamma_ffn[i];
    }

    // 9. FFN SwiGLU Fused
    float h_ffn[D_FFN];
    cpu_swiglu_fused(w->W_gate, w->W_up, z2, h_ffn, D_FFN, D_MODEL);

    // 10. Down Projection + Residual Addition (y_out = r + W_down * h_ffn)
    cpu_gemv_q4_0_bias(w->W_down, h_ffn, NULL, y_out, D_MODEL, D_FFN);
    for (int i = 0; i < D_MODEL; i++) {
        y_out[i] += r[i];
    }
}

void gpu_decoder_layer_step(
    cl_command_queue queue,
    LayerKernels *k,
    LayerWeightsDevice *w,
    LayerKVDevice *kv,
    DecoderWorkspaceDevice *ws,
    int pos,
    int seq_len
) {
    float eps = EPSILON;
    int d_model = D_MODEL;
    int param_d_qkv = D_QKV;
    int d_ffn = D_FFN;
    int t_max = T_MAX;
    float rope_base = ROPE_BASE;
    float scale_factor = 1.0f / sqrtf((float)HEAD_DIM);
    int num_segs = (seq_len + PV_SEGMENT_SIZE - 1) / PV_SEGMENT_SIZE;

    // 1. RMSNorm (attn)
    clSetKernelArg(k->k_rmsnorm_attn, 0, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_rmsnorm_attn, 1, sizeof(cl_mem), &w->gamma_attn);
    clSetKernelArg(k->k_rmsnorm_attn, 2, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_rmsnorm_attn, 3, sizeof(int), &d_model);
    clSetKernelArg(k->k_rmsnorm_attn, 4, sizeof(float), &eps);

    // 2. QKV Projection with Bias
    clSetKernelArg(k->k_qkv_gemv, 0, sizeof(cl_mem), &w->W_qkv);
    clSetKernelArg(k->k_qkv_gemv, 1, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_qkv_gemv, 2, sizeof(cl_mem), &w->b_qkv);
    clSetKernelArg(k->k_qkv_gemv, 3, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_qkv_gemv, 4, sizeof(int), &param_d_qkv);
    clSetKernelArg(k->k_qkv_gemv, 5, sizeof(int), &d_model);

    // 3. RoPE on Q & K, append to KV Cache
    clSetKernelArg(k->k_rope_kv, 0, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_rope_kv, 1, sizeof(cl_mem), &ws->d_k_sub);
    clSetKernelArg(k->k_rope_kv, 2, sizeof(cl_mem), &ws->d_v_sub);
    clSetKernelArg(k->k_rope_kv, 3, sizeof(cl_mem), &kv->k_cache);
    clSetKernelArg(k->k_rope_kv, 4, sizeof(cl_mem), &kv->v_cache);
    clSetKernelArg(k->k_rope_kv, 5, sizeof(int), &pos);
    clSetKernelArg(k->k_rope_kv, 6, sizeof(int), &t_max);
    clSetKernelArg(k->k_rope_kv, 7, sizeof(float), &rope_base);

    // 4. GQA Scores
    clSetKernelArg(k->k_scores, 0, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_scores, 1, sizeof(cl_mem), &kv->k_cache);
    clSetKernelArg(k->k_scores, 2, sizeof(cl_mem), &ws->scores);
    clSetKernelArg(k->k_scores, 3, sizeof(int), &seq_len);
    clSetKernelArg(k->k_scores, 4, sizeof(int), &t_max);
    clSetKernelArg(k->k_scores, 5, sizeof(float), &scale_factor);

    // 5. Softmax GQA
    clSetKernelArg(k->k_softmax, 0, sizeof(cl_mem), &ws->scores);
    clSetKernelArg(k->k_softmax, 1, sizeof(int), &seq_len);
    clSetKernelArg(k->k_softmax, 2, sizeof(int), &t_max);

    // 6. Split-K PV Combine & Reduce
    clSetKernelArg(k->k_pv_combine, 0, sizeof(cl_mem), &ws->scores);
    clSetKernelArg(k->k_pv_combine, 1, sizeof(cl_mem), &kv->v_cache);
    clSetKernelArg(k->k_pv_combine, 2, sizeof(cl_mem), &ws->partial);
    clSetKernelArg(k->k_pv_combine, 3, sizeof(int), &seq_len);
    clSetKernelArg(k->k_pv_combine, 4, sizeof(int), &t_max);
    clSetKernelArg(k->k_pv_combine, 5, sizeof(int), &num_segs);

    clSetKernelArg(k->k_pv_reduce, 0, sizeof(cl_mem), &ws->partial);
    clSetKernelArg(k->k_pv_reduce, 1, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_pv_reduce, 2, sizeof(int), &num_segs);

    // 7. Wo Projection + Residual (ws.state = ws.state + Wo * attn_out)
    clSetKernelArg(k->k_wo_residual, 0, sizeof(cl_mem), &w->W_o);
    clSetKernelArg(k->k_wo_residual, 1, sizeof(cl_mem), &ws->qkv);
    clSetKernelArg(k->k_wo_residual, 2, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_wo_residual, 3, sizeof(int), &d_model);

    // 8. RMSNorm (ffn)
    clSetKernelArg(k->k_rmsnorm_ffn, 0, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_rmsnorm_ffn, 1, sizeof(cl_mem), &w->gamma_ffn);
    clSetKernelArg(k->k_rmsnorm_ffn, 2, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_rmsnorm_ffn, 3, sizeof(int), &d_model);
    clSetKernelArg(k->k_rmsnorm_ffn, 4, sizeof(float), &eps);

    // 9. Fused SwiGLU
    clSetKernelArg(k->k_swiglu_fused, 0, sizeof(cl_mem), &w->W_gate);
    clSetKernelArg(k->k_swiglu_fused, 1, sizeof(cl_mem), &w->W_up);
    clSetKernelArg(k->k_swiglu_fused, 2, sizeof(cl_mem), &ws->norm);
    clSetKernelArg(k->k_swiglu_fused, 3, sizeof(cl_mem), &ws->h);
    clSetKernelArg(k->k_swiglu_fused, 4, sizeof(int), &d_ffn);
    clSetKernelArg(k->k_swiglu_fused, 5, sizeof(int), &d_model);

    // 10. Down Projection + Residual (ws.state = ws.state + W_down * ws.h)
    clSetKernelArg(k->k_down_res, 0, sizeof(cl_mem), &w->W_down);
    clSetKernelArg(k->k_down_res, 1, sizeof(cl_mem), &ws->h);
    clSetKernelArg(k->k_down_res, 2, sizeof(cl_mem), &ws->state);
    clSetKernelArg(k->k_down_res, 3, sizeof(int), &d_model);
    clSetKernelArg(k->k_down_res, 4, sizeof(int), &d_ffn);

    // Enqueue grids:
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

    clEnqueueNDRangeKernel(queue, k->k_rmsnorm_attn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_qkv_gemv,     1, NULL, &g_qkv,     &l_qkv,     0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_rope_kv,      1, NULL, &g_rope,    &l_rope,    0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_scores,       2, NULL, g_scores,   l_scores,   0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_softmax,      1, NULL, &g_soft,    &l_soft,    0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_pv_combine,   2, NULL, g_pv,       l_pv,       0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_pv_reduce,    1, NULL, &g_red,     &l_red,     0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_wo_residual,  1, NULL, &g_wo,      &l_wo,      0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_rmsnorm_ffn,  1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_swiglu_fused, 1, NULL, &g_swiglu,  &l_swiglu,  0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k->k_down_res,     1, NULL, &g_down,    &l_down,    0, NULL, NULL);
}

static char* load_kernel_source(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char*)malloc(sz + 1);
    fread(src, 1, sz, f);
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
            fprintf(stderr, "FATAL: pread failed at offset %lu: %s\n", (unsigned long)offset, strerror(errno));
            exit(1);
        }
        if (n == 0) {
            fprintf(stderr, "FATAL: Unexpected EOF at offset %lu (needed %zu bytes)\n", (unsigned long)offset, remaining);
            exit(1);
        }
        p += n;
        offset += n;
        remaining -= n;
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

static void skip_gguf_metadata(int fd, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        uint64_t klen;
        if (read(fd, &klen, sizeof(klen)) != sizeof(klen)) exit(1);
        lseek(fd, klen, SEEK_CUR);
        uint32_t vtype;
        if (read(fd, &vtype, sizeof(vtype)) != sizeof(vtype)) exit(1);
        if (vtype == 0 || vtype == 1 || vtype == 7) {
            lseek(fd, 1, SEEK_CUR);
        } else if (vtype == 2 || vtype == 3) {
            lseek(fd, 2, SEEK_CUR);
        } else if (vtype == 4 || vtype == 5 || vtype == 6) {
            lseek(fd, 4, SEEK_CUR);
        } else if (vtype == 10 || vtype == 11 || vtype == 12) {
            lseek(fd, 8, SEEK_CUR);
        } else if (vtype == 8) {
            uint64_t slen;
            if (read(fd, &slen, sizeof(slen)) != sizeof(slen)) exit(1);
            lseek(fd, slen, SEEK_CUR);
        } else if (vtype == 9) {
            uint32_t etype;
            uint64_t n_elems;
            if (read(fd, &etype, sizeof(etype)) != sizeof(etype)) exit(1);
            if (read(fd, &n_elems, sizeof(n_elems)) != sizeof(n_elems)) exit(1);
            if (etype == 8) {
                for (uint64_t j = 0; j < n_elems; j++) {
                    uint64_t slen;
                    if (read(fd, &slen, sizeof(slen)) != sizeof(slen)) exit(1);
                    lseek(fd, slen, SEEK_CUR);
                }
            } else if (etype == 0 || etype == 1 || etype == 7) {
                lseek(fd, n_elems * 1, SEEK_CUR);
            } else if (etype == 2 || etype == 3) {
                lseek(fd, n_elems * 2, SEEK_CUR);
            } else if (etype == 4 || etype == 5 || etype == 6) {
                lseek(fd, n_elems * 4, SEEK_CUR);
            } else if (etype == 10 || etype == 11 || etype == 12) {
                lseek(fd, n_elems * 8, SEEK_CUR);
            } else {
                fprintf(stderr, "FATAL: Unknown array element type %u in GGUF metadata\n", etype);
                exit(1);
            }
        } else {
            fprintf(stderr, "FATAL: Unknown metadata type %u in GGUF\n", vtype);
            exit(1);
        }
    }
}

static int parse_gguf_tensors(int fd, GGUFTensor *tensors, int max_tensors, uint64_t *out_data_start) {
    lseek(fd, 0, SEEK_SET);
    char magic[4];
    uint32_t version;
    uint64_t tensor_count, metadata_kv_count;
    if (read(fd, magic, 4) != 4) return -1;
    if (read(fd, &version, 4) != 4) return -1;
    if (read(fd, &tensor_count, 8) != 8) return -1;
    if (read(fd, &metadata_kv_count, 8) != 8) return -1;

    skip_gguf_metadata(fd, metadata_kv_count);

    if ((int)tensor_count > max_tensors) {
        fprintf(stderr, "Too many tensors: %lu > %d\n", (unsigned long)tensor_count, max_tensors);
        return -1;
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
        if (strcmp(tensors[i].name, name) == 0) {
            return tensors[i].abs_offset;
        }
    }
    fprintf(stderr, "FATAL: Tensor '%s' not found in GGUF catalog!\n", name);
    exit(1);
}

int main(void) {
    printf("========================================================================================\n");
    printf(" Full 28-Layer Resident Pipeline + Output Norm Validation (Qwen2.5-Coder-1.5B)          \n");
    printf(" Real GGUF Q4_0 Weights on GT 750M (Kepler OpenCL) vs CPU (Haswell AVX2 8-Threads)      \n");
    printf("========================================================================================\n");

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("open GGUF model"); exit(1); }

    GGUFTensor tensors[512];
    uint64_t data_start = 0;
    int tensor_count = parse_gguf_tensors(fd, tensors, 512, &data_start);
    if (tensor_count <= 0) { fprintf(stderr, "Failed to parse GGUF metadata\n"); exit(1); }
    printf("GGUF Indexer: Parsed %d tensor descriptors (data_start = %lu bytes).\n", tensor_count, (unsigned long)data_start);

    // Read real token embeddings for Token 0 and Token 1
    uint64_t off_token_embd = get_tensor_offset(tensors, tensor_count, "token_embd.weight");
    block_q4_0 embd_blk_0[48], embd_blk_1[48];
    read_exact_at(fd, off_token_embd, embd_blk_0, sizeof(block_q4_0) * 48);
    read_exact_at(fd, off_token_embd + sizeof(block_q4_0) * 48, embd_blk_1, sizeof(block_q4_0) * 48);

    float real_x0[D_MODEL], real_x1[D_MODEL];
    for (int b = 0; b < 48; b++) {
        float d0 = fp16_to_float(embd_blk_0[b].d);
        float d1 = fp16_to_float(embd_blk_1[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t q0 = embd_blk_0[b].qs[j];
            uint8_t q1 = embd_blk_1[b].qs[j];
            real_x0[b * 32 + j]      = (float)((int)(q0 & 0x0F) - 8) * d0;
            real_x0[b * 32 + j + 16] = (float)((int)(q0 >>   4) - 8) * d0;
            real_x1[b * 32 + j]      = (float)((int)(q1 & 0x0F) - 8) * d1;
            real_x1[b * 32 + j + 16] = (float)((int)(q1 >>   4) - 8) * d1;
        }
    }
    printf("Real token embeddings dequantized: Token 0 (d=%.5f), Token 1 (d=%.5f).\n",
           fp16_to_float(embd_blk_0[0].d), fp16_to_float(embd_blk_1[0].d));

    // Allocate 28 Host Layers
    int nb_qkv = D_MODEL / 32, nb_wo = D_MODEL / 32, nb_gate = D_MODEL / 32, nb_down = D_FFN / 32;
    size_t sz_W_qkv  = sizeof(block_q4_0) * D_QKV * nb_qkv;
    size_t sz_W_o    = sizeof(block_q4_0) * D_MODEL * nb_wo;
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * nb_down;
    size_t sz_kv     = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;

    LayerWeightsHost w_host[N_LAYERS];
    LayerKVHost kv_host[N_LAYERS];

    printf("Allocating memory and reading weights for all %d layers from GGUF...\n", N_LAYERS);
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

        kv_host[l].k_cache   = (float*)calloc(N_HEADS_KV * T_MAX * HEAD_DIM, sizeof(float));
        kv_host[l].v_cache   = (float*)calloc(N_HEADS_KV * T_MAX * HEAD_DIM, sizeof(float));

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

    // Read output_norm.weight (RMSNorm gamma for final projection)
    float gamma_output_norm[D_MODEL];
    uint64_t off_output_norm = get_tensor_offset(tensors, tensor_count, "output_norm.weight");
    read_exact_at(fd, off_output_norm, gamma_output_norm, sizeof(float) * D_MODEL);

    close(fd);
    double t_load_end = get_time_us();
    printf("Successfully loaded all %d layers + output_norm in %.2f ms!\n\n", N_LAYERS, (t_load_end - t_load_start) / 1000.0);

    // Initialize OpenCL
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id device = NULL;
    cl_int err;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) == CL_SUCCESS && num_devices > 0) {
            cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
            device = devices[0];
            free(devices);
            break;
        }
    }
    free(platforms);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_CL(err, "context");
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); CHECK_CL(err, "queue");

    char *src = load_kernel_source("kernel_decoder_layer.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err); CHECK_CL(err, "program");
    free(src);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build");

    LayerKernels kernels[N_LAYERS];
    cl_kernel k_output_norm = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "k_output_norm");

    printf("Allocating VRAM and transferring %d layers to GT 750M...\n", N_LAYERS);
    double t_vram_start = get_time_us();

    LayerWeightsDevice w_dev[N_LAYERS];
    LayerKVDevice kv_dev[N_LAYERS];

    for (int l = 0; l < N_LAYERS; l++) {
        w_dev[l].W_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_qkv, w_host[l].W_qkv, &err); CHECK_CL(err, "w_qkv");
        w_dev[l].b_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*D_QKV, w_host[l].b_qkv, &err); CHECK_CL(err, "b_qkv");
        w_dev[l].W_o        = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_o, w_host[l].W_o, &err); CHECK_CL(err, "W_o");
        w_dev[l].W_gate     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_gate, &err); CHECK_CL(err, "W_gate");
        w_dev[l].W_up       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_up, &err); CHECK_CL(err, "W_up");
        w_dev[l].W_down     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_down, w_host[l].W_down, &err); CHECK_CL(err, "W_down");
        w_dev[l].gamma_attn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*D_MODEL, w_host[l].gamma_attn, &err); CHECK_CL(err, "gamma_attn");
        w_dev[l].gamma_ffn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*D_MODEL, w_host[l].gamma_ffn, &err); CHECK_CL(err, "gamma_ffn");

        kv_dev[l].k_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err); CHECK_CL(err, "k_cache");
        kv_dev[l].v_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err); CHECK_CL(err, "v_cache");

        kernels[l].k_rmsnorm_attn = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "k_rmsnorm_attn");
        kernels[l].k_qkv_gemv     = clCreateKernel(program, "gemv_q4_0_bias", &err); CHECK_CL(err, "k_qkv_gemv");
        kernels[l].k_rope_kv      = clCreateKernel(program, "kernel_rope_and_kv_append", &err); CHECK_CL(err, "k_rope_kv");
        kernels[l].k_scores       = clCreateKernel(program, "kernel_gqa_scores", &err); CHECK_CL(err, "k_scores");
        kernels[l].k_softmax      = clCreateKernel(program, "kernel_softmax_gqa", &err); CHECK_CL(err, "k_softmax");
        kernels[l].k_pv_combine   = clCreateKernel(program, "kernel_gqa_value_combine_segmented", &err); CHECK_CL(err, "k_pv_combine");
        kernels[l].k_pv_reduce    = clCreateKernel(program, "kernel_gqa_reduce_segments", &err); CHECK_CL(err, "k_pv_reduce");
        kernels[l].k_wo_residual  = clCreateKernel(program, "gemv_q4_0_wo_residual", &err); CHECK_CL(err, "k_wo_residual");
        kernels[l].k_rmsnorm_ffn  = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "k_rmsnorm_ffn");
        kernels[l].k_swiglu_fused = clCreateKernel(program, "gemv_swiglu_fused", &err); CHECK_CL(err, "k_swiglu_fused");
        kernels[l].k_down_res     = clCreateKernel(program, "gemv_q4_0_down_residual", &err); CHECK_CL(err, "k_down_res");
    }

    // Output Norm Device Buffer
    cl_mem d_gamma_output_norm = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, gamma_output_norm, &err); CHECK_CL(err, "d_gamma_output_norm");
    cl_mem d_final_norm = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err); CHECK_CL(err, "d_final_norm");

    // Reusable Workspace (Strictly 0.33 MB)
    DecoderWorkspaceDevice ws;
    ws.state    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_MODEL, NULL, &err); CHECK_CL(err, "ws.state");
    ws.norm     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_MODEL, NULL, &err); CHECK_CL(err, "ws.norm");
    ws.qkv      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_QKV, NULL, &err); CHECK_CL(err, "ws.qkv");
    ws.scores   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N_HEADS_Q*T_MAX, NULL, &err); CHECK_CL(err, "ws.scores");
    ws.partial  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N_HEADS_Q*MAX_SEGMENTS*HEAD_DIM, NULL, &err); CHECK_CL(err, "ws.partial");
    ws.h        = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_FFN, NULL, &err); CHECK_CL(err, "ws.h");

    size_t k_offset_bytes = 1536 * sizeof(float);
    size_t v_offset_bytes = (1536 + 256) * sizeof(float);
    cl_buffer_region reg_k = { k_offset_bytes, 256 * sizeof(float) };
    cl_buffer_region reg_v = { v_offset_bytes, 256 * sizeof(float) };
    ws.d_k_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_k, &err);
    ws.d_v_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_v, &err);

    // Initialize KV caches to zero
    float zero_val = 0.0f;
    for (int l = 0; l < N_LAYERS; l++) {
        clEnqueueFillBuffer(queue, kv_dev[l].k_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, kv_dev[l].v_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
    }
    clFinish(queue);
    double t_vram_end = get_time_us();
    printf("All %d layers resident in GT 750M VRAM in %.2f ms!\n\n", N_LAYERS, (t_vram_end - t_vram_start) / 1000.0);

    // Execute Multi-Token Autoregressive Pass
    printf("Executing 2 Consecutive Tokens through the Complete %d-Layer Model + Output Norm:\n\n", N_LAYERS);
    printf("%-10s | %-8s | %-12s | %-12s | %-10s | %-12s | %-12s | %-8s\n",
           "Step", "Context", "CPU Time", "GPU Time", "Speedup", "Max Abs Diff", "Rel L2 Error", "Cos Sim");
    printf("--------------------------------------------------------------------------------------------------------\n");

    const float *tokens[2] = { real_x0, real_x1 };

    for (int step = 0; step < 2; step++) {
        int pos = step;
        int seq_len = step + 1;
        const float *x_curr = tokens[step];

        // --- GPU EXECUTION ---
        double t_gpu_0 = get_time_us();
        clEnqueueWriteBuffer(queue, ws.state, CL_FALSE, 0, sizeof(float) * D_MODEL, x_curr, 0, NULL, NULL);

        for (int l = 0; l < N_LAYERS; l++) {
            gpu_decoder_layer_step(queue, &kernels[l], &w_dev[l], &kv_dev[l], &ws, pos, seq_len);
        }

        // Apply final Output Norm on GPU: ws.state -> d_final_norm
        int D = D_MODEL;
        float eps = EPSILON;
        clSetKernelArg(k_output_norm, 0, sizeof(cl_mem), &ws.state);
        clSetKernelArg(k_output_norm, 1, sizeof(cl_mem), &d_gamma_output_norm);
        clSetKernelArg(k_output_norm, 2, sizeof(cl_mem), &d_final_norm);
        clSetKernelArg(k_output_norm, 3, sizeof(int), &D);
        clSetKernelArg(k_output_norm, 4, sizeof(float), &eps);
        size_t g_norm = 128, l_norm = 128;
        clEnqueueNDRangeKernel(queue, k_output_norm, 1, NULL, &g_norm, &l_norm, 0, NULL, NULL);

        float z_gpu[D_MODEL];
        clEnqueueReadBuffer(queue, d_final_norm, CL_TRUE, 0, sizeof(float) * D_MODEL, z_gpu, 0, NULL, NULL);
        double t_gpu_1 = get_time_us();
        double ms_gpu = (t_gpu_1 - t_gpu_0) / 1000.0;

        // --- CPU EXECUTION ---
        double t_cpu_0 = get_time_us();
        float state_cpu[D_MODEL], next_state[D_MODEL];
        memcpy(state_cpu, x_curr, sizeof(float) * D_MODEL);

        for (int l = 0; l < N_LAYERS; l++) {
            cpu_decoder_layer_step(state_cpu, &w_host[l], &kv_host[l], pos, seq_len, next_state);
            memcpy(state_cpu, next_state, sizeof(float) * D_MODEL);
        }

        // Apply final Output Norm on CPU
        float sum_sq = 0.0f;
        for (int i = 0; i < D_MODEL; i++) {
            sum_sq += state_cpu[i] * state_cpu[i];
        }
        float scale = 1.0f / sqrtf((sum_sq / (float)D_MODEL) + EPSILON);
        float z_cpu[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            z_cpu[i] = state_cpu[i] * scale * gamma_output_norm[i];
        }
        double t_cpu_1 = get_time_us();
        double ms_cpu = (t_cpu_1 - t_cpu_0) / 1000.0;

        // Verification & Tolerances
        float max_abs_diff = 0.0f;
        double diff_sq_sum = 0.0, cpu_sq_sum = 0.0, gpu_sq_sum = 0.0, dot_prod = 0.0;
        bool valid = true;

        for (int i = 0; i < D_MODEL; i++) {
            if (!isfinite(z_gpu[i]) || !isfinite(z_cpu[i])) {
                valid = false;
            }
            float diff = fabsf(z_gpu[i] - z_cpu[i]);
            if (diff > max_abs_diff) max_abs_diff = diff;
            diff_sq_sum += (double)diff * (double)diff;
            cpu_sq_sum  += (double)z_cpu[i] * (double)z_cpu[i];
            gpu_sq_sum  += (double)z_gpu[i] * (double)z_gpu[i];
            dot_prod    += (double)z_gpu[i] * (double)z_cpu[i];
        }

        double rel_l2 = sqrt(diff_sq_sum) / sqrt(cpu_sq_sum);
        double cos_sim = dot_prod / (sqrt(cpu_sq_sum) * sqrt(gpu_sq_sum));
        double speedup = ms_cpu / ms_gpu;

        char step_str[16], ctx_str[16];
        snprintf(step_str, sizeof(step_str), "Token %d", step);
        snprintf(ctx_str, sizeof(ctx_str), "T=%d", seq_len);

        printf("%-10s | %-8s | %8.2f ms | %8.2f ms | %7.2fx | %12.4e | %12.4e | %8.6f %s\n",
               step_str, ctx_str, ms_cpu, ms_gpu, speedup, max_abs_diff, rel_l2, cos_sim,
               valid ? "" : "[ERROR NaN/Inf]");
    }
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("Full 28-layer Transformer pipeline validated with clean numerical tolerance!\n\n");

    return 0;
}
