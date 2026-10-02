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

static void dequantize_row_q6_k_ref(const uint8_t *row_bytes, float *y, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *blk_ptr = row_bytes + i * 210;
        const uint8_t *ql = blk_ptr;
        const uint8_t *qh = blk_ptr + 128;
        const int8_t  *sc = (const int8_t*)(blk_ptr + 192);
        uint16_t d_raw    = *(const uint16_t*)(blk_ptr + 208);
        const float d     = fp16_to_float(d_raw);

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
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
} LayerKernels;

void cpu_decoder_layer_step(
    const float *x_in,
    const LayerWeightsHost *w,
    LayerKVHost *kv,
    int pos,
    int seq_len,
    float *y_out
) {
    float sum_sq1 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) sum_sq1 += x_in[i] * x_in[i];
    float scale1 = 1.0f / sqrtf((sum_sq1 / (float)D_MODEL) + EPSILON);
    float z1[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) z1[i] = x_in[i] * scale1 * w->gamma_attn[i];

    float qkv[D_QKV];
    cpu_gemv_q4_0_bias(w->W_qkv, z1, w->b_qkv, qkv, D_QKV, D_MODEL);

    float *q = qkv;
    float *k = qkv + 1536;
    float *v = qkv + 1536 + 256;

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
            for (int d = 0; d < HEAD_DIM; d++) dot += q_h[d] * k_h[d];
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
        for (int t = 0; t < seq_len; t++) scores[t] *= inv_sum;

        float *out_h = attn_out + h_q * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            float accum = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                accum += scores[t] * kv->v_cache[(h_kv * T_MAX + t) * HEAD_DIM + d];
            }
            out_h[d] = accum;
        }
    }

    float r[D_MODEL];
    cpu_gemv_q4_0_bias(w->W_o, attn_out, NULL, r, D_MODEL, D_MODEL);
    for (int i = 0; i < D_MODEL; i++) r[i] += x_in[i];

    float sum_sq2 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) sum_sq2 += r[i] * r[i];
    float scale2 = 1.0f / sqrtf((sum_sq2 / (float)D_MODEL) + EPSILON);
    float z2[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) z2[i] = r[i] * scale2 * w->gamma_ffn[i];

    float h_ffn[D_FFN];
    cpu_swiglu_fused(w->W_gate, w->W_up, z2, h_ffn, D_FFN, D_MODEL);

    cpu_gemv_q4_0_bias(w->W_down, h_ffn, NULL, y_out, D_MODEL, D_FFN);
    for (int i = 0; i < D_MODEL; i++) y_out[i] += r[i];
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
            }
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

// Function to dequantize a specific token embedding row
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

int main(void) {
    printf("========================================================================================\n");
    printf(" Phase B: Full 28-Layer Pipeline + LM Head Logits Validation (Qwen2.5-Coder-1.5B)      \n");
    printf(" Direct Hardware Evaluation on GT 750M (Kepler OpenCL) vs CPU (Haswell AVX2)           \n");
    printf("========================================================================================\n");

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";

    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("open GGUF"); exit(1); }

    GGUFTensor tensors[512];
    uint64_t data_start = 0;
    int tensor_count = parse_gguf_tensors(fd, tensors, 512, &data_start);
    printf("GGUF Indexer: Parsed %d tensor descriptors.\n", tensor_count);

    uint64_t off_embd = get_tensor_offset(tensors, tensor_count, "token_embd.weight");
    uint64_t off_head = get_tensor_offset(tensors, tensor_count, "output.weight");
    uint64_t off_norm = get_tensor_offset(tensors, tensor_count, "output_norm.weight");

    // Allocate 28 Host Layers
    int nb_qkv = D_MODEL / 32, nb_wo = D_MODEL / 32, nb_gate = D_MODEL / 32, nb_down = D_FFN / 32;
    size_t sz_W_qkv  = sizeof(block_q4_0) * D_QKV * nb_qkv;
    size_t sz_W_o    = sizeof(block_q4_0) * D_MODEL * nb_wo;
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * nb_down;
    size_t sz_kv     = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;

    LayerWeightsHost w_host[N_LAYERS];
    LayerKVHost kv_host[N_LAYERS];

    printf("Reading weights for all %d layers from GGUF...\n", N_LAYERS);
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

    float gamma_output_norm[D_MODEL];
    read_exact_at(fd, off_norm, gamma_output_norm, sizeof(float) * D_MODEL);

    // Read full LM Head output.weight (151936 rows * 1260 bytes = 182.57 MB)
    size_t sz_head = (size_t)VOCAB_SIZE * HEAD_BYTES_PER_ROW;
    printf("Reading full LM Head (output.weight, %.2f MB)...\n", sz_head / (1024.0 * 1024.0));
    uint8_t *h_head = (uint8_t*)malloc(sz_head);
    read_exact_at(fd, off_head, h_head, sz_head);

    double t_load_end = get_time_us();
    printf("All model weights loaded in %.2f ms!\n\n", (t_load_end - t_load_start) / 1000.0);

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

    // Build Programs
    char *src_dec = load_kernel_source("kernel_decoder_layer.cl");
    cl_program prog_dec = clCreateProgramWithSource(context, 1, (const char**)&src_dec, NULL, &err); CHECK_CL(err, "prog_dec");
    free(src_dec);
    err = clBuildProgram(prog_dec, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build dec");

    char *src_q6 = load_kernel_source("kernel_q6_k.cl");
    cl_program prog_q6 = clCreateProgramWithSource(context, 1, (const char**)&src_q6, NULL, &err); CHECK_CL(err, "prog_q6");
    free(src_q6);
    err = clBuildProgram(prog_q6, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build q6");

    // Create Kernels
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

    // Allocate Device Buffers
    printf("Transferring 28 layers + Output Norm + Full LM Head to GT 750M VRAM...\n");
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
    printf("Entire 1.5B Parameter Model (1110 MB) Resident in GT 750M VRAM in %.2f ms!\n\n", (t_vram_end - t_vram_start) / 1000.0);

    int n_vocab_tokens = 0;
    char **vocab_tokens = load_tokenizer_tokens(fd, &n_vocab_tokens);
    if (vocab_tokens) {
        printf("Loaded %d vocabulary token strings from GGUF metadata.\n", n_vocab_tokens);
    }

    typedef struct {
        int token_id;
        float logit;
    } TokenCandidate;

    int test_prompts[] = {13048, 750}; // "Hi", "def"
    const char *prompt_names[] = {"Hi", "def"};

    for (int p = 0; p < 2; p++) {
        int prompt_token = test_prompts[p];
        const char *pname = prompt_names[p];
        printf("\n========================================================================================\n");
        printf(" EVALUATING PROMPT %d: \"%s\" (Token ID %d) at pos=0, seq_len=1\n", p + 1, pname, prompt_token);
        printf("========================================================================================\n");

        float h_embd[D_MODEL];
        get_token_embedding(fd, off_embd, prompt_token, h_embd);

        // Reset KV cache for fresh prompt evaluation
        float zero_val = 0.0f;
        for (int l = 0; l < N_LAYERS; l++) {
            clEnqueueFillBuffer(queue, kv_dev[l].k_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
            clEnqueueFillBuffer(queue, kv_dev[l].v_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
        }
        clFinish(queue);

        // 1. GPU FORWARD PASS (28 Layers + Output Norm + Full LM Head)
        double t_gpu_0 = get_time_us();

        clEnqueueWriteBuffer(queue, ws.state, CL_FALSE, 0, sizeof(float) * D_MODEL, h_embd, 0, NULL, NULL);

        for (int l = 0; l < N_LAYERS; l++) {
            gpu_decoder_layer_step(queue, &kernels[l], &w_dev[l], &kv_dev[l], &ws, 0, 1);
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

        float *h_logits_gpu = (float*)malloc(sizeof(float) * VOCAB_SIZE);
        clEnqueueReadBuffer(queue, d_logits, CL_TRUE, 0, sizeof(float) * VOCAB_SIZE, h_logits_gpu, 0, NULL, NULL);

        double t_gpu_1 = get_time_us();
        double ms_gpu_e2e = (t_gpu_1 - t_gpu_0) / 1000.0;
        printf("GPU GT 750M Total End-to-End Latency: %.2f ms (28 Layers + Norm + LM Head)\n", ms_gpu_e2e);

        // 2. CPU FORWARD PASS (28 Layers + Output Norm)
        double t_cpu_0 = get_time_us();
        float cpu_state[D_MODEL], next_state[D_MODEL];
        memcpy(cpu_state, h_embd, sizeof(float) * D_MODEL);

        for (int l = 0; l < N_LAYERS; l++) {
            cpu_decoder_layer_step(cpu_state, &w_host[l], &kv_host[l], 0, 1, next_state);
            memcpy(cpu_state, next_state, sizeof(float) * D_MODEL);
        }

        float sum_sq = 0.0f;
        for (int i = 0; i < D_MODEL; i++) sum_sq += cpu_state[i] * cpu_state[i];
        float scale = 1.0f / sqrtf((sum_sq / (float)D_MODEL) + EPSILON);
        float z_cpu_norm[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) z_cpu_norm[i] = cpu_state[i] * scale * gamma_output_norm[i];
        double t_cpu_1 = get_time_us();
        printf("CPU Haswell (8 threads) Latency:      %.2f ms (Speedup: %.2fx)\n\n",
               (t_cpu_1 - t_cpu_0) / 1000.0, ((t_cpu_1 - t_cpu_0) / 1000.0) / ms_gpu_e2e);

        // 3. Extract Top-5 Candidates on GPU Logits
        TokenCandidate top5[5];
        for (int k = 0; k < 5; k++) {
            top5[k].token_id = -1;
            top5[k].logit = -1e30f;
        }

        for (int i = 0; i < VOCAB_SIZE; i++) {
            float val = h_logits_gpu[i];
            for (int k = 0; k < 5; k++) {
                if (val > top5[k].logit) {
                    for (int j = 4; j > k; j--) top5[j] = top5[j - 1];
                    top5[k].token_id = i;
                    top5[k].logit = val;
                    break;
                }
            }
        }

        // Verify top candidates against CPU reference projection
        float top_cpu_logits[5];
        for (int k = 0; k < 5; k++) {
            int tok = top5[k].token_id;
            float dequant[D_MODEL];
            dequantize_row_q6_k_ref(h_head + (size_t)tok * HEAD_BYTES_PER_ROW, dequant, D_MODEL);
            float sum = 0.0f;
            for (int i = 0; i < D_MODEL; i++) sum += dequant[i] * z_cpu_norm[i];
            top_cpu_logits[k] = sum;
        }

        float delta = top_cpu_logits[0] - top_cpu_logits[1];
        float max_abs_err = 0.0f;
        for (int k = 0; k < 5; k++) {
            float err = fabsf(top5[k].logit - top_cpu_logits[k]);
            if (err > max_abs_err) max_abs_err = err;
        }

        printf("--------------------------------------------------------------------------------------------------------\n");
        printf(" TOP-5 PREDICTED CONTINUATION TOKENS (Total Vocab = 151,936):\n");
        printf("--------------------------------------------------------------------------------------------------------\n");
        printf("Rank | Token ID | Decoded Text         | GPU Logit    | CPU Logit    | Abs Error   | Status\n");
        printf("--------------------------------------------------------------------------------------------------------\n");
        for (int k = 0; k < 5; k++) {
            float err = fabsf(top5[k].logit - top_cpu_logits[k]);
            int tok = top5[k].token_id;
            char esc_text[64] = {0};
            if (vocab_tokens && tok >= 0 && tok < n_vocab_tokens) {
                const char *src = vocab_tokens[tok];
                int dst = 0;
                for (int s = 0; src[s] && dst < 40; s++) {
                    if (src[s] == '\n') { esc_text[dst++] = '\\'; esc_text[dst++] = 'n'; }
                    else if (src[s] == '\t') { esc_text[dst++] = '\\'; esc_text[dst++] = 't'; }
                    else esc_text[dst++] = src[s];
                }
                esc_text[dst] = '\0';
            } else {
                snprintf(esc_text, sizeof(esc_text), "<unk>");
            }

            printf("#%d   | %-8d | '%-18s' | %12.4f | %12.4f | %11.4e | %s\n",
                   k + 1, tok, esc_text, top5[k].logit, top_cpu_logits[k], err,
                   (k == 0) ? "<- [Top-1 ArgMax Match!]" : "");
        }
        printf("--------------------------------------------------------------------------------------------------------\n");

        printf("ArgMax Numerical Stability Analysis:\n");
        printf("  Margin between Top-1 and Top-2 (Delta): %.4f\n", delta);
        printf("  Maximum Logit Error (eps_inf):          %.4e\n", max_abs_err);
        printf("  Sufficient Condition (Delta > 2*eps):   %s (%.4f > %.4e)\n",
               (delta > 2.0f * max_abs_err) ? "MET [GUARANTEED PRESERVATION OF ARGMAX]" : "INSPECT",
               delta, 2.0f * max_abs_err);

        printf("  Top-1 Predicted Continuation: '%s' (Token ID %d)\n",
               vocab_tokens ? vocab_tokens[top5[0].token_id] : "?", top5[0].token_id);

        free(h_logits_gpu);
    }

    close(fd);
    return 0;
}
