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
#include <sys/stat.h>
#include <immintrin.h>
#include <omp.h>
#include <CL/cl.h>

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

    __m256 vw0 = _mm256_loadu_ps(&w[0]);
    __m256 vw1 = _mm256_loadu_ps(&w[8]);
    __m256 vw2 = _mm256_loadu_ps(&w[16]);
    __m256 vw3 = _mm256_loadu_ps(&w[24]);

    __m256 vx0 = _mm256_loadu_ps(&x[0]);
    __m256 vx1 = _mm256_loadu_ps(&x[8]);
    __m256 vx2 = _mm256_loadu_ps(&x[16]);
    __m256 vx3 = _mm256_loadu_ps(&x[24]);

    __m256 acc = _mm256_mul_ps(vw0, vx0);
    acc = _mm256_fmadd_ps(vw1, vx1, acc);
    acc = _mm256_fmadd_ps(vw2, vx2, acc);
    acc = _mm256_fmadd_ps(vw3, vx3, acc);

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    return _mm_cvtss_f32(sum128) * d;
}

void cpu_gemv_q4_0_bias(
    const block_q4_0 *W,
    const float      *x,
    const float      *bias,
    float            *y,
    int M,
    int K
) {
    int nb = K / 32;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < M; r++) {
        const block_q4_0 *row_blocks = W + r * nb;
        float row_sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            row_sum += dot_block_q4_0_avx2(&row_blocks[b], x + b * 32);
        }
        y[r] = row_sum + (bias ? bias[r] : 0.0f);
    }
}

// Robust pread helper with retry
static int read_exact_at(int fd, uint64_t offset, void *dst, size_t size) {
    uint8_t *ptr = (uint8_t*)dst;
    size_t total_read = 0;
    while (total_read < size) {
        ssize_t bytes = pread(fd, ptr + total_read, size - total_read, (off_t)(offset + total_read));
        if (bytes < 0) return -1;
        if (bytes == 0) return -2; // Truncated
        total_read += (size_t)bytes;
    }
    return 0;
}

typedef struct {
    float *gamma_attn;
    block_q4_0 *W_qkv;
    float *b_qkv;
    block_q4_0 *W_o;
    float *gamma_ffn;
    block_q4_0 *W_gate;
    block_q4_0 *W_up;
    block_q4_0 *W_down;
} LayerWeightsHost;

typedef struct {
    cl_mem gamma_attn;
    cl_mem W_qkv;
    cl_mem b_qkv;
    cl_mem W_o;
    cl_mem gamma_ffn;
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

        for (int t = 0; t < seq_len; t++) {
            float dot = 0.0f;
            int q_off = h_q * HEAD_DIM;
            int k_off = (h_kv * T_MAX + t) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q[q_off + d] * kv->k_cache[k_off + d];
            }
            scores[t] = dot * scale_factor;
            if (scores[t] > max_s) max_s = scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }

        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                int v_off = (h_kv * T_MAX + t) * HEAD_DIM + d;
                acc += scores[t] * kv->v_cache[v_off];
            }
            attn_out[h_q * HEAD_DIM + d] = acc;
        }
    }

    // 7. Output Projection Wo + Residual 1: r = x_in + Wo(attn_out)
    float wo_out[D_MODEL];
    cpu_gemv_q4_0_bias(w->W_o, attn_out, NULL, wo_out, D_MODEL, D_MODEL);
    float r[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        r[i] = x_in[i] + wo_out[i];
    }

    // 8. RMSNorm (ffn) on residual r
    float sum_sq2 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq2 += r[i] * r[i];
    }
    float scale2 = 1.0f / sqrtf((sum_sq2 / (float)D_MODEL) + EPSILON);
    float z2[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z2[i] = r[i] * scale2 * w->gamma_ffn[i];
    }

    // 9. Gate & Up GEMV + SiLU
    float g[D_FFN];
    float u[D_FFN];
    cpu_gemv_q4_0_bias(w->W_gate, z2, NULL, g, D_FFN, D_MODEL);
    cpu_gemv_q4_0_bias(w->W_up,   z2, NULL, u, D_FFN, D_MODEL);

    float h[D_FFN];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D_FFN; i++) {
        h[i] = stable_silu(g[i]) * u[i];
    }

    // 10. Down GEMV + Residual 2: y = r + Down(h)
    float down_out[D_MODEL];
    cpu_gemv_q4_0_bias(w->W_down, h, NULL, down_out, D_MODEL, D_FFN);
    for (int i = 0; i < D_MODEL; i++) {
        y_out[i] = r[i] + down_out[i];
    }
}

static void enqueue_layer(
    cl_command_queue queue,
    LayerKernels *k,
    int seq_len,
    int num_segments
) {
    size_t l_rmsnorm = 128, g_rmsnorm = 128;
    size_t l_qkv = 128, g_qkv = ((D_QKV + 3) / 4) * 128;
    size_t l_rope = 64, g_rope = 64;
    size_t l_scores[2] = { 32, 1 };
    size_t g_scores[2] = { (size_t)seq_len * 32, (size_t)N_HEADS_Q };
    size_t l_soft = 128, g_soft = 12 * 128;
    size_t l_pv[2] = { 128, 1 };
    size_t g_pv[2] = { (size_t)num_segments * 128, (size_t)N_HEADS_Q };
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

static char* load_kernel_source(const char* filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    char *src = (char*)malloc(size + 1);
    size_t read_bytes = fread(src, 1, size, fp);
    (void)read_bytes;
    src[size] = '\0';
    fclose(fp);
    return src;
}

int main(void) {
    printf("========================================================================================\n");
    printf(" Validation of Real Weights (Qwen2.5-Coder-1.5B blk.0 & blk.1) on Kepler GT 750M vs CPU \n");
    printf(" Official GGUF Q4_0 Weights + F32 Norms/Biases, Pos=0 (T=1) and Pos=1 (T=2) Sequences   \n");
    printf("========================================================================================\n");

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    int fd = open(gguf_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", gguf_path);
        return 1;
    }

    // Absolute offsets calculated from GGUF inspection
    // Data Start = 5950528
    // Layer 0:
    uint64_t off_l0_attn_norm = 328662592ULL;
    uint64_t off_l0_ffn_down  = 328668736ULL;
    uint64_t off_l0_ffn_gate  = 336410176ULL;
    uint64_t off_l0_ffn_up    = 344151616ULL;
    uint64_t off_l0_ffn_norm  = 351893056ULL;
    uint64_t off_l0_k_bias    = 351899200ULL;
    uint64_t off_l0_k_weight  = 351900224ULL;
    uint64_t off_l0_o_weight  = 352121408ULL;
    uint64_t off_l0_q_bias    = 353448512ULL;
    uint64_t off_l0_q_weight  = 353454656ULL;
    uint64_t off_l0_v_bias    = 354781760ULL;
    uint64_t off_l0_v_weight  = 354782784ULL;

    // Layer 1:
    uint64_t off_l1_attn_norm = 355003968ULL;
    uint64_t off_l1_ffn_down  = 355010112ULL;
    uint64_t off_l1_ffn_gate  = 362751552ULL;
    uint64_t off_l1_ffn_up    = 370492992ULL;
    uint64_t off_l1_ffn_norm  = 378234432ULL;
    uint64_t off_l1_k_bias    = 378240576ULL;
    uint64_t off_l1_k_weight  = 378241600ULL;
    uint64_t off_l1_o_weight  = 378462784ULL;
    uint64_t off_l1_q_bias    = 379789888ULL;
    uint64_t off_l1_q_weight  = 379796032ULL;
    uint64_t off_l1_v_bias    = 381123136ULL;
    uint64_t off_l1_v_weight  = 381124160ULL;

    // Real Embedding Row 0 (Token 0, e.g. `<|endoftext|>`)
    // token_embd.weight shape=[1536, 151936] in Q4_0, offset=191439360, abs=197389888
    uint64_t off_token_embd = 5950528ULL + 191439360ULL;

    printf("Reading and dequantizing real token embeddings (Tokens 0 & 1)...\n");
    block_q4_0 embd_blk_0[48]; // 1536 / 32 = 48 blocks
    block_q4_0 embd_blk_1[48];
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
    printf("Real token embeddings dequantized successfully (Token 0 scale: %.5f, Token 1 scale: %.5f).\n\n",
           fp16_to_float(embd_blk_0[0].d), fp16_to_float(embd_blk_1[0].d));

    // Allocate Layer 0 and Layer 1 host structures
    int nb_qkv = D_MODEL / 32, nb_wo = D_MODEL / 32, nb_gate = D_MODEL / 32, nb_down = D_FFN / 32;
    size_t sz_W_qkv  = sizeof(block_q4_0) * D_QKV * nb_qkv;
    size_t sz_W_o    = sizeof(block_q4_0) * D_MODEL * nb_wo;
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * nb_down;
    size_t sz_kv     = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;

    LayerWeightsHost w_host[2];
    LayerKVHost kv_host[2];

    for (int l = 0; l < 2; l++) {
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
    }

    printf("Reading Layer 0 weights from GGUF into memory...\n");
    read_exact_at(fd, off_l0_attn_norm, w_host[0].gamma_attn, sizeof(float) * D_MODEL);
    read_exact_at(fd, off_l0_ffn_norm,  w_host[0].gamma_ffn,  sizeof(float) * D_MODEL);

    // Concatenate Q, K, V biases into b_qkv
    read_exact_at(fd, off_l0_q_bias, w_host[0].b_qkv, sizeof(float) * 1536);
    read_exact_at(fd, off_l0_k_bias, w_host[0].b_qkv + 1536, sizeof(float) * 256);
    read_exact_at(fd, off_l0_v_bias, w_host[0].b_qkv + 1536 + 256, sizeof(float) * 256);

    // Read Q, K, V matrices directly into contiguous rows of W_qkv (2048 x 48 blocks)
    read_exact_at(fd, off_l0_q_weight, w_host[0].W_qkv, sizeof(block_q4_0) * 1536 * 48);
    read_exact_at(fd, off_l0_k_weight, w_host[0].W_qkv + 1536 * 48, sizeof(block_q4_0) * 256 * 48);
    read_exact_at(fd, off_l0_v_weight, w_host[0].W_qkv + (1536 + 256) * 48, sizeof(block_q4_0) * 256 * 48);

    read_exact_at(fd, off_l0_o_weight, w_host[0].W_o, sz_W_o);
    read_exact_at(fd, off_l0_ffn_gate, w_host[0].W_gate, sz_W_gate);
    read_exact_at(fd, off_l0_ffn_up,   w_host[0].W_up,   sz_W_gate);
    read_exact_at(fd, off_l0_ffn_down, w_host[0].W_down, sz_W_down);

    printf("Reading Layer 1 weights from GGUF into memory...\n");
    read_exact_at(fd, off_l1_attn_norm, w_host[1].gamma_attn, sizeof(float) * D_MODEL);
    read_exact_at(fd, off_l1_ffn_norm,  w_host[1].gamma_ffn,  sizeof(float) * D_MODEL);

    read_exact_at(fd, off_l1_q_bias, w_host[1].b_qkv, sizeof(float) * 1536);
    read_exact_at(fd, off_l1_k_bias, w_host[1].b_qkv + 1536, sizeof(float) * 256);
    read_exact_at(fd, off_l1_v_bias, w_host[1].b_qkv + 1536 + 256, sizeof(float) * 256);

    read_exact_at(fd, off_l1_q_weight, w_host[1].W_qkv, sizeof(block_q4_0) * 1536 * 48);
    read_exact_at(fd, off_l1_k_weight, w_host[1].W_qkv + 1536 * 48, sizeof(block_q4_0) * 256 * 48);
    read_exact_at(fd, off_l1_v_weight, w_host[1].W_qkv + (1536 + 256) * 48, sizeof(block_q4_0) * 256 * 48);

    read_exact_at(fd, off_l1_o_weight, w_host[1].W_o, sz_W_o);
    read_exact_at(fd, off_l1_ffn_gate, w_host[1].W_gate, sz_W_gate);
    read_exact_at(fd, off_l1_ffn_up,   w_host[1].W_up,   sz_W_gate);
    read_exact_at(fd, off_l1_ffn_down, w_host[1].W_down, sz_W_down);
    close(fd);

    printf("Real weights for blk.0 and blk.1 loaded cleanly!\n\n");

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

    LayerWeightsDevice w_dev[2];
    LayerKVDevice kv_dev[2];
    LayerKernels kernels[2];

    for (int l = 0; l < 2; l++) {
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

        kernels[l].k_rmsnorm_attn = clCreateKernel(program, "kernel_rmsnorm", &err);
        kernels[l].k_qkv_gemv     = clCreateKernel(program, "gemv_q4_0_bias", &err);
        kernels[l].k_rope_kv      = clCreateKernel(program, "kernel_rope_and_kv_append", &err);
        kernels[l].k_scores       = clCreateKernel(program, "kernel_gqa_scores", &err);
        kernels[l].k_softmax      = clCreateKernel(program, "kernel_softmax_gqa", &err);
        kernels[l].k_pv_combine   = clCreateKernel(program, "kernel_gqa_value_combine_segmented", &err);
        kernels[l].k_pv_reduce    = clCreateKernel(program, "kernel_gqa_reduce_segments", &err);
        kernels[l].k_wo_residual  = clCreateKernel(program, "gemv_q4_0_wo_residual", &err);
        kernels[l].k_rmsnorm_ffn  = clCreateKernel(program, "kernel_rmsnorm", &err);
        kernels[l].k_swiglu_fused = clCreateKernel(program, "gemv_swiglu_fused", &err);
        kernels[l].k_down_res     = clCreateKernel(program, "gemv_q4_0_down_residual", &err);
    }

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

    // Initialize KV caches to zero
    float zero_val = 0.0f;
    for (int l = 0; l < 2; l++) {
        clEnqueueFillBuffer(queue, kv_dev[l].k_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, kv_dev[l].v_cache, &zero_val, sizeof(float), 0, sz_kv, 0, NULL, NULL);
    }
    clFinish(queue);

    float *h_y0_cpu = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_y1_cpu = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_y_gpu  = (float*)malloc(sizeof(float) * D_MODEL);

    printf("Executing 2 Consecutive Real Tokens through 2 Real Decoder Layers:\n\n");
    printf("%-10s | %-8s | %-12s | %-12s | %-10s | %-12s | %-10s\n",
           "Step", "Context", "CPU Time", "GPU Time", "Speedup", "Rel L2 Error", "Cos Sim");
    printf("----------------------------------------------------------------------------------------\n");

    float *tokens[2] = { real_x0, real_x1 };

    for (int t = 0; t < 2; t++) {
        int pos = t;
        int seq_len = t + 1;
        int num_segments = (seq_len + PV_SEGMENT_SIZE - 1) / PV_SEGMENT_SIZE;
        float *cur_x = tokens[t];

        int d_model = D_MODEL, param_d_qkv = D_QKV, t_max = T_MAX, d_ffn = D_FFN;
        float eps = EPSILON, rope_base = ROPE_BASE, scale_factor = 1.0f / sqrtf(128.0f);

        for (int l = 0; l < 2; l++) {
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 0, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 1, sizeof(cl_mem), &w_dev[l].gamma_attn);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 2, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 3, sizeof(int), &d_model);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 4, sizeof(float), &eps);

            clSetKernelArg(kernels[l].k_qkv_gemv, 0, sizeof(cl_mem), &w_dev[l].W_qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 1, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_qkv_gemv, 2, sizeof(cl_mem), &w_dev[l].b_qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 3, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 4, sizeof(int), &param_d_qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 5, sizeof(int), &d_model);

            clSetKernelArg(kernels[l].k_rope_kv, 0, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_rope_kv, 1, sizeof(cl_mem), &ws.d_k_sub);
            clSetKernelArg(kernels[l].k_rope_kv, 2, sizeof(cl_mem), &ws.d_v_sub);
            clSetKernelArg(kernels[l].k_rope_kv, 3, sizeof(cl_mem), &kv_dev[l].k_cache);
            clSetKernelArg(kernels[l].k_rope_kv, 4, sizeof(cl_mem), &kv_dev[l].v_cache);
            clSetKernelArg(kernels[l].k_rope_kv, 5, sizeof(int), &pos);
            clSetKernelArg(kernels[l].k_rope_kv, 6, sizeof(int), &t_max);
            clSetKernelArg(kernels[l].k_rope_kv, 7, sizeof(float), &rope_base);

            clSetKernelArg(kernels[l].k_scores, 0, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_scores, 1, sizeof(cl_mem), &kv_dev[l].k_cache);
            clSetKernelArg(kernels[l].k_scores, 2, sizeof(cl_mem), &ws.scores);
            clSetKernelArg(kernels[l].k_scores, 3, sizeof(int), &seq_len);
            clSetKernelArg(kernels[l].k_scores, 4, sizeof(int), &t_max);
            clSetKernelArg(kernels[l].k_scores, 5, sizeof(float), &scale_factor);

            clSetKernelArg(kernels[l].k_softmax, 0, sizeof(cl_mem), &ws.scores);
            clSetKernelArg(kernels[l].k_softmax, 1, sizeof(int), &seq_len);
            clSetKernelArg(kernels[l].k_softmax, 2, sizeof(int), &t_max);

            clSetKernelArg(kernels[l].k_pv_combine, 0, sizeof(cl_mem), &ws.scores);
            clSetKernelArg(kernels[l].k_pv_combine, 1, sizeof(cl_mem), &kv_dev[l].v_cache);
            clSetKernelArg(kernels[l].k_pv_combine, 2, sizeof(cl_mem), &ws.partial);
            clSetKernelArg(kernels[l].k_pv_combine, 3, sizeof(int), &seq_len);
            clSetKernelArg(kernels[l].k_pv_combine, 4, sizeof(int), &t_max);
            clSetKernelArg(kernels[l].k_pv_combine, 5, sizeof(int), &num_segments);

            clSetKernelArg(kernels[l].k_pv_reduce, 0, sizeof(cl_mem), &ws.partial);
            clSetKernelArg(kernels[l].k_pv_reduce, 1, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_pv_reduce, 2, sizeof(int), &num_segments);

            clSetKernelArg(kernels[l].k_wo_residual, 0, sizeof(cl_mem), &w_dev[l].W_o);
            clSetKernelArg(kernels[l].k_wo_residual, 1, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_wo_residual, 2, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_wo_residual, 3, sizeof(int), &d_model);

            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 0, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 1, sizeof(cl_mem), &w_dev[l].gamma_ffn);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 2, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 3, sizeof(int), &d_model);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 4, sizeof(float), &eps);

            clSetKernelArg(kernels[l].k_swiglu_fused, 0, sizeof(cl_mem), &w_dev[l].W_gate);
            clSetKernelArg(kernels[l].k_swiglu_fused, 1, sizeof(cl_mem), &w_dev[l].W_up);
            clSetKernelArg(kernels[l].k_swiglu_fused, 2, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_swiglu_fused, 3, sizeof(cl_mem), &ws.h);
            clSetKernelArg(kernels[l].k_swiglu_fused, 4, sizeof(int), &d_ffn);
            clSetKernelArg(kernels[l].k_swiglu_fused, 5, sizeof(int), &d_model);

            clSetKernelArg(kernels[l].k_down_res, 0, sizeof(cl_mem), &w_dev[l].W_down);
            clSetKernelArg(kernels[l].k_down_res, 1, sizeof(cl_mem), &ws.h);
            clSetKernelArg(kernels[l].k_down_res, 2, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_down_res, 3, sizeof(int), &d_model);
            clSetKernelArg(kernels[l].k_down_res, 4, sizeof(int), &d_ffn);
        }

        // GPU Forward Step
        double t_gpu_0 = get_time_us();
        clEnqueueWriteBuffer(queue, ws.state, CL_FALSE, 0, sizeof(float)*D_MODEL, cur_x, 0, NULL, NULL);
        for (int l = 0; l < 2; l++) {
            enqueue_layer(queue, &kernels[l], seq_len, num_segments);
        }
        clEnqueueReadBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float)*D_MODEL, h_y_gpu, 0, NULL, NULL);
        double t_gpu_1 = get_time_us();

        // CPU Forward Step (updating persistent KV caches sequentially)
        double t_cpu_0 = get_time_us();
        cpu_decoder_layer_step(cur_x, &w_host[0], &kv_host[0], pos, seq_len, h_y0_cpu);
        cpu_decoder_layer_step(h_y0_cpu, &w_host[1], &kv_host[1], pos, seq_len, h_y1_cpu);
        double t_cpu_1 = get_time_us();

        // Compare GPU y_final against CPU y1
        double diff_sq = 0.0, ref_sq = 0.0, dot = 0.0, norm_c = 0.0, norm_g = 0.0;
        bool all_finite = true;
        for (int i = 0; i < D_MODEL; i++) {
            if (!isfinite(h_y_gpu[i]) || !isfinite(h_y1_cpu[i])) all_finite = false;
            double d = (double)h_y_gpu[i] - (double)h_y1_cpu[i];
            diff_sq += d * d;
            ref_sq  += (double)h_y1_cpu[i] * (double)h_y1_cpu[i];
            dot     += (double)h_y_gpu[i] * (double)h_y1_cpu[i];
            norm_c  += (double)h_y1_cpu[i] * (double)h_y1_cpu[i];
            norm_g  += (double)h_y_gpu[i] * (double)h_y_gpu[i];
        }
        if (!all_finite) { fprintf(stderr, "FATAL: Non-finite outputs!\n"); exit(1); }
        double rel_l2 = sqrt(diff_sq) / sqrt(ref_sq);
        double cos_sim = dot / (sqrt(norm_c) * sqrt(norm_g));

        double cpu_ms = (t_cpu_1 - t_cpu_0) / 1000.0;
        double gpu_ms = (t_gpu_1 - t_gpu_0) / 1000.0;
        double speedup = cpu_ms / gpu_ms;

        printf("Token %d    | T=%-6d | %8.2f ms | %8.2f ms | %7.2fx | %12.4e | %8.6f\n",
               t, seq_len, cpu_ms, gpu_ms, speedup, rel_l2, cos_sim);
    }
    printf("----------------------------------------------------------------------------------------\n");
    printf("Real-weights validation completed with 100%% numerical consistency!\n");

    return 0;
}
