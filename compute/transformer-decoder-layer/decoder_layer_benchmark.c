#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <CL/cl.h>

#define QK4_0 32
#define D_MODEL 1536
#define HEAD_DIM 128
#define N_HEADS_Q 12
#define N_HEADS_KV 2
#define GQA_GROUP_SIZE 6
#define D_QKV (D_MODEL + N_HEADS_KV * HEAD_DIM + N_HEADS_KV * HEAD_DIM) // 2048
#define D_FFN 8960
#define T_MAX 4096
#define ROPE_BASE 1000000.0f
#define EPSILON 1e-6f
#define PV_SEGMENT_SIZE 256
#define MAX_SEGMENTS (T_MAX / PV_SEGMENT_SIZE) // 16

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

static inline uint16_t float_to_fp16(float f) {
    return (uint16_t)_cvtss_sh(f, 0);
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

// AVX2 + FMA3 Dot-Product for 1 Q4_0 block (32 weights)
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
    int nb = K / QK4_0;

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

// CPU Full Reference Decoder Layer Pipeline (x -> Attn -> r -> FFN -> y)
void cpu_decoder_layer_pipeline(
    const float *x,
    const float *gamma_attn,
    const block_q4_0 *W_qkv,
    const float *b_qkv,
    const block_q4_0 *W_o,
    const float *gamma_ffn,
    const block_q4_0 *W_gate,
    const block_q4_0 *W_up,
    const block_q4_0 *W_down,
    float *k_cache,
    float *v_cache,
    int pos,
    int seq_len,
    float *y_final
) {
    // 1. RMSNorm (attn)
    float sum_sq1 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq1 += x[i] * x[i];
    }
    float scale1 = 1.0f / sqrtf((sum_sq1 / (float)D_MODEL) + EPSILON);
    float z1[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z1[i] = x[i] * scale1 * gamma_attn[i];
    }

    // 2. QKV Projection with Bias
    float qkv[D_QKV];
    cpu_gemv_q4_0_bias(W_qkv, z1, b_qkv, qkv, D_QKV, D_MODEL);

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
            k_cache[out_base + tid]      = k_rot0;
            k_cache[out_base + tid + 64] = k_rot1;

            v_cache[out_base + tid]      = v[in_base + tid];
            v_cache[out_base + tid + 64] = v[in_base + tid + 64];
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
                dot += q[q_off + d] * k_cache[k_off + d];
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
                acc += scores[t] * v_cache[v_off];
            }
            attn_out[h_q * HEAD_DIM + d] = acc;
        }
    }

    // 7. Output Projection Wo + Residual 1: r = x + Wo(attn_out)
    float wo_out[D_MODEL];
    cpu_gemv_q4_0_bias(W_o, attn_out, NULL, wo_out, D_MODEL, D_MODEL);
    float r[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        r[i] = x[i] + wo_out[i];
    }

    // 8. RMSNorm (ffn) on residual r
    float sum_sq2 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq2 += r[i] * r[i];
    }
    float scale2 = 1.0f / sqrtf((sum_sq2 / (float)D_MODEL) + EPSILON);
    float z2[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z2[i] = r[i] * scale2 * gamma_ffn[i];
    }

    // 9. Gate & Up GEMV + SiLU
    float g[D_FFN];
    float u[D_FFN];
    cpu_gemv_q4_0_bias(W_gate, z2, NULL, g, D_FFN, D_MODEL);
    cpu_gemv_q4_0_bias(W_up,   z2, NULL, u, D_FFN, D_MODEL);

    float h[D_FFN];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D_FFN; i++) {
        h[i] = stable_silu(g[i]) * u[i];
    }

    // 10. Down GEMV + Residual 2: y = r + Down(h)
    float down_out[D_MODEL];
    cpu_gemv_q4_0_bias(W_down, h, NULL, down_out, D_MODEL, D_FFN);
    for (int i = 0; i < D_MODEL; i++) {
        y_final[i] = r[i] + down_out[i];
    }
}

static char* load_kernel_source(const char* filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filepath);
        return NULL;
    }
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
    printf(" Complete Resident Transformer Decoder Layer on Kepler GT 750M (E2E Telemetry)        \n");
    printf(" Full Qwen2.5-Coder-1.5B Layer Spec (D=1536, FFN=8960, GQA 12:2, T_max=4096)           \n");
    printf("========================================================================================\n");

    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return 1;
    }
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

    if (!device) {
        fprintf(stderr, "No GPU OpenCL device found.\n");
        return 1;
    }

    char dname[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("Target Device: %s\n", dname);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    char *src = load_kernel_source("kernel_decoder_layer.cl");
    if (!src) return 1;

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    free(src);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char *log = (char*)malloc(log_len + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = '\0';
        fprintf(stderr, "OpenCL Build Error:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel k_rmsnorm      = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "kernel_rmsnorm");
    cl_kernel k_qkv_gemv     = clCreateKernel(program, "gemv_q4_0_bias", &err); CHECK_CL(err, "gemv_q4_0_bias");
    cl_kernel k_rope_kv      = clCreateKernel(program, "kernel_rope_and_kv_append", &err); CHECK_CL(err, "kernel_rope_and_kv_append");
    cl_kernel k_scores       = clCreateKernel(program, "kernel_gqa_scores", &err); CHECK_CL(err, "kernel_gqa_scores");
    cl_kernel k_softmax      = clCreateKernel(program, "kernel_softmax_gqa", &err); CHECK_CL(err, "kernel_softmax_gqa");
    cl_kernel k_pv_combine   = clCreateKernel(program, "kernel_gqa_value_combine_segmented", &err); CHECK_CL(err, "kernel_gqa_value_combine_segmented");
    cl_kernel k_pv_reduce    = clCreateKernel(program, "kernel_gqa_reduce_segments", &err); CHECK_CL(err, "kernel_gqa_reduce_segments");
    cl_kernel k_wo_residual  = clCreateKernel(program, "gemv_q4_0_wo_residual", &err); CHECK_CL(err, "gemv_q4_0_wo_residual");
    cl_kernel k_rmsnorm_ffn  = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "kernel_rmsnorm ffn");
    cl_kernel k_swiglu_fused = clCreateKernel(program, "gemv_swiglu_fused", &err); CHECK_CL(err, "gemv_swiglu_fused");
    cl_kernel k_down_res     = clCreateKernel(program, "gemv_q4_0_down_residual", &err); CHECK_CL(err, "gemv_q4_0_down_residual");

    printf("OpenCL Kernels compiled successfully.\n\n");

    srand(42);
    float *h_x           = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_gamma_attn  = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_b_qkv       = (float*)malloc(sizeof(float) * D_QKV);
    float *h_gamma_ffn   = (float*)malloc(sizeof(float) * D_MODEL);

    for (int i = 0; i < D_MODEL; i++) {
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_gamma_attn[i] = 1.0f + 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
        h_gamma_ffn[i]  = 1.0f + 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
    }
    for (int i = 0; i < D_QKV; i++) {
        h_b_qkv[i] = 0.01f * (((float)rand() / (float)RAND_MAX) - 0.5f);
    }

    int nb_qkv  = D_MODEL / QK4_0;
    int nb_wo   = D_MODEL / QK4_0;
    int nb_gate = D_MODEL / QK4_0;
    int nb_down = D_FFN / QK4_0;

    size_t sz_W_qkv  = sizeof(block_q4_0) * D_QKV * nb_qkv;
    size_t sz_W_o    = sizeof(block_q4_0) * D_MODEL * nb_wo;
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_up   = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * nb_down;

    block_q4_0 *h_W_qkv  = (block_q4_0*)malloc(sz_W_qkv);
    block_q4_0 *h_W_o    = (block_q4_0*)malloc(sz_W_o);
    block_q4_0 *h_W_gate = (block_q4_0*)malloc(sz_W_gate);
    block_q4_0 *h_W_up   = (block_q4_0*)malloc(sz_W_up);
    block_q4_0 *h_W_down = (block_q4_0*)malloc(sz_W_down);

    for (size_t i = 0; i < (size_t)D_QKV * nb_qkv; i++) {
        h_W_qkv[i].d = float_to_fp16(0.02f);
        for (int j = 0; j < 16; j++) h_W_qkv[i].qs[j] = (rand() & 0xFF);
    }
    for (size_t i = 0; i < (size_t)D_MODEL * nb_wo; i++) {
        h_W_o[i].d = float_to_fp16(0.02f);
        for (int j = 0; j < 16; j++) h_W_o[i].qs[j] = (rand() & 0xFF);
    }
    for (size_t i = 0; i < (size_t)D_FFN * nb_gate; i++) {
        h_W_gate[i].d = float_to_fp16(0.02f);
        for (int j = 0; j < 16; j++) h_W_gate[i].qs[j] = (rand() & 0xFF);
    }
    for (size_t i = 0; i < (size_t)D_FFN * nb_gate; i++) {
        h_W_up[i].d = float_to_fp16(0.02f);
        for (int j = 0; j < 16; j++) h_W_up[i].qs[j] = (rand() & 0xFF);
    }
    for (size_t i = 0; i < (size_t)D_MODEL * nb_down; i++) {
        h_W_down[i].d = float_to_fp16(0.02f);
        for (int j = 0; j < 16; j++) h_W_down[i].qs[j] = (rand() & 0xFF);
    }

    size_t sz_kv_cache = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;
    float *h_k_cache_cpu = (float*)malloc(sz_kv_cache);
    float *h_v_cache_cpu = (float*)malloc(sz_kv_cache);
    for (size_t i = 0; i < (size_t)N_HEADS_KV * T_MAX * HEAD_DIM; i++) {
        h_k_cache_cpu[i] = 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
        h_v_cache_cpu[i] = 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
    }

    float *h_y_cpu = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_y_gpu = (float*)malloc(sizeof(float) * D_MODEL);

    cl_mem d_W_qkv  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_qkv, h_W_qkv, &err); CHECK_CL(err, "d_W_qkv");
    cl_mem d_b_qkv  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_QKV, h_b_qkv, &err); CHECK_CL(err, "d_b_qkv");
    cl_mem d_W_o    = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_o, h_W_o, &err); CHECK_CL(err, "d_W_o");
    cl_mem d_W_gate = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, h_W_gate, &err); CHECK_CL(err, "d_W_gate");
    cl_mem d_W_up   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_up, h_W_up, &err); CHECK_CL(err, "d_W_up");
    cl_mem d_W_down = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_down, h_W_down, &err); CHECK_CL(err, "d_W_down");

    cl_mem d_gamma_attn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, h_gamma_attn, &err); CHECK_CL(err, "d_gamma_attn");
    cl_mem d_gamma_ffn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, h_gamma_ffn, &err); CHECK_CL(err, "d_gamma_ffn");

    cl_mem d_k_cache = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz_kv_cache, h_k_cache_cpu, &err); CHECK_CL(err, "d_k_cache");
    cl_mem d_v_cache = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz_kv_cache, h_v_cache_cpu, &err); CHECK_CL(err, "d_v_cache");

    // Resident Activation Arena with d_qkv reuse for attn_out (saves 6 KB and removes d_attn_out!)
    cl_mem d_state    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err); CHECK_CL(err, "d_state");
    cl_mem d_norm     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err); CHECK_CL(err, "d_norm");
    cl_mem d_qkv      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_QKV, NULL, &err); CHECK_CL(err, "d_qkv");
    cl_mem d_scores   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N_HEADS_Q * T_MAX, NULL, &err); CHECK_CL(err, "d_scores");
    cl_mem d_partial  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N_HEADS_Q * MAX_SEGMENTS * HEAD_DIM, NULL, &err); CHECK_CL(err, "d_partial");
    cl_mem d_h        = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_FFN, NULL, &err); CHECK_CL(err, "d_h");

    double total_weights_mb = (double)(sz_W_qkv + sz_W_o + sz_W_gate + sz_W_up + sz_W_down) / (1024.0 * 1024.0);
    double total_kv_mb = (double)(2 * sz_kv_cache) / (1024.0 * 1024.0);
    printf("Total Layer Weights in VRAM: %.2f MB\n", total_weights_mb);
    printf("Persistent KV Cache in VRAM: %.2f MB\n", total_kv_mb);
    printf("Resident Activation Arena:   0.33 MB (with d_qkv buffer reuse)\n\n");

    int context_lens[] = {1, 32, 128, 512, 1024, 2048};
    int num_contexts = sizeof(context_lens) / sizeof(context_lens[0]);

    printf("%-8s | %-10s | %-11s | %-11s | %-9s | %-12s | %-10s\n",
           "Tokens", "CPU (8T)", "GPU Compute", "Full E2E", "Speedup", "Max Abs Diff", "Cos Sim");
    printf("----------------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_contexts; c++) {
        int seq_len = context_lens[c];
        int pos = seq_len - 1;

        // Reset KV Cache from CPU baseline snapshot
        CHECK_CL(clEnqueueWriteBuffer(queue, d_k_cache, CL_TRUE, 0, sz_kv_cache, h_k_cache_cpu, 0, NULL, NULL), "reset k_cache");
        CHECK_CL(clEnqueueWriteBuffer(queue, d_v_cache, CL_TRUE, 0, sz_kv_cache, h_v_cache_cpu, 0, NULL, NULL), "reset v_cache");
        CHECK_CL(clEnqueueWriteBuffer(queue, d_state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_x, 0, NULL, NULL), "reset state");

        int d_model = D_MODEL;
        float eps = EPSILON;
        int param_d_qkv = D_QKV;
        int t_max = T_MAX;
        float rope_base = ROPE_BASE;
        float scale_factor = 1.0f / sqrtf((float)HEAD_DIM);
        int num_segments = (seq_len + PV_SEGMENT_SIZE - 1) / PV_SEGMENT_SIZE;
        int d_ffn = D_FFN;

        size_t k_offset_bytes = 1536 * sizeof(float);
        size_t v_offset_bytes = (1536 + 256) * sizeof(float);
        cl_buffer_region reg_k = { k_offset_bytes, 256 * sizeof(float) };
        cl_buffer_region reg_v = { v_offset_bytes, 256 * sizeof(float) };
        cl_mem d_k_sub = clCreateSubBuffer(d_qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_k, &err); CHECK_CL(err, "d_k_sub");
        cl_mem d_v_sub = clCreateSubBuffer(d_qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_v, &err); CHECK_CL(err, "d_v_sub");

        // Stage 1: RMSNorm (Attn)
        size_t l_rmsnorm = 128, g_rmsnorm = 128;
        clSetKernelArg(k_rmsnorm, 0, sizeof(cl_mem), &d_state);
        clSetKernelArg(k_rmsnorm, 1, sizeof(cl_mem), &d_gamma_attn);
        clSetKernelArg(k_rmsnorm, 2, sizeof(cl_mem), &d_norm);
        clSetKernelArg(k_rmsnorm, 3, sizeof(int), &d_model);
        clSetKernelArg(k_rmsnorm, 4, sizeof(float), &eps);

        // Stage 2: QKV Projection with Bias
        size_t l_qkv = 128, g_qkv = ((D_QKV + 3) / 4) * 128;
        clSetKernelArg(k_qkv_gemv, 0, sizeof(cl_mem), &d_W_qkv);
        clSetKernelArg(k_qkv_gemv, 1, sizeof(cl_mem), &d_norm);
        clSetKernelArg(k_qkv_gemv, 2, sizeof(cl_mem), &d_b_qkv);
        clSetKernelArg(k_qkv_gemv, 3, sizeof(cl_mem), &d_qkv);
        clSetKernelArg(k_qkv_gemv, 4, sizeof(int), &param_d_qkv);
        clSetKernelArg(k_qkv_gemv, 5, sizeof(int), &d_model);

        // Stage 3: RoPE and KV Cache Append
        size_t l_rope = 64, g_rope = 64;
        clSetKernelArg(k_rope_kv, 0, sizeof(cl_mem), &d_qkv);
        clSetKernelArg(k_rope_kv, 1, sizeof(cl_mem), &d_k_sub);
        clSetKernelArg(k_rope_kv, 2, sizeof(cl_mem), &d_v_sub);
        clSetKernelArg(k_rope_kv, 3, sizeof(cl_mem), &d_k_cache);
        clSetKernelArg(k_rope_kv, 4, sizeof(cl_mem), &d_v_cache);
        clSetKernelArg(k_rope_kv, 5, sizeof(int), &pos);
        clSetKernelArg(k_rope_kv, 6, sizeof(int), &t_max);
        clSetKernelArg(k_rope_kv, 7, sizeof(float), &rope_base);

        // Stage 4: Attention Scores
        size_t l_scores[2] = { 32, 1 };
        size_t g_scores[2] = { (size_t)seq_len * 32, (size_t)N_HEADS_Q };
        clSetKernelArg(k_scores, 0, sizeof(cl_mem), &d_qkv);
        clSetKernelArg(k_scores, 1, sizeof(cl_mem), &d_k_cache);
        clSetKernelArg(k_scores, 2, sizeof(cl_mem), &d_scores);
        clSetKernelArg(k_scores, 3, sizeof(int), &seq_len);
        clSetKernelArg(k_scores, 4, sizeof(int), &t_max);
        clSetKernelArg(k_scores, 5, sizeof(float), &scale_factor);

        // Stage 5: Softmax
        size_t l_soft = 128, g_soft = 12 * 128;
        clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores);
        clSetKernelArg(k_softmax, 1, sizeof(int), &seq_len);
        clSetKernelArg(k_softmax, 2, sizeof(int), &t_max);

        // Stage 6: Value Combination (Split-K Segmented)
        size_t l_pv[2] = { 128, 1 };
        size_t g_pv[2] = { (size_t)num_segments * 128, (size_t)N_HEADS_Q };
        clSetKernelArg(k_pv_combine, 0, sizeof(cl_mem), &d_scores);
        clSetKernelArg(k_pv_combine, 1, sizeof(cl_mem), &d_v_cache);
        clSetKernelArg(k_pv_combine, 2, sizeof(cl_mem), &d_partial);
        clSetKernelArg(k_pv_combine, 3, sizeof(int), &seq_len);
        clSetKernelArg(k_pv_combine, 4, sizeof(int), &t_max);
        clSetKernelArg(k_pv_combine, 5, sizeof(int), &num_segments);

        // Stage 6b: Reduce Segments into d_qkv[0:1536] (buffer reuse!)
        size_t l_red = 128, g_red = 12 * 128;
        clSetKernelArg(k_pv_reduce, 0, sizeof(cl_mem), &d_partial);
        clSetKernelArg(k_pv_reduce, 1, sizeof(cl_mem), &d_qkv); // Reuse d_qkv as attn_out!
        clSetKernelArg(k_pv_reduce, 2, sizeof(int), &num_segments);

        // Stage 7: Wo GEMV + Residual (reading from d_qkv!)
        size_t l_wo = 128, g_wo = ((D_MODEL + 3) / 4) * 128;
        clSetKernelArg(k_wo_residual, 0, sizeof(cl_mem), &d_W_o);
        clSetKernelArg(k_wo_residual, 1, sizeof(cl_mem), &d_qkv); // Reused buffer
        clSetKernelArg(k_wo_residual, 2, sizeof(cl_mem), &d_state);
        clSetKernelArg(k_wo_residual, 3, sizeof(int), &d_model);

        // Stage 8: RMSNorm (FFN)
        clSetKernelArg(k_rmsnorm_ffn, 0, sizeof(cl_mem), &d_state);
        clSetKernelArg(k_rmsnorm_ffn, 1, sizeof(cl_mem), &d_gamma_ffn);
        clSetKernelArg(k_rmsnorm_ffn, 2, sizeof(cl_mem), &d_norm);
        clSetKernelArg(k_rmsnorm_ffn, 3, sizeof(int), &d_model);
        clSetKernelArg(k_rmsnorm_ffn, 4, sizeof(float), &eps);

        // Stage 9: SwiGLU Fused Gate+Up+SiLU
        size_t l_swiglu = 128, g_swiglu = ((D_FFN + 3) / 4) * 128;
        clSetKernelArg(k_swiglu_fused, 0, sizeof(cl_mem), &d_W_gate);
        clSetKernelArg(k_swiglu_fused, 1, sizeof(cl_mem), &d_W_up);
        clSetKernelArg(k_swiglu_fused, 2, sizeof(cl_mem), &d_norm);
        clSetKernelArg(k_swiglu_fused, 3, sizeof(cl_mem), &d_h);
        clSetKernelArg(k_swiglu_fused, 4, sizeof(int), &d_ffn);
        clSetKernelArg(k_swiglu_fused, 5, sizeof(int), &d_model);

        // Stage 10: Down GEMV + In-place Residual
        size_t l_down = 128, g_down = ((D_MODEL + 3) / 4) * 128;
        clSetKernelArg(k_down_res, 0, sizeof(cl_mem), &d_W_down);
        clSetKernelArg(k_down_res, 1, sizeof(cl_mem), &d_h);
        clSetKernelArg(k_down_res, 2, sizeof(cl_mem), &d_state);
        clSetKernelArg(k_down_res, 3, sizeof(int), &d_model);
        clSetKernelArg(k_down_res, 4, sizeof(int), &d_ffn);

        // Single Warmup execution
        clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_qkv_gemv, 1, NULL, &g_qkv, &l_qkv, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_rope_kv, 1, NULL, &g_rope, &l_rope, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_scores, 2, NULL, g_scores, l_scores, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_softmax, 1, NULL, &g_soft, &l_soft, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_pv_combine, 2, NULL, g_pv, l_pv, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_pv_reduce, 1, NULL, &g_red, &l_red, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_wo_residual, 1, NULL, &g_wo, &l_wo, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_rmsnorm_ffn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_swiglu_fused, 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_down_res, 1, NULL, &g_down, &l_down, 0, NULL, NULL);
        clFinish(queue);

        CHECK_CL(clEnqueueReadBuffer(queue, d_state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_y_gpu, 0, NULL, NULL), "read y_gpu");

        // CPU Baseline execution
        cpu_decoder_layer_pipeline(
            h_x, h_gamma_attn, h_W_qkv, h_b_qkv, h_W_o,
            h_gamma_ffn, h_W_gate, h_W_up, h_W_down,
            h_k_cache_cpu, h_v_cache_cpu, pos, seq_len, h_y_cpu
        );

        // Compute numerical accuracy metrics
        float max_diff = 0.0f;
        double dot_prod = 0.0, norm_cpu = 0.0, norm_gpu = 0.0;
        for (int i = 0; i < D_MODEL; i++) {
            float diff = fabsf(h_y_gpu[i] - h_y_cpu[i]);
            if (diff > max_diff) max_diff = diff;
            dot_prod += (double)h_y_gpu[i] * (double)h_y_cpu[i];
            norm_cpu += (double)h_y_cpu[i] * (double)h_y_cpu[i];
            norm_gpu += (double)h_y_gpu[i] * (double)h_y_gpu[i];
        }
        double cos_sim = dot_prod / (sqrt(norm_cpu) * sqrt(norm_gpu));

        const int iters = 20;
        double total_gpu_compute_us = 0.0;
        double total_gpu_e2e_us = 0.0;
        double total_cpu_us = 0.0;

        for (int it = 0; it < iters; it++) {
            // Full E2E: Upload x -> Launch Kernels -> clFinish -> Download y
            double t_e2e_0 = get_time_us();
            clEnqueueWriteBuffer(queue, d_state, CL_FALSE, 0, sizeof(float) * D_MODEL, h_x, 0, NULL, NULL);

            double t_gpu_0 = get_time_us();
            clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_qkv_gemv, 1, NULL, &g_qkv, &l_qkv, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rope_kv, 1, NULL, &g_rope, &l_rope, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_scores, 2, NULL, g_scores, l_scores, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_softmax, 1, NULL, &g_soft, &l_soft, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_pv_combine, 2, NULL, g_pv, l_pv, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_pv_reduce, 1, NULL, &g_red, &l_red, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_wo_residual, 1, NULL, &g_wo, &l_wo, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rmsnorm_ffn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_swiglu_fused, 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_down_res, 1, NULL, &g_down, &l_down, 0, NULL, NULL);
            clFinish(queue);
            double t_gpu_1 = get_time_us();

            clEnqueueReadBuffer(queue, d_state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_y_gpu, 0, NULL, NULL);
            double t_e2e_1 = get_time_us();

            total_gpu_compute_us += (t_gpu_1 - t_gpu_0);
            total_gpu_e2e_us     += (t_e2e_1 - t_e2e_0);

            double t_cpu_0 = get_time_us();
            cpu_decoder_layer_pipeline(
                h_x, h_gamma_attn, h_W_qkv, h_b_qkv, h_W_o,
                h_gamma_ffn, h_W_gate, h_W_up, h_W_down,
                h_k_cache_cpu, h_v_cache_cpu, pos, seq_len, h_y_cpu
            );
            double t_cpu_1 = get_time_us();
            total_cpu_us += (t_cpu_1 - t_cpu_0);
        }

        clReleaseMemObject(d_k_sub);
        clReleaseMemObject(d_v_sub);

        double avg_gpu_compute_ms = (total_gpu_compute_us / (double)iters) / 1000.0;
        double avg_gpu_e2e_ms     = (total_gpu_e2e_us / (double)iters) / 1000.0;
        double avg_cpu_ms         = (total_cpu_us / (double)iters) / 1000.0;
        double speedup            = avg_cpu_ms / avg_gpu_e2e_ms;

        printf("T=%-6d | %7.2f ms | %8.2f ms | %8.2f ms | %7.2fx | %12.4e | %8.6f\n",
               seq_len, avg_cpu_ms, avg_gpu_compute_ms, avg_gpu_e2e_ms, speedup, max_diff, cos_sim);
    }

    printf("----------------------------------------------------------------------------------------\n");
    printf("Decoder Layer evaluation completed successfully.\n");

    clReleaseMemObject(d_W_qkv);
    clReleaseMemObject(d_b_qkv);
    clReleaseMemObject(d_W_o);
    clReleaseMemObject(d_W_gate);
    clReleaseMemObject(d_W_up);
    clReleaseMemObject(d_W_down);
    clReleaseMemObject(d_gamma_attn);
    clReleaseMemObject(d_gamma_ffn);
    clReleaseMemObject(d_k_cache);
    clReleaseMemObject(d_v_cache);
    clReleaseMemObject(d_state);
    clReleaseMemObject(d_norm);
    clReleaseMemObject(d_qkv);
    clReleaseMemObject(d_scores);
    clReleaseMemObject(d_partial);
    clReleaseMemObject(d_h);

    clReleaseKernel(k_rmsnorm);
    clReleaseKernel(k_qkv_gemv);
    clReleaseKernel(k_rope_kv);
    clReleaseKernel(k_scores);
    clReleaseKernel(k_softmax);
    clReleaseKernel(k_pv_combine);
    clReleaseKernel(k_pv_reduce);
    clReleaseKernel(k_wo_residual);
    clReleaseKernel(k_rmsnorm_ffn);
    clReleaseKernel(k_swiglu_fused);
    clReleaseKernel(k_down_res);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_x);
    free(h_gamma_attn);
    free(h_b_qkv);
    free(h_gamma_ffn);
    free(h_W_qkv);
    free(h_W_o);
    free(h_W_gate);
    free(h_W_up);
    free(h_W_down);
    free(h_k_cache_cpu);
    free(h_v_cache_cpu);
    free(h_y_cpu);
    free(h_y_gpu);

    return 0;
}
