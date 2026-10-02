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
#define D_QKV (D_MODEL + N_HEADS_KV * HEAD_DIM + N_HEADS_KV * HEAD_DIM) // 1536 + 256 + 256 = 2048
#define T_MAX 4096
#define ROPE_BASE 1000000.0f
#define EPSILON 1e-6f

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

// Explicit AVX2 + FMA3 Dot-Product for 1 Q4_0 block (32 weights)
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

// CPU Reference Full Attention Pipeline
void cpu_attention_gqa_pipeline(
    const float *x,
    const float *gamma,
    const block_q4_0 *W_qkv,
    const float *b_qkv,
    const block_q4_0 *W_o,
    float *k_cache,
    float *v_cache,
    int pos,
    int seq_len,
    float *attn_out,
    float *r_out
) {
    // 1. RMSNorm
    float sum_sq = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf((sum_sq / (float)D_MODEL) + EPSILON);
    float z[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z[i] = x[i] * scale * gamma[i];
    }

    // 2. QKV Projection with Bias
    float qkv[D_QKV];
    cpu_gemv_q4_0_bias(W_qkv, z, b_qkv, qkv, D_QKV, D_MODEL);

    float *q = qkv;
    float *k = qkv + 1536;
    float *v = qkv + 1536 + 256;

    // 3. RoPE on Q & K
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

    // 4 & 5 & 6. GQA Attention Scores -> Softmax -> Value Combination
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
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
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

    // 7. Output Projection Wo + Residual Add
    float y_o[D_MODEL];
    cpu_gemv_q4_0_bias(W_o, attn_out, NULL, y_o, D_MODEL, D_MODEL);
    for (int i = 0; i < D_MODEL; i++) {
        r_out[i] = x[i] + y_o[i];
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
    fread(src, 1, size, fp);
    src[size] = '\0';
    fclose(fp);
    return src;
}

int main(void) {
    printf("===================================================================\n");
    printf("  Resident Grouped-Query Attention (GQA) Engine on GT 750M vs CPU  \n");
    printf("  Qwen2.5-Coder-1.5B (D=1536, H_q=12, H_kv=2, d=128, T_max=4096)   \n");
    printf("===================================================================\n");

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
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) == CL_SUCCESS) {
            char dev_name[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
            printf("Device: %s (Mesa Rusticl OpenCL)\n", dev_name);
            break;
        }
    }
    free(platforms);

    if (!device) {
        fprintf(stderr, "No GPU device found.\n");
        return 1;
    }

    int num_threads = omp_get_max_threads();
    printf("CPU OpenMP Threads: %d threads\n", num_threads);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    char *src = load_kernel_source("kernel_attention_gqa.cl");
    if (!src) { return 1; }

    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    free(src);

    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Kernel build failed: %d\n", err);
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Log:\n%s\n", log);
        return 1;
    }
    printf("✓ OpenCL Attention GQA Program built successfully!\n\n");

    // Kernels
    cl_kernel k_rmsnorm      = clCreateKernel(prog, "kernel_rmsnorm", &err);
    CHECK_CL(err, "k_rmsnorm");
    cl_kernel k_qkv_gemv     = clCreateKernel(prog, "gemv_q4_0_bias", &err);
    CHECK_CL(err, "k_qkv_gemv");
    cl_kernel k_rope_append  = clCreateKernel(prog, "kernel_rope_and_kv_append", &err);
    CHECK_CL(err, "k_rope_append");
    cl_kernel k_gqa_scores   = clCreateKernel(prog, "kernel_gqa_scores", &err);
    CHECK_CL(err, "k_gqa_scores");
    cl_kernel k_softmax      = clCreateKernel(prog, "kernel_softmax_gqa", &err);
    CHECK_CL(err, "k_softmax");
    cl_kernel k_val_segmented = clCreateKernel(prog, "kernel_gqa_value_combine_segmented", &err);
    CHECK_CL(err, "k_val_segmented");
    cl_kernel k_val_reduce    = clCreateKernel(prog, "kernel_gqa_reduce_segments", &err);
    CHECK_CL(err, "k_val_reduce");
    cl_kernel k_wo_residual   = clCreateKernel(prog, "gemv_q4_0_wo_residual", &err);
    CHECK_CL(err, "k_wo_residual");

    // Dimensions
    assert(D_MODEL % 64 == 0);
    assert(D_QKV % 64 == 0);

    int nb_qkv = D_MODEL / QK4_0;
    int nb_wo  = D_MODEL / QK4_0;
    size_t w_qkv_bytes = (size_t)D_QKV * nb_qkv * sizeof(block_q4_0);
    size_t b_qkv_bytes = (size_t)D_QKV * sizeof(float);
    size_t w_wo_bytes  = (size_t)D_MODEL * nb_wo * sizeof(block_q4_0);
    size_t gamma_bytes = D_MODEL * sizeof(float);

    // Resident KV Cache Buffer: [2 heads, 4096 tokens, 128 dim] = 8 MB each
    size_t kv_cache_bytes = (size_t)N_HEADS_KV * T_MAX * HEAD_DIM * sizeof(float);
    size_t scores_bytes   = (size_t)N_HEADS_Q * T_MAX * sizeof(float);

    printf("📦 Resident VRAM Footprint:\n");
    printf("  • W_qkv (2048 x 1536) Q4_0: %6.2f MB\n", w_qkv_bytes / (1024.0 * 1024.0));
    printf("  • Bias_qkv (2048 floats):    %6.2f KB\n", b_qkv_bytes / 1024.0);
    printf("  • W_o (1536 x 1536) Q4_0:   %6.2f MB\n", w_wo_bytes / (1024.0 * 1024.0));
    printf("  • Resident K-Cache (4096):  %6.2f MB\n", kv_cache_bytes / (1024.0 * 1024.0));
    printf("  • Resident V-Cache (4096):  %6.2f MB\n", kv_cache_bytes / (1024.0 * 1024.0));
    printf("  • Gamma + Scratch buffers:  %6.2f MB\n", (gamma_bytes + scores_bytes + D_QKV*4) / (1024.0 * 1024.0));
    printf("  Total Resident Memory:     ~%6.2f MB (Well within 2 GB VRAM!)\n\n",
           (w_qkv_bytes + b_qkv_bytes + w_wo_bytes + 2*kv_cache_bytes + scores_bytes) / (1024.0 * 1024.0));

    // Allocate Host buffers
    block_q4_0 *h_W_qkv = (block_q4_0*)aligned_alloc(64, w_qkv_bytes);
    float      *h_b_qkv = (float*)aligned_alloc(64, b_qkv_bytes);
    block_q4_0 *h_W_o   = (block_q4_0*)aligned_alloc(64, w_wo_bytes);
    float      *h_gamma = (float*)aligned_alloc(64, gamma_bytes);

    float *h_x          = (float*)aligned_alloc(64, D_MODEL * sizeof(float));
    float *h_k_cache    = (float*)aligned_alloc(64, kv_cache_bytes);
    float *h_v_cache    = (float*)aligned_alloc(64, kv_cache_bytes);
    float *h_attn_cpu   = (float*)aligned_alloc(64, D_MODEL * sizeof(float));
    float *h_r_cpu      = (float*)aligned_alloc(64, D_MODEL * sizeof(float));
    float *h_r_gpu      = (float*)aligned_alloc(64, D_MODEL * sizeof(float));

    // Initialize deterministic weights
    srand(42);
    for (size_t i = 0; i < (size_t)D_QKV * nb_qkv; i++) {
        h_W_qkv[i].d = float_to_fp16(0.015f + (float)(rand() % 100) * 0.0001f);
        for (int j = 0; j < 16; j++) h_W_qkv[i].qs[j] = (uint8_t)(rand() % 256);
    }
    for (int i = 0; i < D_QKV; i++) h_b_qkv[i] = ((float)(rand() % 200) - 100.0f) * 0.001f;

    for (size_t i = 0; i < (size_t)D_MODEL * nb_wo; i++) {
        h_W_o[i].d = float_to_fp16(0.012f + (float)(rand() % 100) * 0.0001f);
        for (int j = 0; j < 16; j++) h_W_o[i].qs[j] = (uint8_t)(rand() % 256);
    }
    for (int i = 0; i < D_MODEL; i++) {
        h_gamma[i] = 1.0f + ((float)(rand() % 100) - 50.0f) * 0.001f;
        h_x[i]     = ((float)(rand() % 1000) - 500.0f) * 0.01f;
    }
    // Pre-populate KV cache with random historical context
    for (size_t i = 0; i < (size_t)N_HEADS_KV * T_MAX * HEAD_DIM; i++) {
        h_k_cache[i] = ((float)(rand() % 200) - 100.0f) * 0.01f;
        h_v_cache[i] = ((float)(rand() % 200) - 100.0f) * 0.01f;
    }

    // Allocate GPU Device buffers
    cl_mem d_W_qkv    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_qkv_bytes, h_W_qkv, &err);
    cl_mem d_b_qkv    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_qkv_bytes, h_b_qkv, &err);
    cl_mem d_W_o      = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_wo_bytes, h_W_o, &err);
    cl_mem d_gamma    = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gamma_bytes, h_gamma, &err);

    cl_mem d_k_cache  = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, kv_cache_bytes, h_k_cache, &err);
    cl_mem d_v_cache  = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, kv_cache_bytes, h_v_cache, &err);

    cl_mem d_x        = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_z        = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_qkv      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_QKV * sizeof(float), NULL, &err);
    cl_mem d_scores   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, scores_bytes, NULL, &err);
    // Partial PV Buffer: [12 heads, 32 max segments, 128 dim] = 192 KB
    const int MAX_SEGMENTS = 32;
    size_t partial_pv_bytes = (size_t)N_HEADS_Q * MAX_SEGMENTS * HEAD_DIM * sizeof(float);
    cl_mem d_pv_partial = clCreateBuffer(ctx, CL_MEM_READ_WRITE, partial_pv_bytes, NULL, &err);
    CHECK_CL(err, "d_pv_partial");

    cl_mem d_attn_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_r_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);

    // Bind Persistent Arguments
    // 1. RMSNorm
    float eps = EPSILON;
    int d_model = D_MODEL;
    clSetKernelArg(k_rmsnorm, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_rmsnorm, 1, sizeof(cl_mem), &d_gamma);
    clSetKernelArg(k_rmsnorm, 2, sizeof(cl_mem), &d_z);
    clSetKernelArg(k_rmsnorm, 3, sizeof(int), &d_model);
    clSetKernelArg(k_rmsnorm, 4, sizeof(float), &eps);

    // 2. QKV GEMV
    int m_qkv = D_QKV;
    clSetKernelArg(k_qkv_gemv, 0, sizeof(cl_mem), &d_W_qkv);
    clSetKernelArg(k_qkv_gemv, 1, sizeof(cl_mem), &d_z);
    clSetKernelArg(k_qkv_gemv, 2, sizeof(cl_mem), &d_b_qkv);
    clSetKernelArg(k_qkv_gemv, 3, sizeof(cl_mem), &d_qkv);
    clSetKernelArg(k_qkv_gemv, 4, sizeof(int), &m_qkv);
    clSetKernelArg(k_qkv_gemv, 5, sizeof(int), &d_model);

    // 3. RoPE & Append (Bound in loop per pos)
    int t_max = T_MAX;
    float rope_base = ROPE_BASE;

    // 7. Wo Residual GEMV
    clSetKernelArg(k_wo_residual, 0, sizeof(cl_mem), &d_W_o);
    clSetKernelArg(k_wo_residual, 1, sizeof(cl_mem), &d_attn_out);
    clSetKernelArg(k_wo_residual, 2, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_wo_residual, 3, sizeof(cl_mem), &d_r_out);
    clSetKernelArg(k_wo_residual, 4, sizeof(int), &d_model);

    // Context lengths to sweep
    int sweep_contexts[] = {1, 32, 128, 512, 1024, 2048, 4096};
    int num_sweeps = sizeof(sweep_contexts) / sizeof(sweep_contexts[0]);

    printf("===================================================================\n");
    printf("   Context Length Sweep Benchmark: Kepler GT 750M vs Haswell AVX2   \n");
    printf("===================================================================\n");
    printf(" %-8s | %-12s | %-12s | %-10s | %-10s\n", "Context", "CPU Time", "GPU Time", "Speedup", "Max Error");
    printf("-------------------------------------------------------------------\n");

    const int ITERS = 50;

    for (int s = 0; s < num_sweeps; s++) {
        int seq_len = sweep_contexts[s];
        int pos = seq_len - 1; // appending at the end of context

        // 1. CPU Reference
        double t_cpu_start = get_time_us();
        for (int it = 0; it < ITERS; it++) {
            cpu_attention_gqa_pipeline(h_x, h_gamma, h_W_qkv, h_b_qkv, h_W_o, h_k_cache, h_v_cache, pos, seq_len, h_attn_cpu, h_r_cpu);
        }
        double t_cpu_end = get_time_us();
        double cpu_ms = (t_cpu_end - t_cpu_start) / (ITERS * 1000.0);

        // 2. Set dynamic kernel args for GPU
        clSetKernelArg(k_rope_append, 0, sizeof(cl_mem), &d_qkv);
        // k buffer points to qkv + 1536
        // in our flat buffer, pass d_qkv and handle in kernel or create sub-buffers
        // here kernel expects pointers: pass d_qkv directly since Q, K, V are contiguous!
        // We adjust kernel args:
        // Actually kernel expects q, k, v as separate pointers. In OpenCL, create sub-buffers:
        cl_buffer_region reg_q = {0, 1536 * sizeof(float)};
        cl_buffer_region reg_k = {1536 * sizeof(float), 256 * sizeof(float)};
        cl_buffer_region reg_v = {(1536 + 256) * sizeof(float), 256 * sizeof(float)};

        cl_mem d_q = clCreateSubBuffer(d_qkv, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg_q, &err);
        cl_mem d_k = clCreateSubBuffer(d_qkv, CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &reg_k, &err);
        cl_mem d_v = clCreateSubBuffer(d_qkv, CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &reg_v, &err);

        clSetKernelArg(k_rope_append, 0, sizeof(cl_mem), &d_q);
        clSetKernelArg(k_rope_append, 1, sizeof(cl_mem), &d_k);
        clSetKernelArg(k_rope_append, 2, sizeof(cl_mem), &d_v);
        clSetKernelArg(k_rope_append, 3, sizeof(cl_mem), &d_k_cache);
        clSetKernelArg(k_rope_append, 4, sizeof(cl_mem), &d_v_cache);
        clSetKernelArg(k_rope_append, 5, sizeof(int), &pos);
        clSetKernelArg(k_rope_append, 6, sizeof(int), &t_max);
        clSetKernelArg(k_rope_append, 7, sizeof(float), &rope_base);

        float scale_f = 1.0f / sqrtf((float)HEAD_DIM);
        clSetKernelArg(k_gqa_scores, 0, sizeof(cl_mem), &d_q);
        clSetKernelArg(k_gqa_scores, 1, sizeof(cl_mem), &d_k_cache);
        clSetKernelArg(k_gqa_scores, 2, sizeof(cl_mem), &d_scores);
        clSetKernelArg(k_gqa_scores, 3, sizeof(int), &seq_len);
        clSetKernelArg(k_gqa_scores, 4, sizeof(int), &t_max);
        clSetKernelArg(k_gqa_scores, 5, sizeof(float), &scale_f);

        clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &d_scores);
        clSetKernelArg(k_softmax, 1, sizeof(int), &seq_len);
        clSetKernelArg(k_softmax, 2, sizeof(int), &t_max);

        int num_segments = (seq_len + 255) / 256;
        if (num_segments > 32) num_segments = 32;

        clSetKernelArg(k_val_segmented, 0, sizeof(cl_mem), &d_scores);
        clSetKernelArg(k_val_segmented, 1, sizeof(cl_mem), &d_v_cache);
        clSetKernelArg(k_val_segmented, 2, sizeof(cl_mem), &d_pv_partial);
        clSetKernelArg(k_val_segmented, 3, sizeof(int), &seq_len);
        clSetKernelArg(k_val_segmented, 4, sizeof(int), &t_max);
        clSetKernelArg(k_val_segmented, 5, sizeof(int), &num_segments);

        clSetKernelArg(k_val_reduce, 0, sizeof(cl_mem), &d_pv_partial);
        clSetKernelArg(k_val_reduce, 1, sizeof(cl_mem), &d_attn_out);
        clSetKernelArg(k_val_reduce, 2, sizeof(int), &num_segments);

        // NDRanges
        size_t local_norm = 128, global_norm = 128;
        size_t local_warp = 128;
        size_t global_qkv = ((D_QKV + 3) / 4) * 128;
        size_t global_wo  = ((D_MODEL + 3) / 4) * 128;

        size_t local_rope = 64, global_rope = 64;
        size_t local_scores[2] = {32, 1};
        size_t global_scores[2] = { (size_t)seq_len * 32, (size_t)N_HEADS_Q };
        size_t local_soft = 128, global_soft = 12 * 128;

        // Split-K Segmented PV NDRange: [HEAD_DIM, num_segments, N_HEADS_Q]
        size_t local_val_seg[2]  = {128, 1};
        size_t global_val_seg[2] = { (size_t)128 * num_segments, (size_t)N_HEADS_Q };
        size_t local_val_red     = 128;
        size_t global_val_red    = 12 * 128;

        // Warmup GPU
        for (int it = 0; it < 3; it++) {
            clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_qkv_gemv, 1, NULL, &global_qkv, &local_warp, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rope_append, 1, NULL, &global_rope, &local_rope, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_gqa_scores, 2, NULL, global_scores, local_scores, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_softmax, 1, NULL, &global_soft, &local_soft, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_val_segmented, 2, NULL, global_val_seg, local_val_seg, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_val_reduce, 1, NULL, &global_val_red, &local_val_red, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_wo_residual, 1, NULL, &global_wo, &local_warp, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_r_out, CL_TRUE, 0, D_MODEL * sizeof(float), h_r_gpu, 0, NULL, NULL);
        }
        clFinish(queue);

        // Benchmark GPU
        double t_gpu_start = get_time_us();
        for (int it = 0; it < ITERS; it++) {
            cl_event ev_done;
            clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_qkv_gemv, 1, NULL, &global_qkv, &local_warp, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rope_append, 1, NULL, &global_rope, &local_rope, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_gqa_scores, 2, NULL, global_scores, local_scores, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_softmax, 1, NULL, &global_soft, &local_soft, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_val_segmented, 2, NULL, global_val_seg, local_val_seg, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_val_reduce, 1, NULL, &global_val_red, &local_val_red, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_wo_residual, 1, NULL, &global_wo, &local_warp, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_r_out, CL_FALSE, 0, D_MODEL * sizeof(float), h_r_gpu, 0, NULL, &ev_done);
            clWaitForEvents(1, &ev_done);
            clReleaseEvent(ev_done);
        }
        double t_gpu_end = get_time_us();
        double gpu_ms = (t_gpu_end - t_gpu_start) / (ITERS * 1000.0);

        // Validation (accums over up to 4096 softmax weights)
        double max_err = 0.0;
        const double atol = 5e-4;
        const double rtol = 2e-3;
        for (int i = 0; i < D_MODEL; i++) {
            if (!isfinite(h_r_gpu[i])) {
                fprintf(stderr, "FATAL: Non-finite output at r[%d] in context %d!\n", i, seq_len);
                exit(1);
            }
            double diff = fabs((double)h_r_cpu[i] - (double)h_r_gpu[i]);
            if (diff > max_err) max_err = diff;
            double tol = atol + rtol * fabs((double)h_r_cpu[i]);
            if (diff > tol) {
                fprintf(stderr, "FATAL: Error %e exceeded tol %e at r[%d] in context %d!\n", diff, tol, i, seq_len);
                exit(1);
            }
        }

        printf(" T = %-5d | %8.2f ms | %8.2f ms | %8.2fx | %9.2e  ✓\n",
               seq_len, cpu_ms, gpu_ms, cpu_ms / gpu_ms, max_err);

        clReleaseMemObject(d_q);
        clReleaseMemObject(d_k);
        clReleaseMemObject(d_v);
    }
    printf("-------------------------------------------------------------------\n");
    printf(" ✓ All context sweeps mathematically validated against AVX2 CPU!\n");
    printf("===================================================================\n");

    // Cleanup
    clReleaseMemObject(d_W_qkv);
    clReleaseMemObject(d_b_qkv);
    clReleaseMemObject(d_W_o);
    clReleaseMemObject(d_gamma);
    clReleaseMemObject(d_k_cache);
    clReleaseMemObject(d_v_cache);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_z);
    clReleaseMemObject(d_qkv);
    clReleaseMemObject(d_scores);
    clReleaseMemObject(d_pv_partial);
    clReleaseMemObject(d_attn_out);
    clReleaseMemObject(d_r_out);

    clReleaseKernel(k_rmsnorm);
    clReleaseKernel(k_qkv_gemv);
    clReleaseKernel(k_rope_append);
    clReleaseKernel(k_gqa_scores);
    clReleaseKernel(k_softmax);
    clReleaseKernel(k_val_segmented);
    clReleaseKernel(k_val_reduce);
    clReleaseKernel(k_wo_residual);

    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(h_W_qkv);
    free(h_b_qkv);
    free(h_W_o);
    free(h_gamma);
    free(h_x);
    free(h_k_cache);
    free(h_v_cache);
    free(h_attn_cpu);
    free(h_r_cpu);
    free(h_r_gpu);

    return 0;
}
