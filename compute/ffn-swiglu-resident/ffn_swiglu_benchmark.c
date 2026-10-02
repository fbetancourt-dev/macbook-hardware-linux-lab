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
#define D_FFN   8960
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

// Numerically stable SiLU
static inline float stable_silu(float g) {
    float sig;
    if (g >= 0.0f) {
        sig = 1.0f / (1.0f + expf(-g));
    } else {
        float eg = expf(g);
        sig = eg / (1.0f + eg);
    }
    return g * sig;
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

void cpu_gemv_q4_0(
    const block_q4_0 *W,
    const float      *x,
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
        y[r] = row_sum;
    }
}

void cpu_swiglu_pipeline(
    const float *x,
    const float *gamma,
    const block_q4_0 *W_gate,
    const block_q4_0 *W_up,
    const block_q4_0 *W_down,
    float *z,
    float *g,
    float *u,
    float *h,
    float *ydown,
    float *y
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf((sum_sq / (float)D_MODEL) + EPSILON);
    for (int i = 0; i < D_MODEL; i++) {
        z[i] = x[i] * scale * gamma[i];
    }

    cpu_gemv_q4_0(W_gate, z, g, D_FFN, D_MODEL);
    cpu_gemv_q4_0(W_up, z, u, D_FFN, D_MODEL);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D_FFN; i++) {
        h[i] = stable_silu(g[i]) * u[i];
    }

    cpu_gemv_q4_0(W_down, h, ydown, D_MODEL, D_FFN);

    for (int i = 0; i < D_MODEL; i++) {
        y[i] = x[i] + ydown[i];
    }
}

char* load_kernel_source(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Error opening %s\n", filename); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char*)malloc(len + 1);
    size_t r = fread(src, 1, len, f);
    (void)r;
    src[len] = '\0';
    fclose(f);
    return src;
}

static inline double event_duration_us(cl_event ev) {
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
    return (double)(t_end - t_start) * 1e-3;
}

int main() {
    printf("===================================================================\n");
    printf("  Resident SwiGLU FFN Engine on Kepler GT 750M vs Haswell AVX2 (v3)\n");
    printf("  Comparing: Baseline Modular (6 stages) vs Fused Pipeline (3 stages)\n");
    printf("===================================================================\n");

    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("Device: %s (Mesa Rusticl OpenCL)\n", dev_name);
    printf("CPU OpenMP Threads: %d threads\n", omp_get_max_threads());

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    char *src = load_kernel_source("kernel_ffn_swiglu.cl");
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
    printf("✓ OpenCL SwiGLU Program built successfully!\n\n");

    // Create Dedicated Kernel Objects (per Cloud Sam review)
    cl_kernel k_rmsnorm  = clCreateKernel(prog, "kernel_rmsnorm", &err);
    CHECK_CL(err, "k_rmsnorm");
    cl_kernel k_gate     = clCreateKernel(prog, "gemv_q4_0_dual_block", &err);
    CHECK_CL(err, "k_gate");
    cl_kernel k_up       = clCreateKernel(prog, "gemv_q4_0_dual_block", &err);
    CHECK_CL(err, "k_up");
    cl_kernel k_swiglu   = clCreateKernel(prog, "kernel_swiglu", &err);
    CHECK_CL(err, "k_swiglu");
    cl_kernel k_down     = clCreateKernel(prog, "gemv_q4_0_dual_block", &err);
    CHECK_CL(err, "k_down");
    cl_kernel k_residual = clCreateKernel(prog, "kernel_residual_add", &err);
    CHECK_CL(err, "k_residual");

    // Fused Kernels
    cl_kernel k_fused_gate_up = clCreateKernel(prog, "gemv_swiglu_fused", &err);
    CHECK_CL(err, "k_fused_gate_up");
    cl_kernel k_fused_down_res = clCreateKernel(prog, "gemv_q4_0_down_residual", &err);
    CHECK_CL(err, "k_fused_down_res");

    // Dimensions Assertions (guarantees n_pairs in dual-block reduction doesn't drop blocks)
    assert(D_MODEL % 64 == 0);
    assert(D_FFN % 64 == 0);

    int nb_gate = D_MODEL / QK4_0;
    int nb_down = D_FFN / QK4_0;
    size_t w_gate_bytes = (size_t)D_FFN * nb_gate * sizeof(block_q4_0);
    size_t w_up_bytes   = (size_t)D_FFN * nb_gate * sizeof(block_q4_0);
    size_t w_down_bytes = (size_t)D_MODEL * nb_down * sizeof(block_q4_0);
    size_t gamma_bytes  = D_MODEL * sizeof(float);

    block_q4_0 *h_W_gate = (block_q4_0*)malloc(w_gate_bytes);
    block_q4_0 *h_W_up   = (block_q4_0*)malloc(w_up_bytes);
    block_q4_0 *h_W_down = (block_q4_0*)malloc(w_down_bytes);
    float *h_gamma = (float*)malloc(gamma_bytes);

    for (size_t i = 0; i < (size_t)D_FFN * nb_gate; i++) {
        h_W_gate[i].d = float_to_fp16(0.015f + (float)(rand() % 100) * 0.0001f);
        h_W_up[i].d   = float_to_fp16(0.012f + (float)(rand() % 100) * 0.0001f);
        for (int j = 0; j < 16; j++) {
            h_W_gate[i].qs[j] = (uint8_t)(rand() % 256);
            h_W_up[i].qs[j]   = (uint8_t)(rand() % 256);
        }
    }
    for (size_t i = 0; i < (size_t)D_MODEL * nb_down; i++) {
        h_W_down[i].d = float_to_fp16(0.010f + (float)(rand() % 100) * 0.0001f);
        for (int j = 0; j < 16; j++) {
            h_W_down[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }
    for (int i = 0; i < D_MODEL; i++) {
        h_gamma[i] = 1.0f + ((float)(rand() % 100) - 50.0f) * 0.002f;
    }

    float *h_x = (float*)malloc(D_MODEL * sizeof(float));
    float *h_z_cpu = (float*)malloc(D_MODEL * sizeof(float));
    float *h_g_cpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_u_cpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_h_cpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_ydown_cpu = (float*)malloc(D_MODEL * sizeof(float));
    float *h_y_cpu = (float*)malloc(D_MODEL * sizeof(float));

    float *h_y_mod   = (float*)malloc(D_MODEL * sizeof(float));
    float *h_y_fused = (float*)malloc(D_MODEL * sizeof(float));

    for (int i = 0; i < D_MODEL; i++) {
        h_x[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }

    // Resident Buffers in VRAM
    cl_mem d_W_gate = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_gate_bytes, h_W_gate, &err);
    CHECK_CL(err, "d_W_gate");
    cl_mem d_W_up   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_up_bytes, h_W_up, &err);
    CHECK_CL(err, "d_W_up");
    cl_mem d_W_down = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_down_bytes, h_W_down, &err);
    CHECK_CL(err, "d_W_down");
    cl_mem d_gamma  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gamma_bytes, h_gamma, &err);
    CHECK_CL(err, "d_gamma");

    cl_mem d_x     = clCreateBuffer(ctx, CL_MEM_READ_ONLY, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_z     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_g     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_FFN * sizeof(float), NULL, &err);
    cl_mem d_u     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_FFN * sizeof(float), NULL, &err);
    cl_mem d_h     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_FFN * sizeof(float), NULL, &err);
    cl_mem d_ydown = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_y_mod = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_y_fus = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, D_MODEL * sizeof(float), NULL, &err);

    // Set Arguments ONCE for Modular Kernels
    float eps = EPSILON;
    int d_val = D_MODEL;
    int m_val = D_FFN;

    // k_rmsnorm: (x, gamma, z, D, eps)
    clSetKernelArg(k_rmsnorm, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_rmsnorm, 1, sizeof(cl_mem), &d_gamma);
    clSetKernelArg(k_rmsnorm, 2, sizeof(cl_mem), &d_z);
    clSetKernelArg(k_rmsnorm, 3, sizeof(int), &d_val);
    clSetKernelArg(k_rmsnorm, 4, sizeof(float), &eps);

    // k_gate: (W_gate, z, g, M=8960, K=1536)
    clSetKernelArg(k_gate, 0, sizeof(cl_mem), &d_W_gate);
    clSetKernelArg(k_gate, 1, sizeof(cl_mem), &d_z);
    clSetKernelArg(k_gate, 2, sizeof(cl_mem), &d_g);
    clSetKernelArg(k_gate, 3, sizeof(int), &m_val);
    clSetKernelArg(k_gate, 4, sizeof(int), &d_val);

    // k_up: (W_up, z, u, M=8960, K=1536)
    clSetKernelArg(k_up, 0, sizeof(cl_mem), &d_W_up);
    clSetKernelArg(k_up, 1, sizeof(cl_mem), &d_z);
    clSetKernelArg(k_up, 2, sizeof(cl_mem), &d_u);
    clSetKernelArg(k_up, 3, sizeof(int), &m_val);
    clSetKernelArg(k_up, 4, sizeof(int), &d_val);

    // k_swiglu: (g, u, h, M=8960)
    clSetKernelArg(k_swiglu, 0, sizeof(cl_mem), &d_g);
    clSetKernelArg(k_swiglu, 1, sizeof(cl_mem), &d_u);
    clSetKernelArg(k_swiglu, 2, sizeof(cl_mem), &d_h);
    clSetKernelArg(k_swiglu, 3, sizeof(int), &m_val);

    // k_down: (W_down, h, ydown, M=1536, K=8960)
    clSetKernelArg(k_down, 0, sizeof(cl_mem), &d_W_down);
    clSetKernelArg(k_down, 1, sizeof(cl_mem), &d_h);
    clSetKernelArg(k_down, 2, sizeof(cl_mem), &d_ydown);
    clSetKernelArg(k_down, 3, sizeof(int), &d_val);
    clSetKernelArg(k_down, 4, sizeof(int), &m_val);

    // k_residual: (x, ydown, y, D=1536)
    clSetKernelArg(k_residual, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_residual, 1, sizeof(cl_mem), &d_ydown);
    clSetKernelArg(k_residual, 2, sizeof(cl_mem), &d_y_mod);
    clSetKernelArg(k_residual, 3, sizeof(int), &d_val);

    // Set Arguments ONCE for Fused Kernels
    // k_fused_gate_up: (W_gate, W_up, z, h, M=8960, K=1536)
    clSetKernelArg(k_fused_gate_up, 0, sizeof(cl_mem), &d_W_gate);
    clSetKernelArg(k_fused_gate_up, 1, sizeof(cl_mem), &d_W_up);
    clSetKernelArg(k_fused_gate_up, 2, sizeof(cl_mem), &d_z);
    clSetKernelArg(k_fused_gate_up, 3, sizeof(cl_mem), &d_h);
    clSetKernelArg(k_fused_gate_up, 4, sizeof(int), &m_val);
    clSetKernelArg(k_fused_gate_up, 5, sizeof(int), &d_val);

    // k_fused_down_res: (W_down, h, x, y_fus, D=1536, M=8960)
    clSetKernelArg(k_fused_down_res, 0, sizeof(cl_mem), &d_W_down);
    clSetKernelArg(k_fused_down_res, 1, sizeof(cl_mem), &d_h);
    clSetKernelArg(k_fused_down_res, 2, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_fused_down_res, 3, sizeof(cl_mem), &d_y_fus);
    clSetKernelArg(k_fused_down_res, 4, sizeof(int), &d_val);
    clSetKernelArg(k_fused_down_res, 5, sizeof(int), &m_val);

    // Work-group layouts
    size_t local_norm = 128, global_norm = 128;
    size_t local_warp = 128;
    size_t global_gate = ((D_FFN + 3) / 4) * local_warp;
    size_t global_down = ((D_MODEL + 3) / 4) * local_warp;
    size_t local_swiglu = 256;
    size_t global_swiglu = ((D_FFN + local_swiglu - 1) / local_swiglu) * local_swiglu;
    size_t local_resid = 256;
    size_t global_resid = ((D_MODEL + local_resid - 1) / local_resid) * local_resid;

    // [1] CPU Reference
    printf("⏱️ [1/4] Benchmarking CPU Reference Pipeline (AVX2 + OpenMP 8 threads)...\n");
    for (int i = 0; i < 3; i++) {
        cpu_swiglu_pipeline(h_x, h_gamma, h_W_gate, h_W_up, h_W_down, h_z_cpu, h_g_cpu, h_u_cpu, h_h_cpu, h_ydown_cpu, h_y_cpu);
    }
    const int ITERS = 100;
    double t_cpu_start = get_time_us();
    for (int i = 0; i < ITERS; i++) {
        cpu_swiglu_pipeline(h_x, h_gamma, h_W_gate, h_W_up, h_W_down, h_z_cpu, h_g_cpu, h_u_cpu, h_h_cpu, h_ydown_cpu, h_y_cpu);
    }
    double t_cpu_end = get_time_us();
    double cpu_avg_ms = (t_cpu_end - t_cpu_start) / (ITERS * 1000.0);
    printf("  ✓ CPU Total SwiGLU Time: %8.2f ms\n\n", cpu_avg_ms);

    // [2] Warmup GPU
    printf("⏱️ [2/4] Warming up GPU Pipelines (Modular & Fused)...\n");
    for (int i = 0; i < 5; i++) {
        clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, NULL);
        // Modular
        clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_gate, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_up, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_swiglu, 1, NULL, &global_swiglu, &local_swiglu, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_down, 1, NULL, &global_down, &local_warp, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_residual, 1, NULL, &global_resid, &local_resid, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_y_mod, CL_TRUE, 0, D_MODEL * sizeof(float), h_y_mod, 0, NULL, NULL);

        // Fused
        clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_fused_gate_up, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_fused_down_res, 1, NULL, &global_down, &local_warp, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_y_fus, CL_TRUE, 0, D_MODEL * sizeof(float), h_y_fused, 0, NULL, NULL);
    }
    clFinish(queue);

    // [3 & 4] Alternated Interleaved Benchmarking (eliminates thermal throttling & cache warmup bias)
    printf("⏱️ [3/4] Measuring Modular (6 stages) & Fused (3 stages) Alternated (%d runs)...\n", ITERS);
    double mod_total_us = 0.0;
    double fus_total_us = 0.0;

    for (int i = 0; i < ITERS; i++) {
        // Run Modular iteration
        {
            double t0 = get_time_us();
            cl_event ev_done;
            clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_gate, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_up, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_swiglu, 1, NULL, &global_swiglu, &local_swiglu, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_down, 1, NULL, &global_down, &local_warp, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_residual, 1, NULL, &global_resid, &local_resid, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_y_mod, CL_FALSE, 0, D_MODEL * sizeof(float), h_y_mod, 0, NULL, &ev_done);
            clWaitForEvents(1, &ev_done);
            clReleaseEvent(ev_done);
            double t1 = get_time_us();
            mod_total_us += (t1 - t0);
        }

        // Run Fused iteration
        {
            double t0 = get_time_us();
            cl_event ev_done;
            clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_fused_gate_up, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_fused_down_res, 1, NULL, &global_down, &local_warp, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_y_fus, CL_FALSE, 0, D_MODEL * sizeof(float), h_y_fused, 0, NULL, &ev_done);
            clWaitForEvents(1, &ev_done);
            clReleaseEvent(ev_done);
            double t1 = get_time_us();
            fus_total_us += (t1 - t0);
        }
    }
    double mod_avg_ms = mod_total_us / (ITERS * 1000.0);
    double fus_avg_ms = fus_total_us / (ITERS * 1000.0);
    printf("  ✓ Modular Pipeline Wall-Clock E2E: %8.2f ms\n", mod_avg_ms);
    printf("  ✓ Fused Pipeline Wall-Clock E2E:   %8.2f ms\n\n", fus_avg_ms);

    // Validation
    const double atol = 1e-4;
    const double rtol = 1e-3;
    double max_err_mod = 0.0, max_err_fus = 0.0;

    for (int i = 0; i < D_MODEL; i++) {
        if (!isfinite(h_y_mod[i]) || !isfinite(h_y_fused[i])) {
            fprintf(stderr, "FATAL: Non-finite output in y[%d]!\n", i);
            exit(1);
        }
        double diff_m = fabs((double)h_y_cpu[i] - (double)h_y_mod[i]);
        if (diff_m > max_err_mod) max_err_mod = diff_m;
        double tol_m = atol + rtol * fabs((double)h_y_cpu[i]);
        if (diff_m > tol_m) {
            fprintf(stderr, "FATAL: Modular y[%d] error %e exceeded tol %e!\n", i, diff_m, tol_m);
            exit(1);
        }

        double diff_f = fabs((double)h_y_cpu[i] - (double)h_y_fused[i]);
        if (diff_f > max_err_fus) max_err_fus = diff_f;
        double tol_f = atol + rtol * fabs((double)h_y_cpu[i]);
        if (diff_f > tol_f) {
            fprintf(stderr, "FATAL: Fused y[%d] error %e exceeded tol %e!\n", i, diff_f, tol_f);
            exit(1);
        }
    }

    printf("===================================================================\n");
    printf("                     FINAL BENCHMARK COMPARISON\n");
    printf("===================================================================\n");
    printf("  • CPU Reference (AVX2 FMA 8 threads): %8.2f ms\n", cpu_avg_ms);
    printf("  • GPU Modular Pipeline (6 stages):    %8.2f ms (Speedup: %.2fx) [Max Err: %.2e]\n", 
           mod_avg_ms, cpu_avg_ms / mod_avg_ms, max_err_mod);
    printf("  • GPU Fused Pipeline   (3 stages):    %8.2f ms (Speedup: %.2fx) [Max Err: %.2e]\n", 
           fus_avg_ms, cpu_avg_ms / fus_avg_ms, max_err_fus);
    printf("-------------------------------------------------------------------\n");
    printf("  🏆 Fused Pipeline Improvement over Modular: %.2fx faster (saved %.2f ms)\n", 
           mod_avg_ms / fus_avg_ms, mod_avg_ms - fus_avg_ms);
    printf("  🏆 Total Speedup over CPU:                 %.2fx 🚀\n", cpu_avg_ms / fus_avg_ms);
    printf("===================================================================\n");

    clReleaseMemObject(d_W_gate);
    clReleaseMemObject(d_W_up);
    clReleaseMemObject(d_W_down);
    clReleaseMemObject(d_gamma);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_z);
    clReleaseMemObject(d_g);
    clReleaseMemObject(d_u);
    clReleaseMemObject(d_h);
    clReleaseMemObject(d_ydown);
    clReleaseMemObject(d_y_mod);
    clReleaseMemObject(d_y_fus);

    clReleaseKernel(k_rmsnorm);
    clReleaseKernel(k_gate);
    clReleaseKernel(k_up);
    clReleaseKernel(k_swiglu);
    clReleaseKernel(k_down);
    clReleaseKernel(k_residual);
    clReleaseKernel(k_fused_gate_up);
    clReleaseKernel(k_fused_down_res);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    free(h_W_gate);
    free(h_W_up);
    free(h_W_down);
    free(h_gamma);
    free(h_x);
    free(h_z_cpu);
    free(h_g_cpu);
    free(h_u_cpu);
    free(h_h_cpu);
    free(h_ydown_cpu);
    free(h_y_cpu);
    free(h_y_mod);
    free(h_y_fused);

    return 0;
}
