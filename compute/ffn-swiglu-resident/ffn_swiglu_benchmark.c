#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
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

// CPU Reference GEMV Q4_0 with AVX2 Intrinsics and OpenMP
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

// Full CPU Reference Pipeline
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
    // 1. RMSNorm
    float sum_sq = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf((sum_sq / (float)D_MODEL) + EPSILON);
    for (int i = 0; i < D_MODEL; i++) {
        z[i] = x[i] * scale * gamma[i];
    }

    // 2. Gate GEMV
    cpu_gemv_q4_0(W_gate, z, g, D_FFN, D_MODEL);

    // 3. Up GEMV
    cpu_gemv_q4_0(W_up, z, u, D_FFN, D_MODEL);

    // 4. SwiGLU Activation: h = SiLU(g) * u
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D_FFN; i++) {
        h[i] = stable_silu(g[i]) * u[i];
    }

    // 5. Down GEMV
    cpu_gemv_q4_0(W_down, h, ydown, D_MODEL, D_FFN);

    // 6. Residual Add: y = x + ydown
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
    printf("  Resident SwiGLU FFN Engine on Kepler GT 750M vs Haswell AVX2\n");
    printf("  Architecture: Qwen2.5-Coder-1.5B (D=%d, M=%d)\n", D_MODEL, D_FFN);
    printf("===================================================================\n");

    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { printf("Failed platform: %d\n", err); return 1; }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { printf("Failed GPU device: %d\n", err); return 1; }

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("Device: %s (Mesa Rusticl OpenCL)\n", dev_name);
    printf("CPU OpenMP Threads: %d threads\n", omp_get_max_threads());

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    char *src = load_kernel_source("kernel_ffn_swiglu.cl");
    if (!src) { return 1; }

    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
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

    cl_kernel k_rmsnorm  = clCreateKernel(prog, "kernel_rmsnorm", &err);
    cl_kernel k_gemv     = clCreateKernel(prog, "gemv_q4_0_dual_block", &err);
    cl_kernel k_swiglu   = clCreateKernel(prog, "kernel_swiglu", &err);
    cl_kernel k_residual = clCreateKernel(prog, "kernel_residual_add", &err);

    // Memory footprints
    int nb_gate = D_MODEL / QK4_0; // 48 blocks
    int nb_down = D_FFN / QK4_0;   // 280 blocks

    size_t w_gate_bytes = (size_t)D_FFN * nb_gate * sizeof(block_q4_0);
    size_t w_up_bytes   = (size_t)D_FFN * nb_gate * sizeof(block_q4_0);
    size_t w_down_bytes = (size_t)D_MODEL * nb_down * sizeof(block_q4_0);
    size_t gamma_bytes  = D_MODEL * sizeof(float);

    size_t total_weights_vram = w_gate_bytes + w_up_bytes + w_down_bytes + gamma_bytes;
    printf("📦 Resident VRAM Footprint:\n");
    printf("  • W_gate (%d x %d Q4_0): %.2f MB\n", D_FFN, D_MODEL, (double)w_gate_bytes / (1024.0 * 1024.0));
    printf("  • W_up   (%d x %d Q4_0): %.2f MB\n", D_FFN, D_MODEL, (double)w_up_bytes / (1024.0 * 1024.0));
    printf("  • W_down (%d x %d Q4_0): %.2f MB\n", D_MODEL, D_FFN, (double)w_down_bytes / (1024.0 * 1024.0));
    printf("  • RMSNorm gamma:          %.2f KB\n", (double)gamma_bytes / 1024.0);
    printf("  • TOTAL RESIDENT WEIGHTS: %.2f MB (stays in VRAM throughout session)\n\n", 
           (double)total_weights_vram / (1024.0 * 1024.0));

    // Allocate host weights
    block_q4_0 *h_W_gate = (block_q4_0*)malloc(w_gate_bytes);
    block_q4_0 *h_W_up   = (block_q4_0*)malloc(w_up_bytes);
    block_q4_0 *h_W_down = (block_q4_0*)malloc(w_down_bytes);
    float *h_gamma = (float*)malloc(gamma_bytes);

    // Synthetic weights initialization
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

    // Host token activations
    float *h_x = (float*)malloc(D_MODEL * sizeof(float));
    float *h_z_cpu = (float*)malloc(D_MODEL * sizeof(float));
    float *h_g_cpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_u_cpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_h_cpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_ydown_cpu = (float*)malloc(D_MODEL * sizeof(float));
    float *h_y_cpu = (float*)malloc(D_MODEL * sizeof(float));

    float *h_z_gpu = (float*)malloc(D_MODEL * sizeof(float));
    float *h_g_gpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_u_gpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_h_gpu = (float*)malloc(D_FFN * sizeof(float));
    float *h_ydown_gpu = (float*)malloc(D_MODEL * sizeof(float));
    float *h_y_gpu = (float*)malloc(D_MODEL * sizeof(float));

    for (int i = 0; i < D_MODEL; i++) {
        h_x[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }

    // Create Resident GPU Buffers
    cl_mem d_W_gate = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_gate_bytes, h_W_gate, &err);
    cl_mem d_W_up   = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_up_bytes, h_W_up, &err);
    cl_mem d_W_down = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_down_bytes, h_W_down, &err);
    cl_mem d_gamma  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gamma_bytes, h_gamma, &err);

    // Working Buffers in VRAM
    cl_mem d_x     = clCreateBuffer(ctx, CL_MEM_READ_ONLY, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_z     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_g     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_FFN * sizeof(float), NULL, &err);
    cl_mem d_u     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_FFN * sizeof(float), NULL, &err);
    cl_mem d_h     = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_FFN * sizeof(float), NULL, &err);
    cl_mem d_ydown = clCreateBuffer(ctx, CL_MEM_READ_WRITE, D_MODEL * sizeof(float), NULL, &err);
    cl_mem d_y     = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, D_MODEL * sizeof(float), NULL, &err);

    // Work-group configurations
    size_t local_norm = 128;
    size_t global_norm = 128;

    size_t local_warp = 128; // 4 warps
    size_t global_gate = ((D_FFN + 3) / 4) * local_warp;
    size_t global_down = ((D_MODEL + 3) / 4) * local_warp;

    size_t local_swiglu = 256;
    size_t global_swiglu = ((D_FFN + local_swiglu - 1) / local_swiglu) * local_swiglu;

    size_t local_resid = 256;
    size_t global_resid = ((D_MODEL + local_resid - 1) / local_resid) * local_resid;

    float eps = EPSILON;
    int d_val = D_MODEL;
    int m_val = D_FFN;

    // [1] Warmup and Benchmark CPU Reference
    printf("⏱️ [1/3] Benchmarking CPU Reference Pipeline (AVX2 + OpenMP 8 threads)...\n");
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
    printf("⏱️ [2/3] Warming up GPU Pipeline...\n");
    for (int i = 0; i < 5; i++) {
        clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, NULL);
        // 1. RMSNorm
        clSetKernelArg(k_rmsnorm, 0, sizeof(cl_mem), &d_x);
        clSetKernelArg(k_rmsnorm, 1, sizeof(cl_mem), &d_gamma);
        clSetKernelArg(k_rmsnorm, 2, sizeof(cl_mem), &d_z);
        clSetKernelArg(k_rmsnorm, 3, sizeof(int), &d_val);
        clSetKernelArg(k_rmsnorm, 4, sizeof(float), &eps);
        clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
        // 2. Gate
        clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W_gate);
        clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_z);
        clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_g);
        clSetKernelArg(k_gemv, 3, sizeof(int), &m_val);
        clSetKernelArg(k_gemv, 4, sizeof(int), &d_val);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
        // 3. Up
        clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W_up);
        clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_z);
        clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_u);
        clSetKernelArg(k_gemv, 3, sizeof(int), &m_val);
        clSetKernelArg(k_gemv, 4, sizeof(int), &d_val);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_gate, &local_warp, 0, NULL, NULL);
        // 4. SwiGLU
        clSetKernelArg(k_swiglu, 0, sizeof(cl_mem), &d_g);
        clSetKernelArg(k_swiglu, 1, sizeof(cl_mem), &d_u);
        clSetKernelArg(k_swiglu, 2, sizeof(cl_mem), &d_h);
        clSetKernelArg(k_swiglu, 3, sizeof(int), &m_val);
        clEnqueueNDRangeKernel(queue, k_swiglu, 1, NULL, &global_swiglu, &local_swiglu, 0, NULL, NULL);
        // 5. Down
        clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W_down);
        clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_h);
        clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_ydown);
        clSetKernelArg(k_gemv, 3, sizeof(int), &d_val);
        clSetKernelArg(k_gemv, 4, sizeof(int), &m_val);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_down, &local_warp, 0, NULL, NULL);
        // 6. Residual
        clSetKernelArg(k_residual, 0, sizeof(cl_mem), &d_x);
        clSetKernelArg(k_residual, 1, sizeof(cl_mem), &d_ydown);
        clSetKernelArg(k_residual, 2, sizeof(cl_mem), &d_y);
        clSetKernelArg(k_residual, 3, sizeof(int), &d_val);
        clEnqueueNDRangeKernel(queue, k_residual, 1, NULL, &global_resid, &local_resid, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, D_MODEL * sizeof(float), h_y_gpu, 0, NULL, NULL);
    }
    clFinish(queue);

    // [3] Benchmark GPU Pipeline with Detailed Profiling Breakdown
    printf("⏱️ [3/3] Benchmarking 100 Iterations on GPU (GT 750M via Mesa Rusticl)...\n");

    double t_pcie_up_total = 0.0;
    double t_norm_total    = 0.0;
    double t_gate_total    = 0.0;
    double t_up_total      = 0.0;
    double t_swiglu_total  = 0.0;
    double t_down_total    = 0.0;
    double t_resid_total   = 0.0;
    double t_pcie_dn_total = 0.0;

    double t_host_total_start = get_time_us();

    for (int i = 0; i < ITERS; i++) {
        cl_event ev_up, ev_norm, ev_gate, ev_up_gemv, ev_swiglu, ev_down, ev_resid, ev_dn;

        // Stage 0: PCIe Upload x (6 KB)
        clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, D_MODEL * sizeof(float), h_x, 0, NULL, &ev_up);

        // Stage 1: RMSNorm
        clSetKernelArg(k_rmsnorm, 0, sizeof(cl_mem), &d_x);
        clSetKernelArg(k_rmsnorm, 1, sizeof(cl_mem), &d_gamma);
        clSetKernelArg(k_rmsnorm, 2, sizeof(cl_mem), &d_z);
        clSetKernelArg(k_rmsnorm, 3, sizeof(int), &d_val);
        clSetKernelArg(k_rmsnorm, 4, sizeof(float), &eps);
        clEnqueueNDRangeKernel(queue, k_rmsnorm, 1, NULL, &global_norm, &local_norm, 0, NULL, &ev_norm);

        // Stage 2: Gate GEMV (8960x1536)
        clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W_gate);
        clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_z);
        clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_g);
        clSetKernelArg(k_gemv, 3, sizeof(int), &m_val);
        clSetKernelArg(k_gemv, 4, sizeof(int), &d_val);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_gate, &local_warp, 0, NULL, &ev_gate);

        // Stage 3: Up GEMV (8960x1536)
        clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W_up);
        clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_z);
        clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_u);
        clSetKernelArg(k_gemv, 3, sizeof(int), &m_val);
        clSetKernelArg(k_gemv, 4, sizeof(int), &d_val);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_gate, &local_warp, 0, NULL, &ev_up_gemv);

        // Stage 4: SwiGLU Activation
        clSetKernelArg(k_swiglu, 0, sizeof(cl_mem), &d_g);
        clSetKernelArg(k_swiglu, 1, sizeof(cl_mem), &d_u);
        clSetKernelArg(k_swiglu, 2, sizeof(cl_mem), &d_h);
        clSetKernelArg(k_swiglu, 3, sizeof(int), &m_val);
        clEnqueueNDRangeKernel(queue, k_swiglu, 1, NULL, &global_swiglu, &local_swiglu, 0, NULL, &ev_swiglu);

        // Stage 5: Down GEMV (1536x8960)
        clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W_down);
        clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_h);
        clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_ydown);
        clSetKernelArg(k_gemv, 3, sizeof(int), &d_val);
        clSetKernelArg(k_gemv, 4, sizeof(int), &m_val);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_down, &local_warp, 0, NULL, &ev_down);

        // Stage 6: Residual Add
        clSetKernelArg(k_residual, 0, sizeof(cl_mem), &d_x);
        clSetKernelArg(k_residual, 1, sizeof(cl_mem), &d_ydown);
        clSetKernelArg(k_residual, 2, sizeof(cl_mem), &d_y);
        clSetKernelArg(k_residual, 3, sizeof(int), &d_val);
        clEnqueueNDRangeKernel(queue, k_residual, 1, NULL, &global_resid, &local_resid, 0, NULL, &ev_resid);

        // Stage 7: PCIe Download y (6 KB)
        clEnqueueReadBuffer(queue, d_y, CL_FALSE, 0, D_MODEL * sizeof(float), h_y_gpu, 0, NULL, &ev_dn);

        // Synchronize on the final read event ONLY
        clWaitForEvents(1, &ev_dn);

        t_pcie_up_total += event_duration_us(ev_up);
        t_norm_total    += event_duration_us(ev_norm);
        t_gate_total    += event_duration_us(ev_gate);
        t_up_total      += event_duration_us(ev_up_gemv);
        t_swiglu_total  += event_duration_us(ev_swiglu);
        t_down_total    += event_duration_us(ev_down);
        t_resid_total   += event_duration_us(ev_resid);
        t_pcie_dn_total += event_duration_us(ev_dn);

        clReleaseEvent(ev_up);
        clReleaseEvent(ev_norm);
        clReleaseEvent(ev_gate);
        clReleaseEvent(ev_up_gemv);
        clReleaseEvent(ev_swiglu);
        clReleaseEvent(ev_down);
        clReleaseEvent(ev_resid);
        clReleaseEvent(ev_dn);
    }
    double t_host_total_end = get_time_us();
    double host_wall_avg_ms = (t_host_total_end - t_host_total_start) / (ITERS * 1000.0);

    // Read intermediate buffers for strict stage-by-stage validation
    clEnqueueReadBuffer(queue, d_z, CL_TRUE, 0, D_MODEL * sizeof(float), h_z_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_g, CL_TRUE, 0, D_FFN * sizeof(float), h_g_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_u, CL_TRUE, 0, D_FFN * sizeof(float), h_u_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_h, CL_TRUE, 0, D_FFN * sizeof(float), h_h_gpu, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_ydown, CL_TRUE, 0, D_MODEL * sizeof(float), h_ydown_gpu, 0, NULL, NULL);

    // Strict numerical verification
    double max_err_z = 0.0, max_err_g = 0.0, max_err_u = 0.0, max_err_h = 0.0, max_err_ydown = 0.0, max_err_y = 0.0;
    for (int i = 0; i < D_MODEL; i++) {
        double diff = fabs((double)h_z_cpu[i] - (double)h_z_gpu[i]);
        if (diff > max_err_z) max_err_z = diff;
        diff = fabs((double)h_ydown_cpu[i] - (double)h_ydown_gpu[i]);
        if (diff > max_err_ydown) max_err_ydown = diff;
        diff = fabs((double)h_y_cpu[i] - (double)h_y_gpu[i]);
        if (diff > max_err_y) max_err_y = diff;
    }
    for (int i = 0; i < D_FFN; i++) {
        double diff = fabs((double)h_g_cpu[i] - (double)h_g_gpu[i]);
        if (diff > max_err_g) max_err_g = diff;
        diff = fabs((double)h_u_cpu[i] - (double)h_u_gpu[i]);
        if (diff > max_err_u) max_err_u = diff;
        diff = fabs((double)h_h_cpu[i] - (double)h_h_gpu[i]);
        if (diff > max_err_h) max_err_h = diff;
    }

    printf("===================================================================\n");
    printf("                  STAGE-BY-STAGE PROFILING BREAKDOWN\n");
    printf("===================================================================\n");
    printf("  Stage 0: PCIe Upload x (6 KB):     %8.2f µs (%.3f ms)\n", t_pcie_up_total / ITERS, (t_pcie_up_total / ITERS) / 1000.0);
    printf("  Stage 1: RMSNorm (1536):           %8.2f µs (%.3f ms) [Max Err: %.2e]\n", t_norm_total / ITERS, (t_norm_total / ITERS) / 1000.0, max_err_z);
    printf("  Stage 2: Gate GEMV (8960x1536):    %8.2f µs (%.3f ms) [Max Err: %.2e]\n", t_gate_total / ITERS, (t_gate_total / ITERS) / 1000.0, max_err_g);
    printf("  Stage 3: Up GEMV (8960x1536):      %8.2f µs (%.3f ms) [Max Err: %.2e]\n", t_up_total / ITERS, (t_up_total / ITERS) / 1000.0, max_err_u);
    printf("  Stage 4: SwiGLU Activation:        %8.2f µs (%.3f ms) [Max Err: %.2e]\n", t_swiglu_total / ITERS, (t_swiglu_total / ITERS) / 1000.0, max_err_h);
    printf("  Stage 5: Down GEMV (1536x8960):    %8.2f µs (%.3f ms) [Max Err: %.2e]\n", t_down_total / ITERS, (t_down_total / ITERS) / 1000.0, max_err_ydown);
    printf("  Stage 6: Residual Add (1536):      %8.2f µs (%.3f ms)\n", t_resid_total / ITERS, (t_resid_total / ITERS) / 1000.0);
    printf("  Stage 7: PCIe Download y (6 KB):   %8.2f µs (%.3f ms)\n", t_pcie_dn_total / ITERS, (t_pcie_dn_total / ITERS) / 1000.0);
    printf("-------------------------------------------------------------------\n");
    double gpu_device_sum_ms = (t_norm_total + t_gate_total + t_up_total + t_swiglu_total + t_down_total + t_resid_total) / (ITERS * 1000.0);
    double gpu_full_e2e_ms   = (t_pcie_up_total + t_norm_total + t_gate_total + t_up_total + t_swiglu_total + t_down_total + t_resid_total + t_pcie_dn_total) / (ITERS * 1000.0);
    printf("  • GPU Pure Compute Time:           %8.2f ms\n", gpu_device_sum_ms);
    printf("  • GPU Pipeline (incl. PCIe I/O):   %8.2f ms\n", gpu_full_e2e_ms);
    printf("  • Host Wall-Clock E2E Time:        %8.2f ms\n", host_wall_avg_ms);
    printf("  • CPU Reference (AVX2 8 threads):  %8.2f ms\n", cpu_avg_ms);
    printf("-------------------------------------------------------------------\n");
    printf("  🏆 SPEEDUP (CPU vs GPU Compute):   %8.2fx %s\n", cpu_avg_ms / gpu_device_sum_ms, (cpu_avg_ms > gpu_device_sum_ms) ? "🚀 (GPU Faster!)" : "🐢 (CPU Faster)");
    printf("  🏆 SPEEDUP (CPU vs GPU Host E2E):  %8.2fx %s\n", cpu_avg_ms / host_wall_avg_ms, (cpu_avg_ms > host_wall_avg_ms) ? "🚀 (GPU Faster!)" : "🐢 (CPU Faster)");
    printf("  • Final Max Absolute Error:        %.6e\n", max_err_y);
    printf("===================================================================\n");

    // Clean up
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
    clReleaseMemObject(d_y);

    clReleaseKernel(k_rmsnorm);
    clReleaseKernel(k_gemv);
    clReleaseKernel(k_swiglu);
    clReleaseKernel(k_residual);
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
    free(h_z_gpu);
    free(h_g_gpu);
    free(h_u_gpu);
    free(h_h_gpu);
    free(h_ydown_gpu);
    free(h_y_gpu);

    return 0;
}
