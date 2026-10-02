#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <CL/cl.h>

#define QK4_0 32

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

// CPU AVX2 Reference GEMV Q4_0
void gemv_q4_0_cpu_avx2(
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
            const block_q4_0 *blk = &row_blocks[b];
            float d = fp16_to_float(blk->d);
            const float *x_blk = x + b * 32;

            float blk_sum = 0.0f;
            for (int j = 0; j < 16; j++) {
                uint8_t q = blk->qs[j];
                int x0 = (q & 0x0F) - 8;
                int x1 = (q >>   4) - 8;

                blk_sum += (float)x0 * x_blk[j] + (float)x1 * x_blk[j + 16];
            }
            row_sum += d * blk_sum;
        }
        y[r] = row_sum;
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

typedef struct {
    double cpu_time_us;
    double gpu_row_time_us;
    double gpu_warp_time_us;
    double gb_per_sec;
    int M;
    int K;
} BenchResult;

BenchResult run_benchmark(
    cl_context ctx,
    cl_command_queue queue,
    cl_kernel k_row,
    cl_kernel k_warp,
    int M,
    int K,
    const char *label
) {
    printf("\n=======================================================\n");
    printf("📊 Benchmark: %s (M=%d, K=%d)\n", label, M, K);
    printf("=======================================================\n");

    int nb = K / QK4_0;
    size_t w_bytes = (size_t)M * nb * sizeof(block_q4_0);
    size_t x_bytes = (size_t)K * sizeof(float);
    size_t y_bytes = (size_t)M * sizeof(float);

    printf("  • Matrix Weights Size (Q4_0): %.2f MB\n", (double)w_bytes / (1024.0 * 1024.0));
    printf("  • Input Vector Size (FP32):   %.2f KB\n", (double)x_bytes / 1024.0);
    printf("  • Total Elements:             %d weights\n", M * K);

    block_q4_0 *h_W = (block_q4_0*)malloc(w_bytes);
    float *h_x = (float*)malloc(x_bytes);
    float *h_y_cpu = (float*)malloc(y_bytes);
    float *h_y_row = (float*)malloc(y_bytes);
    float *h_y_warp = (float*)malloc(y_bytes);

    for (int i = 0; i < K; i++) {
        h_x[i] = ((float)(rand() % 100) - 50.0f) / 50.0f;
    }
    size_t total_blocks = (size_t)M * nb;
    for (size_t i = 0; i < total_blocks; i++) {
        h_W[i].d = float_to_fp16(0.015f + (float)(rand() % 100) * 0.0001f);
        for (int j = 0; j < 16; j++) {
            h_W[i].qs[j] = (uint8_t)(rand() % 256);
        }
    }

    // 1. CPU Benchmark
    printf("\n[1/4] Benchmarking CPU (AVX2 Haswell i7-4870HQ)...\n");
    for (int i = 0; i < 5; i++) gemv_q4_0_cpu_avx2(h_W, h_x, h_y_cpu, M, K);
    const int ITERS = 100;
    double t0 = get_time_us();
    for (int i = 0; i < ITERS; i++) gemv_q4_0_cpu_avx2(h_W, h_x, h_y_cpu, M, K);
    double t1 = get_time_us();
    double cpu_avg_us = (t1 - t0) / ITERS;
    printf("  ✓ CPU Avg Time:  %8.2f µs (%.3f ms)\n", cpu_avg_us, cpu_avg_us / 1000.0);

    // 2. OpenCL Buffers
    cl_int err;
    cl_mem d_W = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_bytes, h_W, &err);
    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_bytes, h_x, &err);
    cl_mem d_y_row = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, y_bytes, NULL, &err);
    cl_mem d_y_warp = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, y_bytes, NULL, &err);

    // 3. Kernel 1: Naive 1-thread-per-row
    printf("\n[2/4] Benchmarking GPU Kernel 1: Naive (1-thread/row)...\n");
    clSetKernelArg(k_row, 0, sizeof(cl_mem), &d_W);
    clSetKernelArg(k_row, 1, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_row, 2, sizeof(cl_mem), &d_y_row);
    clSetKernelArg(k_row, 3, sizeof(int), &K);

    size_t local_row = 64;
    size_t global_row = ((M + local_row - 1) / local_row) * local_row;

    for (int i = 0; i < 5; i++) {
        clEnqueueNDRangeKernel(queue, k_row, 1, NULL, &global_row, &local_row, 0, NULL, NULL);
    }
    clFinish(queue);

    double gpu_row_ns = 0.0;
    for (int i = 0; i < ITERS; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_row, 1, NULL, &global_row, &local_row, 0, NULL, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong ts, te;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, NULL);
        gpu_row_ns += (double)(te - ts);
        clReleaseEvent(ev);
    }
    clFinish(queue);
    double gpu_row_avg_us = (gpu_row_ns / 1e3) / ITERS;
    clEnqueueReadBuffer(queue, d_y_row, CL_TRUE, 0, y_bytes, h_y_row, 0, NULL, NULL);

    // Check error Kernel 1
    double diff_row = 0.0;
    for (int i = 0; i < M; i++) {
        double d = fabs((double)h_y_cpu[i] - (double)h_y_row[i]);
        if (d > diff_row) diff_row = d;
    }
    printf("  ✓ GPU Naive Kernel Time: %8.2f µs (%.3f ms) [Max Err: %.2e]\n", 
           gpu_row_avg_us, gpu_row_avg_us / 1000.0, diff_row);

    // 4. Kernel 2: Warp-Cooperative (32 threads per row)
    printf("\n[3/4] Benchmarking GPU Kernel 2: Warp-Cooperative (Coalesced 32 th/row)...\n");
    clSetKernelArg(k_warp, 0, sizeof(cl_mem), &d_W);
    clSetKernelArg(k_warp, 1, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_warp, 2, sizeof(cl_mem), &d_y_warp);
    clSetKernelArg(k_warp, 3, sizeof(int), &M);
    clSetKernelArg(k_warp, 4, sizeof(int), &K);

    size_t local_warp = 128; // 4 warps per work-group
    size_t num_wgs = (M + 3) / 4;
    size_t global_warp = num_wgs * local_warp;

    for (int i = 0; i < 5; i++) {
        clEnqueueNDRangeKernel(queue, k_warp, 1, NULL, &global_warp, &local_warp, 0, NULL, NULL);
    }
    clFinish(queue);

    double gpu_warp_ns = 0.0;
    for (int i = 0; i < ITERS; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_warp, 1, NULL, &global_warp, &local_warp, 0, NULL, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong ts, te;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, NULL);
        gpu_warp_ns += (double)(te - ts);
        clReleaseEvent(ev);
    }
    clFinish(queue);
    double gpu_warp_avg_us = (gpu_warp_ns / 1e3) / ITERS;
    clEnqueueReadBuffer(queue, d_y_warp, CL_TRUE, 0, y_bytes, h_y_warp, 0, NULL, NULL);

    // Check error Kernel 2
    double diff_warp = 0.0;
    for (int i = 0; i < M; i++) {
        double d = fabs((double)h_y_cpu[i] - (double)h_y_warp[i]);
        if (d > diff_warp) diff_warp = d;
    }
    printf("  ✓ GPU Warp Kernel Time:  %8.2f µs (%.3f ms) [Max Err: %.2e]\n", 
           gpu_warp_avg_us, gpu_warp_avg_us / 1000.0, diff_warp);

    // Summary
    double total_bytes = (double)w_bytes + (double)x_bytes + (double)y_bytes;
    double best_gpu_us = (gpu_warp_avg_us < gpu_row_avg_us) ? gpu_warp_avg_us : gpu_row_avg_us;
    double gb_sec = (total_bytes / (best_gpu_us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    printf("\n[4/4] Layer Summary:\n");
    printf("  • CPU Time:       %8.2f µs\n", cpu_avg_us);
    printf("  • GPU Best Time:  %8.2f µs\n", best_gpu_us);
    printf("  • Speedup (Best): %8.2fx %s\n", 
           cpu_avg_us / best_gpu_us,
           (cpu_avg_us > best_gpu_us) ? "🚀 (GPU Faster!)" : "🐢 (CPU Faster)");

    clReleaseMemObject(d_W);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y_row);
    clReleaseMemObject(d_y_warp);
    free(h_W);
    free(h_x);
    free(h_y_cpu);
    free(h_y_row);
    free(h_y_warp);

    BenchResult res;
    res.cpu_time_us = cpu_avg_us;
    res.gpu_row_time_us = gpu_row_avg_us;
    res.gpu_warp_time_us = gpu_warp_avg_us;
    res.gb_per_sec = gb_sec;
    res.M = M;
    res.K = K;
    return res;
}

int main() {
    printf("===============================================================\n");
    printf("  Kepler GT 750M vs Haswell AVX2: GEMV Q4_0 Microbenchmark\n");
    printf("===============================================================\n");

    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { printf("Failed to get platform: %d\n", err); return 1; }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { printf("Failed to get GPU device: %d\n", err); return 1; }

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("Device: %s\n", dev_name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    char *src = load_kernel_source("kernel_gemv_q4.cl");
    if (!src) { return 1; }

    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
    free(src);

    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Kernel build failed: %d\n", err);
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Log:\n%s\n", log);
        return 1;
    }
    printf("✓ OpenCL Program built successfully!\n");

    cl_kernel k_row = clCreateKernel(prog, "gemv_q4_0_row", &err);
    cl_kernel k_warp = clCreateKernel(prog, "gemv_q4_0_warp", &err);

    BenchResult r1 = run_benchmark(ctx, queue, k_row, k_warp, 1536, 1536, "Attention Projection Layer (Qwen/DeepSeek)");
    BenchResult r2 = run_benchmark(ctx, queue, k_row, k_warp, 8960, 1536, "Feed-Forward Up-Projection (FFN/SwiGLU)");

    printf("\n========================================================================================\n");
    printf("                                  FINAL VERDICT TABLE\n");
    printf("========================================================================================\n");
    printf("| Test Layer | Dimension | CPU AVX2 | GPU (Naive) | GPU (Warp Coalesced) | Best Speedup |\n");
    printf("| :--- | :---: | :---: | :---: | :---: | :---: |\n");
    printf("| Attention | 1536x1536 | %6.2f µs | %6.2f µs | %6.2f µs | %5.2fx |\n",
           r1.cpu_time_us, r1.gpu_row_time_us, r1.gpu_warp_time_us,
           r1.cpu_time_us / ((r1.gpu_warp_time_us < r1.gpu_row_time_us) ? r1.gpu_warp_time_us : r1.gpu_row_time_us));
    printf("| FFN Layer | 8960x1536 | %6.2f µs | %6.2f µs | %6.2f µs | %5.2fx |\n",
           r2.cpu_time_us, r2.gpu_row_time_us, r2.gpu_warp_time_us,
           r2.cpu_time_us / ((r2.gpu_warp_time_us < r2.gpu_row_time_us) ? r2.gpu_warp_time_us : r2.gpu_row_time_us));
    printf("========================================================================================\n");

    clReleaseKernel(k_row);
    clReleaseKernel(k_warp);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
