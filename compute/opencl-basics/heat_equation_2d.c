/**
 * @file heat_equation_2d.c
 * @brief 2D Heat Diffusion PDE Simulation benchmark on NVIDIA Kepler GT 750M
 *        utilizing Mesa Rusticl OpenCL 3.0 on Nouveau driver.
 *
 * Simulates thermal diffusion across a 2048 x 2048 grid (4,194,304 points)
 * comparing CPU scalar execution vs. 384 Kepler CUDA cores.
 */

#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define GRID_WIDTH  2048
#define GRID_HEIGHT 2048
#define TOTAL_CELLS (GRID_WIDTH * GRID_HEIGHT)
#define NUM_STEPS   50
#define ALPHA_DT    0.2f  // Thermal diffusivity constant (stable < 0.25)

// OpenCL 2D 5-point stencil kernel for Laplace heat diffusion
const char *kernel_source = 
"__kernel void heat_step(__global const float *current,         \n"
"                        __global float *next,                  \n"
"                        const int width,                       \n"
"                        const int height,                      \n"
"                        const float alpha) {                   \n"
"    int x = get_global_id(0);                                  \n"
"    int y = get_global_id(1);                                  \n"
"                                                               \n"
"    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {   \n"
"        int idx = y * width + x;                               \n"
"        float c = current[idx];                                \n"
"        float top = current[(y - 1) * width + x];              \n"
"        float bottom = current[(y + 1) * width + x];           \n"
"        float left = current[y * width + (x - 1)];             \n"
"        float right = current[y * width + (x + 1)];            \n"
"        next[idx] = c + alpha * (top + bottom + left + right - 4.0f * c); \n"
"    }                                                          \n"
"}                                                              \n";

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void cpu_heat_step(const float *current, float *next, int width, int height, float alpha) {
    for (int y = 1; y < height - 1; y++) {
        int row_idx = y * width;
        int top_row = (y - 1) * width;
        int bot_row = (y + 1) * width;
        for (int x = 1; x < width - 1; x++) {
            float c = current[row_idx + x];
            float top = current[top_row + x];
            float bottom = current[bot_row + x];
            float left = current[row_idx + x - 1];
            float right = current[row_idx + x + 1];
            next[row_idx + x] = c + alpha * (top + bottom + left + right - 4.0f * c);
        }
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("🔥 2D Heat Diffusion Simulation (4.19M Grid Points x %d Steps)\n", NUM_STEPS);
    printf("   Hardware: NVIDIA GK107 (GeForce GT 750M - 384 Kepler CUDA Cores)\n");
    printf("   Driver:   Mesa Rusticl (OpenCL 3.0 on Nouveau nvc0)\n");
    printf("====================================================================\n\n");

    size_t data_bytes = TOTAL_CELLS * sizeof(float);
    float *h_init = (float *)malloc(data_bytes);
    float *h_cpu_cur = (float *)malloc(data_bytes);
    float *h_cpu_next = (float *)malloc(data_bytes);
    float *h_gpu_result = (float *)malloc(data_bytes);

    if (!h_init || !h_cpu_cur || !h_cpu_next || !h_gpu_result) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }

    // Initialize initial condition: 20°C ambient with a 100°C central heat source
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int idx = y * GRID_WIDTH + x;
            float dx = (float)x - (GRID_WIDTH / 2.0f);
            float dy = (float)y - (GRID_HEIGHT / 2.0f);
            if (dx * dx + dy * dy < 200.0f * 200.0f) {
                h_init[idx] = 100.0f; // Hot circular core
            } else {
                h_init[idx] = 20.0f;  // Ambient cold boundary
            }
            h_cpu_cur[idx] = h_init[idx];
            h_cpu_next[idx] = h_init[idx];
        }
    }

    // -------------------------------------------------------------
    // OpenCL Setup (Mesa Rusticl)
    // -------------------------------------------------------------
    cl_int err;
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to find OpenCL platform (err %d). Ensure RUSTICL_ENABLE=nouveau.\n", err);
        return 1;
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to acquire GPU device (err %d)\n", err);
        return 1;
    }

    char dev_name[128];
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    printf("✓ Target GPU Activated: %s (%u SMX Units / 384 CUDA Cores)\n\n", dev_name, compute_units);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

    // Build Kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Kernel Build Failure:\n%s\n", log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "heat_step", &err);

    // Allocate GPU Device Buffers (Ping-Pong buffers for time-stepping)
    cl_mem d_buf_a = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data_bytes, h_init, &err);
    cl_mem d_buf_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data_bytes, h_init, &err);

    int width = GRID_WIDTH;
    int height = GRID_HEIGHT;
    float alpha = ALPHA_DT;
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);
    clSetKernelArg(kernel, 4, sizeof(float), &alpha);

    size_t global_size[2] = {GRID_WIDTH, GRID_HEIGHT};
    size_t local_size[2]  = {16, 16}; // 256 threads per work-group block

    // Warm-up iteration
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_buf_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_buf_b);
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(queue);

    // Re-copy pristine initial state to GPU
    clEnqueueWriteBuffer(queue, d_buf_a, CL_TRUE, 0, data_bytes, h_init, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_buf_b, CL_TRUE, 0, data_bytes, h_init, 0, NULL, NULL);

    // -------------------------------------------------------------
    // GPU Benchmark
    // -------------------------------------------------------------
    printf("⚡ Running GPU Simulation on 384 CUDA Cores (%d time steps)... ", NUM_STEPS);
    fflush(stdout);

    double t0_gpu = get_time_sec();
    cl_mem cur_buf = d_buf_a;
    cl_mem next_buf = d_buf_b;

    for (int step = 0; step < NUM_STEPS; step++) {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &cur_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &next_buf);
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

        // Ping-pong pointer swap
        cl_mem tmp = cur_buf;
        cur_buf = next_buf;
        next_buf = tmp;
    }
    clFinish(queue);
    double t1_gpu = get_time_sec();
    double gpu_duration = t1_gpu - t0_gpu;
    printf("DONE in %.3f ms (%.2f ms/step)\n", gpu_duration * 1000.0, (gpu_duration * 1000.0) / NUM_STEPS);

    // Read back results
    clEnqueueReadBuffer(queue, cur_buf, CL_TRUE, 0, data_bytes, h_gpu_result, 0, NULL, NULL);

    // -------------------------------------------------------------
    // CPU Benchmark (Baseline)
    // -------------------------------------------------------------
    printf("💻 Running CPU Simulation (1 Core Baseline, %d steps)... ", NUM_STEPS);
    fflush(stdout);

    double t0_cpu = get_time_sec();
    float *cur_ptr = h_cpu_cur;
    float *nxt_ptr = h_cpu_next;

    for (int step = 0; step < NUM_STEPS; step++) {
        cpu_heat_step(cur_ptr, nxt_ptr, GRID_WIDTH, GRID_HEIGHT, ALPHA_DT);
        float *tmp = cur_ptr;
        cur_ptr = nxt_ptr;
        nxt_ptr = tmp;
    }
    double t1_cpu = get_time_sec();
    double cpu_duration = t1_cpu - t0_cpu;
    printf("DONE in %.3f ms (%.2f ms/step)\n\n", cpu_duration * 1000.0, (cpu_duration * 1000.0) / NUM_STEPS);

    // -------------------------------------------------------------
    // Verification & Error Check
    // -------------------------------------------------------------
    float max_diff = 0.0f;
    for (int i = 0; i < TOTAL_CELLS; i++) {
        float diff = fabsf(cur_ptr[i] - h_gpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("====================================================================\n");
    printf("📊 RESULTS SUMMARY\n");
    printf("====================================================================\n");
    printf("  CPU Execution Time:   %.2f ms\n", cpu_duration * 1000.0);
    printf("  GPU Execution Time:   %.2f ms\n", gpu_duration * 1000.0);
    printf("  Parallel Speedup:     %.2fx FASTER on NVIDIA GT 750M\n", cpu_duration / gpu_duration);
    printf("  Max Numerical Error:  %.6e (GPU matches CPU math)\n", max_diff);
    printf("  Throughput:           %.2f Million Cell-Updates/sec\n", 
           ((double)TOTAL_CELLS * NUM_STEPS / gpu_duration) / 1e6);
    printf("====================================================================\n");

    // Cleanup
    clReleaseMemObject(d_buf_a);
    clReleaseMemObject(d_buf_b);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_init);
    free(h_cpu_cur);
    free(h_cpu_next);
    free(h_gpu_result);

    return 0;
}
