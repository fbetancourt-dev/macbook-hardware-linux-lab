/**
 * @file nbody_simulation.c
 * @brief Gravitational N-Body Simulation Benchmark on NVIDIA Kepler GT 750M
 *        utilizing Mesa Rusticl OpenCL 3.0 on Nouveau driver.
 *
 * Simulates 16,384 gravitationally interacting bodies (268.4 Million pair interactions/step).
 * High arithmetic intensity benchmark showcasing the 384 Kepler CUDA Cores.
 */

#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define NUM_BODIES    16384
#define NUM_STEPS     20
#define SOFTENING_SQ  1e-3f
#define DT            0.01f

typedef struct {
    float x, y, z, m;
} Body;

typedef struct {
    float vx, vy, vz, pad;
} Velocity;

const char *nbody_kernel_source =
"typedef struct { float x, y, z, m; } Body;                     \n"
"typedef struct { float vx, vy, vz, pad; } Velocity;            \n"
"                                                               \n"
"__kernel void nbody_step(__global Body *bodies,                \n"
"                         __global Velocity *vel,               \n"
"                         const int num_bodies,                 \n"
"                         const float dt,                       \n"
"                         const float softening_sq) {           \n"
"    int i = get_global_id(0);                                  \n"
"    if (i >= num_bodies) return;                               \n"
"                                                               \n"
"    Body bi = bodies[i];                                       \n"
"    float fx = 0.0f, fy = 0.0f, fz = 0.0f;                     \n"
"                                                               \n"
"    for (int j = 0; j < num_bodies; j++) {                     \n"
"        Body bj = bodies[j];                                   \n"
"        float dx = bj.x - bi.x;                                \n"
"        float dy = bj.y - bi.y;                                \n"
"        float dz = bj.z - bi.z;                                \n"
"        float dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;\n"
"        float inv_dist = rsqrt(dist_sq);                       \n"
"        float inv_dist3 = inv_dist * inv_dist * inv_dist;      \n"
"        float s = bj.m * inv_dist3;                            \n"
"        fx += dx * s;                                          \n"
"        fy += dy * s;                                          \n"
"        fz += dz * s;                                          \n"
"    }                                                          \n"
"                                                               \n"
"    Velocity vi = vel[i];                                      \n"
"    vi.vx += fx * dt;                                          \n"
"    vi.vy += fy * dt;                                          \n"
"    vi.vz += fz * dt;                                          \n"
"    vel[i] = vi;                                               \n"
"                                                               \n"
"    bi.x += vi.vx * dt;                                        \n"
"    bi.y += vi.vy * dt;                                        \n"
"    bi.z += vi.vz * dt;                                        \n"
"    bodies[i] = bi;                                            \n"
"}                                                              \n";

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void cpu_nbody_step(Body *bodies, Velocity *vel, int num_bodies, float dt, float softening_sq) {
    for (int i = 0; i < num_bodies; i++) {
        Body bi = bodies[i];
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int j = 0; j < num_bodies; j++) {
            Body bj = bodies[j];
            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float dz = bj.z - bi.z;
            float dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
            float inv_dist = 1.0f / sqrtf(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            float s = bj.m * inv_dist3;
            fx += dx * s;
            fy += dy * s;
            fz += dz * s;
        }
        vel[i].vx += fx * dt;
        vel[i].vy += fy * dt;
        vel[i].vz += fz * dt;
        bodies[i].x += vel[i].vx * dt;
        bodies[i].y += vel[i].vy * dt;
        bodies[i].z += vel[i].vz * dt;
    }
}

int main(void) {
    printf("====================================================================\n");
    printf("🌌 Gravitational N-Body Simulation (%d Bodies x %d Steps)\n", NUM_BODIES, NUM_STEPS);
    printf("   Workload: 268.4 Million Pair-Interactions per Step (High Arithmetic Intensity)\n");
    printf("   Hardware: NVIDIA GK107 (GeForce GT 750M - 384 Kepler CUDA Cores)\n");
    printf("   Driver:   Mesa Rusticl (OpenCL 3.0 on Nouveau nvc0)\n");
    printf("====================================================================\n\n");

    size_t bodies_size = NUM_BODIES * sizeof(Body);
    size_t vel_size = NUM_BODIES * sizeof(Velocity);

    Body *h_bodies_init = (Body *)malloc(bodies_size);
    Velocity *h_vel_init = (Velocity *)malloc(vel_size);
    Body *h_bodies_cpu = (Body *)malloc(bodies_size);
    Velocity *h_vel_cpu = (Velocity *)malloc(vel_size);
    Body *h_bodies_gpu = (Body *)malloc(bodies_size);

    srand(42);
    for (int i = 0; i < NUM_BODIES; i++) {
        h_bodies_init[i].x = ((float)rand() / RAND_MAX - 0.5f) * 100.0f;
        h_bodies_init[i].y = ((float)rand() / RAND_MAX - 0.5f) * 100.0f;
        h_bodies_init[i].z = ((float)rand() / RAND_MAX - 0.5f) * 100.0f;
        h_bodies_init[i].m = ((float)rand() / RAND_MAX) * 10.0f + 1.0f;
        h_vel_init[i].vx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_vel_init[i].vy = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_vel_init[i].vz = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_vel_init[i].pad = 0.0f;

        h_bodies_cpu[i] = h_bodies_init[i];
        h_vel_cpu[i] = h_vel_init[i];
    }

    // OpenCL Setup
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    char dev_name[128];
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    printf("✓ Target GPU Activated: %s (%u SMX Units / 384 CUDA Cores)\n\n", dev_name, compute_units);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &nbody_kernel_source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "nbody_step", &err);

    cl_mem d_bodies = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodies_size, h_bodies_init, &err);
    cl_mem d_vel    = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vel_size, h_vel_init, &err);

    int num_bodies = NUM_BODIES;
    float dt = DT;
    float softening = SOFTENING_SQ;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_bodies);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_vel);
    clSetKernelArg(kernel, 2, sizeof(int), &num_bodies);
    clSetKernelArg(kernel, 3, sizeof(float), &dt);
    clSetKernelArg(kernel, 4, sizeof(float), &softening);

    size_t global_work = NUM_BODIES;
    size_t local_work = 256;

    // -------------------------------------------------------------
    // GPU Benchmark
    // -------------------------------------------------------------
    printf("⚡ Running GPU N-Body Simulation on 384 CUDA Cores (%d steps)... ", NUM_STEPS);
    fflush(stdout);

    double t0_gpu = get_time_sec();
    for (int step = 0; step < NUM_STEPS; step++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work, &local_work, 0, NULL, NULL);
    }
    clFinish(queue);
    double t1_gpu = get_time_sec();
    double gpu_duration = t1_gpu - t0_gpu;
    printf("DONE in %.3f s (%.2f ms/step)\n", gpu_duration, (gpu_duration * 1000.0) / NUM_STEPS);

    clEnqueueReadBuffer(queue, d_bodies, CL_TRUE, 0, bodies_size, h_bodies_gpu, 0, NULL, NULL);

    // -------------------------------------------------------------
    // CPU Benchmark (Baseline: 2 steps scaled for time)
    // -------------------------------------------------------------
    printf("💻 Running CPU Baseline (Testing 2 steps to extrapolate)... ");
    fflush(stdout);

    double t0_cpu = get_time_sec();
    for (int step = 0; step < 2; step++) {
        cpu_nbody_step(h_bodies_cpu, h_vel_cpu, NUM_BODIES, DT, SOFTENING_SQ);
    }
    double t1_cpu = get_time_sec();
    double cpu_2step = t1_cpu - t0_cpu;
    double cpu_duration = (cpu_2step / 2.0) * NUM_STEPS;
    printf("DONE (Extrapolated %d steps: %.2f s | %.2f ms/step)\n\n", NUM_STEPS, cpu_duration, (cpu_duration * 1000.0) / NUM_STEPS);

    // Calculate Interactions and GFLOPS
    // Each pair interaction does:
    // 3 subs, 3 muls, 3 adds (dx*dx+dy*dy+dz*dz+soft), 1 rsqrt, 2 muls, 1 mul(bj.m), 3 muls, 3 adds = ~20 FLOPs
    double total_interactions = (double)NUM_BODIES * NUM_BODIES * NUM_STEPS;
    double total_flops = total_interactions * 20.0;
    double gflops = (total_flops / gpu_duration) / 1e9;

    printf("====================================================================\n");
    printf("📊 PERFORMANCE SHOWDOWN (Arithmetic-Heavy Compute)\n");
    printf("====================================================================\n");
    printf("  CPU Execution Time:   %.2f s (%.1f ms/step)\n", cpu_duration, (cpu_duration * 1000.0) / NUM_STEPS);
    printf("  GPU Execution Time:   %.2f s (%.1f ms/step)\n", gpu_duration, (gpu_duration * 1000.0) / NUM_STEPS);
    printf("  Parallel Speedup:     %.2fx FASTER on NVIDIA GT 750M!\n", cpu_duration / gpu_duration);
    printf("  Effective Compute:    %.2f GFLOPS sustained\n", gflops);
    printf("  Total Interactions:   %.2f Billion pairs computed\n", total_interactions / 1e9);
    printf("====================================================================\n");

    clReleaseMemObject(d_bodies);
    clReleaseMemObject(d_vel);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_bodies_init);
    free(h_vel_init);
    free(h_bodies_cpu);
    free(h_vel_cpu);
    free(h_bodies_gpu);

    return 0;
}
