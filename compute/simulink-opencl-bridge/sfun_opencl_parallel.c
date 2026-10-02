/**
 * @file sfun_opencl_parallel.c
 * @brief Simulink Level-2 C-MEX S-Function for GPU-Accelerated Simulations
 *        using Mesa Rusticl OpenCL 3.0 on NVIDIA Kepler GT 750M.
 *
 * Simulates an array of N coupled second-order dynamical systems
 * (mass-spring-damper / wave propagation / thermal array) in parallel
 * directly inside Simulink time-stepping engine.
 */

#define S_FUNCTION_NAME  sfun_opencl_parallel
#define S_FUNCTION_LEVEL 2

#define CL_TARGET_OPENCL_VERSION 300
#include "simstruc.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_PARAMS          2
#define PARAM_CHANNELS(S)   ssGetSFcnParam(S, 0) // Number of parallel channels (e.g. 1024)
#define PARAM_DAMPING(S)    ssGetSFcnParam(S, 1) // Damping coefficient gamma

typedef struct {
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem           d_u;
    cl_mem           d_pos;
    cl_mem           d_vel;
    cl_mem           d_y;
    int              num_nodes;
    float            damping;
    float            *h_in_buf;
    float            *h_out_buf;
    int              step_count;
} GpuSimContext;

// Embedded OpenCL dynamic kernel
static const char *sfun_kernel_src =
"__kernel void simulate_step(__global const float *u,         \n"
"                            __global float *pos,             \n"
"                            __global float *vel,             \n"
"                            __global float *y,               \n"
"                            const int n,                     \n"
"                            const float dt,                  \n"
"                            const float damping,             \n"
"                            const float k_stiffness) {       \n"
"    int i = get_global_id(0);                                \n"
"    if (i >= n) return;                                      \n"
"                                                             \n"
"    float p = pos[i];                                        \n"
"    float v = vel[i];                                        \n"
"    float p_left  = (i > 0) ? pos[i - 1] : 0.0f;             \n"
"    float p_right = (i < n - 1) ? pos[i + 1] : 0.0f;         \n"
"                                                             \n"
"    // Restoring spring force from neighbors + input drive   \n"
"    float force = u[i] - k_stiffness * (2.0f * p - p_left - p_right) - damping * v;\n"
"                                                             \n"
"    // Symplectic Euler integration                          \n"
"    v += force * dt;                                         \n"
"    p += v * dt;                                             \n"
"                                                             \n"
"    vel[i] = v;                                              \n"
"    pos[i] = p;                                              \n"
"    y[i]   = p; // Output position to Simulink               \n"
"}                                                            \n";

static void mdlInitializeSizes(SimStruct *S) {
    int num_channels = 1024;

    ssSetNumSFcnParams(S, NUM_PARAMS);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        return;
    }

    if (ssGetSFcnParamsCount(S) >= 1 && mxGetNumberOfElements(PARAM_CHANNELS(S)) >= 1) {
        num_channels = (int)mxGetScalar(PARAM_CHANNELS(S));
    }

    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, num_channels);
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    ssSetInputPortRequiredContiguous(S, 0, 1);

    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, num_channels);

    ssSetNumSampleTimes(S, 1);
    ssSetNumPWork(S, 1);
    ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
}

static void mdlInitializeSampleTimes(SimStruct *S) {
    // Fixed sample time: 0.005s (200 Hz) matching the model solver
    ssSetSampleTime(S, 0, 0.005);
    ssSetOffsetTime(S, 0, 0.0);
}

#define MDL_START
#if defined(MDL_START)
static void mdlStart(SimStruct *S) {
    setenv("RUSTICL_ENABLE", "nouveau", 1);

    int num_channels = 1024;
    if (ssGetSFcnParamsCount(S) >= 1 && mxGetNumberOfElements(PARAM_CHANNELS(S)) >= 1) {
        num_channels = (int)mxGetScalar(PARAM_CHANNELS(S));
    }

    float damping = 0.05f;
    if (ssGetSFcnParamsCount(S) >= 2 && mxGetNumberOfElements(PARAM_DAMPING(S)) >= 1) {
        damping = (float)mxGetScalar(PARAM_DAMPING(S));
    }

    GpuSimContext *ctx = (GpuSimContext *)calloc(1, sizeof(GpuSimContext));
    if (!ctx) {
        ssSetErrorStatus(S, "Failed to allocate GPU host context memory");
        return;
    }

    ctx->num_nodes = num_channels;
    ctx->damping   = damping;
    ctx->h_in_buf  = (float *)malloc(num_channels * sizeof(float));
    ctx->h_out_buf = (float *)malloc(num_channels * sizeof(float));
    ctx->step_count = 0;

    cl_int err;
    cl_platform_id platform = NULL;
    cl_uint num_platforms = 0;
    clGetPlatformIDs(1, &platform, &num_platforms);

    if (num_platforms == 0 || platform == NULL) {
        ssSetErrorStatus(S, "Simulink OpenCL: No OpenCL platform available!");
        return;
    }

    cl_device_id device = NULL;
    cl_uint num_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    if (num_devices == 0 || device == NULL) {
        ssSetErrorStatus(S, "Simulink OpenCL: No GPU device available on platform!");
        return;
    }

    ctx->context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    ctx->queue   = clCreateCommandQueueWithProperties(ctx->context, device, NULL, &err);

    ctx->program = clCreateProgramWithSource(ctx->context, 1, &sfun_kernel_src, NULL, &err);
    err = clBuildProgram(ctx->program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[2048];
        clGetProgramBuildInfo(ctx->program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        mexPrintf("Kernel build error: %s\n", build_log);
        ssSetErrorStatus(S, "Failed to build OpenCL kernel!");
        return;
    }

    ctx->kernel = clCreateKernel(ctx->program, "simulate_step", &err);
    if (err != CL_SUCCESS) {
        ssSetErrorStatus(S, "Failed to create OpenCL kernel object!");
        return;
    }

    size_t buf_bytes = num_channels * sizeof(float);
    ctx->d_u   = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY,  buf_bytes, NULL, &err);
    ctx->d_pos = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, buf_bytes, NULL, &err);
    ctx->d_vel = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, buf_bytes, NULL, &err);
    ctx->d_y   = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, buf_bytes, NULL, &err);

    // Explicitly zero initial conditions
    float *zero_mem = (float *)calloc(num_channels, sizeof(float));
    clEnqueueWriteBuffer(ctx->queue, ctx->d_pos, CL_TRUE, 0, buf_bytes, zero_mem, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, ctx->d_vel, CL_TRUE, 0, buf_bytes, zero_mem, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, ctx->d_y,   CL_TRUE, 0, buf_bytes, zero_mem, 0, NULL, NULL);
    free(zero_mem);

    mexPrintf("⚡ [S-Function GPU] Initialized on 384 CUDA Cores for %d nodes\n", num_channels);

    ssGetPWork(S)[0] = (void *)ctx;
}
#endif

static void mdlOutputs(SimStruct *S, int_T tid) {
    (void)tid;
    GpuSimContext *ctx = (GpuSimContext *)ssGetPWork(S)[0];
    if (!ctx) return;

    const real_T *u = (const real_T *)ssGetInputPortSignal(S, 0);
    real_T *y = ssGetOutputPortRealSignal(S, 0);
    int n = ctx->num_nodes;
    float dt_f = 0.005f; // Explicit stable step size

    // Copy inputs
    if (u) {
        for (int i = 0; i < n; i++) {
            ctx->h_in_buf[i] = (float)u[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            ctx->h_in_buf[i] = 0.0f;
        }
    }

    size_t buf_bytes = n * sizeof(float);

    // Upload inputs to GPU
    clEnqueueWriteBuffer(ctx->queue, ctx->d_u, CL_FALSE, 0, buf_bytes, ctx->h_in_buf, 0, NULL, NULL);

    float k_stiffness = 20.0f;
    clSetKernelArg(ctx->kernel, 0, sizeof(cl_mem), &ctx->d_u);
    clSetKernelArg(ctx->kernel, 1, sizeof(cl_mem), &ctx->d_pos);
    clSetKernelArg(ctx->kernel, 2, sizeof(cl_mem), &ctx->d_vel);
    clSetKernelArg(ctx->kernel, 3, sizeof(cl_mem), &ctx->d_y);
    clSetKernelArg(ctx->kernel, 4, sizeof(int),    &n);
    clSetKernelArg(ctx->kernel, 5, sizeof(float),  &dt_f);
    clSetKernelArg(ctx->kernel, 6, sizeof(float),  &ctx->damping);
    clSetKernelArg(ctx->kernel, 7, sizeof(float),  &k_stiffness);

    size_t global_work = n;
    size_t local_work  = (n >= 256) ? 256 : n;

    // Launch on GPU
    clEnqueueNDRangeKernel(ctx->queue, ctx->kernel, 1, NULL, &global_work, &local_work, 0, NULL, NULL);

    // Read back results
    clEnqueueReadBuffer(ctx->queue, ctx->d_y, CL_TRUE, 0, buf_bytes, ctx->h_out_buf, 0, NULL, NULL);

    for (int i = 0; i < n; i++) {
        y[i] = (real_T)ctx->h_out_buf[i];
    }

    ctx->step_count++;
}

static void mdlTerminate(SimStruct *S) {
    GpuSimContext *ctx = (GpuSimContext *)ssGetPWork(S)[0];
    if (ctx) {
        if (ctx->d_u)   clReleaseMemObject(ctx->d_u);
        if (ctx->d_pos) clReleaseMemObject(ctx->d_pos);
        if (ctx->d_vel) clReleaseMemObject(ctx->d_vel);
        if (ctx->d_y)   clReleaseMemObject(ctx->d_y);
        if (ctx->kernel)  clReleaseKernel(ctx->kernel);
        if (ctx->program) clReleaseProgram(ctx->program);
        if (ctx->queue)   clReleaseCommandQueue(ctx->queue);
        if (ctx->context) clReleaseContext(ctx->context);
        if (ctx->h_in_buf)  free(ctx->h_in_buf);
        if (ctx->h_out_buf) free(ctx->h_out_buf);
        free(ctx);
        ssGetPWork(S)[0] = NULL;
    }
}

#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
