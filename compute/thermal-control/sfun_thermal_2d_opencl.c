/*
 * sfun_thermal_2d_opencl.c
 *
 * Level-2 C-MEX S-Function for MATLAB Simulink R2025b
 * 2D Distributed Thermal Convection-Diffusion Plant (256x256 = 65,536 nodes)
 * Accelerated on NVIDIA GeForce GT 750M (384 CUDA cores) via OpenCL / Rusticl.
 *
 * Inputs:  [u1, u2, u3, u4] -> Actuator heating power in % (0 - 100)
 * Outputs: [T1, T2, T3, T4] -> Measured zone temperatures in deg C
 */

#define S_FUNCTION_NAME  sfun_thermal_2d_opencl
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define GRID_W 256
#define GRID_H 256
#define TOTAL_NODES (GRID_W * GRID_H)
#define NUM_ZONES 4
#define SUBSTEPS_PER_STEP 5
#define SAMPLE_TIME 0.05

/* OpenCL Kernel Source */
static const char *thermal_kernel_source =
"__kernel void thermal_step(\n"
"    __global const float *T_in,\n"
"    __global float *T_out,\n"
"    const float alpha,\n"
"    const float h_conv,\n"
"    const float T_amb,\n"
"    const float dt,\n"
"    const float u1, const float u2, const float u3, const float u4)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int W = get_global_size(0);\n"
"    int H = get_global_size(1);\n"
"    int idx = y * W + x;\n"
"\n"
"    if (x == 0 || x == W - 1 || y == 0 || y == H - 1) {\n"
"        /* Boundary conditions: fixed to ambient / heat sink */\n"
"        T_out[idx] = T_amb;\n"
"        return;\n"
"    }\n"
"\n"
"    float Tc = T_in[idx];\n"
"    float Tl = T_in[idx - 1];\n"
"    float Tr = T_in[idx + 1];\n"
"    float Tu = T_in[idx - W];\n"
"    float Td = T_in[idx + W];\n"
"\n"
"    /* 5-point discrete Laplace operator (dx = dy = 1.0) */\n"
"    float laplacian = (Tl + Tr + Tu + Td - 4.0f * Tc);\n"
"\n"
"    /* Convective loss to ambient */\n"
"    float loss = -h_conv * (Tc - T_amb);\n"
"\n"
"    /* 4 Localized Gaussian Heat Sources (Zones centered at 64, 192) */\n"
"    float d1_sq = (x - 64.0f) * (x - 64.0f) + (y - 64.0f) * (y - 64.0f);\n"
"    float d2_sq = (x - 192.0f) * (x - 192.0f) + (y - 64.0f) * (y - 64.0f);\n"
"    float d3_sq = (x - 64.0f) * (x - 64.0f) + (y - 192.0f) * (y - 192.0f);\n"
"    float d4_sq = (x - 192.0f) * (x - 192.0f) + (y - 192.0f) * (y - 192.0f);\n"
"\n"
"    const float inv_2sig2 = 1.0f / (2.0f * 100.0f); /* sigma = 10 */\n"
"    const float Q_max = 18.0f;\n"
"\n"
"    float q1 = (u1 * 0.01f) * Q_max * exp(-d1_sq * inv_2sig2);\n"
"    float q2 = (u2 * 0.01f) * Q_max * exp(-d2_sq * inv_2sig2);\n"
"    float q3 = (u3 * 0.01f) * Q_max * exp(-d3_sq * inv_2sig2);\n"
"    float q4 = (u4 * 0.01f) * Q_max * exp(-d4_sq * inv_2sig2);\n"
"    float Q_total = q1 + q2 + q3 + q4;\n"
"\n"
"    /* Forward Euler update */\n"
"    T_out[idx] = Tc + dt * (alpha * laplacian + loss + Q_total);\n"
"}\n";

/* Persistent OpenCL Structure */
typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_T_curr;
    cl_mem d_T_next;
    float *h_T;
    int is_initialized;
} OpenCLThermalState;

static OpenCLThermalState g_gpu_state = {0};

/* Zone Center Coordinates */
static const int ZONE_X[NUM_ZONES] = {64, 192, 64, 192};
static const int ZONE_Y[NUM_ZONES] = {64, 64, 192, 192};
#define SENSOR_RADIUS 4

/* Helper: Calculate sensor readings from host grid */
static void read_sensor_temperatures(const float *grid, real_T *out_sensors) {
    for (int z = 0; z < NUM_ZONES; z++) {
        int cx = ZONE_X[z];
        int cy = ZONE_Y[z];
        float sum = 0.0f;
        int count = 0;
        for (int dy = -SENSOR_RADIUS; dy <= SENSOR_RADIUS; dy++) {
            for (int dx = -SENSOR_RADIUS; dx <= SENSOR_RADIUS; dx++) {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < GRID_W && py >= 0 && py < GRID_H) {
                    sum += grid[py * GRID_W + px];
                    count++;
                }
            }
        }
        out_sensors[z] = (real_T)(sum / (float)count);
    }
}

static void mdlInitializeSizes(SimStruct *S) {
    ssSetNumSFcnParams(S, 0);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) return;

    /* Input Port 0: 4 Actuator power inputs (u1, u2, u3, u4) */
    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, NUM_ZONES);
    ssSetInputPortDirectFeedThrough(S, 0, 0);
    ssSetInputPortRequiredContiguous(S, 0, 1);

    /* Output Port 0: 4 Measured temperatures (T1, T2, T3, T4) */
    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, NUM_ZONES);

    ssSetNumSampleTimes(S, 1);
    ssSetNumDWork(S, 0);
    ssSetSimStateCompliance(S, USE_DEFAULT_SIM_STATE);
}

static void mdlInitializeSampleTimes(SimStruct *S) {
    ssSetSampleTime(S, 0, SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);
}

#define MDL_START
static void mdlStart(SimStruct *S) {
    cl_int err;
    if (g_gpu_state.is_initialized) {
        return;
    }

    g_gpu_state.h_T = (float *)malloc(TOTAL_NODES * sizeof(float));
    for (int i = 0; i < TOTAL_NODES; i++) {
        g_gpu_state.h_T[i] = 25.0f; /* Initial temperature: 25 deg C */
    }

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(1, &g_gpu_state.platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        ssSetErrorStatus(S, "Failed to find OpenCL platform");
        return;
    }

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(g_gpu_state.platform, CL_DEVICE_TYPE_GPU, 1, &g_gpu_state.device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        ssSetErrorStatus(S, "Failed to find GPU device");
        return;
    }

    g_gpu_state.context = clCreateContext(NULL, 1, &g_gpu_state.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        ssSetErrorStatus(S, "Failed to create OpenCL context");
        return;
    }

    g_gpu_state.queue = clCreateCommandQueueWithProperties(g_gpu_state.context, g_gpu_state.device, NULL, &err);
    if (err != CL_SUCCESS) {
        ssSetErrorStatus(S, "Failed to create command queue");
        return;
    }

    g_gpu_state.program = clCreateProgramWithSource(g_gpu_state.context, 1, &thermal_kernel_source, NULL, &err);
    err = clBuildProgram(g_gpu_state.program, 1, &g_gpu_state.device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(g_gpu_state.program, g_gpu_state.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("[OpenCL Build Error]: %s\n", log);
        ssSetErrorStatus(S, "Kernel build failed");
        return;
    }

    g_gpu_state.kernel = clCreateKernel(g_gpu_state.program, "thermal_step", &err);

    size_t bytes = TOTAL_NODES * sizeof(float);
    g_gpu_state.d_T_curr = clCreateBuffer(g_gpu_state.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, g_gpu_state.h_T, &err);
    g_gpu_state.d_T_next = clCreateBuffer(g_gpu_state.context, CL_MEM_READ_WRITE, bytes, NULL, &err);

    g_gpu_state.is_initialized = 1;
    printf("⚡ [S-Function OpenCL GPU] Initialized 2D Thermal Plant (%dx%d = %d nodes) on GT 750M\n",
           GRID_W, GRID_H, TOTAL_NODES);
}

static void mdlOutputs(SimStruct *S, int_T tid) {
    real_T *y = ssGetOutputPortRealSignal(S, 0);
    if (g_gpu_state.is_initialized && g_gpu_state.h_T) {
        read_sensor_temperatures(g_gpu_state.h_T, y);
    } else {
        for (int i = 0; i < NUM_ZONES; i++) y[i] = 25.0;
    }
}

#define MDL_UPDATE
static void mdlUpdate(SimStruct *S, int_T tid) {
    if (!g_gpu_state.is_initialized) return;

    const real_T *u = (const real_T *)ssGetInputPortSignal(S, 0);
    float u1 = (float)u[0];
    float u2 = (float)u[1];
    float u3 = (float)u[2];
    float u4 = (float)u[3];

    /* Physics Parameters */
    const float alpha = 0.22f;
    const float h_conv = 0.035f;
    const float T_amb = 25.0f;
    const float dt_sub = (float)(SAMPLE_TIME / (double)SUBSTEPS_PER_STEP);

    size_t global_work_size[2] = {GRID_W, GRID_H};
    size_t local_work_size[2] = {16, 16};

    cl_mem in_buf = g_gpu_state.d_T_curr;
    cl_mem out_buf = g_gpu_state.d_T_next;

    /* Execute sub-steps on GPU */
    for (int step = 0; step < SUBSTEPS_PER_STEP; step++) {
        clSetKernelArg(g_gpu_state.kernel, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(g_gpu_state.kernel, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(g_gpu_state.kernel, 2, sizeof(float), &alpha);
        clSetKernelArg(g_gpu_state.kernel, 3, sizeof(float), &h_conv);
        clSetKernelArg(g_gpu_state.kernel, 4, sizeof(float), &T_amb);
        clSetKernelArg(g_gpu_state.kernel, 5, sizeof(float), &dt_sub);
        clSetKernelArg(g_gpu_state.kernel, 6, sizeof(float), &u1);
        clSetKernelArg(g_gpu_state.kernel, 7, sizeof(float), &u2);
        clSetKernelArg(g_gpu_state.kernel, 8, sizeof(float), &u3);
        clSetKernelArg(g_gpu_state.kernel, 9, sizeof(float), &u4);

        clEnqueueNDRangeKernel(g_gpu_state.queue, g_gpu_state.kernel, 2, NULL,
                               global_work_size, local_work_size, 0, NULL, NULL);

        /* Swap ping-pong buffers */
        cl_mem temp = in_buf;
        in_buf = out_buf;
        out_buf = temp;
    }

    g_gpu_state.d_T_curr = in_buf;
    g_gpu_state.d_T_next = out_buf;

    /* Read back temperature grid from GPU to Host */
    clEnqueueReadBuffer(g_gpu_state.queue, g_gpu_state.d_T_curr, CL_TRUE, 0,
                        TOTAL_NODES * sizeof(float), g_gpu_state.h_T, 0, NULL, NULL);
}

static void mdlTerminate(SimStruct *S) {
    if (g_gpu_state.is_initialized) {
        /* Save final field to binary file for visualization */
        FILE *fp = fopen("thermal_field_final_gpu.bin", "wb");
        if (fp && g_gpu_state.h_T) {
            fwrite(g_gpu_state.h_T, sizeof(float), TOTAL_NODES, fp);
            fclose(fp);
        }

        if (g_gpu_state.d_T_curr) clReleaseMemObject(g_gpu_state.d_T_curr);
        if (g_gpu_state.d_T_next) clReleaseMemObject(g_gpu_state.d_T_next);
        if (g_gpu_state.kernel) clReleaseKernel(g_gpu_state.kernel);
        if (g_gpu_state.program) clReleaseProgram(g_gpu_state.program);
        if (g_gpu_state.queue) clReleaseCommandQueue(g_gpu_state.queue);
        if (g_gpu_state.context) clReleaseContext(g_gpu_state.context);
        if (g_gpu_state.h_T) free(g_gpu_state.h_T);
        memset(&g_gpu_state, 0, sizeof(OpenCLThermalState));
    }
}

#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
