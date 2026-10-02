/*
 * sfun_pendulum_ensemble_opencl.c
 *
 * Level-2 C-MEX S-Function for MATLAB Simulink R2025b
 * Massive Parallel Inverted Pendulum Ensemble Simulator (1,024 systems)
 * Accelerated on NVIDIA GeForce GT 750M (384 Kepler CUDA Cores) via OpenCL / Rusticl.
 *
 * Simulates 1,024 independent nonlinear Cart-Pole dynamics in parallel with
 * dual-mode Energy Swing-Up and LQR Upright Stabilization.
 */

#define S_FUNCTION_NAME  sfun_pendulum_ensemble_opencl
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#define ENSEMBLE_SIZE 1024
#define SUBSTEPS 5
#define SAMPLE_TIME 0.01

/* OpenCL Kernel Source */
static const char *pendulum_kernel_source =
"typedef struct {\n"
"    float x;        /* cart position (m) */\n"
"    float x_dot;    /* cart velocity (m/s) */\n"
"    float theta;    /* pole angle from upright (rad) */\n"
"    float theta_dot;/* pole angular velocity (rad/s) */\n"
"} CartPoleState;\n"
"\n"
"/* Derivative function for Cart-Pole nonlinear dynamics */\n"
"void get_derivatives(CartPoleState s, float F, float *dx, float *dv, float *dth, float *domega) {\n"
"    const float M = 1.0f;\n"
"    const float m = 0.2f;\n"
"    const float l = 0.5f;\n"
"    const float I = 0.016667f;\n"
"    const float g = 9.81f;\n"
"    const float b = 0.1f;\n"
"    const float c = 0.01f;\n"
"\n"
"    float sin_th = sin(s.theta);\n"
"    float cos_th = cos(s.theta);\n"
"    float ml_cos = m * l * cos_th;\n"
"    float ml_sin = m * l * sin_th;\n"
"\n"
"    /* Mass matrix determinant */\n"
"    float det = (M + m) * (I + m * l * l) - ml_cos * ml_cos;\n"
"\n"
"    /* Right-hand side terms */\n"
"    float rhs1 = F - b * s.x_dot + ml_sin * s.theta_dot * s.theta_dot;\n"
"    float rhs2 = m * g * l * sin_th - c * s.theta_dot;\n"
"\n"
"    *dx = s.x_dot;\n"
"    *dv = ((I + m * l * l) * rhs1 + ml_cos * rhs2) / det;\n"
"    *dth = s.theta_dot;\n"
"    *domega = (ml_cos * rhs1 + (M + m) * rhs2) / det;\n"
"}\n"
"\n"
"__kernel void simulate_pendulums(\n"
"    __global CartPoleState *states,\n"
"    __global float *forces,\n"
"    const float dt_sub,\n"
"    const int num_substeps)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id >= 1024) return;\n"
"\n"
"    CartPoleState s = states[id];\n"
"    float last_F = 0.0f;\n"
"\n"
"    for (int step = 0; step < num_substeps; step++) {\n"
"        /* Normalize theta to [-pi, pi] */\n"
"        while (s.theta > 3.14159265f) s.theta -= 6.2831853f;\n"
"        while (s.theta < -3.14159265f) s.theta += 6.2831853f;\n"
"\n"
"        /* Dual-Mode Control Law */\n"
"        float F = 0.0f;\n"
"        if (fabs(s.theta) < 0.40f) {\n"
"            /* Upright LQR Stabilization */\n"
"            /* K = [-31.62, -28.45, 185.32, 42.15] */\n"
"            F = -(-31.62f * s.x - 28.45f * s.x_dot + 185.32f * s.theta + 42.15f * s.theta_dot);\n"
"        } else {\n"
"            /* Energy-Based Swing-Up Control */\n"
"            const float m = 0.2f, l = 0.5f, g = 9.81f, I = 0.016667f;\n"
"            float E = 0.5f * (I + m * l * l) * s.theta_dot * s.theta_dot + m * g * l * (cos(s.theta) - 1.0f);\n"
"            float sgn = (s.theta_dot * cos(s.theta) >= 0.0f) ? 1.0f : -1.0f;\n"
"            F = 18.0f * E * sgn - 2.5f * s.x - 2.0f * s.x_dot;\n"
"        }\n"
"\n"
"        /* Actuator Saturation [-25N, 25N] */\n"
"        if (F > 25.0f) F = 25.0f;\n"
"        if (F < -25.0f) F = -25.0f;\n"
"        last_F = F;\n"
"\n"
"        /* 4th-Order Runge-Kutta (RK4) Integration */\n"
"        float k1_x, k1_v, k1_th, k1_w;\n"
"        get_derivatives(s, F, &k1_x, &k1_v, &k1_th, &k1_w);\n"
"\n"
"        CartPoleState s2;\n"
"        s2.x = s.x + 0.5f * dt_sub * k1_x;\n"
"        s2.x_dot = s.x_dot + 0.5f * dt_sub * k1_v;\n"
"        s2.theta = s.theta + 0.5f * dt_sub * k1_th;\n"
"        s2.theta_dot = s.theta_dot + 0.5f * dt_sub * k1_w;\n"
"        float k2_x, k2_v, k2_th, k2_w;\n"
"        get_derivatives(s2, F, &k2_x, &k2_v, &k2_th, &k2_w);\n"
"\n"
"        CartPoleState s3;\n"
"        s3.x = s.x + 0.5f * dt_sub * k2_x;\n"
"        s3.x_dot = s.x_dot + 0.5f * dt_sub * k2_v;\n"
"        s3.theta = s.theta + 0.5f * dt_sub * k2_th;\n"
"        s3.theta_dot = s.theta_dot + 0.5f * dt_sub * k2_w;\n"
"        float k3_x, k3_v, k3_th, k3_w;\n"
"        get_derivatives(s3, F, &k3_x, &k3_v, &k3_th, &k3_w);\n"
"\n"
"        CartPoleState s4;\n"
"        s4.x = s.x + dt_sub * k3_x;\n"
"        s4.x_dot = s.x_dot + dt_sub * k3_v;\n"
"        s4.theta = s.theta + dt_sub * k3_th;\n"
"        s4.theta_dot = s.theta_dot + dt_sub * k3_w;\n"
"        float k4_x, k4_v, k4_th, k4_w;\n"
"        get_derivatives(s4, F, &k4_x, &k4_v, &k4_th, &k4_w);\n"
"\n"
"        s.x += (dt_sub / 6.0f) * (k1_x + 2.0f * k2_x + 2.0f * k3_x + k4_x);\n"
"        s.x_dot += (dt_sub / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);\n"
"        s.theta += (dt_sub / 6.0f) * (k1_th + 2.0f * k2_th + 2.0f * k3_th + k4_th);\n"
"        s.theta_dot += (dt_sub / 6.0f) * (k1_w + 2.0f * k2_w + 2.0f * k3_w + k4_w);\n"
"    }\n"
"\n"
"    states[id] = s;\n"
"    forces[id] = last_F;\n"
"}\n";

typedef struct {
    float x;
    float x_dot;
    float theta;
    float theta_dot;
} CartPoleState;

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_states;
    cl_mem d_forces;
    CartPoleState *h_states;
    float *h_forces;
    int is_initialized;
} GPUPendulumContext;

static GPUPendulumContext g_gpu = {0};

static void mdlInitializeSizes(SimStruct *S) {
    ssSetNumSFcnParams(S, 0);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) return;

    /* Input Port 0: Optional disturbance force on reference cart */
    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 1);
    ssSetInputPortDirectFeedThrough(S, 0, 0);
    ssSetInputPortRequiredContiguous(S, 0, 1);

    /* Output Port 0: Reference Cart States [x0, x_dot0, theta0, theta_dot0, F0] */
    if (!ssSetNumOutputPorts(S, 2)) return;
    ssSetOutputPortWidth(S, 0, 5);

    /* Output Port 1: Ensemble Telemetry [% stabilized, mean_err, max_err] */
    ssSetOutputPortWidth(S, 1, 3);

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
    if (g_gpu.is_initialized) return;

    g_gpu.h_states = (CartPoleState *)malloc(ENSEMBLE_SIZE * sizeof(CartPoleState));
    g_gpu.h_forces = (float *)malloc(ENSEMBLE_SIZE * sizeof(float));

    /* Initialize 1,024 systems with diverse initial conditions */
    /* System 0: Pure downward hanging (theta = pi), needs swing-up */
    g_gpu.h_states[0].x = 0.0f;
    g_gpu.h_states[0].x_dot = 0.0f;
    g_gpu.h_states[0].theta = 3.14159f;
    g_gpu.h_states[0].theta_dot = 0.01f;
    g_gpu.h_forces[0] = 0.0f;

    /* Systems 1..1023: Basin of Attraction Grid sweep */
    /* 32 angles x 32 angular velocities = 1024 initial conditions */
    for (int i = 1; i < ENSEMBLE_SIZE; i++) {
        int angle_idx = i % 32;
        int vel_idx = i / 32;

        float init_th = -3.14159f + angle_idx * (6.28318f / 31.0f);
        float init_w = -4.0f + vel_idx * (8.0f / 31.0f);

        g_gpu.h_states[i].x = 0.0f;
        g_gpu.h_states[i].x_dot = 0.0f;
        g_gpu.h_states[i].theta = init_th;
        g_gpu.h_states[i].theta_dot = init_w;
        g_gpu.h_forces[i] = 0.0f;
    }

    cl_uint num_platforms = 0;
    clGetPlatformIDs(1, &g_gpu.platform, &num_platforms);
    cl_uint num_devices = 0;
    clGetDeviceIDs(g_gpu.platform, CL_DEVICE_TYPE_GPU, 1, &g_gpu.device, &num_devices);

    g_gpu.context = clCreateContext(NULL, 1, &g_gpu.device, NULL, NULL, &err);
    g_gpu.queue = clCreateCommandQueueWithProperties(g_gpu.context, g_gpu.device, NULL, &err);

    g_gpu.program = clCreateProgramWithSource(g_gpu.context, 1, &pendulum_kernel_source, NULL, &err);
    clBuildProgram(g_gpu.program, 1, &g_gpu.device, "-cl-fast-relaxed-math", NULL, NULL);

    g_gpu.kernel = clCreateKernel(g_gpu.program, "simulate_pendulums", &err);

    size_t state_bytes = ENSEMBLE_SIZE * sizeof(CartPoleState);
    size_t force_bytes = ENSEMBLE_SIZE * sizeof(float);
    g_gpu.d_states = clCreateBuffer(g_gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, state_bytes, g_gpu.h_states, &err);
    g_gpu.d_forces = clCreateBuffer(g_gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, force_bytes, g_gpu.h_forces, &err);

    g_gpu.is_initialized = 1;
    printf("⚡ [S-Function GPU] Initialized 1,024 Parallel Inverted Pendulums on 384 CUDA Cores!\n");
}

static void mdlOutputs(SimStruct *S, int_T tid) {
    real_T *y0 = ssGetOutputPortRealSignal(S, 0);
    real_T *y1 = ssGetOutputPortRealSignal(S, 1);

    if (g_gpu.is_initialized && g_gpu.h_states) {
        /* Output Reference Cart (System 0) */
        y0[0] = (real_T)g_gpu.h_states[0].x;
        y0[1] = (real_T)g_gpu.h_states[0].x_dot;
        y0[2] = (real_T)g_gpu.h_states[0].theta;
        y0[3] = (real_T)g_gpu.h_states[0].theta_dot;
        y0[4] = (real_T)g_gpu.h_forces[0];

        /* Calculate Ensemble Statistics */
        int stabilized_count = 0;
        float sum_err = 0.0f;
        float max_err = 0.0f;

        for (int i = 0; i < ENSEMBLE_SIZE; i++) {
            float err_th = fabsf(g_gpu.h_states[i].theta);
            if (err_th < 0.05f && fabsf(g_gpu.h_states[i].x) < 0.5f) {
                stabilized_count++;
            }
            sum_err += err_th;
            if (err_th > max_err) max_err = err_th;
        }

        y1[0] = (real_T)stabilized_count / (real_T)ENSEMBLE_SIZE * 100.0; /* % stabilized */
        y1[1] = (real_T)(sum_err / (float)ENSEMBLE_SIZE);                 /* mean angle error */
        y1[2] = (real_T)max_err;                                          /* max angle error */
    } else {
        for (int i = 0; i < 5; i++) y0[i] = 0.0;
        for (int i = 0; i < 3; i++) y1[i] = 0.0;
    }
}

#define MDL_UPDATE
static void mdlUpdate(SimStruct *S, int_T tid) {
    if (!g_gpu.is_initialized) return;

    float dt_sub = (float)(SAMPLE_TIME / (double)SUBSTEPS);
    int num_substeps = SUBSTEPS;

    clSetKernelArg(g_gpu.kernel, 0, sizeof(cl_mem), &g_gpu.d_states);
    clSetKernelArg(g_gpu.kernel, 1, sizeof(cl_mem), &g_gpu.d_forces);
    clSetKernelArg(g_gpu.kernel, 2, sizeof(float), &dt_sub);
    clSetKernelArg(g_gpu.kernel, 3, sizeof(int), &num_substeps);

    size_t global_size = ENSEMBLE_SIZE;
    size_t local_size = 64;

    clEnqueueNDRangeKernel(g_gpu.queue, g_gpu.kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    /* Read back results to host */
    clEnqueueReadBuffer(g_gpu.queue, g_gpu.d_states, CL_TRUE, 0,
                        ENSEMBLE_SIZE * sizeof(CartPoleState), g_gpu.h_states, 0, NULL, NULL);
    clEnqueueReadBuffer(g_gpu.queue, g_gpu.d_forces, CL_TRUE, 0,
                        ENSEMBLE_SIZE * sizeof(float), g_gpu.h_forces, 0, NULL, NULL);
}

static void mdlTerminate(SimStruct *S) {
    if (g_gpu.is_initialized) {
        /* Save final ensemble states for basin of attraction mapping */
        FILE *fp = fopen("pendulum_ensemble_final.bin", "wb");
        if (fp && g_gpu.h_states) {
            fwrite(g_gpu.h_states, sizeof(CartPoleState), ENSEMBLE_SIZE, fp);
            fclose(fp);
        }

        if (g_gpu.d_states) clReleaseMemObject(g_gpu.d_states);
        if (g_gpu.d_forces) clReleaseMemObject(g_gpu.d_forces);
        if (g_gpu.kernel) clReleaseKernel(g_gpu.kernel);
        if (g_gpu.program) clReleaseProgram(g_gpu.program);
        if (g_gpu.queue) clReleaseCommandQueue(g_gpu.queue);
        if (g_gpu.context) clReleaseContext(g_gpu.context);
        if (g_gpu.h_states) free(g_gpu.h_states);
        if (g_gpu.h_forces) free(g_gpu.h_forces);
        memset(&g_gpu, 0, sizeof(GPUPendulumContext));
    }
}

#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
