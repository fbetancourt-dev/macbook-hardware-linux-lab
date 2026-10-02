/*
 * sfun_pendulum_ensemble_cpu.c
 *
 * Level-2 C-MEX S-Function for MATLAB Simulink R2025b
 * 1,024 Parallel Inverted Pendulums simulated on Host CPU (Reference FP32)
 */

#define S_FUNCTION_NAME  sfun_pendulum_ensemble_cpu
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ENSEMBLE_SIZE 1024
#define SUBSTEPS 5
#define SAMPLE_TIME 0.01

typedef struct {
    float x;
    float x_dot;
    float theta;
    float theta_dot;
} CartPoleState;

typedef struct {
    CartPoleState *states;
    float *forces;
    int is_initialized;
} CPUPendulumContext;

static CPUPendulumContext g_cpu = {0};

static void get_derivatives(CartPoleState s, float F, float *dx, float *dv, float *dth, float *domega) {
    const float M = 1.0f;
    const float m = 0.2f;
    const float l = 0.5f;
    const float I = 0.016667f;
    const float g = 9.81f;
    const float b = 0.1f;
    const float c = 0.01f;

    float sin_th = sinf(s.theta);
    float cos_th = cosf(s.theta);
    float ml_cos = m * l * cos_th;
    float ml_sin = m * l * sin_th;

    float det = (M + m) * (I + m * l * l) - ml_cos * ml_cos;

    float rhs1 = F - b * s.x_dot + ml_sin * s.theta_dot * s.theta_dot;
    float rhs2 = m * g * l * sin_th - c * s.theta_dot;

    *dx = s.x_dot;
    *dv = ((I + m * l * l) * rhs1 + ml_cos * rhs2) / det;
    *dth = s.theta_dot;
    *domega = (ml_cos * rhs1 + (M + m) * rhs2) / det;
}

static void mdlInitializeSizes(SimStruct *S) {
    ssSetNumSFcnParams(S, 0);
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) return;

    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, 1);
    ssSetInputPortDirectFeedThrough(S, 0, 0);
    ssSetInputPortRequiredContiguous(S, 0, 1);

    if (!ssSetNumOutputPorts(S, 2)) return;
    ssSetOutputPortWidth(S, 0, 5);
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
    if (g_cpu.is_initialized) return;

    g_cpu.states = (CartPoleState *)malloc(ENSEMBLE_SIZE * sizeof(CartPoleState));
    g_cpu.forces = (float *)malloc(ENSEMBLE_SIZE * sizeof(float));

    /* System 0: Downward hanging */
    g_cpu.states[0].x = 0.0f;
    g_cpu.states[0].x_dot = 0.0f;
    g_cpu.states[0].theta = 3.14159f;
    g_cpu.states[0].theta_dot = 0.01f;
    g_cpu.forces[0] = 0.0f;

    for (int i = 1; i < ENSEMBLE_SIZE; i++) {
        int angle_idx = i % 32;
        int vel_idx = i / 32;

        float init_th = -3.14159f + angle_idx * (6.28318f / 31.0f);
        float init_w = -4.0f + vel_idx * (8.0f / 31.0f);

        g_cpu.states[i].x = 0.0f;
        g_cpu.states[i].x_dot = 0.0f;
        g_cpu.states[i].theta = init_th;
        g_cpu.states[i].theta_dot = init_w;
        g_cpu.forces[i] = 0.0f;
    }

    g_cpu.is_initialized = 1;
    printf("⏱️  [S-Function CPU] Initialized 1,024 Parallel Inverted Pendulums on Host CPU\n");
}

static void mdlOutputs(SimStruct *S, int_T tid) {
    real_T *y0 = ssGetOutputPortRealSignal(S, 0);
    real_T *y1 = ssGetOutputPortRealSignal(S, 1);

    if (g_cpu.is_initialized && g_cpu.states) {
        y0[0] = (real_T)g_cpu.states[0].x;
        y0[1] = (real_T)g_cpu.states[0].x_dot;
        y0[2] = (real_T)g_cpu.states[0].theta;
        y0[3] = (real_T)g_cpu.states[0].theta_dot;
        y0[4] = (real_T)g_cpu.forces[0];

        int stabilized_count = 0;
        float sum_err = 0.0f;
        float max_err = 0.0f;

        for (int i = 0; i < ENSEMBLE_SIZE; i++) {
            float err_th = fabsf(g_cpu.states[i].theta);
            if (err_th < 0.05f && fabsf(g_cpu.states[i].x) < 0.5f) {
                stabilized_count++;
            }
            sum_err += err_th;
            if (err_th > max_err) max_err = err_th;
        }

        y1[0] = (real_T)stabilized_count / (real_T)ENSEMBLE_SIZE * 100.0;
        y1[1] = (real_T)(sum_err / (float)ENSEMBLE_SIZE);
        y1[2] = (real_T)max_err;
    } else {
        for (int i = 0; i < 5; i++) y0[i] = 0.0;
        for (int i = 0; i < 3; i++) y1[i] = 0.0;
    }
}

#define MDL_UPDATE
static void mdlUpdate(SimStruct *S, int_T tid) {
    if (!g_cpu.is_initialized) return;

    float dt_sub = (float)(SAMPLE_TIME / (double)SUBSTEPS);

    for (int id = 0; id < ENSEMBLE_SIZE; id++) {
        CartPoleState s = g_cpu.states[id];
        float last_F = 0.0f;

        for (int step = 0; step < SUBSTEPS; step++) {
            while (s.theta > 3.14159265f) s.theta -= 6.2831853f;
            while (s.theta < -3.14159265f) s.theta += 6.2831853f;

            float F = 0.0f;
            if (fabsf(s.theta) < 0.40f) {
                F = -(-31.62f * s.x - 28.45f * s.x_dot + 185.32f * s.theta + 42.15f * s.theta_dot);
            } else {
                const float m = 0.2f, l = 0.5f, g = 9.81f, I = 0.016667f;
                float E = 0.5f * (I + m * l * l) * s.theta_dot * s.theta_dot + m * g * l * (cosf(s.theta) - 1.0f);
                float sgn = (s.theta_dot * cosf(s.theta) >= 0.0f) ? 1.0f : -1.0f;
                F = 18.0f * E * sgn - 2.5f * s.x - 2.0f * s.x_dot;
            }

            if (F > 25.0f) F = 25.0f;
            if (F < -25.0f) F = -25.0f;
            last_F = F;

            float k1_x, k1_v, k1_th, k1_w;
            get_derivatives(s, F, &k1_x, &k1_v, &k1_th, &k1_w);

            CartPoleState s2;
            s2.x = s.x + 0.5f * dt_sub * k1_x;
            s2.x_dot = s.x_dot + 0.5f * dt_sub * k1_v;
            s2.theta = s.theta + 0.5f * dt_sub * k1_th;
            s2.theta_dot = s.theta_dot + 0.5f * dt_sub * k1_w;
            float k2_x, k2_v, k2_th, k2_w;
            get_derivatives(s2, F, &k2_x, &k2_v, &k2_th, &k2_w);

            CartPoleState s3;
            s3.x = s.x + 0.5f * dt_sub * k2_x;
            s3.x_dot = s.x_dot + 0.5f * dt_sub * k2_v;
            s3.theta = s.theta + 0.5f * dt_sub * k2_th;
            s3.theta_dot = s.theta_dot + 0.5f * dt_sub * k2_w;
            float k3_x, k3_v, k3_th, k3_w;
            get_derivatives(s3, F, &k3_x, &k3_v, &k3_th, &k3_w);

            CartPoleState s4;
            s4.x = s.x + dt_sub * k3_x;
            s4.x_dot = s.x_dot + dt_sub * k3_v;
            s4.theta = s.theta + dt_sub * k3_th;
            s4.theta_dot = s.theta_dot + dt_sub * k3_w;
            float k4_x, k4_v, k4_th, k4_w;
            get_derivatives(s4, F, &k4_x, &k4_v, &k4_th, &k4_w);

            s.x += (dt_sub / 6.0f) * (k1_x + 2.0f * k2_x + 2.0f * k3_x + k4_x);
            s.x_dot += (dt_sub / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
            s.theta += (dt_sub / 6.0f) * (k1_th + 2.0f * k2_th + 2.0f * k3_th + k4_th);
            s.theta_dot += (dt_sub / 6.0f) * (k1_w + 2.0f * k2_w + 2.0f * k3_w + k4_w);
        }

        g_cpu.states[id] = s;
        g_cpu.forces[id] = last_F;
    }
}

static void mdlTerminate(SimStruct *S) {
    if (g_cpu.is_initialized) {
        if (g_cpu.states) free(g_cpu.states);
        if (g_cpu.forces) free(g_cpu.forces);
        memset(&g_cpu, 0, sizeof(CPUPendulumContext));
    }
}

#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
