/*
 * sfun_thermal_2d_cpu.c
 *
 * Level-2 C-MEX S-Function for MATLAB Simulink R2025b
 * 2D Distributed Thermal Convection-Diffusion Plant (256x256 = 65,536 nodes)
 * Reference Single-Precision (FP32) implementation running on Host CPU.
 *
 * Inputs:  [u1, u2, u3, u4] -> Actuator heating power in % (0 - 100)
 * Outputs: [T1, T2, T3, T4] -> Measured zone temperatures in deg C
 */

#define S_FUNCTION_NAME  sfun_thermal_2d_cpu
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GRID_W 256
#define GRID_H 256
#define TOTAL_NODES (GRID_W * GRID_H)
#define NUM_ZONES 4
#define SUBSTEPS_PER_STEP 5
#define SAMPLE_TIME 0.05

typedef struct {
    float *T_curr;
    float *T_next;
    int is_initialized;
} CPUThermalState;

static CPUThermalState g_cpu_state = {0};

/* Zone Center Coordinates */
static const int ZONE_X[NUM_ZONES] = {64, 192, 64, 192};
static const int ZONE_Y[NUM_ZONES] = {64, 64, 192, 192};
#define SENSOR_RADIUS 4

/* Helper: Calculate sensor readings from grid */
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
    ssSetInputPortDirectFeedThrough(S, 0, 1);
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
    if (g_cpu_state.is_initialized) return;

    g_cpu_state.T_curr = (float *)malloc(TOTAL_NODES * sizeof(float));
    g_cpu_state.T_next = (float *)malloc(TOTAL_NODES * sizeof(float));

    for (int i = 0; i < TOTAL_NODES; i++) {
        g_cpu_state.T_curr[i] = 25.0f;
        g_cpu_state.T_next[i] = 25.0f;
    }
    g_cpu_state.is_initialized = 1;
    printf("⏱️  [S-Function CPU FP32] Initialized 2D Thermal Plant (%dx%d = %d nodes)\n",
           GRID_W, GRID_H, TOTAL_NODES);
}

static void mdlOutputs(SimStruct *S, int_T tid) {
    real_T *y = ssGetOutputPortRealSignal(S, 0);
    if (g_cpu_state.is_initialized && g_cpu_state.T_curr) {
        read_sensor_temperatures(g_cpu_state.T_curr, y);
    } else {
        for (int i = 0; i < NUM_ZONES; i++) y[i] = 25.0;
    }
}

#define MDL_UPDATE
static void mdlUpdate(SimStruct *S, int_T tid) {
    if (!g_cpu_state.is_initialized) return;

    const real_T *u = (const real_T *)ssGetInputPortSignal(S, 0);
    float u1 = (float)u[0];
    float u2 = (float)u[1];
    float u3 = (float)u[2];
    float u4 = (float)u[3];

    const float alpha = 0.22f;
    const float h_conv = 0.035f;
    const float T_amb = 25.0f;
    const float dt_sub = (float)(SAMPLE_TIME / (double)SUBSTEPS_PER_STEP);
    const float inv_2sig2 = 1.0f / (2.0f * 100.0f);
    const float Q_max = 18.0f;

    float *in_grid = g_cpu_state.T_curr;
    float *out_grid = g_cpu_state.T_next;

    for (int step = 0; step < SUBSTEPS_PER_STEP; step++) {
        for (int y = 0; y < GRID_H; y++) {
            for (int x = 0; x < GRID_W; x++) {
                int idx = y * GRID_W + x;

                if (x == 0 || x == GRID_W - 1 || y == 0 || y == GRID_H - 1) {
                    out_grid[idx] = T_amb;
                    continue;
                }

                float Tc = in_grid[idx];
                float Tl = in_grid[idx - 1];
                float Tr = in_grid[idx + 1];
                float Tu = in_grid[idx - GRID_W];
                float Td = in_grid[idx + GRID_W];

                float laplacian = (Tl + Tr + Tu + Td - 4.0f * Tc);
                float loss = -h_conv * (Tc - T_amb);

                float d1_sq = (x - 64.0f) * (x - 64.0f) + (y - 64.0f) * (y - 64.0f);
                float d2_sq = (x - 192.0f) * (x - 192.0f) + (y - 64.0f) * (y - 64.0f);
                float d3_sq = (x - 64.0f) * (x - 64.0f) + (y - 192.0f) * (y - 192.0f);
                float d4_sq = (x - 192.0f) * (x - 192.0f) + (y - 192.0f) * (y - 192.0f);

                float q1 = (u1 * 0.01f) * Q_max * expf(-d1_sq * inv_2sig2);
                float q2 = (u2 * 0.01f) * Q_max * expf(-d2_sq * inv_2sig2);
                float q3 = (u3 * 0.01f) * Q_max * expf(-d3_sq * inv_2sig2);
                float q4 = (u4 * 0.01f) * Q_max * expf(-d4_sq * inv_2sig2);
                float Q_total = q1 + q2 + q3 + q4;

                out_grid[idx] = Tc + dt_sub * (alpha * laplacian + loss + Q_total);
            }
        }
        /* Swap pointers */
        float *tmp = in_grid;
        in_grid = out_grid;
        out_grid = tmp;
    }

    g_cpu_state.T_curr = in_grid;
    g_cpu_state.T_next = out_grid;
}

static void mdlTerminate(SimStruct *S) {
    if (g_cpu_state.is_initialized) {
        FILE *fp = fopen("thermal_field_final_cpu.bin", "wb");
        if (fp && g_cpu_state.T_curr) {
            fwrite(g_cpu_state.T_curr, sizeof(float), TOTAL_NODES, fp);
            fclose(fp);
        }

        if (g_cpu_state.T_curr) free(g_cpu_state.T_curr);
        if (g_cpu_state.T_next) free(g_cpu_state.T_next);
        memset(&g_cpu_state, 0, sizeof(CPUThermalState));
    }
}

#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
