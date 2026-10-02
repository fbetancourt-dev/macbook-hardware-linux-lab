#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define NUM_LAYERS 2
#define QK4_0 32
#define D_MODEL 1536
#define HEAD_DIM 128
#define N_HEADS_Q 12
#define N_HEADS_KV 2
#define GQA_GROUP_SIZE 6
#define D_QKV 2048
#define D_FFN 8960
#define T_MAX 4096
#define ROPE_BASE 1000000.0f
#define EPSILON 1e-6f
#define PV_SEGMENT_SIZE 256
#define MAX_SEGMENTS (T_MAX / PV_SEGMENT_SIZE)

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} block_q4_0;

#define CHECK_CL(err, msg) do { \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FATAL OpenCL Error at %s:%d: %s (code %d)\n", __FILE__, __LINE__, msg, err); \
        exit(1); \
    } \
} while (0)

static char* load_kernel_source(const char* filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    char *src = (char*)malloc(size + 1);
    size_t read_bytes = fread(src, 1, size, fp);
    (void)read_bytes;
    src[size] = '\0';
    fclose(fp);
    return src;
}

int main(void) {
    printf("========================================================================================\n");
    printf(" Kernel Execution Breakdown & Inter-Kernel Gaps Diagnostic (T=1 vs T=1024)             \n");
    printf("========================================================================================\n");

    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id device = NULL;
    cl_int err;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices) == CL_SUCCESS && num_devices > 0) {
            cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
            device = devices[0];
            free(devices);
            break;
        }
    }
    free(platforms);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_CL(err, "context");
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); CHECK_CL(err, "queue");

    char *src = load_kernel_source("kernel_decoder_layer.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err); CHECK_CL(err, "program");
    free(src);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL); CHECK_CL(err, "build");

    const char* kernel_names[11] = {
        "RMSNorm (Attn)", "QKV GEMV", "RoPE & KV", "Scores", "Softmax",
        "PV Combine", "PV Reduce", "Wo Residual", "RMSNorm (FFN)", "SwiGLU Fused", "Down Residual"
    };

    cl_kernel k_l0[11], k_l1[11];
    const char* func_names[11] = {
        "kernel_rmsnorm", "gemv_q4_0_bias", "kernel_rope_and_kv_append", "kernel_gqa_scores",
        "kernel_softmax_gqa", "kernel_gqa_value_combine_segmented", "kernel_gqa_reduce_segments",
        "gemv_q4_0_wo_residual", "kernel_rmsnorm", "gemv_swiglu_fused", "gemv_q4_0_down_residual"
    };

    for (int k = 0; k < 11; k++) {
        k_l0[k] = clCreateKernel(program, func_names[k], &err); CHECK_CL(err, func_names[k]);
        k_l1[k] = clCreateKernel(program, func_names[k], &err); CHECK_CL(err, func_names[k]);
    }

    // Dummy Buffers for diagnostics
    size_t sz_W_qkv = sizeof(block_q4_0) * D_QKV * (D_MODEL / 32);
    size_t sz_W_o   = sizeof(block_q4_0) * D_MODEL * (D_MODEL / 32);
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * (D_MODEL / 32);
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * (D_FFN / 32);
    size_t sz_kv = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;

    cl_mem d_W_qkv[2], d_b_qkv[2], d_W_o[2], d_W_gate[2], d_W_up[2], d_W_down[2], d_gamma_attn[2], d_gamma_ffn[2];
    cl_mem d_k_cache[2], d_v_cache[2];

    for (int l = 0; l < 2; l++) {
        d_W_qkv[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sz_W_qkv, NULL, &err);
        d_b_qkv[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*D_QKV, NULL, &err);
        d_W_o[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sz_W_o, NULL, &err);
        d_W_gate[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sz_W_gate, NULL, &err);
        d_W_up[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sz_W_gate, NULL, &err);
        d_W_down[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sz_W_down, NULL, &err);
        d_gamma_attn[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*D_MODEL, NULL, &err);
        d_gamma_ffn[l] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*D_MODEL, NULL, &err);
        d_k_cache[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err);
        d_v_cache[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, sz_kv, NULL, &err);
    }

    cl_mem d_state = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_MODEL, NULL, &err);
    cl_mem d_norm = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_MODEL, NULL, &err);
    cl_mem d_qkv = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_QKV, NULL, &err);
    cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N_HEADS_Q*T_MAX, NULL, &err);
    cl_mem d_partial = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*N_HEADS_Q*MAX_SEGMENTS*HEAD_DIM, NULL, &err);
    cl_mem d_h = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*D_FFN, NULL, &err);

    int test_contexts[] = {1, 1024};
    for (int tc = 0; tc < 2; tc++) {
        int seq_len = test_contexts[tc];
        int pos = seq_len - 1;
        int d_model = D_MODEL, param_d_qkv = D_QKV, t_max = T_MAX, d_ffn = D_FFN;
        float eps = EPSILON, rope_base = ROPE_BASE, scale_factor = 1.0f / sqrtf(128.0f);
        int num_segments = (seq_len + PV_SEGMENT_SIZE - 1) / PV_SEGMENT_SIZE;

        size_t k_offset_bytes = 1536 * sizeof(float);
        size_t v_offset_bytes = (1536 + 256) * sizeof(float);
        cl_buffer_region reg_k = { k_offset_bytes, 256 * sizeof(float) };
        cl_buffer_region reg_v = { v_offset_bytes, 256 * sizeof(float) };
        cl_mem d_k_sub = clCreateSubBuffer(d_qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_k, &err);
        cl_mem d_v_sub = clCreateSubBuffer(d_qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_v, &err);

        for (int l = 0; l < 2; l++) {
            cl_kernel *k = (l == 0) ? k_l0 : k_l1;
            clSetKernelArg(k[0], 0, sizeof(cl_mem), &d_state); clSetKernelArg(k[0], 1, sizeof(cl_mem), &d_gamma_attn[l]);
            clSetKernelArg(k[0], 2, sizeof(cl_mem), &d_norm); clSetKernelArg(k[0], 3, sizeof(int), &d_model); clSetKernelArg(k[0], 4, sizeof(float), &eps);

            clSetKernelArg(k[1], 0, sizeof(cl_mem), &d_W_qkv[l]); clSetKernelArg(k[1], 1, sizeof(cl_mem), &d_norm);
            clSetKernelArg(k[1], 2, sizeof(cl_mem), &d_b_qkv[l]); clSetKernelArg(k[1], 3, sizeof(cl_mem), &d_qkv);
            clSetKernelArg(k[1], 4, sizeof(int), &param_d_qkv); clSetKernelArg(k[1], 5, sizeof(int), &d_model);

            clSetKernelArg(k[2], 0, sizeof(cl_mem), &d_qkv); clSetKernelArg(k[2], 1, sizeof(cl_mem), &d_k_sub);
            clSetKernelArg(k[2], 2, sizeof(cl_mem), &d_v_sub); clSetKernelArg(k[2], 3, sizeof(cl_mem), &d_k_cache[l]);
            clSetKernelArg(k[2], 4, sizeof(cl_mem), &d_v_cache[l]); clSetKernelArg(k[2], 5, sizeof(int), &pos);
            clSetKernelArg(k[2], 6, sizeof(int), &t_max); clSetKernelArg(k[2], 7, sizeof(float), &rope_base);

            clSetKernelArg(k[3], 0, sizeof(cl_mem), &d_qkv); clSetKernelArg(k[3], 1, sizeof(cl_mem), &d_k_cache[l]);
            clSetKernelArg(k[3], 2, sizeof(cl_mem), &d_scores); clSetKernelArg(k[3], 3, sizeof(int), &seq_len);
            clSetKernelArg(k[3], 4, sizeof(int), &t_max); clSetKernelArg(k[3], 5, sizeof(float), &scale_factor);

            clSetKernelArg(k[4], 0, sizeof(cl_mem), &d_scores); clSetKernelArg(k[4], 1, sizeof(int), &seq_len); clSetKernelArg(k[4], 2, sizeof(int), &t_max);

            clSetKernelArg(k[5], 0, sizeof(cl_mem), &d_scores); clSetKernelArg(k[5], 1, sizeof(cl_mem), &d_v_cache[l]);
            clSetKernelArg(k[5], 2, sizeof(cl_mem), &d_partial); clSetKernelArg(k[5], 3, sizeof(int), &seq_len);
            clSetKernelArg(k[5], 4, sizeof(int), &t_max); clSetKernelArg(k[5], 5, sizeof(int), &num_segments);

            clSetKernelArg(k[6], 0, sizeof(cl_mem), &d_partial); clSetKernelArg(k[6], 1, sizeof(cl_mem), &d_qkv); clSetKernelArg(k[6], 2, sizeof(int), &num_segments);

            clSetKernelArg(k[7], 0, sizeof(cl_mem), &d_W_o[l]); clSetKernelArg(k[7], 1, sizeof(cl_mem), &d_qkv);
            clSetKernelArg(k[7], 2, sizeof(cl_mem), &d_state); clSetKernelArg(k[7], 3, sizeof(int), &d_model);

            clSetKernelArg(k[8], 0, sizeof(cl_mem), &d_state); clSetKernelArg(k[8], 1, sizeof(cl_mem), &d_gamma_ffn[l]);
            clSetKernelArg(k[8], 2, sizeof(cl_mem), &d_norm); clSetKernelArg(k[8], 3, sizeof(int), &d_model); clSetKernelArg(k[8], 4, sizeof(float), &eps);

            clSetKernelArg(k[9], 0, sizeof(cl_mem), &d_W_gate[l]); clSetKernelArg(k[9], 1, sizeof(cl_mem), &d_W_up[l]);
            clSetKernelArg(k[9], 2, sizeof(cl_mem), &d_norm); clSetKernelArg(k[9], 3, sizeof(cl_mem), &d_h);
            clSetKernelArg(k[9], 4, sizeof(int), &d_ffn); clSetKernelArg(k[9], 5, sizeof(int), &d_model);

            clSetKernelArg(k[10], 0, sizeof(cl_mem), &d_W_down[l]); clSetKernelArg(k[10], 1, sizeof(cl_mem), &d_h);
            clSetKernelArg(k[10], 2, sizeof(cl_mem), &d_state); clSetKernelArg(k[10], 3, sizeof(int), &d_model); clSetKernelArg(k[10], 4, sizeof(int), &d_ffn);
        }

        size_t l_rmsnorm = 128, g_rmsnorm = 128;
        size_t l_qkv = 128, g_qkv = ((D_QKV + 3) / 4) * 128;
        size_t l_rope = 64, g_rope = 64;
        size_t l_scores[2] = { 32, 1 };
        size_t g_scores[2] = { (size_t)seq_len * 32, (size_t)N_HEADS_Q };
        size_t l_soft = 128, g_soft = 12 * 128;
        size_t l_pv[2] = { 128, 1 };
        size_t g_pv[2] = { (size_t)num_segments * 128, (size_t)N_HEADS_Q };
        size_t l_red = 128, g_red = 12 * 128;
        size_t l_wo = 128, g_wo = ((D_MODEL + 3) / 4) * 128;
        size_t l_swiglu = 128, g_swiglu = ((D_FFN + 3) / 4) * 128;
        size_t l_down = 128, g_down = ((D_MODEL + 3) / 4) * 128;

        // Warmup
        for (int l = 0; l < 2; l++) {
            cl_kernel *k = (l == 0) ? k_l0 : k_l1;
            clEnqueueNDRangeKernel(queue, k[0], 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[1], 1, NULL, &g_qkv, &l_qkv, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[2], 1, NULL, &g_rope, &l_rope, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[3], 2, NULL, g_scores, l_scores, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[4], 1, NULL, &g_soft, &l_soft, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[5], 2, NULL, g_pv, l_pv, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[6], 1, NULL, &g_red, &l_red, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[7], 1, NULL, &g_wo, &l_wo, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[8], 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[9], 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k[10], 1, NULL, &g_down, &l_down, 0, NULL, NULL);
        }
        clFinish(queue);

        // Profiling run
        cl_event ev[22];
        for (int l = 0; l < 2; l++) {
            cl_kernel *k = (l == 0) ? k_l0 : k_l1;
            int b = l * 11;
            clEnqueueNDRangeKernel(queue, k[0], 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, &ev[b + 0]);
            clEnqueueNDRangeKernel(queue, k[1], 1, NULL, &g_qkv, &l_qkv, 0, NULL, &ev[b + 1]);
            clEnqueueNDRangeKernel(queue, k[2], 1, NULL, &g_rope, &l_rope, 0, NULL, &ev[b + 2]);
            clEnqueueNDRangeKernel(queue, k[3], 2, NULL, g_scores, l_scores, 0, NULL, &ev[b + 3]);
            clEnqueueNDRangeKernel(queue, k[4], 1, NULL, &g_soft, &l_soft, 0, NULL, &ev[b + 4]);
            clEnqueueNDRangeKernel(queue, k[5], 2, NULL, g_pv, l_pv, 0, NULL, &ev[b + 5]);
            clEnqueueNDRangeKernel(queue, k[6], 1, NULL, &g_red, &l_red, 0, NULL, &ev[b + 6]);
            clEnqueueNDRangeKernel(queue, k[7], 1, NULL, &g_wo, &l_wo, 0, NULL, &ev[b + 7]);
            clEnqueueNDRangeKernel(queue, k[8], 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, &ev[b + 8]);
            clEnqueueNDRangeKernel(queue, k[9], 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, &ev[b + 9]);
            clEnqueueNDRangeKernel(queue, k[10], 1, NULL, &g_down, &l_down, 0, NULL, &ev[b + 10]);
        }
        clFinish(queue);

        printf("\n--- Breakdown for T = %d ---\n", seq_len);
        printf("%-3s | %-24s | %-12s | %-12s\n", "Idx", "Kernel Stage Name", "Duration", "Gap to Next");
        printf("------------------------------------------------------------------------\n");

        double total_dur = 0.0, total_gap = 0.0;
        for (int i = 0; i < 22; i++) {
            cl_ulong s, e;
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL);
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &e, NULL);
            double dur_us = (double)(e - s) * 1e-3;
            total_dur += dur_us;

            double gap_us = 0.0;
            if (i < 21) {
                cl_ulong next_s;
                clGetEventProfilingInfo(ev[i + 1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &next_s, NULL);
                gap_us = (double)(next_s - e) * 1e-3;
                total_gap += gap_us;
            }

            char name[64];
            int layer = i / 11;
            int stage = i % 11;
            snprintf(name, sizeof(name), "L%d.%s", layer, kernel_names[stage]);

            if (i < 21) {
                printf("%2d  | %-24s | %8.2f µs | %8.2f µs %s\n",
                       i, name, dur_us, gap_us, (i == 10) ? "<-- L0 -> L1 TRANSITION" : "");
            } else {
                printf("%2d  | %-24s | %8.2f µs | %8s\n", i, name, dur_us, "-");
            }
            clReleaseEvent(ev[i]);
        }
        printf("------------------------------------------------------------------------\n");
        printf("Total Kernels Duration: %8.2f ms\n", total_dur / 1000.0);
        printf("Total Inter-Kernel Gaps: %8.2f ms\n", total_gap / 1000.0);
        printf("Total Span on Silicon:   %8.2f ms\n", (total_dur + total_gap) / 1000.0);

        clReleaseMemObject(d_k_sub);
        clReleaseMemObject(d_v_sub);
    }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
