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

#define NUM_LAYERS 2
#define QK4_0 32
#define D_MODEL 1536
#define HEAD_DIM 128
#define N_HEADS_Q 12
#define N_HEADS_KV 2
#define GQA_GROUP_SIZE 6
#define D_QKV (D_MODEL + N_HEADS_KV * HEAD_DIM + N_HEADS_KV * HEAD_DIM) // 2048
#define D_FFN 8960
#define T_MAX 4096
#define ROPE_BASE 1000000.0f
#define EPSILON 1e-6f
#define PV_SEGMENT_SIZE 256
#define MAX_SEGMENTS (T_MAX / PV_SEGMENT_SIZE) // 16

typedef struct {
    uint16_t d;       // IEEE 754 half
    uint8_t qs[16];   // 32 4-bit nibbles
} block_q4_0;

#define CHECK_CL(err, msg) do { \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FATAL OpenCL Error at %s:%d: %s (code %d)\n", __FILE__, __LINE__, msg, err); \
        exit(1); \
    } \
} while (0)

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

static inline float stable_silu(float x) {
    if (x >= 0.0f) {
        return x / (1.0f + expf(-x));
    } else {
        float ex = expf(x);
        return x * ex / (1.0f + ex);
    }
}

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

void cpu_gemv_q4_0_bias(
    const block_q4_0 *W,
    const float      *x,
    const float      *bias,
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
        y[r] = row_sum + (bias ? bias[r] : 0.0f);
    }
}

typedef struct {
    float *gamma_attn;
    block_q4_0 *W_qkv;
    float *b_qkv;
    block_q4_0 *W_o;
    float *gamma_ffn;
    block_q4_0 *W_gate;
    block_q4_0 *W_up;
    block_q4_0 *W_down;
} LayerWeightsHost;

typedef struct {
    cl_mem gamma_attn;
    cl_mem W_qkv;
    cl_mem b_qkv;
    cl_mem W_o;
    cl_mem gamma_ffn;
    cl_mem W_gate;
    cl_mem W_up;
    cl_mem W_down;
} LayerWeightsDevice;

typedef struct {
    float *k_cache;
    float *v_cache;
} LayerKVHost;

typedef struct {
    cl_mem k_cache;
    cl_mem v_cache;
} LayerKVDevice;

typedef struct {
    cl_mem state;
    cl_mem norm;
    cl_mem qkv;
    cl_mem scores;
    cl_mem partial;
    cl_mem h;
} DecoderWorkspaceDevice;

typedef struct {
    cl_kernel k_rmsnorm_attn;
    cl_kernel k_qkv_gemv;
    cl_kernel k_rope_kv;
    cl_kernel k_scores;
    cl_kernel k_softmax;
    cl_kernel k_pv_combine;
    cl_kernel k_pv_reduce;
    cl_kernel k_wo_residual;
    cl_kernel k_rmsnorm_ffn;
    cl_kernel k_swiglu_fused;
    cl_kernel k_down_res;
} LayerKernels;

void cpu_decoder_layer_step(
    const float *x_in,
    const LayerWeightsHost *w,
    LayerKVHost *kv,
    int pos,
    int seq_len,
    float *y_out
) {
    // 1. RMSNorm (attn)
    float sum_sq1 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq1 += x_in[i] * x_in[i];
    }
    float scale1 = 1.0f / sqrtf((sum_sq1 / (float)D_MODEL) + EPSILON);
    float z1[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z1[i] = x_in[i] * scale1 * w->gamma_attn[i];
    }

    // 2. QKV Projection with Bias
    float qkv[D_QKV];
    cpu_gemv_q4_0_bias(w->W_qkv, z1, w->b_qkv, qkv, D_QKV, D_MODEL);

    float *q = qkv;
    float *k = qkv + 1536;
    float *v = qkv + 1536 + 256;

    // 3. RoPE on Q & K, append to KV Cache
    for (int tid = 0; tid < 64; tid++) {
        float theta = (float)pos * powf(ROPE_BASE, -2.0f * (float)tid / (float)HEAD_DIM);
        float cos_th = cosf(theta);
        float sin_th = sinf(theta);

        for (int h = 0; h < N_HEADS_Q; h++) {
            int base = h * HEAD_DIM;
            float q0 = q[base + tid];
            float q1 = q[base + tid + 64];
            q[base + tid]      = q0 * cos_th - q1 * sin_th;
            q[base + tid + 64] = q0 * sin_th + q1 * cos_th;
        }

        for (int h = 0; h < N_HEADS_KV; h++) {
            int in_base = h * HEAD_DIM;
            float k0 = k[in_base + tid];
            float k1 = k[in_base + tid + 64];

            float k_rot0 = k0 * cos_th - k1 * sin_th;
            float k_rot1 = k0 * sin_th + k1 * cos_th;

            int out_base = (h * T_MAX + pos) * HEAD_DIM;
            kv->k_cache[out_base + tid]      = k_rot0;
            kv->k_cache[out_base + tid + 64] = k_rot1;

            kv->v_cache[out_base + tid]      = v[in_base + tid];
            kv->v_cache[out_base + tid + 64] = v[in_base + tid + 64];
        }
    }

    // 4 & 5 & 6. GQA Scores -> Softmax -> Value Combination
    float attn_out[D_MODEL];
    float scale_factor = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h_q = 0; h_q < N_HEADS_Q; h_q++) {
        int h_kv = h_q / GQA_GROUP_SIZE;
        float scores[seq_len];
        float max_s = -1e30f;

        for (int t = 0; t < seq_len; t++) {
            float dot = 0.0f;
            int q_off = h_q * HEAD_DIM;
            int k_off = (h_kv * T_MAX + t) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q[q_off + d] * kv->k_cache[k_off + d];
            }
            scores[t] = dot * scale_factor;
            if (scores[t] > max_s) max_s = scores[t];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }

        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                int v_off = (h_kv * T_MAX + t) * HEAD_DIM + d;
                acc += scores[t] * kv->v_cache[v_off];
            }
            attn_out[h_q * HEAD_DIM + d] = acc;
        }
    }

    // 7. Output Projection Wo + Residual 1: r = x_in + Wo(attn_out)
    float wo_out[D_MODEL];
    cpu_gemv_q4_0_bias(w->W_o, attn_out, NULL, wo_out, D_MODEL, D_MODEL);
    float r[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        r[i] = x_in[i] + wo_out[i];
    }

    // 8. RMSNorm (ffn) on residual r
    float sum_sq2 = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        sum_sq2 += r[i] * r[i];
    }
    float scale2 = 1.0f / sqrtf((sum_sq2 / (float)D_MODEL) + EPSILON);
    float z2[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) {
        z2[i] = r[i] * scale2 * w->gamma_ffn[i];
    }

    // 9. Gate & Up GEMV + SiLU
    float g[D_FFN];
    float u[D_FFN];
    cpu_gemv_q4_0_bias(w->W_gate, z2, NULL, g, D_FFN, D_MODEL);
    cpu_gemv_q4_0_bias(w->W_up,   z2, NULL, u, D_FFN, D_MODEL);

    float h[D_FFN];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D_FFN; i++) {
        h[i] = stable_silu(g[i]) * u[i];
    }

    // 10. Down GEMV + Residual 2: y = r + Down(h)
    float down_out[D_MODEL];
    cpu_gemv_q4_0_bias(w->W_down, h, NULL, down_out, D_MODEL, D_FFN);
    for (int i = 0; i < D_MODEL; i++) {
        y_out[i] = r[i] + down_out[i];
    }
}

static char* load_kernel_source(const char* filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filepath);
        return NULL;
    }
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
    printf(" Complete Two-Layer Resident Transformer Decoder on Kepler GT 750M (Hardware Profiling) \n");
    printf(" Multi-Layer Resident Pipeline (Qwen2.5-Coder-1.5B Specs, D=1536, FFN=8960, GQA 12:2)   \n");
    printf("========================================================================================\n");

    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return 1;
    }
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

    if (!device) {
        fprintf(stderr, "No GPU OpenCL device found.\n");
        return 1;
    }

    char dname[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    printf("Target Device: %s\n", dname);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    char *src = load_kernel_source("kernel_decoder_layer.cl");
    if (!src) return 1;

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    free(src);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char *log = (char*)malloc(log_len + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = '\0';
        fprintf(stderr, "OpenCL Build Error:\n%s\n", log);
        free(log);
        return 1;
    }

    LayerKernels kernels[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; l++) {
        kernels[l].k_rmsnorm_attn = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "kernel_rmsnorm");
        kernels[l].k_qkv_gemv     = clCreateKernel(program, "gemv_q4_0_bias", &err); CHECK_CL(err, "gemv_q4_0_bias");
        kernels[l].k_rope_kv      = clCreateKernel(program, "kernel_rope_and_kv_append", &err); CHECK_CL(err, "kernel_rope_and_kv_append");
        kernels[l].k_scores       = clCreateKernel(program, "kernel_gqa_scores", &err); CHECK_CL(err, "kernel_gqa_scores");
        kernels[l].k_softmax      = clCreateKernel(program, "kernel_softmax_gqa", &err); CHECK_CL(err, "kernel_softmax_gqa");
        kernels[l].k_pv_combine   = clCreateKernel(program, "kernel_gqa_value_combine_segmented", &err); CHECK_CL(err, "kernel_gqa_value_combine_segmented");
        kernels[l].k_pv_reduce    = clCreateKernel(program, "kernel_gqa_reduce_segments", &err); CHECK_CL(err, "kernel_gqa_reduce_segments");
        kernels[l].k_wo_residual  = clCreateKernel(program, "gemv_q4_0_wo_residual", &err); CHECK_CL(err, "gemv_q4_0_wo_residual");
        kernels[l].k_rmsnorm_ffn  = clCreateKernel(program, "kernel_rmsnorm", &err); CHECK_CL(err, "kernel_rmsnorm ffn");
        kernels[l].k_swiglu_fused = clCreateKernel(program, "gemv_swiglu_fused", &err); CHECK_CL(err, "gemv_swiglu_fused");
        kernels[l].k_down_res     = clCreateKernel(program, "gemv_q4_0_down_residual", &err); CHECK_CL(err, "gemv_q4_0_down_residual");
    }

    printf("OpenCL Kernels initialized for %d layers.\n\n", NUM_LAYERS);

    // Host memory initialization with distinct random seeds per layer
    srand(42);
    float *h_x = (float*)malloc(sizeof(float) * D_MODEL);
    for (int i = 0; i < D_MODEL; i++) {
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    int nb_qkv  = D_MODEL / QK4_0;
    int nb_wo   = D_MODEL / QK4_0;
    int nb_gate = D_MODEL / QK4_0;
    int nb_down = D_FFN / QK4_0;

    size_t sz_W_qkv  = sizeof(block_q4_0) * D_QKV * nb_qkv;
    size_t sz_W_o    = sizeof(block_q4_0) * D_MODEL * nb_wo;
    size_t sz_W_gate = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_up   = sizeof(block_q4_0) * D_FFN * nb_gate;
    size_t sz_W_down = sizeof(block_q4_0) * D_MODEL * nb_down;
    size_t sz_kv_cache = sizeof(float) * N_HEADS_KV * T_MAX * HEAD_DIM;

    LayerWeightsHost w_host[NUM_LAYERS];
    LayerWeightsDevice w_dev[NUM_LAYERS];
    LayerKVHost kv_host[NUM_LAYERS];
    LayerKVDevice kv_dev[NUM_LAYERS];

    for (int l = 0; l < NUM_LAYERS; l++) {
        w_host[l].gamma_attn = (float*)malloc(sizeof(float) * D_MODEL);
        w_host[l].gamma_ffn  = (float*)malloc(sizeof(float) * D_MODEL);
        w_host[l].b_qkv       = (float*)malloc(sizeof(float) * D_QKV);

        w_host[l].W_qkv  = (block_q4_0*)malloc(sz_W_qkv);
        w_host[l].W_o    = (block_q4_0*)malloc(sz_W_o);
        w_host[l].W_gate = (block_q4_0*)malloc(sz_W_gate);
        w_host[l].W_up   = (block_q4_0*)malloc(sz_W_up);
        w_host[l].W_down = (block_q4_0*)malloc(sz_W_down);

        for (int i = 0; i < D_MODEL; i++) {
            w_host[l].gamma_attn[i] = 1.0f + 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
            w_host[l].gamma_ffn[i]  = 1.0f + 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
        }
        for (int i = 0; i < D_QKV; i++) {
            w_host[l].b_qkv[i] = 0.01f * (((float)rand() / (float)RAND_MAX) - 0.5f);
        }
        for (size_t i = 0; i < (size_t)D_QKV * nb_qkv; i++) {
            w_host[l].W_qkv[i].d = float_to_fp16(0.02f);
            for (int j = 0; j < 16; j++) w_host[l].W_qkv[i].qs[j] = (rand() & 0xFF);
        }
        for (size_t i = 0; i < (size_t)D_MODEL * nb_wo; i++) {
            w_host[l].W_o[i].d = float_to_fp16(0.02f);
            for (int j = 0; j < 16; j++) w_host[l].W_o[i].qs[j] = (rand() & 0xFF);
        }
        for (size_t i = 0; i < (size_t)D_FFN * nb_gate; i++) {
            w_host[l].W_gate[i].d = float_to_fp16(0.02f);
            for (int j = 0; j < 16; j++) w_host[l].W_gate[i].qs[j] = (rand() & 0xFF);
        }
        for (size_t i = 0; i < (size_t)D_FFN * nb_gate; i++) {
            w_host[l].W_up[i].d = float_to_fp16(0.02f);
            for (int j = 0; j < 16; j++) w_host[l].W_up[i].qs[j] = (rand() & 0xFF);
        }
        for (size_t i = 0; i < (size_t)D_MODEL * nb_down; i++) {
            w_host[l].W_down[i].d = float_to_fp16(0.02f);
            for (int j = 0; j < 16; j++) w_host[l].W_down[i].qs[j] = (rand() & 0xFF);
        }

        kv_host[l].k_cache = (float*)malloc(sz_kv_cache);
        kv_host[l].v_cache = (float*)malloc(sz_kv_cache);
        for (size_t i = 0; i < (size_t)N_HEADS_KV * T_MAX * HEAD_DIM; i++) {
            kv_host[l].k_cache[i] = 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
            kv_host[l].v_cache[i] = 0.05f * (((float)rand() / (float)RAND_MAX) - 0.5f);
        }

        // Create Device Buffers for Layer l
        w_dev[l].W_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_qkv, w_host[l].W_qkv, &err); CHECK_CL(err, "W_qkv");
        w_dev[l].b_qkv      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_QKV, w_host[l].b_qkv, &err); CHECK_CL(err, "b_qkv");
        w_dev[l].W_o        = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_o, w_host[l].W_o, &err); CHECK_CL(err, "W_o");
        w_dev[l].W_gate     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_gate, w_host[l].W_gate, &err); CHECK_CL(err, "W_gate");
        w_dev[l].W_up       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_up, w_host[l].W_up, &err); CHECK_CL(err, "W_up");
        w_dev[l].W_down     = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_W_down, w_host[l].W_down, &err); CHECK_CL(err, "W_down");
        w_dev[l].gamma_attn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, w_host[l].gamma_attn, &err); CHECK_CL(err, "gamma_attn");
        w_dev[l].gamma_ffn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * D_MODEL, w_host[l].gamma_ffn, &err); CHECK_CL(err, "gamma_ffn");

        kv_dev[l].k_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz_kv_cache, kv_host[l].k_cache, &err); CHECK_CL(err, "k_cache");
        kv_dev[l].v_cache   = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz_kv_cache, kv_host[l].v_cache, &err); CHECK_CL(err, "v_cache");
    }

    // Shared Single Activation Workspace across all layers (0.33 MB total)
    DecoderWorkspaceDevice ws;
    ws.state    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err); CHECK_CL(err, "state");
    ws.norm     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_MODEL, NULL, &err); CHECK_CL(err, "norm");
    ws.qkv      = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_QKV, NULL, &err); CHECK_CL(err, "qkv");
    ws.scores   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N_HEADS_Q * T_MAX, NULL, &err); CHECK_CL(err, "scores");
    ws.partial  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N_HEADS_Q * MAX_SEGMENTS * HEAD_DIM, NULL, &err); CHECK_CL(err, "partial");
    ws.h        = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * D_FFN, NULL, &err); CHECK_CL(err, "h");

    double total_weights_mb = (double)(NUM_LAYERS * (sz_W_qkv + sz_W_o + sz_W_gate + sz_W_up + sz_W_down)) / (1024.0 * 1024.0);
    double total_kv_mb = (double)(NUM_LAYERS * 2 * sz_kv_cache) / (1024.0 * 1024.0);
    printf("Total 2-Layer Weights in VRAM: %.2f MB\n", total_weights_mb);
    printf("Total 2-Layer KV Cache (FP32): %.2f MB\n", total_kv_mb);
    printf("Shared Resident Workspace:     0.33 MB (reused across both layers!)\n");
    printf("Total VRAM footprint:          %.2f MB\n\n", total_weights_mb + total_kv_mb + 0.33);

    float *h_y_cpu_0 = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_y_cpu_1 = (float*)malloc(sizeof(float) * D_MODEL);
    float *h_y_gpu   = (float*)malloc(sizeof(float) * D_MODEL);

    int context_lens[] = {1, 32, 128, 512, 1024, 2048};
    int num_contexts = sizeof(context_lens) / sizeof(context_lens[0]);

    printf("%-8s | %-10s | %-12s | %-11s | %-11s | %-9s | %-12s | %-12s | %-10s\n",
           "Tokens", "CPU (8T)", "GPU Silicon", "Kernel Sum", "Full E2E", "Speedup", "Max Abs Diff", "Rel L2 Error", "Cos Sim");
    printf("----------------------------------------------------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_contexts; c++) {
        int seq_len = context_lens[c];
        int pos = seq_len - 1;

        // Reset KV Caches for both layers
        for (int l = 0; l < NUM_LAYERS; l++) {
            CHECK_CL(clEnqueueWriteBuffer(queue, kv_dev[l].k_cache, CL_TRUE, 0, sz_kv_cache, kv_host[l].k_cache, 0, NULL, NULL), "reset k_cache");
            CHECK_CL(clEnqueueWriteBuffer(queue, kv_dev[l].v_cache, CL_TRUE, 0, sz_kv_cache, kv_host[l].v_cache, 0, NULL, NULL), "reset v_cache");
        }
        CHECK_CL(clEnqueueWriteBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_x, 0, NULL, NULL), "reset state");

        int d_model = D_MODEL;
        float eps = EPSILON;
        int param_d_qkv = D_QKV;
        int t_max = T_MAX;
        float rope_base = ROPE_BASE;
        float scale_factor = 1.0f / sqrtf((float)HEAD_DIM);
        int num_segments = (seq_len + PV_SEGMENT_SIZE - 1) / PV_SEGMENT_SIZE;
        int d_ffn = D_FFN;

        size_t k_offset_bytes = 1536 * sizeof(float);
        size_t v_offset_bytes = (1536 + 256) * sizeof(float);
        cl_buffer_region reg_k = { k_offset_bytes, 256 * sizeof(float) };
        cl_buffer_region reg_v = { v_offset_bytes, 256 * sizeof(float) };
        cl_mem d_k_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_k, &err); CHECK_CL(err, "d_k_sub");
        cl_mem d_v_sub = clCreateSubBuffer(ws.qkv, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg_v, &err); CHECK_CL(err, "d_v_sub");

        // Dimensions
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

        // Configure kernel arguments for Layer 0 & Layer 1
        for (int l = 0; l < NUM_LAYERS; l++) {
            // Stage 1: RMSNorm (Attn)
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 0, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 1, sizeof(cl_mem), &w_dev[l].gamma_attn);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 2, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 3, sizeof(int), &d_model);
            clSetKernelArg(kernels[l].k_rmsnorm_attn, 4, sizeof(float), &eps);

            // Stage 2: QKV Projection
            clSetKernelArg(kernels[l].k_qkv_gemv, 0, sizeof(cl_mem), &w_dev[l].W_qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 1, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_qkv_gemv, 2, sizeof(cl_mem), &w_dev[l].b_qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 3, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 4, sizeof(int), &param_d_qkv);
            clSetKernelArg(kernels[l].k_qkv_gemv, 5, sizeof(int), &d_model);

            // Stage 3: RoPE & KV Cache
            clSetKernelArg(kernels[l].k_rope_kv, 0, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_rope_kv, 1, sizeof(cl_mem), &d_k_sub);
            clSetKernelArg(kernels[l].k_rope_kv, 2, sizeof(cl_mem), &d_v_sub);
            clSetKernelArg(kernels[l].k_rope_kv, 3, sizeof(cl_mem), &kv_dev[l].k_cache);
            clSetKernelArg(kernels[l].k_rope_kv, 4, sizeof(cl_mem), &kv_dev[l].v_cache);
            clSetKernelArg(kernels[l].k_rope_kv, 5, sizeof(int), &pos);
            clSetKernelArg(kernels[l].k_rope_kv, 6, sizeof(int), &t_max);
            clSetKernelArg(kernels[l].k_rope_kv, 7, sizeof(float), &rope_base);

            // Stage 4: Scores
            clSetKernelArg(kernels[l].k_scores, 0, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_scores, 1, sizeof(cl_mem), &kv_dev[l].k_cache);
            clSetKernelArg(kernels[l].k_scores, 2, sizeof(cl_mem), &ws.scores);
            clSetKernelArg(kernels[l].k_scores, 3, sizeof(int), &seq_len);
            clSetKernelArg(kernels[l].k_scores, 4, sizeof(int), &t_max);
            clSetKernelArg(kernels[l].k_scores, 5, sizeof(float), &scale_factor);

            // Stage 5: Softmax
            clSetKernelArg(kernels[l].k_softmax, 0, sizeof(cl_mem), &ws.scores);
            clSetKernelArg(kernels[l].k_softmax, 1, sizeof(int), &seq_len);
            clSetKernelArg(kernels[l].k_softmax, 2, sizeof(int), &t_max);

            // Stage 6: PV Combine
            clSetKernelArg(kernels[l].k_pv_combine, 0, sizeof(cl_mem), &ws.scores);
            clSetKernelArg(kernels[l].k_pv_combine, 1, sizeof(cl_mem), &kv_dev[l].v_cache);
            clSetKernelArg(kernels[l].k_pv_combine, 2, sizeof(cl_mem), &ws.partial);
            clSetKernelArg(kernels[l].k_pv_combine, 3, sizeof(int), &seq_len);
            clSetKernelArg(kernels[l].k_pv_combine, 4, sizeof(int), &t_max);
            clSetKernelArg(kernels[l].k_pv_combine, 5, sizeof(int), &num_segments);

            // Stage 6b: Reduce Segments into ws.qkv[0:1536]
            clSetKernelArg(kernels[l].k_pv_reduce, 0, sizeof(cl_mem), &ws.partial);
            clSetKernelArg(kernels[l].k_pv_reduce, 1, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_pv_reduce, 2, sizeof(int), &num_segments);

            // Stage 7: Wo GEMV + Residual (ws.state = ws.state + Wo(ws.qkv))
            clSetKernelArg(kernels[l].k_wo_residual, 0, sizeof(cl_mem), &w_dev[l].W_o);
            clSetKernelArg(kernels[l].k_wo_residual, 1, sizeof(cl_mem), &ws.qkv);
            clSetKernelArg(kernels[l].k_wo_residual, 2, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_wo_residual, 3, sizeof(int), &d_model);

            // Stage 8: RMSNorm (FFN)
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 0, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 1, sizeof(cl_mem), &w_dev[l].gamma_ffn);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 2, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 3, sizeof(int), &d_model);
            clSetKernelArg(kernels[l].k_rmsnorm_ffn, 4, sizeof(float), &eps);

            // Stage 9: SwiGLU Fused
            clSetKernelArg(kernels[l].k_swiglu_fused, 0, sizeof(cl_mem), &w_dev[l].W_gate);
            clSetKernelArg(kernels[l].k_swiglu_fused, 1, sizeof(cl_mem), &w_dev[l].W_up);
            clSetKernelArg(kernels[l].k_swiglu_fused, 2, sizeof(cl_mem), &ws.norm);
            clSetKernelArg(kernels[l].k_swiglu_fused, 3, sizeof(cl_mem), &ws.h);
            clSetKernelArg(kernels[l].k_swiglu_fused, 4, sizeof(int), &d_ffn);
            clSetKernelArg(kernels[l].k_swiglu_fused, 5, sizeof(int), &d_model);

            // Stage 10: Down GEMV + Residual (ws.state = ws.state + Down(ws.h))
            clSetKernelArg(kernels[l].k_down_res, 0, sizeof(cl_mem), &w_dev[l].W_down);
            clSetKernelArg(kernels[l].k_down_res, 1, sizeof(cl_mem), &ws.h);
            clSetKernelArg(kernels[l].k_down_res, 2, sizeof(cl_mem), &ws.state);
            clSetKernelArg(kernels[l].k_down_res, 3, sizeof(int), &d_model);
            clSetKernelArg(kernels[l].k_down_res, 4, sizeof(int), &d_ffn);
        }

        // Warmup: Execute Layer 0 followed immediately by Layer 1 completely resident in VRAM
        for (int l = 0; l < NUM_LAYERS; l++) {
            clEnqueueNDRangeKernel(queue, kernels[l].k_rmsnorm_attn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_qkv_gemv, 1, NULL, &g_qkv, &l_qkv, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_rope_kv, 1, NULL, &g_rope, &l_rope, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_scores, 2, NULL, g_scores, l_scores, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_softmax, 1, NULL, &g_soft, &l_soft, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_pv_combine, 2, NULL, g_pv, l_pv, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_pv_reduce, 1, NULL, &g_red, &l_red, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_wo_residual, 1, NULL, &g_wo, &l_wo, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_rmsnorm_ffn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_swiglu_fused, 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, kernels[l].k_down_res, 1, NULL, &g_down, &l_down, 0, NULL, NULL);
        }
        clFinish(queue);

        CHECK_CL(clEnqueueReadBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_y_gpu, 0, NULL, NULL), "read y_gpu");

        // CPU Baseline Execution: Layer 0 -> y0 -> Layer 1 -> y1
        cpu_decoder_layer_step(h_x, &w_host[0], &kv_host[0], pos, seq_len, h_y_cpu_0);
        cpu_decoder_layer_step(h_y_cpu_0, &w_host[1], &kv_host[1], pos, seq_len, h_y_cpu_1);

        // Verification & Numerical Metrics
        float max_diff = 0.0f;
        double diff_sq_sum = 0.0, ref_sq_sum = 0.0;
        double dot_prod = 0.0, norm_cpu = 0.0, norm_gpu = 0.0;
        bool all_finite = true;

        for (int i = 0; i < D_MODEL; i++) {
            if (!isfinite(h_y_gpu[i]) || !isfinite(h_y_cpu_1[i])) all_finite = false;
            double diff = (double)h_y_gpu[i] - (double)h_y_cpu_1[i];
            float abs_diff = fabsf((float)diff);
            if (abs_diff > max_diff) max_diff = abs_diff;
            diff_sq_sum += diff * diff;
            ref_sq_sum  += (double)h_y_cpu_1[i] * (double)h_y_cpu_1[i];
            dot_prod    += (double)h_y_gpu[i] * (double)h_y_cpu_1[i];
            norm_cpu    += (double)h_y_cpu_1[i] * (double)h_y_cpu_1[i];
            norm_gpu    += (double)h_y_gpu[i] * (double)h_y_gpu[i];
        }
        if (!all_finite) {
            fprintf(stderr, "FATAL Error: Outputs contain non-finite numbers (NaN/Inf)!\n");
            exit(1);
        }
        double rel_l2_error = sqrt(diff_sq_sum) / sqrt(ref_sq_sum);
        double cos_sim = dot_prod / (sqrt(norm_cpu) * sqrt(norm_gpu));

        const int iters = 20;
        double total_gpu_span_us = 0.0;
        double total_gpu_sum_us  = 0.0;
        double total_gpu_e2e_us  = 0.0;
        double total_cpu_us      = 0.0;

        cl_event events[22];

        for (int it = 0; it < iters; it++) {
            double t_e2e_0 = get_time_us();
            // Single PCIe upload of input x
            clEnqueueWriteBuffer(queue, ws.state, CL_FALSE, 0, sizeof(float) * D_MODEL, h_x, 0, NULL, NULL);

            // Layer 0 dispatch (11 kernels)
            clEnqueueNDRangeKernel(queue, kernels[0].k_rmsnorm_attn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, &events[0]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_qkv_gemv, 1, NULL, &g_qkv, &l_qkv, 0, NULL, &events[1]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_rope_kv, 1, NULL, &g_rope, &l_rope, 0, NULL, &events[2]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_scores, 2, NULL, g_scores, l_scores, 0, NULL, &events[3]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_softmax, 1, NULL, &g_soft, &l_soft, 0, NULL, &events[4]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_pv_combine, 2, NULL, g_pv, l_pv, 0, NULL, &events[5]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_pv_reduce, 1, NULL, &g_red, &l_red, 0, NULL, &events[6]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_wo_residual, 1, NULL, &g_wo, &l_wo, 0, NULL, &events[7]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_rmsnorm_ffn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, &events[8]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_swiglu_fused, 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, &events[9]);
            clEnqueueNDRangeKernel(queue, kernels[0].k_down_res, 1, NULL, &g_down, &l_down, 0, NULL, &events[10]);

            // Layer 1 dispatch (11 kernels, consuming ws.state in-place!)
            clEnqueueNDRangeKernel(queue, kernels[1].k_rmsnorm_attn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, &events[11]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_qkv_gemv, 1, NULL, &g_qkv, &l_qkv, 0, NULL, &events[12]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_rope_kv, 1, NULL, &g_rope, &l_rope, 0, NULL, &events[13]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_scores, 2, NULL, g_scores, l_scores, 0, NULL, &events[14]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_softmax, 1, NULL, &g_soft, &l_soft, 0, NULL, &events[15]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_pv_combine, 2, NULL, g_pv, l_pv, 0, NULL, &events[16]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_pv_reduce, 1, NULL, &g_red, &l_red, 0, NULL, &events[17]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_wo_residual, 1, NULL, &g_wo, &l_wo, 0, NULL, &events[18]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_rmsnorm_ffn, 1, NULL, &g_rmsnorm, &l_rmsnorm, 0, NULL, &events[19]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_swiglu_fused, 1, NULL, &g_swiglu, &l_swiglu, 0, NULL, &events[20]);
            clEnqueueNDRangeKernel(queue, kernels[1].k_down_res, 1, NULL, &g_down, &l_down, 0, NULL, &events[21]);

            // Single PCIe download of final output y1
            clEnqueueReadBuffer(queue, ws.state, CL_TRUE, 0, sizeof(float) * D_MODEL, h_y_gpu, 0, NULL, NULL);
            double t_e2e_1 = get_time_us();

            // Extract hardware profiling metrics from all 22 kernel events
            cl_ulong t_first_start = 0, t_last_end = 0;
            clGetEventProfilingInfo(events[0],  CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_first_start, NULL);
            clGetEventProfilingInfo(events[21], CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &t_last_end, NULL);
            double span_us = (double)(t_last_end - t_first_start) * 1e-3;

            double sum_kernels_us = 0.0;
            for (int k = 0; k < 22; k++) {
                cl_ulong k_start = 0, k_end = 0;
                clGetEventProfilingInfo(events[k], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &k_start, NULL);
                clGetEventProfilingInfo(events[k], CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &k_end, NULL);
                sum_kernels_us += (double)(k_end - k_start) * 1e-3;
                clReleaseEvent(events[k]);
            }

            total_gpu_span_us += span_us;
            total_gpu_sum_us  += sum_kernels_us;
            total_gpu_e2e_us  += (t_e2e_1 - t_e2e_0);

            // CPU Baseline step
            double t_cpu_0 = get_time_us();
            cpu_decoder_layer_step(h_x, &w_host[0], &kv_host[0], pos, seq_len, h_y_cpu_0);
            cpu_decoder_layer_step(h_y_cpu_0, &w_host[1], &kv_host[1], pos, seq_len, h_y_cpu_1);
            double t_cpu_1 = get_time_us();
            total_cpu_us += (t_cpu_1 - t_cpu_0);
        }

        clReleaseMemObject(d_k_sub);
        clReleaseMemObject(d_v_sub);

        double avg_gpu_span_ms = (total_gpu_span_us / (double)iters) / 1000.0;
        double avg_gpu_sum_ms  = (total_gpu_sum_us  / (double)iters) / 1000.0;
        double avg_gpu_e2e_ms  = (total_gpu_e2e_us  / (double)iters) / 1000.0;
        double avg_cpu_ms      = (total_cpu_us      / (double)iters) / 1000.0;
        double speedup         = avg_cpu_ms / avg_gpu_e2e_ms;

        printf("T=%-6d | %7.2f ms | %8.2f ms  | %8.2f ms | %8.2f ms | %7.2fx | %12.4e | %12.4e | %8.6f\n",
               seq_len, avg_cpu_ms, avg_gpu_span_ms, avg_gpu_sum_ms, avg_gpu_e2e_ms, speedup, max_diff, rel_l2_error, cos_sim);
    }

    printf("----------------------------------------------------------------------------------------------------------------------------\n");
    printf("Two-Layer Resident Transformer evaluation completed successfully.\n");

    // Clean up
    for (int l = 0; l < NUM_LAYERS; l++) {
        clReleaseMemObject(w_dev[l].W_qkv);
        clReleaseMemObject(w_dev[l].b_qkv);
        clReleaseMemObject(w_dev[l].W_o);
        clReleaseMemObject(w_dev[l].W_gate);
        clReleaseMemObject(w_dev[l].W_up);
        clReleaseMemObject(w_dev[l].W_down);
        clReleaseMemObject(w_dev[l].gamma_attn);
        clReleaseMemObject(w_dev[l].gamma_ffn);
        clReleaseMemObject(kv_dev[l].k_cache);
        clReleaseMemObject(kv_dev[l].v_cache);

        clReleaseKernel(kernels[l].k_rmsnorm_attn);
        clReleaseKernel(kernels[l].k_qkv_gemv);
        clReleaseKernel(kernels[l].k_rope_kv);
        clReleaseKernel(kernels[l].k_scores);
        clReleaseKernel(kernels[l].k_softmax);
        clReleaseKernel(kernels[l].k_pv_combine);
        clReleaseKernel(kernels[l].k_pv_reduce);
        clReleaseKernel(kernels[l].k_wo_residual);
        clReleaseKernel(kernels[l].k_rmsnorm_ffn);
        clReleaseKernel(kernels[l].k_swiglu_fused);
        clReleaseKernel(kernels[l].k_down_res);

        free(w_host[l].gamma_attn);
        free(w_host[l].gamma_ffn);
        free(w_host[l].b_qkv);
        free(w_host[l].W_qkv);
        free(w_host[l].W_o);
        free(w_host[l].W_gate);
        free(w_host[l].W_up);
        free(w_host[l].W_down);
        free(kv_host[l].k_cache);
        free(kv_host[l].v_cache);
    }

    clReleaseMemObject(ws.state);
    clReleaseMemObject(ws.norm);
    clReleaseMemObject(ws.qkv);
    clReleaseMemObject(ws.scores);
    clReleaseMemObject(ws.partial);
    clReleaseMemObject(ws.h);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_x);
    free(h_y_cpu_0);
    free(h_y_cpu_1);
    free(h_y_gpu);

    return 0;
}
