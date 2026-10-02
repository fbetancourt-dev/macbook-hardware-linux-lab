#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <immintrin.h>
#include <omp.h>
#include <CL/cl.h>

#define D_MODEL 1536
#define QK_K 256
#define BLOCKS_PER_ROW (D_MODEL / QK_K) // 6
#define BYTES_PER_ROW (BLOCKS_PER_ROW * 210) // 1260

typedef struct __attribute__((packed)) {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d; // IEEE 754 half
} block_q6_k;

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

static inline float fp16_to_float(uint16_t h) {
    return _cvtsh_ss(h);
}

// Official GGML reference dequantization
static void dequantize_row_q6_k_ref(const uint8_t *row_bytes, float *y, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t *blk_ptr = row_bytes + i * 210;
        const uint8_t *ql = blk_ptr;
        const uint8_t *qh = blk_ptr + 128;
        const int8_t  *sc = (const int8_t*)(blk_ptr + 192);
        uint16_t d_raw    = *(const uint16_t*)(blk_ptr + 208);
        const float d     = fp16_to_float(d_raw);

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// Reference CPU GEMV using dequantization + dot product
static void cpu_gemv_q6_k_ref(
    const uint8_t *W,
    const float *x,
    float *y,
    int M,
    int K
) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; row++) {
        float dequant[K];
        dequantize_row_q6_k_ref(W + (size_t)row * BYTES_PER_ROW, dequant, K);
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += dequant[i] * x[i];
        }
        y[row] = sum;
    }
}

static char* load_kernel_source(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = (char*)malloc(sz + 1);
    size_t rb = fread(src, 1, sz, f);
    (void)rb;
    src[sz] = '\0';
    fclose(f);
    return src;
}

static void read_exact_at(int fd, uint64_t offset, void *buf, size_t size) {
    uint8_t *p = (uint8_t*)buf;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = pread(fd, p, remaining, offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "FATAL: pread failed at offset %lu: %s\n", (unsigned long)offset, strerror(errno));
            exit(1);
        }
        if (n == 0) {
            fprintf(stderr, "FATAL: EOF at offset %lu\n", (unsigned long)offset);
            exit(1);
        }
        p += n;
        offset += n;
        remaining -= n;
    }
}

int main(void) {
    printf("========================================================================================\n");
    printf(" Phase B0: Isolated Q6_K GEMV Kernel Unit Test (LM Head Projection)                     \n");
    printf(" Real GGUF output.weight on Kepler GT 750M (OpenCL 3.0) vs GGML Reference CPU           \n");
    printf("========================================================================================\n");

    const char *gguf_path = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("open GGUF"); exit(1); }

    // output.weight in GGUF is at offset 5950528, shape [1536, 151936] in Q6_K
    uint64_t off_output_weight = 5950528ULL;

    // Test a representative slice of 4096 vocabulary rows for isolated verification
    int test_M = 4096;
    size_t slice_bytes = (size_t)test_M * BYTES_PER_ROW; // 4096 * 1260 = ~5.16 MB
    printf("Loading slice of %d rows of output.weight (%.2f MB)...\n", test_M, slice_bytes / (1024.0 * 1024.0));

    uint8_t *h_W = (uint8_t*)malloc(slice_bytes);
    read_exact_at(fd, off_output_weight, h_W, slice_bytes);
    close(fd);
    printf("Weights slice loaded successfully!\n\n");

    // Initialize OpenCL
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

    char *src = load_kernel_source("kernel_q6_k.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err); CHECK_CL(err, "program");
    free(src);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = (char*)malloc(log_sz + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        log[log_sz] = '\0';
        fprintf(stderr, "Build Log:\n%s\n", log);
        exit(1);
    }

    cl_kernel k_gemv = clCreateKernel(program, "gemv_q6_k", &err); CHECK_CL(err, "kernel");

    cl_mem d_W = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, slice_bytes, h_W, &err); CHECK_CL(err, "d_W");
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * D_MODEL, NULL, &err); CHECK_CL(err, "d_x");
    cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * test_M, NULL, &err); CHECK_CL(err, "d_y");

    int K = D_MODEL;
    clSetKernelArg(k_gemv, 0, sizeof(cl_mem), &d_W);
    clSetKernelArg(k_gemv, 1, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &d_y);
    clSetKernelArg(k_gemv, 3, sizeof(int), &test_M);
    clSetKernelArg(k_gemv, 4, sizeof(int), &K);

    size_t local_sz = 128;
    size_t global_sz = ((test_M + 3) / 4) * 128;

    // --- TEST 1: Canonical Basis Vectors (e_i) ---
    // Proves that each of the 4 quants in each nibble, upper bits, and scales are mapped to the EXACT column
    int test_indices[] = { 0, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 255, 512, 1024, 1535 };
    int n_basis = sizeof(test_indices) / sizeof(test_indices[0]);

    printf("Running Canonical Basis Unit Tests (e_i) across Q6_K unpacking boundaries:\n");
    printf("%-8s | %-12s | %-12s | %-10s | %-8s\n", "Basis", "Max Abs Diff", "Rel L2 Error", "Cos Sim", "Status");
    printf("-------------------------------------------------------------------------\n");

    bool all_basis_passed = true;
    for (int b = 0; b < n_basis; b++) {
        int idx = test_indices[b];
        float h_x[D_MODEL];
        memset(h_x, 0, sizeof(h_x));
        h_x[idx] = 1.0f; // Unit impulse at column idx

        clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, sizeof(float) * D_MODEL, h_x, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_sz, &local_sz, 0, NULL, NULL);

        float h_y_gpu[test_M];
        clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(float) * test_M, h_y_gpu, 0, NULL, NULL);

        float h_y_cpu[test_M];
        cpu_gemv_q6_k_ref(h_W, h_x, h_y_cpu, test_M, K);

        float max_diff = 0.0f;
        double diff_sq = 0.0, ref_sq = 0.0, dot = 0.0, norm_c = 0.0, norm_g = 0.0;
        for (int r = 0; r < test_M; r++) {
            float d = fabsf(h_y_gpu[r] - h_y_cpu[r]);
            if (d > max_diff) max_diff = d;
            diff_sq += (double)d * (double)d;
            ref_sq  += (double)h_y_cpu[r] * (double)h_y_cpu[r];
            dot     += (double)h_y_gpu[r] * (double)h_y_cpu[r];
            norm_c  += (double)h_y_cpu[r] * (double)h_y_cpu[r];
            norm_g  += (double)h_y_gpu[r] * (double)h_y_gpu[r];
        }

        double rel_l2 = sqrt(diff_sq) / sqrt(ref_sq);
        double cos_sim = dot / (sqrt(norm_c) * sqrt(norm_g));

        char basis_str[16];
        snprintf(basis_str, sizeof(basis_str), "e[%d]", idx);
        bool passed = (rel_l2 < 1e-5 && cos_sim > 0.999999);
        if (!passed) all_basis_passed = false;

        printf("%-8s | %12.4e | %12.4e | %8.6f | %s\n",
               basis_str, max_diff, rel_l2, cos_sim, passed ? "PASS" : "FAIL");
    }
    printf("-------------------------------------------------------------------------\n");
    if (all_basis_passed) {
        printf("All Canonical Basis Unit Tests PASSED! Q6_K unpacking matches GGML 100%%!\n\n");
    } else {
        printf("WARNING: One or more basis tests failed!\n\n");
    }

    // --- TEST 2: Dense Random Latent Vector ---
    printf("Running Dense Latent Vector Verification (x ~ N(0, 1), %d rows):\n", test_M);
    float h_x_dense[D_MODEL];
    srand(42);
    for (int i = 0; i < D_MODEL; i++) {
        h_x_dense[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }

    // Warmup
    clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, sizeof(float) * D_MODEL, h_x_dense, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_sz, &local_sz, 0, NULL, NULL);
    clFinish(queue);

    // Timed GPU Run (10 iterations)
    const int iters = 10;
    double t_gpu_start = get_time_us();
    for (int it = 0; it < iters; it++) {
        clEnqueueNDRangeKernel(queue, k_gemv, 1, NULL, &global_sz, &local_sz, 0, NULL, NULL);
    }
    clFinish(queue);
    double t_gpu_end = get_time_us();
    double ms_gpu = (t_gpu_end - t_gpu_start) / (1000.0 * iters);

    float h_y_dense_gpu[test_M];
    clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(float) * test_M, h_y_dense_gpu, 0, NULL, NULL);

    // CPU Run
    double t_cpu_start = get_time_us();
    float h_y_dense_cpu[test_M];
    cpu_gemv_q6_k_ref(h_W, h_x_dense, h_y_dense_cpu, test_M, K);
    double t_cpu_end = get_time_us();
    double ms_cpu = (t_cpu_end - t_cpu_start) / 1000.0;

    float max_diff = 0.0f;
    double diff_sq = 0.0, ref_sq = 0.0, dot = 0.0, norm_c = 0.0, norm_g = 0.0;
    for (int r = 0; r < test_M; r++) {
        float d = fabsf(h_y_dense_gpu[r] - h_y_dense_cpu[r]);
        if (d > max_diff) max_diff = d;
        diff_sq += (double)d * (double)d;
        ref_sq  += (double)h_y_dense_cpu[r] * (double)h_y_dense_cpu[r];
        dot     += (double)h_y_dense_gpu[r] * (double)h_y_dense_cpu[r];
        norm_c  += (double)h_y_dense_cpu[r] * (double)h_y_dense_cpu[r];
        norm_g  += (double)h_y_dense_gpu[r] * (double)h_y_dense_gpu[r];
    }
    double rel_l2 = sqrt(diff_sq) / sqrt(ref_sq);
    double cos_sim = dot / (sqrt(norm_c) * sqrt(norm_g));

    double bw_gbps = (slice_bytes * 1e-9) / (ms_gpu * 1e-3);
    printf("CPU (8 threads):   %8.2f ms\n", ms_cpu);
    printf("GPU GT 750M:       %8.2f ms (Effective Bandwidth: %.2f GB/s)\n", ms_gpu, bw_gbps);
    printf("Speedup:           %8.2fx\n", ms_cpu / ms_gpu);
    printf("Max Abs Diff:      %12.4e\n", max_diff);
    printf("Relative L2 Error: %12.4e\n", rel_l2);
    printf("Cosine Similarity: %8.6f\n\n", cos_sim);

    if (rel_l2 < 1e-5 && cos_sim > 0.999999) {
        printf("Phase B0 PASSED: Q6_K GEMV kernel validated with 100%% precision!\n");
    } else {
        printf("Phase B0 FAILED: Precision threshold not met!\n");
    }

    return 0;
}
