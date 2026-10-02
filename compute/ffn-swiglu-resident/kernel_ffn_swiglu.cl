// OpenCL 1.2 / 3.0 SwiGLU FFN Pipeline Kernels for Kepler GT 750M
// Implements: RMSNorm -> Gate/Up Dual GEMV -> Numerically Stable SiLU -> Down GEMV -> Residual Add

typedef struct {
    ushort d;      // FP16 scale delta
    uchar qs[16];  // 32 4-bit nibbles
} block_q4_0;

#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define WG_THREADS (WARP_SIZE * WARPS_PER_WG)

// 1. RMSNorm Kernel (D elements, 1 work-group of 128 threads)
__kernel void kernel_rmsnorm(
    __global const float *x,
    __global const float *gamma,
    __global float       *z,
    const int D,
    const float eps
) {
    int tid = get_local_id(0);
    int n_threads = get_local_size(0);

    __local float sh_sq[128];
    __local float l_scale;

    // Accumulate sum of squares per thread
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += n_threads) {
        float val = x[i];
        sum_sq += val * val;
    }
    sh_sq[tid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory (128 -> 1)
    if (tid < 64) { sh_sq[tid] += sh_sq[tid + 64]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 32) { sh_sq[tid] += sh_sq[tid + 32]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16) { sh_sq[tid] += sh_sq[tid + 16]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)  { sh_sq[tid] += sh_sq[tid + 8]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)  { sh_sq[tid] += sh_sq[tid + 4]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)  { sh_sq[tid] += sh_sq[tid + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        float mean = (sh_sq[0] + sh_sq[1]) / (float)D;
        l_scale = rsqrt(mean + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Scale each element and apply gamma
    float scale = l_scale;
    for (int i = tid; i < D; i += n_threads) {
        z[i] = x[i] * scale * gamma[i];
    }
}

// 2. Dual-Block Warp GEMV Q4_0 Kernel
// Used for Gate (8960x1536), Up (8960x1536), and Down (1536x8960)
__kernel void gemv_q4_0_dual_block(
    __global const block_q4_0 *W,
    __global const float      *x,
    __global float            *y,
    const int M,
    const int K
) {
    int local_id = get_local_id(0);
    int warp_id  = local_id / WARP_SIZE;
    int lane_id  = local_id % WARP_SIZE;

    int row = (get_group_id(0) * WARPS_PER_WG) + warp_id;
    bool is_valid_row = (row < M);

    int nb = K / 32;
    __global const block_q4_0 *row_blocks = is_valid_row ? (W + row * nb) : NULL;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    __local float sh_mem[WG_THREADS];

    int blk_half = lane_id >> 4; // 0 for lanes 0..15, 1 for lanes 16..31
    int byte_idx = lane_id & 15; // 0..15 for all lanes

    if (is_valid_row) {
        int n_pairs = nb / 2;
        for (int i = 0; i < n_pairs; i++) {
            int b = 2 * i + blk_half;
            __global const block_q4_0 *blk = &row_blocks[b];
            float d = vload_half(0, (const __global half *)&blk->d);

            uchar q = blk->qs[byte_idx];
            int x0 = (q & 0x0F) - 8;
            int x1 = (q >>   4) - 8;

            int k_base = b * 32;
            acc0 += d * ((float)x0 * x[k_base + byte_idx]);
            acc1 += d * ((float)x1 * x[k_base + byte_idx + 16]);
        }
    }

    sh_mem[local_id] = acc0 + acc1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 16) sh_mem[local_id] += sh_mem[local_id + 16];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 8)  sh_mem[local_id] += sh_mem[local_id + 8];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 4)  sh_mem[local_id] += sh_mem[local_id + 4];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 2)  sh_mem[local_id] += sh_mem[local_id + 2];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id == 0 && is_valid_row) {
        y[row] = sh_mem[local_id] + sh_mem[local_id + 1];
    }
}

// 3. Numerically Stable SwiGLU Kernel: h[i] = SiLU(g[i]) * u[i]
__kernel void kernel_swiglu(
    __global const float *g,
    __global const float *u,
    __global float       *h,
    const int M
) {
    int i = get_global_id(0);
    if (i >= M) return;

    float val_g = g[i];
    float sig;
    if (val_g >= 0.0f) {
        sig = 1.0f / (1.0f + exp(-val_g));
    } else {
        float eg = exp(val_g);
        sig = eg / (1.0f + eg);
    }
    h[i] = val_g * sig * u[i];
}

// 4. Residual Addition Kernel: y[i] = x[i] + ydown[i]
__kernel void kernel_residual_add(
    __global const float *x,
    __global const float *ydown,
    __global float       *y,
    const int D
) {
    int i = get_global_id(0);
    if (i >= D) return;

    y[i] = x[i] + ydown[i];
}
