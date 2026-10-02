// OpenCL 1.2 / 3.0 SwiGLU FFN Pipeline Kernels for Kepler GT 750M
// Supports both Modular (Separate 6-stage) and Fused (3-stage) Execution

typedef struct {
    ushort d;      // FP16 scale delta
    uchar qs[16];  // 32 4-bit nibbles
} block_q4_0;

#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define WG_THREADS (WARP_SIZE * WARPS_PER_WG)

// =========================================================================
// 1. RMSNorm Kernel (D elements, 1 work-group of 128 threads)
// =========================================================================
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

    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += n_threads) {
        float val = x[i];
        sum_sq += val * val;
    }
    sh_sq[tid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);

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

    float scale = l_scale;
    for (int i = tid; i < D; i += n_threads) {
        z[i] = x[i] * scale * gamma[i];
    }
}

// =========================================================================
// 2. Standard Dual-Block Warp GEMV Q4_0 Kernel
// =========================================================================
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

    int blk_half = lane_id >> 4;
    int byte_idx = lane_id & 15;

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

// =========================================================================
// 3. Numerically Stable SwiGLU Kernel: h[i] = SiLU(g[i]) * u[i]
// =========================================================================
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

// =========================================================================
// 4. Residual Addition Kernel: y[i] = x[i] + ydown[i]
// =========================================================================
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

// =========================================================================
// 5. Fused Optimization 1: Down GEMV + In-Place Residual Add
// Writes directly: y[row] = x[row] + Down_GEMV(h)
// Eliminates ydown buffer and residual kernel launch!
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(WG_THREADS, 1, 1)))
void gemv_q4_0_down_residual(
    __global const block_q4_0 *W_down,
    __global const float      *h,
    __global const float      *x,
    __global float            *y,
    const int D,
    const int M
) {
    int local_id = get_local_id(0);
    int warp_id  = local_id / WARP_SIZE;
    int lane_id  = local_id % WARP_SIZE;

    int row = (get_group_id(0) * WARPS_PER_WG) + warp_id;
    bool is_valid_row = (row < D);

    int nb = M / 32;
    __global const block_q4_0 *row_blocks = is_valid_row ? (W_down + row * nb) : NULL;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    __local float sh_mem[WG_THREADS];

    int blk_half = lane_id >> 4;
    int byte_idx = lane_id & 15;

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
            acc0 += d * ((float)x0 * h[k_base + byte_idx]);
            acc1 += d * ((float)x1 * h[k_base + byte_idx + 16]);
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
        y[row] = (sh_mem[local_id] + sh_mem[local_id + 1]) + x[row];
    }
}

// =========================================================================
// 6. Fused Optimization 2: Gate + Up GEMV + SwiGLU Fused Kernel
// Simultaneously projects Gate and Up for row 'row', applies SiLU, and writes h[row]
// Uses 4 accumulators (gate_a, gate_b, up_a, up_b) for dual dual-issue Kepler ILP
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(WG_THREADS, 1, 1)))
void gemv_swiglu_fused(
    __global const block_q4_0 *W_gate,
    __global const block_q4_0 *W_up,
    __global const float      *z,
    __global float            *h,
    const int M,
    const int K
) {
    int local_id = get_local_id(0);
    int warp_id  = local_id / WARP_SIZE;
    int lane_id  = local_id % WARP_SIZE;

    int row = (get_group_id(0) * WARPS_PER_WG) + warp_id;
    bool is_valid_row = (row < M);

    int nb = K / 32;
    __global const block_q4_0 *gate_blocks = is_valid_row ? (W_gate + row * nb) : NULL;
    __global const block_q4_0 *up_blocks   = is_valid_row ? (W_up   + row * nb) : NULL;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    __local float sh_gate[WG_THREADS];
    __local float sh_up[WG_THREADS];

    int blk_half = lane_id >> 4;
    int byte_idx = lane_id & 15;

    if (is_valid_row) {
        int n_pairs = nb / 2;
        for (int i = 0; i < n_pairs; i++) {
            int b = 2 * i + blk_half;
            int k_base = b * 32;
            float z0 = z[k_base + byte_idx];
            float z1 = z[k_base + byte_idx + 16];

            // Gate block
            __global const block_q4_0 *blk_g = &gate_blocks[b];
            float dg = vload_half(0, (const __global half *)&blk_g->d);
            uchar qg = blk_g->qs[byte_idx];
            int g0 = (qg & 0x0F) - 8;
            int g1 = (qg >>   4) - 8;
            acc_gate += dg * ((float)g0 * z0 + (float)g1 * z1);

            // Up block
            __global const block_q4_0 *blk_u = &up_blocks[b];
            float du = vload_half(0, (const __global half *)&blk_u->d);
            uchar qu = blk_u->qs[byte_idx];
            int u0 = (qu & 0x0F) - 8;
            int u1 = (qu >>   4) - 8;
            acc_up += du * ((float)u0 * z0 + (float)u1 * z1);
        }
    }

    sh_gate[local_id] = acc_gate;
    sh_up[local_id]   = acc_up;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 16) {
        sh_gate[local_id] += sh_gate[local_id + 16];
        sh_up[local_id]   += sh_up[local_id + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 8) {
        sh_gate[local_id] += sh_gate[local_id + 8];
        sh_up[local_id]   += sh_up[local_id + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 4) {
        sh_gate[local_id] += sh_gate[local_id + 4];
        sh_up[local_id]   += sh_up[local_id + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 2) {
        sh_gate[local_id] += sh_gate[local_id + 2];
        sh_up[local_id]   += sh_up[local_id + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id == 0 && is_valid_row) {
        float g_val = sh_gate[local_id] + sh_gate[local_id + 1];
        float u_val = sh_up[local_id]   + sh_up[local_id + 1];

        float sig;
        if (g_val >= 0.0f) {
            sig = 1.0f / (1.0f + exp(-g_val));
        } else {
            float eg = exp(g_val);
            sig = eg / (1.0f + eg);
        }
        h[row] = g_val * sig * u_val;
    }
}
