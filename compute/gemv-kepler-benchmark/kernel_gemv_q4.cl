// OpenCL 1.2 / 3.0 GEMV Q4_0 Coalesced Kernel for Kepler GT 750M
// High memory coalescence: 32 threads (1 warp) cooperate on each row.
// Robust bounds protection: all threads unconditionally participate in workgroup barriers.

typedef struct {
    ushort d;      // FP16 scale delta
    uchar qs[16];  // 32 4-bit nibbles
} block_q4_0;

#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define WG_THREADS (WARP_SIZE * WARPS_PER_WG)

// Kernel 1: 1 thread per row
__kernel void gemv_q4_0_row(
    __global const block_q4_0 *W,
    __global const float      *x,
    __global float            *y,
    const int M,
    const int K
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int nb = K / 32;
    __global const block_q4_0 *row_blocks = W + row * nb;
    float acc = 0.0f;

    for (int b = 0; b < nb; b++) {
        __global const block_q4_0 *blk = &row_blocks[b];
        float d = vload_half(0, (const __global half *)&blk->d);

        int k_offset = b * 32;
        __global const float *x_blk = x + k_offset;

        float block_sum = 0.0f;
        #pragma unroll 16
        for (int j = 0; j < 16; j++) {
            uchar q = blk->qs[j];
            int x0 = (q & 0x0F) - 8;
            int x1 = (q >>   4) - 8;

            block_sum += (float)x0 * x_blk[j];
            block_sum += (float)x1 * x_blk[j + 16];
        }
        acc += d * block_sum;
    }
    y[row] = acc;
}

// Kernel 2: Warp-Cooperative (32 threads per row, 1 block per iteration)
__kernel void gemv_q4_0_warp(
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

    float warp_acc = 0.0f;
    __local float sh_mem[WG_THREADS];

    if (is_valid_row) {
        for (int b = 0; b < nb; b++) {
            __global const block_q4_0 *blk = &row_blocks[b];
            float d = vload_half(0, (const __global half *)&blk->d);

            int x_val;
            if (lane_id < 16) {
                uchar q = blk->qs[lane_id];
                x_val = (q & 0x0F) - 8;
            } else {
                uchar q = blk->qs[lane_id - 16];
                x_val = (q >> 4) - 8;
            }

            float term = (float)x_val * x[b * 32 + lane_id];
            warp_acc += d * term;
        }
    }

    sh_mem[local_id] = warp_acc;
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

// Kernel 3: Dual-Block Warp (Optimization from Cloud Sam review)
// 32 threads process 2 blocks (64 weights) per iteration:
// - Lanes 0..15 handle block (2*i + 0)
// - Lanes 16..31 handle block (2*i + 1)
// - Each thread loads 1 byte of qs, extracts BOTH nibbles, and multiplies by x
// - Cuts loop iterations by 50% (from 48 down to 24) and eliminates branch divergence!
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
