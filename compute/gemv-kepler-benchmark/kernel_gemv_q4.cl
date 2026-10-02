// OpenCL 1.2 / 3.0 GEMV Q4_0 Coalesced Kernel for Kepler GT 750M
// High memory coalescence: 32 threads (1 warp) cooperate on each row.
// Robust bounds protection: all threads unconditionally participate in workgroup barriers.

typedef struct {
    ushort d;      // FP16 scale delta
    uchar qs[16];  // 32 4-bit nibbles
} block_q4_0;

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

// Kernel 2: Warp-Cooperative (32 threads per row)
// 1 Workgroup has 4 warps (128 threads) processing 4 rows.
// Threads for out-of-bounds rows participate in all barriers with zero contribution,
// ensuring strict conformance to OpenCL workgroup barrier rules.
#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define WG_THREADS (WARP_SIZE * WARPS_PER_WG)

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

    // Shared memory reduction (all 128 threads reach the barrier unconditionally)
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
