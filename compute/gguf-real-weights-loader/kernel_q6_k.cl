// OpenCL 1.2 / 3.0 High-Performance GEMV for Q6_K Superblocks on Kepler GT 750M
// Weights: block_q6_K (210 bytes per 256 weights)
// K = 1536 (6 superblocks per row = 1260 bytes per row)
// Workgroup: 128 threads (4 warps of 32 threads, computing 4 rows per workgroup)

typedef struct __attribute__((packed)) {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    ushort d; // IEEE 754 half
} block_q6_k_packed;

#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define BLOCKS_PER_ROW 6 // 1536 / 256

__kernel
__attribute__((reqd_work_group_size(128, 1, 1)))
void gemv_q6_k(
    __global const uchar *W,          // [M, 1260] bytes
    __global const float *x,          // [1536] FP32
    __global float       *y,          // [M] FP32
    const int M,
    const int K
) {
    int tid       = get_local_id(0);
    int warp_id   = tid / WARP_SIZE;
    int lane_id   = tid % WARP_SIZE;
    int row       = get_group_id(0) * WARPS_PER_WG + warp_id;

    float acc = 0.0f;

    if (row < M) {
        __global const uchar *row_ptr = W + (size_t)row * 1260;

        // Loop over 6 superblocks (each 256 weights = 210 bytes)
        for (int b = 0; b < BLOCKS_PER_ROW; b++) {
        __global const uchar *blk_ptr = row_ptr + b * 210;

        __global const uchar *ql = blk_ptr;
        __global const uchar *qh = blk_ptr + 128;
        __global const char  *sc = (__global const char*)(blk_ptr + 192);
        ushort d_raw             = *(__global const ushort*)(blk_ptr + 208);
        float d                  = vload_half(0, &d_raw);

        __global const float *x_blk = x + b * 256;

        int is = lane_id / 16; // 0 for lanes 0..15, 1 for lanes 16..31

        // First half: elements 0..127
        uchar ql_lo = ql[lane_id +  0];
        uchar ql_hi = ql[lane_id + 32];
        uchar qh_val = qh[lane_id];

        int q1 = (int)(ql_lo & 0x0F)        | (((int)(qh_val >> 0) & 3) << 4);
        int q2 = (int)(ql_hi & 0x0F)        | (((int)(qh_val >> 2) & 3) << 4);
        int q3 = (int)(ql_lo >> 4)          | (((int)(qh_val >> 4) & 3) << 4);
        int q4 = (int)(ql_hi >> 4)          | (((int)(qh_val >> 6) & 3) << 4);

        q1 -= 32; q2 -= 32; q3 -= 32; q4 -= 32;

        float w1 = d * (float)sc[is + 0] * (float)q1;
        float w2 = d * (float)sc[is + 2] * (float)q2;
        float w3 = d * (float)sc[is + 4] * (float)q3;
        float w4 = d * (float)sc[is + 6] * (float)q4;

        acc += w1 * x_blk[lane_id +  0];
        acc += w2 * x_blk[lane_id + 32];
        acc += w3 * x_blk[lane_id + 64];
        acc += w4 * x_blk[lane_id + 96];

        // Second half: elements 128..255
        // ql advances by 64 bytes, qh advances by 32 bytes, sc advances by 8 bytes
        __global const uchar *ql2 = ql + 64;
        __global const uchar *qh2 = qh + 32;
        __global const char  *sc2 = sc + 8;
        __global const float *x_blk2 = x_blk + 128;

        uchar ql2_lo = ql2[lane_id +  0];
        uchar ql2_hi = ql2[lane_id + 32];
        uchar qh2_val = qh2[lane_id];

        int q5 = (int)(ql2_lo & 0x0F)       | (((int)(qh2_val >> 0) & 3) << 4);
        int q6 = (int)(ql2_hi & 0x0F)       | (((int)(qh2_val >> 2) & 3) << 4);
        int q7 = (int)(ql2_lo >> 4)         | (((int)(qh2_val >> 4) & 3) << 4);
        int q8 = (int)(ql2_hi >> 4)         | (((int)(qh2_val >> 6) & 3) << 4);

        q5 -= 32; q6 -= 32; q7 -= 32; q8 -= 32;

        float w5 = d * (float)sc2[is + 0] * (float)q5;
        float w6 = d * (float)sc2[is + 2] * (float)q6;
        float w7 = d * (float)sc2[is + 4] * (float)q7;
        float w8 = d * (float)sc2[is + 6] * (float)q8;

        acc += w5 * x_blk2[lane_id +  0];
        acc += w6 * x_blk2[lane_id + 32];
        acc += w7 * x_blk2[lane_id + 64];
        acc += w8 * x_blk2[lane_id + 96];
    }
    }

    // Warp Reduction using Shared Memory
    __local float sh_red[128];
    sh_red[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Intra-warp tree reduction across 32 threads
    if (lane_id < 16) { sh_red[tid] += sh_red[tid + 16]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 8)  { sh_red[tid] += sh_red[tid + 8]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 4)  { sh_red[tid] += sh_red[tid + 4]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 2)  { sh_red[tid] += sh_red[tid + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 1)  { sh_red[tid] += sh_red[tid + 1]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id == 0 && row < M) {
        y[row] = sh_red[warp_id * WARP_SIZE];
    }
}
