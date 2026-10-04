// OpenCL 1.2 / 3.0 High-Performance GEMV for Q8_0 Blocks on Kepler GT 750M
// Weights: block_q8_0 (34 bytes per 32 weights: 2 bytes half delta + 32 bytes int8)
// K = 896 (28 blocks per row = 952 bytes per row)
// Workgroup: 128 threads (4 warps of 32 threads, computing 4 rows per workgroup)

#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define BLOCKS_PER_ROW_Q8 28 // 896 / 32

__kernel
__attribute__((reqd_work_group_size(128, 1, 1)))
void gemv_q8_0(
    __global const uchar *W,          // [M, 952] bytes
    __global const float *x,          // [896] FP32
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
        __global const uchar *row_ptr = W + (size_t)row * 952;

        for (int b = 0; b < BLOCKS_PER_ROW_Q8; b++) {
            __global const uchar *blk_ptr = row_ptr + b * 34;
            ushort d_raw = *(__global const ushort*)blk_ptr;
            float d = vload_half(0, &d_raw);

            char q = *(__global const char*)(blk_ptr + 2 + lane_id);
            float x_val = x[b * 32 + lane_id];

            acc += d * (float)q * x_val;
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
