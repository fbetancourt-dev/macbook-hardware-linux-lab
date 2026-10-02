// OpenCL 1.2 / 3.0 Complete Resident Transformer Decoder Layer for Kepler GT 750M
// Matching official Qwen2.5-Coder-1.5B Architecture:
// D_MODEL = 1536, HEAD_DIM = 128, N_HEADS_Q = 12, N_HEADS_KV = 2 (GQA ratio = 6)
// D_FFN = 8960
//
// Combines:
// 1. Attention Block: RMSNorm(attn) -> QKV GEMV (2048x1536) + RoPE + GQA Scores + Softmax + Split-K PV + Wo GEMV (1536x1536) + Residual (r = x + Wo)
// 2. FFN Block:       RMSNorm(ffn)  -> Fused Gate+Up+SiLU (8960x1536) -> Fused Down+Residual (1536x8960, y = r + Down)
//
// Residuals and all intermediate activations remain 100% resident in GPU VRAM!

typedef struct {
    ushort d;      // FP16 scale delta
    uchar qs[16];  // 32 4-bit nibbles
} block_q4_0;

#define WARP_SIZE 32
#define WARPS_PER_WG 4
#define WG_THREADS (WARP_SIZE * WARPS_PER_WG)

#define HEAD_DIM 128
#define N_HEADS_Q 12
#define N_HEADS_KV 2
#define GQA_GROUP_SIZE 6
#define PV_SEGMENT_SIZE 256

// =========================================================================
// 1. RMSNorm Kernel (D elements, 1 work-group of 128 threads)
// Reused for both RMSNorm_attn and RMSNorm_ffn
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(128, 1, 1)))
void kernel_rmsnorm(
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
// 2. Dual-Block Warp GEMV Q4_0 with Optional Bias Kernel
// Computes y[row] = W[row, :] * x + bias[row]
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(WG_THREADS, 1, 1)))
void gemv_q4_0_bias(
    __global const block_q4_0 *W,
    __global const float      *x,
    __global const float      *bias,
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
        float b_val = (bias != NULL) ? bias[row] : 0.0f;
        y[row] = (sh_mem[local_id] + sh_mem[local_id + 1]) + b_val;
    }
}

// =========================================================================
// 3. RoPE & KV-Cache Append Kernel
// =========================================================================
__kernel void kernel_rope_and_kv_append(
    __global float       *q,            // [N_HEADS_Q * HEAD_DIM] = [1536] in-place
    __global const float *k,            // [N_HEADS_KV * HEAD_DIM] = [256]
    __global const float *v,            // [N_HEADS_KV * HEAD_DIM] = [256]
    __global float       *k_cache,      // [N_HEADS_KV, T_MAX, HEAD_DIM]
    __global float       *v_cache,      // [N_HEADS_KV, T_MAX, HEAD_DIM]
    const int pos,
    const int t_max,
    const float rope_base               // 1000000.0f for Qwen2
) {
    int tid = get_global_id(0); // 0 to 63
    if (tid >= 64) return;

    float theta = (float)pos * powr(rope_base, -2.0f * (float)tid / (float)HEAD_DIM);
    float cos_th = cos(theta);
    float sin_th = sin(theta);

    // RoPE on Q (12 heads)
    for (int h = 0; h < N_HEADS_Q; h++) {
        int base = h * HEAD_DIM;
        float q0 = q[base + tid];
        float q1 = q[base + tid + 64];

        q[base + tid]      = q0 * cos_th - q1 * sin_th;
        q[base + tid + 64] = q0 * sin_th + q1 * cos_th;
    }

    // RoPE on K & write to k_cache, and write V to v_cache (2 heads)
    for (int h = 0; h < N_HEADS_KV; h++) {
        int in_base = h * HEAD_DIM;
        float k0 = k[in_base + tid];
        float k1 = k[in_base + tid + 64];

        float k_rot0 = k0 * cos_th - k1 * sin_th;
        float k_rot1 = k0 * sin_th + k1 * cos_th;

        int out_base = (h * t_max + pos) * HEAD_DIM;
        k_cache[out_base + tid]      = k_rot0;
        k_cache[out_base + tid + 64] = k_rot1;

        v_cache[out_base + tid]      = v[in_base + tid];
        v_cache[out_base + tid + 64] = v[in_base + tid + 64];
    }
}

// =========================================================================
// 4. Attention Scores Q * K^T / sqrt(d)
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(WARP_SIZE, 1, 1)))
void kernel_gqa_scores(
    __global const float *q,         // [12, 128]
    __global const float *k_cache,   // [2, T_MAX, 128]
    __global float       *scores,    // [12, T_MAX]
    const int seq_len,
    const int t_max,
    const float scale_factor
) {
    int lane_id = get_local_id(0);   // 0 to 31
    int t       = get_group_id(0);   // 0 to seq_len - 1
    int h_q     = get_group_id(1);   // 0 to 11

    if (t >= seq_len || h_q >= N_HEADS_Q) return;

    int h_kv = h_q / GQA_GROUP_SIZE;
    int q_offset = h_q * HEAD_DIM;
    int k_offset = (h_kv * t_max + t) * HEAD_DIM;

    float dot = 0.0f;
    for (int i = 0; i < 4; i++) {
        int idx = lane_id + i * 32;
        dot += q[q_offset + idx] * k_cache[k_offset + idx];
    }

    __local float sh_dot[WARP_SIZE];
    sh_dot[lane_id] = dot;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id < 16) sh_dot[lane_id] += sh_dot[lane_id + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 8)  sh_dot[lane_id] += sh_dot[lane_id + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 4)  sh_dot[lane_id] += sh_dot[lane_id + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane_id < 2)  sh_dot[lane_id] += sh_dot[lane_id + 2];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id == 0) {
        scores[h_q * t_max + t] = (sh_dot[0] + sh_dot[1]) * scale_factor;
    }
}

// =========================================================================
// 5. Stable Softmax across context tokens for each query head
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(128, 1, 1)))
void kernel_softmax_gqa(
    __global float *scores,    // [12, T_MAX] in-place
    const int seq_len,
    const int t_max
) {
    int h_q = get_group_id(0); // 0 to 11
    int tid = get_local_id(0); // 0 to 127
    int n_threads = get_local_size(0);

    __global float *head_scores = scores + h_q * t_max;

    __local float sh_val[128];
    __local float l_max;
    __local float l_sum;

    // 1. Find Max
    float local_max = -INFINITY;
    for (int t = tid; t < seq_len; t += n_threads) {
        float val = head_scores[t];
        if (val > local_max) local_max = val;
    }
    sh_val[tid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64) { if (sh_val[tid + 64] > sh_val[tid]) sh_val[tid] = sh_val[tid + 64]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 32) { if (sh_val[tid + 32] > sh_val[tid]) sh_val[tid] = sh_val[tid + 32]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16) { if (sh_val[tid + 16] > sh_val[tid]) sh_val[tid] = sh_val[tid + 16]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)  { if (sh_val[tid + 8]  > sh_val[tid]) sh_val[tid] = sh_val[tid + 8]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)  { if (sh_val[tid + 4]  > sh_val[tid]) sh_val[tid] = sh_val[tid + 4]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)  { if (sh_val[tid + 2]  > sh_val[tid]) sh_val[tid] = sh_val[tid + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        l_max = (sh_val[0] > sh_val[1]) ? sh_val[0] : sh_val[1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Compute Exp & Sum
    float max_val = l_max;
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += n_threads) {
        float ev = exp(head_scores[t] - max_val);
        head_scores[t] = ev;
        local_sum += ev;
    }
    sh_val[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64) { sh_val[tid] += sh_val[tid + 64]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 32) { sh_val[tid] += sh_val[tid + 32]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16) { sh_val[tid] += sh_val[tid + 16]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8)  { sh_val[tid] += sh_val[tid + 8]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4)  { sh_val[tid] += sh_val[tid + 4]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2)  { sh_val[tid] += sh_val[tid + 2]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        l_sum = sh_val[0] + sh_val[1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Normalize
    float inv_sum = 1.0f / l_sum;
    for (int t = tid; t < seq_len; t += n_threads) {
        head_scores[t] *= inv_sum;
    }
}

// =========================================================================
// 6. Split-K Context-Partitioned PV Combination
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(HEAD_DIM, 1, 1)))
void kernel_gqa_value_combine_segmented(
    __global const float *probs,    // [12, T_MAX]
    __global const float *v_cache,  // [2, T_MAX, 128]
    __global float       *partial,  // [12, MAX_SEGMENTS, 128]
    const int seq_len,
    const int t_max,
    const int num_segments
) {
    int dim = get_local_id(0);      // 0 to 127
    int seg = get_group_id(0);      // 0 to num_segments - 1
    int h_q = get_group_id(1);      // 0 to 11

    if (h_q >= N_HEADS_Q || seg >= num_segments) return;

    int h_kv = h_q / GQA_GROUP_SIZE;

    int t_start = seg * PV_SEGMENT_SIZE;
    int t_end   = t_start + PV_SEGMENT_SIZE;
    if (t_end > seq_len) t_end = seq_len;

    __global const float *head_probs = probs + h_q * t_max;

    float acc = 0.0f;
    for (int t = t_start; t < t_end; t++) {
        float p = head_probs[t];
        int v_idx = (h_kv * t_max + t) * HEAD_DIM + dim;
        acc += p * v_cache[v_idx];
    }

    int partial_idx = (h_q * num_segments + seg) * HEAD_DIM + dim;
    partial[partial_idx] = acc;
}

__kernel 
__attribute__((reqd_work_group_size(HEAD_DIM, 1, 1)))
void kernel_gqa_reduce_segments(
    __global const float *partial,  // [12, MAX_SEGMENTS, 128]
    __global float       *attn_out, // [12, 128]
    const int num_segments
) {
    int dim = get_local_id(0);      // 0 to 127
    int h_q = get_group_id(0);      // 0 to 11

    if (h_q >= N_HEADS_Q) return;

    float sum = 0.0f;
    int base_offset = h_q * num_segments * HEAD_DIM + dim;
    for (int s = 0; s < num_segments; s++) {
        sum += partial[base_offset + s * HEAD_DIM];
    }

    attn_out[h_q * HEAD_DIM + dim] = sum;
}

// =========================================================================
// 7. Output Projection Wo (1536 x 1536) + Residual Add
// Writes directly: r[row] = state[row] + Wo(attn_out)[row]
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(WG_THREADS, 1, 1)))
void gemv_q4_0_wo_residual(
    __global const block_q4_0 *W_o,
    __global const float      *attn_out,
    __global float            *state,     // in-place: state = state + Wo(attn_out)
    const int D
) {
    int local_id = get_local_id(0);
    int warp_id  = local_id / WARP_SIZE;
    int lane_id  = local_id % WARP_SIZE;

    int row = (get_group_id(0) * WARPS_PER_WG) + warp_id;
    bool is_valid_row = (row < D);

    int nb = D / 32;
    __global const block_q4_0 *row_blocks = is_valid_row ? (W_o + row * nb) : NULL;

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
            acc0 += d * ((float)x0 * attn_out[k_base + byte_idx]);
            acc1 += d * ((float)x1 * attn_out[k_base + byte_idx + 16]);
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
        state[row] += (sh_mem[local_id] + sh_mem[local_id + 1]);
    }
}

// =========================================================================
// 8. Fused Gate + Up GEMV + SwiGLU Fused Kernel
// Projects Gate and Up for row 'row', applies SiLU, and writes h[row]
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

// =========================================================================
// 9. Fused Down GEMV + In-Place Residual Add
// Writes directly: state[row] = state[row] + Down_GEMV(h)
// =========================================================================
__kernel 
__attribute__((reqd_work_group_size(WG_THREADS, 1, 1)))
void gemv_q4_0_down_residual(
    __global const block_q4_0 *W_down,
    __global const float      *h,
    __global float            *state,    // in-place: state = state + Down(h)
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
        state[row] += (sh_mem[local_id] + sh_mem[local_id + 1]);
    }
}
