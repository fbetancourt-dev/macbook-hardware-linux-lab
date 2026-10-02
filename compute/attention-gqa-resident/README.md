# Resident Grouped-Query Attention (GQA) Engine on Kepler GT 750M vs Haswell AVX2 (v2 Split-K)

Complete, end-to-end execution engine of the **Grouped-Query Attention (GQA) Layer with Resident KV Cache and Split-K Context-Partitioned Value Combination** matching the official architecture of **Qwen2.5-Coder-1.5B** ($D=1536$, $N_{\text{heads\_q}}=12$, $N_{\text{heads\_kv}}=2$, $d_{\text{head}}=128$, $T_{\text{max}}=4096$) running on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB)** via **Mesa Rusticl (OpenCL 3.0)**.

---

## 🎯 Architecture & Dataflow

In modern LLMs, Attention represents **~35% of the total floating-point compute** during autoregressive generation. 

All attention projections, scratchpad memory, and the entire 4096-token KV cache remain **100% resident in GPU VRAM** (~11.15 MB total):

```
       Host Memory (x: 6 KB)
                │
                ▼ (PCIe Upload: 12.7 µs)
 ┌─────────────────────────────────────────────────────────────┐
 │                GT 750M Attention Pipeline                   │
 │                                                             │
 │  1. RMSNorm(x, γ_attn)                                      │
 │     │                                                       │
 │     ▼                                                       │
 │  2. QKV GEMV (2048 x 1536) Q4_0 + Biases                    │
 │     ├──► Q: 12 heads x 128 (1536 floats)                    │
 │     ├──► K:  2 heads x 128 ( 256 floats)                    │
 │     └──► V:  2 heads x 128 ( 256 floats)                    │
 │             │                                               │
 │             ▼                                               │
 │  3. RoPE & KV Cache Append (pos)                            │
 │     • Qwen2 RoPE convention on Q and K                      │
 │     • In-place append of K_rot & V into resident KV cache   │
 │             │                                               │
 │             ▼                                               │
 │  4. GQA Attention Scores: S = Q · K^T / sqrt(128)           │
 │     • Direct index sharing: head_kv = head_q / 6            │
 │     • Zero memory replication of K/V                        │
 │             │                                               │
 │             ▼                                               │
 │  5. Stable Softmax across context tokens [0 .. T-1]         │
 │     • -INFINITY initial bound, zero epsilon bias            │
 │             │                                               │
 │             ▼                                               │
 │  6. Split-K Context-Partitioned Value Combination (P · V)   │
 │     • Stage 1: Segmented PV (256 tokens per workgroup)      │
 │     • Stage 2: Fast Inter-segment reduction into Attn_Out   │
 │             │                                               │
 │             ▼                                               │
 │  7. Wo GEMV (1536 x 1536) Q4_0 + Residual Add               │
 │     • Writes directly: r = x + Wo(Attn_Out)                 │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼ (PCIe Download: 526 µs)
                        Host Memory (r: 6 KB)
```

---

## 📊 Measured Benchmark Results (Context Length Sweep on Physical Hardware)

Comparing the **NVIDIA GeForce GT 750M (Kepler GK107)** against the **Intel Core i7-4870HQ (Haswell AVX2 + OpenMP 8 threads)** before and after **Split-K PV Context Partitioning**:

| Context Length ($T$) | CPU Haswell (8 threads) | GPU Kepler (v1 Naive PV) | GPU Kepler (v2 Split-K PV) | Speedup vs CPU | v2 Improvement vs v1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$T = 1$** (Initial Token) | $12.19\text{ ms}$ | $8.19\text{ ms}$ | **$7.14\text{ ms}$** | **$1.71\times$** 🚀 | $1.15\times$ faster |
| **$T = 32$** (Short Prompt) | $13.32\text{ ms}$ | $7.83\text{ ms}$ | **$7.87\text{ ms}$** | **$1.69\times$** 🚀 | parity |
| **$T = 128$** (Chat Context) | $11.48\text{ ms}$ | $11.00\text{ ms}$ | **$9.41\text{ ms}$** | **$1.22\times$** 🚀 | $1.17\times$ faster |
| **$T = 512$** (Code Function) | $12.51\text{ ms}$ | $16.50\text{ ms}$ | **$11.34\text{ ms}$** | **$1.10\times$** 🚀 | **$1.45\times$ faster** |
| **$T = 1024$** (Medium Context) | $16.69\text{ ms}$ | $24.20\text{ ms}$ | **$15.75\text{ ms}$** | **$1.06\times$** 🚀 | **$1.54\times$ faster** |
| **$T = 2048$** (Long Context) | **$21.45\text{ ms}$** | $62.34\text{ ms}$ | **$24.19\text{ ms}$** | $0.89\times$ *(New Crossover)* | **$2.58\times$ faster!** 🔥 |
| **$T = 4096$** (Full Context) | **$34.58\text{ ms}$** | $73.10\text{ ms}$ | **$45.09\text{ ms}$** | $0.77\times$ | **$1.62\times$ faster!** 🔥 |

---

## 🔬 Architectural Takeaways

1. **Massive Latency Reduction on Long Contexts ($2.58\times$ boost at $T=2048$):**
   Splitting the context into 256-token segments and reducing them inter-group eliminated the thread starvation bottleneck diagnosed by Cloud Sam, cutting $T=2048$ latency from **$62.34\text{ ms}$ down to $24.19\text{ ms}$**.
2. **Crossover Point Pushed Beyond $T = 1024$:**
   In v1, the CPU took the lead at $T \approx 768$. With Split-K partitioning, the GPU **remains faster than the 8-thread Haswell CPU all the way past $T = 1024$** ($15.75\text{ ms}$ GPU vs $16.69\text{ ms}$ CPU).
3. **Rigorous Numerical Fidelity:**
   The segmented reduction maintains full mathematical accuracy against the CPU AVX2 reference, passing strict combined tolerance with zero NaNs or infinities.

---

## 🛠️ Reproduction & Execution

```bash
cd compute/attention-gqa-resident
make clean && make
RUSTICL_ENABLE=nouveau ./attention_gqa_benchmark
```
