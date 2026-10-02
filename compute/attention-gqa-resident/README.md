# Resident Grouped-Query Attention (GQA) Engine on Kepler GT 750M vs Haswell AVX2

Complete, end-to-end execution engine of the **Grouped-Query Attention (GQA) Layer with Resident KV Cache** matching the official architecture of **Qwen2.5-Coder-1.5B** ($D=1536$, $N_{\text{heads\_q}}=12$, $N_{\text{heads\_kv}}=2$, $d_{\text{head}}=128$, $T_{\text{max}}=4096$) running on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB)** via **Mesa Rusticl (OpenCL 3.0)**.

---

## 🎯 Architecture & Dataflow

In modern LLMs, Attention represents **~35% of the total floating-point compute** during autoregressive generation. 

Unlike the Feed-Forward Network (FFN), whose computational cost is constant per token, Attention scales with the historical context length $T$ due to the $Q \cdot K^T$ dot products and $P \cdot V$ context reduction over the KV cache.

All attention projections and the entire 4096-token KV cache remain **100% resident in GPU VRAM** (~11.15 MB total):

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
 │             │                                               │
 │             ▼                                               │
 │  6. Context Value Combination: Attn_Out = P · V             │
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

## 📦 Resident VRAM Footprint

| Component | Dimensions / Precision | Memory Footprint |
| :--- | :---: | :---: |
| **$W_{\text{qkv}}$ Projection** | $2048 \times 1536$ `Q4_0` | $1.69\text{ MB}$ |
| **$b_{\text{qkv}}$ Bias Vector** | $2048$ floats (FP32) | $8.00\text{ KB}$ |
| **$W_o$ Projection** | $1536 \times 1536$ `Q4_0` | $1.27\text{ MB}$ |
| **Resident K-Cache** | $2 \text{ heads} \times 4096 \text{ tokens} \times 128$ (FP32) | $4.00\text{ MB}$ |
| **Resident V-Cache** | $2 \text{ heads} \times 4096 \text{ tokens} \times 128$ (FP32) | $4.00\text{ MB}$ |
| **Gamma + Scratch Buffers** | $\gamma_{\text{attn}}, z, qkv, \text{scores}$ (FP32) | $0.20\text{ MB}$ |
| **TOTAL RESIDENT MEMORY** | — | **$\sim 11.15\text{ MB}$** (fits easily in 2 GB) |

---

## 📊 Measured Benchmark Results (Context Length Sweep on Physical Hardware)

Comparing the **NVIDIA GeForce GT 750M (Kepler GK107)** against the **Intel Core i7-4870HQ (Haswell AVX2 + OpenMP 8 threads)**:

| Context Length ($T$) | CPU Haswell (8 threads) | GPU Kepler GT 750M | Speedup vs CPU | Max Numerical Error | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$T = 1$** (Initial Token) | $16.51\text{ ms}$ | **$8.19\text{ ms}$** | **$2.01\times$** 🚀 | $1.72 \times 10^{-5}$ | PASSED |
| **$T = 32$** (Short Prompt) | $17.84\text{ ms}$ | **$7.83\text{ ms}$** | **$2.28\times$** 🚀 | $2.77 \times 10^{-5}$ | PASSED |
| **$T = 128$** (Chat Context) | $16.37\text{ ms}$ | **$11.00\text{ ms}$** | **$1.49\times$** 🚀 | $9.54 \times 10^{-6}$ | PASSED |
| **$T = 512$** (Code Function) | $17.46\text{ ms}$ | **$16.50\text{ ms}$** | **$1.06\times$** 🚀 | $3.81 \times 10^{-5}$ | PASSED |
| **$T = 1024$** (Medium Context) | **$20.95\text{ ms}$** | $24.20\text{ ms}$ | $0.87\times$ (Crossover) | $3.73 \times 10^{-4}$ | PASSED |
| **$T = 2048$** (Long Context) | **$28.93\text{ ms}$** | $62.34\text{ ms}$ | $0.46\times$ (CPU Faster) | $6.54 \times 10^{-4}$ | PASSED |
| **$T = 4096$** (Full Context) | **$42.65\text{ ms}$** | $73.10\text{ ms}$ | $0.58\times$ (CPU Faster) | $5.42 \times 10^{-4}$ | PASSED |

---

## 🔬 Architectural Insights & The Crossover Point

1. **GPU Dominates for Typical Generation ($T \le 512$):**
   For interactive generation lengths ($T \le 512$), the GT 750M is **up to $2.28\times$ faster than the 8-thread Haswell CPU** ($7.83\text{ ms}$ vs $17.84\text{ ms}$). In this regime, the GEMV projections ($W_{qkv}$ and $W_o$) dominate the execution time, where Kepler's raw memory bandwidth and warp-cooperative unpacking excel.
2. **The Crossover Point ($T \approx 768$):**
   Between $T = 512$ and $T = 1024$, the attention score matrix calculation ($Q \cdot K^T$) and value reduction ($P \cdot V$) become memory-bound over the KV cache.
3. **Haswell L3 Cache vs Kepler Memory Latency:**
   At $T > 1024$, Haswell's large 6 MB L3 cache and fast out-of-order execution handle the scalar softmax and reduction efficiently, while Kepler's naive non-tiled reduction across 4096 tokens suffers from global memory round-trips. This clearly reveals the exact target for the next optimization: **FlashAttention-style tiled online softmax**.

---

## 🛠️ Reproduction & Execution

```bash
cd compute/attention-gqa-resident
make clean && make
RUSTICL_ENABLE=nouveau ./attention_gqa_benchmark
```
