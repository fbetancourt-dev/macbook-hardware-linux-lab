# Resident Transformer Decoder Layer on Kepler GT 750M vs Haswell AVX2

Complete, end-to-end execution of a **Resident Transformer Decoder Layer** ($x \to \text{Attention} \to r \to \text{FFN} \to y$) matching the official architecture of **Qwen2.5-Coder-1.5B** on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB GDDR5)** via **Mesa Rusticl (OpenCL 3.0)** compared to an 8-thread **Intel Core i7-4870HQ (Haswell AVX2/FMA3)** CPU baseline.

---

## 🎯 Architecture & Dataflow

A modern autoregressive decoder layer executes two core sub-layers with residual streams:
1. **Multi-Head / Grouped-Query Attention (GQA)**: $D=1536$, $H_q=12$, $H_{kv}=2$, $d=128$, $T_{\max}=4096$.
2. **SwiGLU Feed-Forward Network (FFN)**: $D=1536$, $M=8960$ ($5.83\times$ expansion).

All weights (~25.10 MB in `Q4_0`), persistent KV caches (8.00 MB in FP32), and dynamic activation arenas (0.34 MB) remain **100% resident in GPU VRAM**. No intermediate activation is ever transferred back to the host CPU via PCIe.

```
Host Memory (x: 6 KB)
       │
       ▼ (PCIe Upload: ~12 µs)
┌─────────────────────────────────────────────────────────────┐
│ 1. RMSNorm (Attention)                                      │
│    │                                                        │
│    ▼                                                        │
│ 2. QKV Projection with Bias (2048 x 1536)                   │
│    │                                                        │
│    ▼                                                        │
│ 3. RoPE (Q & K) + KV Cache In-Place Append                  │
│    │                                                        │
│    ▼                                                        │
│ 4. GQA Attention Scores (Q · K^T / sqrt(d))                 │
│    │                                                        │
│    ▼                                                        │
│ 5. Softmax (Head-parallel numerically stable)               │
│    │                                                        │
│    ▼                                                        │
│ 6. Split-K Segmented Value Combination + Partial Reduce     │
│    │                                                        │
│    ▼                                                        │
│ 7. Wo Projection (1536 x 1536) + Residual Add: r = x + Wo   │
│    │                                                        │
│    ▼                                                        │
│ 8. RMSNorm (FFN) on residual r                              │
│    │                                                        │
│    ▼                                                        │
│ 9. Fused Gate + Up + SiLU GEMV (8960 x 1536) -> h           │
│    │                                                        │
│    ▼                                                        │
│ 10. Fused Down GEMV (1536 x 8960) + In-Place Residual:      │
│     y = r + Down(h)                                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (PCIe Download: ~520 µs)
                     Host Memory (y: 6 KB)
```

---

## 📊 Measured Empirical Results (Physical Hardware Benchmark)

- **Device:** NVIDIA GeForce GT 750M (Kepler GK107, 384 cores, 2 GB GDDR5) via Mesa Rusticl OpenCL 3.0 (`NVE7`).
- **CPU Reference:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell AVX2 + FMA3, 8 OpenMP threads).
- **Benchmark Methodology:** 20 alternated, interleaved iterations per context length to eliminate thermal bias.

| Context Length ($T$) | CPU Haswell (8T) | GPU Kepler GT 750M | Speedup vs CPU | Max Numerical Diff | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$T = 1$** | $65.00\text{ ms}$ | **$33.84\text{ ms}$** | **$1.92\times$** 🚀 | $1.76 \times 10^{-2}$ | PASSED |
| **$T = 32$** | $73.87\text{ ms}$ | **$35.06\text{ ms}$** | **$2.11\times$** 🚀 | $1.17 \times 10^{-2}$ | PASSED |
| **$T = 128$** | $50.77\text{ ms}$ | **$31.41\text{ ms}$** | **$1.62\times$** 🚀 | $8.79 \times 10^{-3}$ | PASSED |
| **$T = 512$** | $64.86\text{ ms}$ | **$32.03\text{ ms}$** | **$2.02\times$** 🚀 | $1.66 \times 10^{-2}$ | PASSED |
| **$T = 1024$** | $81.14\text{ ms}$ | **$51.88\text{ ms}$** | **$1.56\times$** 🚀 | $1.86 \times 10^{-2}$ | PASSED |
| **$T = 2048$** | $66.80\text{ ms}$ | **$51.59\text{ ms}$** | **$1.29\times$** 🚀 | $1.25 \times 10^{-1}$ | PASSED |

---

## 🔬 Key Engineering Insights

1. **End-to-End Speedup on Legacy Hardware:**
   The entire decoder layer runs up to **$2.11\times$ faster on the GT 750M** than on all 8 threads of the Haswell CPU with AVX2. Even at $T=2048$, the GPU sustains a **$1.29\times$ advantage** ($51.59\text{ ms}$ vs $66.80\text{ ms}$).
2. **Minimal VRAM Footprint:**
   - Layer Weights: **25.10 MB** (Q4_0 quantization).
   - Full KV Cache ($T=4096$): **8.00 MB** (FP32).
   - Dynamic Activations Arena: **0.34 MB** (shared in-place buffers for $x \to r \to y$, norm, qkv, scores, and intermediate down).
   - **Total footprint per layer is under 34 MB**, making multi-layer residency easily fit within the GT 750M's 2048 MB VRAM.
3. **Split-K Flash-Decoding Scaling:**
   Segmenting the context into 256-token tiles with 128 threads per workgroup prevents thread starvation at large context windows, maintaining sub-$52\text{ ms}$ total layer execution even at $T=2048$.

---

## 🛠️ Build & Run

```bash
cd compute/transformer-decoder-layer
make clean && make
RUSTICL_ENABLE=nouveau ./decoder_layer_benchmark
```
