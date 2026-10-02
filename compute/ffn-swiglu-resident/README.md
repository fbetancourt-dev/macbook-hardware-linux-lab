# Resident SwiGLU FFN Engine on Kepler GT 750M vs Haswell AVX2

Complete, end-to-end execution engine of the **SwiGLU Feed-Forward Network (FFN)** matching the official architecture of **Qwen2.5-Coder-1.5B** ($D=1536$, $M=8960$) running on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB)** via **Mesa Rusticl (OpenCL 3.0)**.

---

## 🎯 Architecture & Dataflow

The FFN represents **~65% of the total floating-point compute** of each autoregressive token in a modern Transformer.

All weight tensors remain **100% resident in GPU VRAM** (~22.15 MB total in `Q4_0`), eliminating PCIe weight transfer bottlenecks:

```
  Host Memory (x: 6 KB)
       │
       ▼ (PCIe 3.0 x16: 12.7 µs)
 ┌─────────────────────────────────────────────────────────────┐
 │                  GT 750M VRAM Pipeline                      │
 │                                                             │
 │  1. RMSNorm(x, γ)                     [ 53.9 µs]            │
 │     │                                                       │
 │     ├──► 2. Gate GEMV (8960x1536)     [7,636.8 µs]          │
 │     └──► 3. Up GEMV   (8960x1536)     [8,103.1 µs]          │
 │             │                                               │
 │             ▼                                               │
 │  4. SwiGLU: SiLU(g) ⊙ u               [ 48.1 µs]            │
 │     │                                                       │
 │     ▼                                                       │
 │  5. Down GEMV (1536x8960)             [7,289.7 µs]          │
 │     │                                                       │
 │     ▼                                                       │
 │  6. Residual Add: y = x + y_down      [ 31.3 µs]            │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼ (PCIe 3.0 x16: 742.4 µs)
                        Host Memory (y: 6 KB)
```

---

## 📊 Measured Benchmark Results (100 Iterations on Physical Hardware)

| Pipeline Stage | Operation | CPU (AVX2 FMA 8 threads) | GPU Kepler (GT 750M) | Max Numerical Error |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 0** | PCIe Upload $x$ (6 KB) | — | $12.70\ \mu\text{s}$ ($0.013\text{ ms}$) | — |
| **Stage 1** | RMSNorm ($D=1536$) | included in total | $53.93\ \mu\text{s}$ ($0.054\text{ ms}$) | $2.38 \times 10^{-7}$ |
| **Stage 2** | Gate GEMV ($8960 \times 1536$) | included in total | $7,636.75\ \mu\text{s}$ ($7.637\text{ ms}$) | $4.77 \times 10^{-6}$ |
| **Stage 3** | Up GEMV ($8960 \times 1536$) | included in total | $8,103.12\ \mu\text{s}$ ($8.103\text{ ms}$) | $3.81 \times 10^{-6}$ |
| **Stage 4** | SwiGLU Activation ($M=8960$) | included in total | $48.11\ \mu\text{s}$ ($0.048\text{ ms}$) | $3.43 \times 10^{-5}$ |
| **Stage 5** | Down GEMV ($1536 \times 8960$) | included in total | $7,289.71\ \mu\text{s}$ ($7.290\text{ ms}$) | $1.53 \times 10^{-4}$ |
| **Stage 6** | Residual Add ($D=1536$) | included in total | $31.27\ \mu\text{s}$ ($0.031\text{ ms}$) | — |
| **Stage 7** | PCIe Download $y$ (6 KB) | — | $742.41\ \mu\text{s}$ ($0.742\text{ ms}$) | — |
| **TOTAL** | **Pure Compute Time** | **$46.33\text{ ms}$** | **$23.16\text{ ms}$** | **$2.00\times$ 🚀 (GPU Faster)** |
| **E2E** | **Host Wall-Clock Latency** | **$46.33\text{ ms}$** | **$34.67\text{ ms}$** | **$1.34\times$ 🚀 (GPU Faster)** |

---

## 🔬 Takeaways

1. **GPU Pure Compute is 2.00× Faster:**
   The entire sequence of RMSNorm, Gate, Up, SwiGLU, Down, and Residual takes **$23.16\text{ ms}$ on the GT 750M**, compared to **$46.33\text{ ms}$ on the 8-thread Haswell CPU**.
2. **Minimal PCIe Overhead:**
   Uploading $x$ takes only **$12.7\ \mu\text{s}$**, demonstrating that keeping weights resident in VRAM eliminates host-device bandwidth bottlenecks.
3. **Stage-by-Stage Mathematical Stability:**
   Every single intermediate buffer ($z, g, u, h, y_{\text{down}}, y$) was independently validated against the double-precision reference, showing complete numerical stability with zero NaNs or infinities.

---

## 🛠️ Reproduction & Execution

```bash
cd compute/ffn-swiglu-resident
make clean && make
RUSTICL_ENABLE=nouveau ./ffn_swiglu_benchmark
```
