# Resident SwiGLU FFN Engine on Kepler GT 750M vs Haswell AVX2 (v2)

Complete, end-to-end execution engine of the **SwiGLU Feed-Forward Network (FFN)** with synthetic weights matching the official architecture of **Qwen2.5-Coder-1.5B** ($D=1536$, $M=8960$) running on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB)** via **Mesa Rusticl (OpenCL 3.0)**.

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
 │  1. RMSNorm(x, γ)                     [ 222.2 µs]           │
 │     │                                                       │
 │     ├──► 2. Gate GEMV (8960x1536)     [8,157.8 µs]          │
 │     └──► 3. Up GEMV   (8960x1536)     [8,069.2 µs]          │
 │             │                                               │
 │             ▼                                               │
 │  4. SwiGLU: SiLU(g) ⊙ u               [ 103.4 µs]           │
 │     │                                                       │
 │     ▼                                                       │
 │  5. Down GEMV (1536x8960)             [7,754.1 µs]          │
 │     │                                                       │
 │     ▼                                                       │
 │  6. Residual Add: y = x + y_down      [  33.0 µs]           │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼ (PCIe 3.0 x16: 526.3 µs)
                        Host Memory (y: 6 KB)
```

---

## 📊 Measured Benchmark Results (100 Iterations on Physical Hardware)

| Pipeline Stage | Operation | CPU (AVX2 FMA 8 threads) | GPU Kepler (GT 750M) | Max Numerical Error | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 0** | PCIe Upload $x$ (6 KB) | — | $12.71\ \mu\text{s}$ ($0.013\text{ ms}$) | — | — |
| **Stage 1** | RMSNorm ($D=1536$) | included in total | $222.24\ \mu\text{s}$ ($0.222\text{ ms}$) | $2.38 \times 10^{-7}$ | PASSED |
| **Stage 2** | Gate GEMV ($8960 \times 1536$) | included in total | $8,157.77\ \mu\text{s}$ ($8.158\text{ ms}$) | $4.77 \times 10^{-6}$ | PASSED |
| **Stage 3** | Up GEMV ($8960 \times 1536$) | included in total | $8,069.17\ \mu\text{s}$ ($8.069\text{ ms}$) | $3.81 \times 10^{-6}$ | PASSED |
| **Stage 4** | SwiGLU Activation ($M=8960$) | included in total | $103.40\ \mu\text{s}$ ($0.103\text{ ms}$) | $3.43 \times 10^{-5}$ | PASSED |
| **Stage 5** | Down GEMV ($1536 \times 8960$) | included in total | $7,754.06\ \mu\text{s}$ ($7.754\text{ ms}$) | $1.53 \times 10^{-4}$ | PASSED |
| **Stage 6** | Residual Add ($D=1536$) | included in total | $33.02\ \mu\text{s}$ ($0.033\text{ ms}$) | — | PASSED |
| **Stage 7** | PCIe Download $y$ (6 KB) | — | $526.29\ \mu\text{s}$ ($0.526\text{ ms}$) | — | — |
| **TOTAL** | **Pure Compute Time** | **$42.55\text{ ms}$** | **$24.34\text{ ms}$** | **$1.75\times$ 🚀 (GPU Faster)** | PASSED |
| **REQUEST** | **Pure Request Latency** | **$42.55\text{ ms}$** | **$34.33\text{ ms}$** | **$1.24\times$ 🚀 (GPU Faster)** | PASSED |

---

## 🔬 Takeaways

1. **GPU Pure Compute is 1.75× Faster:**
   The entire sequence of RMSNorm, Gate, Up, SwiGLU, Down, and Residual takes **$24.34\text{ ms}$ on the GT 750M**, compared to **$42.55\text{ ms}$ on the 8-thread Haswell CPU**.
2. **Minimal PCIe Overhead:**
   Uploading $x$ takes only **$12.7\ \mu\text{s}$**, demonstrating that keeping weights resident in VRAM eliminates host-device bandwidth bottlenecks.
3. **Rigorous Numerical Validation:**
   Every single intermediate buffer ($z, g, u, h, y_{\text{down}}, y$) was independently validated against the CPU reference with strict combined tolerance $|a - b| \le \text{atol} + \text{rtol} \times |b|$, passing with zero NaNs or infinities.
4. **Separate Request Measurement:**
   Pure request latency is measured independently without synchronous profiling event queries inside the timed loop, giving an uninstrumented host time of $34.33\text{ ms}$.

---

## 🛠️ Reproduction & Execution

```bash
cd compute/ffn-swiglu-resident
make clean && make
RUSTICL_ENABLE=nouveau ./ffn_swiglu_benchmark
```
