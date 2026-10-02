# Resident SwiGLU FFN Engine on Kepler GT 750M vs Haswell AVX2 (v3)

Complete, end-to-end execution engine of the **SwiGLU Feed-Forward Network (FFN)** with synthetic weights matching the official architecture of **Qwen2.5-Coder-1.5B** ($D=1536$, $M=8960$) running on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB)** via **Mesa Rusticl (OpenCL 3.0)**.

---

## 🎯 Architecture & Dataflow

The FFN represents **~65% of the total floating-point compute** of each autoregressive token in a modern Transformer.

All weight tensors remain **100% resident in GPU VRAM** (~22.15 MB total in `Q4_0`), eliminating PCIe weight transfer bottlenecks.

### Comparison: Modular (6 Stages) vs Fused Pipeline (3 Stages)

```
        MODULAR PIPELINE (6 Stages)                        FUSED PIPELINE (3 Stages)
     Host Memory (x: 6 KB)                              Host Memory (x: 6 KB)
              │                                                  │
              ▼ (PCIe Upload: 12.7 µs)                           ▼ (PCIe Upload: 12.7 µs)
 ┌──────────────────────────────────────┐           ┌──────────────────────────────────────┐
 │ 1. RMSNorm(x, γ)                     │           │ 1. RMSNorm(x, γ)                     │
 │    │                                 │           │    │                                 │
 │    ├──► 2. Gate GEMV (8960x1536)     │           │    ▼                                 │
 │    └──► 3. Up GEMV   (8960x1536)     │           │ 2. Fused Gate + Up + SiLU            │
 │            │                         │           │    • Factored scale dot products     │
 │            ▼                         │           │    • Vector z reused in registers    │
 │ 4. SwiGLU: SiLU(g) ⊙ u               │           │    • Direct h[row] activation write  │
 │    │                                 │           │    │                                 │
 │    ▼                                 │           │    ▼                                 │
 │ 5. Down GEMV (1536x8960)             │           │ 3. Fused Down GEMV + Residual        │
 │    │                                 │           │    • In-place add: y = x + Down(h)   │
 │    ▼                                 │           │    • Eliminates y_down buffer        │
 │ 6. Residual Add: y = x + y_down      │           └──────────────────┬───────────────────┘
 └──────────────────┬───────────────────┘                              │
                    │                                                  ▼ (PCIe Download: 526 µs)
                    ▼ (PCIe Download: 526 µs)                  Host Memory (y: 6 KB)
            Host Memory (y: 6 KB)
```

---

## 📊 Measured Benchmark Results (Alternated Interleaved 100 Runs on Physical Hardware)

To strictly eliminate thermal throttling and cache warmup bias between pipelines, the benchmark alternates iterations ($M_0, F_0, M_1, F_1, \dots$).

| Pipeline Implementation | Stages | End-to-End Wall Clock | Speedup vs CPU | Max Numerical Error | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU Reference (AVX2 FMA 8 threads)** | — | **$50.30 - 53.58\text{ ms}$** | $1.00\times$ (Baseline) | — | — |
| **GPU Kepler Modular Pipeline** | 6 | **$24.39 - 29.59\text{ ms}$** | **$1.81 - 2.06\times$** 🚀 | $1.53 \times 10^{-4}$ | PASSED |
| **GPU Kepler Fused Pipeline** | 3 | **$22.11 - 23.86\text{ ms}$** | **$2.25 - 2.27\times$** 🚀 | $1.30 \times 10^{-4}$ | PASSED |

### Key Improvements:
- **Fused vs Modular:** Fused execution eliminates 3 kernel launches and intermediate global VRAM buffers ($g, u, y_{\text{down}}$), cutting uninstrumented host request latency down to **$22.11\text{ ms}$** ($1.10 - 1.24\times$ faster, saving up to $5.72\text{ ms}$).
- **Total Speedup over Haswell CPU:** **$2.27\times$ faster on the GT 750M** ($22.11\text{ ms}$ vs $50.30\text{ ms}$).
- **Numerical Fidelity:** Both pipelines match the FP32 CPU reference within $|y_{\text{gpu}} - y_{\text{cpu}}| \le 1.30 \times 10^{-4}$ ($< \text{atol} + \text{rtol} \cdot |y|$), with zero NaNs or infinities.

---

## 🔬 Architectural Details & Optimizations

1. **Explicit Workgroup Contract:**
   All GEMV kernels specify `__attribute__((reqd_work_group_size(WG_THREADS, 1, 1)))` (128 threads = 4 warps), allowing Mesa Rusticl / Nouveau to optimize register allocation without spill hazards.
2. **Factored Scale Dot Product (Low Register Pressure):**
   In `gemv_swiglu_fused`, computing `acc_gate += dg * (g0*z0 + g1*z1)` saves registers compared to naive expansion, preventing local memory spill while keeping maximum arithmetic density.
3. **Alternated Interleaved Benchmarking:**
   Measuring Modular and Fused iterations back-to-back prevents thermal state divergence between measurements.
4. **Zero Host Allocation Overhead:**
   All kernel objects and arguments are pre-bound. The benchmark loop performs purely non-blocking enqueues followed by a single barrier on the final download event.
5. **Dimension Safety Assertions:**
   Host asserts `D_MODEL % 64 == 0` and `D_FFN % 64 == 0`, ensuring that the dual-block reduction ($2 \times 32 = 64$ weights per iteration) never silently drops unaligned blocks.

---

## 🛠️ Reproduction & Execution

```bash
cd compute/ffn-swiglu-resident
make clean && make
RUSTICL_ENABLE=nouveau ./ffn_swiglu_benchmark
```
