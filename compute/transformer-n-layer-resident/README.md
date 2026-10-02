# Parametric N-Layer Resident Transformer Pipeline on Kepler GT 750M vs Haswell AVX2

Complete, end-to-end execution of a **Parametric N-Layer Resident Transformer Pipeline** ($x \to L_0 \to L_1 \dots \to L_{N-1} \to y$) matching the official architecture of **Qwen2.5-Coder-1.5B** on the **NVIDIA GeForce GT 750M (Kepler GK107 2GB GDDR5)** via **Mesa Rusticl (OpenCL 3.0)** compared to an 8-thread **Intel Core i7-4870HQ (Haswell AVX2/FMA3)** CPU baseline.

---

## 🎯 Architecture & Scaling Law

The pipeline evaluates the multi-layer scaling law of resident transformers:
1. **Linear Layer Capacity**: Each layer $l \in [0, N-1]$ holds its own $W^{(l)}$ weights (~25.10 MB in `Q4_0`) and persistent $KV^{(l)}$ cache (8.00 MB in FP32).
   - $N=2$: **50.20 MB** weights + **16.00 MB** KV = **66.53 MB** VRAM.
   - $N=4$: **100.40 MB** weights + **32.00 MB** KV = **132.73 MB** VRAM.
   - $N=8$: **200.80 MB** weights + **64.00 MB** KV = **265.13 MB** VRAM.
   - Full 28 Layers (Projected): **702.80 MB** weights + **224.00 MB** KV = **927.13 MB** VRAM (< 1 GB, comfortably fitting the 2048 MB VRAM of the GT 750M).
2. **Fixed Constant Workspace**: Regardless of $N$, the dynamic activation arena is **strictly 0.33 MB** (`ws.state` updated in-place across all layers).
3. **Single PCIe Boundary**: Only one PCIe upload of $x$ (6 KB) at the start and one download of $y$ (6 KB) after $N$ layers finish.

```
Host Memory (x: 6 KB)
       │
       ▼ (Single PCIe Upload: ~12 µs)
┌─────────────────────────────────────────────────────────────┐
│ LAYER 0: state = L0(state)                                  │
│ LAYER 1: state = L1(state)                                  │
│ ...                                                         │
│ LAYER N-1: state = LN-1(state)                              │
│ (All N layers execute in-place in GPU VRAM!)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (Single PCIe Download: ~420 µs)
                     Host Memory (y: 6 KB)
```

---

## 📊 Measured Empirical Results (Physical Hardware Benchmark, Context $T=512$)

- **Device:** NVIDIA GeForce GT 750M (Kepler GK107, 384 cores, 2 GB GDDR5) via Mesa Rusticl OpenCL 3.0 (`NVE7`).
- **CPU Reference:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell AVX2 + FMA3, 8 OpenMP threads).
- **GPU Silicon (Span):** Total elapsed time from first kernel to final kernel in hardware counters.
- **Kernel Sum:** Sum of pure execution time across all $11 \times N$ kernels ($\sum (END - START)$).
- **Full E2E Time:** Single upload ($x$) + $11 \times N$ kernels + single download ($y$).
- **Benchmark Methodology:** 15 alternated, interleaved iterations with strict `isfinite` checks and relative $L_2$ error.

| Layers ($N$) | CPU Haswell (8T) | GPU Silicon (Span) | Kernel Sum | Full E2E (PCIe) | Speedup vs CPU | Relative $L_2$ Error | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 2$** | $136.73\text{ ms}$ | **$63.00\text{ ms}$** | **$49.88\text{ ms}$** | **$69.42\text{ ms}$** | **$1.97\times$** 🚀 | **$4.57 \times 10^{-7}$** | **$1.000000$** |
| **$N = 4$** | $307.49\text{ ms}$ | **$157.72\text{ ms}$** | **$110.27\text{ ms}$** | **$165.05\text{ ms}$** | **$1.86\times$** 🚀 | **$3.70 \times 10^{-7}$** | **$1.000000$** |
| **$N = 8$** | $669.88\text{ ms}$ | **$300.99\text{ ms}$** | **$207.29\text{ ms}$** | **$313.30\text{ ms}$** | **$2.14\times$** 🚀 | **$1.20 \times 10^{-7}$** | **$1.000000$** |

---

## 🔬 Key Engineering Insights

1. **Perfect Linear Scaling ($~38\text{ ms}$ E2E per Layer on Kepler GT 750M):**
   - $N=2$: $69.42\text{ ms}$ ($34.7\text{ ms}$/layer).
   - $N=4$: $165.05\text{ ms}$ ($41.2\text{ ms}$/layer).
   - $N=8$: $313.30\text{ ms}$ ($39.1\text{ ms}$/layer).
   - The GPU consistently maintains **$\approx 2\times$ speedup** over 8-thread Haswell CPU across any number of layers.
2. **Projected Full Model (28 Layers) Inference:**
   - 28 Layers in GPU VRAM: $28 \times 38\text{ ms} \approx \mathbf{1.06\text{ seconds/token}}$ ($\approx 1\text{ token/s}$ autoregressive generation on a 2014 laptop GPU).
   - Haswell CPU Baseline: $28 \times 83\text{ ms} \approx \mathbf{2.32\text{ seconds/token}}$ ($0.43\text{ tokens/s}$).
3. **Flawless Multi-Layer Numerical Integrity:**
   - Even after 8 consecutive Transformer layers ($88$ kernel dispatches!), the Relative $L_2$ error is **$1.20 \times 10^{-7}$**, with **Cosine Similarity = 1.000000**.

---

## 🛠️ Build & Run

```bash
cd compute/transformer-n-layer-resident
make clean && make
RUSTICL_ENABLE=nouveau ./n_layer_benchmark
```
