# Real Weights GGUF Loader & Full 28-Layer Transformer Pipeline (Qwen2.5-Coder-1.5B)

Direct hardware execution and numerical validation of real model weights loaded from an official GGUF model file (**Qwen2.5-Coder-1.5B-Instruct-Q4_0.gguf**) running across the complete 28 decoder layers ($blk.0$ through $blk.27$) + Output Norm on the **NVIDIA GeForce GT 750M (Kepler GK107, 2 GB GDDR5)** under **Mesa Rusticl (OpenCL 3.0)** compared against an 8-thread **Intel Core i7-4870HQ (Haswell AVX2 + FMA3)** CPU reference.

---

## 🎯 Architecture & Scaling Law

1. **Dynamic GGUF Indexer in C:**
   - Full header and metadata parsing directly in native C (`parse_gguf_tensors`).
   - Dynamically resolves byte offsets for all 339 tensors with robust `read_exact_at()` handling `EINTR` signal retries and EOF verification.
2. **Full Model Residency in GT 750M VRAM:**
   - 28 Layers $\times$ 25.10 MB = **702.80 MB** weights.
   - 28 Layers $\times$ 8.00 MB = **224.00 MB** KV cache ($T_{\max}=4096$).
   - Dynamic workspace: strictly **0.33 MB** (reused in-place across all 28 layers).
   - Output Norm (`output_norm.weight`): **6 KB** (1536 floats).
   - **Total VRAM Allocated:** **927.13 MB** (< 1 GB, comfortably fitting into the 2048 MB VRAM of the GT 750M).
3. **Autoregressive Cache Continuity:**
   - Token 0 evaluated at position $\text{pos}=0$ ($T=1$).
   - Token 1 evaluated at position $\text{pos}=1$ ($T=2$), appending and reading cached KV keys and values across all 28 layers.

---

## 📊 Measured Empirical Results (Physical Hardware Benchmark)

- **GPU Device:** NVIDIA GeForce GT 750M (Kepler GK107, 384 cores, 2 GB GDDR5) via Mesa Rusticl OpenCL 3.0 (`NVE7`).
- **CPU Reference:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell AVX2 + FMA3, 8 OpenMP threads).
- **Model:** `qwen2.5-coder-1.5b-instruct-q4_0.gguf` (1017 MB official GGUF).

### 1. Full 28 Layers + Output Norm (`test_28_layers_real`)

| Step | Sequence Position / Context | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Max Absolute Diff | Relative $L_2$ Error | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | $\text{pos}=0$ ($T=1$) | $1586.20\text{ ms}$ | **$881.77\text{ ms}$** | **$1.80\times$** 🚀 | $1.14 \times 10^{-3}$ | **$1.44 \times 10^{-5}$** | **$1.000000$** |
| **Token 1** | $\text{pos}=1$ ($T=2$) | $1400.55\text{ ms}$ | **$913.64\text{ ms}$** | **$1.53\times$** 🚀 | $1.26 \times 10^{-4}$ | **$4.84 \times 10^{-6}$** | **$1.000000$** |

### 2. Phase B0: Isolated Q6_K GEMV Kernel (`test_gemv_q6_k`)

Canonical Basis Unit Tests ($e_i$) and dense random latent projection ($M=4096, K=1536$) comparing OpenCL `gemv_q6_k` vs official GGML reference:

| Test Vector | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Max Absolute Diff | Relative $L_2$ Error | Cosine Similarity | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Canonical Bases $e_i$** | — | — | — | **$0.0000\text{ e+}00$** | **$0.0000\text{ e+}00$** | **$1.000000$** | **PASS (100%)** |
| **Dense Latent ($x \sim \mathcal{N}(0, 1)$)** | $13.71\text{ ms}$ | **$4.82\text{ ms}$** | **$2.84\times$** 🚀 | $2.62 \times 10^{-6}$ | **$6.93 \times 10^{-7}$** | **$1.000000$** | **PASS** |

### 3. Isolated 2-Layer Subsystem (`test_real_layers`)

| Step | Sequence Position / Context | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Relative $L_2$ Error | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | $\text{pos}=0$ ($T=1$) | $190.09\text{ ms}$ | **$105.54\text{ ms}$** | **$1.80\times$** 🚀 | **$2.62 \times 10^{-6}$** | **$1.000000$** |
| **Token 1** | $\text{pos}=1$ ($T=2$) | $297.30\text{ ms}$ | **$89.55\text{ ms}$** | **$3.32\times$** 🚀 | **$6.12 \times 10^{-6}$** | **$1.000000$** |

---

## 🔬 Key Engineering Insights

1. **Flawless End-to-End Numerical Stability:**
   - After traversing all 28 consecutive decoder layers ($28 \times 11 = 308$ kernel launches per token) and final RMSNorm, the Relative $L_2$ error remains below **$1.45 \times 10^{-5}$** with **Cosine Similarity = 1.000000**.
   - Zero NaNs, zero Infs, confirming that quantization error does not accumulate uncontrollably across 28 layers.
2. **Generation Rate on GT 750M:**
   - Full 28-layer autoregressive forward step executes in **$\approx 880 - 913\text{ ms}$** on the GT 750M, achieving **$\approx 1.1\text{ tokens/s}$** on hardware from 2014.
   - Consistently outperforms the 8-thread Haswell CPU baseline ($1.40 - 1.58\text{ s}$ per token).
3. **Ultra-Fast Weights Loading & Transfer:**
   - 702.8 MB loaded from SSD via `pread()` in **495 ms**.
   - Entire 28-layer model transferred to GT 750M VRAM over PCIe in **1103 ms**.

---

## 🛠️ Build & Run

```bash
# Build both the 2-layer and 28-layer binaries:
make clean && make

# Run 28-layer full model validation:
RUSTICL_ENABLE=nouveau ./test_28_layers_real

# Run 2-layer isolated subsystem validation:
RUSTICL_ENABLE=nouveau ./test_real_layers
```
