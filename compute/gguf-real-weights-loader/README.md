# Real Weights GGUF Loader & Transformer Decoder Validation (Qwen2.5-Coder-1.5B)

Direct hardware execution and numerical validation of real model weights loaded from an official GGUF model file (**Qwen2.5-Coder-1.5B-Instruct-Q4_0.gguf**) running across consecutive decoder layers ($blk.0$ and $blk.1$) on the **NVIDIA GeForce GT 750M (Kepler GK107, 2 GB GDDR5)** under **Mesa Rusticl (OpenCL 3.0)** compared against an 8-thread **Intel Core i7-4870HQ (Haswell AVX2 + FMA3)** CPU reference.

---

## 🎯 Scope & Architecture

1. **Direct GGUF Binary Loading (`pread`):**
   - Direct extraction of tensors from `/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf` using precise byte offsets calculated from the GGUF header without full-model host memory overhead.
   - Header metadata verified via `inspect_model.py` and `validate_real_layers.py`.
   - Native `Q4_0` quantized weight matrices ($1536 \times 1536$, $1536 \times 256$, $1536 \times 8960$, $8960 \times 1536$) with FP16 scales and 4-bit signed nibbles.
   - Native `F32` RMSNorm weights (`attn_norm.weight`, `ffn_norm.weight`) and QKV biases (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`).
2. **Real Embedding Vector Dequantization:**
   - Real token embedding table (`token_embd.weight`, shape $1536 \times 151936$ in `Q4_0`).
   - Tokens 0 and 1 are extracted and dequantized into real continuous FP32 activation vectors ($D=1536$).
3. **Consecutive Multi-Token Autoregressive Stepping:**
   - **Token 0:** Evaluated at sequence position $\text{pos}=0$ (Context $T=1$).
   - **Token 1:** Evaluated at sequence position $\text{pos}=1$ (Context $T=2$) utilizing the cached Key/Value states from Token 0.

---

## 📊 Measured Empirical Results (Physical Hardware)

- **GPU Device:** NVIDIA GeForce GT 750M (Kepler GK107, 384 cores, 2 GB GDDR5) via Mesa Rusticl OpenCL 3.0 (`NVE7`).
- **CPU Reference:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell AVX2 + FMA3, 8 OpenMP threads).
- **Model:** `qwen2.5-coder-1.5b-instruct-q4_0.gguf` (1017 MB official GGUF).

| Step | Sequence Position / Context | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Relative $L_2$ Error | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | $\text{pos}=0$ ($T=1$) | $190.09\text{ ms}$ | **$105.54\text{ ms}$** | **$1.80\times$** 🚀 | **$2.62 \times 10^{-6}$** | **$1.000000$** |
| **Token 1** | $\text{pos}=1$ ($T=2$) | $297.30\text{ ms}$ | **$89.55\text{ ms}$** | **$3.32\times$** 🚀 | **$6.12 \times 10^{-6}$** | **$1.000000$** |

---

## 🔬 Key Technical Insights

1. **Numerical Integrity on Real Production Weights:**
   - Zero NaNs, zero Infs, and mathematical identity between Kepler OpenCL kernels and AVX2 CPU reference:
     - Token 0: Relative $L_2 = 2.62 \times 10^{-6}$, Cosine Similarity = $1.000000$.
     - Token 1: Relative $L_2 = 6.12 \times 10^{-6}$, Cosine Similarity = $1.000000$.
2. **Autoregressive Cache Continuity:**
   - Token 1 correctly reads Token 0's KV state from persistent GPU VRAM KV cache, updating the attention distribution smoothly.
3. **Execution Throughput:**
   - 2 full Transformer decoder layers with real weights execute in **$89.55\text{ ms}$** on the GT 750M (**$3.32\times$ faster than 8-thread Haswell CPU**).

---

## 🛠️ Build & Run

```bash
# Verify model inspection:
python3 inspect_model.py
python3 validate_real_layers.py

# Compile native test runner:
make clean && make

# Run real weights validation on Kepler GT 750M:
RUSTICL_ENABLE=nouveau ./test_real_layers
```
