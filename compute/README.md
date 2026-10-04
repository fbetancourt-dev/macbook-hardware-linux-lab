# Scientific Computing & Local AI Acceleration Lab

High-performance physical simulations, control theory ensembles, and local Large Language Model (LLM) inference on the **Apple MacBook Pro (Retina, 15-inch, Mid 2014)** running Linux.

---

## 🧭 Modules Directory

| Module | Category | Hardware Target | Description | Primary Interfaces |
| :--- | :--- | :--- | :--- | :--- |
| **[`gguf-real-weights-loader/`](gguf-real-weights-loader/)** | Full LLM Engine | NVIDIA GT 750M (1110 MB VRAM) | Complete 28-layer + LM Head resident Qwen2.5-Coder-1.5B pipeline ($2.23\times$ faster decode vs 8T CPU, 1.1 t/s) with `qwen_server` Unix socket daemon & `ask-qwen` transactional memory CLI | OpenCL 3.0 + Unix Socket |
| **[`ffn-swiglu-resident/`](ffn-swiglu-resident/)** | Transformer Engine | NVIDIA GT 750M (22 MB VRAM) | Resident SwiGLU FFN pipeline ($2.00\times$ GPU compute speedup, 23.16 ms) | OpenCL 3.0 + C99 AVX2 |
| **[`attention-gqa-resident/`](attention-gqa-resident/)** | Transformer Engine | NVIDIA GT 750M (OpenCL FP32) | Resident Grouped Query Attention (GQA) with multi-head KV projection | OpenCL 3.0 + C99 AVX2 |
| **[`gemv-kepler-benchmark/`](gemv-kepler-benchmark/)** | Microbenchmarks | NVIDIA GT 750M vs Haswell AVX2 | Warp-coalesced Q4_0 GEMV with FP32 accumulation ($6.57\times$ GPU speedup) | OpenCL 3.0 + C99 AVX2 |
| **[`cartpole/`](cartpole/)** | Dynamics & Control | NVIDIA GT 750M (OpenCL FP32) | 1,024-pendulum parallel GPU ensemble + 60 FPS interactive steering visualizer | OpenCL C-MEX + MATLAB Simulink |
| **[`thermal-control/`](thermal-control/)** | PDE & Process Control | NVIDIA GT 750M (OpenCL FP32) | 2D heat diffusion PDE ($256 \times 256$ grid) with multi-zone anti-windup PID | OpenCL C-MEX + MATLAB Simulink |
| **[`simulink-opencl-bridge/`](simulink-opencl-bridge/)** | Architecture | NVIDIA GT 750M (Mesa Rusticl) | Generic zero-allocation Level-2 C-MEX gateway architecture for Simulink | OpenCL 3.0 + Simulink Engine |
| **[`opencl-basics/`](opencl-basics/)** | Baseline Kernels | NVIDIA GT 750M (OpenCL 3.0) | Standalone C implementations of N-Body gravitation and 2D heat diffusion | Native C99 + OpenCL 3.0 |
| **[`local-llm/`](local-llm/)** | Local AI / LLMs | Intel Haswell (AVX2 / FMA3) | Low-latency autocompletion daemon for Continue + DeepSeek/Qwen benchmarks & legacy llama-server setup | OpenAI API (`:8080`) + Ollama |

---

## ⚡ Architectural Division: GPU vs CPU

* **GPU (NVIDIA GeForce GT 750M Kepler GK107):** Hosts high-throughput **single-precision (FP32)** scientific simulations alongside the resident layers of the **Qwen2.5-Coder-1.5B transformer model** (28 Decoder Layers + Output Norm + Full Q6_K LM Head + KV Cache) within an **1110 MB static VRAM budget** (< 55% of 2 GB VRAM; input embeddings prepared on CPU host). In autoregressive decode benchmarks, it achieves **$961\text{ ms/tok}$** ($1.04 - 1.1\text{ t/s}$), running **$2.23\times$ faster** than official `llama.cpp` on 8 Haswell CPU threads ($2143\text{ ms}$).
* **CPU (Intel Core i7-4870HQ Haswell):** Used for token embedding dequantization/lookup before PCIe dispatch, large prompt batch prefill, fast host tokenizer execution, OS virtualization, and secondary LLM fallback.

For in-depth architectural and driver analysis, see:
* [`../docs/KEPLER_GPGPU_LLM_ARCHITECTURE.md`](../docs/KEPLER_GPGPU_LLM_ARCHITECTURE.md): **Whitepaper: Why, How, and What We Achieved** with Kepler GPGPU LLM Acceleration.
* [`../docs/opencl-rusticl-setup.md`](../docs/opencl-rusticl-setup.md): Package installation, `RUSTICL_ENABLE=nouveau`, and isolating Vulkan workloads.
