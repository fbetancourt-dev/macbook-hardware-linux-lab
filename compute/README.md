# Scientific Computing & Local AI Acceleration Lab

High-performance physical simulations, control theory ensembles, and local Large Language Model (LLM) inference on the **Apple MacBook Pro (Retina, 15-inch, Mid 2014)** running Linux.

---

## 🧭 Modules Directory

| Module | Category | Hardware Target | Description | Primary Interfaces |
| :--- | :--- | :--- | :--- | :--- |
| **[`ffn-swiglu-resident/`](ffn-swiglu-resident/)** | Transformer Engine | NVIDIA GT 750M (22 MB VRAM) | Resident SwiGLU FFN pipeline ($2.00\times$ GPU compute speedup, 23.16 ms) | OpenCL 3.0 + C99 AVX2 |
| **[`gemv-kepler-benchmark/`](gemv-kepler-benchmark/)** | Microbenchmarks | NVIDIA GT 750M vs Haswell AVX2 | Warp-coalesced Q4_0 GEMV with FP32 accumulation ($6.57\times$ GPU speedup) | OpenCL 3.0 + C99 AVX2 |
| **[`cartpole/`](cartpole/)** | Dynamics & Control | NVIDIA GT 750M (OpenCL FP32) | 1,024-pendulum parallel GPU ensemble + 60 FPS interactive steering visualizer | OpenCL C-MEX + MATLAB Simulink |
| **[`thermal-control/`](thermal-control/)** | PDE & Process Control | NVIDIA GT 750M (OpenCL FP32) | 2D heat diffusion PDE ($256 \times 256$ grid) with multi-zone anti-windup PID | OpenCL C-MEX + MATLAB Simulink |
| **[`simulink-opencl-bridge/`](simulink-opencl-bridge/)** | Architecture | NVIDIA GT 750M (Mesa Rusticl) | Generic zero-allocation Level-2 C-MEX gateway architecture for Simulink | OpenCL 3.0 + Simulink Engine |
| **[`opencl-basics/`](opencl-basics/)** | Baseline Kernels | NVIDIA GT 750M (OpenCL 3.0) | Standalone C implementations of N-Body gravitation and 2D heat diffusion | Native C99 + OpenCL 3.0 |
| **[`local-llm/`](local-llm/)** | Local AI / LLMs | Intel Haswell (AVX2 / FMA3) | Low-latency autocompletion daemon for Continue + DeepSeek/Qwen benchmarks & Kepler limits | OpenAI API (`:8080`) + Ollama |

---

## ⚡ Architectural Division: GPU vs CPU

* **GPU (NVIDIA GeForce GT 750M Kepler):** Dedicated to **single-precision (FP32)** scientific computing and quantized Transformer pipelines (GEMV & SwiGLU FFN). Delivers $2.00\times$ compute speedup over Haswell CPU when weights remain resident in VRAM.
* **CPU (Intel Core i7-4870HQ Haswell):** Dedicated to **full pipeline Local LLM inference** via AVX2/FMA3 256-bit vector SIMD for everyday pair programming in Continue.

For OpenCL setup and driver configuration, see:
* [`../docs/opencl-rusticl-setup.md`](../docs/opencl-rusticl-setup.md): Package installation, `RUSTICL_ENABLE=nouveau`, and isolating Vulkan workloads.
