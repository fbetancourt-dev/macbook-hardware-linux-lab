# Scientific Computing & Local AI Acceleration Lab

High-performance physical simulations, control theory ensembles, and local Large Language Model (LLM) inference accelerated on the discrete **NVIDIA GeForce GT 750M (384 Kepler CUDA Cores / 2 GB GDDR5)** using **Mesa Rusticl (OpenCL 3.0)** on Linux.

---

## 🧭 Modules Directory

| Module | Category | Description | Primary Interfaces |
| :--- | :--- | :--- | :--- |
| **[`cartpole/`](cartpole/)** | Dynamics & Control | 1,024-pendulum parallel GPU ensemble + 60 FPS interactive steering visualizer | OpenCL C-MEX + MATLAB Simulink |
| **[`thermal-control/`](thermal-control/)** | PDE & Process Control | 2D heat diffusion PDE ($256 \times 256$ grid) with multi-zone anti-windup PID | OpenCL C-MEX + MATLAB Simulink |
| **[`simulink-opencl-bridge/`](simulink-opencl-bridge/)** | Architecture | Generic zero-allocation Level-2 C-MEX gateway architecture for Simulink | OpenCL 3.0 + Simulink Engine |
| **[`opencl-basics/`](opencl-basics/)** | Baseline Kernels | Standalone C implementations of N-Body gravitation and 2D heat diffusion | Native C99 + OpenCL 3.0 |
| **[`local-llm/`](local-llm/)** | Local AI / LLMs | `llama.cpp` OpenCL server + Qwen2.5-Coder autocompletion daemon for Continue | OpenAI API (`:8080`) + systemd |

---

## ⚡ Prerequisites & Setup

All modules in this directory rely on **Mesa Rusticl OpenCL 3.0**. Please review the central setup guide:
* [`../docs/opencl-rusticl-setup.md`](../docs/opencl-rusticl-setup.md): Complete instructions on package installation, enabling `RUSTICL_ENABLE=nouveau`, and isolating Vulkan workloads.
