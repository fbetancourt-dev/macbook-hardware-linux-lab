# Local LLM Inference & Autocomplete on Haswell AVX2 & Kepler Hardware

Production-grade deployment and hardware acceleration of local Large Language Models (LLMs) on the **Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)**, combining **Intel Haswell AVX2 + FMA3 SIMD** for low-latency coding assistance in **Continue (VS Code)**, and establishing empirical hardware constraints on the legacy **NVIDIA Kepler (GK107 / GT 750M 2GB)** GPU.

---

## 🎯 Executive Hardware Architecture & Workload Division

A critical finding in this lab is the strict architectural boundary between CPU SIMD and legacy GPU compute:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                            VS Code + Continue                               │
 │                       (AI-Assisted Pair Programming)                        │
 └──────────────────────┬──────────────────────────────┬───────────────────────┘
                        │ HTTP / OpenAI API            │ HTTP / Ollama API
                        │ (Port 8080 - Autocomplete)   │ (Port 11434 - Chat)
                        ▼                              ▼
 ┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
 │  llama-server (Systemd User Daemon)  │ │      Ollama Service Daemon         │
 │  - Model: Qwen2.5-Coder 1.5B Q4_K_M  │ │  - Models: deepseek-coder:1.3b/6.7b│
 │  - Engine: libggml-cpu-haswell.so    │ │  - Environment: OLLAMA_VULKAN=false│
 └──────────────────┬───────────────────┘ └─────────────────┬──────────────────┘
                    │                                       │
                    ▼                                       ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                Intel Core i7-4870HQ Haswell CPU (4C / 8T)                   │
 │                - 256-bit AVX2 + FMA3 SIMD Execution Units                   │
 │                - Throughput: 10 - 12 tokens/sec (sub-second completions)    │
 │                - Resident Memory: ~809 MB RSS                               │
 └─────────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │             NVIDIA GeForce GT 750M (Kepler GK107, 2 GB GDDR5)               │
 │             - Role: Dedicated to FP32 Scientific Computing (Simulink)       │
 │             - Hardware Limitation: Lacks cl_khr_fp16 (No FP16 ALUs)         │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 The Kepler OpenCL Investigation: Empirical Proof

### Why Modern LLMs Require CPU AVX2 Instead of Kepler GPU
While the GT 750M excels at single-precision (**FP32**) scientific computing (e.g. 1,024-pendulum Cart-Pole ensembles and 2D thermal PDEs), it **cannot execute modern `llama.cpp` quantization kernels**.

We empirically patched `ggml/src/ggml-opencl/ggml-opencl.cpp` to bypass device whitelists and forced detection of the Nouveau/Mesa Rusticl driver (`NVE7`):

1. **Detection Succeeded:**
   ```text
   llama-bench --list-devices
   GPUOpenCL: NVE7 (2005 MiB, 981 MiB free)
   ```
2. **JIT Compilation Failed (Hardware Silicon Constraint):**
   When `llama.cpp` dispatched its quantized matrix-multiplication kernels (`Q4_K`, `Q5_K`), the Mesa Rusticl LLVM OpenCL compiler aborted:
   ```text
   input.cl:1:26: warning: unsupported OpenCL extension 'cl_khr_fp16' - ignoring
   input.cl:138:14: error: declaring variable of type '__private half' is not allowed
   input.cl:140:18: error: use of undeclared identifier 'convert_half'
   input.cl:160:16: error: unknown type name 'half4'
   ```
3. **Engineering Root Cause:**
   * Modern LLM inference engines rely on native 16-bit floating point arithmetic (`half` / `half4`) and subgroup operations (*warp shuffles*) to unpack quantized weights efficiently.
   * NVIDIA Kepler GK107 (2012–2014) is a pure **FP32-only** microarchitecture at the silicon level.
   * Consequently, the Intel Haswell CPU with **AVX2 + FMA3 (256-bit SIMD)** provides far superior performance and 100% stability.

---

## ⚡ Comparative Coding Benchmarks (Haswell CPU)

We subjected local models to an identical engineering challenge: implementing an accurate **Runge-Kutta 4th-Order (RK4)** numerical ODE integrator in Python with static typing, docstrings, and a damped harmonic oscillator simulation.

| Metric | DeepSeek-Coder 6.7B (Ollama CPU) | DeepSeek-Coder 1.3B (CPU AVX2) | Qwen2.5-Coder 1.5B (llama-server) |
| :--- | :---: | :---: | :---: |
| **Generation Speed** | 2.43 tokens/sec | **9.71 tokens/sec** | **11.5 - 12.6 tokens/sec** ⚡ |
| **Prompt Eval Speed** | 12.88 tokens/sec | **32.46 tokens/sec** | **37.3 tokens/sec** 🚀 |
| **Wall Clock Time (512 tok)** | 220.46 s (~3.67 min) | **59.44 s (< 1 min)** | **~42 s** |
| **RAM Footprint (RSS)** | ~4.2 GB | ~850 MB | **~809 MB** |
| **Implementation Correctness** | Syntax error in $k_3$ formula (`t + 0 dt`) | **100% correct**, zero syntax errors | **100% correct**, clean typing |
| **Ideal Role in Continue** | Deep architectural chat (on-demand) | Ghost-text & Tab autocomplete | **Ghost-text & Tab autocomplete** |

> **Key Takeaway:** For interactive pairing in Continue (VS Code), **latency is king**. Compact models ($\le 1.5$B) running on Haswell AVX2 deliver immediate sub-second completions without breaking developer flow.

---

## 🛠️ Native Application Structure (`~/Applications/llama.cpp/`)

Organized according to system out-of-source standards:
* **Source Tree:** `~/Applications/llama.cpp/src/llama.cpp/` (commit `5fc4f3c`)
* **Build Configuration:**
  ```bash
  cmake -B build -S . \
    -DGGML_OPENCL=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release
  ```
* **Installed Binaries:** `~/Applications/llama.cpp/bin/` (`llama-cli`, `llama-server`, `llama-bench`, `libggml-cpu-haswell.so`)
* **Automated Rebuild Script:** `~/Applications/llama.cpp/rebuild.sh`
* **Daemon Runner:** `~/Applications/llama.cpp/start_llama_server.sh`
* **User Binaries:** Symlinked in `~/.local/bin/` for direct terminal access.

---

## 🔧 Service Configuration

### 1. Ollama Vulkan Isolation (`~/.config/systemd/user/ollama.service`)
Prevents Nouveau kernel driver FIFO pushbuf stalls:
```ini
[Service]
Environment="OLLAMA_VULKAN=false"
```

### 2. llama-server Systemd Service (`~/.config/systemd/user/llama-server.service`)
```ini
[Unit]
Description=llama.cpp Haswell AVX2 Server for Continue
After=network.target

[Service]
Type=simple
ExecStart=/home/fbetancourt/Applications/llama.cpp/start_llama_server.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

### 3. Continue Configuration (`~/.continue/config.yaml`)
```yaml
models:
  - name: Qwen2.5-Coder 1.5B (llama.cpp AVX2)
    provider: openai
    apiBase: http://127.0.0.1:8080/v1
    model: /home/fbetancourt/Applications/llama.cpp/models/qwen2.5-coder-1.5b-base.gguf
    roles:
      - autocomplete

  - name: DeepSeek-Coder 6.7B (Ollama Chat)
    provider: ollama
    apiBase: http://127.0.0.1:11434
    model: deepseek-coder:6.7b
    roles:
      - chat
```
