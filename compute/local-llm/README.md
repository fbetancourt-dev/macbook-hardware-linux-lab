# Local LLM Inference via Mesa Rusticl (OpenCL 3.0) on Kepler GT 750M

Production-grade deployment and hardware acceleration of local Large Language Models (LLMs) on legacy **NVIDIA Kepler (GK107 / GT 750M 2GB GDDR5)** architecture using **Mesa Rusticl (OpenCL 3.0)** and `llama.cpp` (upstream commit `5fc4f3c`), providing OpenAI-compatible autocompletion for **Continue in VS Code**.

---

## 🎯 Background & The "Kepler Trap"

The NVIDIA GeForce GT 750M (Kepler Compute Capability 3.0) presents unique challenges on modern Linux systems:
1. **CUDA Deprecation:** NVIDIA removed Kepler support in modern CUDA toolchains (CUDA 11/12). Modern PyTorch and proprietary wheels cannot run on Kepler.
2. **Proprietary Driver Incompatibility:** Legacy `nvidia-390` or `nvidia-470` drivers cannot compile on modern Linux kernels (6.x/7.x) and completely break Wayland desktop compositors.
3. **Mesa Vulkan (NVK) Instability:** Mesa's experimental Kepler Vulkan driver (`nvk`) suffers from command stream serialization bugs under compute load. When running heavy shaders alongside desktop apps, the kernel driver throws:
   ```
   nouveau 0000:01:00.0: kernel rejected pushbuf: No such device
   nouveau 0000:01:00.0: fifo: ch 6 [mutter] killed
   ```
   Crashing GNOME Shell and terminating the user session.

---

## 💡 The Solution: Mesa Rusticl OpenCL 3.0

**Mesa Rusticl** is the modern OpenCL 3.0 implementation written in Rust within Mesa. It interfaces cleanly with the Nouveau Gallium3D driver and provides stable, hardware-accelerated compute shaders directly on Kepler silicon without touching experimental Vulkan compute paths.

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    VS Code + Continue                       │
 │             (Code autocompletion & assistance)              │
 └──────────────────────────────┬──────────────────────────────┘
                                │ HTTP / OpenAI API (Port 8080)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │       llama.cpp Server Daemon (llama-server.service)        │
 │              - Model: Qwen2.5-Coder 1.5B Q4_K_M             │
 │              - Process Resident Memory: ~809 MB RSS         │
 │              - Raw Model Weights: 986 MB                    │
 └──────────────────────────────┬──────────────────────────────┘
                                │ GGML OpenCL Compute Backend
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                 Mesa Rusticl (OpenCL 3.0)                   │
 │           (RUSTICL_ENABLE=nouveau / Gallium3D)              │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Hardware Execution (PCIe x16)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │        NVIDIA GeForce GT 750M Mac Edition (GK107)           │
 │        - 384 Kepler CUDA Cores                              │
 │        - 2,048 MB GDDR5 VRAM                                │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Native Application Structure (`~/Applications/llama.cpp/`)

Organized according to local system standards:
* **Source:** `~/Applications/llama.cpp/src/llama.cpp/` (commit `5fc4f3c`)
* **Build Flags:** `-DGGML_OPENCL=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release`
* **Binaries:** `~/Applications/llama.cpp/bin/` (`llama-cli`, `llama-server`, `llama-bench`, `libggml-opencl.so`, `libggml-cpu.so`)
* **Automation:** `~/Applications/llama.cpp/rebuild.sh`
* **Daemon Launcher:** `~/Applications/llama.cpp/start_llama_server.sh`
* **CLI Symlinks:** `~/.local/bin/llama-cli`, `~/.local/bin/llama-server`

---

## ⚡ Performance Benchmark

Benchmarked on `Qwen2.5-Coder-1.5B` executing code completion:

| Metric | Ollama (Vulkan NVK) | llama.cpp (Mesa Rusticl OpenCL + AVX2) | Improvement |
| :--- | :---: | :---: | :---: |
| **Generation Speed** | 2.78 tokens/sec | **11.5 - 12.6 tokens/sec** | **~4.5× faster** ⚡ |
| **Prompt Processing** | 2.78 tokens/sec | **18.3 - 37.3 tokens/sec** | **~6.5× - 13× faster** 🚀 |
| **Desktop Stability** | Crashed Mutter/GNOME | **No crashes observed across continuous requests** | 100% Stable |
| **VRAM / Process RSS** | ~1.6 GB | **~809 MB** | ~50% Less Memory |

---

## 🔧 Service Configuration

### 1. Ollama Vulkan Isolation (`~/.config/systemd/user/ollama.service`)
To ensure Ollama does not trigger Vulkan pushbuf rejections:
```ini
[Service]
Environment="OLLAMA_VULKAN=false"
```

### 2. llama-server Systemd Service (`~/.config/systemd/user/llama-server.service`)
```ini
[Unit]
Description=llama.cpp OpenCL Server for Continue
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
  - name: Qwen2.5-Coder 1.5B (llama.cpp OpenCL)
    provider: openai
    apiBase: http://127.0.0.1:8080/v1
    model: /home/fbetancourt/Applications/llama.cpp/models/qwen2.5-coder-1.5b-base.gguf
    roles:
      - autocomplete
```
