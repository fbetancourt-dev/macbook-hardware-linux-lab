# Engineering Whitepaper: Kepler GPGPU LLM Acceleration
## Executing Modern 1.5B Transformers on Legacy 2014 Silicon (NVIDIA GT 750M, Mesa Rusticl & OpenCL 3.0)

**Project:** `macbook-hardware-linux-lab`  
**Author:** Francisco Betancourt (`fbetancourt-dev`)  
**Hardware Target:** Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)  
**GPU:** NVIDIA GeForce GT 750M Mac Edition (Kepler GK107, 384 CUDA Cores, 2 GB GDDR5)  
**Host CPU:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell Crystalwell, AVX2 + FMA3, 16 GB DDR3L)  
**OS & Graphics Stack:** Ubuntu 24.04 LTS (Kernel 6.8+), Wayland / GNOME Shell, Mesa Rusticl (OpenCL 3.0) over `nouveau`  

<p align="center">
  <img src="../assets/kepler_transformer_prime.jpg" alt="Optimus Prime Kepler Transformer Engine" width="100%"/>
  <br>
  <em>Figure 1: The Kepler Transformer Engine — Fusing Optimus Prime robotics aesthetic with real NVIDIA GT 750M GPGPU silicon execution.</em>
</p>

---

## 1. The "Why" (Context & Motivation)

### 1.1 The Silicon Obsolescence Trap
In the mainstream AI ecosystem (PyTorch, TensorFlow, Ollama, llama.cpp), hardware older than NVIDIA Maxwell (Compute Capability 5.0) is officially classified as dead silicon:
1. **CUDA Deprecation:** NVIDIA removed Kepler support (Compute Capability 3.0 / 3.5) starting in CUDA 11 and completely eliminated it in CUDA 12. Standard LLM runtimes built for modern Linux distributions require CUDA 12+, immediately disabling GPU offload on Kepler hardware.
2. **Proprietary Driver Incompatibility:** The legacy NVIDIA proprietary driver (`nvidia-470xx`) is incompatible with modern Linux kernels (6.8+) and triggers severe stability failures, memory corruption, and display crashes under Wayland and GNOME Shell.
3. **Open-Source Driver Realities:** The open-source `nouveau` driver provides exceptional Wayland display stability (especially with our patched Mesa Kepler driver tier), but it does not support the proprietary NVIDIA CUDA runtime (`libcuda.so` or `nvidia-uvm`).
4. **VRAM Constraints:** The GT 750M provides strictly **2,048 MB (2 GB) of GDDR5 VRAM**. Standard quantization paradigms easily exceed 2 GB when loading modern model weights alongside dynamic KV caches and scratchpad tensors.

### 1.2 The Engineering Challenge
The standard response to this dilemma is to abandon the discrete GPU and execute all local AI workloads exclusively on the CPU. However:
- Sustained 8-thread AVX2 workloads on the Haswell CPU generate thermal loads exceeding 85°C–90°C, triggering aggressive thermal throttling down to 1.8–2.0 GHz.
- Leaving the 384 Kepler CUDA cores completely unutilized leaves untapped parallel compute capacity.

**Our Core Engineering Hypothesis:**  
*Can we bypass the proprietary CUDA stack entirely, leverage Mesa's modern OpenCL 3.0 Rusticl driver over Nouveau, and design a custom, zero-waste C/OpenCL transformer engine that fits resident within 2 GB of VRAM, beats 8 CPU threads, and preserves 100% desktop stability?*

---

## 2. The "How" (Implementation & Architecture)

To turn this hypothesis into reality, we engineered a ground-up inference engine implemented in native C99 and OpenCL 3.0 located at [`compute/gguf-real-weights-loader/`](../compute/gguf-real-weights-loader/).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Space / CLI Layer                          │
│   • ask-qwen CLI client (~/.local/bin/ask-qwen)                        │
│   • Transactional memory (profile.md, facts.md, memory_manifest.json)  │
│   • Staged prepare + atomic replacement (facts.tmp ➔ fsync ➔ rename)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Unix Domain Socket
                                    │ (/run/user/1000/qwen.sock)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    qwen_server Daemon (Host CPU)                       │
│   • Native C99 GGUF Parser (inspects all 339 tensor offsets)           │
│   • Token Embedding Dequantization (~445 MB kept in CPU RAM)           │
│   • PCIe Dispatch of 1536-float embedding vector per token             │
│   • Frozen Base KV Cache Manager (SET_BASE prefilled once)             │
│   • Per-token command queue drainage (clFinish)                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ OpenCL 3.0 API / PCIe Gen3 x16
                                    │ Mesa Rusticl (RUSTICL_ENABLE=nouveau)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             NVIDIA GeForce GT 750M VRAM (Strict 1,110 MB Budget)       │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 28 Decoder Layers (Q4_0 Weights): 702.80 MB                     │   │
│   │   • Q, K, V Projections (Warp-coalesced GEMV)                  │   │
│   │   • Grouped Query Attention (GQA, 12 Q heads, 2 KV heads)      │   │
│   │   • FFN SwiGLU (Gate, Up, and Down Projections)                │   │
│   │   • RMSNorm (FP32 normalization)                               │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │ Autoregressive KV Cache (T_max = 4096, 28 layers): 224.00 MB   │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │ Language Model (LM) Head (151,936 rows of Q6_K): 182.57 MB      │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │ Output RMSNorm (1536 FP32 floats): 0.006 MB                    │   │
│   ├────────────────────────────────────────────────────────────────┤   │
│   │ Shared Dynamic Scratchpad Workspace: 0.33 MB                   │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│   Total GPU Resident: 1,110 MB (< 55% of 2,048 MB physical VRAM)       │
│   Headroom for Desktop / Wayland Buffers: ~938 MB                      │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The Silicon Limitation: Emulating FP16 on Kepler
The Kepler GK107 architecture lacks native FP16 arithmetic hardware (`cl_khr_fp16` is unavailable). Running unquantized FP32 weights would require >3.5 GB of VRAM, exceeding physical memory.
- **The Solution:** We implemented custom OpenCL kernels that read quantized nibbles (**Q4_0** for decoder layers and **Q6_K** for the LM Head) directly from GDDR5 VRAM and dequantize them on-the-fly into 32-bit floating-point registers using SIMD integer unpacks and software FP32 multiply-accumulate operations.

### 2.2 Strict VRAM Budgeting (1,110 MB / 2,048 MB)
Running a full 1.5-billion-parameter LLM on a 2 GB GPU requires surgical memory management:
1. **Weight Residency:** The 28 decoder layers ($blk.0$ through $blk.27$) are converted to Q4_0 and kept resident in VRAM:
   $$\text{Layer Weights} = 28 \times 25.10\text{ MB} = 702.80\text{ MB}$$
2. **Output Projection:** The vocabulary projection (LM Head) contains 151,936 rows. Quantized to Q6_K, it requires strictly **182.57 MB**.
3. **KV Cache:** Sized for an autoregressive context window of $T_{\max} = 4096$:
   $$\text{KV Cache} = 28 \text{ layers} \times 8.00\text{ MB} = 224.00\text{ MB}$$
4. **The Embedding Split Strategy:** Rather than loading the dense embedding matrix (`token_embd.weight`, ~445 MB) into VRAM, we store it in host CPU RAM (where 16 GB is available). When a token is evaluated, the host extracts and dequantizes the 1,536-dimensional float vector and streams it to the GPU via PCIe Gen3 x16 in **$< 15\ \mu\text{s}$**. This single decision saved nearly 0.5 GB of VRAM.
5. **Total Allocation:** The GPU maintains strictly **1,110 MB resident**, leaving over **938 MB of headroom** for Wayland compositing buffers and desktop windows.

### 2.3 Command Ring Stabilization
During large prompt prefills (e.g. 218 system prompt tokens dispatching over 67,000 OpenCL kernel launches across 28 layers), standard OpenCL command submission can saturate the `nouveau` DMA ringbuffer. We mitigated this by enforcing per-token command drainage (`clFinish(queue)`), eliminating kernel ring lockups and fence timeouts (`dma_fence_default_wait`).

### 2.4 Production Persistence & Transactional Memory
We paired the engine with a client CLI (`ask-qwen`) and daemon (`qwen_server`):
- **Frozen Base KV Cache:** The system prompt and assistant profile are prefilled into the KV cache once (`SET_BASE`). Subsequent user queries append after `base_pos`, cutting conversational latency to zero prefill overhead.
- **Transactional Consistency:** Context additions (`facts.md`) utilize staged two-phase commits (`facts.tmp` $\to$ `fsync` $\to$ atomic rename $\to$ directory `fsync`) alongside POSIX advisory file locking (`fcntl.flock`), ensuring zero memory corruption upon sudden system reboot.

---

## 3. The "What We Achieved" (Empirical Results)

The system was benchmarked against the official CPU reference implementation (8 OpenMP threads with AVX2 + FMA3 on the Haswell i7-4870HQ) running the official `qwen2.5-coder-1.5b-instruct-q4_0.gguf` weights.

### 3.1 Numerical Correctness & Exactness
Traversing 28 consecutive decoder layers ($28 \times 11 = 308$ kernel dispatches per token) through RMSNorm and Softmax poses severe numerical drift risks on custom OpenCL kernels:

| Metric | Measured Value | Standard Required | Validation Status |
| :--- | :---: | :---: | :---: |
| **Cosine Similarity vs CPU** | **1.000000** | $\ge 0.9999$ | **PERFECT** ✅ |
| **Relative $L_2$ Error (28 Layers)** | **$1.44 \times 10^{-5}$** | $< 1.0 \times 10^{-4}$ | **PASS** ✅ |
| **ArgMax Top-1 Logit Match** | **Identical (Token 16: `'1'`)** | Exact Match | **PASS** ✅ |
| **Top-5 Predicted Token Ordering** | **100% Identical** | Top-5 Matching | **PASS** ✅ |
| **Numerical Integrity** | **0 NaNs, 0 Infs** | Strictly Finite | **STABLE** ✅ |

### 3.2 Speedup & Throughput
Evaluating the forward pass of the complete 28-layer transformer:

1. **First-Token Forward Pass ($T=1$, pos=0):**
   - **Intel Haswell CPU (8 Threads AVX2):** $1586.20\text{ ms}$
   - **NVIDIA GT 750M (Kepler OpenCL):** **$881.77\text{ ms}$**
   - **Speedup:** **$1.80\times$ faster on GPU** 🚀

2. **Autoregressive Token-by-Token Decode:**
   - **Official `llama.cpp` (`llama-completion`, 8 CPU Threads):** $2143.42\text{ ms/token}$ ($0.47\text{ tokens/s}$)
   - **Our GT 750M Engine (`generate_stream`):** **$961.51\text{ ms/token}$ ($1.04\text{ tokens/s}$)**
   - **Speedup:** **$2.23\times$ faster decode on GT 750M** 🚀
   - **Sequence Integrity:** Token-for-token identical output to official `llama.cpp`.

3. **Thermal and System Stability:**
   - Wayland and GNOME Shell maintain flawless 60 FPS operation with zero screen tearing or compositor restarts.
   - Operating temperature remains controlled, avoiding the aggressive thermal ceiling reached when stressing all 8 CPU threads simultaneously.

---

## 4. Key Takeaways & Significance

1. **Hardware Recycling at the Frontier:** Proved that modern Generative AI models are not exclusively reserved for latest-generation hardware. A laptop discrete GPU from 2014 can run state-of-the-art 1.5B coding LLMs with full mathematical accuracy.
2. **Open-Source Graphics Triumph:** Mesa Rusticl and Nouveau have reached production-grade maturity, capable of executing complex multi-kernel OpenCL 3.0 pipelines with strict memory isolation.
3. **Engineering Craftsmanship:** By combining low-level C99 GGUF indexing, custom SIMD-emulated OpenCL quantization kernels, and intelligent PCIe memory partitioning, we extracted modern utility from hardware that mainstream industry vendors abandoned a decade ago.
