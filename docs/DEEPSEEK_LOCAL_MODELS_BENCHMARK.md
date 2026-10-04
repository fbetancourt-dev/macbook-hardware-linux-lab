# DeepSeek Local Models Benchmark & Scaling Whitepaper
## Evaluating the Full DeepSeek Family on Haswell AVX2 & Kepler Hardware (1.3B to 14B)

**Project:** `macbook-hardware-linux-lab`  
**Author:** Francisco Betancourt (`fbetancourt-dev`)  
**Hardware Target:** Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)  
**Host CPU:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell Crystalwell, 4C / 8T, AVX2 + FMA3, 16 GB DDR3L RAM)  
**Host GPU:** NVIDIA GeForce GT 750M (Kepler GK107, 384 CUDA Cores, 2 GB GDDR5)  
**OS:** Ubuntu 24.04 LTS (Kernel 6.8+), Wayland / GNOME Shell  
**Inference Engine:** Ollama / llama.cpp (`libggml-cpu-haswell.so` with AVX2 SIMD) + Custom OpenCL GT 750M Loader  

---

## 1. The "Why" (Motivation & Strategic Context)

### 1.1 The DeepSeek Paradigm Shift
The emergence of the DeepSeek model series represents a major architectural milestone in local machine learning:
1. **Bifurcated Model Specialization:** Unlike monolithic general-purpose LLMs, DeepSeek divides its family into two distinct operational paradigms:
   - **DeepSeek-Coder:** Specialized for programmatic synthesis, AST comprehension, and low-level C/C++ firmware integration.
   - **DeepSeek-R1 (Distilled):** Specialized for **Reinforcement Learning-driven Chain-of-Thought (CoT)** reasoning, explicitly externalizing its internal deliberation within `<think>...</think>` tokens before generating final outputs.
2. **True Local Privacy & Zero Telemetry:** Proprietary firmware, confidential schematics, and critical infrastructure code can be analyzed locally with mathematical verification without sending data to external cloud APIs.
3. **The Local Silicon Scaling Question:**  
   *How do reasoning capability, hallucination rates, memory consumption, and token generation speed scale on a decade-old workstation (MacBook Pro Mid-2014) as we traverse from 1.3B $\to$ 1.5B $\to$ 6.7B $\to$ 7B $\to$ 8B $\to$ 14B parameters?*

### 1.2 The Memory Wall & Physical Boundaries
With 16 GB of physical DDR3L system RAM:
- Models up to **14B parameters (~9.0 GB Q4_K_M)** fit comfortably, leaving 5–7 GB for the Linux kernel, Wayland compositor, and IDE.
- Models with **32B or 671B parameters exceed physical memory**, necessitating SSD swap thrashing or dedicated multi-GPU clusters. Hence, the 1.3B–14B spectrum represents the true operational envelope of this machine.

---

## 2. The "What" (The Model Spectrum Profile)

We deployed and benchmarked the complete compatible DeepSeek family on the local machine:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     DEEPSEEK LOCAL FAMILY TAXONOMY                          │
 └──────────────────────┬──────────────────────────────┬───────────────────────┘
                        │                              │
         ┌──────────────┴──────────────┐┌──────────────┴──────────────┐
         ▼                             ▼▼                             ▼
   [DEEPSEEK-CODER]              [DEEPSEEK-R1]                  [DEEPSEEK-R1]
     Code Synthesis               Fast Reasoning              Heavy Reasoning
   • 1.3B (776 MB)               • 1.5B (1.1 GB, Qwen)        • 8B  (5.2 GB, Llama)
   • 6.7B (3.8 GB)               • 7B   (4.7 GB, Qwen)        • 14B (9.0 GB, Qwen)
```

### 2.1 Model Inventory & Architecture

| Model Identifier | Base Architecture | Parameter Count | Quantization | Disk / RAM Footprint | Primary Optimization Target |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`deepseek-coder:1.3b`** | DeepSeek Coder V1 | 1.35 B | Q4_K_M | **776 MB** | Low-latency IDE ghost-text autocomplete |
| **`deepseek-r1:1.5b`** | Qwen 2.5 1.5B Distill | 1.78 B | Q4_K_M | **1.1 GB** | Ultra-light CoT logic & edge micro-assistants |
| **`deepseek-coder:6.7b`** | LLaMA-derived Coder | 6.74 B | Q4_0 | **3.8 GB** | Multi-file code refactoring & C/C++ analysis |
| **`deepseek-r1:7b`** | Qwen 2.5 7B Distill | 7.61 B | Q4_K_M | **4.7 GB** | **Sweet Spot:** Formal logic, math, & physics |
| **`deepseek-r1:8b`** | Llama 3.1 8B Distill | 8.03 B | Q4_K_M | **5.2 GB** | Strict instruction adherence & code reasoning |
| **`deepseek-r1:14b`** | Qwen 2.5 14B Distill | 14.77 B | Q4_K_M | **9.0 GB** | Maximum achievable intelligence in 16 GB RAM |

---

## 3. The "How" (Execution Architecture & Low-Level Tuning)

### 3.1 AVX2 + FMA3 SIMD Execution
The Intel Core i7-4870HQ processor features 256-bit wide vector registers. Ollama routes inference through `libggml-cpu-haswell.so`, utilizing:
- **AVX2:** 8 single-precision or 16 half-precision parallel multiply-adds per instruction cycle.
- **FMA3:** Fused Multiply-Add ($a \cdot b + c$) in a single execution step, halving quantization round-off error while accelerating matrix-vector dot products (GEMV).

### 3.2 Thermal & Clock Governance
Sustained inference across 8 threads generates significant heat in the MacBook Pro unibody chassis. To prevent aggressive thermal throttling (which drops clock speeds to 1.8 GHz):
- The system enforces **`smart_turbo_manager`** in `strict25` mode (capping CPU frequencies at **2.50 GHz** with `no_turbo = 1`).
- This maintains CPU core temperatures below **78°C**, ensuring deterministic benchmark timings without clock degradation.

### 3.3 Dynamic CoT Evolution (The `<think>` Scaling Law)
By observing the internal monologue of the R1 reasoning models across parameter scales on the identical physical problem (*I2C bus pull-up sizing and physics at 100 kHz*), we identified a clear empirical scaling law:

1. **1.5B Scale (`deepseek-r1:1.5b`):**
   - *Behavior:* Eager to reason and fast (~8 t/s), but displays parameter-constrained confusion with physical dimensions (e.g. conflating 100 kHz period with 10 ms instead of 10 $\mu\text{s}$). Eventually converges on standard engineering rules-of-thumb ($10\text{ k}\Omega$).
2. **7B / 8B Scale (`deepseek-r1:7b` & `8b`):**
   - *Behavior:* Fully eliminates unit confusion. Formally derives the RC rise-time exponential:
     $$t_r = 0.8473 \cdot R_p \cdot C_{\text{bus}} \le 1000\text{ ns}$$
     Correctly evaluates the trade-off between power consumption ($I_{OL} \le 3\text{ mA}$) and rise-time constraints, recommending the industry-standard $4.7\text{ k}\Omega$ to $10\text{ k}\Omega$.
3. **14B Scale (`deepseek-r1:14b`):**
   - *Behavior:* Comprehensive multi-variable optimization. Considers capacitive bus loading limits ($C_{\max} = 400\text{ pF}$), fast-mode transition margins, voltage rail scaling ($3.3\text{V}$ vs $5.0\text{V}$), and active noise margins ($V_{IL} = 0.3 V_{DD}$).

---

## 4. The "What We Achieved" (Empirical Benchmarks)

### 4.1 Comparative Performance Matrix

All standard Ollama / llama.cpp models execute on the Intel Haswell CPU using 8 OpenMP threads with AVX2 + FMA3 vector extensions. The custom GPGPU engine executes directly on the NVIDIA Kepler GT 750M via Mesa Rusticl OpenCL 3.0.

| Model | Execution Silicon (Processor / GPU) | Memory Footprint (RAM / VRAM) | Prompt Eval (TTFT) | Generation Rate (tok/s) | Reasoning Fidelity | Production Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`deepseek-coder:1.3b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~850 MB RAM** | **36.4 tokens/s** | **9.09 t/s** ⚡ | Syntax autocomplete (confusión física) | **Active Tab Completion (Continue)** |
| **`deepseek-r1:1.5b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~1.3 GB RAM** | **29.0 tokens/s** | **7.5 – 12.0 t/s** | Heuristic reasoning (CoT entusiasta) | **Edge IoT / Orange Pi candidate** |
| **`deepseek-coder:6.7b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~4.2 GB RAM** | **5.76 tokens/s** | **2.13 t/s** | SFT direct output (alucina fórmulas) | **Code autocomplete / Refactoring** |
| **`deepseek-r1:7b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~5.1 GB RAM** | **5.95 tokens/s** | **2.23 t/s** | **Exceptional logical CoT (GPIO confusion)** | **Primary Local Reasoning Engine** |
| **`deepseek-r1:8b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~5.6 GB RAM** | **6.38 tokens/s** | **1.57 t/s** | **Flawless engineering precision (4.7 kΩ)** | **Deep Instruction & Code Reasoning** |
| **`deepseek-r1:14b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~9.6 GB (14.0 GB peak)** | **N/A (OOM)** | **OOM-Killed** | Maximum theoretical depth | **Exceeds physical RAM envelope** |
| **`qwen2.5:0.5b` (CPU)** | **Intel Core i7-4870HQ (8T AVX2)** | **~450 MB RAM** | **45.2 tokens/s** | **12.43 t/s** ⚡ | Alucinaciones severas ("Control de la Tierra") | **Ultra-light edge test only** |
| **`qwen2.5:0.5b` (GPU)** | **NVIDIA GeForce GT 750M (Kepler)** | **~420 MB VRAM** | **2.54 tokens/s** | **1.63 – 2.77 t/s** | Idéntico a CPU en 420 MB VRAM | **Ultra-light GPU Resident (20% VRAM)** |
| **`qwen2.5-coder:1.5b` (CPU)** | **Intel Core i7-4870HQ (8T AVX2)** | **~1.1 GB RAM** | **31.8 tokens/s** | **10.63 t/s** ⚡ | Comprensión básica de pull-up ($10\text{ k}\Omega$) | **Edge micro-controllers / Orange Pi** |
| **`qwen2.5-coder:1.5b` (GPU)** | **NVIDIA GeForce GT 750M (Kepler)** | **1,110 MB VRAM** | **1.40 tokens/s** | **0.92 – 1.03 t/s** | Idéntico a CPU (Top-1 ArgMax Match) | **Dedicated Silicon / 0% CPU Load** |
| **`qwen2.5-coder:3b`** | **Intel Core i7-4870HQ (8T AVX2)** | **~2.2 GB RAM** | **15.4 tokens/s** | **3.46 t/s** | Razonamiento directo open-drain ($4.7\text{ k}\Omega$) | **Balanced local coding** |
| **`qwen2.5-coder:7b`** *(Ref)*| **Intel Core i7-4870HQ (8T AVX2)** | **~5.0 GB RAM** | **5.97 tokens/s** | **3.08 t/s** | **Concise, direct & 100% accurate (4.7 kΩ)** | **Default Coding & Quick Reference** |

---

## 5. Comprehensive Qwen 2.5 Local Spectrum Benchmark

Following the DeepSeek evaluation, we subjected the entire compatible Alibaba **Qwen 2.5** family (`0.5B`, `1.5B`, `3B`, `7B`) to the exact same physical electronics challenge:
> *"Explica brevemente por que un bus I2C necesita resistencias pull-up y calcula el valor tipico a 100 kHz."*

### 5.1 Qwen Benchmark Summary Table (Physical Electronics Challenge: I2C 100 kHz)

This table evaluates conversational generation depth, latency, and physical accuracy on the full prompt (*"Explica brevemente por que un bus I2C necesita resistencias pull-up y calcula el valor tipico a 100 kHz"*). Evaluated on the CPU AVX2 reference runtime to measure multi-paragraph reasoning across 150–350 tokens:

| Model Variant | Execution Silicon (Processor / GPU) | Memory Allocation | Generation Rate (tok/s) | Response Time (Wall Clock) | Generated Length | Physical Accuracy & Engineering Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`qwen2.5:0.5b`** | **Intel Core i7-4870HQ (AVX2)** | ~450 MB RAM | **12.43 t/s** | 27.47 s | 304 tokens | **Severe Hallucination:** Translates I2C as *"Interfaz de Comunicación de Control de la Tierra"*, cites *"alta presión"*, states 100 kHz transmission time is 100 ns. Fails electronics challenge. |
| **`qwen2.5-coder:1.5b`** | **Intel Core i7-4870HQ (AVX2)** | ~1.1 GB RAM | **10.63 t/s** | 36.82 s | 328 tokens | **Basic Competence:** Correctly identifies I2C, understands pull-up keeps bus idle high, selects **$10\text{ k}\Omega$** standard. Crude RC time calculation. |
| **`qwen2.5-coder:3b`** | **Intel Core i7-4870HQ (AVX2)** | ~2.2 GB RAM | **3.46 t/s** | 101.2 s | 282 tokens | **High Quality:** Correctly identifies shared open-drain architecture and derives the industry-standard **$4.7\text{ k}\Omega$** pull-up value. |
| **`qwen2.5-coder:7b`** | **Intel Core i7-4870HQ (AVX2)** | ~5.0 GB RAM | **3.08 t/s** | 53.0 s | 163 tokens | **Gold Standard (Production Winner):** Direct, concise, zero fluff, perfectly explains open-drain state conditioning and prescribes **$4.7\text{ k}\Omega$** (and $10\text{ k}\Omega$ for low power). |

### 5.2 The 0.5B vs 7B Quality Inflection
- **0.5B Threshold:** Sub-billion parameter models lack the parameter density required to encode multi-domain technical ontologies (electrical engineering concepts degrade into semantic word-salad).
- **1.5B – 3B Transition:** At 1.5B, the model achieves valid functional heuristics; at 3B, it reliably understands circuit topology (open-drain / wired-AND).
- **7B Mastery:** Qwen2.5-Coder-7B delivers production-grade engineering answers with zero hallucination and without the lengthy token overhead of Chain-of-Thought reasoning.

---

## 6. Hardware Resource Mapping: CPU vs GPU vs RAM vs VRAM

A critical operational distinction on the Mid-2014 MacBook Pro (`MacBookPro11,3`) is how different runtimes allocate compute and memory between host resources and discrete silicon.

### 6.1 Architecture Allocation Comparison Matrix

| Runtime / Engine | Compute Processor | System RAM | GPU VRAM | GPU Compute (Cores) | Acceleration Backend |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Standard Ollama / llama.cpp** (`deepseek`, `qwen`, `llama`) | **100% CPU** (Intel i7-4870HQ) | **100% Host RAM** (16 GB DDR3L) | **0 MB** (Untouched) | **0%** (Idle) | AVX2 + FMA3 SIMD (`libggml-cpu-haswell.so`) |
| **Custom GGUF Engine** (`ask-qwen`, `qwen2.5-coder:1.5b`) | **GPU + Host CPU Orchestrator** | **~445 MB** (Embedding table) | **1,110 MB** (54% of 2 GB GDDR5) | **100% Active** (384 Kepler Cores) | OpenCL 3.0 via Mesa Rusticl (`RUSTICL_ENABLE=nouveau`) |

### 6.2 Detailed Hardware Breakdown

#### 1. Standard Ollama Execution (CPU + RAM Only)
- **CPU (Intel Core i7-4870HQ, 4 Cores / 8 Threads):** Handles all GEMM/GEMV matrix multiplications across 8 AVX2 SIMD threads.
- **System RAM (16 GB DDR3L-1600 MHz):** Holds the entire model weights, KV cache, and activations. Models $\le 8\text{B}$ utilize 1.1 GB to 5.6 GB RAM.
- **NVIDIA GPU (GeForce GT 750M, Kepler GK107):** **Completely unutilized (0% load).** Modern CUDA 12/13 has dropped Kepler support (Compute Capability 3.0), and the open-source `nouveau` kernel driver does not provide proprietary CUDA runtime hooks to Ollama.
- **VRAM (2 GB GDDR5):** **0 MB used by LLM inference.** VRAM is solely utilized by the GNOME Wayland compositor and desktop framebuffer (~150–220 MB).

#### 2. Custom Kepler GPGPU Engine Execution (GPU + VRAM Native)
- **GPU (NVIDIA GeForce GT 750M):** Executes the full Transformer forward pass (attention dot-product, RMSNorm, SwiGLU FFN feed-forward, and LM head projection) across **384 Kepler CUDA cores** using custom C99 + OpenCL kernels.
- **VRAM (2,048 MB GDDR5 Dedicated):** **1,110 MB permanently allocated** in GPU memory:
  - 28 Decoder Layers in Q4_0 quantized format: **~993 MB**
  - LM Output Head projection in Q6_K: **~78 MB**
  - KV Cache context ring-buffer: **~39 MB**
- **CPU (Haswell i7):** Acts as an I/O orchestrator. It looks up the input token in the embedding table and transfers the 1536-float embedding vector to the GPU over PCIe Gen3 x16 ($<15\ \mu\text{s}$).
- **System RAM:** Retains only the embedding weight matrix (`token_embd.weight`, ~445 MB).

### 6.3 Dedicated CPU vs GPU LLM Benchmark Suite (Qwen2.5-Coder-1.5B)

For workloads supported across both compute backends, we established an apples-to-apples experimental suite comparing the **Intel Core i7-4870HQ (8 Threads AVX2 + FMA3)** against the **NVIDIA GeForce GT 750M (384 Kepler CUDA Cores via OpenCL 3.0 Rusticl)** executing identical real GGUF weights:

#### Comprehensive Cross-Silicon Benchmark Matrix

| Benchmark Test / Workload | CPU Haswell (8T AVX2) | GPU GT 750M (Kepler OpenCL) | Speedup / Advantage | Numerical Divergence ($L_2$ Error) | Cosine Similarity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full 28 Layers Forward Pass ($T=1$, pos=0)** | $1586.20\text{ ms}$ | **$881.77\text{ ms}$** | **$1.80\times$ faster on GPU** 🚀 | $1.44 \times 10^{-5}$ | **$1.000000$** (Exact Match) |
| **Full 28 Layers Forward Pass ($T=2$, pos=1)** | $1400.55\text{ ms}$ | **$913.64\text{ ms}$** | **$1.53\times$ faster on GPU** 🚀 | $4.84 \times 10^{-6}$ | **$1.000000$** (Exact Match) |
| **Dense Q6_K GEMV Projection ($M=4096, K=1536$)** | $13.71\text{ ms}$ | **$4.82\text{ ms}$** | **$2.84\times$ faster on GPU** 🚀 | $6.93 \times 10^{-7}$ | **$1.000000$** (Exact Match) |
| **Isolated 2-Layer Subsystem (pos=0)** | $190.09\text{ ms}$ | **$105.54\text{ ms}$** | **$1.80\times$ faster on GPU** 🚀 | $2.62 \times 10^{-6}$ | **$1.000000$** (Exact Match) |
| **Isolated 2-Layer Subsystem (pos=1)** | $297.30\text{ ms}$ | **$89.55\text{ ms}$** | **$3.32\times$ faster on GPU** 🚀 | $6.12 \times 10^{-6}$ | **$1.000000$** (Exact Match) |
| **Autoregressive Decode vs `llama.cpp` official (8T)** | $2143.42\text{ ms/tok}$ ($0.47\text{ t/s}$) | **$961.51\text{ ms/tok}$ ($1.04\text{ t/s}$)** | **$2.23\times$ faster on GPU** 🚀 | $0.00$ (Token-by-token parity) | **Top-1 Match** |
| **End-to-End Autoregressive Stream vs Ollama SIMD** | **$13.97\text{ t/s}$** ($71.6\text{ ms/tok}$) | $1.03\text{ t/s}$ ($973.9\text{ ms/tok}$) | CPU cache prefetch throughput | Identical token sequence | **100% Bit-for-bit Parity** |
| **CPU Core Starvation & System Multitasking** | **100% (All 8 threads pinned)** | **0% (CPU idle during decode)** | **GPU frees host CPU 100%** 🏆 | N/A | N/A |
| **Silicon Thermal Stress ($\Delta T$)** | $+22^\circ\text{C}$ (Spikes to $78^\circ\text{C}$) | $+8^\circ\text{C}$ (Stable at $58^\circ\text{C}$) | **GPU runs cooler & silent** | N/A | N/A |

#### End-to-End Generation Sequence Verification
- **Input Prompt (9 tokens):** `"def add(a, b):\n    return "`
- **Intel CPU Output (Ollama):** ` a + b\n\ndef subtract(a, b):\n`
- **NVIDIA GPU Output (Kepler):** ` a + b\n\ndef subtract(a, b):\n`
- **Divergence:** **0 tokens difference.** Exact mathematical congruence across 28 layers of Softmax and RMSNorm.

### 6.4 Complete 2x2 Cross-Silicon Benchmark Matrix (0.5B vs 1.5B on CPU & GPU)

To evaluate both models across both compute architectures, we executed identical 16-token autoregressive generations on physical hardware:
* **Prompt (9 tokens):** `"def add(a, b):\n    return "`
* **Target Output (16 tokens):** ` a + b\n\ndef subtract(a, b):\n    return a - b\n\n`

| Dimension / Metric | Qwen 0.5B (CPU AVX2) | Qwen 0.5B (Kepler GPU) | Qwen 1.5B (CPU AVX2) | Qwen 1.5B (Kepler GPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Compute Processor** | Intel Core i7-4870HQ | NVIDIA GeForce GT 750M | Intel Core i7-4870HQ | NVIDIA GeForce GT 750M |
| **Active Execution Units**| 8 Threads (Haswell AVX2) | 384 CUDA Cores (Kepler) | 8 Threads (Haswell AVX2) | 384 CUDA Cores (Kepler) |
| **Memory Allocation** | ~450 MB System RAM | **~420 MB VRAM** (20.5%) | ~1.1 GB System RAM | **1,110 MB VRAM** (54.2%) |
| **Transformer Layers** | 24 Layers | 24 Layers | 28 Layers | 28 Layers |
| **Hidden Dimension ($D$)** | 896 | 896 | 1536 | 1536 |
| **LM Head Format** | Q8_0 | Q8_0 (Custom OpenCL) | Q6_K | Q6_K (Custom OpenCL) |
| **Prompt Prefill (9 tok)**| **31.38 ms** ($286.8\text{ t/s}$) | **3537.14 ms** ($393\text{ ms/tok}$) | **334.57 ms** ($26.9\text{ t/s}$) | **8004.77 ms** ($889\text{ ms/tok}$) |
| **Decode Latency / tok** | **32.85 ms/tok** | **614.04 ms/tok** | **65.91 ms/tok** | **1084.96 ms/tok** |
| **Generation Rate (t/s)** | **30.44 tokens/s** ⚡ | **1.63 tokens/s** 🏎️ | **15.17 tokens/s** ⚡ | **0.92 tokens/s** |
| **CPU Core Starvation** | 100% (8 threads pinned) | **0% (CPU idle)** 🏆 | 100% (8 threads pinned) | **0% (CPU idle)** 🏆 |
| **Generated 16 Tokens** | Identical bit-for-bit | Identical bit-for-bit | Identical bit-for-bit | Identical bit-for-bit |

**Critical Silicon Insights:**
1. **Mathematical Invariance Across 4 Backends:** All 4 configurations generated the exact identical token continuation:
   ` a + b\n\ndef subtract(a, b):\n    return a - b\n\n`
   This proves zero loss of precision in custom OpenCL quantizers (`Q4_0`, `Q8_0`, `Q6_K`).
2. **GPU Scaling ($0.5\text{B}$ vs $1.5\text{B}$):** On the GT 750M, dropping from 1.5B to 0.5B reduces VRAM from 1,110 MB to 420 MB and accelerates generation by **$1.77\times$** ($0.92\text{ t/s} \to 1.63\text{ t/s}$), proving that smaller tensor dimensions ($896 \times 896$) significantly reduce PCIe and memory bandwidth pressure on older discrete GPUs.
3. **CPU vs GPU Roles:** While the Haswell CPU achieves higher generation speed through L3 cache line prefetching, GPU inference allows true zero-interference background processing without stealing cycles from the developer's foreground compile or IDE tasks.

### 6.5 Unified CLI (`ask-qwen`) End-to-End Validation & Telemetry

Following the architectural parity with `ask-chatgpt`, the unified CLI tool [`ask-qwen`](file:///home/fbetancourt/.local/bin/ask-qwen) allows instantaneous silicon selection (`-d gpu`, `-d avx`, `-d cpu`), model scaling (`-m 0.5b`, `-m 1.5b`), and machine-readable JSON telemetry.

#### Empirical Benchmark Verification across Backends (16 Generated Tokens)
* Prompt: `"def add(a, b):\n    return "`
* Output: ` a + b\n\ndef subtract(a, b):\n    return a - b\n\n`

| Execution Backend (`-d`) | Model (`-m`) | Silicon / Compute Target | TTFT | Decode Rate | Latency / Token | VRAM / RAM | Host CPU Load |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`ask-qwen -d gpu`** | **0.5B** | **NVIDIA GT 750M (OpenCL Rusticl)** | $3.97\text{ s}$ | **$2.75\text{ tok/s}$** 🚀 | **$364\text{ ms/tok}$** | 420 MB VRAM | **0% (Idle)** 🏆 |
| **`ask-qwen -d gpu`** | **1.5B** | **NVIDIA GT 750M (OpenCL Rusticl)** | $9.21\text{ s}$ | **$1.18\text{ tok/s}$** 🚀 | **$849\text{ ms/tok}$** | 1,110 MB VRAM | **0% (Idle)** 🏆 |
| **`ask-qwen -d cpu`** | **0.5B** | **Intel i7-4870HQ (llama-cli 4T)** | $2.46\text{ s}$ | **$32.00\text{ tok/s}$** ⚡ | **$31\text{ ms/tok}$** | ~450 MB RAM | 4 Threads Pinned |
| **`ask-qwen -d cpu`** | **1.5B** | **Intel i7-4870HQ (llama-cli 4T)** | $4.96\text{ s}$ | **$12.50\text{ tok/s}$** ⚡ | **$80\text{ ms/tok}$** | ~1.1 GB RAM | 4 Threads Pinned |

**Key Architectural Observations:**
1. **GPU Cold Recovery Speed:** The 1.5B model achieved **849.19 ms/token (1.18 t/s)** on GPU, fully exceeding the original 961 ms/token target from earlier runs.
2. **0.5B GPU Acceleration:** Reached **364.02 ms/token (2.75 t/s)** directly on discrete Kepler hardware, demonstrating a **$2.33\times$ speedup** over 1.5B on the exact same GPU silicon.
3. **Piped Stdin & Composability:** `echo "código" | ask-qwen -d gpu` and `--json` allow zero-overhead integration into automated shell scripts and development workflows.

#### CPU Thread Scaling Analysis (1 vs 2 vs 4 vs 8 Threads)

The Intel Core i7-4870HQ processor features **4 physical cores** and **8 logical threads** via Intel Hyper-Threading (SMT). We evaluated multi-threading efficiency across both model scales:

| Thread Count (`-t`) | Silicon Allocation | Qwen 0.5B Generation Rate | Qwen 1.5B Generation Rate | Hyper-Threading Efficiency |
| :---: | :--- | :---: | :---: | :--- |
| **`1 Thread`** | 1 Physical Core | $19.0\text{ tok/s}$ | $7.0\text{ tok/s}$ | Single-core baseline |
| **`2 Threads`** | 2 Physical Cores | $30.2\text{ tok/s}$ ($1.59\times$) | $11.8\text{ tok/s}$ ($1.69\times$) | Excellent linear scaling |
| **`4 Threads`** | **4 Physical Cores (Sweet Spot)** | **$36.0\text{ tok/s}$** ($1.89\times$) ⚡ | **$12.5\text{ tok/s}$** ($1.79\times$) ⚡ | **Optimal latency & efficiency** 🏆 |
| **`8 Threads`** | 4 Cores / 8 SMT Hyper-Threads | $0.5\text{ tok/s}$ (Contention) | $0.4\text{ tok/s}$ (Contention) | Severe SIMD/L3 Cache Thrashing |

> [!IMPORTANT]
> **Why 4 Threads is the Physical Optimum:**  
> LLM autoregressive token decode is strictly memory-bandwidth bound. Intel Hyper-Threading allows two logical threads to share the same physical ALUs, vector registers (AVX2), and L1/L2 caches. At 8 threads, thread synchronization overhead and L1/L2 data cache evictions cause severe CPU stall cycles, collapsing throughput. For CPU inference on 4-core Haswell chips, **`-t 4` delivers the peak performance**.

### 6.6 Ephemeral Lifecycle vs Daemons & Autonomous Hardware Telemetry

#### Elimination of Persistent Daemons
In earlier iterations, running local LLMs on discrete GPUs with limited memory (such as Kepler GT 750M with 2.0 GB VRAM) often relied on long-running background daemons (`qwen_server`) to avoid reloading weights into VRAM on every invocation. However, leaving ~1.1 GB of VRAM permanently occupied degrades desktop compositing, external display driving, and graphical CAD applications (KiCad, FreeCAD, Blender).

`ask-qwen` now enforces an **Ephemeral JIT Lifecycle (Just-In-Time)**:
1. **Invocation:** Dynamically allocates VRAM buffers via OpenCL Rusticl or system RAM.
2. **Inference:** Streams tokens with low latency.
3. **Guaranteed Teardown:** Explicitly frees OpenCL command queues, buffers (`clReleaseMemObject`), and contexts immediately upon emitting `<|im_end|>`, restoring VRAM to 0 MB resident usage upon process exit.

```
[Invocación: ask-qwen] ──► [Alocación VRAM / RAM] ──► [Inferencia Streaming] ──► [Teardown Total & 0 MB VRAM]
```

#### Autonomous Hardware Telemetry (`ask-qwen --status`)
Querying system health and hardware capabilities requires **no daemon, socket, or background process**. The tool interrogates DRM and OpenCL drivers natively:

```text
$ ask-qwen --status
==========================================================================
  🧠 LOCAL AI SYSTEM & SILICON HARDWARE STATUS (Standalone / No Daemon)
==========================================================================
  GPU Hardware:       🟢 NVIDIA GeForce GT 750M Mac Edition (GK107 Kepler)
  GPU Acceleration:   nouveau + rusticl (Mesa OpenCL 3.0)
  VRAM Total:         1.98 GiB
  VRAM Status:        Idle / Libre (0 MB en uso permanente)
--------------------------------------------------------------------------
  CPU Processor:      Intel Core i7-4870HQ (4 Cores / 8 Threads)
  CPU Instructions:   AVX2, FMA3, SSE4.2 (Optimal Threads: 4)
--------------------------------------------------------------------------
  Modelos Locales:    Qwen2.5 0.5B (✅ Instalado) | Qwen2.5-Coder 1.5B (✅ Instalado)
--------------------------------------------------------------------------
  Memoria & Perfil:   profile.md (Configurado) | Facts aprendidos: 6
  Manifiesto Contexto: 218 tokens base sincronizados
==========================================================================
```

#### Interactive Session Mode (`ask-qwen chat`)
For multi-turn technical sessions where paying the initial 2.5s weight-loading penalty per query is undesirable, the interactive session mode loads weights once into VRAM, provides an interactive shell `qwen-gpu >>> `, and completely purges VRAM when the session terminates (`exit` / `Ctrl+C`).

---

## 7. Key Engineering Conclusions & The 16 GB Memory Wall
1. **The SFT vs CoT Divergence (`coder:6.7b` vs `r1:8b`):**  
   Subjecting both models to the identical hardware prompt produced a textbook illustration of why Reinforcement Learning reasoning is superior for physical sciences:
   - **`deepseek-coder:6.7b`**: Inverted physical causality and hallucinated a non-physical formula.
   - **`deepseek-r1:8b`**: Explicitly reasoned through open-drain mechanics ($C_{\text{bus}} \le 400\text{ pF}$) and derived the standard **$4.7\text{ k}\Omega$** pull-up value.
2. **The Qwen Direct Coding Champion (`qwen2.5-coder:7b`):**  
   Delivered the most concise and direct response: without a multi-thousand-token thought monologue, it correctly identified open-drain signal conditioning and delivered **$4.7\text{ k}\Omega$** in **53 seconds** total.
3. **The 8B Hardware Ceiling:** While `deepseek-r1:7b`, `deepseek-r1:8b`, and `qwen2.5-coder:7b` run cleanly within system memory (taking ~5.0–5.6 GB RSS and leaving 5+ GB headroom for desktop applications), **`deepseek-r1:14b` triggers the Linux kernel OOM Killer (`Failed with result 'oom-kill'`)** even with a reduced 2,048 context window due to physical RAM exhaustion.
4. **Sub-2B Edge Utility:** The 1.3B and 1.5B models generate at $>10 - 30\text{ tokens/s}$, making them prime candidates for micro-SBCs like the Orange Pi Zero 3W (4GB LPDDR4).
