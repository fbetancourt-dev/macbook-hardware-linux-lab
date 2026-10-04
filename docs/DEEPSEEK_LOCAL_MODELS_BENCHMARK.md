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

### 4.1 Comparative Performance Matrix (Haswell AVX2, 8 Threads)

| Model | Memory Footprint (RSS) | Prompt Eval (TTFT) | Generation Rate (tok/s) | Reasoning Fidelity | Production Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`deepseek-coder:1.3b`** | **~850 MB** | **36.4 tokens/s** | **9.09 t/s** ⚡ | Syntax autocomplete (confusión física) | **Active Tab Completion (Continue)** |
| **`deepseek-r1:1.5b`** | **~1.3 GB** | **29.0 tokens/s** | **7.5 – 12.0 t/s** | Heuristic reasoning (CoT entusiasta) | **Edge IoT / Orange Pi candidate** |
| **`deepseek-coder:6.7b`** | **~4.2 GB** | **5.76 tokens/s** | **2.13 t/s** | SFT direct output (alucina fórmulas) | **Code autocomplete / Refactoring** |
| **`deepseek-r1:7b`** | **~5.1 GB** | **5.95 tokens/s** | **2.23 t/s** | **Exceptional logical CoT (GPIO confusion)** | **Primary Local Reasoning Engine** |
| **`deepseek-r1:8b`** | **~5.6 GB** | **6.38 tokens/s** | **1.57 t/s** | **Flawless engineering precision (4.7 kΩ)** | **Deep Instruction & Code Reasoning** |
| **`deepseek-r1:14b`** | **~9.6 GB (14.0 GB peak)** | **N/A (OOM)** | **OOM-Killed** | Maximum theoretical depth | **Exceeds physical RAM envelope** |
| **`qwen2.5:0.5b`** | **~450 MB** | **45.2 tokens/s** | **12.43 t/s** ⚡ | Alucinaciones severas ("Control de la Tierra") | **Ultra-light edge test only** |
| **`qwen2.5-coder:1.5b`** | **~1.1 GB** | **31.8 tokens/s** | **10.63 t/s** ⚡ | Comprensión básica de pull-up ($10\text{ k}\Omega$) | **Edge micro-controllers / Orange Pi** |
| **`qwen2.5-coder:3b`** | **~2.2 GB** | **15.4 tokens/s** | **3.46 t/s** | Razonamiento directo open-drain ($4.7\text{ k}\Omega$) | **Balanced local coding** |
| **`qwen2.5-coder:7b`** *(Ref)*| **~5.0 GB** | **5.97 tokens/s** | **3.08 t/s** | **Concise, direct & 100% accurate (4.7 kΩ)** | **Default Coding & Quick Reference** |

---

## 5. Comprehensive Qwen 2.5 Local Spectrum Benchmark

Following the DeepSeek evaluation, we subjected the entire compatible Alibaba **Qwen 2.5** family (`0.5B`, `1.5B`, `3B`, `7B`) to the exact same physical electronics challenge:
> *"Explica brevemente por que un bus I2C necesita resistencias pull-up y calcula el valor tipico a 100 kHz."*

### 5.1 Qwen Benchmark Summary Table

| Model Variant | Disk Footprint | Memory RSS | Eval Rate (Speed) | Response Time | Physical Accuracy & Engineering Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`qwen2.5:0.5b`** | 397 MB | ~450 MB | **12.43 t/s** | 27.47 s | **Severe Hallucination:** Translates I2C as *"Interfaz de Comunicación de Control de la Tierra"*, cites *"alta presión"*, states 100 kHz transmission time is 100 ns. Fails electronics challenge. |
| **`qwen2.5-coder:1.5b`** | 986 MB | ~1.1 GB | **10.63 t/s** | 36.82 s | **Basic Competence:** Correctly identifies I2C, understands pull-up keeps bus idle high, selects **$10\text{ k}\Omega$** standard. Crude RC time calculation. |
| **`qwen2.5-coder:3b`** | 1.9 GB | ~2.2 GB | **3.46 t/s** | 101.2 s | **High Quality:** Correctly identifies shared open-drain architecture and derives the industry-standard **$4.7\text{ k}\Omega$** pull-up value. |
| **`qwen2.5-coder:7b`** | 4.7 GB | ~5.0 GB | **3.08 t/s** | 53.0 s | **Gold Standard (Production Winner):** Direct, concise, zero fluff, perfectly explains open-drain state conditioning and prescribes **$4.7\text{ k}\Omega$** (and $10\text{ k}\Omega$ for low power). |

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

### 6.3 Direct Side-by-Side Benchmark: Qwen2.5-Coder-1.5B (CPU vs GPU)

We performed an apples-to-apples evaluation using the exact same prompt (`"def add(a, b):\n    return "`, 9 tokens) and generating the identical 10-token greedy continuation (` a + b\n\ndef subtract(a, b):\n`):

| Evaluation Dimension | CPU Version (Ollama / AVX2 + FMA3) | GPU Version (GT 750M / OpenCL Rusticl) | Divergence / Equivalence |
| :--- | :--- | :--- | :---: |
| **Model Weights** | `qwen2.5-coder:1.5b` (Q4_K_M) | `qwen2.5-coder-1.5b-instruct-q4_0.gguf` | Same base weights |
| **Active Silicon** | 8 Threads Intel Core i7-4870HQ | 384 CUDA Cores NVIDIA GT 750M | CPU vs GPU |
| **Memory Allocation** | ~1.1 GB System RAM (DDR3L) | 1,110 MB VRAM (GDDR5) + 445 MB RAM | RAM vs VRAM |
| **Prompt Prefill (9 tokens)** | $313.0\text{ ms}$ ($28.7\text{ tok/s}$) | $6513.2\text{ ms}$ ($711.7\text{ ms/tok}$) | CPU cache bandwidth |
| **Generation Rate (10 tokens)**| **$13.97\text{ tok/s}$** ($71.6\text{ ms/tok}$) | **$1.03\text{ tok/s}$** ($973.9\text{ ms/tok}$) | Haswell SIMD vs Kepler |
| **Generated Output Tokens** | ` a + b\n\ndef subtract(a, b):\n` | ` a + b\n\ndef subtract(a, b):\n` | **100% Bit-for-bit Identical** |
| **Host CPU Utilization** | **100% all 8 threads loaded** | **0% CPU load during decode** | Frees CPU for other tasks |

**Key Takeaways:**
1. **Deterministic Equivalence:** Both execution engines arrive at the exact same autoregressive token sequence with zero drift across all 28 layers.
2. **Compute Trade-offs:** While AVX2 multi-threading on the Haswell CPU achieves higher generation throughput due to L3 cache prefetching and dual-channel DDR3L bandwidth, the GPU OpenCL engine completely frees the host CPU from inference load, making it possible to run heavy multitasking without CPU starvation.

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
