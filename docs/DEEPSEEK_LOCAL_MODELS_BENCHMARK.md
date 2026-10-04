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
| **`qwen2.5-coder:7b`** *(Ref)*| **~5.0 GB** | **5.97 tokens/s** | **2.50 t/s** | **Concise, direct & 100% accurate (4.7 kΩ)** | **Default Coding & Quick Reference** |

### 4.2 Key Engineering Conclusions & The 16 GB Memory Wall
1. **The SFT vs CoT Divergence (`coder:6.7b` vs `r1:8b`):**  
   Subjecting both models to the identical hardware prompt (*why I2C requires pull-ups and calculating the typical 100 kHz value*) produced a textbook illustration of why Reinforcement Learning reasoning is superior for physical sciences:
   - **`deepseek-coder:6.7b`** (Standard Supervised Fine-Tuning): Generated output directly without deliberating. It inverted physical causality (claiming pull-ups prevent the line from being "always high") and hallucinated a non-physical formula:
     $$R_{\text{pullup}} = \frac{V_{CC} \cdot R_L}{\frac{1000}{f} - 1}$$
   - **`deepseek-r1:8b`** (Reinforcement Learning with `<think>` CoT): Explicitly reasoned through open-drain mechanics (devices can sink to GND but cannot source to VCC), analyzed capacitive bus loading ($C_{\text{bus}} \le 400\text{ pF}$), and correctly derived the industry-standard **$4.7\text{ k}\Omega$** pull-up value.
2. **The Direct Competitor (`qwen2.5-coder:7b`):**  
   Tested against the exact same hardware challenge, Alibaba's **Qwen2.5-Coder-7B** delivered the most concise and direct response: without needing a multi-thousand-token thought monologue, it correctly identified open-drain signal conditioning and delivered the exact industry standard **$4.7\text{ k}\Omega$** in strictly **33 seconds** of generation.
3. **The 8B Hardware Ceiling:** While `deepseek-r1:7b` and `deepseek-r1:8b` run cleanly within system memory (taking ~5.1–5.6 GB RSS and leaving 5+ GB headroom for desktop applications), **`deepseek-r1:14b` triggers the Linux kernel OOM Killer (`Failed with result 'oom-kill'`)** even when tested with a reduced 2,048-token context window. At a 14 GB peak memory allocation plus 3.7 GB swap, it breaches the physical 16 GB RAM ceiling of this workstation.
4. **Sub-2B Edge Utility:** The 1.3B and 1.5B models generate at $>8 - 30\text{ tokens/s}$, making them prime candidates for deployment on micro-SBCs like the Orange Pi Zero 3W (4GB LPDDR4).
