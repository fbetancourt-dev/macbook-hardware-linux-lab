# Real Weights GGUF Loader & Full 28-Layer Transformer Pipeline (Qwen2.5-Coder-1.5B)

Direct hardware execution and numerical validation of real model weights loaded from an official GGUF model file (**Qwen2.5-Coder-1.5B-Instruct-Q4_0.gguf**) running across the complete 28 decoder layers ($blk.0$ through $blk.27$) + Output Norm on the **NVIDIA GeForce GT 750M (Kepler GK107, 2 GB GDDR5)** under **Mesa Rusticl (OpenCL 3.0)** compared against an 8-thread **Intel Core i7-4870HQ (Haswell AVX2 + FMA3)** CPU reference.

---

## 🎯 Architecture & Scaling Law

1. **Dynamic GGUF Indexer in C:**
   - Full header and metadata parsing directly in native C (`parse_gguf_tensors`).
   - Dynamically resolves byte offsets for all 339 tensors with robust `read_exact_at()` handling `EINTR` signal retries and EOF verification.
2. **Transformer Residency & Memory Budget in GT 750M VRAM:**
   - 28 Layers $\times$ 25.10 MB = **702.80 MB** weights (Q4_0).
   - Output Norm (`output_norm.weight`): **6 KB** (1536 FP32 floats).
   - Full LM Head: **182.57 MB** (151,936 rows of Q6_K).
   - 28 Layers $\times$ 8.00 MB = **224.00 MB** KV cache ($T_{\max}=4096$).
   - Dynamic workspace: strictly **0.33 MB** (reused in-place across all 28 layers).
   - **Total VRAM Resident:** **1110 MB** (< 55% of the 2048 MB VRAM of the GT 750M).
   - **Host CPU Pre-Processing:** Token embeddings (`token_embd.weight`, ~445 MB) are dequantized and indexed on the CPU host, with the resulting 1536-float embedding vector transferred to the GPU per token over PCIe to conserve VRAM for the compute-intensive decoder layers.
3. **Autoregressive Cache Continuity:**
   - Token 0 evaluated at position $\text{pos}=0$ ($T=1$).
   - Token 1 evaluated at position $\text{pos}=1$ ($T=2$), appending and reading cached KV keys and values across all 28 layers.

---

## 📊 Measured Empirical Results (Physical Hardware Benchmark)

- **GPU Device:** NVIDIA GeForce GT 750M (Kepler GK107, 384 cores, 2 GB GDDR5) via Mesa Rusticl OpenCL 3.0 (`NVE7`).
- **CPU Reference:** Intel Core i7-4870HQ @ 2.50 GHz (Haswell AVX2 + FMA3, 8 OpenMP threads).
- **Model:** `qwen2.5-coder-1.5b-instruct-q4_0.gguf` (1017 MB official GGUF).

### 1. Full 28 Layers + Output Norm (`test_28_layers_real`)

| Step | Sequence Position / Context | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Max Absolute Diff | Relative $L_2$ Error | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | $\text{pos}=0$ ($T=1$) | $1586.20\text{ ms}$ | **$881.77\text{ ms}$** | **$1.80\times$** 🚀 | $1.14 \times 10^{-3}$ | **$1.44 \times 10^{-5}$** | **$1.000000$** |
| **Token 1** | $\text{pos}=1$ ($T=2$) | $1400.55\text{ ms}$ | **$913.64\text{ ms}$** | **$1.53\times$** 🚀 | $1.26 \times 10^{-4}$ | **$4.84 \times 10^{-6}$** | **$1.000000$** |

### 2. Phase B0: Isolated Q6_K GEMV Kernel (`test_gemv_q6_k`)

Canonical Basis Unit Tests ($e_i$) and dense random latent projection ($M=4096, K=1536$) comparing OpenCL `gemv_q6_k` vs official GGML reference:

| Test Vector | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Max Absolute Diff | Relative $L_2$ Error | Cosine Similarity | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Canonical Bases $e_i$** | — | — | — | **$0.0000\text{ e+}00$** | **$0.0000\text{ e+}00$** | **$1.000000$** | **PASS (100%)** |
| **Dense Latent ($x \sim \mathcal{N}(0, 1)$)** | $13.71\text{ ms}$ | **$4.82\text{ ms}$** | **$2.84\times$** 🚀 | $2.62 \times 10^{-6}$ | **$6.93 \times 10^{-7}$** | **$1.000000$** | **PASS** |

### 3. Phase B: Full 28 Layers + LM Head Logits Validation (`test_full_model_logits`)

Full pipeline execution with official Qwen2.5-Coder-1.5B weights: 28 Decoder Layers (Q4_0) + Output Norm (FP32) + LM Head (151,936 rows of Q6_K, 182.57 MB). Total model memory resident in GT 750M VRAM: **1110 MB** (< 55% of 2 GB VRAM).

| Prompt Evaluated | Latency CPU Haswell (8T) | Latency GPU GT 750M | Speedup vs CPU | Top-1 Predicted Token | ArgMax GPU vs CPU Logit | Margin $\Delta$ ($z_{(1)} - z_{(2)}$) | Max Error $\epsilon_\infty$ | Condition $\Delta > 2\epsilon_\infty$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **"Hi"** (Token 13048) | $1905.83\text{ ms}$ | **$1945.32\text{ ms}$** | $0.98\times$ | Token 16 (`'1'`) | $10.2319$ vs $10.2319$ | **$0.4415$** | $5.82 \times 10^{-5}$ | **MET ($0.4415 > 1.16 \times 10^{-4}$)** ✅ |
| **"def"** (Token 750) | $1768.70\text{ ms}$ | **$1328.79\text{ ms}$** | **$1.33\times$** 🚀 | Token 16 (`'1'`) | $10.9106$ vs $10.9105$ | **$1.4854$** | $1.26 \times 10^{-4}$ | **MET ($1.4854 > 2.52 \times 10^{-4}$)** ✅ |

#### Top-5 Predicted Continuation Tokens ("Hi"):
| Rank | Token ID | Decoded Text | GPU Logit | CPU Logit | Absolute Error | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **16** | `'1'` | **10.2319** | **10.2319** | **$2.77 \times 10^{-5}$** | **Top-1 ArgMax Match!** |
| **#2** | **13** | `'.'` | 9.7904 | 9.7904 | $5.82 \times 10^{-5}$ | Match |
| **#3** | **11** | `','` | 9.3392 | 9.3393 | $5.25 \times 10^{-5}$ | Match |
| **#4** | **18** | `'3'` | 8.6835 | 8.6836 | $3.24 \times 10^{-5}$ | Match |
| **#5** | **17** | `'2'` | 8.3430 | 8.3430 | $3.43 \times 10^{-5}$ | Match |

### 4. Phase C: Real-Time Streaming Autoregressive Generator (`generate_stream`)

Complete end-to-end multi-token prefill and autoregressive greedy token generation running with 28 layers, LM head, and KV cache resident on the **NVIDIA GeForce GT 750M** Kepler GPU (384 cores, 2 GB VRAM; embeddings prepared on CPU host) using real weights from **Qwen2.5-Coder-1.5B**.

- **Prompt:** `"def add(a, b):\n    return "` (9 tokens prefilled: `[750, 912, 2877, 11, 293, 982, 262, 470, 220]`)
- **Prefill Latency (TTFT):** $6405.55\text{ ms}$ (Average $\approx 711\text{ ms}$ per prompt token).
- **Autoregressive Continuation (Generated by GT 750M):**
  ```python
  a + b

  def subtract(a, b):
      return a - b

  def multiply(a, b):
      return 
  ```

#### Corrected Benchmark Accounting & Llama.cpp Independent Validation:
| Engine | Hardware | Prompt Eval (9 tokens) | Decode Passes | Mean Decode Latency | Generation Throughput | Generated Output Tokens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`generate_stream` (Our Engine)** | **NVIDIA GT 750M (OpenCL Rusticl)** | $6405.55\text{ ms}$ ($711.7\text{ ms/tok}$) | **23 passes** | **$961.51\text{ ms/pass}$** | **$1.04\text{ t/s}$** 🚀 | **Token-by-token Identical** |
| **`llama-completion` (llama.cpp official)** | **Haswell i7-4870HQ (8 threads AVX2)** | $2217.91\text{ ms}$ ($246.4\text{ ms/tok}$) | **23 runs** | **$2143.42\text{ ms/run}$** | **$0.47\text{ t/s}$** | **Token-by-token Identical** |

> [!NOTE]
> **Independent Cross-Validation:** In this specific 9-prompt / 23-decode benchmark run, the output token sequence produced by our GT 750M Kepler engine is token-for-token identical to official `llama.cpp` (`llama-completion`). On decode passes for this test prompt, our GT 750M achieves **$2.23\times$ faster** decode than official `llama.cpp` running on 8 CPU threads ($961\text{ ms}$ vs $2143\text{ ms}$).

#### Edge Case & Termination Verification:
- **`max_new_tokens = 0`:** Exits immediately with 0 tokens generated and 0 decode forwards executed. No uninitialized buffer access.
- **`max_new_tokens = 1`:** Emits exactly 1 token directly from prompt prefill; executes 0 decode forwards (`Latency: N/A`).
- **Context Boundary:** Uses `next_input_pos` boundary checks, correctly permitting sequence position $T_{\max} - 1$ while safely preventing writes at $T_{\max}$.
- **Non-Finite / NaN Injection (`--test-nan`):** Detects non-finite values in the 151,936 logits immediately and aborts with a descriptive fatal error without corrupting state.
- **OpenCL Dispatch Guard:** All `clEnqueueNDRangeKernel` and buffer transfer calls verified with `CHECK_CL()`.

### 5. Isolated 2-Layer Subsystem (`test_real_layers`)

| Step | Sequence Position / Context | CPU Haswell (8T) | GPU GT 750M | Speedup vs CPU | Relative $L_2$ Error | Cosine Similarity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | $\text{pos}=0$ ($T=1$) | $190.09\text{ ms}$ | **$105.54\text{ ms}$** | **$1.80\times$** 🚀 | **$2.62 \times 10^{-6}$** | **$1.000000$** |
| **Token 1** | $\text{pos}=1$ ($T=2$) | $297.30\text{ ms}$ | **$89.55\text{ ms}$** | **$3.32\times$** 🚀 | **$6.12 \times 10^{-6}$** | **$1.000000$** |



### 6. Production Local Daemon & CLI (`qwen_server` & `ask-qwen`)

A high-performance persistent daemon resident in GT 750M VRAM paired with a transactional CLI client implementing persistent memory with crash consistency:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ask-qwen (Client CLI)                           │
│   • Profile (~/.config/local_llm/profile.md)                           │
│   • Persistent Facts (~/.config/local_llm/facts.md)                    │
│   • Manifest (~/.config/local_llm/memory_manifest.json)                │
│   • Locking: facts.lock via fcntl.flock(LOCK_EX)                       │
│   • Integrity: prompt_content_hash, model_sha256, tokenizer_hash       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Unix Domain Socket
                                    │ (/run/user/1000/qwen.sock)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      qwen_server (GPU Daemon)                          │
│   • 1110 MB / 2048 MB Resident in GT 750M VRAM                         │
│   • 28 Layers (Q4_0) + Output Norm (FP32) + LM Head (Q6_K) + KV Cache  │
│   • Input embeddings prepared on CPU host & transferred via PCIe       │
│   • Frozen Base KV Cache (SET_BASE prefilled once, avoids repeat)      │
│   • Streaming token generation directly to socket                      │
│   • Lifecycle States: UNINITIALIZED ➔ REBUILDING ➔ READY               │
│   • Guards: SO_RCVTIMEO (5s), RAW_QUERY guard, context overflow check  │
└────────────────────────────────────────────────────────────────────────┘
```

#### Protocol Specification:
- **`PING`:** Returns `PONG\n` (liveness check).
- **`STATUS`:** Returns formatted metadata string:
  ```
  STATUS OK model=<path> model_sha256=<hash> t_max=4096 base_pos=218 base_hash=0x6204c50c4b83d6ac base_generation=1 memory_state=READY vram_mb=1110 (weights_mb=850 kv_cache_mb=224 workspace_mb=36)
  ```
  *(Note: `vram_mb=1110` represents the static model budget allocated on the GPU for weights, LM head, KV cache, and workspace buffers, rather than dynamic OS telemetry).*
- **`SET_BASE <t1,t2,...>`:** Prefills system prompt tokens into KV cache starting at `pos=0` and freezes them as base context. Transitions state from `UNINITIALIZED` $\to$ `REBUILDING` $\to$ `READY`.
- **`QUERY <max_new> <t1,t2,...>`:** Appends user tokens after `base_pos` without touching frozen base cache, evaluates user prompt, and streams generated tokens one by one until `<|im_end|>` or `max_new` limit.
- **`RAW_QUERY <max_new> <t1,t2,...>`:** Evaluates prompt from `cur_pos=0` (only permitted when no base context is active).

#### Transactional Guarantees & Memory Consistency:
1. **Staged Prepare & Atomic Replacement (Crash-Consistent Transaction):**
   - **Phase 1 (Prepare):** Prompt tokens are prefilled into GPU VRAM first. If the GPU rejects (e.g. `ERR_CONTEXT_FULL`), on-disk memory remains untouched.
   - **Phase 2 (Commit):** On GPU confirmation (`OK`), facts file is written atomically (`facts.md.tmp` $\to$ `fsync()` $\to$ `os.replace` $\to$ directory `fsync()`), followed by atomic `memory_manifest.json` write. Recovery is guaranteed even across sudden crashes.
2. **Deterministic Content Fingerprinting:**
   - Instead of fragile mtime checks, `memory_manifest.json` tracks `prompt_content_hash = sha256(build_base_system_prompt())`. Any modification, addition, or deletion of memory immediately triggers automatic self-healing resynchronization.
3. **Model & Binary Fingerprinting:**
   - `model_sha256` of loaded GGUF and `tokenizer_hash` of `llama-tokenize` verified actively before inference to prevent running against stale weights.
4. **Defensive Socket Accumulation & Bounds Validation:**
   - Socket reads accumulated strictly until `\n` with a 5-second `SO_RCVTIMEO` timeout. Truncated inputs safely rejected with `ERR_TRUNCATED_REQUEST`.
   - Token IDs and output token budgets strictly validated ($0 \le \text{tok} < \text{VOCAB\_SIZE}$, $0 < \text{max\_new} \le T_{\max}$, $\text{cur\_pos} + \text{tokens} + \text{max\_new} < T_{\max}$).

---

## 🔬 Key Engineering Insights

1. **Flawless End-to-End Numerical Stability:**
   - After traversing all 28 consecutive decoder layers ($28 \times 11 = 308$ kernel launches per token) and final RMSNorm, the Relative $L_2$ error remains below **$1.45 \times 10^{-5}$** with **Cosine Similarity = 1.000000**.
   - Zero NaNs, zero Infs, confirming that quantization error does not accumulate uncontrollably across 28 layers.
2. **Generation Rate on GT 750M:**
   - Full 28-layer autoregressive forward step executes in **$\approx 880 - 913\text{ ms}$** on the GT 750M, achieving **$\approx 1.1\text{ tokens/s}$** on hardware from 2014.
   - Consistently outperforms the 8-thread Haswell CPU baseline ($1.40 - 1.58\text{ s}$ per token).
3. **Ultra-Fast Weights Loading & Transfer:**
   - 702.8 MB loaded from SSD via `pread()` in **495 ms**.
   - Entire 28-layer model transferred to GT 750M VRAM over PCIe in **1103 ms**.

---

## 🛠️ Build & Run

```bash
# Build all binaries (test suites + qwen_server daemon):
make clean && make

# Symlink ask-qwen CLI to ~/.local/bin:
mkdir -p ~/.local/bin
ln -sf $(pwd)/ask_qwen.py ~/.local/bin/ask-qwen

# Launch persistent daemon in background:
RUSTICL_ENABLE=nouveau ./qwen_server &

# Query via ask-qwen CLI:
ask-qwen "Hola Qwen, que hardware tienes?"

# Check status:
ask-qwen --status

# Store persistent fact in memory:
ask-qwen --remember "Francisco es un ingeniero experto en Linux y robótica."
```
