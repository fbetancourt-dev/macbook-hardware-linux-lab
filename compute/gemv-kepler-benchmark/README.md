# Kepler GT 750M vs Haswell AVX2: GEMV Q4_0 Microbenchmark (v3)

Empirical microbenchmark demonstrating that the **NVIDIA GeForce GT 750M (Kepler GK107 2GB, 384 cores)** running **Mesa Rusticl (OpenCL 3.0)** outperforms the **Intel Core i7-4870HQ (Haswell AVX2 + FMA3 across 8 OpenMP threads)** on all Matrix-Vector Multiplications (GEMV) with `Q4_0` quantized weights and on-the-fly FP32 accumulation.

---

## 🎯 Architectural Findings & The Dual-Block Breakthrough

Through iterative hardware optimization and architectural review:
1. **Kernel 1 (Naive 1-thread/row):** Uncoalesced global memory reads caused severe thrashing on large matrices.
2. **Kernel 2 (Warp-Cooperative 1-block/iter):** 32 threads in each warp coalesced 32-weight blocks, achieving $2.60\times$ speedup on attention.
3. **Kernel 3 (Dual-Block 2-nibbles/thread - Cloud Sam Optimization):**
   * 32 threads in the warp process **2 blocks (64 weights) per iteration**:
     * Lanes 0..15 handle block $2i + 0$.
     * Lanes 16..31 handle block $2i + 1$.
   * Each thread reads 1 single byte of `qs`, extracting **both low and high nibbles** in a single memory load:
     $$x_0 = (q\ \&\ 0x0F) - 8, \quad x_1 = (q \gg 4) - 8$$
   * Multiplies both weights by their respective $x$ elements with two independent accumulators (`acc0`, `acc1`) to break instruction dependency chains.
   * Cuts total loop iterations by **50% (from 48 down to 24)** and completely eliminates branch divergence (`if (lane_id < 16)`).

---

## 📊 Benchmark Results (100 Iterations on Physical Hardware)

| Test Layer | Matrix Dim | Size in VRAM | CPU (Explicit AVX2 FMA) | GPU Warp (v2) | GPU Dual-Block (v3) | Best Speedup | Max Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention Projection** | $1536 \times 1536$ | **1.27 MB** | $8,442.00\ \mu\text{s}$ ($8.44\text{ ms}$) | $1,790.23\ \mu\text{s}$ | **$1,285.49\ \mu\text{s}$ ($1.28\text{ ms}$)** | **$6.57\times$ 🚀 (GPU Faster)** | $1.91 \times 10^{-6}$ |
| **FFN Up-Projection** | $8960 \times 1536$ | **7.38 MB** | $12,636.76\ \mu\text{s}$ ($12.64\text{ ms}$) | $10,163.40\ \mu\text{s}$ | **$6,738.76\ \mu\text{s}$ ($6.74\text{ ms}$)** | **$1.88\times$ 🚀 (GPU Faster)** | $2.38 \times 10^{-6}$ |

---

## 🔬 Takeaways

* **Attention Projection ($1536 \times 1536$):** The Kepler GPU executes the full projection in just **$1.28\text{ ms}$**, delivering a **$6.57\times$ speedup** over the 8-thread Haswell AVX2 CPU.
* **FFN Layer ($8960 \times 1536$):** The dual-block optimization broke the tie conclusively: the GPU executes the 7.38 MB layer in **$6.74\text{ ms}$**, beating the CPU by nearly **$2\times$ ($1.88\times$ speedup)**.
* **Numerical Equivalence:** Max deviation is $2.38 \times 10^{-6}$, confirming high-precision numerical fidelity between CPU FMA3 and GPU MAD.

---

## 🛠️ Reproduction & Execution

```bash
# Clean build with AVX2 + FMA + OpenCL
make clean && make

# Run on NVIDIA GT 750M via Mesa Rusticl
RUSTICL_ENABLE=nouveau ./gemv_benchmark
```
