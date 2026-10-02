# NVIDIA Kepler GPGPU Compute Lab (Mesa Rusticl OpenCL 3.0)

Hardware acceleration and parallel scientific simulations running on the **384 CUDA Cores** of the **NVIDIA GeForce GT 750M Mac Edition (GK107 / NVE7)** using the modern open-source **Mesa Rusticl OpenCL 3.0** stack on top of the Linux `nouveau` driver.

---

## 🎯 Architecture & Specifications

| Feature | Specification |
| :--- | :--- |
| **Silicon Architecture** | NVIDIA Kepler (GK107 / NVE7) |
| **PCI Device ID** | `10de:0fe9` (Subsystem: Apple `106b:0130`) |
| **Streaming Multiprocessors (SMX)** | 2 SMX Units |
| **Total CUDA Cores** | **384 Cores** (192 Cores / SMX) |
| **Dedicated VRAM** | **2048 MB (2 GB) GDDR5** |
| **Bus Interface** | PCIe Gen3 x16 (via Apple gmux switch) |
| **OpenCL Platform** | `rusticl` (Mesa 25.2.8 / OpenCL 3.0 FULL_PROFILE) |
| **Compiler Backend** | LLVM / Gallium `nvc0` code generator |

---

## 🚀 Benchmarks & Physical Simulations

### 1. Gravitational N-Body Simulation (`nbody_simulation.c`)
* **Workload:** 16,384 gravitationally interacting bodies across 20 time steps.
* **Arithmetic Intensity:** High (20 FLOPs per particle pair interaction, 268.4 Million pair interactions per time step, **5.37 Billion total pairs computed**).
* **Measured Performance:**
  * **Single-Core Intel Core i7 CPU:** ~40.98 s (2049 ms/step)
  * **NVIDIA GT 750M (384 Kepler Cores):** **2.28 s** (114 ms/step)
  * **Speedup Factor:** **`17.95x FASTER`** on GPU!

### 2. 2D Heat Diffusion PDE Simulation (`heat_equation_2d.c`)
* **Workload:** 5-point Laplace stencil over a $2048 \times 2048$ thermal grid (**4.19 Million points** x 50 iterations).
* **Mathematical Precision:** GPU matches CPU calculation within `7.6e-6` floating point tolerance.
* **Throughput:** ~412 Million cell-updates per second.

---

## 🛠️ How to Compile and Run

```bash
cd /home/fbetancourt/Gemini/macbook-hardware-linux-lab/subsystems/03-display-and-gmux/compute-kepler-opencl

# Compile both benchmarks
make

# Run Gravitational N-Body Simulation (384 CUDA Cores vs CPU)
make run-nbody

# Run 2D Thermal PDE Simulation
make run-heat
```

---

## 💡 Notes on System Stability

* Running OpenCL kernels via `RUSTICL_ENABLE=nouveau` requires **zero modifications to VBIOS clocks or VRM voltages**.
* The 384 cores execute in parallel under the rock-solid stock boot profile without introducing graphical display lockups or GNOME Wayland session resets.
