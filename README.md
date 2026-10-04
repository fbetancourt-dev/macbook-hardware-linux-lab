# MacBook Pro Linux Hardware Engineering & GPU Compute Lab

Low-level hardware reverse engineering, Linux kernel driver research, and high-performance GPU computing on legacy silicon (**NVIDIA Kepler OpenCL 3.0 via Mesa Rusticl, MATLAB Simulink C-MEX, and Intel Haswell AVX2 local LLM acceleration**) for the **Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)**.

---

## 🎯 Executive Engineering Overview

This repository transforms the 2014 MacBook Pro into a dual-purpose engineering platform under modern Linux kernels (6.x/7.x) and Wayland desktop environments:
1. **A Hardware Reverse Engineering & Driver Lab ([`hardware/`](hardware/)):** Demystifying proprietary Apple ASICs, CPLDs, and PCIe peripherals (gmux, SMC, Broadcom FaceTime HD ISP, Cirrus HD audio).
2. **A Scientific GPU Compute & Local AI Lab ([`compute/`](compute/)):** Extracting maximum performance from the discrete **NVIDIA GeForce GT 750M (384 Kepler Cores, 2 GB GDDR5)** using **Mesa Rusticl (OpenCL 3.0)** for single-precision scientific computing, alongside **Intel Haswell AVX2 vector SIMD** for sub-second local LLM coding autocompletion in VS Code.

---

## 🧭 Navigation Tracks: Choose Your Entry Point

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                  macbook-hardware-linux-lab Portal                         │
 └──────────────────────┬──────────────────────────────┬──────────────────────┘
                        │                              │
         ┌──────────────┴──────────────┐┌──────────────┴──────────────┐
         ▼                             ▼▼                             ▼
   [HARDWARE LAB]               [COMPUTE LAB]                  [LOCAL AI]
  Apple ASICs & Drivers       Simulink & Scientific        llama.cpp + Continue
  • Camera BCM1570            • Cart-Pole 1,024 Ensemble   • 12.6 tokens/sec AVX2
  • Apple SMC Thermals        • 2D Thermal Anti-Windup     • DeepSeek & Qwen Coder
  • gmux Display Switcher     • Simulink-OpenCL Bridge     • Systemd background daemon
```

### 🏎️ Track 1: High-Performance GPU Computing & Control ([`compute/`](compute/))
* **[`compute/cartpole/`](compute/cartpole/):** Massive parallel ensemble simulation of **1,024 nonlinear inverted pendulums** computed on 384 GPU cores via C-MEX S-Function with Åström-Furuta energy swing-up and LQR control, plus an interactive 60 FPS real-time visualizer with keyboard steering.
* **[`compute/thermal-control/`](compute/thermal-control/):** 2D transient heat diffusion PDE solver ($256 \times 256 = 65,536$ nodes) running on GPU coupled with multi-zone closed-loop PID control and anti-windup clamping in Simulink.
* **[`compute/simulink-opencl-bridge/`](compute/simulink-opencl-bridge/):** Zero-allocation Level-2 C-MEX gateway architecture for streaming data between MATLAB Simulink and OpenCL kernels.
* **[`compute/opencl-basics/`](compute/opencl-basics/):** Standalone C OpenCL kernels demonstrating $17.95\times$ speedup on N-body gravitation and 412M updates/sec 2D Laplacian stencils.

### 🤖 Track 2: GPU-Accelerated Resident Transformer & Local AI ([`compute/gguf-real-weights-loader/`](compute/gguf-real-weights-loader/) & [`compute/local-llm/`](compute/local-llm/))
* **[`compute/gguf-real-weights-loader/`](compute/gguf-real-weights-loader/):** Complete 28-layer + LM Head Transformer pipeline executing resident in GT 750M VRAM (1110 MB allocated; input embeddings prepared on CPU host) for **Qwen2.5-Coder-1.5B**.
  * **Hardware Breakthrough:** Overcomes the absence of silicon-level `cl_khr_fp16` on Kepler GK107 via custom warp-coalesced OpenCL kernels with FP32 software accumulation.
  * **Empirical Speedup:** Achieves **$961\text{ ms/tok}$** in autoregressive decode (**$2.23\times$ faster** than official `llama.cpp` on 8 Haswell CPU threads at $2143\text{ ms}$ on the tested benchmark prompt), yielding token-by-token sequence match with llama.cpp.
  * **Production Daemon (`qwen_server`):** Persistent Unix domain socket daemon with prefilled frozen base KV cache (`SET_BASE`), socket streaming, avoiding redundant prefix prefill for system context.
  * **Transactional Memory CLI (`ask-qwen`):** Pair programming CLI with staged prepare and atomic replacement (`facts.md.tmp` $\to$ fsync $\to$ `os.replace` $\to$ dir fsync), concurrency lock (`facts.lock`), and deterministic content-hash self-healing memory verification.
* **[`compute/local-llm/`](compute/local-llm/):** CPU-side Haswell AVX2/FMA3 pipeline for high-batch prompt prefill and background `llama-server` integration powering **Continue** in VS Code.

### 🛠️ Track 3: Proprietary Apple Hardware Reverse Engineering & Linux Drivers ([`hardware/`](hardware/))
* **[`hardware/camera-bcm1570/`](hardware/camera-bcm1570/):** Complete reverse engineering of the Broadcom BCM1570 PCIe ISP FaceTime HD camera, DDR ringbuffer IPC protocol, and out-of-tree V4L2 driver (`bcwc_pcie`).
* **[`hardware/smc-thermals/`](hardware/smc-thermals/):** Apple System Management Controller (SMC) protocol on LPC bus `0x300`, dual-fan tachometer PID control, and 661 hardware monitoring keys.
* **[`hardware/display-gmux/`](hardware/display-gmux/):** Custom Lattice MachXO CPLD hardware architecture, embedded DisplayPort (eDP) lane multiplexing between Intel Iris Pro and NVIDIA GT 750M, and PWM backlight synthesis.
* **[`hardware/input-trackpad/`](hardware/input-trackpad/):** Broadcom BCM5974 multi-touch controller USB protocol and evdev pressure event mapping.
* **[`hardware/audio-cs4208/`](hardware/audio-cs4208/):** Cirrus Logic CS4208 HD Audio codec, multi-channel DAC/ADC pathing, and SPDIF optical routing.
* **[`hardware/thunderbolt-pcie/`](hardware/thunderbolt-pcie/):** Intel DSL5520 Falcon Ridge 4C Thunderbolt 2 controller, dual 20 Gbps channels, and Linux PCIe hotplug.

---

## 💻 Hardware Architecture Profile (`MacBookPro11,3`)

```
                        ┌──────────────────────────────────────────────┐
                        │   Intel Core i7-4870HQ (Haswell Crystalwell) │
                        │   - 4C / 8T @ 2.5 - 3.7 GHz                  │
                        │   - 128 MB eDRAM Iris Pro Graphics 5200      │
                        │   - DRAM Controller [8086:0d04]              │
                        └───────┬──────────────────────────────┬───────┘
                                │ PCIe Gen3 x16                │ DMI 2.0 (20 Gbps)
                                ▼                              ▼
                 ┌─────────────────────────────┐ ┌──────────────────────────────────────────┐
                 │ NVIDIA GeForce GT 750M Mac  │ │ Intel 8-Series HM87 Lynx Point PCH       │
                 │ [10de:0fe9] (2 GB GDDR5)    │ │ [8086:8c4b]                              │
                 │ OpenCL 3.0 (Mesa Rusticl)   │ │                                          │
                 └──────────────┬──────────────┘ └──────┬────────────┬────────────┬─────────┘
                                │                       │            │            │
                                ▼                       │            │            │
                 ┌─────────────────────────────┐        │            │            │
                 │ Apple gmux (Lattice CPLD)   │        │            │            │
                 │ DisplayPort / PWM Backlight │        │            │            │
                 └──────────────┬──────────────┘        │            │            │
                                │                       │            │            │
                                ▼                       ▼            ▼            ▼
                           Retina Panel          PCIe Root    PCIe Root    LPC Bus (IO)
                          (2880 x 1800)            Port #3      Port #4    (0x300-0x31f)
                                                        │            │            │
                                                        ▼            ▼            ▼
                                                 ┌────────────┐┌────────────┐┌────────────┐
                                                 │ Broadcom   ││ Samsung    ││ Apple SMC  │
                                                 │ BCM1570    ││ NVMe SSD   ││ (661 Keys) │
                                                 │ FaceTime HD││ PCIe Gen3  ││ Fans / Temp│
                                                 │ (BAR0,2,4) ││ [144d:a80c]││ Batt / ALS │
                                                 └────────────┘└────────────┘└────────────┘
```

---

## 📚 Central Documentation & Guides

* [`docs/DEEPSEEK_LOCAL_MODELS_BENCHMARK.md`](docs/DEEPSEEK_LOCAL_MODELS_BENCHMARK.md): **DeepSeek Local Models Benchmark & Scaling Whitepaper: Why, How, and What We Achieved** comparing 1.3B, 1.5B, 6.7B, 7B, 8B, and 14B on Haswell AVX2.
* [`docs/KEPLER_GPGPU_LLM_ARCHITECTURE.md`](docs/KEPLER_GPGPU_LLM_ARCHITECTURE.md): **Engineering Whitepaper: Why, How, and What We Achieved** running Qwen2.5-Coder-1.5B directly on the NVIDIA GT 750M (Kepler OpenCL 3.0 via Mesa Rusticl).
* [`docs/opencl-rusticl-setup.md`](docs/opencl-rusticl-setup.md): Complete guide to configuring Mesa Rusticl, environment variables, compiler flags, and avoiding Vulkan pushbuf collisions.
* [`docs/MOTHERBOARD_TOPOLOGY.md`](docs/MOTHERBOARD_TOPOLOGY.md): In-depth hardware topology, bus mappings, and PCIe BAR assignments.
* [`tools/macbook_system_audit.sh`](tools/macbook_system_audit.sh): Automated diagnostic script auditing system sensors, PCIe links, and kernel drivers.
