# MacBook Pro Linux Hardware Engineering & GPU Compute Lab

Low-level hardware reverse engineering, Linux kernel driver research, and high-performance GPU computing on legacy silicon (**NVIDIA Kepler OpenCL 3.0 via Mesa Rusticl, MATLAB Simulink C-MEX, and local LLM acceleration**) for the **Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)**.

---

## 🎯 Executive Engineering Overview

This repository transforms the 2014 MacBook Pro into a dual-purpose engineering platform under modern Linux kernels (6.x/7.x) and Wayland desktop environments:
1. **A Hardware Reverse Engineering & Driver Lab (`hardware/`):** Demystifying proprietary Apple ASICs, CPLDs, and PCIe peripherals (gmux, SMC, Broadcom FaceTime HD ISP, Cirrus HD audio).
2. **A Scientific GPU Compute & Local AI Lab (`compute/`):** Extracting maximum performance from the discrete **NVIDIA GeForce GT 750M (384 Kepler Cores, 2 GB GDDR5)** using **Mesa Rusticl (OpenCL 3.0)** rather than proprietary dead-end CUDA or unstable Vulkan drivers.

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
  • Camera BCM1570            • Cart-Pole 1,024 Ensemble   • 12.6 tokens/sec OpenCL
  • Apple SMC Thermals        • 2D Thermal Anti-Windup     • Qwen2.5-Coder in VRAM
  • gmux Display Switcher     • Simulink-OpenCL Bridge     • Systemd background daemon
```

### 🏎️ Track 1: High-Performance GPU Computing & Control ([`compute/`](compute/))
* **[`compute/cartpole/`](compute/cartpole/):** Massive parallel ensemble simulation of **1,024 nonlinear inverted pendulums** computed on 384 GPU cores via C-MEX S-Function with Åström-Furuta energy swing-up and LQR control, plus an interactive 60 FPS real-time visualizer with keyboard steering.
* **[`compute/thermal-control/`](compute/thermal-control/):** 2D transient heat diffusion PDE solver ($256 \times 256 = 65,536$ nodes) running on GPU coupled with multi-zone closed-loop PID control and anti-windup clamping in Simulink.
* **[`compute/simulink-opencl-bridge/`](compute/simulink-opencl-bridge/):** Zero-allocation Level-2 C-MEX gateway architecture for streaming data between MATLAB Simulink and OpenCL kernels.
* **[`compute/opencl-basics/`](compute/opencl-basics/):** Standalone C OpenCL kernels demonstrating $17.95\times$ speedup on N-body gravitation and 412M updates/sec 2D Laplacian stencils.

### 🤖 Track 2: Local AI & LLM Acceleration on Legacy GPU ([`compute/local-llm/`](compute/local-llm/))
* **[`compute/local-llm/`](compute/local-llm/):** Production deployment of `llama.cpp` using Mesa Rusticl OpenCL 3.0.
  * **Model:** Qwen2.5-Coder 1.5B (986 MB in VRAM, ~809 MB RSS).
  * **Throughput:** **11.5 – 12.6 tokens/sec** generation and **37.3 tokens/sec** prompt eval (~4.5× faster than Vulkan).
  * **Integration:** OpenAI-compatible API on `127.0.0.1:8080` powering code autocompletion in **Continue (VS Code)**.
  * **Stability:** Permanently eliminates the Mesa Vulkan NVK pushbuf crash bug (`OLLAMA_VULKAN=false`).

### 🛠️ Track 3: Apple Silicon Reverse Engineering & Linux Drivers ([`hardware/`](hardware/))
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

* [`docs/opencl-rusticl-setup.md`](docs/opencl-rusticl-setup.md): Complete guide to configuring Mesa Rusticl, environment variables, compiler flags, and avoiding Vulkan pushbuf collisions.
* [`docs/MOTHERBOARD_TOPOLOGY.md`](docs/MOTHERBOARD_TOPOLOGY.md): In-depth hardware topology, bus mappings, and PCIe BAR assignments.
* [`tools/macbook_system_audit.sh`](tools/macbook_system_audit.sh): Automated diagnostic script auditing system sensors, PCIe links, and kernel drivers.
