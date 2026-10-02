# Subsystem: Dual GPU & Apple gmux (Display Multiplexer)

The `MacBookPro11,3` incorporates a dual-GPU hybrid graphics architecture orchestrated by a proprietary hardware multiplexer: **Apple gmux**.

---

## 1. GPUs on the Bus

1. **Integrated GPU (iGPU):**
   * **Model:** Intel Iris Pro Graphics 5200 (Haswell Crystalwell GT3e).
   * **PCI ID:** `8086:0d26` at `00:02.0`.
   * **Dedicated Cache:** 128 MB on-package embedded DRAM (eDRAM) running at CPU ring frequency.
   * **Kernel Driver:** `i915`.
2. **Discrete GPU (dGPU):**
   * **Model:** NVIDIA GeForce GT 750M Mac Edition (Kepler architecture - GK107M core).
   * **PCI ID:** `10de:0fe9` at `01:00.0`.
   * **VRAM:** 2 GB GDDR5.
   * **Kernel Driver:** `nouveau` (open-source) or proprietary `nvidia-driver`.

---

## 2. The Apple gmux Controller

* **Hardware Identity:** Custom Lattice MachXO CPLD (or Apple custom ASIC).
* **ACPI Device:** `PNP0C02` / `APP000B` (`gmux`).
* **I/O Ports:** `0x700` - `0x70f`.
* **Kernel Driver:** `drivers/platform/x86/apple-gmux.c`.

### Functions
1. **Display Switching (eDP MUX):**
   * Physically switches the embedded DisplayPort (eDP) data lanes between the Intel display engine and the NVIDIA display engine.
   * Enables runtime power-down of the NVIDIA GPU (entering PCIe D3cold) when idle to preserve battery life.
2. **Display Backlight PWM Generation:**
   * Instead of CPU GPU backlight PWM, gmux generates the high-frequency PWM signal driving the Retina panel's LED driver.
   * Exposed in Linux as `/sys/class/backlight/gmux_backlight`.

---

## 3. High-Performance GPU Compute Modules (Mesa Rusticl / OpenCL 3.0)

Due to NVIDIA deprecating the Kepler architecture in modern CUDA and the instability of experimental Mesa Vulkan (NVK), all accelerated compute workloads leverage **OpenCL 3.0 via Mesa Rusticl**:

1. [`compute-kepler-opencl/`](compute-kepler-opencl/): Standalone C OpenCL 3.0 kernels for 2D Heat Diffusion and Gravitational N-body simulations.
2. [`simulink_opencl_bridge/`](simulink_opencl_bridge/): Generic C-MEX Level-2 S-Function bridge connecting MATLAB Simulink directly to OpenCL compute queues.
3. [`simulink_multizone_thermal/`](simulink_multizone_thermal/): 2D multizone thermal field finite-difference solver with anti-windup PID temperature regulation.
4. [`simulink_inverted_pendulum/`](simulink_inverted_pendulum/): Real-time ensemble simulation of 1,024 nonlinear Cart-Pole pendulums with Åström-Furuta energy swing-up, LQR stabilization, and interactive keyboard steering GUI.
5. [`local-llm-opencl/`](local-llm-opencl/): Production deployment of `llama.cpp` using Mesa Rusticl OpenCL 3.0, systemd daemon management, and Continue autocompletion integration.

