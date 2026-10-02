# Hardware Subsystem: Dual GPU & Apple gmux (Display Multiplexer)

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
   * **Kernel Driver:** `nouveau` (open-source DRM kernel driver).

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

## 3. High-Performance GPU Compute

Accelerated scientific simulations and local AI workloads utilizing the GT 750M have been centralized in the top-level **[`compute/`](../../compute/)** directory:

* [`compute/opencl-basics/`](../../compute/opencl-basics/): Standalone C OpenCL 3.0 kernels (2D Heat Diffusion and Gravitational N-body).
* [`compute/simulink-opencl-bridge/`](../../compute/simulink-opencl-bridge/): Generic C-MEX Level-2 S-Function bridge for MATLAB Simulink.
* [`compute/thermal-control/`](../../compute/thermal-control/): 2D multizone thermal field solver with anti-windup PID control.
* [`compute/cartpole/`](../../compute/cartpole/): Real-time ensemble simulation of 1,024 nonlinear Cart-Pole pendulums and interactive steering GUI.
* [`compute/local-llm/`](../../compute/local-llm/): Production deployment of `llama.cpp` using Mesa Rusticl OpenCL 3.0 for Continue.
