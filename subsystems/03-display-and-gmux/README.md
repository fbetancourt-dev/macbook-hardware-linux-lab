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
