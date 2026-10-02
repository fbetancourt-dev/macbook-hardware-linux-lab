# MacBook Pro Linux Hardware Engineering & GPU Compute Lab

Low-level hardware reverse engineering, Linux kernel driver research, and high-performance GPU computing on legacy silicon (**NVIDIA Kepler OpenCL 3.0 via Mesa Rusticl, MATLAB Simulink C-MEX, and local LLM acceleration**) for the **Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)**.

---

## 🎯 Mission & Philosophy

This laboratory demystifies every piece of silicon, microcontroller, and firmware layer inside the 2014 MacBook Pro under modern Linux kernels (6.x/7.x) and modern Wayland desktop environments.

Rather than letting older hardware become obsolete, we focus on:
1. **High-Performance GPU Computing without Proprietary CUDA:**
   - Harnessing the **NVIDIA GeForce GT 750M (384 Kepler Cores, 2 GB GDDR5)** through modern **OpenCL 3.0 on Mesa Rusticl**.
   - Massively parallel simulations in **MATLAB Simulink R2025b** via custom Level-2 C-MEX S-Functions (e.g. 1,024 parallel nonlinear inverted pendulums, 2D transient thermal field solvers).
   - Local LLM inference acceleration with **`llama.cpp`** and **Continue** in VS Code at ~12.6 tokens/sec with rock-solid system stability.
2. **Proprietary Hardware Reverse Engineering:**
   - Mapping PCIe BARs, shared memory IPC ringbuffers, and bus protocols (PCIe, LPC, I2C, SPI, ACPI, USB).
   - Documenting custom chips: **Apple gmux CPLD** (eDP display switching & PWM), **Apple SMC** (661 thermal/voltage/fan keys on LPC bus `0x300`), and the **Broadcom BCM1570 ISP** FaceTime HD camera.
3. **Linux Kernel Driver Engineering:**
   - Maintaining and validating drivers across kernel updates (`apple_gmux`, `applesmc`, `bcwc_pcie`, `bcm5974`, `snd_hda_intel`, `nouveau`).

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

## 📂 Subsystems Directory

| Directory | Subsystem | Controller / Hardware | Scope & Implementations |
| :--- | :--- | :--- | :--- |
| [`subsystems/03-display-and-gmux/`](subsystems/03-display-and-gmux/) | **Dual GPU, gmux & OpenCL Compute** | Intel Iris Pro 5200 + Nvidia GT 750M + Apple gmux CPLD | **GPU Compute Lab:** 1,024-pendulum ensemble simulation (`simulink_inverted_pendulum`), 2D thermal anti-windup control (`simulink_multizone_thermal`), `llama.cpp` OpenCL inference daemon (`local-llm-opencl`), and `apple_gmux` backlight control. |
| [`subsystems/02-smc-and-thermals/`](subsystems/02-smc-and-thermals/) | **System Management & Thermals** | Apple SMC (Renesas H8S/custom MCU on LPC bus `0x300`) | Hardware register protocol, dual-fan tachometer PID control, CPU/GPU thermal diode monitoring via `applesmc`. |
| [`subsystems/01-camera-bcm1570/`](subsystems/01-camera-bcm1570/) | **FaceTime HD Camera** | Broadcom BCM1570 PCIe ISP (`14e4:1570`) + OmniVision CMOS | PCIe BAR layout reverse engineering, firmware loading protocol, and V4L2 kernel driver (`bcwc_pcie` / `facetimehd`). |
| [`subsystems/04-input-trackpad/`](subsystems/04-input-trackpad/) | **Keyboard & Trackpad** | Broadcom BCM5974 Multitouch Controller (USB `05ac:0263`) | Multi-finger pressure gestures, SPI/USB packet decoding (`bcm5974`, `hid-apple`). |
| [`subsystems/05-audio-codec/`](subsystems/05-audio-codec/) | **Audio Subsystem** | Cirrus Logic CS4208 + Intel Lynx Point HD Audio (`8086:8c20`) | Multi-channel DAC/ADC path routing, SPDIF optical output, and jack sensing (`snd_hda_intel`, `snd_hda_codec_cirrus`). |
| [`subsystems/06-thunderbolt-pcie/`](subsystems/06-thunderbolt-pcie/) | **Thunderbolt 2** | Intel DSL5520 Falcon Ridge 4C (`8086:156d/156c`) | PCIe hotplug tunneling, domain controller power states, and security levels. |

---

## 🛠️ Global Tools & Diagnostics

* [`tools/macbook_system_audit.sh`](tools/macbook_system_audit.sh): Complete hardware topology, PCIe bus, SMC sensor and driver audit script.
