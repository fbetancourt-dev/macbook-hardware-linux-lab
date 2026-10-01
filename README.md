# MacBook Broadcom BCM1570 FaceTime HD Camera Research

Technical research, architectural documentation, and Linux driver engineering for the **Broadcom BCM1570 (PCIe ID 14e4:1570)** FaceTime HD Camera found in Apple MacBook Pro / MacBook Air models (specifically tested and analyzed on `MacBookPro11,3` running Linux kernel 7.x).

---

## 📌 Target Hardware Profile

| Parameter | Specification |
| :--- | :--- |
| **Host System** | MacBookPro11,3 (Retina 15-inch, Mid 2014) |
| **PCI Device** | `04:00.0 Multimedia controller [0480]` |
| **Vendor & Device ID** | `14e4:1570` (Broadcom Inc. and subsidiaries) |
| **Interconnect** | Direct PCI Express x1 (Not USB / Not UVC) |
| **SoC / Controller** | Broadcom BCM1570 ISP & PCIe Bridge |
| **Camera Sensor** | OmniVision CMOS Sensor via MIPI CSI-2 |
| **Sensor Control** | Internal I2C / SCCB bus managed by BCM1570 firmware |
| **PCI BAR 0** | `0xc1d00000` (64 KB) - Control & Status Registers (CSR), PLL, Clocks |
| **PCI BAR 2** | `0xa0000000` (256 MB) - High-speed DMA streaming aperture |
| **PCI BAR 4** | `0xc1c00000` (1 MB) - Internal SRAM for firmware loading |

---

## 🧠 Architectural Overview

Unlike typical PC webcams that interface via USB Video Class (UVC), the BCM1570 is a full standalone Image Signal Processor (ISP) SoC wired directly to the system's PCIe bus:

```
[ OmniVision CMOS Sensor ]
           │ (MIPI CSI-2 bus)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Broadcom BCM1570 SoC (ISP)                                  │
│                                                             │
│  [ Embedded Microcontroller Core ]                          │
│     └── Executes firmware loaded into SRAM (BAR 4)          │
│     └── Manages internal RTOS & IPC message queues          │
│                                                             │
│  [ Hardware ISP Processing Pipeline ]                       │
│     ├── Debayering (Raw Bayer to YUV/RGB)                   │
│     ├── 3A Algorithms: Auto-Exposure (AE), AWB, AF          │
│     ├── Noise reduction, Lens shading, Color matrix (CCM)   │
│     └── Direct I2C master controlling the optical sensor    │
│                                                             │
│  [ DDR Memory Controller & PCIe DMA Engine ]                │
│     └── Direct DMA frame transmission to Host RAM (BAR 2)   │
└─────────────────────────────────────────────────────────────┘
           │ (PCIe Bus: BAR0, BAR2, BAR4)
           ▼
[ Linux Host CPU: Kernel Driver (bcwc_pcie) -> /dev/video0 ]
```

---

## 📂 Repository Structure

* `docs/`
  * [`HARDWARE_ARCHITECTURE.md`](docs/HARDWARE_ARCHITECTURE.md): Deep-dive into BCM1570 registers, PLL, clocks, DDR controller, and PCIe regions.
  * [`DRIVER_INTERNALS.md`](docs/DRIVER_INTERNALS.md): Analysis of the driver source (`src/`), IPC channels, command table, and V4L2 bridge.
  * [`FIRMWARE_AND_TOOLS.md`](docs/FIRMWARE_AND_TOOLS.md): How firmware extraction and calibration files work.
* `src/`: Source code of the reverse-engineered Linux kernel driver (`bcwc_pcie`):
  * `fthd_reg.h`: Hardware register definitions.
  * `fthd_hw.c`: PLL, clocking, DDR PHY and hardware bringup.
  * `fthd_isp.c`: Firmware loader, IPC channels and ISP commands.
  * `fthd_drv.c`: PCI driver probe, IRQ handling and message routing.
  * `fthd_v4l2.c`: Video4Linux2 subsystem bridge.
* `tools/`: Diagnostic scripts, firmware extraction helpers, and testing utilities.

---

## 🚀 Goals & Research Directions

1. **Hardware & Protocol Reverse Engineering:** Document all known registers, IPC channels, and undocumented firmware commands.
2. **Modern Linux Kernel Compatibility:** Maintain and adapt the driver for kernel 6.x and 7.x (fixing video buffer freeze regressions and V4L2 API transitions).
3. **Driver Customization:**
   * Expose additional controls through sysfs / V4L2 controls (manual gain, exposure, raw sensor access).
   * Tap into `TERMINAL` and `DEBUG` firmware channels to read live telemetry from the BCM1570 micro.
