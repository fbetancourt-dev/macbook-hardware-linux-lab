# MacBookPro11,3 Motherboard Topology & Bus Architecture

## 1. High-Level Chipset Architecture

The 15-inch Retina MacBook Pro (Mid 2014, `MacBookPro11,3`) is architected around two primary processing complexes:
1. **CPU & System Agent (Intel Haswell Crystalwell i7-4870HQ):**
   * Features 128 MB on-package eDRAM (L4 cache / Iris Pro Graphics cache).
   * Directly hosts the DDR3L-1600 memory controller (16 GB integrated).
   * Generates 16 PCI Express 3.0 lanes dedicated to the discrete graphics controller (NVIDIA GeForce GT 750M Mac Edition).
   * Connects to the Platform Controller Hub (PCH) via Direct Media Interface 2.0 (DMI 2.0 @ 5.0 GT/s, x4 link).
2. **Platform Controller Hub (Intel HM87 Lynx Point):**
   * Manages PCIe Gen 2.0 downstream root ports, USB 3.0 xHCI controller, High Definition Audio (HDA), Low Pin Count (LPC) legacy interface, SPI flash interface, and SMBus/I2C.

---

## 2. PCI Express Bus Allocation

Audited on live hardware:

```
[Bus 00] Host Controller & Chipset Root Ports
 ├── 00:00.0 Host bridge: Intel Crystal Well DRAM Controller [8086:0d04]
 ├── 00:01.0 PCIe Bridge x16 -> [Bus 01:00.0] NVIDIA GeForce GT 750M (GK107M)
 ├── 00:01.1 PCIe Bridge x8  -> [Bus 06:00.0] Intel DSL5520 Thunderbolt 2 Bridge
 ├── 00:02.0 Integrated GPU: Intel Iris Pro 5200 [8086:0d26]
 ├── 00:03.0 Audio: Intel Crystal Well HD Audio Controller [8086:0d0c]
 ├── 00:14.0 USB: Intel 8 Series xHCI Controller [8086:8c31]
 ├── 00:16.0 Management Engine: Intel MEI Controller #1 [8086:8c3a]
 ├── 00:1b.0 Audio: Intel 8 Series HD Audio Controller [8086:8c20]
 ├── 00:1c.0 PCIe Root Port #1 -> [Bus 03:00.0] Broadcom BCM4360 802.11ac Wi-Fi [14e4:43a0]
 ├── 00:1c.2 PCIe Root Port #3 -> [Bus 04:00.0] Broadcom BCM1570 FaceTime HD Camera [14e4:1570]
 ├── 00:1c.3 PCIe Root Port #4 -> [Bus 05:00.0] Samsung NVMe SSD Controller [144d:a80c]
 ├── 00:1f.0 ISA Bridge: Intel HM87 LPC Controller [8086:8c4b] (Links to Apple SMC)
 └── 00:1f.3 SMBus: Intel 8 Series SMBus Controller [8086:8c22]
```

---

## 3. The Low Pin Count (LPC) Bus & Apple SMC

The **Apple System Management Controller (SMC)** is an embedded microcontroller wired to the PCH via the LPC bus at classic x86 I/O port address space:
* **Base I/O Ports:** `0x0300` - `0x031f`
* **Driver in Linux:** `drivers/hwmon/applesmc.c`
* **Functions:**
  * Power sequencing, battery management (SBS over internal I2C/SMBus).
  * Dual fan tachometer reading and PWM target control (`fan1` Left, `fan2` Right).
  * Thermal diode readings across CPU cores, GPU die, memory, heatpipe, palm rest, and battery cells.
  * Ambient Light Sensor (ALS) readings (`light` sysfs entry).
  * Keyboard backlight LED pulse-width modulation (`leds/smc::kbd_backlight`).

---

## 4. Display Multiplexer: Apple gmux

Because the laptop features both an integrated Intel GPU (energy efficient) and a discrete NVIDIA GPU (high performance), Apple engineered a custom hardware multiplexer:
* **Hardware Chip:** Lattice CPLD (or custom ASIC).
* **Control Interface:** ACPI PNP device `APP000B` exposed at I/O ports `0x700` - `0x70f`.
* **Operation:**
  * Connects the embedded DisplayPort (eDP) lines of the Retina LCD panel to either the Intel or NVIDIA display engine.
  * Controls the display backlight inverter / LED driver via hardware PWM (`gmux_backlight`).
