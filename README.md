# MacBook Pro Hardware & Linux Driver Engineering Lab

Technical research, low-level reverse engineering, hardware architecture documentation, and Linux kernel driver development for the **Apple MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)**.

---

## 🎯 Mission & Philosophy

The goal of this project is to demystify every piece of silicon, microcontroller, and firmware layer inside the laptop down to the Linux kernel drivers and user-space APIs. 

By understanding the exact hardware topologies, PCIe BAR mappings, shared memory IPC ringbuffers, and bus protocols (PCIe, LPC, I2C, SPI, ACPI, USB), we can:
1. **Master Linux driver development** through real, complex proprietary hardware.
2. **Revitalize and maintain hardware** on modern Linux kernels (6.x & 7.x).
3. **Build ultra-robust user-space applications** that interact directly and cleanly with system APIs, kernel interfaces, and hardware registers.

---

## 💻 Hardware Profile (`MacBookPro11,3`)

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

| Directory | Subsystem | Controller / Hardware | Linux Driver / Subsystem |
| :--- | :--- | :--- | :--- |
| [`subsystems/01-camera-bcm1570/`](subsystems/01-camera-bcm1570/) | **FaceTime HD Camera** | Broadcom BCM1570 PCIe ISP (`14e4:1570`) + OmniVision CMOS | `bcwc_pcie` / `facetimehd` (`/dev/video0`) |
| [`subsystems/02-smc-and-thermals/`](subsystems/02-smc-and-thermals/) | **System Management** | Apple SMC (Renesas H8S/custom MCU on LPC bus `0x300`) | `applesmc` (`/sys/devices/platform/applesmc.768`) |
| [`subsystems/03-display-and-gmux/`](subsystems/03-display-and-gmux/) | **Dual GPU & Display** | Intel Iris Pro 5200 + Nvidia GT 750M + Apple gmux CPLD | `i915`, `nouveau` / `nvidia`, `apple_gmux` |
| [`subsystems/04-input-trackpad/`](subsystems/04-input-trackpad/) | **Keyboard & Trackpad** | Broadcom BCM5974 Multitouch Controller (USB `05ac:0263`) | `bcm5974`, `hid-apple` (`evdev`) |
| [`subsystems/05-audio-codec/`](subsystems/05-audio-codec/) | **Audio Subsystem** | Cirrus Logic CS4208 + Intel Lynx Point HD Audio (`8086:8c20`) | `snd_hda_intel`, `snd_hda_codec_cirrus` |
| [`subsystems/06-thunderbolt-pcie/`](subsystems/06-thunderbolt-pcie/) | **Thunderbolt 2** | Intel DSL5520 Falcon Ridge 4C (`8086:156d/156c`) | `thunderbolt`, PCIe hotplug |

---

## 🛠️ Global Tools & Diagnostics

* [`tools/macbook_system_audit.sh`](tools/macbook_system_audit.sh): Complete hardware topology, PCIe bus, SMC sensor and driver audit script.
