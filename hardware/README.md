# Hardware Lab: Proprietary Apple Hardware & Linux Drivers

Low-level reverse engineering, hardware architecture documentation, and Linux kernel driver development for proprietary ASICs, CPLDs, and PCIe peripherals in the **MacBook Pro (Retina, 15-inch, Mid 2014 - `MacBookPro11,3`)**.

---

## 📂 Subsystem Modules

| Subsystem | Hardware / Chipset | Bus / Protocol | Linux Driver / Interface |
| :--- | :--- | :--- | :--- |
| **[`camera-bcm1570/`](camera-bcm1570/)** | Broadcom BCM1570 PCIe ISP + OmniVision CMOS | PCIe Gen2 x1, DDR ringbuffer IPC | `bcwc_pcie` (`/dev/video0`) |
| **[`smc-thermals/`](smc-thermals/)** | Apple SMC (Renesas H8S/custom MCU) | LPC I/O Ports `0x300` - `0x31f` | `applesmc` (`/sys/devices/platform/applesmc.768`) |
| **[`display-gmux/`](display-gmux/)** | Custom Lattice MachXO CPLD | I/O Ports `0x700` - `0x70f` | `apple_gmux` (`/sys/class/backlight/gmux_backlight`) |
| **[`input-trackpad/`](input-trackpad/)** | Broadcom BCM5974 Multitouch | USB HID `05ac:0263` | `bcm5974`, `hid-apple` (`evdev`) |
| **[`audio-cs4208/`](audio-cs4208/)** | Cirrus CS4208 + Lynx Point HDA | Intel HD Audio bus (`8086:8c20`) | `snd_hda_intel`, `snd_hda_codec_cirrus` |
| **[`thunderbolt-pcie/`](thunderbolt-pcie/)** | Intel DSL5520 Falcon Ridge 4C | PCIe Gen2 x4 (20 Gbps) | `thunderbolt`, PCIe hotplug (`pciehp`) |

---

## 🛠️ Global Hardware Tools

* [`../../tools/macbook_system_audit.sh`](../../tools/macbook_system_audit.sh): Automated hardware topology, PCIe bus, SMC sensor, and driver diagnostic script.
* [`../../docs/MOTHERBOARD_TOPOLOGY.md`](../../docs/MOTHERBOARD_TOPOLOGY.md): Comprehensive motherboard bus layout and PCIe BAR mapping.
