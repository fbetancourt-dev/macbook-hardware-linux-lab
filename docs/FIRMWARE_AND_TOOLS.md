# Firmware Extraction & Calibration Data

The Broadcom BCM1570 hardware cannot function without its proprietary firmware blob. This firmware is loaded into BAR 4 SRAM during the driver initialization sequence.

---

## 1. Firmware Origin

Due to copyright and licensing, the firmware is not distributed inside Linux kernel source trees. It must be extracted from Apple's official driver packages:
* Target file: `AppleCameraInterface` (located inside macOS `/System/Library/Extensions/AppleCameraInterface.kext/Contents/MacOS/AppleCameraInterface` or inside official Apple Boot Camp ESD update downloads).

---

## 2. Firmware Extraction Process

The community tool `facetimehd-firmware` automates the extraction:

```bash
# Clone the extractor utility
git clone https://github.com/patjak/facetimehd-firmware.git
cd facetimehd-firmware

# Download Apple update payload and extract firmware.bin
make
sudo make install
```

This installs the binary blob to:
`/lib/firmware/facetimehd/firmware.bin`

When `fthd_pci_probe()` runs, the kernel calls:
```c
request_firmware(&fw, "facetimehd/firmware.bin", &dev_priv->pdev->dev);
```

---

## 3. Sensor Calibration Data

In addition to `firmware.bin`, the BCM1570 ISP utilizes sensor-specific calibration tables:
* Sensor defect pixel maps.
* Lens shading correction (vignetting compensation).
* Noise profiles per ISO / analog gain stage.

These tables are read from `/lib/firmware/facetimehd/` or generated during sensor initialization.
