# Subsystem: Broadcom BCM1570 FaceTime HD Camera

* **PCI Identifier:** `14e4:1570` on `04:00.0`
* **Linux Driver:** `bcwc_pcie` (`facetimehd`) -> `/dev/video0`

## Documentation & Code Links

1. [`docs_hardware.md`](docs_hardware.md): Deep-dive into BCM1570 silicon, PLL clock tree (24 MHz vs 25 MHz to 450 MHz), DDR40 PHY calibration registers, and MIPI CSI-2 / I2C sensor links.
2. [`docs_driver.md`](docs_driver.md): Kernel driver internals, memory-mapped I/O, IPC ringbuffer channels (`TERMINAL`, `DEBUG`, `BUF_H2T`, `BUF_T2H`), and 3A image signal processor commands.
3. [`docs_firmware.md`](docs_firmware.md): Firmware extraction from Apple kext packages and sensor calibration tables.
4. [`src/`](src/): Full C source code of the kernel module.
5. [`inspect_camera_pci.sh`](inspect_camera_pci.sh): Diagnostic tool to test device status, BARs, and driver loading.
