# Driver Internals & IPC Protocol

The Linux driver (`bcwc_pcie`) manages the bridge between the Linux V4L2 subsystem and the proprietary BCM1570 firmware protocol.

---

## 1. Module Bringup Flow

```
pci_register_driver(&fthd_pci_driver)
  │
  ├── fthd_pci_probe()
  │     ├── pci_enable_device()
  │     ├── pci_request_regions()
  │     ├── Map BAR 0 (CSR MMIO)
  │     ├── Map BAR 2 (DMA MMIO)
  │     ├── Map BAR 4 (Firmware SRAM)
  │     ├── fthd_hw_init() -> Setup PLL & DDR PHY
  │     ├── isp_load_firmware() -> Copy firmware.bin into BAR 4
  │     ├── fthd_irq_install() -> Request PCI IRQ / MSI
  │     └── fthd_v4l2_register() -> Expose /dev/video0
```

---

## 2. Shared Memory IPC Channels

Communication between the host kernel driver and the BCM1570 firmware is based on circular ringbuffers defined in `fthd_ringbuf.c` and managed in `fthd_isp.c`.

When the firmware boots, it writes a channel descriptor table into memory at offset `0x0128`:

| Channel Name | Direction | Purpose |
| :--- | :--- | :--- |
| **`TERMINAL`** | Target -> Host | ASCII text log stream from the embedded RTOS. Handled in `terminal_handler()` and logged with `pr_info("FWMSG: %.*s")`. |
| **`DEBUG`** | Target -> Host | Low-level execution trace and register debug data. Accessible via debugfs `channel_debug`. |
| **`SHAREDMALLOC`** | Bi-directional | Dynamic heap allocator coordination between host and ISP. |
| **`IO`** | Host -> Target | Command dispatch ring. The host pushes `CISP_CMD_*` requests here. |
| **`IO_T2H`** | Target -> Host | Command completion and response notifications from the ISP. |
| **`BUF_H2T`** | Host -> Target | Video buffer descriptor queue provided by the host for incoming frames. |
| **`BUF_T2H`** | Target -> Host | Completed frame buffers delivered back to host with timestamps and lengths. |

---

## 3. ISP Command Set (`CISP_CMD_*`)

Defined in `fthd_isp.h`:

### Core Lifecycle Commands
* `CISP_CMD_START` (`0x0`): Wake up the ISP core.
* `CISP_CMD_STOP` (`0x1`): Stop streaming and idle the sensor.
* `CISP_CMD_RESET` (`0x2`): Soft reset ISP processing pipeline.
* `CISP_CMD_POWER_DOWN` (`0xa`): Enter ultra-low power standby.

### Sensor Channel Configuration
* `CISP_CMD_CH_START` (`0x100`) / `STOP` (`0x101`).
* `CISP_CMD_CH_CAMERA_CONFIG_SELECT` (`0x107`): Choose optical configuration.
* `CISP_CMD_CH_I2C_READ` (`0x10b`) / `WRITE` (`0x10c`): Read/Write raw registers of the CMOS sensor over I2C.
* `CISP_CMD_CH_CAMERA_MIPI_FREQ_SELECT` (`0x11b`): Set MIPI clock rate.

### 3A Engine (Auto-Exposure, White Balance, Focus)
* **Auto Exposure:**
  * `CISP_CMD_CH_AE_START` (`0x200`) / `STOP` (`0x201`).
  * `CISP_CMD_CH_AE_FRAME_RATE_MAX_SET` (`0x208`) / `MIN_SET` (`0x20a`).
  * `CISP_CMD_CH_AE_GAIN_CAP_SET` (`0x20c`).
  * `CISP_CMD_CH_AE_NOISE_REDUCTION_CONTROL_PARAM_SET` (`0x211`).
* **Auto White Balance:**
  * `CISP_CMD_CH_AWB_START` (`0x300`) / `STOP` (`0x301`).
  * `CISP_CMD_CH_AWB_CCT_MANUAL` (`0x305`): Force manual correlated color temperature.
  * `CISP_CMD_CH_AWB_CCM_WARMUP_MATRIX_SET` (`0x309`): Color Correction Matrix.
* **Auto Focus:**
  * `CISP_CMD_CH_AF_START` (`0x400`) / `FOCUS_POS_GET` (`0x404`).

---

## 4. Video4Linux2 Subsystem Integration (`fthd_v4l2.c`)

The driver registers a standard V4L2 device node (`/dev/video0`) using `videobuf2-dma-sg`:

* **Formats:** Native uncompressed format is `V4L2_PIX_FMT_YUYV` (4:2:2 uncompressed). Fallback format `V4L2_PIX_FMT_YVYU`.
* **Resolution:** Default `1280x720` at 30 fps (MacBook Pro FaceTime HD).
* **Controls:** Standard controls mapped via `v4l2_ctrl_new_std`:
  * `V4L2_CID_BRIGHTNESS`
  * `V4L2_CID_CONTRAST`
  * `V4L2_CID_SATURATION`
  * `V4L2_CID_HUE`
  * `V4L2_CID_AUTO_WHITE_BALANCE`
