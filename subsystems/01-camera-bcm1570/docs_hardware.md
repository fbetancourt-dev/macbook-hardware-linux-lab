# BCM1570 Hardware Architecture & Register Specifications

## 1. PCIe Interconnect & Memory Map

The Broadcom BCM1570 is a PCIe endpoint controller (Vendor: `0x14e4`, Device: `0x1570`, Subsystem: `0x14e4:0x1570`). On the `MacBookPro11,3` host, it is connected to PCI root port at bus address `04:00.0`.

### Base Address Registers (BARs)

| BAR | Physical Base (Host) | Size | Cacheability | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **BAR 0** | `0xc1d00000` | 64 KB | Non-prefetchable (MMIO) | Core Control & Status Registers (CSR), PLL, DDR PHY control, IRQ config. |
| **BAR 2** | `0xa0000000` | 256 MB | Prefetchable (MMIO/DMA) | High-speed frame aperture. Target for direct PCIe DMA transfers to/from host RAM. |
| **BAR 4** | `0xc1c00000` | 1 MB | Non-prefetchable (SRAM) | Internal SRAM of the BCM1570 embedded processor. The host loads `firmware.bin` here. |

---

## 2. Clock Tree & Phase-Locked Loop (PLL)

The BCM1570 requires explicit clock generation sequencing before its internal core or memory bus can operate:

1. **Reference Clock Detection:**
   * Register `S2_PLL_REFCLK` (`0x04` in BAR 0).
   * Bit 3 (`S2_PLL_REFCLK_25MHZ`):
     * If set (`1`): 25 MHz oscillator input.
     * If clear (`0`): 24 MHz oscillator input (common on MacBookPro11,x).
2. **Core PLL Configuration:**
   * Register `S2_PLL_CTRL_2C` (`0x002c`): Toggle bit 6 to reset PLL.
   * Register `S2_PLL_CTRL_100` (`0x0100`): Frequency multiplier registers.
   * Target core operating frequency: **450 MHz**.
3. **Lock Verification:**
   * Register `S2_PLL_CMU_STATUS` (`0x0c`): Read until Bit 15 (`S2_PLL_CMU_STATUS_LOCKED`) asserts high.

---

## 3. DDR40 Memory Controller & PHY

The BCM1570 contains an integrated DDR memory controller to buffer uncompressed video streams directly on-chip:

* **PHY Base Address:** `0x2800` (in BAR 0).
* **Initialization Sequence:** Handled in `fthd_ddr.c`. The driver programmatically cycles through 80+ PHY registers (`fthd_ddr_phy_reg_map`) to calibrate:
  * ZQ Impedance Calibration (`S2_DDR40_PHY_ZQ_PVT_COMP_CTL`).
  * Pad Drive Strength (`S2_DDR40_PHY_DRV_PAD_CTL`).
  * Variable Delay Line (VDL) coarse and fine steps (`S2_DDR40_PHY_VDL_OVR_COARSE` / `FINE`).
  * Read Enable calibration per byte lane (`S2_DDR40_RDEN_BYTE0`, `BYTE1`).

---

## 4. Optical Sensor & MIPI CSI-2

The camera sensor placed at the top bezel of the laptop display is an OmniVision CMOS image sensor connected to the BCM1570 via MIPI CSI-2 differential lanes:

* **MIPI Data Lanes:** 1 or 2 differential data pairs + clock pair.
* **Camera Sensor Control (I2C):**
  * The sensor registers are not exposed directly to the PCIe bus of the laptop CPU.
  * Instead, the BCM1570 embedded MCU acts as the I2C master.
  * The host kernel driver issues `CISP_CMD_CH_I2C_READ` and `CISP_CMD_CH_I2C_WRITE` packets over PCIe IPC, and the BCM1570 firmware converts them into physical I2C transactions towards the sensor.
