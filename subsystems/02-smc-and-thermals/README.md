# Subsystem: Apple System Management Controller (SMC) & Thermal Control

The **System Management Controller (SMC)** is the central hardware housekeeping brain of Apple laptops.

---

## 1. Hardware Architecture

* **Hardware Controller:** Custom microcontroller (Renesas H8S derivative) running Apple proprietary firmware.
* **Bus Interface:** Intel Lynx Point PCH LPC (Low Pin Count) bus.
* **Port Addresses:** Base port `0x0300`, Command/Status port `0x0304`.
* **Kernel Driver:** `drivers/hwmon/applesmc.c`.
* **Sysfs Platform Device:** `/sys/devices/platform/applesmc.768` (Note: 768 in decimal is `0x0300` in hex!).

---

## 2. The SMC Key-Value Protocol

The SMC firmware exposes internal sensors, registers, and controls through a 4-character ASCII key (FourCC) protocol:
* Total Keys detected on this machine: **661 keys**.

### Key Data Types
* `ui8`: 8-bit unsigned integer (e.g., fan count `FNum`).
* `ui16`: 16-bit unsigned integer.
* `fpe2`: 16-bit fixed point (14-bit integer, 2-bit fractional). Used for fan RPM (`F0Ac`, `F1Ac`).
* `sp78`: Signed fixed-point temperature format (8 bits sign/int, 8 bits fractional).
* `ch8*`: Byte string array.

---

## 3. Key Sensor Map on `MacBookPro11,3`

### Fans
* `FNum`: Total fans detected (`2` - Left and Right).
* `F0Ac`: Fan 0 Actual RPM (Left Fan).
* `F0Tg`: Fan 0 Target RPM.
* `F0Mn` / `F0Mx`: Fan 0 Minimum and Maximum allowed RPM.
* `F1Ac`: Fan 1 Actual RPM (Right Fan).
* `F1Tg`: Fan 1 Target RPM.

### Temperature Diodes
* `TC0P`: CPU Proximity temperature.
* `TC0D`: CPU Die temperature.
* `TG0D`: GPU Die temperature (NVIDIA GK107M).
* `TM0P`: Memory module proximity.
* `Th0H`: Heatpipe temperature.
* `TB0T`: Battery temperature.

### Ambient Sensors & LEDs
* `ALV0`: Ambient Light Sensor (ALS) valid flag.
* `ALRV`: Ambient Light Sensor raw lux reading.
* `LKSB`: Keyboard backlight LED brightness duty cycle.
