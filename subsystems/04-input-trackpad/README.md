# Subsystem: Keyboard & Multitouch Trackpad

* **USB Identifier:** `05ac:0263 Apple Internal Keyboard / Trackpad (MacBook Retina)` on Bus 001 Device 004
* **Controllers:** Broadcom BCM5974 Multitouch ASIC
* **Kernel Drivers:** `drivers/input/mouse/bcm5974.c` & `drivers/hid/hid-apple.c`

---

## 1. Hardware Architecture

On `MacBookPro11,3`, the physical keyboard matrix and glass trackpad communicate via internal USB 2.0 connected to the Intel xHCI root hub:

```
[ Glass Trackpad Capacitive Grid ]
              │ (Analog sensing traces)
              ▼
[ Broadcom BCM5974 Multitouch Controller ]
              │ (USB HID Interface: 05ac:0263)
              ▼
[ Intel 8-Series xHCI Host Controller (00:14.0) ]
              │ (USB Bus 1)
              ▼
[ Linux Kernel: bcm5974.ko (multitouch) + hid-apple.ko (keyboard) ]
              │
              ▼
[ /dev/input/event* (evdev subsystem) -> Libinput ]
```

---

## 2. Multitouch Packet Protocol (`bcm5974`)

The BCM5974 sends USB bulk/interrupt packets containing:
1. **Finger Count & Pressure:** Raw capacitive pressure per contact point.
2. **Contact Ellipse:** Major axis, minor axis, and orientation angle of each touch contact.
3. **Physical Click Button:** The integrated mechanical switch under the diving-board trackpad mechanism.

The Linux driver `bcm5974` translates these raw packets into standard Linux multi-touch input events (`ABS_MT_POSITION_X`, `ABS_MT_PRESSURE`, `ABS_MT_TOUCH_MAJOR`).
