#!/usr/bin/env bash
# macbook_system_audit.sh - Comprehensive hardware audit tool for MacBookPro11,3 on Linux

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
NC="\033[0m"

echo -e "${BOLD}${BLUE}================================================================${NC}"
echo -e "${BOLD}${GREEN} MacBookPro11,3 Hardware & Linux Driver Engineering Audit${NC}"
echo -e "${BOLD}${BLUE}================================================================${NC}"

# 1. System Platform
echo -e "\n${BOLD}[1] System Platform & Kernel${NC}"
echo "Model:      $(cat /sys/class/dmi/id/product_name 2>/dev/null || echo 'Unknown')"
echo "Board:      $(cat /sys/class/dmi/id/board_name 2>/dev/null || echo 'Unknown')"
echo "Kernel:     $(uname -r)"

# 2. CPU & Memory
echo -e "\n${BOLD}[2] Processor & Graphics Cache${NC}"
lscpu | grep -E "Model name|CPU MHz|CPU max MHz|L1d|L1i|L2|L3" || true

# 3. Graphics & gmux
echo -e "\n${BOLD}[3] Dual GPU & Display Multiplexer (gmux)${NC}"
lspci -nn | grep -E "VGA|3D" || true
if [ -d "/sys/class/backlight/gmux_backlight" ]; then
    echo -e "gmux Backlight: ${GREEN}Active${NC} (/sys/class/backlight/gmux_backlight)"
    echo "Current Brightness: $(cat /sys/class/backlight/gmux_backlight/actual_brightness 2>/dev/null || echo '?') / $(cat /sys/class/backlight/gmux_backlight/max_brightness 2>/dev/null || echo '?')"
else
    echo -e "gmux Backlight: ${YELLOW}Not detected${NC}"
fi

# 4. Apple SMC
echo -e "\n${BOLD}[4] Apple System Management Controller (SMC)${NC}"
if [ -d "/sys/devices/platform/applesmc.768" ]; then
    echo -e "SMC Interface:  ${GREEN}Active at LPC port 0x0300 (768)${NC}"
    echo "Total SMC Keys: $(cat /sys/devices/platform/applesmc.768/key_count 2>/dev/null || echo '?')"
    echo "Left Fan RPM:   $(cat /sys/devices/platform/applesmc.768/fan1_input 2>/dev/null || echo 'N/A') (Min: $(cat /sys/devices/platform/applesmc.768/fan1_min 2>/dev/null || echo '?'), Max: $(cat /sys/devices/platform/applesmc.768/fan1_max 2>/dev/null || echo '?'))"
    echo "Right Fan RPM:  $(cat /sys/devices/platform/applesmc.768/fan2_input 2>/dev/null || echo 'N/A') (Min: $(cat /sys/devices/platform/applesmc.768/fan2_min 2>/dev/null || echo '?'), Max: $(cat /sys/devices/platform/applesmc.768/fan2_max 2>/dev/null || echo '?'))"
else
    echo -e "SMC Interface:  ${YELLOW}applesmc driver not active${NC}"
fi

# 5. FaceTime HD Camera
echo -e "\n${BOLD}[5] Broadcom BCM1570 FaceTime HD Camera${NC}"
lspci -nn -s 04:00.0 || echo "Camera PCI 04:00.0 not found!"
if [ -f "/lib/firmware/facetimehd/firmware.bin" ]; then
    echo -e "Firmware Blob:  ${GREEN}Present${NC} (/lib/firmware/facetimehd/firmware.bin, $(stat -c%s /lib/firmware/facetimehd/firmware.bin) bytes)"
else
    echo -e "Firmware Blob:  ${YELLOW}Missing in /lib/firmware/facetimehd/${NC}"
fi
echo -n "V4L2 Video Nodes: "
if compgen -G "/dev/video*" > /dev/null; then
    ls -d /dev/video* | tr '\n' ' '
    echo ""
else
    echo "None active."
fi

# 6. Keyboard & Trackpad
echo -e "\n${BOLD}[6] Input Devices (Multitouch Trackpad & Keyboard)${NC}"
lsusb | grep -i apple || echo "No Apple USB input devices detected."

# 7. Audio
echo -e "\n${BOLD}[7] High Definition Audio Controllers${NC}"
lspci -nn | grep -i audio || true

# 8. Thunderbolt
echo -e "\n${BOLD}[8] Intel Thunderbolt 2 Bridge${NC}"
lspci -nn | grep -i thunderbolt || true

echo -e "\n${BOLD}${BLUE}================================================================${NC}"
