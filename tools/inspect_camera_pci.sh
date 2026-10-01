#!/usr/bin/env bash
# inspect_camera_pci.sh - Quick diagnostic tool for Broadcom BCM1570 FaceTime HD camera

set -euo pipefail

echo "========================================================"
echo " FaceTime HD Camera (BCM1570) Hardware Diagnostic"
echo "========================================================"

echo -e "\n[1] PCI Device Info (14e4:1570):"
lspci -vnn -s 04:00.0 || echo "Device 04:00.0 not found!"

echo -e "\n[2] Loaded Kernel Modules:"
lsmod | grep -E "facetimehd|bcwc_pcie|uvcvideo|videobuf2" || echo "No camera kernel modules currently active."

echo -e "\n[3] V4L2 Device Nodes:"
if compgen -G "/dev/video*" > /dev/null; then
    ls -la /dev/video*
else
    echo "No /dev/video* nodes found in system."
fi

echo -e "\n[4] Firmware status in /lib/firmware/facetimehd:"
if [ -d "/lib/firmware/facetimehd" ]; then
    ls -la /lib/firmware/facetimehd/
else
    echo "Directory /lib/firmware/facetimehd does not exist yet."
fi

echo -e "\n[5] System DMI Product Name:"
cat /sys/class/dmi/id/product_name || echo "DMI info unavailable"

echo "========================================================"
