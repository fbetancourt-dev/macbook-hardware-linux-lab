#!/usr/bin/env python3
import struct
import numpy as np
import os
import sys

def dequantize_q4_0_row(raw_bytes, K):
    nb = K // 32
    weights = np.zeros(K, dtype=np.float32)
    for b in range(nb):
        offset = b * 18
        # FP16 scale
        scale = np.frombuffer(raw_bytes[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
        qs = raw_bytes[offset+2:offset+18]
        for j in range(16):
            byte = qs[j]
            v0 = (byte & 0x0F) - 8
            v1 = (byte >> 4) - 8
            weights[b * 32 + j] = v0 * scale
            weights[b * 32 + j + 16] = v1 * scale
    return weights

print("Dequantizer test script created.")
