#!/usr/bin/env python3
import struct
import sys
import os

# GGUF Value Types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")

def read_val(f, val_type):
    if val_type == GGUF_TYPE_UINT8:   return struct.unpack("<B", f.read(1))[0]
    if val_type == GGUF_TYPE_INT8:    return struct.unpack("<b", f.read(1))[0]
    if val_type == GGUF_TYPE_UINT16:  return struct.unpack("<H", f.read(2))[0]
    if val_type == GGUF_TYPE_INT16:   return struct.unpack("<h", f.read(2))[0]
    if val_type == GGUF_TYPE_UINT32:  return struct.unpack("<I", f.read(4))[0]
    if val_type == GGUF_TYPE_INT32:   return struct.unpack("<i", f.read(4))[0]
    if val_type == GGUF_TYPE_FLOAT32: return struct.unpack("<f", f.read(4))[0]
    if val_type == GGUF_TYPE_UINT64:  return struct.unpack("<Q", f.read(8))[0]
    if val_type == GGUF_TYPE_INT64:   return struct.unpack("<q", f.read(8))[0]
    if val_type == GGUF_TYPE_FLOAT64: return struct.unpack("<d", f.read(8))[0]
    if val_type == GGUF_TYPE_BOOL:    return bool(struct.unpack("<B", f.read(1))[0])
    if val_type == GGUF_TYPE_STRING:  return read_string(f)
    if val_type == GGUF_TYPE_ARRAY:
        elem_type, n_elems = struct.unpack("<IQ", f.read(12))
        return [read_val(f, elem_type) for _ in range(n_elems)]
    raise ValueError(f"Unknown GGUF type: {val_type}")

def inspect_qwen(filepath):
    print("=" * 80)
    print(f" Inspecting GGUF Model: {filepath}")
    print("=" * 80)
    with open(filepath, "rb") as f:
        magic, version, tensor_count, metadata_kv_count = struct.unpack("<IIQQ", f.read(24))
        print(f"Version: {version} | Tensors: {tensor_count} | Metadata keys: {metadata_kv_count}\n")
        
        metadata = {}
        for _ in range(metadata_kv_count):
            key = read_string(f)
            val_type = struct.unpack("<I", f.read(4))[0]
            val = read_val(f, val_type)
            metadata[key] = val
        
        # Print relevant architecture metadata
        arch = metadata.get("general.architecture", "unknown")
        print(f"Architecture:            {arch}")
        print(f"Block count (layers):    {metadata.get(f'{arch}.block_count')}")
        print(f"Embedding length (D):    {metadata.get(f'{arch}.embedding_length')}")
        print(f"Feed Forward length (M): {metadata.get(f'{arch}.feed_forward_length')}")
        print(f"Attention Head Count:    {metadata.get(f'{arch}.attention.head_count')}")
        print(f"Attention Head Count KV: {metadata.get(f'{arch}.attention.head_count_kv')}")
        print(f"RoPE Base Frequency:     {metadata.get(f'{arch}.rope.freq_base')}")
        print(f"RMSNorm Epsilon:         {metadata.get(f'{arch}.attention.layer_norm_rms_epsilon')}")
        alignment = metadata.get("general.alignment", 32)
        print(f"General Alignment:       {alignment} bytes\n")

        print("--- Layer 0 & 1 Tensors Inventory ---")
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, ne, ggml_type, offset))
        
        header_end = f.tell()
        data_start = (header_end + alignment - 1) & ~(alignment - 1)
        print(f"Header End: {header_end} bytes | Data Start: {data_start} bytes (offset: {data_start})\n")
        
        print(f"{'Tensor Name':<32} | {'Shape (ne)':<18} | {'Type':<6} | {'Relative Offset':<16} | {'Absolute Offset'}")
        print("-" * 92)
        for name, ne, gtype, off in tensors:
            if name.startswith("blk.0.") or name.startswith("blk.1."):
                type_name = GGML_TYPE_NAMES.get(gtype, f"Type_{gtype}")
                abs_off = data_start + off
                print(f"{name:<32} | {str(ne):<18} | {type_name:<6} | {off:<16} | {abs_off}")

if __name__ == "__main__":
    inspect_qwen("/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf")
