#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/bin/llama-server"
DEFAULT_MODEL="${SCRIPT_DIR}/models/qwen2.5-coder-1.5b-base.gguf"
MODEL="${1:-${DEFAULT_MODEL}}"
NGL="${2:-99}"
PORT="${3:-8080}"

if [ ! -f "${MODEL}" ]; then
    echo "❌ Model file not found: ${MODEL}"
    exit 1
fi

echo "===================================================================="
echo "⚡ STARTING LLAMA-SERVER WITH MESA RUSTICL OPENCL ACCELERATION"
echo "   Model:       ${MODEL}"
echo "   GPU Layers:  ${NGL} (Target: NVIDIA GT 750M via Rusticl)"
echo "   Endpoint:    http://127.0.0.1:${PORT}/v1"
echo "===================================================================="

export RUSTICL_ENABLE=nouveau
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

exec "${BIN}" \
    --model "${MODEL}" \
    --n-gpu-layers "${NGL}" \
    --ctx-size 4096 \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --threads 4
