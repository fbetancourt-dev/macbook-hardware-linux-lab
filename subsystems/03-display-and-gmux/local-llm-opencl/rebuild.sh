#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src/llama.cpp"
BUILD_DIR="${SRC_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin"

echo "===================================================================="
echo "⚡ REBUILDING LLAMA.CPP WITH OPENCL (MESA RUSTICL) SUPPORT"
echo "===================================================================="

if [ "${1:-}" == "--pull" ]; then
    echo "📥 Pulling latest upstream changes..."
    cd "${SRC_DIR}" && git pull --rebase
fi

if [ "${1:-}" == "--clean" ]; then
    echo "🧹 Cleaning previous build artifacts..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}" "${BIN_DIR}"

echo "🔧 Configuring CMake with OpenCL 3.0 & Haswell AVX2 optimizations..."
cmake -B "${BUILD_DIR}" -S "${SRC_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON

echo "🔨 Compiling with Ninja..."
ninja -C "${BUILD_DIR}" -j4

echo "📦 Installing binaries to ${BIN_DIR}..."
cp "${BUILD_DIR}/bin/llama-cli" "${BIN_DIR}/"
cp "${BUILD_DIR}/bin/llama-server" "${BIN_DIR}/"
cp "${BUILD_DIR}/bin/llama-bench" "${BIN_DIR}/" 2>/dev/null || true
cp "${BUILD_DIR}"/bin/libggml*.so* "${BIN_DIR}/" 2>/dev/null || true
cp "${BUILD_DIR}"/bin/libllama*.so* "${BIN_DIR}/" 2>/dev/null || true

# Symlink to ~/.local/bin for global accessibility
mkdir -p "${HOME}/.local/bin"
ln -sf "${BIN_DIR}/llama-cli" "${HOME}/.local/bin/llama-cli"
ln -sf "${BIN_DIR}/llama-server" "${HOME}/.local/bin/llama-server"

echo "✅ llama.cpp successfully rebuilt and installed!"
echo "   Binaries available in: ${BIN_DIR}"
echo "   Global shortcuts in:   ~/.local/bin/llama-cli, ~/.local/bin/llama-server"
