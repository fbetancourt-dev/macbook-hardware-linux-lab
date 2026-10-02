# Engineering Guide: OpenCL 3.0 via Mesa Rusticl on NVIDIA Kepler (GT 750M)

A practical engineering guide for setting up and utilizing **hardware-accelerated OpenCL 3.0** on legacy **NVIDIA Kepler (GK107 / GeForce GT 750M)** architecture using **Mesa Rusticl** under modern Linux kernels (6.x/7.x) and Wayland desktop environments.

---

## 🎯 Architecture Context

```
 ┌─────────────────────────────────────────────────────────────┐
 │    User-Space Application (MATLAB MEX, llama.cpp, C App)    │
 └──────────────────────────────┬──────────────────────────────┘
                                │ OpenCL 3.0 API calls (clEnqueueNDRangeKernel)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                ocl-icd Loader (/usr/lib/libOpenCL.so)       │
 └──────────────────────────────┬──────────────────────────────┘
                                │ ICD Dispatch (/etc/OpenCL/vendors/rusticl.icd)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           Mesa Rusticl (Rust OpenCL 3.0 Implementation)     │
 │                    libRusticlOpenCL.so                      │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Gallium3D Pipe Driver / SPIR-V Compiler
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │               Nouveau Gallium3D (nouveau_dri.so)            │
 └──────────────────────────────┬──────────────────────────────┘
                                │ DRM IOCTLs (/dev/dri/renderD128)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │               Linux Kernel DRM Driver (nouveau.ko)          │
 └──────────────────────────────┬──────────────────────────────┘
                                │ PCIe Gen3 x16 Bus
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │      NVIDIA GeForce GT 750M Mac Edition (Kepler GK107)      │
 │      - 384 FP32 Cores (2 SMX Engines @ 926 MHz)             │
 │      - 2,048 MB GDDR5 VRAM (64-bit @ 2500 MHz)              │
 └─────────────────────────────────────────────────────────────┘
```

---

## ❓ Why Rusticl Instead of CUDA or Vulkan?

1. **CUDA Deprecation:** NVIDIA dropped Compute Capability 3.0 (Kepler) starting in CUDA 11. Modern frameworks (PyTorch, TensorFlow, Triton) no longer provide binaries or kernels for Kepler.
2. **Proprietary Driver Incompatibility:** The legacy `nvidia-470` driver fails to compile on modern Linux kernels (6.8+) and breaks Wayland / Mutter window compositing.
3. **Mesa Vulkan (NVK) Instability:** The experimental Kepler Vulkan backend in Mesa suffers from pushbuf submission deadlocks when running compute shaders simultaneously with desktop windows, triggering kernel FIFO channel aborts:
   ```
   nouveau 0000:01:00.0: kernel rejected pushbuf: No such device
   nouveau 0000:01:00.0: fifo: ch 6 [mutter] killed
   ```
4. **Rusticl OpenCL 3.0 Stability:** Mesa Rusticl targets the stable Gallium compute pipeline. It executes compute shaders stably with zero GPU hangs and full access to the 384 CUDA cores and GDDR5 VRAM.

---

## 🛠️ System Prerequisites & Installation

### 1. Install OpenCL Runtime & Development Headers
On Ubuntu / Debian systems:
```bash
sudo apt update
sudo apt install -y mesa-opencl-icd ocl-icd-opencl-dev clinfo
```

### 2. Enable Rusticl for Nouveau
Rusticl requires explicit driver enablement via an environment variable:
```bash
export RUSTICL_ENABLE=nouveau
```
To persist this across sessions, add it to `~/.bashrc` or `~/.profile`:
```bash
echo 'export RUSTICL_ENABLE=nouveau' >> ~/.bashrc
```

### 3. Verification
Verify that Rusticl detects the Kepler GPU:
```bash
export RUSTICL_ENABLE=nouveau
clinfo -l
```
**Expected Output:**
```
Platform #0: rusticl
 `-- Device #0: NV107
```

Detailed query:
```bash
clinfo | grep -E "Platform Name|Device Name|OpenCL C version|Device OpenCL C version|Max compute units"
```

---

## 💻 Compiling OpenCL Applications

### Standalone C Programs
```bash
gcc -O3 -Wall \
    -o my_simulation main.c \
    -lOpenCL -lm
```

Run with Rusticl enabled:
```bash
RUSTICL_ENABLE=nouveau ./my_simulation
```

### MATLAB Level-2 C-MEX S-Functions
Within MATLAB:
```matlab
setenv('RUSTICL_ENABLE', 'nouveau');
mex -O CFLAGS="\$CFLAGS -std=c99 -Wall" ...
    LDFLAGS="\$LDFLAGS -lOpenCL" ...
    sfun_opencl_kernel.c
```

### Native llama.cpp Compilation
```bash
cmake -B build \
    -DGGML_OPENCL=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

---

## 🛡️ Best Practices & Stability Guidelines

1. **Isolate Vulkan Compute:** If using Ollama alongside Rusticl, set `OLLAMA_VULKAN=false` in its systemd service to avoid NVK pushbuf collisions.
2. **Workgroup Sizing:** Kepler GK107 SMX architecture groups threads into warps of 32. Use local workgroup sizes that are multiples of 32 (typically `64` or `128`) for optimal occupancy.
3. **Memory Transfers:** Minimize PCIe host-to-device transfers by keeping iterative simulation state inside on-device `cl_mem` buffers across time steps.
