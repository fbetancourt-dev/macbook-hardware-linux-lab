# Simulink GPU OpenCL Bridge (NVIDIA Kepler GT 750M Acceleration)

A high-performance C-MEX S-Function bridge enabling **Simulink** models in **MATLAB R2025b** to dispatch parallel physical simulations directly to the **384 CUDA Cores** of the **NVIDIA GeForce GT 750M (GK107)** using **Mesa Rusticl OpenCL 3.0** on Linux.

---

## 🎯 Architecture Overview

```
 ┌────────────────────────────────────────────────────────┐
 │           Simulink Dynamic Simulation Model            │
 │             (Solvers: Discrete, ODE45, etc.)           │
 └───────────────────────────┬────────────────────────────┘
                             │ u(t) Input Excitation Vector
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │         S-Function Block: sfun_opencl_parallel         │
 │          (Level-2 C-MEX Gateway Architecture)          │
 └─────────────┬────────────────────────────▲─────────────┘
   DMA Upload  │                            │ DMA Readback
   (Host->Dev) │                            │ (Dev->Host)
               ▼                            │
 ┌──────────────────────────────────────────┴─────────────┐
 │       NVIDIA GeForce GT 750M (384 Kepler Cores)        │
 │           Persistent 2 GB GDDR5 GPU Memory             │
 │         OpenCL Kernel (Symplectic Euler Step)          │
 └────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Technical Highlights

1. **Zero-Overhead Persistent Buffers (`mdlStart`):**
   - OpenCL context, command queue, program compilation, and GPU VRAM allocation occur once during simulation start (`mdlStart`).
   - During active time steps (`mdlOutputs`), there is **zero allocation overhead**—only lightning-fast asynchronous DMA stream transfers and kernel dispatches.
2. **Linux Modern Toolchain Bridge (`LD_PRELOAD`):**
   - Resolves the classic MATLAB bundled `libstdc++` version mismatch on Ubuntu 24.04 (`GLIBCXX_3.4.32`) to allow smooth dynamic linking of Mesa LLVM and Rusticl.
3. **Parametric Scaling:**
   - Supports scaling up to thousands of parallel physical nodes/channels (e.g. 512, 1024, 4096) directly from the Simulink block parameter dialog.

---

## 🛠️ Files in this Module

* [`sfun_opencl_parallel.c`](sfun_opencl_parallel.c): The Level-2 C-MEX S-Function block source code.
* [`sfun_opencl_parallel.mexa64`](sfun_opencl_parallel.mexa64): Precompiled 64-bit Linux MEX binary.
* [`compile_sfunction.m`](compile_sfunction.m): MATLAB script to recompile the S-Function with `mex -O -lOpenCL`.
* [`run_simulink_demo.m`](run_simulink_demo.m): Automated MATLAB script creating and running a complete 512-channel coupled oscillator wave propagation model (`gpu_oscillator_array_model.slx`).
* [`test_opencl_platform.c`](test_opencl_platform.c): Diagnostic MEX utility validating platform and GPU device discovery inside the MATLAB process.

---

## 💻 How to Launch MATLAB & Run the Demo

Because MATLAB bundles an older `libstdc++`, launch MATLAB with the system library preloaded:

```bash
# Launch MATLAB with Rusticl & system libstdc++ enabled:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 RUSTICL_ENABLE=nouveau matlab
```

Then in the MATLAB Command Window:

```matlab
cd /home/fbetancourt/Gemini/macbook-hardware-linux-lab/subsystems/03-display-and-gmux/simulink_opencl_bridge

% Run the automated Simulink GPU simulation demo:
run_simulink_demo
```

---

## 🧩 Adding the GPU Accelerator to Any Custom Simulink Model

1. Open your Simulink model or canvas.
2. Drag an **S-Function** block from the Simulink Library Browser (`Simulink > User-Defined Functions > S-Function`).
3. Double-click the block:
   * **S-function name:** `sfun_opencl_parallel`
   * **S-function parameters:** `512, 0.05` *(Number of channels, damping coefficient)*
4. Connect your vector input signal to the input port, and connect the output port to a Scope or To Workspace.
5. Click **Run**! The simulation states are now computed in hardware by your NVIDIA GPU.
