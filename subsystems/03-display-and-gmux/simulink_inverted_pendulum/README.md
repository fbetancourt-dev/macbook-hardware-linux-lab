# Massive Parallel Inverted Pendulum Ensemble Simulator (GPU OpenCL / Simulink)

High-performance real-time simulation of **1,024 independent nonlinear Cart-Pole dynamical systems** computed in parallel on the **NVIDIA GeForce GT 750M (384 Kepler CUDA Cores / 2 GB GDDR5)** via **OpenCL 3.0 (Mesa Rusticl)** and integrated into **MATLAB Simulink R2025b** through custom Level-2 C-MEX S-Functions.

---

## 🎯 Architecture Overview

```
 ┌─────────────────────────────────────────────────────────────┐
 │                MATLAB Simulink R2025b                      │
 │     [live_pendulum_realtime_gui.m] / Simulink Engine       │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Level-2 C-MEX S-Function
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           sfun_pendulum_ensemble_opencl.mexa64              │
 │   - Host-device memory management                           │
 │   - Non-blocking asynchronous kernel dispatch               │
 └──────────────────────────────┬──────────────────────────────┘
                                │ OpenCL 3.0 API (clEnqueueNDRangeKernel)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                 Mesa Rusticl OpenCL Driver                  │
 │          (RUSTICL_ENABLE=nouveau / Gallium3D)               │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Hardware Execution (PCIe x16)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │        NVIDIA GeForce GT 750M Mac Edition (GK107)           │
 │        - 384 Kepler Stream Cores @ 926 MHz                  │
 │        - 2,048 MB GDDR5 VRAM                                │
 │        - 1,024 concurrent work-items (local workgroup: 64)  │
 └─────────────────────────────────────────────────────────────┘
```

> **Note on Execution Model:** The 1,024 pendulum dynamical systems are scheduled as 1,024 independent OpenCL work-items partitioned into workgroups of 64 threads. The GPU hardware orchestrates execution across its 2 SMX multiprocessors (384 physical FP32 ALUs) through hardware warp schedulers (32 threads/warp), yielding massive throughput via SIMD lockstep execution.

---

## 📐 Mathematical Formulation

### 1. Coupled Nonlinear Cart-Pole Dynamics

Each pendulum system in the ensemble is governed by the continuous-time nonlinear equations of motion:

$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \left( \frac{u + m l \dot{\theta}^2 \sin\theta - b_c \dot{x}}{M + m} \right) - \frac{b_p \dot{\theta}}{m l}}{l \left( \frac{4}{3} - \frac{m \cos^2\theta}{M + m} \right)}$$

$$\ddot{x} = \frac{u + m l (\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta) - b_c \dot{x}}{M + m}$$

Where:
* $M = 1.0\,\text{kg}$ (Cart mass)
* $m = 0.15\,\text{kg}$ (Pendulum pole mass)
* $l = 0.5\,\text{m}$ (Half-length of pole)
* $g = 9.81\,\text{m/s}^2$ (Gravitational acceleration)
* $b_c = 0.1\,\text{N}\cdot\text{s/m}$ (Cart friction coefficient)
* $b_p = 0.002\,\text{N}\cdot\text{m}\cdot\text{s/rad}$ (Pivot damping)
* $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T$ (State vector)

Integration is performed on-device using a fixed-step 4th-order Runge-Kutta (RK4) integrator ($\Delta t = 2.0\,\text{ms}$, 5 substeps per solver step).

### 2. Dual-Mode Control Strategy

Each thread dynamically executes one of two control regimes:

1. **Åström-Furuta Energy Swing-Up ($|\theta| > 25^\circ$):**
   Pumps mechanical energy into the pendulum until it reaches the homoclinic orbit corresponding to the upright equilibrium:
   $$E = \frac{1}{2} J_p \dot{\theta}^2 + m g l (1 - \cos\theta)$$
   $$E_0 = 2 m g l$$
   $$u_{\text{swing}} = \text{clip}\left(k_E (E - E_0) \operatorname{sign}(\dot{\theta} \cos\theta), -u_{\max}, u_{\max}\right)$$

2. **LQR Upright Stabilization & Trajectory Tracking ($|\theta| \le 25^\circ$):**
   Linear Quadratic Regulator with setpoint tracking:
   $$u_{\text{LQR}} = -K (\mathbf{x} - \mathbf{x}_{\text{ref}})$$
   $$\mathbf{x}_{\text{ref}} = [x_{\text{target}}, 0, 0, 0]^T$$
   Where $K = [-12.5, -16.8, 142.6, 28.4]$ provides rapid disturbance rejection and precise positioning of the cart.

---

## 🎮 Interactive Soft Real-Time Visualizer (`live_pendulum_realtime_gui.m`)

The GUI provides an interactive visual control bench rendering at ~50–60 FPS while the GPU solves the 1,024 parallel states:
* **Interactive Cart Steering:** 
  * Keyboard: `Left Arrow` / `A` to move Left, `Right Arrow` / `D` to move Right.
  * GUI Buttons: On-screen `◀◀ Izquierda` and `Derecha ▶▶` buttons.
* **Wind Perturbation Injection:** `🌪️ Perturbación Viento` applies sudden impulsive lateral forces to challenge the LQR controller.
* **Live Telemetry:** Tracks cart position $x(t)$, pole angle $\theta(t)$, control force $u(t)$, and system mechanical energy $E(t)$.
* **Visual Kinematic Trace:** Multi-point trail tracing the trajectory of the pendulum tip.

---

## 🚀 Compilation & Running

### 1. Compile C-MEX S-Functions in MATLAB
```matlab
cd /home/fbetancourt/Gemini/macbook-hardware-linux-lab/subsystems/03-display-and-gmux/simulink_inverted_pendulum
compile_pendulum_sfunction
```

### 2. Run Interactive Real-Time Visualizer
```matlab
live_pendulum_realtime_gui
```

### 3. Run Benchmark (1,024 Parallel Systems)
```matlab
run_pendulum_benchmark
```
