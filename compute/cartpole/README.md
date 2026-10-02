# Inverted Pendulum (Cart-Pole) Control Lab: GPU OpenCL Ensemble & Interactive GUI

High-performance nonlinear Cart-Pole control engineering featuring:
1. **Massive Parallel Ensemble Simulation (1,024 systems):** Accelerated on the **NVIDIA GeForce GT 750M (384 Kepler Cores)** via **OpenCL 3.0 (Mesa Rusticl)** and custom Level-2 C-MEX S-Functions in **MATLAB Simulink R2025b**.
2. **Interactive Real-Time Visualizer GUI:** Direct keyboard steering (`Left`/`Right` arrows or `A`/`D`), click-to-position on track, wind perturbation injection, and phase portraits.

---

## 🎯 Dual-Track Architecture

```
                                  compute/cartpole/
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Track A: Massive GPU Ensemble]                [Track B: Interactive GUI]
  1,024 Parallel Dynamical Systems                Single System Interactive Bench
  • Kernel: sfun_pendulum_ensemble_opencl.c       • Script: live_pendulum_realtime_gui.m
  • Simulink: pendulum_gpu_ensemble_model.slx     • Refresh: ~50-60 FPS Soft Real-Time
  • Target: Parameter sweep & Monte Carlo         • Control: Real-time keyboard steering
  • Hardware: 384 Kepler Cores via Rusticl        • Features: Wind injection & phase plots
```

---

## 📐 Mathematical Formulation

### 1. Coupled Nonlinear Cart-Pole Dynamics

Each pendulum system is governed by continuous-time nonlinear equations of motion:

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

Integration is executed using a 4th-order Runge-Kutta (RK4) scheme ($\Delta t = 2.0\,\text{ms}$, 5–8 substeps per frame).

### 2. Dual-Mode Hybrid Control Strategy

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

## 🚀 Execution & Quick Start

### 1. Compile C-MEX S-Functions
Prerequisites: OpenCL headers installed and Rusticl enabled (see [`docs/opencl-rusticl-setup.md`](../../docs/opencl-rusticl-setup.md)).
```matlab
cd /home/fbetancourt/Gemini/macbook-hardware-linux-lab/compute/cartpole
compile_pendulum_sfunction
```

### 2. Launch Interactive Real-Time Visualizer
```matlab
live_pendulum_realtime_gui
```
* **Steering:** Use `Left Arrow` / `A` or `Right Arrow` / `D` to steer the cart along the track.
* **Disturbance:** Press `Space` to kick the cart right (+18N) or `Z` to kick left (-18N).
* **Click-to-Position:** Click anywhere on the track in the GUI to send the cart to that setpoint.

### 3. Run Benchmark (1,024 Parallel Systems)
```matlab
run_pendulum_benchmark
```
Executes 1,024 independent state vectors simultaneously across 384 GPU cores, comparing simulation time against the single-threaded CPU reference.
