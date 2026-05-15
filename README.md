<div align="center">

```
 ███████╗███╗   ██╗██╗ ██████╗ ███╗   ███╗ █████╗   
██╔════╝████╗  ██║██║██╔════╝ ████╗ ████║██╔══██╗
█████╗  ██╔██╗ ██║██║██║  ███╗██╔████╔██║███████║
██╔══╝  ██║╚██╗██║██║██║   ██║██║╚██╔╝██║██╔══██║
███████╗██║ ╚████║██║╚██████╔╝██║ ╚═╝ ██║██║  ██║
╚══════╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝
```

### **Bayesian MPC Predictive Agent for Autonomous Highway Driving**
*Phase 3.2 · Temporal Transformer · GMM · MC Dropout · Hard Shield AEB · ROS2*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-009688?style=for-the-badge)](https://gymnasium.farama.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Ready-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://docs.ros.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7CA18?style=for-the-badge)](LICENSE)

<br/>

[![highway-env](https://img.shields.io/badge/Simulation-highway--env%20(Open%20Source)-FF6B35?style=for-the-badge&logo=github&logoColor=white)](https://github.com/eleurent/highway-env)
[![Captum](https://img.shields.io/badge/XAI-Captum%20%7C%20Integrated%20Gradients-7952B3?style=for-the-badge)](https://captum.ai/)
[![FileLock](https://img.shields.io/badge/Training-Thread--Safe%20%7C%20FileLock-4CAF50?style=for-the-badge)](https://pypi.org/project/filelock/)

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-enigmaa.space.z.ai-0A0A0A?style=for-the-badge&logoColor=white)](https://enigmaa.space.z.ai)

</div>

---

<div align="center">

> 🚨 **Test Case Simulation Notice**
>
> All driving scenarios used in **test cases** are simulated using **[highway-env](https://github.com/eleurent/highway-env)** —
> a free, open-source autonomous driving environment by Édouard Leurent, maintained by the
> **[Farama Foundation](https://farama.org)** under the MIT License.
> We use a **modified fork** with custom patches for intersection logic,
> coordinate encoding, and distributed training support.

</div>

---

## 📌 Table of Contents

| # | Section |
|---|---------|
| 1 | [🌐 Live Deployment](#-live-deployment) |
| 2 | [🔭 Project Overview](#-project-overview) |
| 3 | [🏗️ Open-Source Simulation Foundation](#️-open-source-simulation-foundation) |
| 4 | [🗂️ Repository Structure](#️-repository-structure) |
| 5 | [🧠 System Architecture](#-system-architecture) |
| 6 | [⚡ Test Scenarios](#-test-scenarios) |
| 7 | [🛡️ Hard Shield — AEB Safety System](#️-hard-shield--aeb-safety-system) |
| 8 | [🛰️ Server / Client Bridge](#️-server--client-bridge) |
| 9 | [🤖 ROS2 Deployment](#-ros2-deployment) |
| 10 | [🔄 Continuous Online Learning](#-continuous-online-learning) |
| 11 | [🔧 Patch System](#-patch-system) |
| 12 | [🧪 Evaluation & Benchmarks](#-evaluation--benchmarks) |
| 13 | [🔬 Explainability (XAI)](#-explainability-xai) |
| 14 | [📦 Installation](#-installation) |
| 15 | [🚀 Quick Start](#-quick-start) |
| 16 | [⚙️ Configuration Reference](#️-configuration-reference) |
| 17 | [📊 Model Files](#-model-files) |
| 18 | [🙏 Acknowledgements](#-acknowledgements) |

---

## 🌐 Live Deployment

> **ENIGMAA is live and accessible at:**

<div align="center">

### 🚀 [enigmaa.space.z.ai](https://enigmaa.space.z.ai)

</div>

The deployed instance runs the full ENIGMAA stack — Bayesian MPC agent, Hard Shield AEB, and the multi-environment evaluation suite — accessible directly from your browser without any local setup.

| Detail | Info |
|---|---|
| 🌍 URL | [https://enigmaa.space.z.ai](https://enigmaa.space.z.ai) |
| 🧠 Agent | Bayesian Trajectory Predictor (Phase 3.2) |
| 🛡️ Safety | Hard Shield AEB active |
| 🌐 Environments | `highway-v0`, `intersection-v0`, `two-way-v0` |

---

## 🔭 Project Overview

**ENIGMAA** is a research-grade **Bayesian Model Predictive Control (MPC)** agent for safe autonomous driving in complex, dynamic traffic environments. It wraps a neural trajectory predictor inside a physics-level safety guardrail, enabling confident action selection with quantified uncertainty at every timestep.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ENIGMAA PHASE 3.2                             │
│                                                                      │
│  Observation ──▶ Feature Extraction ──▶ Temporal Transformer         │
│                                               │                      │
│                                    GMM Trajectory Prediction         │
│                                    (3 modes × 30-step horizon)       │
│                                               │                      │
│                                     MPC Action Selection             │
│                                               │                      │
│                                   Hard Shield AEB Override           │
│                                               │                      │
│                                        Environment Step              │
│                                               │                      │
│                                    Online Model Fine-tuning          │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Capabilities

| Capability | Technology |
|---|---|
| Sequence modeling | Temporal Transformer Encoder (2L, nhead=3) |
| Multi-modal prediction | Gaussian Mixture Model head (3 modes) |
| Uncertainty quantification | MC Dropout — epistemic + aleatoric |
| Rule-based safety | Hard Shield AEB (physics proximity checks) |
| Explainability | Captum Integrated Gradients |
| Continuous learning | File-locked background fine-tuning |
| Real-robot bridge | ROS2 node (`/vehicle/cmd_vel`) |
| Remote control | TCP socket server/client protocol |

---

## 🏗️ Open-Source Simulation Foundation

> **All test case driving scenarios in ENIGMAA are simulated using the open-source [`highway-env`](https://github.com/eleurent/highway-env) library.**

### What is highway-env?

[`highway-env`](https://github.com/eleurent/highway-env) is a **free, open-source collection of environments for autonomous driving and tactical decision-making**, originally developed by **Édouard Leurent** and now maintained by the **[Farama Foundation](https://farama.org)** as part of the Gymnasium ecosystem.

It provides:
- Kinematic vehicle simulation with realistic lane/road geometry
- Multi-agent NPC traffic with configurable spawn rates and behaviors
- Observation spaces: kinematics, occupancy grids, grayscale frames
- Action spaces: discrete (lane change/speed) and continuous (steering/acceleration)
- Multiple pre-built environments covering highway, intersection, roundabout, parking, and racetrack scenarios

**Our modifications** on top of the base library:

```
highway-env (upstream, MIT)
        │
        ├── patch_intersection.py      → intersection-v0 AEB logic + action remapping
        ├── patch_coordinates.py       → relative ego-coordinate trajectory encoding
        ├── patch_data_architecture.py → per-scenario isolated .npz data files
        └── patch_files.py             → FileLock-safe concurrent training support
```

### Environments Used

| `highway-env` Environment | ENIGMAA Scenario |
|---|---|
| `highway-v0` | Cruising, Braking, Overtake, Lane Selection, Gap Change, Cluster, Marginal Gap |
| `intersection-v0` | Urban intersection crossing with yielding |
| `two-way-v0` | Oncoming traffic head-on avoidance |

> ⭐ If you use this project, please also star and cite the original [`highway-env`](https://github.com/eleurent/highway-env) repository.

---

## 🗂️ Repository Structure

```
env/
│
├── 📦 bayesian_model.pt                ← Root pre-trained Bayesian Transformer
├── 📊 live_calib_data.npz              ← Shared live calibration dataset
├── 📊 live_calib_data_oncoming.npz     ← Oncoming-specific calibration buffer
│
└── 📁 HighwayEnv/
    │
    ├── 📁 highway_env/                 ← Modified fork of highway-env (open-source base)
    │   ├── envs/                       ← Environment definitions (highway, intersection…)
    │   │   ├── highway_env.py
    │   │   ├── intersection_env.py
    │   │   ├── two_way_env.py
    │   │   ├── roundabout_env.py
    │   │   ├── parking_env.py
    │   │   └── common/                 ← Observation, action, abstract, graphics
    │   ├── road/                       ← Lane, road, graphics, spline, regulation
    │   └── vehicle/                    ← Kinematics, behavior, controller, uncertainty
    │
    ├── 📁 test_cases/                  ← ENIGMAA scenario-specific agents
    │   ├── test_case_mpc.py            ← Baseline MPC (highway cruising)
    │   ├── test_case_braking.py        ← Emergency braking
    │   ├── test_case_overtake.py       ← Safe overtaking
    │   ├── test_case_lane_selection.py ← Lane selection & merging
    │   ├── test_case_gap_change.py     ← Dynamic gap exploitation
    │   ├── test_case_cluster.py        ← Dense cluster navigation
    │   ├── test_case_marginal_gap.py   ← Marginal gap acceptance
    │   ├── test_case_intersection.py   ← Urban intersection crossing
    │   ├── test_case_oncoming.py       ← Head-on collision avoidance
    │   └── test_case_ros2.py           ← ROS2 deployment node
    │
    ├── 🖥️  server_env.py               ← TCP socket environment server
    ├── 🔌  client_script.py            ← TCP socket agent client
    ├── 📈  eval.py                     ← Quick 3-episode evaluation
    ├── 📋  eval_report.py              ← Multi-environment benchmark
    │
    ├── 🔧  patch_files.py              ← FileLock + shared model patching
    ├── 🔧  patch_intersection.py       ← Intersection-v0 adaptation
    ├── 🔧  patch_data_architecture.py  ← Per-scenario data isolation
    ├── 🔧  patch_coordinates.py        ← Relative coordinate encoding fix
    │
    ├── 📦  bayesian_model.pt           ← Working checkpoint (online-updated)
    ├── 📦  bayesian_model_oncoming.pt  ← Oncoming-specific weights
    ├── 📊  data_test_case_*.npz        ← Per-scenario training datasets
    ├── 🔒  training.lock               ← FileLock for concurrent safety
    │
    ├── pyproject.toml                  ← Package metadata & deps
    └── setup.py                        ← Legacy setuptools entry
```

---

## 🧠 System Architecture

### Bayesian Trajectory Predictor

```
 ╔══════════════════════════════════════════════════════════════╗
 ║                  BayesianTrajectoryPredictor                 ║
 ╠══════════════════════════════════════════════════════════════╣
 ║                                                              ║
 ║  Input:  (batch, seq_len=3, 69-dim)                          ║
 ║           └── 64-dim kinematics + 5-dim one-hot action       ║
 ║                                                              ║
 ║  ┌──────────────────────────────────────────────────────┐    ║
 ║  │       Temporal Transformer Encoder                   │    ║
 ║  │   2 layers · nhead=3 · d_ff=128 · dropout=0.2        │    ║
 ║  └─────────────────────┬────────────────────────────────┘    ║
 ║                        │  last-timestep token                ║
 ║               ┌────────▼────────┐                            ║
 ║               │  Linear + ReLU  │  (→ 128-dim hidden)        ║
 ║               │  + MC Dropout   │  (p=0.2, active at infer)  ║
 ║               └────────┬────────┘                            ║
 ║        ┌───────────────┼───────────────┐                     ║
 ║        ▼               ▼               ▼                     ║
 ║   mean_head       logvar_head     weight_head                 ║
 ║  (3×30×2 μ)      (3×30×2 σ²)    (3,) softmax                 ║
 ║        └───────────────┴───────────────┘                     ║
 ║               Gaussian Mixture Model                          ║
 ║               3 modes · 30-step horizon · (x,y) coords       ║
 ╚══════════════════════════════════════════════════════════════╝
```

**MC Dropout** remains active during inference. Each forward pass samples a different trajectory distribution, and N passes are aggregated to estimate:

- **Epistemic uncertainty** — what the model *doesn't know* (reducible with more data)
- **Aleatoric uncertainty** — inherent observation noise (irreducible)

### MPC Action Selection Loop

```
 For each timestep t:
 ┌─────────────────────────────────────────────────────────┐
 │  1. Extract 64-dim features from obs_t                   │
 │  2. Append to 3-frame temporal buffer                    │
 │                                                          │
 │  For each candidate action a ∈ {0,1,2,3,4}:             │
 │    ┌─────────────────────────────────────────────────┐  │
 │    │  Concat one-hot(a) → temporal sequence           │  │
 │    │  Run N MC forward passes through Transformer     │  │
 │    │  Compute GMM-weighted mean trajectory            │  │
 │    │  Compute epistemic std (uncertainty)             │  │
 │    │  Score = progress − lateral_dev − uncertainty    │  │
 │    └─────────────────────────────────────────────────┘  │
 │                                                          │
 │  3. Select action with highest score (best_action)       │
 │  4. Apply Hard Shield AEB override if needed  🛡️         │
 │  5. env.step(final_action) → obs_{t+1}, reward           │
 │  6. Log (feature, true_future) for online training       │
 └─────────────────────────────────────────────────────────┘
```

---

## ⚡ Test Scenarios

> All scenarios below are simulated using the **open-source [`highway-env`](https://github.com/eleurent/highway-env)** library — used exclusively for **test case simulation**.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Script</th>
<th>Environment</th>
<th>Challenge</th>
<th>AEB Mode</th>
</tr>
</thead>
<tbody>
<tr>
<td>🛣️ Baseline Cruising</td>
<td><code>test_case_mpc.py</code></td>
<td><code>highway-v0</code></td>
<td>Maintain high speed in flowing traffic</td>
<td>Lane-based</td>
</tr>
<tr>
<td>🛑 Emergency Braking</td>
<td><code>test_case_braking.py</code></td>
<td><code>highway-v0</code></td>
<td>Hard braking behind slow/stopped vehicle</td>
<td>Lane-based</td>
</tr>
<tr>
<td>⬆️ Safe Overtake</td>
<td><code>test_case_overtake.py</code></td>
<td><code>highway-v0</code></td>
<td>Overtake slower vehicle without sideswipe</td>
<td>Lane-based</td>
</tr>
<tr>
<td>🔀 Lane Selection</td>
<td><code>test_case_lane_selection.py</code></td>
<td><code>highway-v0</code></td>
<td>Choose optimal lane under variable traffic</td>
<td>Lane-based</td>
</tr>
<tr>
<td>🔓 Gap Change</td>
<td><code>test_case_gap_change.py</code></td>
<td><code>highway-v0</code></td>
<td>Exploit dynamic gaps as they open/close</td>
<td>Lane-based</td>
</tr>
<tr>
<td>🚦 Dense Cluster</td>
<td><code>test_case_cluster.py</code></td>
<td><code>highway-v0</code></td>
<td>Navigate through dense NPC clusters safely</td>
<td>Lane-based</td>
</tr>
<tr>
<td>📏 Marginal Gap</td>
<td><code>test_case_marginal_gap.py</code></td>
<td><code>highway-v0</code></td>
<td>Accept/reject tight gaps at decision threshold</td>
<td>Lane-based</td>
</tr>
<tr>
<td>🏙️ Intersection</td>
<td><code>test_case_intersection.py</code></td>
<td><code>intersection-v0</code></td>
<td>Urban crossing with yielding + arrival reward</td>
<td>Radial (8m)</td>
</tr>
<tr>
<td>🔄 Oncoming Traffic</td>
<td><code>test_case_oncoming.py</code></td>
<td><code>two-way-v0</code></td>
<td>Head-on avoidance on two-lane road</td>
<td>Lane-based</td>
</tr>
<tr>
<td>🤖 ROS2 Node</td>
<td><code>test_case_ros2.py</code></td>
<td>ROS2 Topics</td>
<td>Real-robot deployment via pub/sub</td>
<td>Both modes</td>
</tr>
</tbody>
</table>

---

## 🛡️ Hard Shield — AEB Safety System

The **Hard Shield** is a **physics-level, rule-based safety override** that executes *after* the neural network selects an action. It acts as a non-negotiable last-resort collision barrier — the neural policy is never solely responsible for collision avoidance.

### Highway / Two-Way Mode (Lane-Based Geometry)

```
  Ego Vehicle ──▶ Check surrounding vehicles:

  ┌─────────────────────────────────────────────────────────────┐
  │  crash_ahead   (|dy| < 2.0m,  0 < dx < 15.0m)  →  FRONT   │
  │  blocked_left  (-6.0m < dy ≤ -2.0m, |dx| < 10m) →  LEFT   │
  │  blocked_right  (2.0m ≤ dy < 6.0m,  |dx| < 10m) →  RIGHT  │
  └─────────────────────────────────────────────────────────────┘

  Decision Tree:
  crash_ahead + action ∈ {IDLE, FASTER}
      ├── NOT blocked_left  ──▶  action = LANE_LEFT   (evade)
      ├── NOT blocked_right ──▶  action = LANE_RIGHT  (evade)
      └── BOTH blocked      ──▶  action = SLOWER       (brake)

  action = LANE_LEFT  + blocked_left  ──▶  IDLE (or BRAKE if crash)
  action = LANE_RIGHT + blocked_right ──▶  IDLE (or BRAKE if crash)
```

### Intersection Mode (Radial Proximity)

```
  For every NPC vehicle v:
    dist = sqrt[(v.x − ego.x)² + (v.y − ego.y)²]
    dist < 8.0m  ──▶  SLAM BRAKES  🛑  (action = SLOWER)

  Action remapping (NN 5-class → intersection-v0 Discrete(3)):
    NN:  LEFT=0  IDLE=1  RIGHT=2  FASTER=3  SLOWER=4
    Env:   →1      →1      →1       →2         →0
```

---

## 🛰️ Server / Client Bridge

ENIGMAA includes a **TCP socket bridge** that fully decouples the environment renderer from the agent controller, allowing any external process to drive the simulation.

```
 ┌──────────────────────┐          TCP:5005         ┌──────────────────────┐
 │   server_env.py      │◄─────────────────────────►│  client_script.py    │
 │                      │                            │  (or any controller) │
 │  highway-env         │  ① {"status": "ready"}    │                      │
 │  pygame render loop  │─────────────────────────► │                      │
 │                      │  ② action integer (0-4)   │                      │
 │                      │◄───────────────────────── │                      │
 │                      │  ③ {"reward":r,"done":d}  │                      │
 │                      │─────────────────────────► │                      │
 └──────────────────────┘                            └──────────────────────┘
```

**Action Space:**

| Code | Action |
|:---:|---|
| `0` | Lane Left |
| `1` | Idle |
| `2` | Lane Right |
| `3` | Faster |
| `4` | Slower |

```bash
# Terminal 1 — Launch environment server (pygame window opens)
python server_env.py

# Terminal 2 — Connect your agent or test client
python client_script.py
```

The bridge is compatible with any controller: another Python process, a ROS2 node, a remote machine, or a custom RL training loop.

---

## 🤖 ROS2 Deployment

`test_case_ros2.py` provides a production-ready **`SafeTrajectoryNode`** that bridges ENIGMAA's Bayesian MPC agent directly to real ROS2 vehicle hardware.

### Topics

| Direction | Topic | Message Type | Purpose |
|---|---|---|---|
| Subscribe | `/highway_env/observations` | `Float32MultiArray` | Sensor feature vector input |
| Publish | `/vehicle/cmd_vel` | `Twist` | Velocity commands to vehicle |
| Publish | `/explainability/rationale` | `String` | Integrated Gradients per-step |

> 💡 Replace the `/highway_env/observations` source with your real LiDAR / camera feature extractor pipeline in production.

### Launch

```bash
# 1. Source your ROS2 workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

# 2. Copy model weights to working directory
cp env/HighwayEnv/bayesian_model.pt .

# 3. Run the node
ros2 run your_package test_case_ros2
```

Expected log output:
```
[INFO] [safe_trajectory_ai]: Initializing SafeTrajectory AI ROS2 Node...
[INFO] [safe_trajectory_ai]: Successfully loaded Temporal Transformer weights.
```

---

## 🔄 Continuous Online Learning

ENIGMAA continuously improves its Bayesian model during inference using a **background training thread** with file-locked data access to prevent corruption across concurrent test cases.

```
Main Thread                          Background Thread
──────────────────────────────────   ──────────────────────────────────────────
Inference step t
  │
  ├── extract features
  ├── predict trajectory
  ├── select action (+ AEB)
  ├── env.step()
  └── DataLogger.append(feat, future)

                                     Every N steps:
                                       ┌─────────────────────────────────┐
                                       │  FileLock(training.lock)         │
                                       │                                  │
                                       │  load  data_test_case_X.npz     │
                                       │  merge new + historical          │
                                       │  trim  → max 10,000 rows         │
                                       │  save  data_test_case_X.npz     │
                                       │                                  │
                                       │  load  bayesian_model.pt         │
                                       │  fine-tune 2 epochs (bs=32)      │
                                       │  save  bayesian_model.pt         │
                                       └─────────────────────────────────┘
```

The **`training.lock`** file acts as a mutex, preventing two simultaneously running test cases from writing to the same `.npz` or `.pt` file at the same time.

---

## 🔧 Patch System

Rather than duplicating code across scenarios, ENIGMAA ships **surgical patch scripts** that transform a base test case to target a specific environment or architecture configuration.

```
Base test_case_mpc.py
        │
        ├── patch_files.py              → Add FileLock + shared MODEL_PATH
        │
        ├── patch_data_architecture.py  → Per-scenario isolated .npz data files
        │                                 (data_test_case_braking.npz, etc.)
        │
        ├── patch_coordinates.py        → Fix ego position from env.unwrapped
        │                                 Encode futures as relative (dx,dy) offsets
        │
        └── patch_intersection.py       → Switch to intersection-v0
                                          Radial AEB (8m threshold)
                                          Discrete(3) action remapping
                                          Arrival reward (+10) + spawn_probability
```

### Running Patches

```bash
cd env/HighwayEnv

# Step 1 — Shared model + FileLock support
python patch_files.py

# Step 2 — Isolate per-scenario calibration data
python patch_data_architecture.py

# Step 3 — Fix ego coordinates + relative trajectory encoding
python patch_coordinates.py

# Step 4 — Only if running intersection scenario
python patch_intersection.py
```

> ⚠️ Run patches in the order listed above. Each patch assumes the previous one has already been applied.

---

## 🧪 Evaluation & Benchmarks

### Quick Evaluation (3 Episodes)

```bash
cd env/HighwayEnv
python eval.py
```

```
Eval Episode 1 Survived for: 423 steps
Eval Episode 2 Survived for: 389 steps
Eval Episode 3 Survived for: 461 steps
AVERAGE_STEPS: 424.33
```

### Full Multi-Environment Benchmark

```bash
python eval_report.py
```

Runs the pre-trained model across three distinct highway-env environments:

| Benchmark | Environment | Metric |
|---|---|---|
| Baseline MPC (Cruising) | `highway-v0` | Steps survived |
| Two-Way Oncoming (Avoidance) | `two-way-v0` | Steps survived |
| Urban Intersection (Yielding) | `intersection-v0` | Steps survived |

Both scripts use **`rgb_array` render mode** (headless) and cap episodes at 100–500 steps for repeatable benchmarking.

---

## 🔬 Explainability (XAI)

ENIGMAA uses **[Captum](https://captum.ai/) Integrated Gradients** to trace every action decision back to the raw 69-dimensional input features — fulfilling safety and audit requirements in regulated deployments.

```
ModelWrapper(BayesianTrajectoryPredictor)
        │
        └── IntegratedGradients.attribute(input_tensor)
                │
                ▼
        Attribution scores per input feature
        (69 values: 64 kinematic dims + 5 action dims)
                │
                ▼
        Published to /explainability/rationale  (ROS2)
        or logged inline in terminal output
```

The `ModelWrapper` reduces the multi-modal GMM output (3 modes × 30 steps × 2 coords) to a scalar score so that Integrated Gradients can compute gradients through the full prediction stack.

> This enables safety engineers and regulators to audit *why* the agent selected a particular action at any timestep — a key requirement for ISO 26262 / SOTIF compliance in production AV systems.

---

## 📦 Installation

### Requirements

- Python **3.9 – 3.13**
- pip
- *(Optional)* ROS2 Humble or later for `test_case_ros2.py`

### Step 1 — Clone

```bash
git clone https://github.com/your-username/enigmaa-highway
cd enigmaa-highway
```

### Step 2 — Install the modified highway-env fork

```bash
cd env/HighwayEnv
pip install -e .
```

### Step 3 — Install ENIGMAA dependencies

```bash
pip install torch captum filelock
```

### Full dependency list (`pyproject.toml`)

```toml
dependencies = [
    "gymnasium >= 1.0.0a2",
    "farama-notifications >= 0.0.1",
    "numpy >= 1.23.0",
    "pygame >= 2.0.2",
    "matplotlib",
    "pandas",
    "scipy",
    "torch",
    "captum",
    "filelock",
]
```

---

## 🚀 Quick Start

```bash
cd env/HighwayEnv

# ── Option A: Run a scenario directly ──────────────────────────────────
python test_cases/test_case_mpc.py          # Highway baseline
python test_cases/test_case_braking.py      # Emergency braking
python test_cases/test_case_intersection.py # Urban intersection
python test_cases/test_case_oncoming.py     # Oncoming avoidance

# ── Option B: Server/Client split (decoupled renderer) ─────────────────
python server_env.py &       # Starts the pygame render server
python client_script.py      # Agent connects and sends actions

# ── Option C: Headless evaluation ──────────────────────────────────────
python eval.py               # Quick 3-episode benchmark
python eval_report.py        # Full 3-environment report

# ── Option D: Apply patches then run ───────────────────────────────────
python patch_files.py
python patch_data_architecture.py
python patch_coordinates.py
python test_cases/test_case_mpc.py
```

---

## ⚙️ Configuration Reference

All environments are configured by passing a `config` dict to `gym.make()`:

```python
env = gym.make("highway-v0", render_mode="human", config={
    "lanes_count": 2,
    "duration": 1_000_000,
    "simulation_frequency": 15,
    "collision_reward": -50,
    "high_speed_reward": 1,
    "right_lane_reward": 0.1,
    "reward_speed_range": [20, 30],
})
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lanes_count` | `int` | `4` | Number of highway lanes |
| `duration` | `int` | `40` | Episode length in steps |
| `simulation_frequency` | `int` | `15` | Simulation steps per second |
| `collision_reward` | `float` | `-1` | Penalty applied on collision |
| `high_speed_reward` | `float` | `0.4` | Reward for high-speed travel |
| `right_lane_reward` | `float` | `0.1` | Reward for right-lane preference |
| `reward_speed_range` | `list` | `[20, 30]` | Speed range for full reward |
| `spawn_probability` | `float` | `0.6` | NPC vehicle spawn rate (intersection) |
| `arrived_reward` | `float` | `1.0` | Reward for completing intersection cross |
| `manual_control` | `bool` | `False` | Enable keyboard control |

---

## 📊 Model Files

| File | Location | Size | Description |
|---|---|---|---|
| `bayesian_model.pt` | `env/` | ~167 KB | Root pre-trained Bayesian Transformer |
| `bayesian_model.pt` | `env/HighwayEnv/` | ~534 KB | Working checkpoint (updated by online learning) |
| `bayesian_model_oncoming.pt` | `env/HighwayEnv/` | ~534 KB | Oncoming-scenario specialized weights |
| `live_calib_data.npz` | `env/` | ~4.5 MB | Shared live calibration buffer |
| `live_calib_data.npz` | `env/HighwayEnv/` | ~10 MB | Expanded working calibration buffer |
| `live_calib_data_oncoming.npz` | `env/HighwayEnv/` | ~1.4 MB | Oncoming-specific calibration buffer |
| `data_test_case_*.npz` | `env/HighwayEnv/` | ~2 MB each | Per-scenario isolated training data |

---

## 🙏 Acknowledgements

<table>
<tr>
<td align="center" width="50%">
<h4><a href="https://github.com/eleurent/highway-env">highway-env</a></h4>
<em>The open-source driving simulation library powering all ENIGMAA test scenarios</em><br/><br/>
By <strong>Édouard Leurent</strong> · Maintained by <strong>Farama Foundation</strong><br/>
Licensed under <strong>MIT</strong>
</td>
<td align="center" width="50%">
<h4><a href="https://farama.org">Farama Foundation</a></h4>
<em>Gymnasium ecosystem maintenance and stewardship of open-source RL environments</em><br/><br/>
<a href="https://gymnasium.farama.org">gymnasium.farama.org</a>
</td>
</tr>
<tr>
<td align="center" width="50%">
<h4><a href="https://captum.ai">Captum</a></h4>
<em>Model interpretability and explainability — Integrated Gradients XAI</em><br/><br/>
By <strong>Meta Research</strong> · Open Source
</td>
<td align="center" width="50%">
<h4><a href="https://pytorch.org">PyTorch</a></h4>
<em>Neural network backbone for the Temporal Transformer GMM architecture</em><br/><br/>
By <strong>Meta AI</strong> · Open Source
</td>
</tr>
</table>

---

<div align="center">

**ENIGMAA Phase 3.2** · MIT License · Built on open-source foundations

🌐 **Deployed at [enigmaa.space.z.ai](https://enigmaa.space.z.ai)**

*Test case simulation powered by [`highway-env`](https://github.com/eleurent/highway-env)*

</div>
