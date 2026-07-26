# Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation

Official code for the RL4IL @ ICRA 2026 paper — a controlled comparison of dual and unified
critic architectures for whole-body loco-manipulation on the Unitree G1 in NVIDIA Isaac Lab.

[![arXiv](https://img.shields.io/badge/arXiv-2606.11891-b31b1b.svg)](https://arxiv.org/abs/2606.11891)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

The policy controls **17 joints (12 leg + 5 right arm)** of the G1's 23 active DoF; wrist and
hand joints are held fixed.

## Results

Standardized benchmark — 3,000 steps, 1 environment, deterministic actions, standing mode.

| Metric | Unified critic (S6u) | Dual critic (S6s) |
|---|---|---|
| Steps to target | 22.6 | **6.5** (3.5x faster) |
| Throughput (validated reaches / 1,000 steps) | 7.0 | **14.3** (2x) |
| Validated reach rate | 53.8% | **65.2%** |

Adding anti-gaming reward shaping on top of the dual critic (S7) yields **60.9%** — no gain
over the architectural change alone.

## Installation

Requires **Isaac Sim 5.1.0** and **Isaac Lab 2.3.1**, CUDA 12.8.

```powershell
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all]==5.1.0"

git clone -b v2.3.1 https://github.com/isaac-sim/IsaacLab.git C:\IsaacLab
cd C:\IsaacLab
.\isaaclab.bat --install

git clone https://github.com/mturan33/isaac-g1-ulc.git `
  source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc
```

Verified environment: Python 3.11.15, PyTorch 2.7.0+cu128, `isaaclab` 0.48.0, `isaacsim` 5.1.0.0.

The G1 robot USD is streamed from the Omniverse asset server at runtime — no manual asset
download is required for the experiments in the paper.

**GPU:** 12 GB VRAM is sufficient. All reported results were produced on a single RTX 5070 Ti
Laptop GPU (12 GB) at ~17K simulation steps/s.

## Reproduce

All commands run from `C:\IsaacLab` with `env_isaaclab` active. `REPO` below stands for
`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc`.

### Train the dual-critic policy (S6s)

13-level curriculum, 4 phases, variable end-effector orientation up to an 80° cone.

```powershell
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/train/23dof/train_ulc_stage_6_simplified.py `
  --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt `
  --num_envs 2048 `
  --max_iterations 20000 `
  --headless
```

Run of record: `ulc_g1_stage6_simplified_2026-02-04_23-41-18` — reached curriculum level 12/12,
best checkpoint at iteration 19,730.

### Train the unified-critic policy (S6u)

Identical setup except for a single 109-dim critic over the concatenated observation.

```powershell
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/train/23dof/train_ulc_stage_6_unified.py `
  --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt `
  --num_envs <TBD - verify> `
  --max_iterations <TBD - verify> `
  --experiment_name ulc_g1_stage6_complete `
  --headless
```

Run of record: `ulc_g1_stage6_complete_2026-01-31_20-49-39` — stopped at iteration 20,000 at
curriculum level 10/12, with 3.3M training reaches and best reward 36.2. The exact
`--num_envs` / `--max_iterations` for this run are not recorded in any config on disk.

### Train the anti-gaming variant (S7)

Frozen locomotion branch, freshly initialised arm policy, 8-level curriculum, 55-dim arm obs.

```powershell
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/train/23dof/train_ulc_stage_7.py `
  --stage6_checkpoint logs/ulc/ulc_g1_stage6_simplified_2026-02-04_23-41-18/model_best.pt `
  --num_envs 4096 `
  --headless
```

Run of record: `ulc_g1_stage7_antigaming_2026-02-06_17-41-47` — level 7/7, best checkpoint at
iteration 14,878.

### Run the three-way evaluation

```powershell
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/test/benchmark_s6_vs_s7.py `
  --s6_checkpoint  logs/ulc/ulc_g1_stage6_simplified_2026-02-04_23-41-18/model_best.pt `
  --s6u_checkpoint logs/ulc/ulc_g1_stage6_complete_2026-01-31_20-49-39/model_final.pt `
  --s7_checkpoint  logs/ulc/ulc_g1_stage7_antigaming_2026-02-06_17-41-47/model_best.pt `
  --steps 3000 --num_envs 1 --mode both --seed 42
```

Evaluation defaults match the paper: `--pos_threshold 0.06`, `--min_displacement 0.10`,
`--max_target_steps 150`, `--min_target_dist 0.12`. Results are written as `summary.json` and
`per_target.csv`, plus plots, in the output directory.

By default the benchmark loads one shared locomotion branch into all three policies so that the
arm branch is the only difference; pass `--s6u_own_loco` to evaluate S6u with its own
locomotion weights instead.

> ### Known pitfall — use the correct S6u checkpoint
>
> There are two unified-critic run directories from the same day:
>
> | Directory | State |
> |---|---|
> | `ulc_g1_stage6_complete_2026-01-31_18-54-44` | **Broken.** Aborted after 22 iterations at curriculum level 0. Its `model_best.pt` is an untrained snapshot and benchmarks at 0% reach rate / 100% timeout. |
> | `ulc_g1_stage6_complete_2026-01-31_20-49-39` | **Correct.** The run behind the paper's 53.8%. Use `model_final.pt` (iteration 20,000, level 10). |
>
> Benchmarking the `18-54-44` run instead of `20-49-39` reproduces 0%, not 53.8%.

### Checkpoints

The trained policies for the paper are **not yet in this repository**. They will be published on
Hugging Face: _link to be added_. Until then the commands above expect local run directories
under `C:\IsaacLab\logs\ulc\`.

Note: training scripts do not set a random seed, so runs are single-seed and not bit-reproducible.
This is stated as a limitation in the paper.

## Repository structure

```
isaac_g1_ulc/
├── g1/               Main package. Paper code lives in train/23dof (S6u/S6s/S7),
│                     test/benchmark_s6_vs_s7.py, play-23dof/, plus envs/,
│                     rewards/, config/, curriculum/, utils/
├── checkpoints/      Pre-trained weights for the 29-DoF pipeline (see below)
├── agents/           skrl PPO configuration
├── tasks/            Task configs and observation definitions
├── external/         Unitree hardware bridge (DDS, action provider, image server)
├── vlm_integration/  Florence-2 interface
├── go2/              Unitree Go2 quadruped environment
└── old/              Archived scripts
```

## Citation

```bibtex
@article{yardimci2026critic,
  title   = {Critic Architecture Matters: Dual vs. Unified Critics for
             Humanoid Loco-Manipulation},
  author  = {Yard{\i}mc{\i}, Mehmet Turan},
  journal = {arXiv preprint arXiv:2606.11891},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.11891},
  url     = {https://github.com/mturan33/isaac-g1-ulc},
  note    = {ICRA 2026 Workshop on Reinforcement Learning for
             Imitation Learning (RL4IL)}
}
```

---

## Other work in this repository (not part of the paper)

The repository also hosts a separate, ongoing 29-DoF line of work on the same robot. It is not
evaluated in the paper and uses different observation spaces, action spaces and checkpoints.

**29-DoF + DEX3 hierarchical pipeline** — Unitree G1 with DEX3 hands (43 joints: 12 leg, 3 waist,
14 arm, 14 finger), Triple Actor-Critic (locomotion / arm / grasp), sequential stages with frozen
predecessors.

| Stage | Task | Obs → Act | Status |
|---|---|---|---|
| 1 | Omnidirectional locomotion | 66 → 15 | Complete |
| 2 | Arm position reaching | 39 → 7 | Complete |
| 2L | Perturbation-robust locomotion | 66 → 15 | Complete |
| 3G | DEX3 finger grasping | 45 → 7 | In progress |
| VLM | Florence-2 task planning | — | Planned |

Shipped weights for this pipeline are in `checkpoints/`: `loco_stage1.pt`, `arm_stage2.pt`,
`loco_stage2.pt`. Training and play scripts live in `g1/isaac_g1_ulc/train/29dof/` and
`g1/isaac_g1_ulc/play/`.

```powershell
# Example: Stage 1 locomotion playback
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/play/play_unified_stage_1.py `
  --checkpoint REPO/checkpoints/loco_stage1.pt --num_envs 1 --mode mixed
```

## Author

**Mehmet Turan Yardımcı** — Çukurova University, Computer Engineering
[ORCID 0009-0004-8416-8368](https://orcid.org/0009-0004-8416-8368) ·
[@mturan33](https://github.com/mturan33)

## References

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — NVIDIA
- [ULC: A Unified and Fine-Grained Controller for Humanoid Loco-Manipulation](https://arxiv.org/abs/2507.06905) — Sun et al.
- [Unitree G1](https://www.unitree.com/g1/) — Unitree Robotics
