# Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation

Official code for the RL4IL @ ICRA 2026 paper, which compares dual and unified critic
architectures for whole-body loco-manipulation on the Unitree G1 in NVIDIA Isaac Lab. The two
runs differ in more than the critic; see [Experimental caveats](#experimental-caveats) before
reading the results as an effect of critic architecture.

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

Run of record: `ulc_g1_stage6_simplified_2026-02-04_23-41-18` — reached curriculum level 12/13 (the last level of its 13-level curriculum),
best checkpoint at iteration 19,730.

### Train the unified-critic policy (S6u)

A single 109-dim critic over the concatenated observation. Note this script is not otherwise
identical to the dual-critic one — it also has its own 40-level curriculum, its own reward table,
policy-driven fingers from level 10 on, and a different PPO update (one summed advantage, one
likelihood ratio for both actors); see [Experimental caveats](#experimental-caveats).

**Use the as-run file.** `train_ulc_stage_6_unified_asrun_c545e8a.py` is the exact script that
produced the S6u run, restored byte-for-byte from commit `c545e8a` (blob `73b05dc`). The file
`train_ulc_stage_6_unified.py` at the tip of this branch is **not** that script: it was rewritten
afterwards and now builds `DualActorCritic` with separate `loco_critic` and `arm_critic`, so it
does not reproduce S6u. Running it will train a dual-critic policy.

```powershell
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/train/23dof/train_ulc_stage_6_unified_asrun_c545e8a.py `
  --stage3_checkpoint <NOT RECORDED - see below> `
  --num_envs 2048 `
  --max_iterations 20000 `
  --experiment_name ulc_g1_stage6_complete `
  --headless
```

Run of record: `ulc_g1_stage6_complete_2026-01-31_20-49-39` — stopped at iteration 20,000 at
curriculum level 10 of 40, with 3.3M training reaches and best reward 36.2.

**None of this run's launch flags were saved to disk.** The values above are reconstructed:

- `--num_envs 2048` — *inferred, not recorded.* A run can log at most `num_envs × 24` reaches
  per iteration; across all 20,000 iterations the observed maximum is 49,046, which never
  exceeds the 2048 ceiling of 49,152 (99.78% of it) and exceeds the 1024 ceiling 69 times.
  See [Experimental caveats](#experimental-caveats).
- `--max_iterations 20000` — *read off the artifacts, not the flag.* The saved checkpoints run
  to `model_20000.pt` and `model_final.pt` carries `iteration = 20000`, so the run ended there;
  the flag value itself was never recorded (the script's own default is 25000).
- `--stage3_checkpoint` — **unknown.** The dual-critic run started from
  `logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt`, but there is no record of what
  this run used, and nothing in the checkpoint identifies its initialisation. Do not assume the
  two runs shared an upstream checkpoint.

### Train the anti-gaming variant (S7)

Frozen locomotion branch, freshly initialised arm policy, 8-level curriculum, 55-dim arm obs.

```powershell
.\isaaclab.bat -p REPO/g1/isaac_g1_ulc/train/23dof/train_ulc_stage_7.py `
  --stage6_checkpoint logs/ulc/ulc_g1_stage6_simplified_2026-02-04_23-41-18/model_best.pt `
  --num_envs 4096 `
  --headless
```

Run of record: `ulc_g1_stage7_antigaming_2026-02-06_17-41-47` — level 7/8 (the last level of its 8-level curriculum), best checkpoint at
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

The trained policies for the paper are published on Hugging Face:
**https://huggingface.co/mturan33/g1-dual-critic-locomanip**

```bash
hf download mturan33/g1-dual-critic-locomanip --local-dir checkpoints_paper
```

The commands above expect local run directories under `C:\IsaacLab\logs\ulc\`; point the
`--s6_checkpoint` / `--s6u_checkpoint` / `--s7_checkpoint` flags at the downloaded files instead
(`dual_critic_s6s.pt`, `unified_critic_s6u.pt`, `dual_critic_antigaming_s7.pt`).

## Experimental caveats

The headline comparison rests on a single pair of training runs, and those runs differ in more
than the critic. This section states what was and was not held constant.

### Held constant between S6u and S6s

Robot and simulator; the 17 driven joints (12 leg + 5 arm); observation spaces (57-dim
locomotion, 52-dim arm); network sizes ([512, 256, 128] locomotion, [256, 256, 128] arm); PPO
settings (lr 3e-4 with cosine annealing, γ = 0.99, λ = 0.95, clip 0.2, 24-step rollouts);
20,000 training iterations; and the entire evaluation path — one benchmark script, one shared
locomotion branch, fingers pinned open for every policy, seed 42.

Two of those "held constant" items cut against S6u specifically. The shared locomotion branch is
the dual run's: S6s and S7 carry bit-identical locomotion weights (15 of 15 tensors equal), while
S6u's own locomotion weights differ from them (largest absolute difference 3.38), so the harness
replaces S6u's locomotion branch at load time unless `--s6u_own_loco` is passed. And pinning the
fingers open matches how S6s and S7 trained, but not how S6u trained. The PPO *settings* are
shared; the PPO *update rule* is not — see the update-rule item below.

### Parallel environments

| Run | `num_envs` | Basis |
|---|---|---|
| S6s (dual critic) | 2048 | recorded in the launch notes written before the run |
| S7 (dual + anti-gaming) | 4096 | recorded in the launch command |
| S6u (unified critic) | 2048 — **inferred** | not recorded anywhere; see below |

S6u's launch flags were never written to disk. Wall-clock timing cannot settle the question: the
known-2048 and known-4096 runs take 3.39 and 3.22 s/iteration, so iteration cost on this machine
does not scale with environment count. The reach counter can. A run can log at most
`num_envs × 24` reaches in a single iteration, so the counter's peak bounds the environment
count from below. Scanning all 20,000 iterations of the S6u run:

| Ceiling | `num_envs × 24` | Iterations exceeding it |
|---|---|---|
| 1024 | 24,576 | **69** |
| 2048 | 49,152 | **0** |
| 4096 | 98,304 | 0 |

The largest single-iteration increment anywhere in the run is **49,046**. That puts a hard floor
of `num_envs` ≥ 2044 and rules 1024 out outright, while sitting at 99.78% of the 2048 ceiling —
106 reaches of headroom — without ever crossing it. For 4096 to be the answer, the counter would
have to hover at 49.9% of its ceiling for 20,000 iterations and never once cross half. 2048 is
the only reading consistent with the data, but it remains an inference, not a record.

Version 1 of the paper stated that all experiments used 4096 environments; that held only for
S7. Version 2 (30 July 2026) reports the per-run counts above.

### Curriculum — the two runs did not finish on the same task

This is the largest caveat, and it is not a matter of one run being a shorter version of the
other. Read from the curriculum definitions, the task each policy was training on when its
checkpoint was saved:

| At its final level | S6u — level 10 of 40 | S6s — level 12 of 13 |
|---|---|---|
| Base motion | **standing still** (`vx`, `vy`, `vyaw` all 0) | **walking**, `vx` 0–0.6 m/s, `vy` ±0.13, `vyaw` ±0.22 |
| Target distance | 0.18–0.28 m | 0.18–0.40 m |
| Position tolerance | 0.05 m | 0.04 m |
| Orientation target | fixed palm-down | **arbitrary**, sampled in a widening cone (80° at level 12) |
| Orientation tolerance | 0.8 rad (≈46°) | 1.0 rad (≈57°) |
| Gripper | **active** — the policy drives the seven finger joints and earns a gripper reward | not trained; fingers pinned open |

The 40-level scheme is not a finer-grained version of the 13-level one. Three structural
differences:

1. **Different capability axes.** S6u ramps reaching (0–9) → **orientation and gripper together**
   (10–19) → **commanded torso height** (20–29) → **payload** (30–39). S6s trains none of gripper,
   height command or payload. S6u did enter its gripper phase: `Curriculum/level` first reaches 10
   at iteration 1,162 and stays there, 18,838 of 20,000 iterations.
2. **Different shape.** S6u resets base motion at each phase boundary and re-ramps: its level 9
   already commands `vx` up to 0.60 m/s, then level 10 drops back to `vx` = 0 while the position
   tolerance stays at 0.05 m and orientation and gripper switch on. S6s's ladder is monotonic — commanded velocity only rises
   (0 → 0.2 → 0.3 → 0.35 → 0.4 → 0.45 → 0.5 → 0.55 → 0.6).
3. **Different orientation goal.** S6u only ever trains a fixed palm-down target; the script has
   no variable-orientation mechanism at all. S6s's last four levels train arbitrary end-effector
   orientation.

**Mapping level 10/40 onto the 13-level ladder.** There is no clean equivalent, because S6u's
level 10 combines a locomotion demand from the bottom of the S6s ladder with an orientation term
from its middle. On base motion — the dominant difficulty axis — standing puts it at S6s levels
0–4. On orientation it sits near S6s levels 7–8 (fixed palm-down), though at a tighter tolerance
than either. It is nowhere near S6s level 12.

**So: S6u was evaluated as a policy that had progressed less far, not one that had reached a
comparable task difficulty.** Its curriculum position is not a like-for-like measure of skill
against S6s's, and the gap in final task difficulty is large.

Two things temper how far that undercuts the result, and neither rescues the causal claim:

- The benchmark does not test S6u out of distribution. Its targets average ≈0.21 m at a 0.06 m
  tolerance, which falls inside S6u's own level-10 training range (0.18–0.28 m at 0.05 m) and is
  slightly more forgiving. S6u underperformed on approximately its own final training task.
- Both runs had the same budget — 20,000 iterations at 2048 environments. That the dual-critic
  arm finished a 13-level curriculum in that budget while the unified arm reached level 10 of 40
  is itself an observation about learning speed. But it is a *different* claim from the one v1 of
  the paper made,
  and it is still confounded: the two ladders differ in length, in graduation gates (S6u demands
  10,000 reaches at level 9 against S6s's 2,000–6,000, and the two runs do not even count a reach
  the same way — the unified script has no notion of a *validated* reach), and in what they ask for.

Version 1 of the paper reported S6u at "Level 10/12"; version 2 corrects this to 10/40.

### Not held constant, besides the curriculum

- **Arm action dimensionality and finger control.** S6u's arm actor emits 12 values (5 arm + 7
  finger) against S6s's 5, visible in the released weights as `arm_actor.log_std` with shape
  `(12,)` versus `(5,)`. Finger control switches on at curriculum **level 10**, not level 20, so
  those seven outputs drove the robot's fingers for 18,838 of the 20,000 iterations
  (`if lv["use_gripper"]:` in `_pre_physics_step` of the as-run script). They are discarded at
  evaluation, where the benchmark pins the fingers open for every policy (`arm_out[:, :5]` and
  `target_pos[:, self.finger_idx] = self.finger_lower`). S6u is therefore evaluated in a
  configuration it did not train in, and the confound is in the task performed as well as in
  exploration and policy entropy.
- **Reward table.** The two scripts do not share a reward table. The as-run unified script keeps
  one flat `REWARD_WEIGHTS` dict; the simplified one keeps separate `LOCO_REWARD_WEIGHTS` and
  `ARM_REWARD_WEIGHTS`. Forward velocity tracking is 2.5 in the unified script against 5.0 in the
  simplified one, and the term lists differ: the simplified configuration adds leg posture,
  standing still and foot stability; the unified one adds gripper (5.0), height tracking (3.0)
  and load stability (2.0). On the arm side only the distance weight (4.0) is common to both.
  Unlike the curriculum position, this was active for the entire length of both runs.
- **Update rule.** The difference is not only the critic. The unified script normalizes one
  summed advantage, applies a single likelihood ratio to both actors and clips gradients over the
  whole network with one optimizer. The simplified script computes a separate advantage per
  reward stream, normalizes each one, and updates each actor with its own ratio, its own gradient
  clipping and its own optimizer. Critic architecture and credit assignment move together between
  these two runs, and no measurement here separates them.
- The unified script's gripper reward term (`gripper_grasp`, 5.0) fired: `Env/R/gripper` is
  non-zero for 18,837 of the 20,000 logged iterations, starting at iteration 1,163.

### Seeding

Neither training run set a seed. At the time of the S6u and S6s runs, `torch.manual_seed`,
`np.random.seed`, an Isaac Lab `cfg.seed` and a `--seed` argument were all absent from the
training scripts, and Isaac Lab leaves the seed unset by default. (Scripts in `train/23dof/`
have taken a `--seed` argument since July 2026, defaulting to 42 and recorded in
`run_config.json`; that came after these runs and changes nothing about them.) The scripts do
draw from the RNG, so the runs are genuinely non-deterministic: every number reported here comes
from one run, with no variance estimate. The
benchmark does seed (default 42, re-applied before each policy), so the policies are compared on
matched target sequences even though the training runs behind them are unseeded.

### What survives this

The evaluation itself is apples-to-apples — one harness, one locomotion branch for all three
policies, matched target sequences — so the measured gap between the two arm policies (3.5x on
steps-to-target, 2x on throughput) is a real property of these two checkpoints as the harness
runs them, with S6u's arm driven by the dual run's locomotion branch and with its fingers pinned.

What the experiment cannot carry is the causal claim. Once the curricula are laid side by side,
the simpler explanation for the gap is training progress: the two policies were not merely
trained differently, they finished on tasks of very different difficulty, and the slower one had
last been trained to stand still. Critic architecture may well be why one arm climbed further on
the same budget — that is a reasonable hypothesis and the reason the ablation is worth running —
but the measured 3.5x cannot be attributed to it, because curriculum ladder, graduation gates,
locomotion reward weight and arm action dimensionality all vary alongside the critic, with a
single seed per arm.

Settling it needs a single-variable ablation: one curriculum, one action space, one reward set,
only the critic swapped, several seeds. That run has not been done, and this repository should
be read as its starting point rather than its conclusion.

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
