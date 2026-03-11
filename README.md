# G1 Hierarchical VLM-ULC System

**Unified Loco-manipulation Control for the Unitree G1 Humanoid Robot with Vision-Language Model Integration**

> **Research in progress -- paper in preparation.**
> If you use any part of this codebase, architecture, or methodology, citation is required.
> Contact the author for collaboration inquiries.

---

## Overview

A hierarchical reinforcement learning system for whole-body loco-manipulation on the Unitree G1 humanoid robot (29DoF + DEX3 hands, 43 joints total). The robot learns to walk, balance, and reach arbitrary 3D targets through a sequential curriculum trained entirely in simulation using NVIDIA Isaac Lab.

The system uses a **Separate Policy, Separate Obs** architecture (literature standard: SayCan, Berkeley, SoFTA, Mobile-TeleVision). Each policy has its own observation space, reward function, and PPO update. Previous policies are frozen during subsequent training.

```
                    HIERARCHICAL ARCHITECTURE

  VLM Layer          "Pick up the red cup"
  (Florence-2)    -> Target: {position, orientation}     ~1 Hz
       |
  Triple AC       <- vel_cmd + arm_target + hand_cmd
  (Separate PPO)  -> Joint actions (43 DoF)              50 Hz
       |
  Unitree G1         12 leg + 3 waist + 14 arm + 14 finger
```

---

## Key Results

### Stage 1: Omnidirectional Locomotion
| Metric | Value |
|--------|-------|
| Obs/Act | 66 / 15 (12 leg + 3 waist) |
| Curriculum | 9 levels (standing -> omni walk -> push robustness) |
| Velocity range | vx: -0.3~1.0, vy: +-0.4, vyaw: +-1.0 |

### Stage 2: Arm Position Reaching
| Metric | Value |
|--------|-------|
| Obs/Act | 39 / 7 (right arm) |
| Validated reach rate | 86.9% (play, 1 env) |
| Avg reach distance | 3.08 cm (< 4 cm industry target) |
| Avg EE displacement | 21.9 cm |
| Workspace | ~55 cm reach, spherical shell around shoulder |
| Falls | 0 in 3000 steps |

### Stage 3: Orientation Fine-Tune (Failed -- Decommissioned)
| Metric | Value |
|--------|-------|
| Orient error | ~2.18 rad (no improvement from Stage 2) |
| Position | Preserved (4.4 cm) but orient not learnable via RL |
| Conclusion | Heuristic wrist control or grasp-policy orientation needed |

---

## Training Pipeline (29DoF -- Active)

Each stage loads the previous checkpoint. Policies are trained sequentially with frozen predecessors.

| Stage | Task | Policy | Obs | Act | Status |
|-------|------|--------|-----|-----|--------|
| 1 | Omnidirectional locomotion | LocoAC | 66 | 15 | **Complete** |
| 2 | Arm position reaching | ArmAC (loco frozen) | 39 | 7 | **Complete** |
| 3 | Orientation fine-tune | ArmAC (critic reset) | 39 | 7 | Failed |
| 4 | Hand grasping | HandAC (loco+arm frozen) | ~50 | 14 | Planned |
| 5 | Skill chaining | Full pipeline | -- | -- | Planned |
| VLM | Semantic task execution | Florence-2 + skills | -- | -- | Planned |

---

## Triple Actor-Critic Architecture

```
LocoAC  (66 -> 15)  [512,256,128] + LayerNorm + ELU  -- Legs + waist
ArmAC   (39 -> 7)   [256,256,128] + ELU               -- Right arm (7 joints)
HandAC  (~50 -> 14)  [256,128,64] + ELU               -- DEX3 fingers (planned)
```

- Each policy has its **own obs space** (no shared/unified obs)
- **Legs:** Direct position control -- `leg_targets = default_pose + scale * policy_output`
- **Arms:** Residual actions -- `arm_targets = default_arm + scale * policy_output`
- Inference: `full_action = cat(loco_act, arm_act, hand_act)` -> env.step()

---

## Tech Stack

- **Simulation:** NVIDIA Isaac Lab 2.3.1, Isaac Sim 5.1.0
- **RL:** Custom PPO (PyTorch), Dual/Triple Actor-Critic
- **VLM (planned):** Florence-2 / Molmo2
- **Robot:** Unitree G1 29DoF + DEX3 (43 joints: 12 legs + 3 waist + 14 arms + 14 fingers)
- **Hardware:** NVIDIA RTX 5070 Ti (12 GB VRAM), Intel i9-13900HX, 32 GB RAM
- **Training:** 4096 parallel envs, ~17K steps/sec
- **Platform:** Windows 11 Pro, Python 3.10 (Anaconda env: env_isaaclab)

---

## Commands

All commands run from `C:\IsaacLab` with conda env `env_isaaclab` activated.

```powershell
cd C:\IsaacLab
conda activate env_isaaclab
```

### 29DoF Pipeline (Active)

#### PLAY (Evaluation)

**Stage 1: Omnidirectional Locomotion**
Modes: `--mode walk`, `--mode mixed`, `--mode push`
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_unified_stage_1.py --checkpoint logs/ulc/g1_unified_stage1_2026-02-27_00-05-20/model_best.pt --num_envs 1 --mode mixed
```

**Stage 2: Arm Position Reaching (< 4cm, 55cm reach)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_unified_stage_2_arm.py --checkpoint logs/ulc/g1_stage2_arm_2026-03-06_18-51-31/model_best.pt --num_envs 1 --mode standing --no_orient
```

**Stage 3: Orient Fine-Tune (Failed -- decommissioned)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_unified_stage_3_orient.py --checkpoint logs/ulc/g1_stage3_orient_2026-03-09_13-20-39/model_best.pt --num_envs 1 --mode standing
```

#### TRAIN

**Stage 1: Locomotion (from scratch)**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/29dof/train_unified_stage_1.py --num_envs 4096 --max_iterations 50000 --headless
```

**Stage 2: Arm Reaching (from Stage 1 checkpoint)**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/29dof/train_unified_stage_2_arm.py --stage1_checkpoint logs/ulc/g1_unified_stage1_2026-02-27_00-05-20/model_best.pt --num_envs 4096 --max_iterations 30000 --headless
```

**Stage 3: Orient Fine-Tune (from Stage 2 checkpoint -- failed experiment)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\29dof\train_unified_stage_3_orient.py --stage2_checkpoint logs/ulc/g1_stage2_arm_2026-03-06_18-51-31/model_best.pt --orient_weight 2.0 --num_envs 4096 --max_iterations 20000 --headless
```

### Hierarchical VLM+RL

**Loco+Arm Hierarchical Test**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\test_hierarchical.py --num_envs 4 --max_steps 3000 --checkpoint C:\IsaacLab\logs\ulc\g1_unified_stage1_2026-02-27_00-05-20\model_best.pt --arm_checkpoint C:\IsaacLab\logs\ulc\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\model_best.pt
```

**VLM Planning Demo**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py --num_envs 4 --checkpoint C:\IsaacLab\logs\ulc\g1_unified_stage1_2026-02-27_00-05-20\model_best.pt --arm_checkpoint C:\IsaacLab\logs\ulc\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\model_best.pt --task "Pick up the red cup and place it on the second table" --planner simple
```

### TensorBoard

```powershell
tensorboard --logdir logs/
```

### Legacy 23DoF Pipeline (Archive)

<details>
<summary>Click to expand legacy commands</summary>

#### PLAY (Legacy)

**Stage 1: Standing**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_1.py --checkpoint logs\ulc\ulc_g1_stage1_2026-01-05_17-27-57\model_best.pt --num_envs 4
```

**Stage 2: Walking**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_2.py --checkpoint logs\ulc\ulc_g1_stage2_v2_2026-01-08_16-42-40\model_best.pt --num_envs 4 --vx 0.5
```

**Stage 3: Torso Control**
```powershell
# Forward lean
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_3.py --checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --num_envs 4 --vx 0.0 --pitch -0.35

# Walking + lean
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_3.py --checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --num_envs 4 --vx 0.3 --pitch -0.2

# Side lean
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_3.py --checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --num_envs 4 --vx 0.2 --roll 0.15
```

**Stage 4: Dual Policy Arm Control**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_4_arm_dual.py
```

**Stage 5: Arm Reaching**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_5_arm.py --checkpoint logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt --num_envs 1
```

**Stage 5.5: Combined Loco+Arm**
```powershell
$env:PROJECT_ROOT = "C:\unitree_sim_isaaclab"
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_5.5_both.py --loco_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --arm_checkpoint logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt --num_envs 1 --vx 0.0
```

**Stage 6: Loco-Manipulation (Gaming detected)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_6_simplified.py --checkpoint logs/ulc/ulc_g1_stage6_simplified_2026-02-04_23-41-18/model_best.pt --num_envs 4 --mode walking
```

**Stage 7: Anti-Gaming Arm Reaching**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_7.py --checkpoint logs\ulc\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\model_best.pt --num_envs 1 --mode walking
```

**Paper Video Mode (30s+30s)**
```powershell
# Stage 7 paper demo
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_7.py --checkpoint logs\ulc\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\model_best.pt --mode paper

# Stage 6 gaming demo
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_ulc_stage_6_unified.py --checkpoint logs/ulc/ulc_g1_stage6_complete_2026-01-31_20-49-39/model_final.pt --loco_checkpoint logs/ulc/ulc_g1_stage7_antigaming_2026-02-06_17-41-47/model_best.pt --mode paper
```

#### TRAIN (Legacy)

**Stage 1-3: Standing -> Walking -> Torso**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_ulc.py --num_envs 4096 --headless --max_iterations 1500

.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_ulc_stage_2.py --num_envs 4096 --headless --max_iterations 6000 --stage1_checkpoint logs/ulc/ulc_g1_stage1_2026-01-05_17-27-57/model_best.pt

.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_ulc_stage_3.py --stage2_checkpoint logs/ulc/ulc_g1_stage2_v2_2026-01-08_16-42-40/model_best.pt --num_envs 4096 --headless --max_iterations 4000
```

**Stage 4-5: Arm Control**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_4_arm.py --num_envs 4096 --max_iterations 5000 --headless

.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_5_arm_full.py --num_envs 2048 --max_iterations 15000 --headless
```

**Stage 6-8: Loco-Manipulation**
```powershell
$env:PROJECT_ROOT = "C:\unitree_sim_isaaclab"
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_ulc_stage_6_simplified.py --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --num_envs 2048 --max_iterations 30000 --headless

.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_ulc_stage_7.py --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --num_envs 2048 --max_iterations 30000 --headless

.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_ulc_stage_8.py --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt --num_envs 2048 --max_iterations 30000 --headless
```

</details>

---

## Task Difficulty Reference

| Task | Difficulty | Isaac Lab Feasible | Priority |
|------|-----------|-------------------|----------|
| Pick-place | Easy | Yes | High |
| Drawer opening | Easy | Yes | High |
| Door opening (lever) | Hard | Yes | High |
| Window cleaning | Medium | Yes | Medium |
| Sock inside-out | Medium | Cloth sim needed | Low |
| Pan washing | Hard | Fluid sim needed | Low |
| Key -> Lock | Very Hard | Difficult | Low |
| Peanut butter sandwich | Very Long | No | Low |

> Note: Even pi-0.5 fails when trained "from scratch" (VLM initialization). Fine-tuning is required.

---

## Project Structure

```
g1/isaac_g1_ulc/
  config/                  Environment and scene configuration
  curriculum/              Sequential curriculum definitions
  envs/                    RL environments (ULC, arm reach, dual arm)
  train/
    29dof/                 Active 29DoF training scripts (Stage 1-3)
  play/                    Evaluation and demo scripts
  rewards/                 Modular reward functions
  utils/                   COM tracker, quintic interpolator, delay buffer
  test/                    Kinematics, workspace, joint tests
  demo/                    Decoupled walking + reaching demos
  data/                    Pre-computed workspace maps

vlm_integration/           VLM interface (Florence-2)
agents/                    PPO hyperparameter configs
external/                  Hardware integration (DDS, action provider)
```

---

## Lessons Learned

- **Separate policy, separate obs is essential.** Unified 188-dim obs (65% zeros) caused LayerNorm pollution and gradient dilution. 1 month of failed V1-V5.2. Each policy must define its own obs space.
- **Curriculum gaming is real.** Proximity rewards + smoothness penalties incentivize stillness. Movement-centric rewards (velocity toward target, progress) and 3-condition reach validation are essential.
- **Multi-task needs multi-critic.** A single critic receiving mixed signals produces noisy value estimates. Separate critics with separate GAE and PPO updates work much better.
- **Never change reward weights mid-training.** Changing orient weight from 3.0 to 6.0 on a trained checkpoint caused catastrophic position collapse (4.3cm -> 35cm). The critic's value landscape was invalidated.
- **Critic reset preserves position.** Resetting the critic to Xavier init while loading actor weights from a previous stage avoids reward scale mismatch. Position precision was fully preserved.
- **Orientation is not learnable with small gate.** ORIENT_GATE_DISTANCE=0.08m means orient reward only fires within 8cm of target, just before episode terminates. Two experiments (orient_weight 0.5 and 2.0) both failed. OrErr stuck at ~2.18 rad.
- **Training-play consistency matters.** Observation thresholds, action scales, and workspace definitions must match exactly between training and evaluation.

---

## Checkpoints

| Stage | Checkpoint | Key Metrics |
|-------|-----------|-------------|
| Stage 1 (Loco) | `logs/ulc/g1_unified_stage1_2026-02-27_00-05-20/model_best.pt` | 9-level curriculum complete |
| Stage 2 (Arm) | `logs/ulc/g1_stage2_arm_2026-03-06_18-51-31/model_best.pt` | EE=3.08cm, 86.9% rate |
| Stage 3 (Orient) | `logs/ulc/g1_stage3_orient_2026-03-09_13-20-39/model_best.pt` | Failed, OrErr=2.18 |

---

## Next Steps

1. **Hand Grasping (Stage 4):** HandPolicy (~50 obs -> 14 act), loco+arm frozen, DEX3 per-finger force sensors
2. **Skill Chaining (Stage 5):** walk_to -> squat -> grasp -> stand_up -> walk_to -> place
3. **VLM Planner:** SayCan/Berkeley architecture, task decomposition + skill executor
4. **End-to-end:** "Pick up the cup from the floor, place it on the table"
5. **Workshop paper** (ICRA/RSS)

---

## References

- [ULC: Unified Fine-Grained Controller for Humanoid Loco-Manipulation](https://arxiv.org/abs/2507.06905) -- Sun et al.
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) -- NVIDIA
- [Unitree G1](https://www.unitree.com/g1/) -- Unitree Robotics

---

## License and Citation

This is unpublished research work under MIT license.

```bibtex
@misc{yardimci2026g1ulcvlm,
  author = {Yardimci, Mehmet Turan},
  title  = {Hierarchical VLM-ULC for G1 Humanoid Loco-Manipulation},
  year   = {2026},
  note   = {Paper in preparation},
  url    = {https://github.com/mturan33/isaac-g1-ulc-vlm}
}
```

For collaboration or usage inquiries: mehmetturanyardimci@hotmail.com

---

## Author

**Mehmet Turan Yardimci**
- Cukurova University, Computer Engineering
- GitHub: [@mturan33](https://github.com/mturan33)
- LinkedIn: [/in/mehmetturanyardimci](https://linkedin.com/in/mehmetturanyardimci)
