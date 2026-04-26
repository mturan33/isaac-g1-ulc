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
| 2L | Perturbation-robust loco | LocoAC (arm frozen perturbation) | 66 | 15 | **Complete** |
| 3L | Variable height / squat | LocoAC (arm frozen) | 66 | 15 | Parked (diminishing returns) |
| 3G | DEX3 finger grasping | GraspAC (fix_root_link) | 45 | 7 | **Active** (Phase A V3) |
| 4 | Skill chaining | Full pipeline | -- | -- | Planned |
| VLM | Semantic task execution | Florence-2 + skills | -- | -- | Planned |

### Grasp Training (Stage 3G -- Active)

Fixed-base robot, 3 object shapes (sphere/cylinder/box, 33% each), finger-only policy.

**Key findings:**
- Passive grasp exploit: objects fall between fingers, policy learns "do nothing" (V1-V2)
- Fix: proximity-gated finger closure reward (`closure * exp(-5*dist)`)
- Reward budget analysis via RewardLogger (TensorBoard RR/RW/RB/ prefixes)
- Object spawn must be within 5cm of palm (too far = fingers can't reach)

---

## Triple Actor-Critic Architecture

```
LocoAC   (66 -> 15)  [512,256,128] + LayerNorm + ELU  -- Legs + waist
ArmAC    (39 -> 7)   [256,256,128] + ELU               -- Right arm (7 joints)
GraspAC  (45 -> 7)   [256,128,64]  + ELU               -- DEX3 right hand (7 finger joints)
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
- **Hardware:** NVIDIA RTX 5070 Ti Laptop (12 GB VRAM), Intel i9-13900HX, 64 GB DDR5-5200 dual-channel
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

> **Pre-trained checkpoints are included in the repository under `checkpoints/`**.
> No need to train from scratch — just clone and run play/demo commands below.

**Stage 1: Omnidirectional Locomotion**
Modes: `--mode walk`, `--mode mixed`, `--mode push`
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_unified_stage_1.py --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage1.pt --num_envs 1 --mode mixed
```

**Stage 2: Arm Position Reaching (< 4cm, 55cm reach)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_unified_stage_2_arm.py --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/arm_stage2.pt --num_envs 1 --mode standing --no_orient
```

**Stage 2 Loco (Perturbation-robust, 50K iter)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\play\play_unified_stage_2_loco.py --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage2.pt --num_envs 1 --mode walk
```

**Stage 3: Orient Fine-Tune (Failed -- decommissioned, no checkpoint shipped)**
```powershell
# Stage 3 was decommissioned (worse than Stage 2). Train your own if needed:
# .\isaaclab.bat -p ...play_unified_stage_3_orient.py --checkpoint <your-stage3-ckpt> --num_envs 1 --mode standing
```

#### TRAIN

**Stage 1: Locomotion (from scratch)**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/29dof/train_unified_stage_1.py --num_envs 4096 --max_iterations 50000 --headless
```

**Stage 2: Arm Reaching (from Stage 1 checkpoint)**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/29dof/train_unified_stage_2_arm.py --stage1_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage1.pt --num_envs 4096 --max_iterations 30000 --headless
```

**Stage 2 Loco: Perturbation-robust (from Stage 1 + Stage 2 Arm)**
```powershell
.\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/29dof/train_unified_stage_2_loco.py --stage1_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage1.pt --arm_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/arm_stage2.pt --num_envs 2048 --max_iterations 50000 --headless
```

**Stage 3: Orient Fine-Tune (from Stage 2 checkpoint -- failed experiment)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\29dof\train_unified_stage_3_orient.py --stage2_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/arm_stage2.pt --orient_weight 2.0 --num_envs 4096 --max_iterations 20000 --headless
```

**Grasp Phase A: Fixed-Base Finger Training (3 shapes: sphere/cylinder/box)**
```powershell
# Training (from scratch)
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_grasp_phase_a.py --num_envs 2048 --max_iterations 40000 --headless

# Smoke test (visual)
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_grasp_phase_a.py --num_envs 64 --max_iterations 100
```

**Grasp Phase B: Fixed-Base + Frozen Arm Reaching**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_ulc\g1\isaac_g1_ulc\train\train_grasp_phase_b.py --arm_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage2.pt --num_envs 2048 --max_iterations 50000 --headless
```

### Hierarchical VLM+RL

**Loco+Arm Hierarchical Test**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\test_hierarchical.py --num_envs 4 --max_steps 3000 --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage1.pt --arm_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/arm_stage2.pt
```

**VLM Planning Demo (Pick-and-Place)**
```powershell
.\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\demo_vlm_planning.py --num_envs 4 --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/loco_stage2.pt --arm_checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/arm_stage2.pt --task "Pick up the red cup and place it on the second table" --planner simple
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
  config/                  Environment and scene configuration (29DoF joints, actuators)
  curriculum/              Sequential curriculum definitions
  envs/                    RL environments (ULC, arm reach, dual arm)
  train/
    29dof/                 Active 29DoF training scripts (Stage 1-3, loco/arm)
    train_grasp_phase_a.py Grasp training: fixed base, finger-only, 3 shapes
    train_grasp_phase_b.py Grasp training: fixed base + frozen arm reaching
  play/                    Evaluation and demo scripts
  rewards/                 Modular reward functions
  utils/
    reward_logger.py       Per-component reward breakdown (TensorBoard RR/RW/RB/)
    com_tracker.py         Center-of-mass stability tracking
    quintic_interpolator.py Smooth trajectory generation
    delay_buffer.py        Action delay simulation
  test/                    Kinematics, workspace, joint tests (20+ files)
  demo/                    Decoupled walking + reaching demos
  data/                    Pre-computed workspace maps

vlm_integration/           VLM interface (Florence-2)
agents/                    PPO hyperparameter configs
external/                  Hardware integration (DDS, action provider, camera)
```

---

## Lessons Learned

- **Separate policy, separate obs is essential.** Unified 188-dim obs (65% zeros) caused LayerNorm pollution and gradient dilution. 1 month of failed V1-V5.2. Each policy must define its own obs space.
- **Curriculum gaming is real.** Proximity rewards + smoothness penalties incentivize stillness. Movement-centric rewards (velocity toward target, progress) and 3-condition reach validation are essential.
- **Multi-task needs multi-critic.** A single critic receiving mixed signals produces noisy value estimates. Separate critics with separate GAE and PPO updates work much better.
- **Passive grasp exploit is the #1 grasping failure mode.** If objects can fall into the hand via gravity, policy learns "do nothing." Fix: proximity-gated rewards (`reward * exp(-k*dist)`), spawn objects within finger reach (5cm), and ensure approach reward dominates budget.
- **RewardLogger is essential for debugging.** TensorBoard RB/ (reward budget %) reveals dead rewards (<1%) and dominant rewards (>30%) instantly. Without it, reward design is blind guessing.
- **Curriculum gate must check task-specific metrics.** Reward threshold alone is insufficient. Height tracking needs `h_err < 0.05m`, grasping needs `grasp_success_rate > 0.3`. Without task gates, robot advances without learning.
- **KL penalty blocks radical behavior change.** Fine-tuning with KL=0.02 preserves old behavior. For new tasks (squat, grasp), KL must be 0.005 or lower.
- **fix_root_link=True for grasp training.** Free-standing robot with frozen loco policy requires exact obs/action matching. Any mismatch causes immediate collapse. Use fixed root until loco integration is validated separately.
- **Training-play consistency matters.** Observation thresholds, action scales, and workspace definitions must match exactly between training and evaluation.

---

## Checkpoints

**Pre-trained checkpoints are shipped in this repository under `checkpoints/`** (no manual training required).

| Stage | Repo Path | Key Metrics |
|-------|-----------|-------------|
| Stage 1 (Loco) | `checkpoints/loco_stage1.pt` | 9-level curriculum complete, 20K iter |
| Stage 2 (Arm) | `checkpoints/arm_stage2.pt` | EE=3.08 cm reach, 86.9% rate, 20K iter |
| Stage 2L (Loco robust) | `checkpoints/loco_stage2.pt` | 50K iter, perturbation-robust, lateral carry stable |

Run all play/demo commands from `C:\IsaacLab\` and reference the path
`source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/checkpoints/<file>.pt`.

**Decommissioned (not shipped):**
- Stage 3 Orient: failed experiment, worse than Stage 2
- Stage 3 Squat: parked (diminishing returns)

**Grasp Phase A:** train your own (see TRAIN section above).

---

## Next Steps

1. **Grasp Phase A V3:** Proximity-gated closure reward, approach=8.0, solve passive exploit
2. **Grasp Phase B:** Frozen arm policy + finger training (fix_root_link=True)
3. **Grasp Phase C:** Frozen loco + arm + finger (full standing robot)
4. **Skill Chaining:** walk_to -> reach -> grasp -> stand_up -> walk_to -> place
5. **VLM Planner:** SayCan/Berkeley architecture, task decomposition + skill executor
6. **End-to-end:** "Pick up the cup from the table, place it in the box"
7. **Workshop paper** (ICRA/RSS)

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
