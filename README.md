# G1 Hierarchical VLM-ULC System

**Unified Loco-manipulation Control for the Unitree G1 Humanoid Robot with Vision-Language Model Integration**

> **Research in progress -- paper in preparation.**
> If you use any part of this codebase, architecture, or methodology, citation is required.
> Contact the author for collaboration inquiries.

[https://github.com/mturan33/isaac-g1-ulc-vlm/blob/main/g1_stage7_reaching_showcase.mp4]

---

## Overview

A hierarchical reinforcement learning system for whole-body loco-manipulation on the Unitree G1 humanoid robot. The robot learns to walk, balance, and reach arbitrary 3D targets through a sequential curriculum trained entirely in simulation using NVIDIA Isaac Lab.

The low-level controller (ULC) uses a Dual Actor-Critic PPO architecture: a frozen locomotion policy maintains stable bipedal walking while a separately trained arm policy learns to reach targets across the full workspace. An anti-gaming curriculum ensures the arm actively moves toward targets rather than exploiting resampling luck.

A Vision-Language Model layer (planned) will sit on top to enable semantic task execution such as "touch the red ball."

```
                    HIERARCHICAL ARCHITECTURE

  VLM Layer          "Touch the red ball"
  (Florence-2)    -> Target: {position, orientation}     ~1 Hz
       |
  ULC Policy      <- vel_cmd + arm_target + height_cmd
  (Dual PPO)      -> Joint actions (17 DoF)              50 Hz
       |
  Unitree G1         12 leg joints + 5 arm joints
```

---

## Key Results (Stage 7)

| Metric | Value |
|--------|-------|
| Validated reaches | 297 in 3000 steps (60s) |
| Success rate | 100% |
| Workspace | 18--40 cm spherical shell around shoulder |
| Avg reach distance | 0.068 m |
| Avg EE displacement | 0.184 m |
| Training throughput | ~17,000 steps/s with 4096 parallel envs |

---

## Training Curriculum

The system is trained through a sequential curriculum where each stage loads the previous checkpoint and adds new control complexity.

| Stage | Task | Architecture | Obs | Act | Status |
|-------|------|-------------|-----|-----|--------|
| 1 | Standing (height control) | Single Actor-Critic | 45 | 12 | Complete |
| 2 | Bipedal locomotion | Single Actor-Critic | 51 | 12 | Complete |
| 3 | Torso control (pitch/roll/yaw) | Single Actor-Critic | 57 | 12 | Complete |
| 4 | Fixed-base arm reaching | Single Actor-Critic | 77 | 22 | Complete |
| 5 | Arm reaching (workspace mapping) | Single Actor-Critic | 77 | 22 | Complete |
| 6 | Loco-manipulation | Dual Actor-Critic | 57+52 | 17 | Complete (gaming detected) |
| 7 | Anti-gaming arm reaching | Dual AC (loco frozen) | 57+55 | 17 | **Complete** |
| 8 | Orientation control | Planned | -- | -- | Planned |
| VLM | Semantic task execution | Planned | -- | -- | Planned |

### Stage 6: Loco-Manipulation (Dual Actor-Critic)

Introduces separate actor-critic networks for locomotion and arm control with independent reward functions, GAE computation, and PPO updates.

```
LocoActor (57 -> 12 leg actions)   + LocoCritic (57 -> 1)
ArmActor  (52 -> 5 arm actions)    + ArmCritic  (52 -> 1)
```

A 13-level curriculum progresses from standing+reaching through walking+reaching to variable end-effector orientation control. **Problem discovered:** the robot learned to stay still and let targets resample nearby by chance rather than actively reaching (curriculum gaming).

### Stage 7: Anti-Gaming Arm Reaching

Freezes the locomotion policy from Stage 6 and retrains the arm policy from scratch with five anti-gaming mechanisms:

1. **Absolute-only target sampling** with minimum distance enforcement
2. **3-condition reach validation:** position threshold + EE displacement + time limit
3. **Validated reach rate** for curriculum advancement (not total count)
4. **Movement-centric rewards:** velocity-toward-target, progress, stillness penalty
5. **Gaming detection:** refuses to advance if timeout rate exceeds 90%

The arm observation space is extended to 55 dimensions (52 base + steps_since_spawn, ee_displacement, initial_distance).

---

## Dual Actor-Critic Architecture

```
                   Observation (112 dim)
                  /                     \
         Loco obs (57)             Arm obs (55)
              |                         |
    LocoActor [512,256,128]    ArmActor [256,256,128]
    LayerNorm + ELU            ELU (no LayerNorm)
              |                         |
       12 leg actions             5 arm actions
              |                         |
               \                       /
                Combined (17 actions)
                        |
                    Unitree G1
```

- **Legs:** Direct position control -- `leg_targets = default_pose + scale * policy_output`
- **Arms:** Residual actions -- `arm_targets = arm_commands + scale * tanh(policy_output)`
- Locomotion branch is fully frozen (`requires_grad=False`) during Stage 7
- Arm branch uses high initial exploration (`log_std = log(0.8)`)

---

## Workspace Definition

Targets are sampled in a spherical shell around the right shoulder in the robot's body frame:

- **Radius:** 18--40 cm (inner limit from physical reachability)
- **Azimuth:** [-0.3, 1.2] rad (front-right region)
- **Elevation:** [-0.4, 0.6] rad (below shoulder to above head)
- **Coordinate system:** -X = forward, -Y = right (G1 convention, inverted from standard)

```
     -X (FRONT)
          |
   -Y ----+---- +Y
          |
     +X (BACK)
```

---

## Tech Stack

- **Simulation:** NVIDIA Isaac Lab 2.3.1, Isaac Sim 5.1.0
- **RL:** RSL-RL, PyTorch, PPO (Proximal Policy Optimization)
- **VLM (planned):** Florence-2 / Molmo2
- **Robot:** Unitree G1 (29 DoF, 22 DoF policy output)
- **Hardware:** NVIDIA RTX 5070 Ti (12 GB VRAM), Intel i9-13900HX, 32 GB RAM
- **Platform:** Windows 11 Pro, Python 3.10

---

## Project Structure

```
g1/isaac_g1_ulc/
  config/                  Environment and scene configuration
  curriculum/              Sequential curriculum definitions
  envs/                    RL environments (ULC, arm reach, dual arm)
  train/                   Training scripts per stage
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

## Quick Start

All commands run from the Isaac Lab root directory.

```bash
# Stage 7 training (from Stage 6 checkpoint)
./isaaclab.bat -p source/isaaclab_tasks/.../train/train_ulc_stage_7.py \
    --stage6_checkpoint logs/ulc/.../model_best.pt \
    --num_envs 4096 --headless

# Evaluation (position reaching, no orientation check)
./isaaclab.bat -p source/isaaclab_tasks/.../play/play_ulc_stage_7.py \
    --checkpoint logs/ulc/.../model_best.pt \
    --num_envs 1 --mode standing --no_orient

# Showcase demo with video recording
./isaaclab.bat -p source/isaaclab_tasks/.../play/play_ulc_stage_7.py \
    --checkpoint logs/ulc/.../model_best.pt \
    --mode showcase --record --record_duration 10
```

---

## Lessons Learned

- **Curriculum gaming is real.** Proximity rewards combined with smoothness penalties incentivize stillness. Movement-centric rewards (velocity toward target, progress tracking) are essential.
- **Multi-task needs multi-critic.** A single critic receiving mixed locomotion and arm signals produces noisy value estimates. Separate critics with separate GAE and PPO updates work much better.
- **Reach validation needs three conditions.** Position threshold alone is insufficient -- EE displacement and time limits prevent counting lucky spawns as genuine reaches.
- **Training-play consistency matters.** Observation thresholds, action scales, and workspace definitions must match exactly between training and evaluation. A 0.04 m vs 0.08 m mismatch in the `target_reached` observation caused the policy to receive incorrect signals.
- **Orientation is much harder than position.** The arm policy learned reliable position reaching but failed to learn palm orientation. Statistical success in 4096 parallel environments does not transfer to single-environment evaluation.

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
- GitHub: [@mturan33](https://github.com/mturan33)
- LinkedIn: [/in/mehmetturanyardimci](https://linkedin.com/in/mehmetturanyardimci)
