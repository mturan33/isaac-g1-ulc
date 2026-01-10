# 🤖 G1 Hierarchical VLM-ULC System

**Unified Loco-manipulation Control for Unitree G1 Humanoid Robot with Vision-Language Model Integration**

> ⚠️ **RESEARCH IN PROGRESS — PAPER IN PREPARATION**  
> This repository contains original research work. If you use any part of this codebase, architecture, or methodology, **citation is required**. Unauthorized reproduction or publication of this work is prohibited. Contact the author for collaboration inquiries.

---

## 📋 Overview

A hierarchical control system combining Vision-Language Models (VLM) with Unified Loco-manipulation Control (ULC) for long-horizon task solving on the Unitree G1 humanoid robot.

```
┌─────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     "Go to the blue chair"                │
│  │     VLM     │ ← RGB Image + Language Command            │
│  │ Florence-2  │ → Target: {x, y, object_class}            │
│  └──────┬──────┘                                           │
│         │ ~1 Hz (Semantic Understanding)                   │
│  ┌──────▼──────┐                                           │
│  │  Semantic   │ → Object positions, scene graph           │
│  │  World Map  │ → Geometric tracking (100x faster)        │
│  └──────┬──────┘                                           │
│         │                                                  │
│  ┌──────▼──────┐                                           │
│  │    ULC      │ ← cmd_vel + arm_commands + torso_cmd      │
│  │   Policy    │ → Joint Actions (22 DoF)                  │
│  │    (PPO)    │   [12 legs + 10 arms]                     │
│  └──────┬──────┘                                           │
│         │ 50 Hz (Motor Control)                            │
│      [G1 🤖]                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Contributions

1. **Sequential Curriculum Learning:** 5-stage training from standing to full loco-manipulation
2. **Residual Action Modeling:** Stable arm control via small corrections around default poses
3. **Semantic World Model:** VLM for initial understanding + geometric tracking for real-time updates (~100x speedup)
4. **Unified Policy:** Single PPO policy for whole-body control (locomotion + torso + arms)

---

## 📊 Training Progress

| Stage | Task | Obs Dim | Act Dim | Status |
|-------|------|---------|---------|--------|
| 1 | Standing (Height Control) | 45 | 12 | ✅ Complete |
| 2 | Locomotion (Velocity Tracking) | 51 | 12 | ✅ Complete |
| 3 | Torso Control (Pitch/Roll/Yaw) | 57 | 12 | ✅ Complete |
| 4 | Arm Tracking (Residual Actions) | 77 | 22 | ✅ Complete |
| 5 | Full Integration + Workspace | 77 | 22 | 🔄 In Progress |
| 6 | VLM Integration | TBD | 22 | 📋 Planned |

---

## 🛠️ Tech Stack

- **Simulation:** NVIDIA Isaac Lab 2.3.1, Isaac Sim 5.1.0
- **RL Framework:** RSL-RL, PyTorch, PPO
- **VLM:** Florence-2 / Molmo2
- **Robot:** Unitree G1 (29 DoF configuration)
- **Hardware:** RTX 5070 Ti (12GB VRAM), 4096 parallel environments

---

## 🏗️ Architecture Details

### ULC Policy
- **Input:** Proprioception (joint pos/vel) + Commands (velocity, torso, arm targets) + Gait phase
- **Output:** Joint position targets for legs (12) + Residual corrections for arms (10)
- **Training:** ~17,000 steps/second with domain randomization

### Residual Action Modeling
```python
# Arms use residual actions around commanded positions
arm_targets = arm_commands + scale * tanh(policy_output)
# Legs use direct position control
leg_targets = default_pose + scale * policy_output
```

### Sequential Curriculum
Each stage builds on the previous checkpoint, progressively adding control complexity while maintaining stability.

---

## 📁 Project Structure

```
isaac-g1-ulc-vlm/
├── config/
│   └── ulc_g1_env_cfg.py      # Environment configuration
├── envs/
│   └── ulc_g1_env.py          # ULC environment implementation
├── train/
│   ├── train_ulc_stage_*.py   # Stage-specific training scripts
│   └── play_ulc_stage_*.py    # Evaluation scripts
├── vlm/
│   ├── vlm_wrapper.py         # Florence-2/Molmo2 interface
│   └── semantic_map.py        # World model with geometric tracking
└── README.md
```

---

## 🚀 Quick Start

```powershell
# Stage 4 Training (from Stage 3 checkpoint)
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/.../train/train_ulc_stage_4.py \
    --stage3_checkpoint logs/ulc/stage3_best.pt \
    --num_envs 4096 --headless

# Evaluation
./isaaclab.bat -p .../play/play_ulc_stage_4.py \
    --checkpoint logs/ulc/stage4_best.pt \
    --num_envs 4
```

---

## 📚 References

This work builds upon:

- [ULC: Unified Fine-Grained Controller for Humanoid Loco-Manipulation](https://arxiv.org/abs/2507.06905) - Sun et al.
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - NVIDIA
- [Unitree G1 Simulation](https://github.com/unitreerobotics/unitree_sim_isaaclab) - Unitree Robotics

---

## ⚖️ License & Citation

**This is unpublished research work.** The code is provided for reference only.

If you use this work, please cite:
```bibtex
@misc{yardimci2026g1ulcvlm,
  author = {Yardımcı, Mehmet Turan},
  title = {Hierarchical VLM-ULC for G1 Humanoid Loco-Manipulation},
  year = {2026},
  note = {Paper in preparation}
}
```

For collaboration or usage inquiries: mehmetturanyardimci@hotmail.com

---

## 👤 Author

**Mehmet Turan Yardımcı**  
- GitHub: [@mturan33](https://github.com/mturan33)  
- LinkedIn: [/in/mehmetturanyardimci](https://linkedin.com/in/mehmetturanyardimci)
