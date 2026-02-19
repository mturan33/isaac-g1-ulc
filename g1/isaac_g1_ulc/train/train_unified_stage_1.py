"""
Unified Stage 1: Standing + Omnidirectional Locomotion with Decoupled Heading Control
======================================================================================
Future-proof unified obs/action space — 188 dim obs, 43 dim act.
Only the loco branch (15 act) is active in this stage.

ARCHITECTURE: Triple Actor-Critic (only LocoAC trained here)
- LocoActor:  188 obs -> 15 act (12 leg + 3 waist) — ACTIVE
- ArmActor:   188 obs -> 14 act (placeholder, frozen)
- HandActor:  188 obs -> 14 act (placeholder, frozen)

COMMAND SPACE: Decoupled Heading Control
- speed_cmd:        [0, 1.2] m/s (scalar, always >= 0)
- heading_offset:   [-pi, pi] rad (movement direction vs body facing)
- body_yaw_cmd:     yaw_rate [-1.0, 1.0] rad/s OR absolute yaw [-pi, pi]
- yaw_mode:         0.0 = rate mode, 1.0 = absolute mode

Body frame decomposition:
  vx_body = speed * cos(heading_offset)
  vy_body = speed * sin(heading_offset)

OBS SPACE (188 dim — NEVER changes across stages):
  [0:73]     Loco branch (active from this stage)
  [73:143]   Arm branch  (zeros — active from Stage 3)
  [143:188]  Hand branch (zeros — active from Stage 4)

CURRICULUM (10 levels):
  L0: Standing (speed=0, all commands zero)
  L1-L2: Forward walk (speed increasing, heading=0)
  L3-L4: Turning + lateral (heading_offset active)
  L5: Full omnidirectional (rate mode)
  L6: Absolute yaw transition (50/50 rate/absolute)
  L7: Full absolute yaw
  L8-L9: Variable height + strong push (FINAL)

2026-02-12: Initial implementation — decoupled heading, unified obs space.
2026-02-12: Expanded obs 143→188 — added all future-needed obs (anti-gaming,
            per-finger forces, ee velocities, payload estimate, etc.) to prevent
            weight surgery in later stages.
2026-02-15: V2 — Major gait quality fixes:
            - Quaternion order fix (wxyz for Isaac Lab)
            - quat_to_euler_xyz fix (wxyz input)
            - enabled_self_collisions=True
            - hip_yaw_penalty added (prevents leg crossing/rotation)
            - feet_air_time reward added (prevents toe-walking, encourages ground contact cycle)
            - foot_flatness increased (1.5→3.0, scale -8→-15)
            - height reward reduced (5.0/-15→3.5/-10, was causing toe-walking)
            - standing_posture reduced (5.0/-5→3.0/-3)
            - orientation reduced (5.0/-15→4.0/-12)
            - action clamp on hip_yaw (prevents extreme leg rotation)
            - solver iterations increased (4→8 for self-collision)
            - jerk penalty increased (-0.05→-0.08)
2026-02-16: V3 — Anti-gaming posture enforcement (iter ~16400 posture collapse fix):
            - V2 problem: robot found exploit — bent knees (1.15→0.64), leaned forward
              (wP=-0.18), rotated waist (wY=-0.14), reward INCREASED (18→24.5) due to
              alive accumulation (EpLen 24000+) and weak speed penalty (3x overshoot=%74).
            - Waist termination tightened: pitch 0.35→0.20, roll 0.25→0.15
            - waist_posture weight 3.0→4.0, scales increased (yaw:-30, roll:-25, pitch:-20)
            - NEW: knee_min_penalty=-5.0 (knee<0.8 rad = linear penalty, prevents squat)
            - NEW: gait_stance_posture=2.0 (stance knee must return to default)
            - speed scale -3.0→-6.0 (3x overshoot now=56% reward, was 74%)
            - height scale -10.0→-15.0 (tighter height tracking)
            - orientation 4.0→5.0, scale -12.0→-15.0 (Stage 2 level enforcement)
            - gait_knee 3.0→3.5 (stronger gait pattern)
            - alive 1.0→0.5 (reduce long-episode accumulation gaming)
            - standing_posture min floor 0.3 (partially active even during walking)
2026-02-18: V4 — Heading gate fix + posture collapse prevention:
            - V3 problem: heading gate blocked L2 (heading_err=1.3-1.4, threshold=0.5)
              because heading_offset=0 but yaw_cmd active — body rotation makes heading
              tracking impossible. Also knee collapsed to 0.61 despite knee_min_penalty.
            - Heading gate skip for L0-L2 (heading_offset=0 levels, gate only L3+)
            - knee_collapse termination: knee < 0.3 rad = terminate (default knee=0.42, 0.5 was too high)
            - waist_roll termination tightened: 0.15→0.10
            - knee_min_penalty: -5.0→-8.0, threshold 0.8→0.7 (heavier penalty, slightly wider range)
            - alive: 0.5→0.2 (further reduce accumulation gaming)
2026-02-19: V5 — Stage 2 alignment + Isaac Lab reference rebalance:
            - V4 problem: posture collapse at L2 transition (knee 1.14→0.82, tilt 2.7→9.3
              in 200 iter). Same pattern as V2/V3. Root cause: reward/termination too
              aggressive vs working Stage 2 (29DoF, 66 obs) which reached L4 without collapse.
            - Isaac Lab official G1 config confirms: ~7 reward terms, low weights, wide tolerances.
            - Philosophy: "Return to the working system, keep only heading gate fix."
            - foot_flatness: 3.0/-15→1.5/-8 (Stage 2, was 4x too aggressive)
            - standing_posture floor: min=0.3→min=0.0 (off during walking, Stage 2 behavior)
            - knee_min_penalty: -8.0→0.0 (REMOVED, prevented natural knee bending)
            - knee_collapse termination: REMOVED (prevented natural gait)
            - waist termination: pitch 0.20→0.35, roll 0.10→0.25 (Stage 2 values)
            - alive: 0.2→1.0 (Stage 2 value, encourages survival/exploration)
            - speed scale: -6.0→-3.0 (Stage 2 value)
            - orientation: 5.0→6.0 (Stage 2 value)
            - waist_posture: 4.0→3.0, scales -30/-25/-20→-25/-20/-15 (Stage 2)
            - height scale: -15.0→-10.0 (Stage 2 value)
            - gait_knee: 3.5→3.0 (Stage 2 value)
            - KEPT from V4: heading gate skip L0-L2
            - KEPT from V2: hip_yaw_penalty, self_collisions, quaternion wxyz fix
            - KEPT from V3: gait_stance_posture
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime

# ============================================================================
# IMPORTS & CONFIG
# ============================================================================

import importlib.util
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "config", "ulc_g1_29dof_cfg.py")
_spec = importlib.util.spec_from_file_location("ulc_g1_29dof_cfg", _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

G1_29DOF_USD = _cfg_mod.G1_29DOF_USD
LOCO_JOINT_NAMES = _cfg_mod.LOCO_JOINT_NAMES
LEG_JOINT_NAMES = _cfg_mod.LEG_JOINT_NAMES
WAIST_JOINT_NAMES = _cfg_mod.WAIST_JOINT_NAMES
ARM_JOINT_NAMES = _cfg_mod.ARM_JOINT_NAMES
HAND_JOINT_NAMES = _cfg_mod.HAND_JOINT_NAMES
DEFAULT_LOCO_LIST = _cfg_mod.DEFAULT_LOCO_LIST
DEFAULT_ARM_LIST = _cfg_mod.DEFAULT_ARM_LIST
DEFAULT_HAND_LIST = _cfg_mod.DEFAULT_HAND_LIST
DEFAULT_ALL_POSES = _cfg_mod.DEFAULT_ALL_POSES
NUM_LOCO_JOINTS = _cfg_mod.NUM_LOCO_JOINTS
NUM_ARM_JOINTS = _cfg_mod.NUM_ARM_JOINTS
NUM_HAND_JOINTS = _cfg_mod.NUM_HAND_JOINTS
LEG_ACTION_SCALE = _cfg_mod.LEG_ACTION_SCALE
WAIST_ACTION_SCALE = _cfg_mod.WAIST_ACTION_SCALE
ARM_ACTION_SCALE = _cfg_mod.ARM_ACTION_SCALE
HAND_ACTION_SCALE = _cfg_mod.HAND_ACTION_SCALE
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS

# ============================================================================
# UNIFIED OBS/ACT DIMENSIONS — NEVER CHANGE THESE
# ============================================================================
# Design principle: Include ALL obs that ANY future stage might need.
# Unused obs are zero in early stages. This prevents weight surgery forever.
# Never remove obs for "faster training" — correctness > speed.

# Loco branch: 73 dim
# [0:3]   lin_vel_b             (3)  — body frame linear velocity
# [3:6]   ang_vel_b             (3)  — body frame angular velocity
# [6:9]   proj_gravity          (3)  — gravity projection (uprightness)
# [9:21]  leg_joint_pos         (12) — 12 leg joint positions
# [21:33] leg_joint_vel         (12) — 12 leg joint velocities (x0.1)
# [33:36] waist_joint_pos       (3)  — waist yaw/roll/pitch positions
# [36:39] waist_joint_vel       (3)  — waist yaw/roll/pitch velocities (x0.1)
# [39:40] height_cmd            (1)  — target height (0.40-0.85m)
# [40:41] speed_cmd             (1)  — speed magnitude (0-1.2 m/s)
# [41:42] heading_offset_cmd    (1)  — heading angle offset (rad, -pi to pi)
# [42:43] body_yaw_cmd          (1)  — yaw rate OR absolute yaw target
# [43:44] yaw_mode              (1)  — 0.0=rate mode, 1.0=absolute mode
# [44:46] gait_phase            (2)  — sin/cos @ GAIT_FREQUENCY Hz
# [46:61] prev_loco_actions     (15) — previous loco action history
# [61:64] torso_euler           (3)  — current roll, pitch, yaw from quaternion
# [64:67] torso_cmd             (3)  — target roll, pitch, yaw
# [67:69] foot_contact          (2)  — left/right foot ground contact (bool->float)
# [69:70] current_height        (1)  — actual root height (for squat awareness)
# [70:71] height_err            (1)  — height_cmd - current_height (signed)
# [71:72] root_height_vel       (1)  — vertical velocity of root (squat transitions)
# [72:73] payload_estimate      (1)  — estimated additional mass (0 if empty-handed)
LOCO_OBS_DIM = 73

# Arm branch: 70 dim (zeros in this stage, active from Stage 3)
# [73:87]   arm_joint_pos         (14) — 7 left + 7 right arm positions
# [87:101]  arm_joint_vel         (14) — arm joint velocities (x0.1)
# [101:104] left_ee_pos_b         (3)  — left end-effector pos (body frame)
# [104:107] right_ee_pos_b        (3)  — right end-effector pos (body frame)
# [107:110] left_ee_vel_b         (3)  — left end-effector velocity (body frame)
# [110:113] right_ee_vel_b        (3)  — right end-effector velocity (body frame)
# [113:116] left_target_pos_b     (3)  — left arm target pos (body frame)
# [116:119] right_target_pos_b    (3)  — right arm target pos (body frame)
# [119:122] left_ee_orient_err    (3)  — left orientation error (axis-angle)
# [122:125] right_ee_orient_err   (3)  — right orientation error (axis-angle)
# [125:126] left_steps_norm       (1)  — anti-gaming: normalized time since left target spawn
# [126:127] right_steps_norm      (1)  — anti-gaming: normalized time since right target spawn
# [127:128] left_ee_displacement  (1)  — anti-gaming: left arm displacement since spawn
# [128:129] right_ee_displacement (1)  — anti-gaming: right arm displacement since spawn
# [129:130] left_initial_dist     (1)  — anti-gaming: initial distance to left target
# [130:131] right_initial_dist    (1)  — anti-gaming: initial distance to right target
# [131:132] left_grip_force       (1)  — left hand total grip force (N)
# [132:133] right_grip_force      (1)  — right hand total grip force (N)
# [133:134] left_contact_detected (1)  — left hand contact bool
# [134:135] right_contact_detected(1)  — right hand contact bool
# [135:138] object_pos_b          (3)  — target object position (body frame)
# [138:141] object_orient_err     (3)  — object orientation error (axis-angle)
# [141:143] estimated_load        (2)  — estimated load on each arm (left, right)
ARM_OBS_DIM = 70

# Hand branch: 45 dim (zeros in this stage, active from Stage 4)
# [143:157] hand_joint_pos         (14) — 7 left + 7 right finger positions
# [157:171] hand_joint_vel         (14) — 7 left + 7 right finger velocities (x0.1)
# [171:178] left_finger_forces     (7)  — per-link force: index0,index1,mid0,mid1,thumb0,thumb1,thumb2
# [178:185] right_finger_forces    (7)  — per-link force: index0,index1,mid0,mid1,thumb0,thumb1,thumb2
# [185:186] grasp_cmd              (1)  — 0=open, 1=close
# [186:187] left_object_in_hand    (1)  — 0=empty, 1=holding in left hand
# [187:188] right_object_in_hand   (1)  — 0=empty, 1=holding in right hand
HAND_OBS_DIM = 45

OBS_DIM = LOCO_OBS_DIM + ARM_OBS_DIM + HAND_OBS_DIM  # 188
LOCO_ACT_DIM = NUM_LOCO_JOINTS   # 15
ARM_ACT_DIM = NUM_ARM_JOINTS     # 14
HAND_ACT_DIM = NUM_HAND_JOINTS   # 14
ACT_DIM = LOCO_ACT_DIM + ARM_ACT_DIM + HAND_ACT_DIM  # 43

# ============================================================================
# CURRICULUM — 10 LEVELS (Standing through Full Omnidirectional + Variable Height)
# ============================================================================

# NOTE: 29DoF G1 body frame — +X = FORWARD, -X = BACKWARD
CURRICULUM = [
    {   # L0: Standing — all commands zero, learn to stand
        "description": "L0: Standing (all cmds zero)",
        "threshold": 18.0,
        "speed": (0.0, 0.0),
        "heading_offset": (0.0, 0.0),
        "yaw_cmd": (0.0, 0.0),
        "yaw_mode": "rate",       # rate only
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 0),
        "push_interval": (9999, 9999),
        "mass_scale": (1.0, 1.0),
    },
    {   # L1: Slow forward walk — heading=0 (straight ahead)
        "description": "L1: Slow forward walk",
        "threshold": 19.0,
        "speed": (0.0, 0.3),
        "heading_offset": (0.0, 0.0),  # Only forward
        "yaw_cmd": (-0.2, 0.2),
        "yaw_mode": "rate",
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 0),
        "push_interval": (9999, 9999),
        "mass_scale": (0.97, 1.03),
    },
    {   # L2: Medium walk with slight yaw
        "description": "L2: Medium walk + yaw",
        "threshold": 20.0,
        "speed": (0.0, 0.5),
        "heading_offset": (0.0, 0.0),
        "yaw_cmd": (-0.4, 0.4),
        "yaw_mode": "rate",
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 5),
        "push_interval": (400, 800),
        "mass_scale": (0.95, 1.05),
    },
    {   # L3: Turning — heading_offset introduced
        "description": "L3: Turning + heading offset",
        "threshold": 21.0,
        "speed": (0.0, 0.5),
        "heading_offset": (-0.3, 0.3),  # ~17 degrees
        "yaw_cmd": (-0.7, 0.7),
        "yaw_mode": "rate",
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 5),
        "push_interval": (400, 800),
        "mass_scale": (0.95, 1.05),
    },
    {   # L4: Lateral walking — heading up to +-90 degrees
        "description": "L4: Lateral + aggressive yaw",
        "threshold": 22.0,
        "speed": (0.0, 0.7),
        "heading_offset": (-1.57, 1.57),  # +-pi/2
        "yaw_cmd": (-0.8, 0.8),
        "yaw_mode": "rate",
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 10),
        "push_interval": (300, 600),
        "mass_scale": (0.93, 1.07),
    },
    {   # L5: Full omnidirectional — all directions, rate mode
        "description": "L5: Full omni (rate mode)",
        "threshold": 23.0,
        "speed": (0.0, 1.0),
        "heading_offset": (-3.14, 3.14),  # Full circle
        "yaw_cmd": (-1.0, 1.0),
        "yaw_mode": "rate",
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 15),
        "push_interval": (200, 500),
        "mass_scale": (0.90, 1.10),
    },
    {   # L6: Absolute yaw transition — 50/50 rate/absolute
        "description": "L6: Absolute yaw transition (50/50)",
        "threshold": 23.5,
        "speed": (0.0, 1.0),
        "heading_offset": (-3.14, 3.14),
        "yaw_cmd": (-3.14, 3.14),  # Now absolute range
        "yaw_mode": "mixed",       # 50% rate, 50% absolute
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 15),
        "push_interval": (200, 500),
        "mass_scale": (0.90, 1.10),
    },
    {   # L7: Full absolute yaw — "walk north while facing east"
        "description": "L7: Full absolute yaw",
        "threshold": 24.0,
        "speed": (0.0, 1.0),
        "heading_offset": (-3.14, 3.14),
        "yaw_cmd": (-3.14, 3.14),
        "yaw_mode": "absolute",
        "height": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "push_force": (0, 20),
        "push_interval": (200, 500),
        "mass_scale": (0.88, 1.12),
    },
    {   # L8: Variable height — squat preparation
        "description": "L8: Variable height (0.65-0.85)",
        "threshold": 24.5,
        "speed": (0.0, 1.2),
        "heading_offset": (-3.14, 3.14),
        "yaw_cmd": (-3.14, 3.14),
        "yaw_mode": "absolute",
        "height": (0.65, 0.85),
        "push_force": (0, 25),
        "push_interval": (200, 500),
        "mass_scale": (0.85, 1.15),
    },
    {   # L9: FINAL — full range + strong push
        "description": "L9: FINAL (full range + strong push)",
        "threshold": None,
        "speed": (0.0, 1.2),
        "heading_offset": (-3.14, 3.14),
        "yaw_cmd": (-3.14, 3.14),
        "yaw_mode": "absolute",
        "height": (0.55, 0.85),
        "push_force": (0, 30),
        "push_interval": (100, 300),
        "mass_scale": (0.85, 1.15),
    },
]

# ============================================================================
# REWARD WEIGHTS
# ============================================================================

REWARD_WEIGHTS = {
    # Velocity / Heading tracking — V5: speed scale reverted to Stage 2 (-3.0)
    "speed": 4.0,                   # exp(-3.0 * speed_err) — V5: -6→-3 (Stage 2 value)
    "heading": 4.0,                 # exp(-2.0 * heading_err) * heading_active
    "yaw": 4.0,                     # exp(-3.0 * yaw_err), mode-dependent
    # Gait control (scaled by speed magnitude)
    "gait_knee": 3.0,               # V5: 3.5→3.0 (Stage 2 value)
    "gait_clearance": 2.0,          # Hip pitch swing
    "gait_contact": 2.0,            # Contact pattern matching
    "gait_stance_posture": 2.0,     # V3: stance knee must return to default (KEPT from V3)
    # Stability — V5: orientation=6.0 (Stage 2), height scale=-10 (Stage 2)
    "height": 3.5,                  # exp(-10.0 * height_err) — V5: scale -15→-10 (Stage 2)
    "orientation": 6.0,             # exp(-15.0 * tilt_err) — V5: 5.0→6.0 (Stage 2 value)
    "ang_vel_penalty": 1.0,         # ONLY roll+pitch (z-axis excluded)
    # Posture — V5: foot_flatness reverted, knee_min removed
    "ankle_penalty": 2.0,           # exp(-15.0 * ankle_roll_err)
    "foot_flatness": 1.5,           # V5: 3.0→1.5, scale -15→-8 (Stage 2 value, was 4x too aggressive)
    "symmetry_gait": 1.5,           # Phase-shifted L/R range matching
    "hip_roll_penalty": 1.5,        # exp(-10.0 * hip_roll_err)
    "hip_yaw_penalty": 2.5,         # exp(-12.0 * hip_yaw_err) — KEPT from V2
    "knee_negative_penalty": -8.0,  # Linear, prevents backward bending (knee < 0.1)
    "knee_overbend_penalty": -5.0,  # Linear, prevents deep squat (knee > 1.2 rad)
    "knee_min_penalty": 0.0,        # V5: DISABLED (was -8.0, prevented natural knee bending)
    # Waist and standing — V5: reverted to Stage 2 values
    "waist_posture": 3.0,           # V5: 4.0→3.0, scales: yaw -25, roll -20, pitch -15 (Stage 2)
    "standing_posture": 3.0,        # V5: floor min=0.0 (off during walking, Stage 2 behavior)
    "yaw_rate_penalty": -2.0,       # CONDITIONAL (off when yaw_cmd active)
    # Physics
    "vz_penalty": -2.0,             # Vertical bouncing
    "feet_slip": -0.1,              # Contact foot sliding
    # Gait quality
    "feet_air_time": 1.0,           # Reward proper gait timing
    # Smoothness
    "action_rate": -0.02,
    "jerk": -0.08,
    "energy": -0.0003,
    "alive": 1.0,                   # V5: 0.2→1.0 (Stage 2 value, encourages exploration)
}

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Stage 1: Standing + Omnidirectional Locomotion")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=35000)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--experiment_name", type=str, default="g1_unified_stage1")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sensors import ContactSensorCfg
from torch.utils.tensorboard import SummaryWriter

print("=" * 80)
print("UNIFIED STAGE 1: STANDING + OMNIDIRECTIONAL LOCOMOTION")
print(f"  Decoupled Heading Control: (speed, heading_offset, body_yaw_cmd)")
print(f"  USD: {G1_29DOF_USD}")
print(f"  Unified Obs: {OBS_DIM} (loco={LOCO_OBS_DIM} + arm={ARM_OBS_DIM} + hand={HAND_OBS_DIM})")
print(f"  Unified Act: {ACT_DIM} (loco={LOCO_ACT_DIM} + arm={ARM_ACT_DIM} + hand={HAND_ACT_DIM})")
print(f"  Active in this stage: loco only (15 act)")
print(f"  Joints: {NUM_LOCO_JOINTS} loco + {NUM_ARM_JOINTS} arm + {NUM_HAND_JOINTS} hand = {NUM_LOCO_JOINTS + NUM_ARM_JOINTS + NUM_HAND_JOINTS}")
print(f"  Gait frequency: {GAIT_FREQUENCY} Hz")
print("=" * 80)
for i, lv in enumerate(CURRICULUM):
    print(f"  Level {i}: {lv['description']}")
    print(f"    speed={lv['speed']}, heading={lv['heading_offset']}, yaw={lv['yaw_cmd']}, "
          f"yaw_mode={lv['yaw_mode']}, height={lv['height']}, push={lv['push_force']}")

# ============================================================================
# NETWORK — Triple Actor-Critic (only LocoAC trained here)
# ============================================================================

class LocoActorCritic(nn.Module):
    """Locomotion Actor-Critic — [512, 256, 128] + LayerNorm + ELU"""
    def __init__(self, num_obs=OBS_DIM, num_act=LOCO_ACT_DIM, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def act(self, x, det=False):
        mean = self.actor(x)
        if det:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, x, a):
        mean = self.actor(x)
        val = self.critic(x).squeeze(-1)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return val, dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


# ============================================================================
# PPO
# ============================================================================

class PPO:
    def __init__(self, net, device, lr=3e-4, max_iter=35000):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, max_iter, 1e-5)

    def gae(self, r, v, d, nv):
        adv = torch.zeros_like(r)
        last = 0
        for t in reversed(range(len(r))):
            nxt = nv if t == len(r) - 1 else v[t + 1]
            delta = r[t] + 0.99 * nxt * (1 - d[t]) - v[t]
            adv[t] = last = delta + 0.99 * 0.95 * (1 - d[t]) * last
        return adv, adv + v

    def update(self, obs, act, old_lp, ret, adv, old_v):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tot_a, tot_c, tot_e, n = 0, 0, 0, 0
        bs = obs.shape[0]

        for _ in range(5):
            idx = torch.randperm(bs, device=self.device)
            for i in range(0, bs, 4096):
                mb = idx[i:i + 4096]
                val, lp, ent = self.net.evaluate(obs[mb], act[mb])
                ratio = (lp - old_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(0.8, 1.2) * adv[mb]
                a_loss = -torch.min(s1, s2).mean()
                v_clip = old_v[mb] + (val - old_v[mb]).clamp(-0.2, 0.2)
                c_loss = 0.5 * torch.max((val - ret[mb]) ** 2, (v_clip - ret[mb]) ** 2).mean()
                loss = a_loss + 0.5 * c_loss - 0.01 * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                n += 1

        self.sched.step()
        return {"a": tot_a / n, "c": tot_c / n, "e": tot_e / n, "lr": self.sched.get_last_lr()[0]}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quat_to_euler_xyz(quat):
    """Convert wxyz quaternion (Isaac Lab convention) to roll, pitch, yaw.
    Isaac Lab root_quat_w returns (w, x, y, z) — confirmed in articulation_data.py line 455.
    V2 fix: previously treated col 0 as x, now correctly treats col 0 as w.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


def wrap_to_pi(angles):
    """Wrap angles to [-pi, pi] range."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# ENVIRONMENT
# ============================================================================

def create_env(num_envs, device):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_29DOF_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,   # V2: enabled — prevents leg crossing (was False)
                    solver_position_iteration_count=8,   # V2: increased from 4 for self-collision stability
                    solver_velocity_iteration_count=2,    # V2: increased from 1 for self-collision stability
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.80),
                joint_pos=DEFAULT_ALL_POSES,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.90,
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[
                        ".*_hip_yaw_joint", ".*_hip_roll_joint",
                        ".*_hip_pitch_joint", ".*_knee_joint",
                        ".*waist.*",
                    ],
                    effort_limit_sim=ACTUATOR_PARAMS["legs"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["legs"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["legs"]["stiffness"],
                    damping=ACTUATOR_PARAMS["legs"]["damping"],
                    armature=ACTUATOR_PARAMS["legs"]["armature"],
                ),
                "feet": ImplicitActuatorCfg(
                    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["feet"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["feet"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["feet"]["stiffness"],
                    damping=ACTUATOR_PARAMS["feet"]["damping"],
                    armature=ACTUATOR_PARAMS["feet"]["armature"],
                ),
                "shoulders": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["shoulders"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["shoulders"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["shoulders"]["stiffness"],
                    damping=ACTUATOR_PARAMS["shoulders"]["damping"],
                    armature=ACTUATOR_PARAMS["shoulders"]["armature"],
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["arms"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["arms"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["arms"]["stiffness"],
                    damping=ACTUATOR_PARAMS["arms"]["damping"],
                    armature=ACTUATOR_PARAMS["arms"]["armature"],
                ),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=[".*_wrist_.*"],
                    effort_limit_sim=ACTUATOR_PARAMS["wrist"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["wrist"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["wrist"]["stiffness"],
                    damping=ACTUATOR_PARAMS["wrist"]["damping"],
                    armature=ACTUATOR_PARAMS["wrist"]["armature"],
                ),
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=[
                        ".*_hand_index_.*_joint",
                        ".*_hand_middle_.*_joint",
                        ".*_hand_thumb_.*_joint",
                    ],
                    effort_limit=ACTUATOR_PARAMS["hands"]["effort_limit"],
                    velocity_limit=ACTUATOR_PARAMS["hands"]["velocity_limit"],
                    stiffness={".*": ACTUATOR_PARAMS["hands"]["stiffness"]},
                    damping={".*": ACTUATOR_PARAMS["hands"]["damping"]},
                    armature={".*": ACTUATOR_PARAMS["hands"]["armature"]},
                ),
            },
        )
        # Foot contact sensors — one per foot (prim_path must be single body)
        left_foot_contact = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/left_ankle_roll_link",
            update_period=0.02,
            history_length=1,
            track_air_time=False,
            force_threshold=5.0,
        )
        right_foot_contact = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ankle_roll_link",
            update_period=0.02,
            history_length=1,
            track_air_time=False,
            force_threshold=5.0,
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 20.0
        action_space = LOCO_ACT_DIM  # Only loco actions go through env.step()
        observation_space = OBS_DIM   # Full 188-dim unified obs
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=4.0)

    class Env(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)

            # ============================================================
            # Joint index resolution
            # ============================================================
            jn = self.robot.joint_names
            print(f"\n[Env] Robot joint names ({len(jn)} total): {jn}")

            # Loco
            self.loco_idx = []
            for name in LOCO_JOINT_NAMES:
                if name in jn:
                    self.loco_idx.append(jn.index(name))
                else:
                    print(f"  [WARN] Loco joint not found: {name}")
            self.loco_idx = torch.tensor(self.loco_idx, device=self.device)
            print(f"  Loco joints: {len(self.loco_idx)} / {NUM_LOCO_JOINTS}")

            # Arm
            self.arm_idx = []
            for name in ARM_JOINT_NAMES:
                if name in jn:
                    self.arm_idx.append(jn.index(name))
            self.arm_idx = torch.tensor(self.arm_idx, device=self.device)
            print(f"  Arm joints: {len(self.arm_idx)} / {NUM_ARM_JOINTS}")

            # Hand
            self.hand_idx = []
            for name in HAND_JOINT_NAMES:
                if name in jn:
                    self.hand_idx.append(jn.index(name))
            self.hand_idx = torch.tensor(self.hand_idx, device=self.device)
            print(f"  Hand joints: {len(self.hand_idx)} / {NUM_HAND_JOINTS}")

            # ============================================================
            # Per-joint indices within LOCO_JOINT_NAMES for reward compute
            # ============================================================
            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")
            self.left_hip_pitch_idx = LOCO_JOINT_NAMES.index("left_hip_pitch_joint")
            self.right_hip_pitch_idx = LOCO_JOINT_NAMES.index("right_hip_pitch_joint")

            ankle_roll_names = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
            self.ankle_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_roll_names], device=self.device)

            ankle_pitch_names = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
            self.ankle_pitch_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_pitch_names], device=self.device)

            hip_roll_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
            self.hip_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_roll_names], device=self.device)

            # V2: hip_yaw indices for leg rotation penalty
            hip_yaw_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
            self.hip_yaw_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_yaw_names], device=self.device)

            # Symmetry pairs
            sym_pairs = [
                ("left_hip_pitch_joint", "right_hip_pitch_joint"),
                ("left_hip_roll_joint", "right_hip_roll_joint"),
                ("left_hip_yaw_joint", "right_hip_yaw_joint"),
                ("left_knee_joint", "right_knee_joint"),
                ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
                ("left_ankle_roll_joint", "right_ankle_roll_joint"),
            ]
            self.sym_left_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(l) for l, r in sym_pairs], device=self.device)
            self.sym_right_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(r) for l, r in sym_pairs], device=self.device)

            # ============================================================
            # Default poses as tensors
            # ============================================================
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)
            self.default_knee = DEFAULT_LOCO_LIST[self.left_knee_idx]

            # Action scales (loco only for this stage)
            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # ============================================================
            # State variables — decoupled heading commands
            # ============================================================
            self.curr_level = 0
            self.curr_hist = []

            # Commands
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.speed_cmd = torch.zeros(self.num_envs, device=self.device)
            self.heading_offset = torch.zeros(self.num_envs, device=self.device)
            self.body_yaw_cmd = torch.zeros(self.num_envs, device=self.device)
            self.yaw_mode = torch.zeros(self.num_envs, device=self.device)  # 0=rate, 1=absolute
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

            # Gait and actions
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, LOCO_ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, LOCO_ACT_DIM, device=self.device)
            self._prev_jvel = None

            # Push timer
            self.push_timer = torch.full((self.num_envs,), 9999, device=self.device, dtype=torch.long)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # Command resample — random interval per-env
            self.cmd_resample_lo = 150
            self.cmd_resample_hi = 400
            self.cmd_timer = torch.randint(0, self.cmd_resample_hi,
                                           (self.num_envs,), device=self.device)
            self.cmd_resample_targets = torch.randint(
                self.cmd_resample_lo, self.cmd_resample_hi + 1,
                (self.num_envs,), device=self.device)

            # Foot contact sensor refs (one per foot)
            self._left_foot_sensor = self.scene["left_foot_contact"]
            self._right_foot_sensor = self.scene["right_foot_contact"]

            # Sample initial commands
            self._sample_commands(torch.arange(self.num_envs, device=self.device))

            print(f"\n[Env] {self.num_envs} envs, level {self.curr_level}")
            print(f"  Obs: {OBS_DIM} (loco={LOCO_OBS_DIM}+arm={ARM_OBS_DIM}+hand={HAND_OBS_DIM})")
            print(f"  Act: {LOCO_ACT_DIM} (loco only, arm/hand inactive)")
            print(f"  Resample interval: {self.cmd_resample_lo}-{self.cmd_resample_hi} steps")

        @property
        def robot(self):
            return self.scene["robot"]

        # ============================================================
        # Command sampling — decoupled heading
        # ============================================================
        def _sample_commands(self, env_ids):
            """Sample commands from current curriculum level."""
            lv = CURRICULUM[self.curr_level]
            n = len(env_ids)

            # Speed
            sp_lo, sp_hi = lv["speed"]
            self.speed_cmd[env_ids] = torch.rand(n, device=self.device) * (sp_hi - sp_lo) + sp_lo

            # Heading offset
            ho_lo, ho_hi = lv["heading_offset"]
            self.heading_offset[env_ids] = torch.rand(n, device=self.device) * (ho_hi - ho_lo) + ho_lo

            # Body yaw command
            yc_lo, yc_hi = lv["yaw_cmd"]
            self.body_yaw_cmd[env_ids] = torch.rand(n, device=self.device) * (yc_hi - yc_lo) + yc_lo

            # Yaw mode
            ym = lv["yaw_mode"]
            if ym == "rate":
                self.yaw_mode[env_ids] = 0.0
            elif ym == "absolute":
                self.yaw_mode[env_ids] = 1.0
            elif ym == "mixed":
                # 50% rate, 50% absolute
                self.yaw_mode[env_ids] = (torch.rand(n, device=self.device) > 0.5).float()

            # Height
            h_lo, h_hi = lv["height"]
            if h_lo == h_hi:
                self.height_cmd[env_ids] = h_lo
            else:
                self.height_cmd[env_ids] = torch.rand(n, device=self.device) * (h_hi - h_lo) + h_lo

            # 15% standing samples — all commands zero
            standing_mask = torch.rand(n, device=self.device) < 0.15
            if standing_mask.any():
                s_ids = env_ids[standing_mask]
                self.speed_cmd[s_ids] = 0.0
                self.heading_offset[s_ids] = 0.0
                self.body_yaw_cmd[s_ids] = 0.0
                self.yaw_mode[s_ids] = 0.0  # rate mode for standing

        # ============================================================
        # Curriculum
        # ============================================================
        def update_curriculum(self, r):
            """Multi-axis curriculum gating."""
            self.curr_hist.append(r)
            if len(self.curr_hist) >= 100:
                avg = np.mean(self.curr_hist[-100:])
                thr = CURRICULUM[self.curr_level]["threshold"]

                if thr is not None and avg > thr and self.curr_level < len(CURRICULUM) - 1:
                    # Check tracking quality
                    lv_b = quat_apply_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)
                    av_b = quat_apply_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)
                    torso_euler = quat_to_euler_xyz(self.robot.data.root_quat_w)

                    # Speed tracking
                    actual_speed = torch.sqrt(lv_b[:, 0]**2 + lv_b[:, 1]**2).mean().item()
                    cmd_speed = self.speed_cmd.mean().item()
                    speed_err = abs(actual_speed - cmd_speed) / max(cmd_speed, 0.3)  # denominator 0.1→0.3 to soften standing gate
                    speed_ok = speed_err < 0.8  # threshold 0.5→0.8 to allow slightly imperfect standing

                    # Heading tracking — per-env speed filter (match reward logic)
                    # V4: Skip heading gate for L0-L2 (heading_offset=0, yaw rotation makes
                    # heading tracking impossible). L3+ has heading_offset non-zero.
                    heading_ok = True
                    if self.curr_level >= 3:
                        moving_mask = self.speed_cmd.squeeze() > 0.05  # per-env filter
                        if moving_mask.any():
                            actual_heading = torch.atan2(lv_b[:, 1], lv_b[:, 0])
                            heading_err_all = torch.abs(wrap_to_pi(actual_heading - self.heading_offset))
                            heading_err = heading_err_all[moving_mask].mean().item()
                            heading_ok = heading_err < 0.5

                    # Yaw tracking
                    yaw_ok = True
                    yaw_cmd_mag = self.body_yaw_cmd.abs().mean().item()
                    if yaw_cmd_mag > 0.1:
                        if self.yaw_mode.mean().item() < 0.5:  # rate mode
                            yaw_err = (av_b[:, 2] - self.body_yaw_cmd).abs().mean().item()
                        else:  # absolute mode
                            yaw_err = torch.abs(wrap_to_pi(torso_euler[:, 2] - self.body_yaw_cmd)).mean().item()
                        yaw_ok = yaw_err < 0.3

                    if speed_ok and heading_ok and yaw_ok:
                        self.curr_level += 1
                        lv = CURRICULUM[self.curr_level]
                        print(f"\n*** LEVEL UP! Now {self.curr_level}: {lv['description']} ***")
                        print(f"    speed={lv['speed']}, heading={lv['heading_offset']}, "
                              f"yaw={lv['yaw_cmd']}, yaw_mode={lv['yaw_mode']}")
                        print(f"    height={lv['height']}, push={lv['push_force']}, mass={lv['mass_scale']}")
                        print(f"    (speed_err={speed_err:.2f}, reward={avg:.1f})")
                        self.curr_hist = []
                        self._sample_commands(torch.arange(self.num_envs, device=self.device))
                    else:
                        if len(self.curr_hist) % 200 == 0:
                            reasons = []
                            if not speed_ok:
                                reasons.append(f"speed_err={speed_err:.2f}>0.5")
                            if not heading_ok:
                                reasons.append(f"heading_err={heading_err:.2f}>0.5")
                            if not yaw_ok:
                                reasons.append("yaw_err>0.3")
                            print(f"  [Gate] Level {self.curr_level} blocked: {', '.join(reasons)} (R={avg:.1f})")

        # ============================================================
        # Push and command resample
        # ============================================================
        def _apply_push(self):
            lv = CURRICULUM[self.curr_level]
            self.step_count += 1
            push_mask = self.step_count >= self.push_timer

            # Command resample
            self.cmd_timer += 1
            cmd_mask = self.cmd_timer >= self.cmd_resample_targets
            if cmd_mask.any():
                cmd_ids = cmd_mask.nonzero(as_tuple=False).squeeze(-1)
                self._sample_commands(cmd_ids)
                self.cmd_timer[cmd_ids] = 0
                self.cmd_resample_targets[cmd_ids] = torch.randint(
                    self.cmd_resample_lo, self.cmd_resample_hi + 1,
                    (len(cmd_ids),), device=self.device)

            # Push forces
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                force = torch.zeros(n, 3, device=self.device)
                force[:, :2] = torch.randn(n, 2, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                fmin, fmax = lv["push_force"]
                mag = torch.rand(n, 1, device=self.device) * (fmax - fmin) + fmin
                force = force * mag
                forces[ids, 0] = force

                pi_lo, pi_hi = lv["push_interval"]
                if pi_hi <= pi_lo:
                    pi_hi = pi_lo + 1
                self.push_timer[ids] = torch.randint(pi_lo, pi_hi, (n,), device=self.device)
                self.step_count[ids] = 0

            self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        # ============================================================
        # Physics step
        # ============================================================
        def _pre_physics_step(self, act):
            self.actions = act.clone()

            # Apply loco actions
            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.loco_idx] = self.default_loco + act * self.action_scales

            # V2: Clamp hip_yaw to prevent extreme leg rotation
            left_hip_yaw_idx = self.loco_idx[LOCO_JOINT_NAMES.index("left_hip_yaw_joint")]
            right_hip_yaw_idx = self.loco_idx[LOCO_JOINT_NAMES.index("right_hip_yaw_joint")]
            tgt[:, left_hip_yaw_idx].clamp_(-0.25, 0.25)    # ~14 degrees max rotation
            tgt[:, right_hip_yaw_idx].clamp_(-0.25, 0.25)

            # Clamp waist
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)

            # Hold arms and hands at default
            tgt[:, self.arm_idx] = self.default_arm
            tgt[:, self.hand_idx] = self.default_hand

            self.robot.set_joint_position_target(tgt)

            # Update state
            self._prev_act = self.prev_act.clone()
            self.prev_act = act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            self._apply_push()

        def _apply_action(self):
            pass

        # ============================================================
        # Observations — 188 dim unified space
        # ============================================================
        def _get_observations(self):
            r = self.robot
            q = r.data.root_quat_w
            lv = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            # Leg joints
            jp_leg = r.data.joint_pos[:, self.loco_idx[:12]]
            jv_leg = r.data.joint_vel[:, self.loco_idx[:12]] * 0.1

            # Waist joints
            jp_waist = r.data.joint_pos[:, self.loco_idx[12:15]]
            jv_waist = r.data.joint_vel[:, self.loco_idx[12:15]] * 0.1

            # Gait phase
            gait = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)
            ], -1)

            # Torso euler
            torso_euler = quat_to_euler_xyz(q)

            # Foot contact from sensors (one per foot → concat)
            lf = self._left_foot_sensor.data.net_forces_w   # [N, 1, 3]
            rf = self._right_foot_sensor.data.net_forces_w  # [N, 1, 3]
            foot_contact = torch.cat([
                (lf.norm(dim=-1) > 5.0).float(),            # [N, 1]
                (rf.norm(dim=-1) > 5.0).float(),            # [N, 1]
            ], dim=-1)                                       # [N, 2]

            # Height obs
            current_height = r.data.root_pos_w[:, 2:3]           # [N, 1]
            height_err = self.height_cmd[:, None] - current_height  # [N, 1]
            root_height_vel = lv[:, 2:3]                           # [N, 1] vertical vel

            # Payload estimate (0 for this stage — filled by loco-manip stages)
            payload_est = torch.zeros(self.num_envs, 1, device=self.device)

            # === Loco obs (73 dim) ===
            loco_obs = torch.cat([
                lv,                                     # 3  [0:3]
                av,                                     # 3  [3:6]
                g,                                      # 3  [6:9]
                jp_leg,                                 # 12 [9:21]
                jv_leg,                                 # 12 [21:33]
                jp_waist,                               # 3  [33:36]
                jv_waist,                               # 3  [36:39]
                self.height_cmd[:, None],               # 1  [39:40]
                self.speed_cmd[:, None],                # 1  [40:41]
                self.heading_offset[:, None],           # 1  [41:42]
                self.body_yaw_cmd[:, None],             # 1  [42:43]
                self.yaw_mode[:, None],                 # 1  [43:44]
                gait,                                   # 2  [44:46]
                self.prev_act,                          # 15 [46:61]
                torso_euler,                            # 3  [61:64]
                self.torso_cmd,                         # 3  [64:67]
                foot_contact,                           # 2  [67:69]
                current_height,                         # 1  [69:70]
                height_err,                             # 1  [70:71]
                root_height_vel,                        # 1  [71:72]
                payload_est,                            # 1  [72:73]
            ], dim=-1)  # = 73

            # === Arm obs (70 dim) — zeros in this stage ===
            arm_obs = torch.zeros(self.num_envs, ARM_OBS_DIM, device=self.device)

            # === Hand obs (45 dim) — zeros in this stage ===
            hand_obs = torch.zeros(self.num_envs, HAND_OBS_DIM, device=self.device)

            # === Concatenate ===
            obs = torch.cat([loco_obs, arm_obs, hand_obs], dim=-1)  # 188

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        # ============================================================
        # Rewards
        # ============================================================
        def _get_rewards(self):
            r = self.robot
            q = r.data.root_quat_w
            pos = r.data.root_pos_w
            lv_b = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av_b = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))
            jp = r.data.joint_pos[:, self.loco_idx]
            jv = r.data.joint_vel[:, self.loco_idx]
            torso_euler = quat_to_euler_xyz(q)

            # ===========================================
            # SPEED TRACKING
            # ===========================================
            actual_speed = torch.sqrt(lv_b[:, 0]**2 + lv_b[:, 1]**2)
            speed_err = (actual_speed - self.speed_cmd) ** 2
            r_speed = torch.exp(-3.0 * speed_err)  # V5: -6.0→-3.0 (Stage 2 value)

            # ===========================================
            # HEADING TRACKING
            # ===========================================
            actual_heading = torch.atan2(lv_b[:, 1], lv_b[:, 0])
            heading_err_raw = wrap_to_pi(actual_heading - self.heading_offset)
            heading_err = heading_err_raw ** 2
            r_heading_raw = torch.exp(-2.0 * heading_err)
            # Heading only matters when moving (speed > 0.05)
            heading_active = (self.speed_cmd > 0.05).float()
            r_heading = r_heading_raw * heading_active + (1.0 - heading_active)

            # ===========================================
            # YAW TRACKING (dual mode)
            # ===========================================
            # Rate mode: track yaw rate
            yaw_rate_err = (av_b[:, 2] - self.body_yaw_cmd) ** 2
            r_yaw_rate = torch.exp(-3.0 * yaw_rate_err)

            # Absolute mode: track yaw angle
            current_yaw = torso_euler[:, 2]
            yaw_angle_err = wrap_to_pi(current_yaw - self.body_yaw_cmd) ** 2
            r_yaw_abs = torch.exp(-3.0 * yaw_angle_err)

            # Select based on yaw_mode (0=rate, 1=absolute)
            r_yaw = (1.0 - self.yaw_mode) * r_yaw_rate + self.yaw_mode * r_yaw_abs

            # ===========================================
            # GAIT CONTROL
            # ===========================================
            ph = self.phase
            l_swing = (ph < 0.5).float()
            r_swing = (ph >= 0.5).float()

            # Knee alternation
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_swing_target = 0.65
            knee_stance_target = self.default_knee

            left_knee_target = l_swing * knee_swing_target + (1 - l_swing) * knee_stance_target
            right_knee_target = r_swing * knee_swing_target + (1 - r_swing) * knee_stance_target
            knee_err = (lk - left_knee_target) ** 2 + (rk - right_knee_target) ** 2
            r_gait_knee = torch.exp(-5.0 * knee_err)

            # Knee negative penalty
            knee_min = 0.1
            lk_violation = torch.clamp(knee_min - lk, min=0.0)
            rk_violation = torch.clamp(knee_min - rk, min=0.0)
            r_knee_neg_penalty = lk_violation + rk_violation

            # Knee overbend penalty — prevents deep squat (knee > 1.2 rad)
            knee_max = 1.2
            lk_overbend = torch.clamp(lk - knee_max, min=0.0)
            rk_overbend = torch.clamp(rk - knee_max, min=0.0)
            r_knee_overbend = lk_overbend + rk_overbend

            # V4: Knee minimum angle penalty — prevents squat-like posture (knee < 0.7 rad)
            knee_min_walk = 0.7
            lk_min_violation = torch.clamp(knee_min_walk - lk, min=0.0)
            rk_min_violation = torch.clamp(knee_min_walk - rk, min=0.0)
            r_knee_min_penalty = lk_min_violation + rk_min_violation

            # Foot clearance
            lh = jp[:, self.left_hip_pitch_idx]
            rh = jp[:, self.right_hip_pitch_idx]
            hip_swing_target = -0.35
            hip_stance_target = self.default_loco[self.left_hip_pitch_idx].item()

            left_hip_target = l_swing * hip_swing_target + (1 - l_swing) * hip_stance_target
            right_hip_target = r_swing * hip_swing_target + (1 - r_swing) * hip_stance_target
            hip_err = (lh - left_hip_target) ** 2 + (rh - right_hip_target) ** 2
            r_gait_clearance = torch.exp(-4.0 * hip_err)

            # Foot contact pattern
            lk_vel = jv[:, self.left_knee_idx].abs()
            rk_vel = jv[:, self.right_knee_idx].abs()
            swing_activity = l_swing * lk_vel + r_swing * rk_vel
            stance_stability = (1 - l_swing) * lk_vel + (1 - r_swing) * rk_vel
            r_gait_contact = torch.tanh(swing_activity * 0.5) * torch.exp(-2.0 * stance_stability)

            # Gait scale — gait rewards proportional to speed
            gait_scale = torch.clamp(self.speed_cmd / 0.2, 0.0, 1.0)

            r_gait_knee = r_gait_knee * gait_scale + (1 - gait_scale) * 1.0
            r_gait_clearance = r_gait_clearance * gait_scale + (1 - gait_scale) * 1.0
            r_gait_contact = r_gait_contact * gait_scale

            # V3: Stance knee must return to default (not stay bent)
            left_stance_knee_err = (1.0 - l_swing) * (lk - knee_stance_target) ** 2
            right_stance_knee_err = (1.0 - r_swing) * (rk - knee_stance_target) ** 2
            r_gait_stance_posture = torch.exp(-8.0 * (left_stance_knee_err + right_stance_knee_err))
            r_gait_stance_posture = r_gait_stance_posture * gait_scale + (1.0 - gait_scale) * 1.0

            # ===========================================
            # STABILITY
            # ===========================================
            height = pos[:, 2]
            h_err = (height - self.height_cmd).abs()
            r_height = torch.exp(-10.0 * h_err)  # V5: -15→-10 (Stage 2 value)

            r_orient = torch.exp(-15.0 * (g[:, :2] ** 2).sum(-1))  # V3: -12→-15, weight 4→5 (Stage 2 level)

            # Angular velocity penalty — ONLY roll + pitch (z-axis excluded)
            r_ang = torch.exp(-1.0 * (av_b[:, :2] ** 2).sum(-1))

            # ===========================================
            # POSTURE
            # ===========================================
            ankle_roll_err = (jp[:, self.ankle_roll_loco_idx] - self.default_loco[self.ankle_roll_loco_idx]) ** 2
            r_ankle = torch.exp(-15.0 * ankle_roll_err.sum(-1))

            ankle_pitch_err = (jp[:, self.ankle_pitch_loco_idx] - self.default_loco[self.ankle_pitch_loco_idx]) ** 2
            r_foot_flat = torch.exp(-8.0 * ankle_pitch_err.sum(-1))  # V5: -15→-8 (Stage 2 value, -15 was too aggressive)

            left_pos = jp[:, self.sym_left_idx]
            right_pos = jp[:, self.sym_right_idx]
            left_dev = (left_pos - self.default_loco[self.sym_left_idx]).abs()
            right_dev = (right_pos - self.default_loco[self.sym_right_idx]).abs()
            sym_range_err = (left_dev - right_dev) ** 2
            r_sym_gait = torch.exp(-3.0 * sym_range_err.sum(-1))

            hip_roll_err = (jp[:, self.hip_roll_loco_idx] - self.default_loco[self.hip_roll_loco_idx]) ** 2
            r_hip_roll = torch.exp(-10.0 * hip_roll_err.sum(-1))

            # V2: Hip yaw penalty — prevents legs from rotating/crossing
            hip_yaw_err = (jp[:, self.hip_yaw_loco_idx] - self.default_loco[self.hip_yaw_loco_idx]) ** 2
            r_hip_yaw = torch.exp(-12.0 * hip_yaw_err.sum(-1))

            # Waist posture — V5: reverted to Stage 2 scales (-25/-20/-15)
            waist_yaw_err = (jp[:, 12] - self.default_loco[12]) ** 2
            waist_roll_err = (jp[:, 13] - self.default_loco[13]) ** 2
            waist_pitch_err = (jp[:, 14] - self.default_loco[14]) ** 2
            r_waist_posture = (torch.exp(-25.0 * waist_yaw_err)
                             * torch.exp(-20.0 * waist_roll_err)
                             * torch.exp(-15.0 * waist_pitch_err))

            # Standing posture — V5: floor removed (Stage 2: fully off during walking)
            standing_scale = torch.clamp(1.0 - gait_scale, min=0.0)  # V5: min 0.3→0.0 (Stage 2 behavior)
            leg_pos_err = (jp[:, :12] - self.default_loco[:12]) ** 2
            r_standing_posture_raw = torch.exp(-3.0 * leg_pos_err.sum(-1))
            r_standing_posture = standing_scale * r_standing_posture_raw + (1 - standing_scale) * 1.0

            # ===========================================
            # PHYSICS PENALTIES
            # ===========================================
            vz = lv_b[:, 2]
            r_vz_penalty = vz ** 2

            # Conditional yaw_rate_penalty
            yaw_rate = av_b[:, 2]
            yaw_cmd_mag = self.body_yaw_cmd.abs()
            yaw_penalty_scale = torch.clamp(1.0 - yaw_cmd_mag / 0.3, 0.0, 1.0)
            r_yaw_rate_penalty = yaw_rate ** 2 * yaw_penalty_scale

            # Feet slip
            ankle_pitch_vel = jv[:, self.ankle_pitch_loco_idx].abs()
            ankle_roll_vel = jv[:, self.ankle_roll_loco_idx].abs()
            left_stance = 1.0 - l_swing
            right_stance = 1.0 - r_swing
            left_slip = left_stance * (ankle_pitch_vel[:, 0] + ankle_roll_vel[:, 0])
            right_slip = right_stance * (ankle_pitch_vel[:, 1] + ankle_roll_vel[:, 1])
            r_feet_slip = left_slip + right_slip

            # V2: Feet air time reward — encourages proper foot contact cycle
            # Uses ContactSensor: reward feet that land during stance and lift during swing
            lf_contact = (self._left_foot_sensor.data.net_forces_w.norm(dim=-1).squeeze(-1) > 5.0).float()
            rf_contact = (self._right_foot_sensor.data.net_forces_w.norm(dim=-1).squeeze(-1) > 5.0).float()
            # Stance foot should be on ground, swing foot should be in air
            left_contact_correct = left_stance * lf_contact + l_swing * (1.0 - lf_contact)
            right_contact_correct = right_stance * rf_contact + r_swing * (1.0 - rf_contact)
            r_feet_air_time = (left_contact_correct + right_contact_correct) * 0.5
            # Only active during walking (scale with gait_scale)
            r_feet_air_time = r_feet_air_time * gait_scale + (1.0 - gait_scale) * 1.0

            # ===========================================
            # SMOOTHNESS
            # ===========================================
            dact = self.prev_act - self._prev_act
            r_action_rate = (dact ** 2).sum(-1)

            if self._prev_jvel is None:
                self._prev_jvel = jv.clone()
            jerk = ((jv - self._prev_jvel) ** 2).sum(-1)
            self._prev_jvel = jv.clone()

            r_energy = (jv.abs() * jp.abs()).sum(-1)

            # ===========================================
            # TOTAL REWARD
            # ===========================================
            reward = (
                REWARD_WEIGHTS["speed"] * r_speed
                + REWARD_WEIGHTS["heading"] * r_heading
                + REWARD_WEIGHTS["yaw"] * r_yaw
                + REWARD_WEIGHTS["gait_knee"] * r_gait_knee
                + REWARD_WEIGHTS["gait_clearance"] * r_gait_clearance
                + REWARD_WEIGHTS["gait_contact"] * r_gait_contact
                + REWARD_WEIGHTS["gait_stance_posture"] * r_gait_stance_posture  # V3: stance knee default
                + REWARD_WEIGHTS["height"] * r_height
                + REWARD_WEIGHTS["orientation"] * r_orient
                + REWARD_WEIGHTS["ang_vel_penalty"] * r_ang
                + REWARD_WEIGHTS["ankle_penalty"] * r_ankle
                + REWARD_WEIGHTS["foot_flatness"] * r_foot_flat
                + REWARD_WEIGHTS["symmetry_gait"] * r_sym_gait
                + REWARD_WEIGHTS["hip_roll_penalty"] * r_hip_roll
                + REWARD_WEIGHTS["hip_yaw_penalty"] * r_hip_yaw
                + REWARD_WEIGHTS["knee_negative_penalty"] * r_knee_neg_penalty
                + REWARD_WEIGHTS["knee_overbend_penalty"] * r_knee_overbend
                + REWARD_WEIGHTS["knee_min_penalty"] * r_knee_min_penalty        # V3: knee min angle
                + REWARD_WEIGHTS["waist_posture"] * r_waist_posture
                + REWARD_WEIGHTS["standing_posture"] * r_standing_posture
                + REWARD_WEIGHTS["yaw_rate_penalty"] * r_yaw_rate_penalty
                + REWARD_WEIGHTS["vz_penalty"] * r_vz_penalty
                + REWARD_WEIGHTS["feet_slip"] * r_feet_slip
                + REWARD_WEIGHTS["feet_air_time"] * r_feet_air_time
                + REWARD_WEIGHTS["action_rate"] * r_action_rate
                + REWARD_WEIGHTS["jerk"] * jerk
                + REWARD_WEIGHTS["energy"] * r_energy
                + REWARD_WEIGHTS["alive"]
            )
            return reward

        # ============================================================
        # Termination
        # ============================================================
        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gravity_vec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(q, gravity_vec)

            fallen = (pos[:, 2] < 0.35) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_hyperextended = (lk < -0.05) | (rk < -0.05) | (lk > 2.0) | (rk > 2.0)
            # V5: knee_collapse REMOVED (prevented natural gait)

            waist_pitch_val = jp[:, 14]
            waist_roll_val = jp[:, 13]
            waist_excessive = (waist_pitch_val.abs() > 0.35) | (waist_roll_val.abs() > 0.25)  # V5: Stage 2 values (0.20/0.10 was too tight)

            terminated = fallen | bad_orientation | knee_hyperextended | waist_excessive
            time_out = self.episode_length_buf >= self.max_episode_length
            return terminated, time_out

        # ============================================================
        # Reset
        # ============================================================
        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0:
                return
            n = len(env_ids)

            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            default_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
            noise = torch.randn_like(default_pos) * 0.02
            self.robot.write_joint_state_to_sim(default_pos + noise, default_vel, None, env_ids)

            root_pos = torch.tensor([[0.0, 0.0, HEIGHT_DEFAULT]], device=self.device).expand(n, -1).clone()
            root_pos[:, :2] += torch.randn(n, 2, device=self.device) * 0.05
            yaw = torch.randn(n, device=self.device) * 0.1
            # V2 fix: Isaac Lab expects wxyz quaternion order
            qw = torch.cos(yaw / 2)
            qx = torch.zeros(n, device=self.device)
            qy = torch.zeros(n, device=self.device)
            qz = torch.sin(yaw / 2)
            root_quat = torch.stack([qw, qx, qy, qz], dim=-1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)

            lv = CURRICULUM[self.curr_level]
            pi_lo, pi_hi = lv["push_interval"]
            if pi_hi <= pi_lo:
                pi_hi = pi_lo + 1
            self.push_timer[env_ids] = torch.randint(pi_lo, pi_hi, (n,), device=self.device)
            self.step_count[env_ids] = 0

            self._sample_commands(env_ids)
            self.cmd_timer[env_ids] = torch.randint(0, self.cmd_resample_hi, (n,), device=self.device)
            self.cmd_resample_targets[env_ids] = torch.randint(
                self.cmd_resample_lo, self.cmd_resample_hi + 1, (n,), device=self.device)

    return Env(EnvCfg())


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    net = LocoActorCritic(OBS_DIM, LOCO_ACT_DIM).to(device)
    ppo = PPO(net, device, lr=3e-4, max_iter=args_cli.max_iterations)

    start_iter = 0
    best_reward = -1e10

    if args_cli.checkpoint:
        print(f"\n[Load] Resuming from {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        if "iteration" in ckpt:
            start_iter = ckpt["iteration"] + 1
        if "best_reward" in ckpt:
            best_reward = ckpt["best_reward"]
        if "curriculum_level" in ckpt:
            env.curr_level = min(ckpt["curriculum_level"], len(CURRICULUM) - 1)
        # Restore optimizer and scheduler state (prevents LR reset on resume)
        if "optimizer" in ckpt:
            ppo.opt.load_state_dict(ckpt["optimizer"])
            print(f"  Optimizer state restored (AdamW momentum preserved)")
        else:
            print(f"  [WARN] No optimizer state in checkpoint — LR will restart from 3e-4")
        if "scheduler" in ckpt:
            ppo.sched.load_state_dict(ckpt["scheduler"])
            print(f"  Scheduler state restored (LR={ppo.sched.get_last_lr()[0]:.2e})")
        else:
            # Approximate: step scheduler to match resumed iteration
            for _ in range(start_iter):
                ppo.sched.step()
            print(f"  [WARN] No scheduler state — stepped {start_iter}x to approximate LR={ppo.sched.get_last_lr()[0]:.2e}")
        print(f"  Resumed at iter {start_iter}, best_reward={best_reward:.2f}, level={env.curr_level}")
    else:
        print("\n[Train] Sifirdan egitim baslatiliyor (unified obs: 188 dim)")

    # Logging
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\n[Log] {log_dir}")

    # Rollout buffer
    T = 24
    obs_buf = torch.zeros(T, env.num_envs, OBS_DIM, device=device)
    act_buf = torch.zeros(T, env.num_envs, LOCO_ACT_DIM, device=device)
    rew_buf = torch.zeros(T, env.num_envs, device=device)
    done_buf = torch.zeros(T, env.num_envs, device=device)
    val_buf = torch.zeros(T, env.num_envs, device=device)
    lp_buf = torch.zeros(T, env.num_envs, device=device)

    obs, _ = env.reset()
    obs_t = obs["policy"]
    ep_rewards = torch.zeros(env.num_envs, device=device)
    ep_lengths = torch.zeros(env.num_envs, device=device)
    completed_rewards = []

    print(f"\n[Train] Starting from iter {start_iter}, max {args_cli.max_iterations}")
    print(f"  Envs: {env.num_envs}, Rollout: {T} steps, Obs: {OBS_DIM}, Act: {LOCO_ACT_DIM}")

    for iteration in range(start_iter, args_cli.max_iterations):
        # Exploration decay
        progress = iteration / args_cli.max_iterations
        std = 0.8 + (0.2 - 0.8) * progress
        net.log_std.data.fill_(np.log(max(std, 0.2)))

        # Collect rollout
        for t in range(T):
            with torch.no_grad():
                action = net.act(obs_t)
                val = net.critic(obs_t).squeeze(-1)
                dist = torch.distributions.Normal(net.actor(obs_t), net.log_std.clamp(-2, 1).exp())
                lp = dist.log_prob(action).sum(-1)

            obs_buf[t] = obs_t
            act_buf[t] = action
            val_buf[t] = val
            lp_buf[t] = lp

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs_next = obs_dict["policy"]
            done = (terminated | truncated).float()

            rew_buf[t] = reward
            done_buf[t] = done

            ep_rewards += reward
            ep_lengths += 1

            done_mask = done.bool()
            if done_mask.any():
                for r_val in ep_rewards[done_mask].cpu().numpy():
                    completed_rewards.append(r_val)
                ep_rewards[done_mask] = 0
                ep_lengths[done_mask] = 0

            obs_t = obs_next

        # Returns
        with torch.no_grad():
            nv = net.critic(obs_t).squeeze(-1)
        adv, ret = ppo.gae(rew_buf, val_buf, done_buf, nv)

        obs_flat = obs_buf.reshape(-1, OBS_DIM)
        act_flat = act_buf.reshape(-1, LOCO_ACT_DIM)
        lp_flat = lp_buf.reshape(-1)
        ret_flat = ret.reshape(-1)
        adv_flat = adv.reshape(-1)
        val_flat = val_buf.reshape(-1)

        losses = ppo.update(obs_flat, act_flat, lp_flat, ret_flat, adv_flat, val_flat)

        mean_reward = rew_buf.mean().item()
        env.update_curriculum(mean_reward)

        # ============================================================
        # Tensorboard logging
        # ============================================================
        if iteration % 10 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            writer.add_scalar("reward/mean_step", mean_reward, iteration)
            writer.add_scalar("reward/mean_episode", avg_ep, iteration)
            writer.add_scalar("loss/actor", losses["a"], iteration)
            writer.add_scalar("loss/critic", losses["c"], iteration)
            writer.add_scalar("loss/entropy", losses["e"], iteration)
            writer.add_scalar("train/lr", losses["lr"], iteration)
            writer.add_scalar("train/log_std", net.log_std.data.mean().item(), iteration)
            writer.add_scalar("curriculum/level", env.curr_level, iteration)

            # Robot stats
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
            av_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_ang_vel_w)
            writer.add_scalar("robot/height", height, iteration)

            # Speed tracking
            actual_speed = torch.sqrt(lv_b[:, 0]**2 + lv_b[:, 1]**2).mean().item()
            cmd_speed = env.speed_cmd.mean().item()
            writer.add_scalar("robot/speed_actual", actual_speed, iteration)
            writer.add_scalar("robot/speed_cmd", cmd_speed, iteration)
            writer.add_scalar("robot/speed_err", abs(actual_speed - cmd_speed), iteration)

            # Heading tracking — per-env filtered (exclude standing envs)
            moving_mask = env.speed_cmd.squeeze() > 0.05
            if moving_mask.any():
                actual_heading = torch.atan2(lv_b[:, 1], lv_b[:, 0])
                heading_err_all = torch.abs(wrap_to_pi(actual_heading - env.heading_offset))
                heading_err_filtered = heading_err_all[moving_mask].mean().item()
                actual_heading_filtered = actual_heading[moving_mask].mean().item()
                cmd_heading = env.heading_offset[moving_mask].mean().item()
                writer.add_scalar("robot/heading_actual", actual_heading_filtered, iteration)
                writer.add_scalar("robot/heading_cmd", cmd_heading, iteration)
                writer.add_scalar("robot/heading_err", heading_err_filtered, iteration)

            # Yaw tracking
            yaw_actual = av_b[:, 2].mean().item()
            yaw_cmd = env.body_yaw_cmd.mean().item()
            yaw_mode_avg = env.yaw_mode.mean().item()
            writer.add_scalar("robot/yaw_rate_actual", yaw_actual, iteration)
            writer.add_scalar("robot/yaw_cmd", yaw_cmd, iteration)
            writer.add_scalar("robot/yaw_mode", yaw_mode_avg, iteration)

            # Posture
            jp = env.robot.data.joint_pos[:, env.loco_idx]
            ankle_roll_err = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            hip_roll_err = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()
            hip_yaw_err = (jp[:, env.hip_yaw_loco_idx] - env.default_loco[env.hip_yaw_loco_idx]).abs().mean().item()
            writer.add_scalar("posture/ankle_roll_err", ankle_roll_err, iteration)
            writer.add_scalar("posture/hip_roll_err", hip_roll_err, iteration)
            writer.add_scalar("posture/hip_yaw_err", hip_yaw_err, iteration)

            knee_l = jp[:, env.left_knee_idx].mean().item()
            knee_r = jp[:, env.right_knee_idx].mean().item()
            writer.add_scalar("posture/knee_left", knee_l, iteration)
            writer.add_scalar("posture/knee_right", knee_r, iteration)

            waist_yaw = jp[:, 12].mean().item()
            waist_roll = jp[:, 13].mean().item()
            waist_pitch = jp[:, 14].mean().item()
            writer.add_scalar("posture/waist_yaw", waist_yaw, iteration)
            writer.add_scalar("posture/waist_roll", waist_roll, iteration)
            writer.add_scalar("posture/waist_pitch", waist_pitch, iteration)

            # Tilt
            q_root = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=env.device).expand(env.num_envs, -1)
            proj_g = quat_apply_inverse(q_root, gvec)
            tilt_rad = torch.asin(torch.clamp((proj_g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))
            tilt_deg = torch.rad2deg(tilt_rad).mean().item()
            writer.add_scalar("posture/tilt_deg", tilt_deg, iteration)

        # ============================================================
        # Terminal logging
        # ============================================================
        if iteration % 50 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
            av_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_ang_vel_w)

            actual_speed = torch.sqrt(lv_b[:, 0]**2 + lv_b[:, 1]**2).mean().item()
            cmd_speed = env.speed_cmd.mean().item()
            yaw_actual = av_b[:, 2].mean().item()
            yaw_cmd = env.body_yaw_cmd.mean().item()

            jp = env.robot.data.joint_pos[:, env.loco_idx]
            ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            knee_l_p = jp[:, env.left_knee_idx].mean().item()
            knee_r_p = jp[:, env.right_knee_idx].mean().item()
            w_yaw = jp[:, 12].mean().item()
            w_roll = jp[:, 13].mean().item()
            w_pitch = jp[:, 14].mean().item()

            q_root = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=env.device).expand(env.num_envs, -1)
            proj_g = quat_apply_inverse(q_root, gvec)
            tilt_d = torch.rad2deg(torch.asin(torch.clamp((proj_g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))).mean().item()

            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_reward:.2f} EpR={avg_ep:.2f} "
                  f"H={height:.3f} spd={actual_speed:.3f}(cmd:{cmd_speed:.3f}) "
                  f"yaw={yaw_actual:.3f}(cmd:{yaw_cmd:.3f}) "
                  f"Lv={env.curr_level} ankR={ankR:.3f} "
                  f"knL={knee_l_p:.2f} knR={knee_r_p:.2f} "
                  f"tilt={tilt_d:.1f} wY={w_yaw:.3f} wR={w_roll:.3f} wP={w_pitch:.3f} "
                  f"LR={losses['lr']:.2e} std={np.exp(net.log_std.data.mean().item()):.3f}")

        # ============================================================
        # Checkpoint saving
        # ============================================================
        if iteration % 500 == 0 and iteration > 0:
            path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                "model": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
                "scheduler": ppo.sched.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "obs_dim": OBS_DIM,
                "loco_act_dim": LOCO_ACT_DIM,
                "unified_act_dim": ACT_DIM,
            }, path)
            print(f"  [Save] {path}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            path = os.path.join(log_dir, "model_best.pt")
            torch.save({
                "model": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
                "scheduler": ppo.sched.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "obs_dim": OBS_DIM,
                "loco_act_dim": LOCO_ACT_DIM,
                "unified_act_dim": ACT_DIM,
            }, path)

    # Final save
    path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": net.state_dict(),
        "optimizer": ppo.opt.state_dict(),
        "scheduler": ppo.sched.state_dict(),
        "iteration": args_cli.max_iterations - 1,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
        "obs_dim": OBS_DIM,
        "loco_act_dim": LOCO_ACT_DIM,
        "unified_act_dim": ACT_DIM,
    }, path)
    print(f"\n[Done] Egitim tamamlandi. Best reward: {best_reward:.2f}, Level: {env.curr_level}")
    print(f"  Log dir: {log_dir}")

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
