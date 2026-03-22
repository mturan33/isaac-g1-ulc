"""
Unified Stage 2 Loco: Perturbation-Robust Locomotion Fine-Tune (29DoF)
========================================================================
V6.2 loco policy fine-tuned while frozen arm policy perturbs the robot with
random reaching + external payload forces. Loco obs/act space UNCHANGED (66/15).

PROBLEM: V6.2 loco works perfectly with empty arms but fails ~75% when arm
actively reaches + carries payload. Asymmetric torque from arm movement, payload
gravity on palm, combined walk+yaw+load = instability.

ARCHITECTURE:
  LocoActor  (66->15) [512,256,128] + LN + ELU  — V6.2 weights, FINE-TUNE (grad ON)
  LocoCritic (66->1)  [512,256,128] + LN + ELU  — FRESH Xavier init
  ArmActor   (39->7)  [256,256,128] + ELU       — Stage 2 Arm weights, FROZEN

PPO only sees 66-dim loco obs and 15-dim loco act.
Arm runs INSIDE the environment every step (not visible to PPO loop).

PERTURBATION:
  1. Frozen arm policy reaches random targets continuously
  2. External gravity force on palm (0-2kg payload simulation)
  3. Lateral sway force on palm (object momentum simulation)
  4. Enhanced 3D torso push (5-15 step duration, up to 80N)
  5. Arm freeze mode (30% chance hold position 50-200 steps)

CURRICULUM (10 levels, fixed height 0.80m):
  L0: Standing + arm (no walk, no load)
  L1: Slow walk + arm
  L2: Medium walk + light load (0-0.5kg)
  L3: Walk + lateral + medium load (0-1.0kg)
  L4: Omnidirectional + arm + load
  L5: Full range + heavy load (0-2kg) + demo robustness (lateral bias, carry mode)
  L6: Aggressive + walk/stop transitions
  L7: Extreme perturbation
  L8: Drift correction + smooth walking (extra rewards: vyaw=-6, smooth=-3, precision=+2/-2)
  L9: Ultra-smooth precision walking (FINAL)
  Squat levels moved to Stage 3 Loco.

DEMO ROBUSTNESS (2026-03-18):
  - Episode length: 40s (was 20s)
  - Sustained arm: 30% hold 500-1500 steps, 30% demo pose bias (L5+)
  - Lateral bias: 15% pure lateral (L5+)
  - Correlated load+arm: 40% carry mode min 1.0kg (L5+)
  - Yaw drift penalty: 100-step tracking, 15 deg threshold
  - Arm freeze: 50-500 steps (was 50-200)

REWARD: V6.2 base + arm_stability, transition_stability, yaw_drift

CURRICULUM FIX (2026-03-14): Gates tightened to prevent fast advancement.
  - MIN_DWELL=500 iter per level (was 100)
  - vx gate: absolute error < 0.18 m/s (was relative 0.50, changed for perturbation-fairness)
  - vy gate: abs error < 0.08 (was 0.1)
  - vyaw gate: abs error < 0.25 (was 0.3)
  - Standing envs filtered from gate evaluation (15% standing biased averages)

SQUAT UPDATE (2026-03-15): Variable height_cmd + squat curriculum.
  - height_cmd: [0.40, 0.78] random per curriculum level (was fixed 0.80)
  - KL reference: Stage 2 Loco checkpoint (was V6.2) — preserves perturbation robustness
  - Termination: relaxed for squat (min 0.35m floor)
  - Squat knee reward: HOMIE formula couples height with knee bend

USAGE:
    isaaclab.bat -p ... --stage1_checkpoint V6.2_model.pt --arm_checkpoint stage2_arm_model.pt --num_envs 2048 --headless
    isaaclab.bat -p ... --checkpoint stage2_loco_model.pt --num_envs 2048 --headless
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
                         "..", "..", "config", "ulc_g1_29dof_cfg.py")
_spec = importlib.util.spec_from_file_location("ulc_g1_29dof_cfg", _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

G1_29DOF_USD = _cfg_mod.G1_29DOF_USD
LOCO_JOINT_NAMES = _cfg_mod.LOCO_JOINT_NAMES
LEG_JOINT_NAMES = _cfg_mod.LEG_JOINT_NAMES
WAIST_JOINT_NAMES = _cfg_mod.WAIST_JOINT_NAMES
ARM_JOINT_NAMES = _cfg_mod.ARM_JOINT_NAMES
ARM_JOINT_NAMES_RIGHT = _cfg_mod.ARM_JOINT_NAMES_RIGHT
ARM_JOINT_NAMES_LEFT = _cfg_mod.ARM_JOINT_NAMES_LEFT
HAND_JOINT_NAMES = _cfg_mod.HAND_JOINT_NAMES
DEFAULT_LOCO_LIST = _cfg_mod.DEFAULT_LOCO_LIST
DEFAULT_ARM_LIST = _cfg_mod.DEFAULT_ARM_LIST
DEFAULT_HAND_LIST = _cfg_mod.DEFAULT_HAND_LIST
DEFAULT_ALL_POSES = _cfg_mod.DEFAULT_ALL_POSES
DEFAULT_ARM_POSES = _cfg_mod.DEFAULT_ARM_POSES
NUM_LOCO_JOINTS = _cfg_mod.NUM_LOCO_JOINTS
NUM_ARM_JOINTS = _cfg_mod.NUM_ARM_JOINTS
NUM_HAND_JOINTS = _cfg_mod.NUM_HAND_JOINTS
LEG_ACTION_SCALE = _cfg_mod.LEG_ACTION_SCALE
WAIST_ACTION_SCALE = _cfg_mod.WAIST_ACTION_SCALE
ARM_ACTION_SCALE = 2.0  # Stage 2 Arm uses 2.0
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS
SHOULDER_OFFSET_RIGHT = _cfg_mod.SHOULDER_OFFSET_RIGHT

# ============================================================================
# DIMENSIONS
# ============================================================================

OBS_DIM = 66   # V6.2 identical
ACT_DIM = 15   # 12 leg + 3 waist
ARM_OBS_DIM = 39
ARM_ACT_DIM = 7

# ============================================================================
# CURRICULUM — 8 levels (arm perturbation + load + push progression)
# ============================================================================

MIN_DWELL = 500  # Minimum iterations per curriculum level before gate check

CURRICULUM = [
    {
        "description": "L0: Standing + arm random targets (no walk, no load)",
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_active": True,
        "load_range": (0.0, 0.0),
        "push_force": (0, 10),
        "push_interval": (200, 500),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 28.0,
    },
    {
        "description": "L1: Slow walk + arm (no load)",
        "vx": (0.0, 0.3), "vy": (-0.1, 0.1), "vyaw": (-0.2, 0.2),
        "arm_active": True,
        "load_range": (0.0, 0.0),
        "push_force": (0, 10),
        "push_interval": (200, 500),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 26.0,
    },
    {
        "description": "L2: Medium walk + arm + light load (0-0.5kg)",
        "vx": (0.0, 0.5), "vy": (-0.15, 0.15), "vyaw": (-0.4, 0.4),
        "arm_active": True,
        "load_range": (0.0, 0.5),
        "push_force": (0, 15),
        "push_interval": (150, 400),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 24.0,
    },
    {
        "description": "L3: Walk + lateral + medium load (0-1.0kg)",
        "vx": (0.0, 0.5), "vy": (-0.25, 0.25), "vyaw": (-0.5, 0.5),
        "arm_active": True,
        "load_range": (0.0, 1.0),
        "push_force": (0, 20),
        "push_interval": (100, 300),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 23.0,
    },
    {
        "description": "L4: Omnidirectional + arm + load",
        "vx": (-0.2, 0.7), "vy": (-0.3, 0.3), "vyaw": (-0.8, 0.8),
        "arm_active": True,
        "load_range": (0.0, 1.5),
        "push_force": (0, 30),
        "push_interval": (100, 300),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 22.0,
    },
    {
        "description": "L5: Full range + heavy load (0-2kg)",
        "vx": (-0.3, 1.0), "vy": (-0.3, 0.3), "vyaw": (-1.0, 1.0),
        "arm_active": True,
        "load_range": (0.0, 2.0),
        "push_force": (0, 40),
        "push_interval": (80, 250),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 21.0,
    },
    {
        "description": "L6: Aggressive + walk/stop transitions",
        "vx": (-0.5, 1.2), "vy": (-0.4, 0.4), "vyaw": (-1.0, 1.0),
        "arm_active": True,
        "load_range": (0.0, 2.0),
        "push_force": (0, 60),
        "push_interval": (50, 200),
        "cmd_change_interval": (50, 150),
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 20.0,
    },
    {
        "description": "L7: Extreme perturbation",
        "vx": (-0.5, 1.2), "vy": (-0.4, 0.4), "vyaw": (-1.0, 1.0),
        "arm_active": True,
        "load_range": (0.0, 2.0),
        "push_force": (0, 80),
        "push_interval": (50, 150),
        "cmd_change_interval": (30, 100),
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "threshold": 19.0,  # was FINAL, now gates to L8
    },
    # === DRIFT CORRECTION LEVELS (L8-L9) ===
    # L8+ activate extra rewards: vyaw_drift_extra, smooth_vel, vx_precision
    # L0-L7 rewards UNTOUCHED — these are additive on top
    {
        "description": "L8: Drift correction + smooth walking",
        "vx": (-0.3, 0.8), "vy": (-0.4, 0.4), "vyaw": (-0.8, 0.8),
        "arm_active": True,
        "load_range": (0.5, 2.0),
        "push_force": (0, 60),
        "push_interval": (80, 250),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "lateral_bias": 0.20,
        "threshold": 19.0,
    },
    {
        "description": "L9: Ultra-smooth precision walking (FINAL)",
        "vx": (-0.3, 0.8), "vy": (-0.4, 0.4), "vyaw": (-0.8, 0.8),
        "arm_active": True,
        "load_range": (0.5, 2.0),
        "push_force": (0, 80),
        "push_interval": (80, 250),
        "cmd_change_interval": None,
        "height_range": (HEIGHT_DEFAULT, HEIGHT_DEFAULT),
        "lateral_bias": 0.25,
        "threshold": None,  # FINAL
    },
]

# ============================================================================
# REWARD WEIGHTS — V6.2 base + modifications + 2 new terms
# ============================================================================

REWARD_WEIGHTS = {
    # Velocity tracking — vx INCREASED to prevent standing exploit
    "vx": 6.0,                   # was 4.0, policy refused to walk
    "vy": 4.0,                   # vy error scale changed 4->6 inside compute
    "vyaw": 4.0,
    # Gait control
    "gait_knee": 3.0,
    "gait_clearance": 2.0,
    "gait_contact": 2.0,
    # Stability — REVERTED to V6.2 values (increasing caused standing exploit)
    "height": 3.0,               # V6.2 original (squat moved to Stage 3)
    "orientation": 6.0,          # REVERTED: 8.0 -> 6.0 (V6.2 original)
    "ang_vel_penalty": 1.0,
    # Posture
    "ankle_penalty": 2.0,
    "foot_flatness": 3.0,
    "symmetry_gait": 1.5,
    "hip_roll_penalty": 1.5,
    "hip_yaw_penalty": 3.0,
    "knee_negative_penalty": -8.0,
    "waist_posture": 3.0,
    "standing_posture": 3.0,
    # Physics penalties
    "yaw_rate_penalty": -2.0,
    "vz_penalty": -2.0,
    "feet_slip": -0.1,
    # Smoothness penalties
    "action_rate": -0.02,
    "jerk": -0.05,
    "energy": -0.0003,
    "alive": 1.0,
    # NEW: Perturbation stability rewards
    "arm_stability_bonus": 2.0,
    "transition_stability": 1.5,
    # Squat rewards — DISABLED (moved to Stage 3 Loco)
    "squat_knee": 0.0,
    "homie_knee": 0.0,
    # Yaw drift penalty — penalizes accumulated yaw change over 100 steps
    "yaw_drift": -2.0,           # prevents slow heading drift during sustained arm perturbation
}

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Loco: Perturbation-Robust Fine-Tune")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage 1 V6.2 loco checkpoint (required for fresh start)")
    parser.add_argument("--arm_checkpoint", type=str, default=None,
                        help="Stage 2 arm checkpoint (required for fresh start)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 2 Loco checkpoint")
    parser.add_argument("--experiment_name", type=str, default="g1_stage2_loco")
    parser.add_argument("--curriculum_level", type=int, default=None,
                        help="Force curriculum level (overrides checkpoint level)")
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
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from torch.utils.tensorboard import SummaryWriter

print("=" * 80)
print("STAGE 2 LOCO: PERTURBATION-ROBUST LOCOMOTION FINE-TUNE")
print(f"USD: {G1_29DOF_USD}")
print(f"Loco: {OBS_DIM} obs -> {ACT_DIM} act (FINE-TUNE)")
print(f"Arm:  {ARM_OBS_DIM} obs -> {ARM_ACT_DIM} act (FROZEN perturbation)")
print("=" * 80)
for i, lv in enumerate(CURRICULUM):
    print(f"  Level {i}: {lv['description']}")


# ============================================================================
# QUATERNION UTILS
# ============================================================================

def quat_to_euler_xyz(quat):
    """Convert wxyz quaternion (Isaac Lab convention) to roll, pitch, yaw."""
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


def get_palm_forward(quat):
    """Get forward direction from wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fwd_x = 1 - 2 * (y * y + z * z)
    fwd_y = 2 * (x * y + w * z)
    fwd_z = 2 * (x * z - w * y)
    return torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)


def compute_orientation_error(palm_quat, target_dir):
    """Compute angle between palm forward and target direction."""
    forward = get_palm_forward(palm_quat)
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# NETWORK — DualActorCritic (Loco fine-tune + Arm frozen)
# ============================================================================

KL_COEFF = 0.02  # Light brake — prevents KL explosion while allowing squat learning


def build_ref_actor(device):
    """Build frozen copy of V6.2 loco actor for KL reference."""
    layers = []
    prev = OBS_DIM
    for h in [512, 256, 128]:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, ACT_DIM))
    actor = nn.Sequential(*layers).to(device)
    log_std = torch.zeros(ACT_DIM, device=device)
    return actor, log_std


class DualActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Loco Actor: 66 -> [512,256,128](LN+ELU) -> 15 — FINE-TUNE
        layers = []
        prev = OBS_DIM
        for h in [512, 256, 128]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        self.loco_actor = nn.Sequential(*layers)

        # Loco Critic: FRESH Xavier init (new reward landscape)
        layers = []
        prev = OBS_DIM
        for h in [512, 256, 128]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.loco_critic = nn.Sequential(*layers)

        # Arm Actor: 39 -> [256,256,128](ELU, NO LayerNorm) -> 7 — FROZEN
        layers = []
        prev = ARM_OBS_DIM
        for h in [256, 256, 128]:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ARM_ACT_DIM))
        self.arm_actor = nn.Sequential(*layers)

        self.loco_log_std = nn.Parameter(torch.zeros(ACT_DIM))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.loco_actor[-1].weight, gain=0.01)

    def act_loco(self, x, det=False):
        mean = self.loco_actor(x)
        if det:
            return mean
        std = self.loco_log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate_loco(self, x, a):
        mean = self.loco_actor(x)
        val = self.loco_critic(x).squeeze(-1)
        std = self.loco_log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return val, dist.log_prob(a).sum(-1), dist.entropy().sum(-1)

    def act_arm(self, arm_obs):
        """Frozen arm inference (deterministic)."""
        return self.arm_actor(arm_obs)


# ============================================================================
# LOCO PPO — dual LR (actor=1e-4, critic=3e-4), FIXED (no cosine)
# ============================================================================

class LocoPPO:
    def __init__(self, net, device, ref_actor=None, ref_log_std=None,
                 actor_lr=1e-4, critic_lr=3e-4, kl_coeff=KL_COEFF):
        self.net = net
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.kl_coeff = kl_coeff
        # Reference policy (frozen V6.2) for KL penalty
        self.ref_actor = ref_actor  # nn.Sequential, frozen, no grad
        self.ref_log_std = ref_log_std  # tensor, frozen
        # Dual parameter groups
        self.opt = torch.optim.AdamW([
            {"params": list(net.loco_actor.parameters()) + [net.loco_log_std],
             "lr": actor_lr},
            {"params": net.loco_critic.parameters(),
             "lr": critic_lr},
        ], weight_decay=1e-5)

    def gae(self, r, v, d, nv):
        adv = torch.zeros_like(r)
        last = 0
        for t in reversed(range(len(r))):
            nxt = nv if t == len(r) - 1 else v[t + 1]
            delta = r[t] + 0.99 * nxt * (1 - d[t]) - v[t]
            adv[t] = last = delta + 0.99 * 0.95 * (1 - d[t]) * last
        return adv, adv + v

    def _compute_kl(self, obs, act):
        """Mean-only KL penalty — prevents forgetting V6.2 walking.
        Only penalizes mean divergence, NOT std divergence.
        Std is managed by external schedule, so full KL would fight it
        (V6.2 ref_std ~0.2, current starts at 0.5 -> KL ~25 from std alone).
        At iter 0 (same weights): cur_mean == ref_mean -> KL = 0. Correct.
        """
        if self.ref_actor is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            ref_mean = self.ref_actor(obs)
            ref_std = self.ref_log_std.clamp(-2, 1).exp()
        cur_mean = self.net.loco_actor(obs)
        # Mean-only KL: penalize mean divergence, ignore std term
        kl = ((cur_mean - ref_mean) ** 2 / (2 * ref_std ** 2 + 1e-8)).sum(-1)
        return kl.mean()

    def update(self, obs, act, old_lp, ret, adv, old_v):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tot_a, tot_c, tot_e, tot_kl, n = 0, 0, 0, 0, 0
        bs = obs.shape[0]

        for _ in range(5):
            idx = torch.randperm(bs, device=self.device)
            for i in range(0, bs, 4096):
                mb = idx[i:i + 4096]
                val, lp, ent = self.net.evaluate_loco(obs[mb], act[mb])
                ratio = (lp - old_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(0.8, 1.2) * adv[mb]
                a_loss = -torch.min(s1, s2).mean()
                v_clip = old_v[mb] + (val - old_v[mb]).clamp(-0.2, 0.2)
                c_loss = 0.5 * torch.max((val - ret[mb]) ** 2, (v_clip - ret[mb]) ** 2).mean()
                # KL penalty to V6.2 reference
                kl = self._compute_kl(obs[mb], act[mb])
                loss = a_loss + 0.5 * c_loss - 0.01 * ent.mean() + self.kl_coeff * kl
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.net.loco_actor.parameters()) +
                    [self.net.loco_log_std] +
                    list(self.net.loco_critic.parameters()),
                    0.5)
                self.opt.step()
                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                tot_kl += kl.item()
                n += 1

        return {"a": tot_a / max(n, 1), "c": tot_c / max(n, 1),
                "e": tot_e / max(n, 1), "kl": tot_kl / max(n, 1),
                "lr_a": self.actor_lr, "lr_c": self.critic_lr}


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
        contact_forces = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3, track_air_time=False,
        )
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_29DOF_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=True,
                    linear_damping=0.0, angular_damping=0.0,
                    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, HEIGHT_DEFAULT),
                joint_pos=DEFAULT_ALL_POSES,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.90,
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[
                        ".*_hip_yaw_joint", ".*_hip_roll_joint",
                        ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*",
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
                        ".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
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

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 40.0  # increased from 20s for long-horizon stability (demo ~90s)
        action_space = ACT_DIM
        observation_space = OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=5.0)

    class Stage2LocoEnv(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            jn = self.robot.joint_names
            print(f"\n[Stage2Loco] Robot joints ({len(jn)}): {jn}")

            # Loco joint indices (15)
            self.loco_idx = []
            for name in LOCO_JOINT_NAMES:
                if name in jn:
                    self.loco_idx.append(jn.index(name))
            self.loco_idx = torch.tensor(self.loco_idx, device=self.device)
            print(f"  Loco joints: {len(self.loco_idx)} / {NUM_LOCO_JOINTS}")

            # Right arm joint indices (7)
            self.right_arm_idx = []
            for name in ARM_JOINT_NAMES_RIGHT:
                if name in jn:
                    self.right_arm_idx.append(jn.index(name))
            self.right_arm_idx = torch.tensor(self.right_arm_idx, device=self.device)

            # Left arm joint indices (7)
            self.left_arm_idx = []
            for name in ARM_JOINT_NAMES_LEFT:
                if name in jn:
                    self.left_arm_idx.append(jn.index(name))
            self.left_arm_idx = torch.tensor(self.left_arm_idx, device=self.device)

            # Hand indices
            self.hand_idx = []
            for name in HAND_JOINT_NAMES:
                if name in jn:
                    self.hand_idx.append(jn.index(name))
            self.hand_idx = torch.tensor(self.hand_idx, device=self.device)

            # Per-joint loco indices for reward/termination (V6.2 identical)
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
                [LOCO_JOINT_NAMES.index(ln) for ln, rn in sym_pairs], device=self.device)
            self.sym_right_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(rn) for ln, rn in sym_pairs], device=self.device)

            # Find palm body for EE computation + external force
            body_names = self.robot.data.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "palm" in name.lower():
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                self.palm_idx = len(body_names) - 1
                print(f"  [WARN] right_palm not found! Using body {self.palm_idx}")
            else:
                print(f"  Palm body: '{body_names[self.palm_idx]}' (idx={self.palm_idx})")

            # Default poses
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)
            self.default_knee = DEFAULT_LOCO_LIST[self.left_knee_idx]

            self.default_right_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_RIGHT],
                device=self.device, dtype=torch.float32)
            self.default_left_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_LEFT],
                device=self.device, dtype=torch.float32)

            # Action scales (V6.2 identical)
            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # Shoulder offset for arm target sampling
            self.shoulder_offset = torch.tensor(SHOULDER_OFFSET_RIGHT, device=self.device, dtype=torch.float32)

            # State buffers (V6.2 identical)
            self.curr_level = 0
            self.curr_hist = []
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_jvel = None

            # Arm perturbation state
            self.prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)
            self.arm_target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.arm_target_orient = torch.zeros(self.num_envs, 3, device=self.device)
            self.arm_target_orient[:, 2] = -1.0  # palm down default
            self.arm_target_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.arm_target_change_at = torch.randint(100, 300, (self.num_envs,), device=self.device)
            self.arm_steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # Arm freeze mode (30% chance, 50-200 steps)
            self.arm_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.arm_freeze_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # External force state
            self.load_mass = torch.zeros(self.num_envs, device=self.device)

            # Push timer (enhanced from V6.2)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.push_timer = torch.randint(200, 500, (self.num_envs,), device=self.device)
            self.push_duration = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.push_force_active = torch.zeros(self.num_envs, 3, device=self.device)

            # Yaw drift tracking (ring buffer of last 100 yaw values)
            self._yaw_history = torch.zeros(self.num_envs, 100, device=self.device)
            self._yaw_history_idx = 0

            # L8+ smooth velocity tracking (step-to-step velocity change)
            self._prev_lv_b = None
            self._prev_av_b = None

            # Command resample timer
            self.cmd_timer = torch.randint(0, 400, (self.num_envs,), device=self.device)
            self.cmd_resample_lo = 150
            self.cmd_resample_hi = 400
            self.cmd_resample_targets = torch.randint(150, 401, (self.num_envs,), device=self.device)

            # Contact sensor
            self._contact_sensor = self.scene["contact_forces"]
            self._illegal_contact_ids, illegal_names = self._contact_sensor.find_bodies(
                "pelvis|torso_link|.*knee_link")
            print(f"  Illegal contact bodies: {illegal_names}")

            # Initial sampling
            self._sample_commands(torch.arange(self.num_envs, device=self.device))
            self._sample_arm_targets(torch.arange(self.num_envs, device=self.device))
            self._sample_load(torch.arange(self.num_envs, device=self.device))

            print(f"\n[Stage2Loco] {self.num_envs} envs, Level {self.curr_level}")
            print(f"  Obs: {OBS_DIM}, Act: {ACT_DIM}")

        @property
        def robot(self):
            return self.scene["robot"]

        # ================================================================
        # COMMAND SAMPLING (V6.2 identical + 15% standing samples)
        # ================================================================

        def _sample_commands(self, env_ids):
            lv = CURRICULUM[self.curr_level]
            n = len(env_ids)
            vx_lo, vx_hi = lv["vx"]
            self.vel_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (vx_hi - vx_lo) + vx_lo
            vy_lo, vy_hi = lv["vy"]
            self.vel_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (vy_hi - vy_lo) + vy_lo
            vyaw_lo, vyaw_hi = lv["vyaw"]
            self.vel_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (vyaw_hi - vyaw_lo) + vyaw_lo
            # 15% standing samples
            standing_mask = torch.rand(n, device=self.device) < 0.15
            if standing_mask.any():
                self.vel_cmd[env_ids[standing_mask]] = 0.0
            # Pure lateral bias (L5+): vx≈0, |vy|=0.2-0.4 (carry walk simulation)
            # L5-L7: 15%, L8+: from curriculum lateral_bias field
            lateral_pct = lv.get("lateral_bias", 0.15) if self.curr_level >= 5 else 0.0
            if lateral_pct > 0:
                lateral_mask = (~standing_mask) & (torch.rand(n, device=self.device) < lateral_pct)
                if lateral_mask.any():
                    n_lat = lateral_mask.sum().item()
                    self.vel_cmd[env_ids[lateral_mask], 0] = torch.empty(n_lat, device=self.device).uniform_(-0.05, 0.05)
                    vy_sign = torch.sign(torch.randn(n_lat, device=self.device))
                    self.vel_cmd[env_ids[lateral_mask], 1] = vy_sign * torch.empty(n_lat, device=self.device).uniform_(0.20, 0.40)
            # Height sampling (variable for squat levels L8+, fixed for L0-L7)
            h_lo, h_hi = lv.get("height_range", (HEIGHT_DEFAULT, HEIGHT_DEFAULT))
            if h_hi - h_lo > 0.01:
                self.height_cmd[env_ids] = torch.empty(n, device=self.device).uniform_(h_lo, h_hi)
            else:
                self.height_cmd[env_ids] = h_lo

        # ================================================================
        # ARM TARGET SAMPLING (Stage 2 workspace, body frame)
        # ================================================================

        def _sample_arm_targets(self, env_ids):
            n = len(env_ids)

            # --- Demo pose bias: 30% forward-high targets (L5+ only) ---
            # Demo position: arm forward and elevated (reach/lift/carry)
            demo_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            if self.curr_level >= 5:
                demo_mask = torch.rand(n, device=self.device) < 0.30

            # Standard spherical sampling from shoulder (Stage 2 identical)
            azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
            elevation = torch.empty(n, device=self.device).uniform_(-0.3, 0.5)
            radius = torch.empty(n, device=self.device).uniform_(0.15, 0.45)

            target_x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
            target_y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
            target_z = radius * torch.sin(elevation) + self.shoulder_offset[2]

            # Override demo envs with forward-high targets
            if demo_mask.any():
                n_demo = demo_mask.sum().item()
                target_x[demo_mask] = torch.empty(n_demo, device=self.device).uniform_(0.20, 0.35)
                target_y[demo_mask] = torch.empty(n_demo, device=self.device).uniform_(-0.15, -0.05)
                target_z[demo_mask] = torch.empty(n_demo, device=self.device).uniform_(0.10, 0.25)

            target_x = target_x.clamp(-0.10, 0.55)
            target_y = target_y.clamp(-0.60, -0.05)
            target_z = target_z.clamp(-0.15, 0.55)

            self.arm_target_pos_body[env_ids] = torch.stack([target_x, target_y, target_z], dim=-1)
            # Random orientation (unit vector on sphere)
            orient = torch.randn(n, 3, device=self.device)
            orient = orient / (orient.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            self.arm_target_orient[env_ids] = orient

            self.arm_steps_since_spawn[env_ids] = 0

            # Arm freeze mode: 30% chance, 50-500 step duration (increased from 50-200)
            freeze_mask = torch.rand(n, device=self.device) < 0.3
            self.arm_frozen[env_ids] = freeze_mask
            freeze_dur = torch.randint(50, 500, (n,), device=self.device)  # extended for demo robustness
            self.arm_freeze_timer[env_ids] = torch.where(freeze_mask, freeze_dur,
                                                          torch.zeros(n, dtype=torch.long, device=self.device))

        # ================================================================
        # LOAD SAMPLING (per-episode external force)
        # ================================================================

        def _sample_load(self, env_ids):
            lv = CURRICULUM[self.curr_level]
            lo, hi = lv["load_range"]
            n = len(env_ids)
            self.load_mass[env_ids] = torch.empty(n, device=self.device).uniform_(lo, hi)
            # Correlated carry mode (L5+): 40% envs get high load + forward arm target
            if self.curr_level >= 5 and hi > 0.5:
                carry_mask = torch.rand(n, device=self.device) < 0.40
                if carry_mask.any():
                    n_carry = carry_mask.sum().item()
                    # Minimum 1.0kg load for carry envs
                    carry_load = torch.empty(n_carry, device=self.device).uniform_(max(1.0, lo), hi)
                    self.load_mass[env_ids[carry_mask]] = carry_load
                    # Set arm target to forward-high (carry position)
                    carry_ids = env_ids[carry_mask]
                    self.arm_target_pos_body[carry_ids, 0] = torch.empty(n_carry, device=self.device).uniform_(0.20, 0.35)
                    self.arm_target_pos_body[carry_ids, 1] = torch.empty(n_carry, device=self.device).uniform_(-0.15, -0.05)
                    self.arm_target_pos_body[carry_ids, 2] = torch.empty(n_carry, device=self.device).uniform_(0.10, 0.25)

        # ================================================================
        # ARM OBS (Stage 2 identical — 39 dim)
        # ================================================================

        def get_arm_obs(self):
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w

            arm_pos = r.data.joint_pos[:, self.right_arm_idx]
            arm_vel = r.data.joint_vel[:, self.right_arm_idx] * 0.1

            # EE in body frame
            palm_pos = r.data.body_pos_w[:, self.palm_idx]
            palm_quat = r.data.body_quat_w[:, self.palm_idx]
            fwd = get_palm_forward(palm_quat)
            ee_w = palm_pos + 0.02 * fwd
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            pos_error = self.arm_target_pos_body - ee_body
            orient_err = compute_orientation_error(palm_quat, self.arm_target_orient)
            orient_err_norm = orient_err.unsqueeze(-1) / np.pi

            steps_norm = (self.arm_steps_since_spawn.float() / 200.0).unsqueeze(-1).clamp(0, 2)

            obs = torch.cat([
                arm_pos,                       # 7
                arm_vel,                       # 7
                ee_body,                       # 3
                palm_quat,                     # 4
                self.arm_target_pos_body,      # 3
                self.arm_target_orient,        # 3
                pos_error,                     # 3
                orient_err_norm,               # 1
                self.prev_arm_act,             # 7
                steps_norm,                    # 1
            ], dim=-1)  # = 39
            return obs.clamp(-10, 10).nan_to_num()

        # ================================================================
        # PRE-PHYSICS STEP
        # ================================================================

        def _pre_physics_step(self, loco_act):
            self.actions = loco_act.clone()
            tgt = self.robot.data.default_joint_pos.clone()

            # Loco joints (V6.2 identical)
            tgt[:, self.loco_idx] = self.default_loco + loco_act * self.action_scales

            # Waist clamp (V6.2)
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)

            # Hip yaw clamp (V6.1)
            for hy_idx in self.hip_yaw_loco_idx:
                tgt[:, self.loco_idx[hy_idx]].clamp_(-0.3, 0.3)

            # RIGHT ARM: frozen arm policy action (stored externally by training loop)
            arm_act = self.prev_arm_act.clone()
            tgt[:, self.right_arm_idx] = self.default_right_arm + arm_act * ARM_ACTION_SCALE

            # LEFT ARM: default
            tgt[:, self.left_arm_idx] = self.default_left_arm

            # HANDS: open
            tgt[:, self.hand_idx] = self.default_hand

            self.robot.set_joint_position_target(tgt)

            # Update state
            self._prev_act = self.prev_act.clone()
            self.prev_act = loco_act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
            self.step_count += 1

            # Command resample
            self.cmd_timer += 1
            lv = CURRICULUM[self.curr_level]
            # Fast command change interval for L6+
            cmd_interval = lv.get("cmd_change_interval", None)
            if cmd_interval is not None:
                lo, hi = cmd_interval
                cmd_mask = self.cmd_timer >= self.cmd_resample_targets
                if cmd_mask.any():
                    cmd_ids = cmd_mask.nonzero(as_tuple=False).squeeze(-1)
                    self._sample_commands(cmd_ids)
                    self.cmd_timer[cmd_ids] = 0
                    self.cmd_resample_targets[cmd_ids] = torch.randint(lo, hi + 1, (len(cmd_ids),), device=self.device)
            else:
                cmd_mask = self.cmd_timer >= self.cmd_resample_targets
                if cmd_mask.any():
                    cmd_ids = cmd_mask.nonzero(as_tuple=False).squeeze(-1)
                    self._sample_commands(cmd_ids)
                    self.cmd_timer[cmd_ids] = 0
                    self.cmd_resample_targets[cmd_ids] = torch.randint(
                        self.cmd_resample_lo, self.cmd_resample_hi + 1, (len(cmd_ids),), device=self.device)

            # Arm target timer
            self.arm_target_timer += 1
            self.arm_steps_since_spawn += 1

            # Arm freeze countdown
            freeze_expired = self.arm_frozen & (self.arm_freeze_timer <= 0)
            self.arm_frozen[freeze_expired] = False
            self.arm_freeze_timer[self.arm_frozen] -= 1

            # Resample arm targets
            change_mask = self.arm_target_timer >= self.arm_target_change_at
            if change_mask.any():
                change_ids = change_mask.nonzero(as_tuple=False).squeeze(-1)
                self._sample_arm_targets(change_ids)
                self.arm_target_timer[change_ids] = 0
                n_ch = len(change_ids)
                # Sustained mode: 30% hold same target for 500-1500 steps (L5+)
                if self.curr_level >= 5:
                    sustained = torch.rand(n_ch, device=self.device) < 0.30
                    normal_dur = torch.randint(100, 300, (n_ch,), device=self.device)
                    sustained_dur = torch.randint(500, 1500, (n_ch,), device=self.device)
                    self.arm_target_change_at[change_ids] = torch.where(sustained, sustained_dur, normal_dur)
                else:
                    self.arm_target_change_at[change_ids] = torch.randint(100, 300, (n_ch,), device=self.device)

            # Apply external forces (load + push)
            self._apply_forces()

        def _apply_action(self):
            pass

        # ================================================================
        # EXTERNAL FORCES (palm load + torso push)
        # ================================================================

        def _apply_forces(self):
            num_bodies = self.robot.data.body_pos_w.shape[1]
            forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
            torques = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)

            # Palm load: gravity + sway
            load_mask = self.load_mass > 0.01
            if load_mask.any():
                # Gravity on palm
                forces[load_mask, self.palm_idx, 2] = -self.load_mass[load_mask] * 9.81
                # Sway (object momentum sim)
                sway = torch.randn(load_mask.sum(), 2, device=self.device)
                sway = sway * self.load_mass[load_mask, None] * 2.0
                forces[load_mask, self.palm_idx, :2] += sway

            # Torso push (enhanced: 3D, duration-based)
            lv = CURRICULUM[self.curr_level]
            push_mask = self.step_count >= self.push_timer

            # Continue active push
            active_push = self.push_duration > 0
            if active_push.any():
                forces[active_push, 0, :] += self.push_force_active[active_push]
                self.push_duration[active_push] -= 1

            # Start new push
            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                # 3D random direction
                force = torch.randn(n, 3, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                fmin, fmax = lv["push_force"]
                mag = torch.rand(n, 1, device=self.device) * (fmax - fmin) + fmin
                self.push_force_active[ids] = force * mag
                # Duration: 5-15 steps
                self.push_duration[ids] = torch.randint(5, 16, (n,), device=self.device)
                # Reset timer
                pi_lo, pi_hi = lv["push_interval"]
                if pi_hi <= pi_lo:
                    pi_hi = pi_lo + 1
                self.push_timer[ids] = self.step_count[ids] + torch.randint(pi_lo, pi_hi, (n,), device=self.device)

            self.robot.set_external_force_and_torque(forces, torques)

        # ================================================================
        # OBSERVATIONS (V6.2 identical — 66 dim)
        # ================================================================

        def _get_observations(self):
            r = self.robot
            q = r.data.root_quat_w
            lv = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            jp_leg = r.data.joint_pos[:, self.loco_idx[:12]]
            jv_leg = r.data.joint_vel[:, self.loco_idx[:12]] * 0.1
            jp_waist = r.data.joint_pos[:, self.loco_idx[12:15]]
            jv_waist = r.data.joint_vel[:, self.loco_idx[12:15]] * 0.1

            gait = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)
            ], -1)

            torso_euler = quat_to_euler_xyz(q)

            obs = torch.cat([
                lv,                          # 3
                av,                          # 3
                g,                           # 3
                jp_leg,                      # 12
                jv_leg,                      # 12
                jp_waist,                    # 3
                jv_waist,                    # 3
                self.height_cmd[:, None],    # 1
                self.vel_cmd,                # 3
                gait,                        # 2
                self.prev_act,               # 15
                torso_euler,                 # 3
                self.torso_cmd,              # 3
            ], dim=-1)

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        # ================================================================
        # REWARDS (V6.2 22 terms + 2 new)
        # ================================================================

        def _get_rewards(self):
            r = self.robot
            q = r.data.root_quat_w
            pos = r.data.root_pos_w

            lv_b = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av_b = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            jp = r.data.joint_pos[:, self.loco_idx]
            jv = r.data.joint_vel[:, self.loco_idx]

            # === VELOCITY TRACKING ===
            vx_err = (lv_b[:, 0] - self.vel_cmd[:, 0]) ** 2
            r_vx = torch.exp(-3.0 * vx_err)

            # vy — enhanced scale (4.0 -> 6.0 for lateral tracking)
            vy_err = (lv_b[:, 1] - self.vel_cmd[:, 1]) ** 2
            r_vy = torch.exp(-6.0 * vy_err)

            vyaw_err = (av_b[:, 2] - self.vel_cmd[:, 2]) ** 2
            r_vyaw = torch.exp(-5.0 * vyaw_err)

            # === GAIT CONTROL (V6.2 identical) ===
            ph = self.phase
            l_swing = (ph < 0.5).float()
            r_swing = (ph >= 0.5).float()

            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_swing_target = 0.65
            knee_stance_target = self.default_knee

            left_knee_target = l_swing * knee_swing_target + (1 - l_swing) * knee_stance_target
            right_knee_target = r_swing * knee_swing_target + (1 - r_swing) * knee_stance_target
            knee_err = (lk - left_knee_target) ** 2 + (rk - right_knee_target) ** 2
            r_gait_knee = torch.exp(-5.0 * knee_err)

            knee_min = 0.1
            lk_violation = torch.clamp(knee_min - lk, min=0.0)
            rk_violation = torch.clamp(knee_min - rk, min=0.0)
            r_knee_neg_penalty = lk_violation + rk_violation

            lh = jp[:, self.left_hip_pitch_idx]
            rh = jp[:, self.right_hip_pitch_idx]
            hip_swing_target = -0.35
            hip_stance_target = self.default_loco[self.left_hip_pitch_idx].item()
            left_hip_target = l_swing * hip_swing_target + (1 - l_swing) * hip_stance_target
            right_hip_target = r_swing * hip_swing_target + (1 - r_swing) * hip_stance_target
            hip_err = (lh - left_hip_target) ** 2 + (rh - right_hip_target) ** 2
            r_gait_clearance = torch.exp(-4.0 * hip_err)

            lk_vel = jv[:, self.left_knee_idx].abs()
            rk_vel = jv[:, self.right_knee_idx].abs()
            swing_activity = l_swing * lk_vel + r_swing * rk_vel
            stance_stability = (1 - l_swing) * lk_vel + (1 - r_swing) * rk_vel
            r_gait_contact = torch.tanh(swing_activity * 0.5) * torch.exp(-2.0 * stance_stability)

            vel_magnitude = (self.vel_cmd[:, 0].abs()
                           + self.vel_cmd[:, 1].abs()
                           + self.vel_cmd[:, 2].abs() * 0.3)
            gait_scale = torch.clamp(vel_magnitude / 0.2, 0.0, 1.0)

            r_gait_knee = r_gait_knee * gait_scale + (1 - gait_scale) * 1.0
            r_gait_clearance = r_gait_clearance * gait_scale + (1 - gait_scale) * 1.0
            r_gait_contact = r_gait_contact * gait_scale

            # Squat gait scaling removed (squat moved to Stage 3)

            # === STABILITY ===
            height = pos[:, 2]
            r_height = torch.exp(-25.0 * (height - self.height_cmd) ** 2)

            r_orient = torch.exp(-15.0 * (g[:, :2] ** 2).sum(-1))
            r_ang = torch.exp(-1.0 * (av_b[:, :2] ** 2).sum(-1))

            # === POSTURE (V6.2 identical) ===
            ankle_roll_err = (jp[:, self.ankle_roll_loco_idx] - self.default_loco[self.ankle_roll_loco_idx]) ** 2
            r_ankle = torch.exp(-15.0 * ankle_roll_err.sum(-1))

            ankle_pitch_err = (jp[:, self.ankle_pitch_loco_idx] - self.default_loco[self.ankle_pitch_loco_idx]) ** 2
            r_foot_flat = torch.exp(-15.0 * ankle_pitch_err.sum(-1))

            left_pos = jp[:, self.sym_left_idx]
            right_pos = jp[:, self.sym_right_idx]
            left_dev = (left_pos - self.default_loco[self.sym_left_idx]).abs()
            right_dev = (right_pos - self.default_loco[self.sym_right_idx]).abs()
            sym_range_err = (left_dev - right_dev) ** 2
            r_sym_gait = torch.exp(-3.0 * sym_range_err.sum(-1))

            hip_roll_err = (jp[:, self.hip_roll_loco_idx] - self.default_loco[self.hip_roll_loco_idx]) ** 2
            r_hip_roll = torch.exp(-10.0 * hip_roll_err.sum(-1))

            hip_yaw_err = (jp[:, self.hip_yaw_loco_idx] - self.default_loco[self.hip_yaw_loco_idx]) ** 2
            r_hip_yaw = torch.exp(-8.0 * hip_yaw_err.sum(-1))

            waist_yaw_err = (jp[:, 12] - self.default_loco[12]) ** 2
            waist_roll_err = (jp[:, 13] - self.default_loco[13]) ** 2
            waist_pitch_err = (jp[:, 14] - self.default_loco[14]) ** 2
            r_waist_posture = (torch.exp(-25.0 * waist_yaw_err)
                             * torch.exp(-20.0 * waist_roll_err)
                             * torch.exp(-15.0 * waist_pitch_err))

            standing_scale = 1.0 - gait_scale
            leg_pos_err = (jp[:, :12] - self.default_loco[:12]) ** 2
            r_standing_posture_raw = torch.exp(-3.0 * leg_pos_err.sum(-1))
            r_standing_posture = standing_scale * r_standing_posture_raw + (1 - standing_scale) * 1.0
            # squat_posture_mask removed (squat moved to Stage 3)

            # === PHYSICS PENALTIES ===
            vz = lv_b[:, 2]
            r_vz_penalty = vz ** 2

            yaw_rate = av_b[:, 2]
            vyaw_cmd_mag = self.vel_cmd[:, 2].abs()
            yaw_penalty_scale = torch.clamp(1.0 - vyaw_cmd_mag / 0.3, 0.0, 1.0)
            r_yaw_rate_penalty = yaw_rate ** 2 * yaw_penalty_scale

            ankle_pitch_vel = jv[:, self.ankle_pitch_loco_idx].abs()
            ankle_roll_vel = jv[:, self.ankle_roll_loco_idx].abs()
            left_stance = 1.0 - l_swing
            right_stance = 1.0 - r_swing
            left_slip = left_stance * (ankle_pitch_vel[:, 0] + ankle_roll_vel[:, 0])
            right_slip = right_stance * (ankle_pitch_vel[:, 1] + ankle_roll_vel[:, 1])
            r_feet_slip = left_slip + right_slip

            # === SMOOTHNESS PENALTIES ===
            dact = self.prev_act - self._prev_act
            r_action_rate = (dact ** 2).sum(-1)

            if self._prev_jvel is None:
                self._prev_jvel = jv.clone()
            jerk = ((jv - self._prev_jvel) ** 2).sum(-1)
            self._prev_jvel = jv.clone()

            r_energy = (jv.abs() * jp.abs()).sum(-1)

            # === NEW: YAW DRIFT PENALTY ===
            # Track accumulated yaw change over 100 steps
            # Penalizes unintended heading drift (e.g. from asymmetric arm perturbation)
            torso_euler_rew = quat_to_euler_xyz(q)
            current_yaw = torso_euler_rew[:, 2]  # rad
            self._yaw_history[:, self._yaw_history_idx] = current_yaw
            oldest_idx = (self._yaw_history_idx + 1) % 100
            oldest_yaw = self._yaw_history[:, oldest_idx]
            # Wrap-safe yaw difference
            yaw_change = torch.atan2(
                torch.sin(current_yaw - oldest_yaw),
                torch.cos(current_yaw - oldest_yaw)
            ).abs()
            self._yaw_history_idx = (self._yaw_history_idx + 1) % 100
            # Penalize drift beyond 15 deg (0.2618 rad), scaled by how much exceeds threshold
            # Only penalize when vyaw_cmd is small (robot shouldn't be turning)
            vyaw_cmd_drift = self.vel_cmd[:, 2].abs()
            drift_active = (vyaw_cmd_drift < 0.2).float()  # only when not commanded to turn
            r_yaw_drift = torch.clamp(yaw_change - 0.2618, min=0.0) * drift_active

            # === NEW: ARM STABILITY BONUS (relative to height_cmd) ===
            r_arm_stability = (height > (self.height_cmd - 0.10)).float()

            # === NEW: SQUAT KNEE (positive exp, secondary to HOMIE) ===
            # Couples height command with knee bend — low height = bent knees
            knee_max = 2.0  # matched with termination limit
            lk_norm = lk.clamp(0, knee_max) / knee_max
            rk_norm = rk.clamp(0, knee_max) / knee_max
            knee_norm = (lk_norm + rk_norm) / 2.0
            h_target_norm = ((self.height_cmd - 0.40) / (0.78 - 0.40)).clamp(0, 1)
            desired_knee_norm = 1.0 - h_target_norm
            r_squat_knee = torch.exp(-8.0 * (knee_norm - desired_knee_norm) ** 2)

            # === NEW: HOMIE Knee-Height Coupled Reward (RSS 2025) ===
            # r = -|height_err * (knee_norm - 0.5)|
            # When h_cmd < h_actual AND knees straight → BIG penalty
            # When h_cmd < h_actual AND knees bent → small penalty (correct direction)
            # When h_cmd ≈ h_actual → penalty ≈ 0 (height correct)
            KNEE_MIN = 0.0   # straight leg (rad)
            KNEE_MAX = 2.0   # deep squat (rad) — G1 knee joint range
            lk_norm_homie = (lk.clamp(KNEE_MIN, KNEE_MAX) - KNEE_MIN) / (KNEE_MAX - KNEE_MIN)
            rk_norm_homie = (rk.clamp(KNEE_MIN, KNEE_MAX) - KNEE_MIN) / (KNEE_MAX - KNEE_MIN)
            h_err_signed = self.height_cmd - height  # negative = need to squat
            r_homie_left = -torch.abs(h_err_signed * (lk_norm_homie - 0.5))
            r_homie_right = -torch.abs(h_err_signed * (rk_norm_homie - 0.5))
            r_homie_knee = (r_homie_left + r_homie_right) / 2.0

            # === NEW: TRANSITION STABILITY ===
            cmd_near_zero = (self.vel_cmd.abs().sum(-1) < 0.1).float()
            vel_near_zero = (lv_b[:, :2].norm(dim=-1) < 0.15).float()
            r_transition = cmd_near_zero * vel_near_zero

            # === L8+ CONDITIONAL EXTRA REWARDS (drift correction) ===
            # These are ADDITIVE on top of L0-L7 rewards. Zero for L0-L7.
            r_drift_extra = torch.zeros(self.num_envs, device=self.device)
            r_smooth_extra = torch.zeros(self.num_envs, device=self.device)
            r_precision_extra = torch.zeros(self.num_envs, device=self.device)

            if self.curr_level >= 8:
                # 1. Strong vyaw drift penalty — tracks vyaw error more aggressively
                vyaw_err_sq = (av_b[:, 2] - self.vel_cmd[:, 2]) ** 2
                r_drift_extra = -6.0 * vyaw_err_sq

                # 2. Smooth velocity — penalize step-to-step velocity oscillation
                if self._prev_lv_b is not None:
                    vx_change = (lv_b[:, 0] - self._prev_lv_b[:, 0]).abs()
                    vy_change = (lv_b[:, 1] - self._prev_lv_b[:, 1]).abs()
                    vyaw_change = (av_b[:, 2] - self._prev_av_b[:, 2]).abs()
                    r_smooth_extra = -3.0 * (vx_change + vy_change + 0.5 * vyaw_change)
                self._prev_lv_b = lv_b.clone()
                self._prev_av_b = av_b.clone()

                # 3. vx precision bonus — reward tight tracking, penalize large error
                vx_err_abs = (lv_b[:, 0] - self.vel_cmd[:, 0]).abs()
                r_precision_extra = torch.where(
                    vx_err_abs < 0.08,
                    torch.tensor(2.0, device=self.device),
                    -2.0 * (vx_err_abs - 0.08)
                )

            # === REWARD SUM ===
            # L0-L7 base rewards (UNTOUCHED)
            reward_base = (
                # Velocity tracking
                REWARD_WEIGHTS["vx"] * r_vx
                + REWARD_WEIGHTS["vy"] * r_vy
                + REWARD_WEIGHTS["vyaw"] * r_vyaw
                # Gait control
                + REWARD_WEIGHTS["gait_knee"] * r_gait_knee
                + REWARD_WEIGHTS["gait_clearance"] * r_gait_clearance
                + REWARD_WEIGHTS["gait_contact"] * r_gait_contact
                # Stability
                + REWARD_WEIGHTS["height"] * r_height
                + REWARD_WEIGHTS["orientation"] * r_orient
                + REWARD_WEIGHTS["ang_vel_penalty"] * r_ang
                # Posture
                + REWARD_WEIGHTS["ankle_penalty"] * r_ankle
                + REWARD_WEIGHTS["foot_flatness"] * r_foot_flat
                + REWARD_WEIGHTS["symmetry_gait"] * r_sym_gait
                + REWARD_WEIGHTS["hip_roll_penalty"] * r_hip_roll
                + REWARD_WEIGHTS["hip_yaw_penalty"] * r_hip_yaw
                + REWARD_WEIGHTS["waist_posture"] * r_waist_posture
                + REWARD_WEIGHTS["standing_posture"] * r_standing_posture
                # Perturbation stability
                + REWARD_WEIGHTS["arm_stability_bonus"] * r_arm_stability
                + REWARD_WEIGHTS["transition_stability"] * r_transition
                # Penalties
                + REWARD_WEIGHTS["knee_negative_penalty"] * r_knee_neg_penalty
                + REWARD_WEIGHTS["vz_penalty"] * r_vz_penalty
                + REWARD_WEIGHTS["yaw_rate_penalty"] * r_yaw_rate_penalty
                + REWARD_WEIGHTS["feet_slip"] * r_feet_slip
                + REWARD_WEIGHTS["action_rate"] * r_action_rate
                + REWARD_WEIGHTS["jerk"] * jerk
                + REWARD_WEIGHTS["energy"] * r_energy
                + REWARD_WEIGHTS["yaw_drift"] * r_yaw_drift
                # Alive
                + REWARD_WEIGHTS["alive"]
            )

            # L8+ extra rewards (conditional, zero for L0-L7)
            reward = reward_base + r_drift_extra + r_smooth_extra + r_precision_extra

            return reward

        # ================================================================
        # TERMINATION (V6.2 identical)
        # ================================================================

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gravity_vec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(q, gravity_vec)

            # Height termination: relaxed for squat (min 0.35m floor)
            min_height = torch.where(self.height_cmd < 0.60,
                                      torch.clamp(self.height_cmd - 0.10, min=0.35),
                                      torch.tensor(0.55, device=self.device))
            fallen = (pos[:, 2] < min_height) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_bad = (lk < -0.05) | (rk < -0.05) | (lk > 1.5) | (rk > 1.5)  # V6.2 original (squat moved to Stage 3)

            waist_pitch_val = jp[:, 14]
            waist_roll_val = jp[:, 13]
            waist_excessive = (waist_pitch_val.abs() > 0.35) | (waist_roll_val.abs() > 0.25)

            hip_yaw_L = jp[:, self.hip_yaw_loco_idx[0]]
            hip_yaw_R = jp[:, self.hip_yaw_loco_idx[1]]
            hip_yaw_excessive = (hip_yaw_L.abs() > 0.6) | (hip_yaw_R.abs() > 0.6)

            if len(self._illegal_contact_ids) > 0:
                net_forces = self._contact_sensor.data.net_forces_w_history
                illegal_contact = torch.any(
                    torch.max(
                        torch.norm(net_forces[:, :, self._illegal_contact_ids], dim=-1), dim=1
                    )[0] > 1.0,
                    dim=1,
                )
            else:
                illegal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            terminated = fallen | bad_orientation | knee_bad | waist_excessive | hip_yaw_excessive | illegal_contact
            time_out = self.episode_length_buf >= self.max_episode_length
            return terminated, time_out

        # ================================================================
        # RESET
        # ================================================================

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0:
                return
            n = len(env_ids)

            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            noise = torch.randn_like(default_pos) * 0.02
            joint_pos = default_pos + noise

            root_pos = torch.tensor([[0.0, 0.0, HEIGHT_DEFAULT]], device=self.device).expand(n, -1).clone()
            root_pos[:, :2] += torch.randn(n, 2, device=self.device) * 0.05

            # Squat initial state removed (moved to Stage 3 Loco)

            self.robot.write_joint_state_to_sim(
                joint_pos,
                torch.zeros_like(joint_pos), None, env_ids)

            yaw = torch.randn(n, device=self.device) * 0.1
            qw = torch.cos(yaw / 2)
            root_quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw),
                                      torch.sin(yaw / 2)], dim=-1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            # Reset state
            self._yaw_history[env_ids] = 0
            # Reset L8+ smooth velocity buffers (set to None forces re-init on next step)
            if self._prev_lv_b is not None:
                self._prev_lv_b[env_ids] = 0
            if self._prev_av_b is not None:
                self._prev_av_b[env_ids] = 0
            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.prev_arm_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)
            self.step_count[env_ids] = 0
            lv = CURRICULUM[self.curr_level]
            pi_lo, pi_hi = lv["push_interval"]
            if pi_hi <= pi_lo:
                pi_hi = pi_lo + 1
            self.push_timer[env_ids] = torch.randint(pi_lo, pi_hi, (n,), device=self.device)
            self.push_duration[env_ids] = 0
            self.push_force_active[env_ids] = 0

            # Resample commands, arm targets, load
            self._sample_commands(env_ids)
            self._sample_arm_targets(env_ids)
            self._sample_load(env_ids)
            self.cmd_timer[env_ids] = torch.randint(0, self.cmd_resample_hi, (n,), device=self.device)
            self.cmd_resample_targets[env_ids] = torch.randint(
                self.cmd_resample_lo, self.cmd_resample_hi + 1, (n,), device=self.device)
            self.arm_target_timer[env_ids] = 0
            self.arm_target_change_at[env_ids] = torch.randint(100, 300, (n,), device=self.device)

        # ================================================================
        # CURRICULUM
        # ================================================================

        def update_curriculum(self, r):
            self.curr_hist.append(r)
            # MIN_DWELL: must stay at each level for at least MIN_DWELL iterations
            if len(self.curr_hist) < max(100, MIN_DWELL):
                return
            avg = np.mean(self.curr_hist[-100:])
            thr = CURRICULUM[self.curr_level]["threshold"]
            if thr is not None and avg > thr and self.curr_level < len(CURRICULUM) - 1:
                # Check tracking quality — filter standing envs from gate
                lv_b = quat_apply_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)
                av_b = quat_apply_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)

                # Filter: only evaluate walking envs (|vx_cmd| > 0.02)
                # Standing envs have vx=0 cmd, their small drift biases the average
                speed_mag = self.vel_cmd[:, 0].abs() + self.vel_cmd[:, 1].abs() + self.vel_cmd[:, 2].abs()
                walk_mask = speed_mag > 0.02
                n_walk = walk_mask.sum().item()

                if n_walk > 0:
                    vx_actual = lv_b[walk_mask, 0].mean().item()
                    vx_cmd = self.vel_cmd[walk_mask, 0].mean().item()
                    vy_actual = lv_b[walk_mask, 1].mean().item()
                    vy_cmd = self.vel_cmd[walk_mask, 1].mean().item()
                    vyaw_actual = av_b[walk_mask, 2].mean().item()
                    vyaw_cmd = self.vel_cmd[walk_mask, 2].mean().item()
                else:
                    # All standing (L0) — use full average
                    vx_actual = lv_b[:, 0].mean().item()
                    vx_cmd = self.vel_cmd[:, 0].mean().item()
                    vy_actual = lv_b[:, 1].mean().item()
                    vy_cmd = self.vel_cmd[:, 1].mean().item()
                    vyaw_actual = av_b[:, 2].mean().item()
                    vyaw_cmd = self.vel_cmd[:, 2].mean().item()

                vx_err_abs = abs(vx_actual - vx_cmd)
                vx_ok = vx_err_abs < 0.18  # absolute error gate (perturbation-fair, was relative 0.50)

                vy_err_abs = abs(vy_actual - vy_cmd)
                vy_ok = vy_err_abs < 0.08 or abs(vy_cmd) < 0.05  # tightened: was 0.1

                vyaw_err_abs = abs(vyaw_actual - vyaw_cmd)
                vyaw_ok = vyaw_err_abs < 0.25 or abs(vyaw_cmd) < 0.1  # tightened: was 0.3

                # drift_ok gate REMOVED — av_b.abs().mean() measures gait oscillation (~0.57),
                # not actual drift. L8 rewards (vyaw_drift=-6.0) handle drift correction.

                if vx_ok and vy_ok and vyaw_ok:
                    self.curr_level += 1
                    new_lv = CURRICULUM[self.curr_level]
                    print(f"\n*** LEVEL UP! Now {self.curr_level}: {new_lv['description']} ***")
                    print(f"    Load: {new_lv['load_range']}, Push: {new_lv['push_force']}")
                    print(f"    Gate: vx={vx_actual:.3f}({vx_cmd:.3f}) abs_err={vx_err_abs:.3f}"
                          f" vy={vy_actual:.3f}({vy_cmd:.3f}) vyaw={vyaw_actual:.3f}({vyaw_cmd:.3f})"
                          f" [walk_envs={n_walk}/{self.num_envs}]")
                    self.curr_hist = []
                    self._sample_commands(torch.arange(self.num_envs, device=self.device))
                    self._sample_load(torch.arange(self.num_envs, device=self.device))

    return Stage2LocoEnv(EnvCfg())


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_checkpoints(net, stage1_path, arm_path, device):
    """Load V6.2 loco weights into loco_actor (fine-tune) and Stage 2 arm weights (frozen).
    Also builds frozen reference actor for KL penalty.
    Returns: (ref_actor, ref_log_std) — frozen V6.2 policy for KL constraint.
    """
    # 1. Loco actor from V6.2
    print(f"\n[Transfer] Loading V6.2 loco: {stage1_path}")
    ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    s1_state = ckpt.get("model", ckpt)

    # V6.2 keys: actor.X -> loco_actor.X, log_std -> loco_log_std
    transferred = 0
    for s1_key, s1_val in s1_state.items():
        if s1_key.startswith("actor."):
            our_key = "loco_actor." + s1_key[6:]
        elif s1_key == "log_std":
            our_key = "loco_log_std"
        else:
            continue  # Skip critic (fresh init)
        if our_key in net.state_dict() and net.state_dict()[our_key].shape == s1_val.shape:
            net.state_dict()[our_key].copy_(s1_val)
            transferred += 1
        else:
            print(f"  [WARN] Skipped {s1_key} -> {our_key}")
    print(f"  Transferred {transferred} loco actor parameters")
    for key in ["best_reward", "iteration", "curriculum_level"]:
        if key in ckpt:
            print(f"  V6.2 {key}: {ckpt[key]}")

    # 2. Build FROZEN reference actor (V6.2 copy for KL penalty)
    print(f"\n[KL Reference] Building frozen V6.2 reference actor...")
    ref_actor, ref_log_std = build_ref_actor(device)
    ref_transferred = 0
    ref_state = ref_actor.state_dict()
    for s1_key, s1_val in s1_state.items():
        if s1_key.startswith("actor."):
            ref_key = s1_key[6:]  # actor.0.weight -> 0.weight
            if ref_key in ref_state and ref_state[ref_key].shape == s1_val.shape:
                ref_state[ref_key].copy_(s1_val)
                ref_transferred += 1
        elif s1_key == "log_std":
            ref_log_std.copy_(s1_val)
            ref_transferred += 1
    ref_actor.load_state_dict(ref_state)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad = False
    print(f"  Ref actor: {ref_transferred} params loaded, FROZEN")

    # 3. Arm actor from Stage 2
    print(f"\n[Transfer] Loading Stage 2 arm: {arm_path}")
    arm_ckpt = torch.load(arm_path, map_location=device, weights_only=False)
    arm_state = arm_ckpt.get("model", arm_ckpt)

    arm_transferred = 0
    for key, val in arm_state.items():
        # Checkpoint keys: arm_actor.net.X -> our keys: arm_actor.X
        # Also try direct match (arm_actor.X)
        if key.startswith("arm_actor.net."):
            our_key = "arm_actor." + key[len("arm_actor.net."):]
        elif key.startswith("arm_actor.") and key != "arm_actor.log_std":
            our_key = key
        else:
            continue
        if our_key in net.state_dict() and net.state_dict()[our_key].shape == val.shape:
            net.state_dict()[our_key].copy_(val)
            arm_transferred += 1
        else:
            print(f"  [WARN] Arm key skip: {key} -> {our_key} (not found or shape mismatch)")
    print(f"  Transferred {arm_transferred} arm actor parameters")

    # 4. Freeze arm
    net.arm_actor.eval()
    frozen = 0
    for name, p in net.named_parameters():
        if name.startswith("arm_actor."):
            p.requires_grad = False
            frozen += 1
    print(f"  Frozen {frozen} arm parameters")

    # 5. Loco log_std for fine-tune (lower exploration than from-scratch)
    net.loco_log_std.data.fill_(np.log(0.5))
    print(f"  Loco log_std = log(0.5) (fine-tune exploration)")

    # 6. Critic is already fresh Xavier init (from constructor)
    print(f"  Loco critic: FRESH Xavier init")

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_n = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"\n  Trainable: {trainable:,} | Frozen: {frozen_n:,}")
    print(f"  KL coeff: {KL_COEFF}")

    return ref_actor, ref_log_std


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)
    net = DualActorCritic().to(device)

    start_iter = 0
    best_reward = float('-inf')

    ref_actor = None
    ref_log_std = None

    if args_cli.checkpoint:
        print(f"\n[Load] Resuming from {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0) + 1
        best_reward = ckpt.get("best_reward", float('-inf'))
        env.curr_level = min(ckpt.get("curriculum_level", 0), len(CURRICULUM) - 1)
        # Re-freeze arm
        net.arm_actor.eval()
        for name, p in net.named_parameters():
            if name.startswith("arm_actor."):
                p.requires_grad = False
        # Build KL reference from THIS checkpoint (Stage 2 Loco weights)
        # This preserves perturbation-robust walking as the reference baseline
        print(f"  [KL] Building reference from resumed checkpoint (Stage 2 Loco)")
        ref_actor, ref_log_std = build_ref_actor(device)
        ref_state = ref_actor.state_dict()
        for key, val in ckpt["model"].items():
            if key.startswith("loco_actor."):
                ref_key = key[len("loco_actor."):]
                if ref_key in ref_state and ref_state[ref_key].shape == val.shape:
                    ref_state[ref_key].copy_(val)
            elif key == "loco_log_std":
                ref_log_std.copy_(val)
        ref_actor.load_state_dict(ref_state)
        ref_actor.eval()
        for p in ref_actor.parameters():
            p.requires_grad = False
        print(f"  [KL] Reference actor loaded from Stage 2 Loco, KL_COEFF={KL_COEFF}")
        ppo = LocoPPO(net, device, ref_actor=ref_actor, ref_log_std=ref_log_std)
        if "loco_optimizer" in ckpt:
            ppo.opt.load_state_dict(ckpt["loco_optimizer"])
        # CLI curriculum level override (e.g., --curriculum_level 8 to restart at L8)
        if args_cli.curriculum_level is not None:
            forced = min(args_cli.curriculum_level, len(CURRICULUM) - 1)
            print(f"  [Curriculum] Forced level: {env.curr_level} -> {forced}")
            env.curr_level = forced
            env.curr_hist = []
        # Read std from checkpoint for decay schedule
        std_start = np.exp(net.loco_log_std.data.mean().item())
        print(f"  Resumed: iter={start_iter}, R={best_reward:.2f}, Lv={env.curr_level}, std={std_start:.3f}")
    else:
        if args_cli.stage1_checkpoint is None or args_cli.arm_checkpoint is None:
            raise ValueError("--stage1_checkpoint AND --arm_checkpoint required for fresh start")
        ref_actor, ref_log_std = load_checkpoints(net, args_cli.stage1_checkpoint, args_cli.arm_checkpoint, device)
        ppo = LocoPPO(net, device, ref_actor=ref_actor, ref_log_std=ref_log_std)
        std_start = 0.5  # fresh start exploration

    # Logging
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\n[Log] {log_dir}")

    # Rollout buffers (LOCO only — arm is internal perturbation)
    T = 24
    obs_buf = torch.zeros(T, env.num_envs, OBS_DIM, device=device)
    act_buf = torch.zeros(T, env.num_envs, ACT_DIM, device=device)
    rew_buf = torch.zeros(T, env.num_envs, device=device)
    done_buf = torch.zeros(T, env.num_envs, device=device)
    val_buf = torch.zeros(T, env.num_envs, device=device)
    lp_buf = torch.zeros(T, env.num_envs, device=device)

    obs, _ = env.reset()
    obs_t = obs["policy"]
    ep_rewards = torch.zeros(env.num_envs, device=device)
    completed_rewards = []

    print(f"\n{'='*80}")
    print("STARTING STAGE 2 LOCO: PERTURBATION-ROBUST FINE-TUNE")
    print(f"  Loco: FINE-TUNE (66->15), actor_lr=1e-4, critic_lr=3e-4")
    print(f"  Arm: FROZEN perturbation (39->7)")
    print(f"  KL penalty: coeff={KL_COEFF}, ref={'V6.2' if ref_actor is not None else 'DISABLED'}")
    print(f"  Reward: vx={REWARD_WEIGHTS['vx']}, orient={REWARD_WEIGHTS['orientation']}, height={REWARD_WEIGHTS['height']}, yaw_drift={REWARD_WEIGHTS['yaw_drift']}")
    print(f"  Levels: L0-L7 (perturbation) + L8-L9 (drift correction)")
    print(f"  L8+ extra: vyaw_drift=-6.0, smooth=-3.0, vx_precision=+2.0/-2.0")
    print(f"{'='*80}\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        # Std decay: start_std -> 0.2 (linear)
        # On resume, start_std comes from checkpoint (e.g. 0.201), not hardcoded 0.5
        # This prevents resetting a converged std back to 0.5
        remaining_iters = args_cli.max_iterations - start_iter
        if remaining_iters > 0:
            progress_since_start = (iteration - start_iter) / remaining_iters
        else:
            progress_since_start = 1.0
        std = max(std_start - (std_start - 0.20) * progress_since_start, 0.20)
        net.loco_log_std.data.fill_(np.log(std))

        # Collect rollout
        for t in range(T):
            # 1. Arm inference (frozen, deterministic) — runs INSIDE env
            with torch.no_grad():
                arm_obs = env.get_arm_obs()
                # Skip arm inference if arm is frozen for some envs
                arm_act = net.act_arm(arm_obs).clamp(-1.5, 1.5)
                # Frozen envs keep previous action
                arm_act = torch.where(env.arm_frozen.unsqueeze(-1), env.prev_arm_act, arm_act)
                env.prev_arm_act = arm_act.clone()

            # 2. Loco inference (training, stochastic)
            with torch.no_grad():
                loco_action = net.act_loco(obs_t)
                val = net.loco_critic(obs_t).squeeze(-1)
                dist = torch.distributions.Normal(net.loco_actor(obs_t), net.loco_log_std.clamp(-2, 1).exp())
                lp = dist.log_prob(loco_action).sum(-1)

            obs_buf[t] = obs_t
            act_buf[t] = loco_action
            val_buf[t] = val
            lp_buf[t] = lp

            # 3. Step (arm action applied inside _pre_physics_step via prev_arm_act)
            obs_dict, reward, terminated, truncated, _ = env.step(loco_action)
            obs_next = obs_dict["policy"]
            done = (terminated | truncated).float()

            rew_buf[t] = reward
            done_buf[t] = done

            ep_rewards += reward
            done_mask = done.bool()
            if done_mask.any():
                for r_val in ep_rewards[done_mask].cpu().numpy():
                    completed_rewards.append(r_val)
                ep_rewards[done_mask] = 0

            obs_t = obs_next

        # Compute returns
        with torch.no_grad():
            nv = net.loco_critic(obs_t).squeeze(-1)
        adv, ret = ppo.gae(rew_buf, val_buf, done_buf, nv)

        # PPO update (LOCO only)
        losses = ppo.update(
            obs_buf.reshape(-1, OBS_DIM),
            act_buf.reshape(-1, ACT_DIM),
            lp_buf.reshape(-1),
            ret.reshape(-1),
            adv.reshape(-1),
            val_buf.reshape(-1),
        )

        mean_reward = rew_buf.mean().item()
        env.update_curriculum(mean_reward)

        # Tensorboard
        if iteration % 10 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            writer.add_scalar("reward/mean_step", mean_reward, iteration)
            writer.add_scalar("reward/mean_episode", avg_ep, iteration)
            writer.add_scalar("loss/actor", losses["a"], iteration)
            writer.add_scalar("loss/critic", losses["c"], iteration)
            writer.add_scalar("loss/entropy", losses["e"], iteration)
            writer.add_scalar("loss/kl_penalty", losses["kl"], iteration)
            writer.add_scalar("train/lr_actor", losses["lr_a"], iteration)
            writer.add_scalar("train/lr_critic", losses["lr_c"], iteration)
            writer.add_scalar("train/log_std", net.loco_log_std.data.mean().item(), iteration)
            writer.add_scalar("curriculum/level", env.curr_level, iteration)

            height = env.robot.data.root_pos_w[:, 2].mean().item()
            lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
            av_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_ang_vel_w)
            writer.add_scalar("robot/height", height, iteration)
            writer.add_scalar("robot/vx_actual", lv_b[:, 0].mean().item(), iteration)
            writer.add_scalar("robot/vx_cmd", env.vel_cmd[:, 0].mean().item(), iteration)
            writer.add_scalar("robot/vy_actual", lv_b[:, 1].mean().item(), iteration)
            writer.add_scalar("robot/vy_cmd", env.vel_cmd[:, 1].mean().item(), iteration)
            writer.add_scalar("robot/vyaw_actual", av_b[:, 2].mean().item(), iteration)
            writer.add_scalar("robot/vyaw_cmd", env.vel_cmd[:, 2].mean().item(), iteration)
            writer.add_scalar("perturbation/load_mass_avg", env.load_mass.mean().item(), iteration)
            writer.add_scalar("robot/height_cmd_avg", env.height_cmd.mean().item(), iteration)
            # Knee angles for squat tracking
            jp_loco = env.robot.data.joint_pos[:, env.loco_idx]
            avg_knee = (jp_loco[:, env.left_knee_idx].mean().item() + jp_loco[:, env.right_knee_idx].mean().item()) / 2
            writer.add_scalar("robot/knee_avg", avg_knee, iteration)

        # Console log
        if iteration % 50 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
            av_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_ang_vel_w)
            avg_vx = lv_b[:, 0].mean().item()
            cmd_vx = env.vel_cmd[:, 0].mean().item()
            avg_vy = lv_b[:, 1].mean().item()
            cmd_vy = env.vel_cmd[:, 1].mean().item()
            avg_vyaw = av_b[:, 2].mean().item()
            cmd_vyaw = env.vel_cmd[:, 2].mean().item()
            load_avg = env.load_mass.mean().item()
            h_cmd_avg = env.height_cmd.mean().item()

            jp_loco_log = env.robot.data.joint_pos[:, env.loco_idx]
            knee_avg = (jp_loco_log[:, env.left_knee_idx].mean().item() + jp_loco_log[:, env.right_knee_idx].mean().item()) / 2
            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_reward:.2f} EpR={avg_ep:.2f} "
                  f"H={height:.3f}({h_cmd_avg:.2f}) K={knee_avg:.2f} vx={avg_vx:.3f}({cmd_vx:.3f}) "
                  f"vy={avg_vy:.3f}({cmd_vy:.3f}) "
                  f"vyaw={avg_vyaw:.3f}({cmd_vyaw:.3f}) "
                  f"Lv={env.curr_level} load={load_avg:.2f}kg "
                  f"KL={losses['kl']:.3f} "
                  f"std={np.exp(net.loco_log_std.data.mean().item()):.3f}")

        # Save checkpoints
        if iteration % 500 == 0 and iteration > 0:
            path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                "model": net.state_dict(),
                "loco_optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "stage1_checkpoint": args_cli.stage1_checkpoint,
                "arm_checkpoint": args_cli.arm_checkpoint,
            }, path)
            print(f"  [Save] {path}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            path = os.path.join(log_dir, "model_best.pt")
            torch.save({
                "model": net.state_dict(),
                "loco_optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "stage1_checkpoint": args_cli.stage1_checkpoint,
                "arm_checkpoint": args_cli.arm_checkpoint,
            }, path)

    # Final save
    path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": net.state_dict(),
        "loco_optimizer": ppo.opt.state_dict(),
        "iteration": args_cli.max_iterations - 1,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
        "stage1_checkpoint": args_cli.stage1_checkpoint,
        "arm_checkpoint": args_cli.arm_checkpoint,
    }, path)

    print(f"\n{'='*80}")
    print("STAGE 2 LOCO TRAINING COMPLETE")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Log Dir: {log_dir}")
    print(f"{'='*80}")

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
