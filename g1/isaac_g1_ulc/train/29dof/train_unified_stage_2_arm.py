"""
Unified Stage 2: Arm Reaching — Dual Actor-Critic (29DoF)
==========================================================
Loco policy (Stage 1 V6.2) FROZEN, arm policy (right arm 7 joints) trained from scratch.

ARCHITECTURE:
  LocoActor  (66→15) [512,256,128] + LN + ELU  — FROZEN (Stage 1 V6.2 checkpoint)
  LocoCritic (66→1)  [512,256,128] + LN + ELU  — FROZEN
  ArmActor   (39→7)  [256,256,128] + ELU       — FRESH (trained)
  ArmCritic  (39→1)  [256,256,128] + ELU       — FRESH (trained)

ARM OBS (39 dim):
  arm_joint_pos_right(7) + arm_joint_vel_right(7) + ee_pos_body(3) + ee_quat(4)
  + target_pos_body(3) + target_orient(3) + pos_error(3) + orient_err(1)
  + prev_arm_action(7) + steps_norm(1) = 39

ARM ACTION (7):
  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
  right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw

ANTI-GAMING (from Stage 7):
  1. Absolute-only target sampling + min distance enforcement
  2. 3-condition reach validation (position + displacement + time)
  3. Validated reach rate for curriculum advancement
  4. Movement-centric rewards (velocity_toward, progress, stillness_penalty)
  5. Strict 8-level curriculum (3 phases)

CURRICULUM:
  Phase 1 (Level 0-3): Standing + Reaching (vx=0)
  Phase 2 (Level 4-5): Walking + Reaching
  Phase 3 (Level 6-7): Walking + Orientation

USAGE:
    # From Stage 1 V6.2 checkpoint (fresh arm):
    isaaclab.bat -p ... --stage1_checkpoint logs/ulc/.../model_best.pt --num_envs 4096 --headless
    # Resume from Stage 2 checkpoint:
    isaaclab.bat -p ... --checkpoint logs/ulc/.../model_best.pt --num_envs 4096 --headless
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
ARM_ACTION_SCALE = _cfg_mod.ARM_ACTION_SCALE
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS
SHOULDER_OFFSET_RIGHT = _cfg_mod.SHOULDER_OFFSET_RIGHT

# ============================================================================
# DIMENSIONS
# ============================================================================

LOCO_OBS_DIM = 66   # Stage 1 V6.2: lin_vel(3)+ang_vel(3)+gravity(3)+leg_pos(12)+leg_vel(12)
                     #   +waist_pos(3)+waist_vel(3)+height_cmd(1)+vel_cmd(3)+gait_phase(2)
                     #   +prev_act(15)+torso_euler(3)+torso_cmd(3)
LOCO_ACT_DIM = 15   # 12 leg + 3 waist
ARM_OBS_DIM = 39     # arm_pos(7)+arm_vel(7)+ee_pos(3)+ee_quat(4)+target_pos(3)
                     #   +target_orient(3)+pos_err(3)+orient_err(1)+prev_arm(7)+steps_norm(1)
ARM_ACT_DIM = 7      # right arm 7 joints
COMBINED_ACT_DIM = LOCO_ACT_DIM + ARM_ACT_DIM  # 22

# ============================================================================
# ARM REWARD WEIGHTS — Movement-centric (anti-gaming)
# ============================================================================

ARM_REWARD_WEIGHTS = {
    "reach": 10.0,                # exp(-8.0 * dist) — primary proximity
    "velocity_toward": 6.0,       # dot(ee_vel, direction_to_target)
    "progress": 5.0,              # (initial_dist - current_dist) / initial_dist
    "reach_bonus": 1.0,           # +10.0 per validated reach (sparse)
    "smooth": 1.0,                # action smoothness (only when close)
    "orient": 3.0,                # palm orientation (Level 6+ only)
    "left_arm_dev": -2.0,         # left arm deviation from default
    "stillness_penalty": -2.0,    # penalize stillness when far
    "height": 2.0,                # height stability
    "tilt": 1.5,                  # tilt penalty
    "action_rate": -0.05,
    "jerk": -0.03,
    "alive": 0.3,
}

# ============================================================================
# ANTI-GAMING CURRICULUM (8 levels, 3 phases)
# ============================================================================

CURRICULUM = [
    # === PHASE 1: STANDING + REACHING (Level 0-3) ===
    {
        "description": "L0: Standing + easy reach",
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "pos_threshold": 0.10,
        "min_target_distance": 0.10,
        "min_displacement": 0.03,
        "max_reach_steps": 200,
        "validated_reach_rate": 0.20,
        "min_validated_reaches": 3000,
        "min_steps": 1500,
        "use_orientation": False,
        "workspace_radius": (0.08, 0.15),
    },
    {
        "description": "L1: Standing + medium reach",
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "pos_threshold": 0.08,
        "min_target_distance": 0.12,
        "min_displacement": 0.04,
        "max_reach_steps": 180,
        "validated_reach_rate": 0.20,
        "min_validated_reaches": 3500,
        "min_steps": 2000,
        "use_orientation": False,
        "workspace_radius": (0.10, 0.20),
    },
    {
        "description": "L2: Standing + far reach",
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "pos_threshold": 0.07,
        "min_target_distance": 0.14,
        "min_displacement": 0.05,
        "max_reach_steps": 170,
        "validated_reach_rate": 0.20,
        "min_validated_reaches": 4000,
        "min_steps": 2500,
        "use_orientation": False,
        "workspace_radius": (0.12, 0.25),
    },
    {
        "description": "L3: Standing + full workspace",
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "pos_threshold": 0.06,
        "min_target_distance": 0.15,
        "min_displacement": 0.06,
        "max_reach_steps": 160,
        "validated_reach_rate": 0.25,
        "min_validated_reaches": 4500,
        "min_steps": 3000,
        "use_orientation": False,
        "workspace_radius": (0.15, 0.30),
    },
    # === PHASE 2: WALKING + REACHING (Level 4-5) ===
    {
        "description": "L4: Slow walk + reach",
        "vx": (0.0, 0.25), "vy": (-0.05, 0.05), "vyaw": (-0.10, 0.10),
        "pos_threshold": 0.06,
        "min_target_distance": 0.15,
        "min_displacement": 0.06,
        "max_reach_steps": 160,
        "validated_reach_rate": 0.25,
        "min_validated_reaches": 4500,
        "min_steps": 3000,
        "use_orientation": False,
        "workspace_radius": (0.15, 0.30),
    },
    {
        "description": "L5: Medium walk + reach",
        "vx": (0.0, 0.40), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.05,
        "min_target_distance": 0.16,
        "min_displacement": 0.07,
        "max_reach_steps": 150,
        "validated_reach_rate": 0.25,
        "min_validated_reaches": 5000,
        "min_steps": 3500,
        "use_orientation": False,
        "workspace_radius": (0.15, 0.35),
    },
    # === PHASE 3: WALKING + ORIENTATION (Level 6-7) ===
    {
        "description": "L6: Walk + palm_down orientation",
        "vx": (0.0, 0.40), "vy": (-0.08, 0.08), "vyaw": (-0.14, 0.14),
        "pos_threshold": 0.05,
        "min_target_distance": 0.16,
        "min_displacement": 0.07,
        "max_reach_steps": 150,
        "validated_reach_rate": 0.20,
        "min_validated_reaches": 5000,
        "min_steps": 3500,
        "orient_threshold": 2.0,
        "use_orientation": True,
        "workspace_radius": (0.15, 0.35),
    },
    {
        "description": "L7: FINAL — fast walk + orientation",
        "vx": (0.0, 0.50), "vy": (-0.10, 0.10), "vyaw": (-0.16, 0.16),
        "pos_threshold": 0.04,
        "min_target_distance": 0.18,
        "min_displacement": 0.08,
        "max_reach_steps": 150,
        "validated_reach_rate": None,
        "min_validated_reaches": None,
        "min_steps": None,
        "orient_threshold": 1.5,
        "use_orientation": True,
        "workspace_radius": (0.18, 0.40),
    },
]

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Stage 2: Arm Reaching (Dual AC, 29DoF)")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage 1 V6.2 checkpoint (loco weights frozen). Required for fresh start.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 2 checkpoint")
    parser.add_argument("--experiment_name", type=str, default="g1_stage2_arm")
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from torch.utils.tensorboard import SummaryWriter

print("=" * 80)
print("UNIFIED STAGE 2: ARM REACHING (Dual AC, 29DoF)")
print(f"USD: {G1_29DOF_USD}")
print(f"Loco: {LOCO_OBS_DIM} obs -> {LOCO_ACT_DIM} act (FROZEN)")
print(f"Arm:  {ARM_OBS_DIM} obs -> {ARM_ACT_DIM} act (FRESH)")
print(f"Combined action: {COMBINED_ACT_DIM}")
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
# DUAL ACTOR-CRITIC NETWORKS
# ============================================================================

class LocoActorCritic(nn.Module):
    """Locomotion AC: 66 obs -> 15 act (FROZEN, from Stage 1 V6.2).
    Same architecture as ActorCritic in train_unified_stage_1.py.
    """
    def __init__(self, num_obs=LOCO_OBS_DIM, num_act=LOCO_ACT_DIM, hidden=[512, 256, 128]):
        super().__init__()
        # Actor
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        # Critic
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

    def forward_actor(self, x):
        return self.actor(x)

    def forward_critic(self, x):
        return self.critic(x).squeeze(-1)


class ArmActor(nn.Module):
    """Arm policy: 39 obs -> 7 act (FRESH, no LayerNorm)"""
    def __init__(self, num_obs=ARM_OBS_DIM, num_act=ARM_ACT_DIM, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        return self.net(x)


class ArmCritic(nn.Module):
    """Arm value: 39 obs -> 1 value (FRESH, no LayerNorm)"""
    def __init__(self, num_obs=ARM_OBS_DIM, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualActorCritic(nn.Module):
    """Dual AC — Loco (FROZEN) + Arm (FRESH)"""
    def __init__(self):
        super().__init__()
        self.loco_ac = LocoActorCritic()
        self.arm_actor = ArmActor()
        self.arm_critic = ArmCritic()

    def get_loco_action(self, loco_obs, deterministic=True):
        """Loco is always deterministic (frozen)."""
        return self.loco_ac.forward_actor(loco_obs)

    def get_arm_action(self, arm_obs, deterministic=False):
        """Arm action with optional stochastic exploration."""
        mean = self.arm_actor(arm_obs)
        if deterministic:
            return mean, torch.zeros(arm_obs.shape[0], device=arm_obs.device)
        std = self.arm_actor.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp

    def evaluate_arm(self, arm_obs, arm_actions):
        mean = self.arm_actor(arm_obs)
        val = self.arm_critic(arm_obs)
        std = self.arm_actor.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(arm_actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return val, logp, ent


# ============================================================================
# ARM PPO (only updates arm parameters)
# ============================================================================

class ArmPPO:
    def __init__(self, net, device, lr=1e-3, max_iter=20000):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(
            list(net.arm_actor.parameters()) + list(net.arm_critic.parameters()),
            lr=lr, weight_decay=1e-5
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, max_iter, eta_min=1e-5)

    def gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        adv = torch.zeros_like(rewards)
        last = 0
        for t in reversed(range(len(rewards))):
            nxt = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
            adv[t] = last = delta + gamma * lam * (1 - dones[t]) * last
        return adv, adv + values

    def update(self, obs, act, old_lp, ret, adv, old_v):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tot_a, tot_c, tot_e, n = 0, 0, 0, 0
        bs = obs.shape[0]

        for _ in range(5):
            idx = torch.randperm(bs, device=self.device)
            for i in range(0, bs, 4096):
                mb = idx[i:i + 4096]
                val, lp, ent = self.net.evaluate_arm(obs[mb], act[mb])
                ratio = (lp - old_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(0.8, 1.2) * adv[mb]
                a_loss = -torch.min(s1, s2).mean()
                v_clip = old_v[mb] + (val - old_v[mb]).clamp(-0.2, 0.2)
                c_loss = 0.5 * torch.max((val - ret[mb]) ** 2, (v_clip - ret[mb]) ** 2).mean()
                loss = a_loss + 0.5 * c_loss - 0.01 * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.net.arm_actor.parameters()) + list(self.net.arm_critic.parameters()),
                    0.5)
                self.opt.step()
                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                n += 1

        self.sched.step()
        return {"a": tot_a / max(n, 1), "c": tot_c / max(n, 1),
                "e": tot_e / max(n, 1), "lr": self.sched.get_last_lr()[0]}


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
        episode_length_s = 20.0
        action_space = COMBINED_ACT_DIM
        observation_space = LOCO_OBS_DIM  # _get_observations returns loco_obs
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=5.0)

    class Stage2ArmEnv(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            jn = self.robot.joint_names
            print(f"\n[Stage2] Robot joints ({len(jn)}): {jn}")

            # Loco joint indices (15 joints)
            self.loco_idx = []
            for name in LOCO_JOINT_NAMES:
                if name in jn:
                    self.loco_idx.append(jn.index(name))
                else:
                    print(f"  [WARN] Loco joint not found: {name}")
            self.loco_idx = torch.tensor(self.loco_idx, device=self.device)
            print(f"  Loco joints: {len(self.loco_idx)} / {NUM_LOCO_JOINTS}")

            # Right arm joint indices (7 joints)
            self.right_arm_idx = []
            for name in ARM_JOINT_NAMES_RIGHT:
                if name in jn:
                    self.right_arm_idx.append(jn.index(name))
                else:
                    print(f"  [WARN] Right arm joint not found: {name}")
            self.right_arm_idx = torch.tensor(self.right_arm_idx, device=self.device)
            print(f"  Right arm joints: {len(self.right_arm_idx)} / 7")

            # Left arm joint indices (7 joints — held at default)
            self.left_arm_idx = []
            for name in ARM_JOINT_NAMES_LEFT:
                if name in jn:
                    self.left_arm_idx.append(jn.index(name))
            self.left_arm_idx = torch.tensor(self.left_arm_idx, device=self.device)
            print(f"  Left arm joints: {len(self.left_arm_idx)} / 7")

            # Hand joint indices (held open)
            self.hand_idx = []
            for name in HAND_JOINT_NAMES:
                if name in jn:
                    self.hand_idx.append(jn.index(name))
            self.hand_idx = torch.tensor(self.hand_idx, device=self.device)
            print(f"  Hand joints: {len(self.hand_idx)} / {NUM_HAND_JOINTS}")

            # Per-joint loco indices for reward/termination
            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")
            hip_yaw_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
            self.hip_yaw_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_yaw_names], device=self.device)

            # Find palm body for EE computation
            body_names = self.robot.data.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "palm" in name.lower():
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                # Fallback: last body
                self.palm_idx = len(body_names) - 1
                print(f"  [WARN] right_palm not found! Using body {self.palm_idx}")
            else:
                print(f"  Palm body: '{body_names[self.palm_idx]}' (idx={self.palm_idx})")

            # Default poses
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)

            # Right arm defaults (7 values)
            self.default_right_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_RIGHT],
                device=self.device, dtype=torch.float32)
            # Left arm defaults (7 values)
            self.default_left_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_LEFT],
                device=self.device, dtype=torch.float32)

            # Action scales
            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.loco_action_scales = torch.tensor(
                leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # Shoulder offset for target sampling
            self.shoulder_offset = torch.tensor(
                SHOULDER_OFFSET_RIGHT, device=self.device, dtype=torch.float32)

            # State buffers
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_loco_act = torch.zeros(self.num_envs, LOCO_ACT_DIM, device=self.device)
            self.prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)
            self._prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)

            # Target state
            self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_orient = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_orient[:, 2] = -1.0  # Default: palm down

            # EE tracking
            self.prev_ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

            # Anti-gaming state
            self.ee_pos_at_spawn = torch.zeros(self.num_envs, 3, device=self.device)
            self.steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.initial_dist = torch.zeros(self.num_envs, device=self.device)

            # Curriculum tracking
            self.curr_level = 0
            self.validated_reaches = 0
            self.timed_out_targets = 0
            self.total_attempts = 0
            self.stage_validated_reaches = 0
            self.stage_timed_out = 0
            self.stage_steps = 0
            self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            # Reward storage
            self.arm_reward = torch.zeros(self.num_envs, device=self.device)
            self.arm_reward_components = {}

            # Push timer (loco perturbation — minimal during arm training)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.push_timer = torch.randint(300, 600, (self.num_envs,), device=self.device)

            # Command resample timer
            self.cmd_timer = torch.randint(0, 400, (self.num_envs,), device=self.device)
            self.cmd_resample_targets = torch.randint(150, 401, (self.num_envs,), device=self.device)

            # ContactSensor for illegal contact termination
            self._contact_sensor = self.scene["contact_forces"]
            self._illegal_contact_ids, illegal_names = self._contact_sensor.find_bodies(
                "pelvis|torso_link|.*knee_link")
            print(f"  Illegal contact bodies: {illegal_names}")

            # Visual markers
            self._markers_initialized = False

            # Initial target sample
            self._sample_targets(torch.arange(self.num_envs, device=self.device))
            self._sample_vel_commands(torch.arange(self.num_envs, device=self.device))

            print(f"\n[Stage2] {self.num_envs} envs, Level {self.curr_level}")
            print(f"  Loco obs: {LOCO_OBS_DIM}, Arm obs: {ARM_OBS_DIM}")
            print(f"  Loco act: {LOCO_ACT_DIM}, Arm act: {ARM_ACT_DIM}")

        @property
        def robot(self):
            return self.scene["robot"]

        # ================================================================
        # TARGET SAMPLING
        # ================================================================

        def _sample_vel_commands(self, env_ids):
            """Sample velocity commands from current curriculum level."""
            lv = CURRICULUM[self.curr_level]
            n = len(env_ids)
            self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*lv["vx"])
            self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*lv["vy"])
            self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*lv["vyaw"])

        def _sample_targets(self, env_ids):
            """ANTI-GAMING: Absolute spherical sampling + min distance enforcement."""
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]
            self.already_reached[env_ids] = False

            # Current EE position in body frame
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]
            ee_w, _ = self._compute_ee()
            ee_w = ee_w[env_ids]
            current_ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            # Spherical sampling from shoulder offset
            workspace_r = lv.get("workspace_radius", (0.10, 0.30))
            azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
            elevation = torch.empty(n, device=self.device).uniform_(-0.3, 0.5)
            radius = torch.empty(n, device=self.device).uniform_(*workspace_r)

            target_x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
            target_y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
            target_z = radius * torch.sin(elevation) + self.shoulder_offset[2]

            # Workspace clamp (body frame, right side)
            target_x = target_x.clamp(0.0, 0.45)
            target_y = target_y.clamp(-0.50, -0.05)
            target_z = target_z.clamp(-0.10, 0.50)
            target_body = torch.stack([target_x, target_y, target_z], dim=-1)

            # Min distance enforcement
            min_dist = lv["min_target_distance"]
            dist_to_ee = torch.norm(target_body - current_ee_body, dim=-1)
            too_close = dist_to_ee < min_dist
            if too_close.any():
                direction = target_body - current_ee_body
                direction = direction / (direction.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                pushed = current_ee_body + min_dist * direction
                pushed[:, 0] = pushed[:, 0].clamp(0.0, 0.45)
                pushed[:, 1] = pushed[:, 1].clamp(-0.50, -0.05)
                pushed[:, 2] = pushed[:, 2].clamp(-0.10, 0.50)
                target_body = torch.where(too_close.unsqueeze(-1).expand_as(target_body),
                                          pushed, target_body)

            self.target_pos_body[env_ids] = target_body

            # Target orientation (palm down default, variable in Phase 3)
            self.target_orient[env_ids, 0] = 0.0
            self.target_orient[env_ids, 1] = 0.0
            self.target_orient[env_ids, 2] = -1.0

            # Record spawn state for anti-gaming validation
            self.ee_pos_at_spawn[env_ids] = current_ee_body
            self.steps_since_spawn[env_ids] = 0
            self.initial_dist[env_ids] = torch.norm(
                target_body - current_ee_body, dim=-1).clamp(min=0.01)

        # ================================================================
        # EE COMPUTATION
        # ================================================================

        def _compute_ee(self):
            """Compute EE position (palm + 2cm forward offset) and quaternion."""
            palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
            palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
            fwd = get_palm_forward(palm_quat)
            ee_pos = palm_pos + 0.02 * fwd
            return ee_pos, palm_quat

        # ================================================================
        # OBSERVATIONS
        # ================================================================

        def get_loco_obs(self):
            """66-dim loco obs — IDENTICAL to Stage 1 V6.2."""
            r = self.robot
            q = r.data.root_quat_w
            lv_b = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av_b = quat_apply_inverse(q, r.data.root_ang_vel_w)
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
                lv_b,                          # 3
                av_b,                          # 3
                g,                             # 3
                jp_leg,                        # 12
                jv_leg,                        # 12
                jp_waist,                      # 3
                jv_waist,                      # 3
                self.height_cmd[:, None],      # 1
                self.vel_cmd,                  # 3
                gait,                          # 2
                self.prev_loco_act,            # 15
                torso_euler,                   # 3
                self.torso_cmd,                # 3
            ], dim=-1)  # = 66
            return obs.clamp(-10, 10).nan_to_num()

        def get_arm_obs(self):
            """39-dim arm obs."""
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w
            lv = CURRICULUM[self.curr_level]

            # Right arm joint state (7 each)
            arm_pos = r.data.joint_pos[:, self.right_arm_idx]
            arm_vel = r.data.joint_vel[:, self.right_arm_idx] * 0.1

            # EE position in body frame
            ee_w, palm_quat = self._compute_ee()
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            # Position error
            pos_error = self.target_pos_body - ee_body

            # Orientation error (scalar, 0-pi range, normalized)
            orient_err = compute_orientation_error(palm_quat, self.target_orient)
            orient_err_norm = orient_err.unsqueeze(-1) / np.pi  # 0-1

            # Steps since spawn (normalized, anti-gaming)
            max_steps = float(lv["max_reach_steps"])
            steps_norm = (self.steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)

            obs = torch.cat([
                arm_pos,                       # 7: right arm joint positions
                arm_vel,                       # 7: right arm joint velocities (* 0.1)
                ee_body,                       # 3: EE position in body frame
                palm_quat,                     # 4: EE quaternion (wxyz)
                self.target_pos_body,          # 3: target position in body frame
                self.target_orient,            # 3: target orientation direction
                pos_error,                     # 3: target - ee_pos
                orient_err_norm,               # 1: orientation error (normalized)
                self.prev_arm_act,             # 7: previous arm action
                steps_norm,                    # 1: steps since spawn (normalized)
            ], dim=-1)  # = 39
            return obs.clamp(-10, 10).nan_to_num()

        # ================================================================
        # PRE-PHYSICS STEP
        # ================================================================

        def _pre_physics_step(self, combined_act):
            self.actions = combined_act.clone()
            loco_act = combined_act[:, :LOCO_ACT_DIM]
            arm_act = combined_act[:, LOCO_ACT_DIM:]

            tgt = self.robot.data.default_joint_pos.clone()

            # Loco (frozen policy output)
            tgt[:, self.loco_idx] = self.default_loco + loco_act * self.loco_action_scales

            # Waist clamp (Stage 1 V6.2)
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)

            # Hip yaw clamp (Stage 1 V6.2)
            for hy_idx in self.hip_yaw_loco_idx:
                tgt[:, self.loco_idx[hy_idx]].clamp_(-0.3, 0.3)

            # RIGHT ARM (residual control)
            tgt[:, self.right_arm_idx] = self.default_right_arm + arm_act * ARM_ACTION_SCALE

            # LEFT ARM (default pose)
            tgt[:, self.left_arm_idx] = self.default_left_arm

            # HANDS (open)
            tgt[:, self.hand_idx] = self.default_hand

            self.robot.set_joint_position_target(tgt)

            # Update phase and actions
            self._prev_arm_act = self.prev_arm_act.clone()
            self.prev_loco_act = loco_act.clone()
            self.prev_arm_act = arm_act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            # Increment step counters
            self.steps_since_spawn += 1
            self.step_count += 1
            self.stage_steps += 1

            # Reach validation + resample
            self._validate_reaches()

            # Velocity command resample
            self.cmd_timer += 1
            cmd_mask = self.cmd_timer >= self.cmd_resample_targets
            if cmd_mask.any():
                cmd_ids = cmd_mask.nonzero(as_tuple=False).squeeze(-1)
                self._sample_vel_commands(cmd_ids)
                self.cmd_timer[cmd_ids] = 0
                self.cmd_resample_targets[cmd_ids] = torch.randint(150, 401, (len(cmd_ids),), device=self.device)

            # Push perturbation (minimal for arm training)
            self._apply_push()

            # Update EE tracking
            ee_w, _ = self._compute_ee()
            self.prev_ee_pos_w = ee_w.clone()

        def _apply_action(self):
            pass

        # ================================================================
        # REACH VALIDATION (3-condition anti-gaming)
        # ================================================================

        def _validate_reaches(self):
            lv = CURRICULUM[self.curr_level]
            ee_w, palm_quat = self._compute_ee()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)
            target_w = root_pos + quat_apply(root_quat, self.target_pos_body)
            dist = torch.norm(ee_w - target_w, dim=-1)

            # Condition 1: position threshold
            pos_reached = dist < lv["pos_threshold"]
            if lv["use_orientation"]:
                orient_err = compute_orientation_error(palm_quat, self.target_orient)
                orient_thresh = lv.get("orient_threshold", 2.0)
                pos_reached = pos_reached & (orient_err < orient_thresh)

            # Condition 2: arm actually moved
            ee_disp = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1)
            moved_enough = ee_disp >= lv["min_displacement"]

            # Condition 3: within time limit
            within_time = self.steps_since_spawn <= lv["max_reach_steps"]

            # All 3 conditions
            validated = pos_reached & moved_enough & within_time
            new_reaches = validated & ~self.already_reached

            if new_reaches.any():
                reached_ids = torch.where(new_reaches)[0]
                self.validated_reaches += len(reached_ids)
                self.stage_validated_reaches += len(reached_ids)
                self.total_attempts += len(reached_ids)
                self.already_reached[reached_ids] = True
                self._sample_targets(reached_ids)

            # Handle timed-out targets
            timed_out = (self.steps_since_spawn > lv["max_reach_steps"]) & ~self.already_reached
            if timed_out.any():
                to_ids = torch.where(timed_out)[0]
                self.timed_out_targets += len(to_ids)
                self.stage_timed_out += len(to_ids)
                self.total_attempts += len(to_ids)
                self._sample_targets(to_ids)

            # Visual markers (only when rendering)
            self._init_markers()
            default_quat = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
            self.target_markers.visualize(translations=target_w, orientations=default_quat)
            self.ee_markers.visualize(translations=ee_w, orientations=default_quat)

        def _init_markers(self):
            if self._markers_initialized:
                return
            self.target_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TargetMarkers",
                    markers={"sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)))},
                ))
            self.ee_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/EEMarkers",
                    markers={"sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
                ))
            self._markers_initialized = True

        # ================================================================
        # PUSH PERTURBATION
        # ================================================================

        def _apply_push(self):
            # Only apply push when walking (Phase 2+)
            if self.curr_level < 4:
                return
            push_mask = self.step_count >= self.push_timer
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                force = torch.zeros(n, 3, device=self.device)
                force[:, :2] = torch.randn(n, 2, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                mag = torch.rand(n, 1, device=self.device) * 15.0  # 0-15N
                forces[ids, 0] = force * mag
                self.push_timer[ids] = torch.randint(200, 500, (n,), device=self.device)
                self.step_count[ids] = 0
            self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        # ================================================================
        # REWARDS
        # ================================================================

        def _get_observations(self):
            return {"policy": self.get_loco_obs()}

        def compute_arm_reward(self):
            """Movement-centric arm reward with anti-gaming signals."""
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w
            lv = CURRICULUM[self.curr_level]

            ee_w, palm_quat = self._compute_ee()
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)
            target_w = root_pos + quat_apply(root_quat, self.target_pos_body)
            dist = torch.norm(ee_w - target_w, dim=-1)

            # EE velocity
            ee_vel_w = (ee_w - self.prev_ee_pos_w) / 0.02
            ee_vel_b = quat_apply_inverse(root_quat, ee_vel_w)
            ee_speed = ee_vel_b.norm(dim=-1)

            # Direction to target
            direction = self.target_pos_body - ee_body
            dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dir_unit = direction / dir_norm

            # 1. Reach proximity (exp-decay)
            r_reach = torch.exp(-8.0 * dist)

            # 2. Velocity toward target
            vel_toward = (ee_vel_b * dir_unit).sum(dim=-1)
            r_velocity = vel_toward.clamp(-0.5, 1.0)

            # 3. Progress (fraction of initial distance closed)
            r_progress = ((self.initial_dist - dist) / self.initial_dist.clamp(min=0.01)).clamp(-0.5, 1.0)

            # 4. Reach bonus (sparse, per validated reach — computed in _validate_reaches)
            # Not per-step, handled via curriculum advancement incentive

            # 5. Action smoothness (only when close)
            arm_diff = self.prev_arm_act - self._prev_arm_act
            close_mask = (dist < 0.15).float()
            r_smooth = -(arm_diff ** 2).sum(-1) * close_mask

            # 6. Orientation (Level 6+ only)
            orient_err = compute_orientation_error(palm_quat, self.target_orient)
            if lv["use_orientation"]:
                r_orient = torch.exp(-3.0 * orient_err)
            else:
                r_orient = torch.zeros(self.num_envs, device=self.device)

            # 7. Left arm deviation penalty
            left_arm_pos = r.data.joint_pos[:, self.left_arm_idx]
            left_dev = (left_arm_pos - self.default_left_arm).abs().sum(-1)
            r_left_dev = left_dev

            # 8. Stillness penalty (when far from target)
            far_mask = (dist > 0.10).float()
            r_stillness = torch.exp(-5.0 * ee_speed) * far_mask

            # 9. Height stability
            height = root_pos[:, 2]
            r_height = torch.exp(-10.0 * (height - HEIGHT_DEFAULT) ** 2)

            # 10. Tilt penalty
            g = quat_apply_inverse(root_quat, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))
            tilt = torch.asin(torch.clamp((g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))
            r_tilt = torch.exp(-3.0 * tilt)

            # 11. Action rate + jerk
            r_action_rate = (arm_diff ** 2).sum(-1)
            # Simple jerk (no _prev_prev needed — keep it simple)
            r_jerk = r_action_rate  # Approximate

            # Store components for logging
            self.arm_reward_components = {
                "reach": (ARM_REWARD_WEIGHTS["reach"] * r_reach).mean().item(),
                "velocity": (ARM_REWARD_WEIGHTS["velocity_toward"] * r_velocity).mean().item(),
                "progress": (ARM_REWARD_WEIGHTS["progress"] * r_progress).mean().item(),
                "smooth": (ARM_REWARD_WEIGHTS["smooth"] * r_smooth).mean().item(),
                "orient": (ARM_REWARD_WEIGHTS["orient"] * r_orient).mean().item(),
                "left_dev": (ARM_REWARD_WEIGHTS["left_arm_dev"] * r_left_dev).mean().item(),
                "stillness": (ARM_REWARD_WEIGHTS["stillness_penalty"] * r_stillness).mean().item(),
                "height": (ARM_REWARD_WEIGHTS["height"] * r_height).mean().item(),
                "tilt": (ARM_REWARD_WEIGHTS["tilt"] * r_tilt).mean().item(),
                "ee_dist": dist.mean().item(),
                "ee_speed": ee_speed.mean().item(),
                "orient_err": orient_err.mean().item(),
            }

            reward = (
                ARM_REWARD_WEIGHTS["reach"] * r_reach
                + ARM_REWARD_WEIGHTS["velocity_toward"] * r_velocity
                + ARM_REWARD_WEIGHTS["progress"] * r_progress
                + ARM_REWARD_WEIGHTS["smooth"] * r_smooth
                + ARM_REWARD_WEIGHTS["orient"] * r_orient
                + ARM_REWARD_WEIGHTS["left_arm_dev"] * r_left_dev
                + ARM_REWARD_WEIGHTS["stillness_penalty"] * r_stillness
                + ARM_REWARD_WEIGHTS["height"] * r_height
                + ARM_REWARD_WEIGHTS["tilt"] * r_tilt
                + ARM_REWARD_WEIGHTS["action_rate"] * r_action_rate
                + ARM_REWARD_WEIGHTS["jerk"] * r_jerk
                + ARM_REWARD_WEIGHTS["alive"]
            )
            return reward.clamp(-10, 30)

        def _get_rewards(self):
            self.arm_reward = self.compute_arm_reward()
            return self.arm_reward

        # ================================================================
        # TERMINATION (Stage 1 V6.2 — all terminations preserved)
        # ================================================================

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_g = quat_apply_inverse(q, gvec)

            # Height
            fallen = (pos[:, 2] < 0.55) | (pos[:, 2] > 1.2)
            bad_orient = proj_g[:, :2].abs().max(dim=-1)[0] > 0.7

            # Knee limits
            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_bad = (lk < -0.05) | (rk < -0.05) | (lk > 1.5) | (rk > 1.5)

            # Waist limits
            waist_pitch = jp[:, 14]
            waist_roll = jp[:, 13]
            waist_bad = (waist_pitch.abs() > 0.35) | (waist_roll.abs() > 0.25)

            # Hip yaw limits
            hip_yaw_L = jp[:, self.hip_yaw_loco_idx[0]]
            hip_yaw_R = jp[:, self.hip_yaw_loco_idx[1]]
            hip_yaw_bad = (hip_yaw_L.abs() > 0.6) | (hip_yaw_R.abs() > 0.6)

            # Illegal contact
            if len(self._illegal_contact_ids) > 0:
                net_forces = self._contact_sensor.data.net_forces_w_history
                illegal_contact = torch.any(
                    torch.max(
                        torch.norm(net_forces[:, :, self._illegal_contact_ids], dim=-1), dim=1
                    )[0] > 1.0, dim=1)
            else:
                illegal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            terminated = fallen | bad_orient | knee_bad | waist_bad | hip_yaw_bad | illegal_contact
            truncated = self.episode_length_buf >= self.max_episode_length
            return terminated, truncated

        # ================================================================
        # RESET
        # ================================================================

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0:
                return
            n = len(env_ids)

            # Reset joint state
            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            noise = torch.randn_like(default_pos) * 0.02
            self.robot.write_joint_state_to_sim(
                default_pos + noise,
                torch.zeros_like(default_pos), None, env_ids)

            # Reset root state (wxyz quaternion)
            root_pos = torch.tensor([[0.0, 0.0, HEIGHT_DEFAULT]], device=self.device).expand(n, -1).clone()
            root_pos[:, :2] += torch.randn(n, 2, device=self.device) * 0.05
            yaw = torch.randn(n, device=self.device) * 0.1
            qw = torch.cos(yaw / 2)
            root_quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw),
                                      torch.sin(yaw / 2)], dim=-1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            # Reset state buffers
            self.prev_loco_act[env_ids] = 0
            self.prev_arm_act[env_ids] = 0
            self._prev_arm_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)
            self.prev_ee_pos_w[env_ids] = 0
            self.already_reached[env_ids] = False
            self.step_count[env_ids] = 0
            self.push_timer[env_ids] = torch.randint(300, 600, (n,), device=self.device)

            # Resample commands and targets
            self._sample_vel_commands(env_ids)
            self._sample_targets(env_ids)
            self.cmd_timer[env_ids] = torch.randint(0, 400, (n,), device=self.device)
            self.cmd_resample_targets[env_ids] = torch.randint(150, 401, (n,), device=self.device)

        # ================================================================
        # CURRICULUM
        # ================================================================

        def update_curriculum(self, mean_reward):
            lv = CURRICULUM[self.curr_level]
            if lv["min_validated_reaches"] is None:
                return  # Final level

            if self.stage_steps < lv.get("min_steps", 0):
                return

            stage_attempts = self.stage_validated_reaches + self.stage_timed_out
            if stage_attempts == 0:
                return

            # Gaming detection
            if stage_attempts > 100:
                timeout_ratio = self.stage_timed_out / stage_attempts
                if timeout_ratio > 0.90:
                    if self.stage_steps % 500 == 0:
                        print(f"  [GAMING] Level {self.curr_level}: timeout {timeout_ratio:.1%} > 90%")
                    return

            if self.stage_validated_reaches < lv["min_validated_reaches"]:
                return

            validated_rate = self.stage_validated_reaches / stage_attempts
            if validated_rate >= lv["validated_reach_rate"]:
                if self.curr_level < len(CURRICULUM) - 1:
                    self.curr_level += 1
                    new_lv = CURRICULUM[self.curr_level]
                    phase = "STAND+REACH" if self.curr_level < 4 else (
                        "WALK+REACH" if self.curr_level < 6 else "WALK+ORIENT")
                    print(f"\n{'='*60}")
                    print(f"  LEVEL UP! Level {self.curr_level}: {new_lv['description']} ({phase})")
                    print(f"  Validated: {self.stage_validated_reaches}, Rate: {validated_rate:.1%}")
                    print(f"  Timed out: {self.stage_timed_out}")
                    print(f"{'='*60}\n")
                    self.stage_validated_reaches = 0
                    self.stage_timed_out = 0
                    self.stage_steps = 0
                    # Resample all targets for new level
                    self._sample_targets(torch.arange(self.num_envs, device=self.device))
                    self._sample_vel_commands(torch.arange(self.num_envs, device=self.device))

    return Stage2ArmEnv(EnvCfg())


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_stage1_and_setup(net, stage1_path, device):
    """Load Stage 1 V6.2 checkpoint into loco_ac, freeze it. Arm stays fresh."""
    print(f"\n[Transfer] Loading Stage 1 V6.2: {stage1_path}")
    ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    s1_state = ckpt.get("model", ckpt)

    # Stage 1 uses ActorCritic with keys: actor.*, critic.*, log_std
    # Our DualAC uses loco_ac.actor.*, loco_ac.critic.*, loco_ac.log_std
    transferred = 0
    for s1_key, s1_val in s1_state.items():
        our_key = f"loco_ac.{s1_key}"
        if our_key in net.state_dict():
            if net.state_dict()[our_key].shape == s1_val.shape:
                net.state_dict()[our_key].copy_(s1_val)
                transferred += 1
            else:
                print(f"  [WARN] Shape mismatch: {our_key} {s1_val.shape} vs {net.state_dict()[our_key].shape}")
        else:
            print(f"  [WARN] Key not found in DualAC: {our_key} (source: {s1_key})")

    print(f"  Transferred {transferred} loco parameters")

    # Freeze loco
    frozen = 0
    for name, p in net.named_parameters():
        if name.startswith("loco_ac."):
            p.requires_grad = False
            frozen += 1
    print(f"  Frozen {frozen} loco parameters")

    # Arm: fresh init with high exploration
    net.arm_actor.log_std.data.fill_(np.log(0.6))
    print(f"  Arm: FRESH init, log_std = log(0.6)")

    # Print checkpoint info
    for key in ["best_reward", "iteration", "curriculum_level"]:
        if key in ckpt:
            print(f"  Stage 1 {key}: {ckpt[key]}")

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_n = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"\n  Trainable: {trainable:,} | Frozen: {frozen_n:,}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)
    net = DualActorCritic().to(device)

    start_iter = 0
    best_reward = float('-inf')

    if args_cli.checkpoint:
        print(f"\n[Load] Resuming from {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0) + 1
        best_reward = ckpt.get("best_reward", float('-inf'))
        env.curr_level = min(ckpt.get("curriculum_level", 0), len(CURRICULUM) - 1)
        env.validated_reaches = ckpt.get("validated_reaches", 0)
        env.timed_out_targets = ckpt.get("timed_out_targets", 0)
        env.total_attempts = ckpt.get("total_attempts", 0)
        # Re-freeze loco
        for name, p in net.named_parameters():
            if name.startswith("loco_ac."):
                p.requires_grad = False
        # Restore optimizer/scheduler
        ppo = ArmPPO(net, device, lr=1e-3, max_iter=args_cli.max_iterations)
        if "arm_optimizer" in ckpt:
            ppo.opt.load_state_dict(ckpt["arm_optimizer"])
        if "arm_scheduler" in ckpt:
            ppo.sched.load_state_dict(ckpt["arm_scheduler"])
        print(f"  Resumed: iter={start_iter}, R={best_reward:.2f}, Lv={env.curr_level}, "
              f"VR={env.validated_reaches}, TO={env.timed_out_targets}")
    else:
        if args_cli.stage1_checkpoint is None:
            raise ValueError("--stage1_checkpoint required for fresh Stage 2 start")
        load_stage1_and_setup(net, args_cli.stage1_checkpoint, device)
        ppo = ArmPPO(net, device, lr=1e-3, max_iter=args_cli.max_iterations)

    # Logging
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\n[Log] {log_dir}")

    # Rollout buffers (ARM only — loco is frozen)
    T = 24
    arm_obs_buf = torch.zeros(T, env.num_envs, ARM_OBS_DIM, device=device)
    arm_act_buf = torch.zeros(T, env.num_envs, ARM_ACT_DIM, device=device)
    arm_rew_buf = torch.zeros(T, env.num_envs, device=device)
    arm_val_buf = torch.zeros(T, env.num_envs, device=device)
    arm_lp_buf = torch.zeros(T, env.num_envs, device=device)
    done_buf = torch.zeros(T, env.num_envs, device=device)

    obs, _ = env.reset()

    print(f"\n{'='*80}")
    print("STARTING STAGE 2: ARM REACHING TRAINING")
    print(f"  Loco: FROZEN (66->15), Arm: FRESH (39->7)")
    print(f"  Phase 1 (L0-3): Standing + Reaching")
    print(f"  Phase 2 (L4-5): Walking + Reaching")
    print(f"  Phase 3 (L6-7): Walking + Orientation")
    print(f"{'='*80}\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        # Exploration decay
        progress = iteration / args_cli.max_iterations
        arm_std = 0.6 + (0.15 - 0.6) * progress
        net.arm_actor.log_std.data.fill_(np.log(max(arm_std, 0.15)))

        # Collect rollout
        for t in range(T):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            with torch.no_grad():
                loco_action = net.get_loco_action(loco_obs)
                arm_action, arm_logp = net.get_arm_action(arm_obs)
                arm_value = net.arm_critic(arm_obs)

            arm_obs_buf[t] = arm_obs
            arm_act_buf[t] = arm_action
            arm_val_buf[t] = arm_value
            arm_lp_buf[t] = arm_logp

            combined = torch.cat([loco_action, arm_action], dim=-1)
            obs_dict, reward, terminated, truncated, _ = env.step(combined)

            arm_rew_buf[t] = env.arm_reward
            done_buf[t] = (terminated | truncated).float()

        # Compute returns (ARM only)
        with torch.no_grad():
            final_arm_obs = env.get_arm_obs()
            arm_next_val = net.arm_critic(final_arm_obs)

        arm_adv, arm_ret = ppo.gae(arm_rew_buf, arm_val_buf, done_buf, arm_next_val)

        # PPO update (ARM only)
        losses = ppo.update(
            arm_obs_buf.reshape(-1, ARM_OBS_DIM),
            arm_act_buf.reshape(-1, ARM_ACT_DIM),
            arm_lp_buf.reshape(-1),
            arm_ret.reshape(-1),
            arm_adv.reshape(-1),
            arm_val_buf.reshape(-1),
        )

        mean_arm_reward = arm_rew_buf.mean().item()
        env.update_curriculum(mean_arm_reward)

        # Tensorboard
        if iteration % 10 == 0:
            writer.add_scalar("reward/arm_step", mean_arm_reward, iteration)
            writer.add_scalar("reward/best", best_reward, iteration)
            writer.add_scalar("loss/actor", losses["a"], iteration)
            writer.add_scalar("loss/critic", losses["c"], iteration)
            writer.add_scalar("loss/entropy", losses["e"], iteration)
            writer.add_scalar("train/lr", losses["lr"], iteration)
            writer.add_scalar("train/arm_std", np.exp(net.arm_actor.log_std.data.mean().item()), iteration)
            writer.add_scalar("curriculum/level", env.curr_level, iteration)
            writer.add_scalar("curriculum/validated_reaches", env.validated_reaches, iteration)
            writer.add_scalar("curriculum/timed_out", env.timed_out_targets, iteration)

            # Arm reward components
            for key, val in env.arm_reward_components.items():
                writer.add_scalar(f"arm_reward/{key}", val, iteration)

            # Robot stats
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            writer.add_scalar("robot/height", height, iteration)

            v_rate = env.validated_reaches / max(env.total_attempts, 1)
            writer.add_scalar("curriculum/validated_rate", v_rate, iteration)

        # Console log
        if iteration % 50 == 0:
            rc = env.arm_reward_components
            ee_dist = rc.get("ee_dist", 0)
            ee_spd = rc.get("ee_speed", 0)
            orient_err = rc.get("orient_err", 0)
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            v_rate = env.validated_reaches / max(env.total_attempts, 1)

            phase = "P1-Stand" if env.curr_level < 4 else (
                "P2-Walk" if env.curr_level < 6 else "P3-Orient")

            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_arm_reward:.2f} Best={best_reward:.2f} "
                  f"Lv={env.curr_level}({phase}) "
                  f"VR={env.validated_reaches} TO={env.timed_out_targets} "
                  f"Rate={v_rate:.1%} "
                  f"EE={ee_dist:.3f} Spd={ee_spd:.3f} "
                  f"H={height:.3f} OrErr={orient_err:.2f} "
                  f"LR={losses['lr']:.2e} std={np.exp(net.arm_actor.log_std.data.mean().item()):.3f}")

        # Save checkpoints
        if iteration % 500 == 0 and iteration > 0:
            path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                "model": net.state_dict(),
                "arm_optimizer": ppo.opt.state_dict(),
                "arm_scheduler": ppo.sched.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "validated_reaches": env.validated_reaches,
                "timed_out_targets": env.timed_out_targets,
                "total_attempts": env.total_attempts,
                "stage1_checkpoint": args_cli.stage1_checkpoint,
            }, path)
            print(f"  [Save] {path}")

        if mean_arm_reward > best_reward:
            best_reward = mean_arm_reward
            path = os.path.join(log_dir, "model_best.pt")
            torch.save({
                "model": net.state_dict(),
                "arm_optimizer": ppo.opt.state_dict(),
                "arm_scheduler": ppo.sched.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "validated_reaches": env.validated_reaches,
                "timed_out_targets": env.timed_out_targets,
                "total_attempts": env.total_attempts,
                "stage1_checkpoint": args_cli.stage1_checkpoint,
            }, path)

    # Final save
    path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": net.state_dict(),
        "arm_optimizer": ppo.opt.state_dict(),
        "arm_scheduler": ppo.sched.state_dict(),
        "iteration": args_cli.max_iterations - 1,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
        "validated_reaches": env.validated_reaches,
        "timed_out_targets": env.timed_out_targets,
        "total_attempts": env.total_attempts,
        "stage1_checkpoint": args_cli.stage1_checkpoint,
    }, path)

    print(f"\n{'='*80}")
    print("STAGE 2 TRAINING COMPLETE")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Validated Reaches: {env.validated_reaches}")
    print(f"  Timed Out: {env.timed_out_targets}")
    if env.total_attempts > 0:
        print(f"  Validated Rate: {env.validated_reaches / env.total_attempts:.1%}")
    print(f"  Log Dir: {log_dir}")
    print(f"{'='*80}")

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
