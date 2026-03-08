"""
Unified Stage 3: Orientation Fine-Tune — Critic Reset (29DoF)
=============================================================
Stage 2 arm_actor (position reaching, 4.3cm) PRESERVED with low LR.
arm_critic RESET from scratch (new reward landscape).
Loco policy (Stage 1 V6.2) FROZEN.

ARCHITECTURE (same as Stage 2):
  LocoActor  (66->15) [512,256,128] + LN + ELU  — FROZEN
  LocoCritic (66->1)  [512,256,128] + LN + ELU  — FROZEN
  ArmActor   (39->7)  [256,256,128] + ELU       — LOADED from Stage 2, LR=5e-5
  ArmCritic  (39->1)  [256,256,128] + ELU       — RESET (Xavier init), LR=3e-4

KEY DESIGN DECISIONS:
  1. Critic reset: Stage 2 critic learned R~23 value landscape. Orient changes
     reward distribution -> critic mismatch -> policy collapse (proven in failed
     experiment: orient weight 3->6 caused EE 4.3cm -> 35cm). Fresh critic learns
     new landscape from scratch.
  2. Dual optimizer: Actor LR 5e-5 (6x lower) preserves position precision.
     Critic LR 3e-4 (normal) allows fast value learning.
  3. L0 warmup: orient OFF for first 2000 iter. Critic learns baseline reward,
     actor confirms position still works. Only then orient activates.
  4. Early stop: if avg_ee > 0.15m for 3000 iter, position lost -> abort.
  5. model_best guard: only saved when EE < 0.08m (ensures usable checkpoint).

CURRICULUM (6 levels, orient-focused):
  L0: pos_thresh=4cm, orient=OFF (warmup — critic learns, actor preserves)
  L1: pos_thresh=4cm, orient_thresh=2.5 rad (very loose)
  L2: pos_thresh=4cm, orient_thresh=2.0 rad
  L3: pos_thresh=4cm, orient_thresh=1.5 rad
  L4: pos_thresh=4cm, orient_thresh=1.2 rad
  L5: pos_thresh=4cm, orient_thresh=1.0 rad (FINAL)

USAGE:
    # From Stage 2 checkpoint:
    isaaclab.bat -p ... --stage2_checkpoint logs/ulc/.../model_best.pt --num_envs 4096 --headless
    # Resume from Stage 3 checkpoint:
    isaaclab.bat -p ... --checkpoint logs/ulc/.../model_best.pt --num_envs 4096 --headless

2026-03-08: Created from Stage 2 script. Critic reset + dual optimizer approach.
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
ARM_ACTION_SCALE = 2.0  # Override cfg (0.5 too small — limits reach to ~0.25m, need 0.55m)
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS
SHOULDER_OFFSET_RIGHT = _cfg_mod.SHOULDER_OFFSET_RIGHT

# ============================================================================
# DIMENSIONS
# ============================================================================

LOCO_OBS_DIM = 66
LOCO_ACT_DIM = 15
ARM_OBS_DIM = 39
ARM_ACT_DIM = 7
COMBINED_ACT_DIM = LOCO_ACT_DIM + ARM_ACT_DIM  # 22

# ============================================================================
# ARM REWARD WEIGHTS — Same as Stage 2 (orient weight overridden by CLI)
# ============================================================================

ARM_REWARD_WEIGHTS = {
    "reaching": 10.0,
    "final_push": 4.0,
    "distance": 3.0,
    "velocity_toward": 6.0,
    "progress": 5.0,
    "pbrs": 3.0,
    "proximity_bonus": 1.0,
    "reach_bonus": 1.0,
    "orient": 3.0,             # Will be overridden by --orient_weight CLI arg
    "left_arm_dev": -0.5,
    "stillness_penalty": -2.0,
    "action_rate": -0.02,
    "height": 1.0,
    "tilt": 0.5,
    "alive": 0.2,
}

REACH_BONUS_VALUE = 50.0
EPISODE_TERMINATE_ON_REACH = True
CURRICULUM_WINDOW_SIZE = 50000
CURRICULUM_MIN_WINDOW = 5000
ORIENT_GATE_DISTANCE = 0.08  # meters — orient reward only when dist < 8cm

# ============================================================================
# CURRICULUM (6 levels, orient-focused)
# L0: warmup (orient OFF, critic learns baseline)
# L1-L5: progressive orient tightening
# ============================================================================

CURRICULUM = [
    {
        "description": "L0: Warmup (orient OFF, critic learns baseline)",
        "vx": (0.0, 0.35), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.06,
        "max_reach_steps": 200,
        "validated_reach_rate": 0.50,      # Should be easy (4cm already working)
        "min_validated_reaches": 5000,
        "min_steps": 2000,                 # 2000 iter warmup before promotion
        "use_orientation": False,
        "workspace_radius": (0.18, 0.55),
    },
    {
        "description": "L1: Orient 2.5 rad (very loose, ~143 deg)",
        "vx": (0.0, 0.35), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.06,
        "max_reach_steps": 200,
        "validated_reach_rate": 0.30,
        "min_validated_reaches": 5000,
        "min_steps": 3000,
        "orient_threshold": 2.5,
        "use_orientation": True,
        "workspace_radius": (0.18, 0.55),
    },
    {
        "description": "L2: Orient 2.0 rad (~115 deg)",
        "vx": (0.0, 0.35), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.06,
        "max_reach_steps": 200,
        "validated_reach_rate": 0.25,
        "min_validated_reaches": 5000,
        "min_steps": 3000,
        "orient_threshold": 2.0,
        "use_orientation": True,
        "workspace_radius": (0.18, 0.55),
    },
    {
        "description": "L3: Orient 1.5 rad (~86 deg)",
        "vx": (0.0, 0.35), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.06,
        "max_reach_steps": 200,
        "validated_reach_rate": 0.20,
        "min_validated_reaches": 5000,
        "min_steps": 3500,
        "orient_threshold": 1.5,
        "use_orientation": True,
        "workspace_radius": (0.18, 0.55),
    },
    {
        "description": "L4: Orient 1.2 rad (~69 deg)",
        "vx": (0.0, 0.35), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.06,
        "max_reach_steps": 200,
        "validated_reach_rate": 0.15,
        "min_validated_reaches": 5000,
        "min_steps": 4000,
        "orient_threshold": 1.2,
        "use_orientation": True,
        "workspace_radius": (0.18, 0.55),
    },
    {
        "description": "L5: FINAL — Orient 1.0 rad (~57 deg)",
        "vx": (0.0, 0.35), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.06,
        "max_reach_steps": 200,
        "validated_reach_rate": None,       # FINAL — no advancement
        "min_validated_reaches": None,
        "min_steps": None,
        "orient_threshold": 1.0,
        "use_orientation": True,
        "workspace_radius": (0.18, 0.55),
    },
]

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Stage 3: Orientation Fine-Tune (Critic Reset, 29DoF)")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Stage 2 model_best.pt (arm actor with position precision)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 3 checkpoint")
    parser.add_argument("--actor_lr", type=float, default=5e-5,
                        help="Arm actor LR (low to preserve position)")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="Arm critic LR (normal, learns new landscape)")
    parser.add_argument("--orient_weight", type=float, default=0.5,
                        help="Orient reward weight (overrides ARM_REWARD_WEIGHTS)")
    parser.add_argument("--experiment_name", type=str, default="g1_stage3_orient")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

args_cli = parse_args()

# Apply orient weight override BEFORE env creation
ARM_REWARD_WEIGHTS["orient"] = args_cli.orient_weight

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
print("UNIFIED STAGE 3: ORIENTATION FINE-TUNE (Critic Reset, 29DoF)")
print(f"  USD: {G1_29DOF_USD}")
print(f"  Loco: {LOCO_OBS_DIM} obs -> {LOCO_ACT_DIM} act (FROZEN)")
print(f"  Arm Actor:  {ARM_OBS_DIM} obs -> {ARM_ACT_DIM} act (LOADED, LR={args_cli.actor_lr})")
print(f"  Arm Critic: {ARM_OBS_DIM} obs -> 1 val (RESET, LR={args_cli.critic_lr})")
print(f"  Orient weight: {args_cli.orient_weight}")
print(f"  Orient gate: {ORIENT_GATE_DISTANCE}m")
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
# DUAL ACTOR-CRITIC NETWORKS (identical to Stage 2)
# ============================================================================

class LocoActorCritic(nn.Module):
    """Locomotion AC: 66 obs -> 15 act (FROZEN, from Stage 1 V6.2)."""
    def __init__(self, num_obs=LOCO_OBS_DIM, num_act=LOCO_ACT_DIM, hidden=[512, 256, 128]):
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

    def forward_actor(self, x):
        return self.actor(x)

    def forward_critic(self, x):
        return self.critic(x).squeeze(-1)


class ArmActor(nn.Module):
    """Arm policy: 39 obs -> 7 act (LOADED from Stage 2)"""
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
    """Arm value: 39 obs -> 1 value (RESET from scratch)"""
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
    """Dual AC — Loco (FROZEN) + Arm Actor (LOADED) + Arm Critic (RESET)"""
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
# ARM PPO — Dual Optimizer (actor LR != critic LR)
# ============================================================================

class ArmPPO:
    def __init__(self, net, device, actor_lr=5e-5, critic_lr=3e-4):
        self.net = net
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # Separate optimizers for different LRs
        self.actor_opt = torch.optim.AdamW(
            net.arm_actor.parameters(), lr=actor_lr, weight_decay=1e-5
        )
        self.critic_opt = torch.optim.AdamW(
            net.arm_critic.parameters(), lr=critic_lr, weight_decay=1e-5
        )

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

                # Update actor
                self.actor_opt.zero_grad()
                actor_loss = a_loss - 0.01 * ent.mean()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.net.arm_actor.parameters(), 0.5)
                self.actor_opt.step()

                # Update critic
                self.critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.net.arm_critic.parameters(), 0.5)
                self.critic_opt.step()

                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                n += 1

        return {"a": tot_a / max(n, 1), "c": tot_c / max(n, 1),
                "e": tot_e / max(n, 1), "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr}


# ============================================================================
# ENVIRONMENT (identical to Stage 2 — copied verbatim)
# ============================================================================

def create_env(num_envs, device):
    """Create Stage 3 environment (same as Stage 2 — reward, obs, actions identical)."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )

        contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
        )

        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_29DOF_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, HEIGHT_DEFAULT + 0.05),
                joint_pos=DEFAULT_ALL_POSES,
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=LEG_JOINT_NAMES,
                    **ACTUATOR_PARAMS["legs"],
                ),
                "feet": ImplicitActuatorCfg(
                    joint_names_expr=["left_ankle_pitch_joint", "right_ankle_pitch_joint",
                                     "left_ankle_roll_joint", "right_ankle_roll_joint"],
                    **ACTUATOR_PARAMS["feet"],
                ),
                "shoulders": ImplicitActuatorCfg(
                    joint_names_expr=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                                     "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                                     "left_shoulder_yaw_joint", "right_shoulder_yaw_joint"],
                    **ACTUATOR_PARAMS["shoulders"],
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["left_elbow_joint", "right_elbow_joint"],
                    **ACTUATOR_PARAMS["arms"],
                ),
                "wrists": ImplicitActuatorCfg(
                    joint_names_expr=["left_wrist_roll_joint", "right_wrist_roll_joint",
                                     "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                                     "left_wrist_yaw_joint", "right_wrist_yaw_joint"],
                    **ACTUATOR_PARAMS["wrists"],
                ),
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=HAND_JOINT_NAMES,
                    **ACTUATOR_PARAMS["hands"],
                ),
            },
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 20.0
        scene = SceneCfg(num_envs=num_envs, env_spacing=5.0)
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        action_space = COMBINED_ACT_DIM
        observation_space = LOCO_OBS_DIM
        env_spacing = 5.0

    class Stage3OrientEnv(DirectRLEnv):
        """Stage 3 Orient Fine-Tune Environment.
        Identical to Stage2ArmEnv — same obs, rewards, reach validation.
        Only curriculum and training loop differ.
        """
        cfg: EnvCfg

        def __init__(self, cfg):
            super().__init__(cfg)
            r = self.scene["robot"]
            self.robot = r
            jn = r.joint_names

            # Joint indices
            self.loco_idx = torch.tensor([jn.index(n) for n in LOCO_JOINT_NAMES], device=self.device)
            self.right_arm_idx = torch.tensor([jn.index(n) for n in ARM_JOINT_NAMES_RIGHT], device=self.device)
            self.left_arm_idx = torch.tensor([jn.index(n) for n in ARM_JOINT_NAMES_LEFT], device=self.device)
            self.hand_idx = torch.tensor([jn.index(n) for n in HAND_JOINT_NAMES], device=self.device)

            self.left_knee_idx = torch.tensor([jn.index("left_knee_joint")], device=self.device)
            self.right_knee_idx = torch.tensor([jn.index("right_knee_joint")], device=self.device)

            # Hip yaw indices for clamp
            hip_yaw_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
            self.hip_yaw_loco_idx = torch.tensor(
                [list(LOCO_JOINT_NAMES).index(n) for n in hip_yaw_names], device=self.device)

            # Palm body index
            body_names = r.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "palm" in name.lower():
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                for i, name in enumerate(body_names):
                    if "right" in name.lower() and ("hand" in name.lower() or "link" in name.lower()):
                        self.palm_idx = i
                        break
            if self.palm_idx is None:
                self.palm_idx = len(body_names) - 1
            print(f"[Env] Palm body idx: {self.palm_idx} ({body_names[self.palm_idx]})")

            # Default poses
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device)
            right_arm_start = 0
            self.default_right_arm = torch.tensor(
                DEFAULT_ARM_LIST[right_arm_start:right_arm_start + len(ARM_JOINT_NAMES_RIGHT)],
                device=self.device)
            left_arm_start = len(ARM_JOINT_NAMES_RIGHT)
            self.default_left_arm = torch.tensor(
                DEFAULT_ARM_LIST[left_arm_start:left_arm_start + len(ARM_JOINT_NAMES_LEFT)],
                device=self.device)

            # Loco action scales
            leg_scales = torch.tensor([LEG_ACTION_SCALE] * len(LEG_JOINT_NAMES), device=self.device)
            waist_scales = torch.tensor([WAIST_ACTION_SCALE] * len(WAIST_JOINT_NAMES), device=self.device)
            self.loco_action_scales = torch.cat([leg_scales, waist_scales])

            # State buffers
            N = self.num_envs
            self.height_cmd = torch.full((N,), HEIGHT_DEFAULT, device=self.device)
            self.vel_cmd = torch.zeros(N, 3, device=self.device)
            self.torso_cmd = torch.zeros(N, 3, device=self.device)
            self.phase = torch.zeros(N, device=self.device)
            self.prev_loco_act = torch.zeros(N, LOCO_ACT_DIM, device=self.device)
            self.prev_arm_act = torch.zeros(N, ARM_ACT_DIM, device=self.device)
            self._prev_arm_act = torch.zeros(N, ARM_ACT_DIM, device=self.device)

            # Target state
            self.target_pos_body = torch.zeros(N, 3, device=self.device)
            self.target_orient = torch.zeros(N, 3, device=self.device)
            self.target_orient[:, 2] = -1.0  # Default: palm facing down

            # EE tracking
            self.prev_ee_pos_w = torch.zeros(N, 3, device=self.device)

            # Anti-gaming
            self.ee_pos_at_spawn = torch.zeros(N, 3, device=self.device)
            self.steps_since_spawn = torch.zeros(N, dtype=torch.long, device=self.device)
            self.initial_dist = torch.ones(N, device=self.device) * 0.3

            # Curriculum state
            self.curr_level = 0
            self.validated_reaches = 0
            self.timed_out_targets = 0
            self.total_attempts = 0
            self.stage_validated_reaches = 0
            self.stage_timed_out = 0
            self.stage_steps = 0

            # Ring buffer for windowed rate
            self.window_buf = torch.zeros(CURRICULUM_WINDOW_SIZE, dtype=torch.bool, device=self.device)
            self.window_idx = 0
            self.window_count = 0

            # Episode termination on reach
            self.reach_terminated = torch.zeros(N, dtype=torch.bool, device=self.device)
            self.reach_bonus_pending = torch.zeros(N, device=self.device)

            # PBRS
            self.prev_potential = torch.zeros(N, device=self.device)

            # Push perturbation
            self.step_count = 0
            self.push_timer = torch.zeros(N, dtype=torch.long, device=self.device)

            # Command resample timer
            self.cmd_timer = torch.zeros(N, dtype=torch.long, device=self.device)
            self.cmd_resample_targets = torch.randint(150, 400, (N,), device=self.device)

            # Reward storage
            self.arm_reward = torch.zeros(N, device=self.device)
            self.arm_reward_components = {}

            # Visual markers (lazy init)
            self._markers = None

            # Illegal contact bodies (pelvis/torso/knee)
            body_names = self.robot.body_names
            self._illegal_body_ids = []
            for i, name in enumerate(body_names):
                nl = name.lower()
                if "pelvis" in nl or "torso" in nl or "knee" in nl:
                    self._illegal_body_ids.append(i)
            self._illegal_body_ids = torch.tensor(self._illegal_body_ids, device=self.device)

        # ================================================================
        # END EFFECTOR
        # ================================================================

        def _compute_ee(self):
            """Get right palm world position and quaternion."""
            body_pos = self.robot.data.body_pos_w
            body_quat = self.robot.data.body_quat_w
            return body_pos[:, self.palm_idx], body_quat[:, self.palm_idx]

        # ================================================================
        # TARGET SAMPLING
        # ================================================================

        def _sample_targets(self, env_ids):
            """Sample random targets in right arm workspace (absolute, body frame)."""
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]
            ws_inner, ws_outer = lv["workspace_radius"]

            # Spherical sampling from shoulder offset
            shoulder = torch.tensor(SHOULDER_OFFSET_RIGHT, device=self.device)
            azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
            elevation = torch.empty(n, device=self.device).uniform_(-0.3, 0.5)
            radius = torch.empty(n, device=self.device).uniform_(ws_inner, ws_outer)

            x = shoulder[0] + radius * torch.cos(elevation) * torch.cos(azimuth)
            y = shoulder[1] + radius * torch.cos(elevation) * torch.sin(azimuth)
            z = shoulder[2] + radius * torch.sin(elevation)

            # Workspace clamp
            x = x.clamp(-0.10, 0.55)
            y = y.clamp(-0.60, -0.05)
            z = z.clamp(-0.15, 0.55)

            targets = torch.stack([x, y, z], dim=-1)
            self.target_pos_body[env_ids] = targets

            # Min distance enforcement
            ee_w, _ = self._compute_ee()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            ee_body = quat_apply_inverse(root_quat[env_ids], ee_w[env_ids] - root_pos[env_ids])
            dist = torch.norm(targets - ee_body, dim=-1)
            min_dist = lv["min_target_distance"]
            too_close = dist < min_dist
            if too_close.any():
                direction = targets[too_close] - ee_body[too_close]
                d = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                targets[too_close] = ee_body[too_close] + direction / d * min_dist
                self.target_pos_body[env_ids] = targets

            # Record spawn state
            self.ee_pos_at_spawn[env_ids] = ee_body
            self.steps_since_spawn[env_ids] = 0
            self.initial_dist[env_ids] = torch.norm(targets - ee_body, dim=-1).clamp(min=0.01)

            # Initialize PBRS potential
            self.prev_potential[env_ids] = torch.exp(-self.initial_dist[env_ids] / 0.15)

            # Sample target orientation (palm down default)
            self.target_orient[env_ids, 0] = 0.0
            self.target_orient[env_ids, 1] = 0.0
            self.target_orient[env_ids, 2] = -1.0

        def _sample_vel_commands(self, env_ids):
            """Sample velocity commands from curriculum range."""
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]
            vx_lo, vx_hi = lv["vx"]
            vy_lo, vy_hi = lv["vy"]
            vyaw_lo, vyaw_hi = lv["vyaw"]

            # 15% standing samples
            standing = torch.rand(n, device=self.device) < 0.15
            vx = torch.empty(n, device=self.device).uniform_(vx_lo, max(vx_hi, vx_lo + 0.01))
            vy = torch.empty(n, device=self.device).uniform_(vy_lo, vy_hi if vy_hi > vy_lo else vy_lo + 0.01)
            vyaw = torch.empty(n, device=self.device).uniform_(vyaw_lo, vyaw_hi if vyaw_hi > vyaw_lo else vyaw_lo + 0.01)
            vx[standing] = 0.0
            vy[standing] = 0.0
            vyaw[standing] = 0.0

            self.vel_cmd[env_ids, 0] = vx
            self.vel_cmd[env_ids, 1] = vy
            self.vel_cmd[env_ids, 2] = vyaw

            self.cmd_timer[env_ids] = 0
            self.cmd_resample_targets[env_ids] = torch.randint(150, 400, (n,), device=self.device)

        # ================================================================
        # OBSERVATIONS (identical to Stage 2)
        # ================================================================

        def get_loco_obs(self):
            """66-dim loco observation (Stage 1 V6.2 format)."""
            r = self.robot
            root_quat = r.data.root_quat_w
            inv_quat = root_quat.clone()
            inv_quat[:, 1:] *= -1

            lin_vel_w = r.data.root_lin_vel_w
            ang_vel_w = r.data.root_ang_vel_w
            lin_vel_b = quat_apply(inv_quat, lin_vel_w)
            ang_vel_b = quat_apply(inv_quat, ang_vel_w)

            g_w = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_g = quat_apply(inv_quat, g_w)

            leg_pos = r.data.joint_pos[:, self.loco_idx[:12]] - self.default_loco[:12]
            leg_vel = r.data.joint_vel[:, self.loco_idx[:12]] * 0.1
            waist_pos = r.data.joint_pos[:, self.loco_idx[12:15]] - self.default_loco[12:15]
            waist_vel = r.data.joint_vel[:, self.loco_idx[12:15]] * 0.1

            euler = quat_to_euler_xyz(root_quat)
            phase_sin = torch.sin(2 * np.pi * self.phase).unsqueeze(-1)
            phase_cos = torch.cos(2 * np.pi * self.phase).unsqueeze(-1)

            obs = torch.cat([
                lin_vel_b, ang_vel_b, proj_g,
                leg_pos, leg_vel, waist_pos, waist_vel,
                self.height_cmd.unsqueeze(-1),
                self.vel_cmd,
                phase_sin, phase_cos,
                self.prev_loco_act,
                euler,
                self.torso_cmd,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num(0)

        def get_arm_obs(self):
            """39-dim arm observation."""
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w

            arm_pos = r.data.joint_pos[:, self.right_arm_idx] - self.default_right_arm
            arm_vel = r.data.joint_vel[:, self.right_arm_idx] * 0.1

            ee_w, palm_quat = self._compute_ee()
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            target_w = root_pos + quat_apply(root_quat, self.target_pos_body)
            pos_error = self.target_pos_body - ee_body

            orient_err = compute_orientation_error(palm_quat, self.target_orient)
            orient_err_norm = orient_err.unsqueeze(-1) / np.pi

            lv = CURRICULUM[self.curr_level]
            max_steps = lv["max_reach_steps"]
            steps_norm = (self.steps_since_spawn.float() / max_steps).clamp(0, 2).unsqueeze(-1)

            obs = torch.cat([
                arm_pos,
                arm_vel,
                ee_body,
                palm_quat,
                self.target_pos_body,
                self.target_orient,
                pos_error,
                orient_err_norm,
                self.prev_arm_act,
                steps_norm,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num(0)

        # ================================================================
        # PRE-PHYSICS STEP (identical to Stage 2)
        # ================================================================

        def _pre_physics_step(self, actions):
            r = self.robot
            loco_act = actions[:, :LOCO_ACT_DIM]
            arm_act = actions[:, LOCO_ACT_DIM:COMBINED_ACT_DIM]

            num_joints = r.data.joint_pos.shape[1]
            targets = r.data.joint_pos.clone()

            # Loco joints
            loco_targets = self.default_loco + loco_act * self.loco_action_scales
            # Hip yaw clamp
            loco_targets[:, self.hip_yaw_loco_idx] = loco_targets[:, self.hip_yaw_loco_idx].clamp(-0.3, 0.3)
            # Waist clamp
            loco_targets[:, 12:15] = loco_targets[:, 12:15].clamp(-0.3, 0.3)
            targets[:, self.loco_idx] = loco_targets

            # Right arm (residual)
            arm_clamped = arm_act.clamp(-1.5, 1.5)
            arm_targets = self.default_right_arm + arm_clamped * ARM_ACTION_SCALE
            targets[:, self.right_arm_idx] = arm_targets

            # Left arm at default
            targets[:, self.left_arm_idx] = self.default_left_arm

            # Hands at default
            targets[:, self.hand_idx] = self.default_hand

            r.set_joint_position_target(targets)

            # Update state
            dt = self.cfg.sim.dt * self.cfg.decimation
            self.phase = (self.phase + GAIT_FREQUENCY * dt) % 1.0
            self._prev_arm_act = self.prev_arm_act.clone()
            self.prev_arm_act = arm_act.clone()
            self.prev_loco_act = loco_act.clone()
            self.step_count += 1
            self.steps_since_spawn += 1

            # Reach validation
            self.reach_terminated[:] = False
            self._validate_reaches()

            # Velocity resample on timer
            self.cmd_timer += 1
            resample_mask = self.cmd_timer >= self.cmd_resample_targets
            if resample_mask.any():
                ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
                self._sample_vel_commands(ids)

            # Push perturbation (only in walk levels)
            lv = CURRICULUM[self.curr_level]
            if lv["vx"][1] > 0:
                self.push_timer += 1
                push_mask = self.push_timer >= 200
                if push_mask.any():
                    push_ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                    force_mag = torch.empty(len(push_ids), device=self.device).uniform_(0, 15.0)
                    angle = torch.empty(len(push_ids), device=self.device).uniform_(0, 2 * np.pi)
                    fx = force_mag * torch.cos(angle)
                    fy = force_mag * torch.sin(angle)
                    forces = torch.zeros(self.num_envs, r.num_bodies, 3, device=self.device)
                    forces[push_ids, 0, 0] = fx
                    forces[push_ids, 0, 1] = fy
                    r.set_external_force_and_torque(
                        forces, torch.zeros_like(forces))
                    self.push_timer[push_ids] = 0

            # Update prev EE pos
            ee_w, _ = self._compute_ee()
            self.prev_ee_pos_w = ee_w.clone()

        # ================================================================
        # REACH VALIDATION (identical to Stage 2)
        # ================================================================

        def _validate_reaches(self):
            """3-condition reach validation: position + displacement + time."""
            lv = CURRICULUM[self.curr_level]
            ee_w, palm_quat = self._compute_ee()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            target_w = root_pos + quat_apply(root_quat, self.target_pos_body)
            dist = torch.norm(ee_w - target_w, dim=-1)

            pos_reached = dist < lv["pos_threshold"]
            if lv["use_orientation"]:
                orient_thresh = lv.get("orient_threshold", 2.0)
                orient_err = compute_orientation_error(palm_quat, self.target_orient)
                pos_reached = pos_reached & (orient_err < orient_thresh)

            disp = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1)
            disp_ok = disp >= lv["min_displacement"]
            within_time = self.steps_since_spawn <= lv["max_reach_steps"]
            timed_out = self.steps_since_spawn > lv["max_reach_steps"]

            validated = pos_reached & disp_ok & within_time

            if validated.any():
                n_val = validated.sum().item()
                self.validated_reaches += n_val
                self.total_attempts += n_val
                self.stage_validated_reaches += n_val
                self.reach_bonus_pending[validated] = REACH_BONUS_VALUE

                # Ring buffer
                for _ in range(n_val):
                    self.window_buf[self.window_idx] = True
                    self.window_idx = (self.window_idx + 1) % CURRICULUM_WINDOW_SIZE
                    self.window_count = min(self.window_count + 1, CURRICULUM_WINDOW_SIZE)

                # Episode termination on reach
                if EPISODE_TERMINATE_ON_REACH:
                    self.reach_terminated[validated] = True

                # Resample validated targets
                val_ids = validated.nonzero(as_tuple=False).squeeze(-1)
                self._sample_targets(val_ids)

            if timed_out.any():
                n_to = timed_out.sum().item()
                self.timed_out_targets += n_to
                self.total_attempts += n_to
                self.stage_timed_out += n_to

                for _ in range(n_to):
                    self.window_buf[self.window_idx] = False
                    self.window_idx = (self.window_idx + 1) % CURRICULUM_WINDOW_SIZE
                    self.window_count = min(self.window_count + 1, CURRICULUM_WINDOW_SIZE)

                to_ids = timed_out.nonzero(as_tuple=False).squeeze(-1)
                self._sample_targets(to_ids)

            # Update visual markers
            if self._markers is not None:
                target_w = root_pos + quat_apply(root_quat, self.target_pos_body)
                self._markers.visualize(target_w[:1])

        def _get_observations(self):
            return {"policy": self.get_loco_obs()}

        # ================================================================
        # REWARD (identical to Stage 2 — orient_weight from CLI)
        # ================================================================

        def compute_arm_reward(self):
            """Comprehensive arm reward: sigmoid reaching + PBRS + proximity + orient."""
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

            # === REACHING REWARDS ===
            pos_threshold = lv["pos_threshold"]
            r_reaching = torch.sigmoid((pos_threshold - dist) * 30.0)
            r_final_push = torch.exp(-15.0 * dist) * torch.sigmoid((0.08 - dist) * 25.0)
            r_distance = torch.exp(-8.0 * dist)

            # === MOVEMENT REWARDS ===
            vel_toward = (ee_vel_b * dir_unit).sum(dim=-1)
            r_velocity = vel_toward.clamp(0.0, 1.0)
            r_progress = ((self.initial_dist - dist) / self.initial_dist.clamp(min=0.01)).clamp(0.0, 1.0)

            # === SHAPING REWARDS ===
            current_potential = torch.exp(-dist / 0.15)
            r_pbrs = ARM_REWARD_WEIGHTS["pbrs"] * (0.99 * current_potential - self.prev_potential)
            self.prev_potential = current_potential.clone()

            r_proximity = torch.zeros_like(dist)
            r_proximity += (dist < 0.15).float() * 1.0
            r_proximity += (dist < 0.10).float() * 2.0
            r_proximity += (dist < 0.05).float() * 5.0

            r_reach_bonus = self.reach_bonus_pending.clone()
            self.reach_bonus_pending.zero_()

            # === ORIENTATION (proximity gated) ===
            orient_err = compute_orientation_error(palm_quat, self.target_orient)
            if lv["use_orientation"]:
                orient_gate = (dist < ORIENT_GATE_DISTANCE).float()
                r_orient = orient_gate * (1.0 - orient_err / np.pi)
            else:
                r_orient = torch.zeros(self.num_envs, device=self.device)

            # === PENALTIES ===
            left_arm_pos = r.data.joint_pos[:, self.left_arm_idx]
            left_dev = (left_arm_pos - self.default_left_arm).abs().sum(-1)
            r_left_dev = left_dev

            far_mask = (dist > 0.15).float()
            r_stillness = torch.exp(-20.0 * ee_speed) * far_mask

            # === STABILITY ===
            height = root_pos[:, 2]
            r_height = torch.exp(-10.0 * (height - HEIGHT_DEFAULT) ** 2)

            g = quat_apply_inverse(root_quat, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))
            tilt = torch.asin(torch.clamp((g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))
            r_tilt = torch.exp(-3.0 * tilt)

            arm_diff = (self.prev_arm_act - self._prev_arm_act).clamp(-2.0, 2.0)
            r_action_rate = (arm_diff ** 2).sum(-1)

            # Store components for logging
            self.arm_reward_components = {
                "reaching": (ARM_REWARD_WEIGHTS["reaching"] * r_reaching).mean().item(),
                "final_push": (ARM_REWARD_WEIGHTS["final_push"] * r_final_push).mean().item(),
                "distance": (ARM_REWARD_WEIGHTS["distance"] * r_distance).mean().item(),
                "velocity": (ARM_REWARD_WEIGHTS["velocity_toward"] * r_velocity).mean().item(),
                "progress": (ARM_REWARD_WEIGHTS["progress"] * r_progress).mean().item(),
                "pbrs": r_pbrs.mean().item(),
                "proximity": (ARM_REWARD_WEIGHTS["proximity_bonus"] * r_proximity).mean().item(),
                "reach_bonus": r_reach_bonus.mean().item(),
                "orient": (ARM_REWARD_WEIGHTS["orient"] * r_orient).mean().item(),
                "left_dev": (ARM_REWARD_WEIGHTS["left_arm_dev"] * r_left_dev).mean().item(),
                "stillness": (ARM_REWARD_WEIGHTS["stillness_penalty"] * r_stillness).mean().item(),
                "height": (ARM_REWARD_WEIGHTS["height"] * r_height).mean().item(),
                "tilt": (ARM_REWARD_WEIGHTS["tilt"] * r_tilt).mean().item(),
                "action_rate": (ARM_REWARD_WEIGHTS["action_rate"] * r_action_rate).mean().item(),
                "ee_dist": dist.mean().item(),
                "ee_speed": ee_speed.mean().item(),
                "orient_err": orient_err.mean().item(),
            }

            reward = (
                ARM_REWARD_WEIGHTS["reaching"] * r_reaching
                + ARM_REWARD_WEIGHTS["final_push"] * r_final_push
                + ARM_REWARD_WEIGHTS["distance"] * r_distance
                + ARM_REWARD_WEIGHTS["velocity_toward"] * r_velocity
                + ARM_REWARD_WEIGHTS["progress"] * r_progress
                + r_pbrs
                + ARM_REWARD_WEIGHTS["proximity_bonus"] * r_proximity
                + ARM_REWARD_WEIGHTS["reach_bonus"] * r_reach_bonus
                + ARM_REWARD_WEIGHTS["orient"] * r_orient
                + ARM_REWARD_WEIGHTS["left_arm_dev"] * r_left_dev
                + ARM_REWARD_WEIGHTS["stillness_penalty"] * r_stillness
                + ARM_REWARD_WEIGHTS["height"] * r_height
                + ARM_REWARD_WEIGHTS["tilt"] * r_tilt
                + ARM_REWARD_WEIGHTS["action_rate"] * r_action_rate
                + ARM_REWARD_WEIGHTS["alive"]
            )
            return reward.clamp(-10, 50)

        def _get_rewards(self):
            self.arm_reward = self.compute_arm_reward()
            return self.arm_reward

        # ================================================================
        # TERMINATION
        # ================================================================

        def _get_dones(self):
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w
            height = root_pos[:, 2]

            g = quat_apply_inverse(root_quat, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            terminated = (height < 0.30) | (height > 1.2) | (g[:, :2].abs().max(dim=-1)[0] > 0.7)
            terminated = terminated | self.reach_terminated
            truncated = torch.zeros_like(terminated)
            return terminated, truncated

        # ================================================================
        # RESET
        # ================================================================

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            r = self.robot
            n = len(env_ids)

            # Joint state
            default_pos = torch.zeros(n, r.data.joint_pos.shape[1], device=self.device)
            for i, name in enumerate(r.joint_names):
                if name in DEFAULT_ALL_POSES:
                    default_pos[:, i] = DEFAULT_ALL_POSES[name]
            noise = torch.randn_like(default_pos) * 0.02
            r.write_joint_state_to_sim(
                default_pos + noise,
                torch.zeros(n, r.data.joint_vel.shape[1], device=self.device),
                env_ids=env_ids)

            # Root pose
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.empty(n, device=self.device).uniform_(-0.05, 0.05)
            pos[:, 1] = torch.empty(n, device=self.device).uniform_(-0.05, 0.05)
            pos[:, 2] = HEIGHT_DEFAULT + 0.05
            yaw = torch.empty(n, device=self.device).uniform_(-0.1, 0.1)
            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0] = torch.cos(yaw / 2)
            quat[:, 3] = torch.sin(yaw / 2)
            r.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            r.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)

            # Clear state
            self.vel_cmd[env_ids] = 0
            self.phase[env_ids] = 0
            self.prev_loco_act[env_ids] = 0
            self.prev_arm_act[env_ids] = 0
            self._prev_arm_act[env_ids] = 0
            self.prev_ee_pos_w[env_ids] = 0
            self.push_timer[env_ids] = 0
            self.reach_terminated[env_ids] = False
            self.reach_bonus_pending[env_ids] = 0

            self._sample_vel_commands(env_ids)
            self._sample_targets(env_ids)

        # ================================================================
        # CURRICULUM UPDATE
        # ================================================================

        def update_curriculum(self, mean_reward):
            lv = CURRICULUM[self.curr_level]
            if lv["min_validated_reaches"] is None:
                return  # Final level

            self.stage_steps += 1

            if self.stage_steps < lv.get("min_steps", 0):
                return

            stage_attempts = self.stage_validated_reaches + self.stage_timed_out
            if stage_attempts == 0:
                return

            if self.stage_validated_reaches < lv["min_validated_reaches"]:
                return

            if self.window_count < CURRICULUM_MIN_WINDOW:
                return

            if self.window_count >= CURRICULUM_WINDOW_SIZE:
                windowed_rate = self.window_buf.float().mean().item()
            else:
                windowed_rate = self.window_buf[:self.window_count].float().mean().item()

            cumulative_rate = self.stage_validated_reaches / stage_attempts

            if windowed_rate >= lv["validated_reach_rate"]:
                if self.curr_level < len(CURRICULUM) - 1:
                    self.curr_level += 1
                    new_lv = CURRICULUM[self.curr_level]
                    orient_info = f"orient={new_lv.get('orient_threshold', 'OFF')}" if new_lv["use_orientation"] else "orient=OFF"
                    print(f"\n{'='*60}")
                    print(f"  LEVEL UP! Level {self.curr_level}: {new_lv['description']}")
                    print(f"  Window Rate: {windowed_rate:.1%} (cumulative: {cumulative_rate:.1%})")
                    print(f"  Validated: {self.stage_validated_reaches}, Timed out: {self.stage_timed_out}")
                    print(f"  {orient_info}")
                    print(f"{'='*60}\n")
                    # Reset stage counters (MUST be inside promotion block!)
                    self.stage_validated_reaches = 0
                    self.stage_timed_out = 0
                    self.stage_steps = 0
                    self.window_buf.zero_()
                    self.window_idx = 0
                    self.window_count = 0
                    self._sample_targets(torch.arange(self.num_envs, device=self.device))
                    self._sample_vel_commands(torch.arange(self.num_envs, device=self.device))

    return Stage3OrientEnv(EnvCfg())


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_stage2_and_setup(net, stage2_path, device):
    """Load Stage 2 checkpoint: loco FROZEN, arm_actor LOADED, arm_critic RESET."""
    print(f"\n[Transfer] Loading Stage 2: {stage2_path}")
    ckpt = torch.load(stage2_path, map_location=device, weights_only=False)
    s2_state = ckpt.get("model", ckpt)

    # Load full model state (loco + arm_actor + arm_critic)
    missing, unexpected = net.load_state_dict(s2_state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected}")

    loaded_actor = sum(1 for k in s2_state if k.startswith("arm_actor."))
    loaded_critic = sum(1 for k in s2_state if k.startswith("arm_critic."))
    loaded_loco = sum(1 for k in s2_state if k.startswith("loco_ac."))
    print(f"  Loaded: {loaded_loco} loco, {loaded_actor} arm_actor, {loaded_critic} arm_critic params")

    # Freeze loco
    frozen = 0
    for name, p in net.named_parameters():
        if name.startswith("loco_ac."):
            p.requires_grad = False
            frozen += 1
    print(f"  Frozen {frozen} loco parameters")

    # RESET arm critic (Xavier init — key difference from Stage 2!)
    print("  Resetting arm_critic to Xavier init...")
    for m in net.arm_critic.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    print("  arm_critic: RESET (Xavier uniform)")

    # Set arm actor STD to 0.5 (moderate — position already learned)
    net.arm_actor.log_std.data.fill_(np.log(0.5))
    print(f"  arm_actor: LOADED (Stage 2 weights), log_std = log(0.5)")

    # Print checkpoint info
    for key in ["best_reward", "iteration", "curriculum_level"]:
        if key in ckpt:
            print(f"  Stage 2 {key}: {ckpt[key]}")

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
        # Resume from Stage 3 checkpoint
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
        # Restore dual optimizers
        ppo = ArmPPO(net, device, actor_lr=args_cli.actor_lr, critic_lr=args_cli.critic_lr)
        if "actor_optimizer" in ckpt:
            ppo.actor_opt.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            ppo.critic_opt.load_state_dict(ckpt["critic_optimizer"])
        print(f"  Resumed: iter={start_iter}, R={best_reward:.2f}, Lv={env.curr_level}")
    else:
        if args_cli.stage2_checkpoint is None:
            raise ValueError("--stage2_checkpoint required for fresh Stage 3 start")
        load_stage2_and_setup(net, args_cli.stage2_checkpoint, device)
        ppo = ArmPPO(net, device, actor_lr=args_cli.actor_lr, critic_lr=args_cli.critic_lr)

    # Logging
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\n[Log] {log_dir}")

    # Rollout buffers
    T = 24
    arm_obs_buf = torch.zeros(T, env.num_envs, ARM_OBS_DIM, device=device)
    arm_act_buf = torch.zeros(T, env.num_envs, ARM_ACT_DIM, device=device)
    arm_rew_buf = torch.zeros(T, env.num_envs, device=device)
    arm_val_buf = torch.zeros(T, env.num_envs, device=device)
    arm_lp_buf = torch.zeros(T, env.num_envs, device=device)
    done_buf = torch.zeros(T, env.num_envs, device=device)

    obs, _ = env.reset()

    # Early stop tracking
    ee_bad_counter = 0
    EE_BAD_THRESHOLD = 0.15   # position lost if avg EE > 15cm
    EE_BAD_MAX_ITERS = 3000   # abort after 3000 consecutive bad iterations

    print(f"\n{'='*80}")
    print("STARTING STAGE 3: ORIENTATION FINE-TUNE (Critic Reset)")
    print(f"  Loco: FROZEN (66->15)")
    print(f"  Arm Actor: LOADED (39->7, LR={args_cli.actor_lr})")
    print(f"  Arm Critic: RESET (39->1, LR={args_cli.critic_lr})")
    print(f"  Orient weight: {args_cli.orient_weight}")
    print(f"  Orient gate: {ORIENT_GATE_DISTANCE}m")
    print(f"  Early stop: EE > {EE_BAD_THRESHOLD}m for {EE_BAD_MAX_ITERS} iter")
    for i, lv in enumerate(CURRICULUM):
        orient_info = f"orient_thresh={lv.get('orient_threshold', 'OFF')}" if lv["use_orientation"] else "orient=OFF"
        print(f"  L{i}: {lv['description']} ({orient_info})")
    print(f"{'='*80}\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        # STD: start at 0.5, decay to floor 0.3
        progress = iteration / args_cli.max_iterations
        arm_std = max(0.5 - 0.2 * progress, 0.3)  # 0.5 -> 0.3 (gentle decay)
        net.arm_actor.log_std.data.fill_(np.log(arm_std))
        net.arm_actor.log_std.data.clamp_(min=np.log(0.3))

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

        # Compute returns
        with torch.no_grad():
            final_arm_obs = env.get_arm_obs()
            arm_next_val = net.arm_critic(final_arm_obs)

        arm_adv, arm_ret = ppo.gae(arm_rew_buf, arm_val_buf, done_buf, arm_next_val)

        # PPO update
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

        # Early stop check
        ee_dist = env.arm_reward_components.get("ee_dist", 0)
        if ee_dist > EE_BAD_THRESHOLD:
            ee_bad_counter += 1
            if ee_bad_counter >= EE_BAD_MAX_ITERS:
                print(f"\n[ABORT] Position lost! EE > {EE_BAD_THRESHOLD}m for {EE_BAD_MAX_ITERS} consecutive iterations.")
                print(f"  Last EE: {ee_dist:.3f}m. Stopping training.")
                break
        else:
            ee_bad_counter = 0

        # TensorBoard
        if iteration % 10 == 0:
            writer.add_scalar("reward/arm_step", mean_arm_reward, iteration)
            writer.add_scalar("reward/best", best_reward, iteration)
            writer.add_scalar("loss/actor", losses["a"], iteration)
            writer.add_scalar("loss/critic", losses["c"], iteration)
            writer.add_scalar("loss/entropy", losses["e"], iteration)
            writer.add_scalar("train/actor_lr", losses["actor_lr"], iteration)
            writer.add_scalar("train/critic_lr", losses["critic_lr"], iteration)
            writer.add_scalar("train/arm_std", np.exp(net.arm_actor.log_std.data.mean().item()), iteration)
            writer.add_scalar("curriculum/level", env.curr_level, iteration)
            writer.add_scalar("curriculum/validated_reaches", env.validated_reaches, iteration)
            writer.add_scalar("curriculum/timed_out", env.timed_out_targets, iteration)

            for key, val in env.arm_reward_components.items():
                writer.add_scalar(f"arm_reward/{key}", val, iteration)

            height = env.robot.data.root_pos_w[:, 2].mean().item()
            writer.add_scalar("robot/height", height, iteration)

            v_rate = env.validated_reaches / max(env.total_attempts, 1)
            writer.add_scalar("curriculum/validated_rate", v_rate, iteration)
            if env.window_count >= CURRICULUM_MIN_WINDOW:
                if env.window_count >= CURRICULUM_WINDOW_SIZE:
                    w_rate = env.window_buf.float().mean().item()
                else:
                    w_rate = env.window_buf[:env.window_count].float().mean().item()
                writer.add_scalar("curriculum/windowed_rate", w_rate, iteration)

            writer.add_scalar("early_stop/ee_bad_counter", ee_bad_counter, iteration)

        # Console log
        if iteration % 50 == 0:
            rc = env.arm_reward_components
            ee_dist = rc.get("ee_dist", 0)
            ee_spd = rc.get("ee_speed", 0)
            orient_err = rc.get("orient_err", 0)
            height = env.robot.data.root_pos_w[:, 2].mean().item()

            if env.window_count >= CURRICULUM_MIN_WINDOW:
                if env.window_count >= CURRICULUM_WINDOW_SIZE:
                    w_rate = env.window_buf.float().mean().item()
                else:
                    w_rate = env.window_buf[:env.window_count].float().mean().item()
            else:
                w_rate = 0.0

            orient_info = f"OrErr={orient_err:.2f}"
            lv = CURRICULUM[env.curr_level]
            if lv["use_orientation"]:
                orient_info += f"(thr={lv.get('orient_threshold', '?')})"

            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_arm_reward:.2f} Best={best_reward:.2f} "
                  f"Lv={env.curr_level} "
                  f"VR={env.validated_reaches} TO={env.timed_out_targets} "
                  f"WR={w_rate:.1%} "
                  f"EE={ee_dist:.3f} Spd={ee_spd:.3f} "
                  f"H={height:.3f} {orient_info} "
                  f"aLR={losses['actor_lr']:.1e} cLR={losses['critic_lr']:.1e} "
                  f"std={np.exp(net.arm_actor.log_std.data.mean().item()):.3f}"
                  f"{' [!POS]' if ee_bad_counter > 500 else ''}")

        # Save checkpoints
        save_dict = {
            "model": net.state_dict(),
            "actor_optimizer": ppo.actor_opt.state_dict(),
            "critic_optimizer": ppo.critic_opt.state_dict(),
            "iteration": iteration,
            "best_reward": best_reward,
            "curriculum_level": env.curr_level,
            "validated_reaches": env.validated_reaches,
            "timed_out_targets": env.timed_out_targets,
            "total_attempts": env.total_attempts,
            "stage2_checkpoint": args_cli.stage2_checkpoint,
        }

        if iteration % 500 == 0 and iteration > 0:
            path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save(save_dict, path)
            print(f"  [Save] {path}")

        # model_best: ONLY when EE < 0.08m (guard against position collapse)
        if mean_arm_reward > best_reward and ee_dist < 0.08:
            best_reward = mean_arm_reward
            save_dict["best_reward"] = best_reward
            path = os.path.join(log_dir, "model_best.pt")
            torch.save(save_dict, path)
            print(f"  [Best] R={best_reward:.2f} EE={ee_dist:.3f} OrErr={orient_err:.2f}")

    # Final save
    save_dict["iteration"] = min(iteration, args_cli.max_iterations - 1)
    save_dict["best_reward"] = best_reward
    path = os.path.join(log_dir, "model_final.pt")
    torch.save(save_dict, path)

    print(f"\n{'='*80}")
    print("STAGE 3 TRAINING COMPLETE")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Validated Reaches: {env.validated_reaches}")
    print(f"  Timed Out: {env.timed_out_targets}")
    if env.total_attempts > 0:
        print(f"  Validated Rate: {env.validated_reaches / env.total_attempts:.1%}")
    if ee_bad_counter >= EE_BAD_MAX_ITERS:
        print(f"  [!] Training ABORTED: position collapse detected")
    print(f"  Log Dir: {log_dir}")
    print(f"{'='*80}")

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
