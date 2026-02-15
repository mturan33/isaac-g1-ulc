#!/usr/bin/env python3
"""
Fair Comparison Benchmark: Stage 6 (Gaming) vs Stage 7 (Anti-Gaming)
=====================================================================
Runs arm reaching policies sequentially in the SAME standardized environment
with identical target sampling, workspace, and reach metrics.

Supports up to 3 policies:
  - S6 Unified (52->12 arm, .actor. naming, gaming checkpoint)
  - S6 Simplified (52->5 arm, .net. naming)
  - S7 Anti-Gaming (55->5 arm, .net. naming)

Outputs:
  - per_target.csv: Per-target metrics (initial_dist, displacement, reached, etc.)
  - summary.json: Aggregated comparison statistics
  - plots/: Matplotlib comparison figures (PNG + PDF)

Key standardizations:
  - Same loco policy (Stage 7 frozen, shared across all)
  - Same target sampling (absolute + min distance 0.12m)
  - Same workspace (spherical, radius 0.18-0.40m)
  - Same timeout (150 steps)
  - Same evaluation thresholds (pos < 0.06m, displacement >= 0.10m)
  - Each policy receives its NATIVE arm obs format (S6: 52, S7: 55)

Usage:
    .\\isaaclab.bat -p benchmark_s6_vs_s7.py \\
        --s6u_checkpoint logs/ulc/.../model_best.pt \\
        --s7_checkpoint logs/ulc/.../model_best.pt \\
        --num_envs 1 --steps 3000 --mode both

    # 3-way comparison:
    .\\isaaclab.bat -p benchmark_s6_vs_s7.py \\
        --s6u_checkpoint logs/ulc/.../model_best.pt \\
        --s6_checkpoint logs/ulc/.../model_best.pt \\
        --s7_checkpoint logs/ulc/.../model_best.pt \\
        --num_envs 1 --steps 3000 --mode both

V1 (2026-02-15): Initial benchmark script for ICRA 2026 paper.
V2 (2026-02-15): Added S6 unified support, 3-way comparison, fixed initial_dist=0 bug.
"""

from __future__ import annotations

import argparse
import os
import json
import csv
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# ============================================================================
# CLI ARGUMENTS (before Isaac imports)
# ============================================================================

parser = argparse.ArgumentParser(description="S6 vs S7 Benchmark - ICRA 2026")
parser.add_argument("--s6_checkpoint", type=str, default=None,
                    help="Stage 6 SIMPLIFIED (52->5 arm) checkpoint path")
parser.add_argument("--s6u_checkpoint", type=str, default=None,
                    help="Stage 6 UNIFIED (52->12 arm+finger) checkpoint path (gaming)")
parser.add_argument("--s7_checkpoint", type=str, required=True,
                    help="Stage 7 (anti-gaming) checkpoint path")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments (default: 1)")
parser.add_argument("--steps", type=int, default=3000,
                    help="Steps per policy per mode (default: 3000)")
parser.add_argument("--mode", type=str, default="both",
                    choices=["standing", "walking", "both"],
                    help="Benchmark mode (default: both)")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: benchmarks/s6_vs_s7_<timestamp>)")
parser.add_argument("--pos_threshold", type=float, default=0.06,
                    help="Position reach evaluation threshold in meters (default: 0.06)")
parser.add_argument("--min_displacement", type=float, default=0.10,
                    help="Min displacement for validated reach in meters (default: 0.10)")
parser.add_argument("--max_target_steps", type=int, default=150,
                    help="Max steps per target before timeout (default: 150)")
parser.add_argument("--min_target_dist", type=float, default=0.12,
                    help="Min distance from EE for new target in meters (default: 0.12)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
parser.add_argument("--no_plots", action="store_true", default=False,
                    help="Skip plot generation")
parser.add_argument("--env_spacing", type=float, default=5.0,
                    help="Spacing between environments")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# ============================================================================
# CONSTANTS
# ============================================================================
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
PALM_FORWARD_OFFSET = 0.08
SHOULDER_OFFSET = torch.tensor([0.0, -0.174, 0.259])

G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

FINGER_JOINT_NAMES = [
    "right_zero_joint", "right_one_joint", "right_two_joint",
    "right_three_joint", "right_four_joint", "right_five_joint",
    "right_six_joint",
]

# Benchmark mode configurations
MODE_CONFIGS = {
    "standing": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.18, 0.40),
        "description": "Standing still, arm reaching"
    },
    "walking": {
        "vx_range": (0.2, 0.4),
        "vy_range": (-0.05, 0.05),
        "vyaw_range": (-0.1, 0.1),
        "workspace_radius": (0.18, 0.40),
        "description": "Walking with arm reaching"
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to euler angles. Handles both xyzw and wxyz."""
    # Stage 6 unified play uses wxyz, Stage 7 uses xyzw
    # Both scripts produce the same obs because the convention is self-consistent
    # For benchmark, use xyzw (matches Stage 7 play)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


def get_palm_forward(quat: torch.Tensor) -> torch.Tensor:
    """Get palm forward direction from quaternion (wxyz format)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=-1)


def compute_orientation_error(palm_quat: torch.Tensor,
                              target_dir: torch.Tensor) -> torch.Tensor:
    """Compute angle between palm forward and target direction."""
    forward = get_palm_forward(palm_quat)
    dot = torch.clamp((forward * target_dir).sum(dim=-1), -1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# NETWORK DEFINITIONS
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion actor: 57 obs -> 12 leg actions. [512,256,128]+LN+ELU."""
    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, x):
        return self.net(x)


class LocoCritic(nn.Module):
    def __init__(self, num_obs=57, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ArmActor(nn.Module):
    """Generic arm actor. [256,256,128]+ELU."""
    def __init__(self, num_obs, num_act, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, x):
        return self.net(x)


class ArmCritic(nn.Module):
    def __init__(self, num_obs, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualACStage6(nn.Module):
    """Stage 6: LocoActor(57->12) + ArmActor(52->5)
    NOTE: S6 'simplified' checkpoint already has 5-dim arm output (not 12).
    """
    def __init__(self):
        super().__init__()
        self.loco_actor = LocoActor(57, 12)
        self.loco_critic = LocoCritic(57)
        self.arm_actor = ArmActor(52, 5)
        self.arm_critic = ArmCritic(52)

    def get_actions(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)  # [N, 5]
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        loco_act = torch.distributions.Normal(loco_mean, loco_std).sample()
        arm_act = torch.distributions.Normal(arm_mean, arm_std).sample()
        return loco_act, arm_act


class DualACStage6Unified(nn.Module):
    """Stage 6 UNIFIED: LocoActor(57->12) + ArmActor(52->12, includes 7 finger)
    Uses .actor. naming in checkpoint (needs remap).
    Has unified critic (109 input) instead of separate loco/arm critics.
    """
    def __init__(self):
        super().__init__()
        self.loco_actor = LocoActor(57, 12)
        self.loco_critic = LocoCritic(57)  # Dummy, unified critic won't load here
        self.arm_actor = ArmActor(52, 12)  # 5 arm + 7 finger
        self.arm_critic = ArmCritic(52)    # Dummy

    def get_actions(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_out = self.arm_actor(arm_obs)  # [N, 12]
        arm_mean = arm_out[:, :5]  # Only first 5 (arm joints, ignore fingers)
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std[:5].clamp(-2, 1).exp()
        loco_act = torch.distributions.Normal(loco_mean, loco_std).sample()
        arm_act = torch.distributions.Normal(arm_mean, arm_std).sample()
        return loco_act, arm_act


class DualACStage7(nn.Module):
    """Stage 7: LocoActor(57->12) + ArmActor(55->5)"""
    def __init__(self):
        super().__init__()
        self.loco_actor = LocoActor(57, 12)
        self.loco_critic = LocoCritic(57)
        self.arm_actor = ArmActor(55, 5)
        self.arm_critic = ArmCritic(55)

    def get_actions(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        loco_act = torch.distributions.Normal(loco_mean, loco_std).sample()
        arm_act = torch.distributions.Normal(arm_mean, arm_std).sample()
        return loco_act, arm_act


def remap_s6_keys(state_dict):
    """Remap Stage 6 checkpoint keys from .actor. to .net. naming."""
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        new_k = new_k.replace("loco_actor.actor.", "loco_actor.net.")
        new_k = new_k.replace("arm_actor.actor.", "arm_actor.net.")
        new_k = new_k.replace("critic.critic.", "loco_critic.net.")
        remapped[new_k] = v
    return remapped


# ============================================================================
# ENVIRONMENT
# ============================================================================

@configclass
class BenchSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=10.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0, "right_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.0, "right_elbow_pitch_joint": 0.0,
                "left_elbow_roll_joint": 0.0, "right_elbow_roll_joint": 0.0,
                "torso_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=150.0, damping=15.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=50.0, damping=5.0,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*zero.*", ".*one.*", ".*two.*", ".*three.*",
                    ".*four.*", ".*five.*", ".*six.*"
                ],
                stiffness=20.0, damping=2.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0, damping=10.0,
            ),
        },
    )


@configclass
class BenchEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 120.0  # Long episodes for benchmark
    action_space = 17  # 12 leg + 5 arm
    observation_space = 57  # Dummy, actual obs are computed manually
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = BenchSceneCfg(num_envs=1, env_spacing=5.0)


class BenchmarkEnv(DirectRLEnv):
    """Standardized environment for fair policy comparison."""
    cfg: BenchEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_names = self.robot.joint_names
        self.leg_idx = torch.tensor(
            [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
            device=self.device
        )
        self.arm_idx = torch.tensor(
            [joint_names.index(n) for n in ARM_JOINT_NAMES if n in joint_names],
            device=self.device
        )
        self.finger_idx = torch.tensor(
            [joint_names.index(n) for n in FINGER_JOINT_NAMES if n in joint_names],
            device=self.device
        )

        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )
        self.default_arm = self.robot.data.default_joint_pos[0, self.arm_idx].clone()

        # Finger limits
        joint_limits = self.robot.root_physx_view.get_dof_limits()
        self.finger_lower = torch.tensor(
            [joint_limits[0, i, 0].item() for i in self.finger_idx], device=self.device
        )
        self.finger_upper = torch.tensor(
            [joint_limits[0, i, 1].item() for i in self.finger_idx], device=self.device
        )

        # Find palm body
        body_names = self.robot.body_names
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right_palm" in name.lower():
                self.palm_idx = i
                break
        if self.palm_idx is None:
            self.palm_idx = len(body_names) - 1
            print("[WARNING] right_palm_link not found, using last body")

        self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

        # Commands and targets
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body[:, 2] = -1.0  # Palm down

        # State
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Anti-gaming tracking buffers (used for BOTH policies)
        self.ee_pos_at_spawn = torch.zeros(self.num_envs, 3, device=self.device)
        self.steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.initial_dist = torch.zeros(self.num_envs, device=self.device)
        self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Benchmark config
        self.eval_pos_threshold = args.pos_threshold
        self.eval_min_displacement = args.min_displacement
        self.max_target_steps = args.max_target_steps
        self.min_target_dist = args.min_target_dist
        self.workspace_radius = (0.18, 0.40)

        # Current mode config
        self.mode_cfg = None

        print(f"\n[BenchmarkEnv] Initialized:")
        print(f"  Envs: {self.num_envs}")
        print(f"  Eval pos threshold: {self.eval_pos_threshold:.3f}m")
        print(f"  Min displacement: {self.eval_min_displacement:.3f}m")
        print(f"  Max target steps: {self.max_target_steps}")
        print(f"  Min target distance: {self.min_target_dist:.3f}m")
        print(f"  Leg: {len(self.leg_idx)}, Arm: {len(self.arm_idx)}, "
              f"Finger: {len(self.finger_idx)}")

    @property
    def robot(self):
        return self.scene["robot"]

    def set_mode(self, mode_name: str):
        """Set benchmark mode (standing/walking)."""
        self.mode_cfg = MODE_CONFIGS[mode_name].copy()
        print(f"\n[BenchmarkEnv] Mode: {mode_name} ({self.mode_cfg['description']})")

    def full_reset(self):
        """Complete environment reset for between-policy switching."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        self.prev_ee_pos[:] = 0
        # Step once to get valid physics state (robot settled at spawn pose)
        obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, 17, device=self.device))
        # Now EE position is valid - initialize tracking buffers
        ee_pos, _ = self._compute_palm_ee()
        self.prev_ee_pos = ee_pos.clone()
        # Re-sample targets with valid EE position (fixes initial_dist=0 bug)
        # This ensures ee_pos_at_spawn and initial_dist are computed from
        # the actual EE position, not from zeros
        self._sample_targets(env_ids)
        # Update ee_pos_at_spawn with body-frame EE
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
        self.ee_pos_at_spawn = ee_body.clone()

    def _compute_palm_ee(self):
        """Compute palm EE position and quaternion."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_forward = get_palm_forward(palm_quat)
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward
        return ee_pos, palm_quat

    def _sample_targets(self, env_ids):
        """STANDARDIZED absolute sampling with min distance enforcement."""
        n = len(env_ids)
        ws = self.workspace_radius

        azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
        elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
        radius = torch.empty(n, device=self.device).uniform_(*ws)

        x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
        y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
        z = radius * torch.sin(elevation) + self.shoulder_offset[2]

        target_body = torch.stack([
            x.clamp(0.05, 0.55),
            y.clamp(-0.55, 0.10),
            z.clamp(-0.25, 0.55)
        ], dim=-1)

        # Get current EE in body frame
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        ee_world, _ = self._compute_palm_ee()
        ee_world = ee_world[env_ids]
        current_ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

        # Enforce min distance from current EE
        dist_to_ee = torch.norm(target_body - current_ee_body, dim=-1)
        too_close = dist_to_ee < self.min_target_dist
        if too_close.any():
            direction = target_body - current_ee_body
            direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            direction = direction / direction_norm
            pushed = current_ee_body + self.min_target_dist * direction
            pushed[:, 0] = pushed[:, 0].clamp(0.05, 0.55)
            pushed[:, 1] = pushed[:, 1].clamp(-0.55, 0.10)
            pushed[:, 2] = pushed[:, 2].clamp(-0.25, 0.55)
            target_body = torch.where(
                too_close.unsqueeze(-1).expand_as(target_body), pushed, target_body
            )

        self.target_pos_body[env_ids] = target_body
        self.target_orient_body[env_ids, 0] = 0.0
        self.target_orient_body[env_ids, 1] = 0.0
        self.target_orient_body[env_ids, 2] = -1.0

        # Record spawn state
        self.ee_pos_at_spawn[env_ids] = current_ee_body
        self.steps_since_spawn[env_ids] = 0
        self.initial_dist[env_ids] = torch.norm(
            target_body - current_ee_body, dim=-1
        ).clamp(min=0.01)
        self.already_reached[env_ids] = False

    def _sample_commands(self, env_ids):
        """Sample velocity commands and arm targets."""
        n = len(env_ids)
        cfg = self.mode_cfg
        self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*cfg["vx_range"])
        self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*cfg["vy_range"])
        self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*cfg["vyaw_range"])
        self.height_cmd[env_ids] = HEIGHT_DEFAULT
        self._sample_targets(env_ids)

    def get_loco_obs(self):
        """57-dim loco obs. Identical for both policies."""
        robot = self.robot
        quat = robot.data.root_quat_w
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)
        gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        joint_pos = robot.data.joint_pos[:, self.leg_idx]
        joint_vel = robot.data.joint_vel[:, self.leg_idx]
        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.phase),
            torch.cos(2 * np.pi * self.phase)
        ], dim=-1)
        torso_euler = quat_to_euler_xyz(quat)
        obs = torch.cat([
            lin_vel_b,              # 3
            ang_vel_b,              # 3
            proj_gravity,           # 3
            joint_pos,              # 12
            joint_vel,              # 12
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,           # 3
            gait_phase,             # 2
            self.prev_leg_actions,  # 12
            self.torso_cmd,         # 3
            torso_euler,            # 3
        ], dim=-1)
        return obs.clamp(-10, 10).nan_to_num()

    def _get_arm_obs_common(self, obs_pos_th: float, obs_orient_th: float):
        """Compute 52-dim base arm obs. Used by both S6 and S7 obs methods."""
        robot = self.robot
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        arm_pos = robot.data.joint_pos[:, self.arm_idx]
        arm_vel = robot.data.joint_vel[:, self.arm_idx] * 0.1
        finger_pos = robot.data.joint_pos[:, self.finger_idx]

        ee_world, palm_quat = self._compute_palm_ee()
        ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)
        ee_vel_world = (ee_world - self.prev_ee_pos) / 0.02
        ee_vel_body = quat_apply_inverse(root_quat, ee_vel_world)

        finger_normalized = (
            (finger_pos - self.finger_lower)
            / (self.finger_upper - self.finger_lower + 1e-6)
        )
        gripper_closed_ratio = finger_normalized.mean(dim=-1, keepdim=True)
        finger_vel = robot.data.joint_vel[:, self.finger_idx]
        grip_force = (
            finger_vel.abs().mean(dim=-1, keepdim=True) * gripper_closed_ratio
        ).clamp(0, 1)

        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist_to_target = torch.norm(ee_world - target_world, dim=-1, keepdim=True)
        contact_detected = (dist_to_target < 0.08).float()

        pos_error = self.target_pos_body - ee_body
        pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5

        palm_forward = get_palm_forward(palm_quat)
        down_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).expand(self.num_envs, -1)
        dot = (palm_forward * down_vec).sum(dim=-1)
        angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
        orient_error = (angle / np.pi).unsqueeze(-1)

        # target_reached using NATIVE threshold for this policy
        target_reached = (
            (dist_to_target < obs_pos_th) &
            (orient_error * np.pi < obs_orient_th)
        ).float()

        current_height = root_pos[:, 2:3]
        height_cmd_obs = self.height_cmd.unsqueeze(-1)
        height_err = (height_cmd_obs - current_height) / 0.4

        estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
        object_in_hand = torch.zeros(self.num_envs, 1, device=self.device)
        target_orient_obs = self.target_orient_body

        lin_vel_b = quat_apply_inverse(root_quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w)
        lin_vel_xy = lin_vel_b[:, :2]
        ang_vel_z = ang_vel_b[:, 2:3]

        base_obs = torch.cat([
            arm_pos,              # 5
            arm_vel,              # 5
            finger_pos,           # 7
            ee_body,              # 3
            ee_vel_body,          # 3
            palm_quat,            # 4
            grip_force,           # 1
            gripper_closed_ratio, # 1
            contact_detected,     # 1
            self.target_pos_body, # 3
            pos_error,            # 3
            pos_dist,             # 1
            orient_error,         # 1
            target_reached,       # 1
            height_cmd_obs,       # 1
            current_height,       # 1
            height_err,           # 1
            estimated_load,       # 3
            object_in_hand,       # 1
            target_orient_obs,    # 3
            lin_vel_xy,           # 2
            ang_vel_z,            # 1
        ], dim=-1)  # Total: 52

        # Return extra data needed for metrics
        return base_obs, ee_body, ee_world, dist_to_target.squeeze(-1)

    def get_arm_obs_s6(self, obs_pos_th: float, obs_orient_th: float):
        """52-dim arm obs for Stage 6 policy."""
        base_obs, ee_body, ee_world, dist = self._get_arm_obs_common(
            obs_pos_th, obs_orient_th
        )
        return base_obs.clamp(-10, 10).nan_to_num(), ee_body, ee_world, dist

    def get_arm_obs_s7(self, obs_pos_th: float, obs_orient_th: float):
        """55-dim arm obs for Stage 7 policy (52 base + 3 anti-gaming)."""
        base_obs, ee_body, ee_world, dist = self._get_arm_obs_common(
            obs_pos_th, obs_orient_th
        )
        # 3 anti-gaming observations
        max_steps = float(self.max_target_steps)
        steps_norm = (self.steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)
        ee_displacement = torch.norm(
            ee_body - self.ee_pos_at_spawn, dim=-1, keepdim=True
        )
        initial_dist_obs = self.initial_dist.unsqueeze(-1) / 0.5

        full_obs = torch.cat([
            base_obs, steps_norm, ee_displacement, initial_dist_obs
        ], dim=-1)  # Total: 55
        return full_obs.clamp(-10, 10).nan_to_num(), ee_body, ee_world, dist

    def _pre_physics_step(self, actions):
        """Apply actions: leg (12) + arm (5). Fingers stay at default."""
        self.actions = actions.clone()
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5
        target_pos[:, self.finger_idx] = self.finger_lower  # Fingers open (default)

        self.robot.set_joint_position_target(target_pos)
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {"policy": self.get_loco_obs()}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        height = self.robot.data.root_pos_w[:, 2]
        gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(self.robot.data.root_quat_w, gravity_vec)
        fallen = (height < 0.3) | (height > 1.2)
        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7
        terminated = fallen | bad_orientation
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        default_pos = torch.tensor(
            [[0.0, 0.0, 0.8]], device=self.device
        ).expand(n, -1).clone()
        default_quat = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], device=self.device
        ).expand(n, -1)
        self.robot.write_root_pose_to_sim(
            torch.cat([default_pos, default_quat], dim=-1), env_ids
        )
        self.robot.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids
        )
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(
            default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids
        )
        if self.mode_cfg is not None:
            self._sample_commands(env_ids)
        self.phase[env_ids] = torch.rand(n, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0
        self.prev_ee_pos[env_ids] = 0
        self.ee_pos_at_spawn[env_ids] = 0
        self.steps_since_spawn[env_ids] = 0
        self.initial_dist[env_ids] = 0
        self.already_reached[env_ids] = False


# ============================================================================
# BENCHMARK SESSION
# ============================================================================

def run_policy_session(env, net, policy_name, get_arm_obs_fn,
                       obs_pos_th, obs_orient_th, steps, mode_name):
    """Run one policy for N steps and collect per-target metrics."""
    device = env.device
    per_target = []
    target_id = 0
    cumulative_pos_reaches = []
    cumulative_val_reaches = []
    total_pos = 0
    total_val = 0
    total_timeouts = 0
    falls = 0

    print(f"\n{'='*60}")
    print(f"  {policy_name} | Mode: {mode_name} | {steps} steps")
    print(f"{'='*60}")

    with torch.no_grad():
        for step in range(steps):
            # Get observations
            loco_obs = env.get_loco_obs()
            arm_obs, ee_body, ee_world, dist = get_arm_obs_fn(obs_pos_th, obs_orient_th)

            # Get actions
            leg_act, arm_act = net.get_actions(loco_obs, arm_obs, deterministic=True)
            actions = torch.cat([leg_act, arm_act], dim=-1)

            # Compute metrics BEFORE step
            ee_displacement = torch.norm(
                ee_body - env.ee_pos_at_spawn, dim=-1
            )
            orient_err = compute_orientation_error(
                env.robot.data.body_quat_w[:, env.palm_idx],
                env.target_orient_body
            )

            # Check position reach (STANDARDIZED threshold)
            pos_reached = dist < env.eval_pos_threshold
            # Check validated reach (STANDARDIZED)
            validated = (
                pos_reached &
                (ee_displacement >= env.eval_min_displacement) &
                (env.steps_since_spawn <= env.max_target_steps)
            )
            # Check timeout
            timed_out = (
                (env.steps_since_spawn >= env.max_target_steps) &
                ~env.already_reached
            )

            # Handle NEW validated reaches (record BEFORE resample)
            new_val = validated & ~env.already_reached
            if new_val.any():
                val_ids = torch.where(new_val)[0]
                for idx in val_ids:
                    i = idx.item()
                    per_target.append({
                        "target_id": target_id,
                        "policy": policy_name,
                        "mode": mode_name,
                        "initial_dist": env.initial_dist[i].item(),
                        "final_dist": dist[i].item(),
                        "ee_displacement": ee_displacement[i].item(),
                        "time_steps": env.steps_since_spawn[i].item(),
                        "position_reached": True,
                        "validated_reached": True,
                        "timed_out": False,
                        "orient_err": orient_err[i].item(),
                    })
                    target_id += 1
                    total_pos += 1
                    total_val += 1
                env.already_reached[val_ids] = True
                env._sample_targets(val_ids)

            # Handle position-only reaches that are NOT validated
            new_pos_only = pos_reached & ~validated & ~env.already_reached
            if new_pos_only.any():
                pos_ids = torch.where(new_pos_only)[0]
                for idx in pos_ids:
                    i = idx.item()
                    per_target.append({
                        "target_id": target_id,
                        "policy": policy_name,
                        "mode": mode_name,
                        "initial_dist": env.initial_dist[i].item(),
                        "final_dist": dist[i].item(),
                        "ee_displacement": ee_displacement[i].item(),
                        "time_steps": env.steps_since_spawn[i].item(),
                        "position_reached": True,
                        "validated_reached": False,
                        "timed_out": False,
                        "orient_err": orient_err[i].item(),
                    })
                    target_id += 1
                    total_pos += 1
                env.already_reached[pos_ids] = True
                env._sample_targets(pos_ids)

            # Handle timeouts
            if timed_out.any():
                to_ids = torch.where(timed_out)[0]
                for idx in to_ids:
                    i = idx.item()
                    per_target.append({
                        "target_id": target_id,
                        "policy": policy_name,
                        "mode": mode_name,
                        "initial_dist": env.initial_dist[i].item(),
                        "final_dist": dist[i].item(),
                        "ee_displacement": ee_displacement[i].item(),
                        "time_steps": env.steps_since_spawn[i].item(),
                        "position_reached": False,
                        "validated_reached": False,
                        "timed_out": True,
                        "orient_err": orient_err[i].item(),
                    })
                    target_id += 1
                    total_timeouts += 1
                env._sample_targets(to_ids)

            # Step environment
            env.steps_since_spawn += 1
            obs, reward, terminated, truncated, info = env.step(actions)

            # Update prev actions
            env.prev_leg_actions = leg_act.clone()
            env.prev_arm_actions = arm_act.clone()
            ee_pos_now, _ = env._compute_palm_ee()
            env.prev_ee_pos = ee_pos_now.clone()

            # Handle falls (reset)
            done = terminated | truncated
            if done.any():
                falls += done.sum().item()

            # Track cumulative
            cumulative_pos_reaches.append(total_pos)
            cumulative_val_reaches.append(total_val)

            # Progress print every 500 steps
            if (step + 1) % 500 == 0:
                total_targets = len(per_target)
                pos_rate = (total_pos / total_targets * 100) if total_targets > 0 else 0
                val_rate = (total_val / total_targets * 100) if total_targets > 0 else 0
                print(f"  [{step+1:5d}/{steps}] Targets: {total_targets}, "
                      f"Pos: {total_pos} ({pos_rate:.1f}%), "
                      f"Val: {total_val} ({val_rate:.1f}%), "
                      f"Timeout: {total_timeouts}, Falls: {falls}")

    # Compute summary
    total_targets = len(per_target)
    pos_rate = (total_pos / total_targets * 100) if total_targets > 0 else 0
    val_rate = (total_val / total_targets * 100) if total_targets > 0 else 0
    to_rate = (total_timeouts / total_targets * 100) if total_targets > 0 else 0

    val_times = [t["time_steps"] for t in per_target if t["validated_reached"]]
    mean_time = np.mean(val_times) if val_times else 0
    std_time = np.std(val_times) if val_times else 0

    all_disps = [t["ee_displacement"] for t in per_target]
    mean_disp = np.mean(all_disps) if all_disps else 0
    std_disp = np.std(all_disps) if all_disps else 0

    all_init = [t["initial_dist"] for t in per_target]
    mean_init = np.mean(all_init) if all_init else 0

    all_final = [t["final_dist"] for t in per_target]
    mean_final = np.mean(all_final) if all_final else 0

    summary = {
        "policy": policy_name,
        "mode": mode_name,
        "total_steps": steps,
        "total_targets": total_targets,
        "position_reach_count": total_pos,
        "position_reach_rate": round(pos_rate, 2),
        "validated_reach_count": total_val,
        "validated_reach_rate": round(val_rate, 2),
        "timeout_count": total_timeouts,
        "timeout_rate": round(to_rate, 2),
        "mean_time_to_reach": round(mean_time, 1),
        "std_time_to_reach": round(std_time, 1),
        "mean_ee_displacement": round(mean_disp, 4),
        "std_ee_displacement": round(std_disp, 4),
        "mean_initial_distance": round(mean_init, 4),
        "mean_final_distance": round(mean_final, 4),
        "validated_per_1k_steps": round(total_val / steps * 1000, 2),
        "falls": falls,
    }

    print(f"\n  --- {policy_name} | {mode_name} SONUC ---")
    print(f"  Toplam hedef: {total_targets}")
    print(f"  Position-only: {total_pos} ({pos_rate:.1f}%)")
    print(f"  Validated:     {total_val} ({val_rate:.1f}%)")
    print(f"  Timeout:       {total_timeouts} ({to_rate:.1f}%)")
    print(f"  Ort. displacement: {mean_disp:.4f}m")
    print(f"  Ort. time-to-reach: {mean_time:.1f} steps")
    print(f"  Val/1K steps:  {total_val / steps * 1000:.2f}")
    print(f"  Dusme:         {falls}")

    return {
        "per_target": per_target,
        "cumulative_pos": cumulative_pos_reaches,
        "cumulative_val": cumulative_val_reaches,
        "summary": summary,
    }


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_per_target_csv(all_results, output_dir):
    """Save per-target data to CSV."""
    path = os.path.join(output_dir, "per_target.csv")
    all_targets = []
    for r in all_results:
        all_targets.extend(r["per_target"])

    if not all_targets:
        print("[WARNING] No target data to save")
        return

    fieldnames = list(all_targets[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_targets:
            writer.writerow(row)
    print(f"[OUTPUT] per_target.csv: {len(all_targets)} kayit -> {path}")


def save_summary_json(all_results, output_dir):
    """Save summary statistics to JSON."""
    path = os.path.join(output_dir, "summary.json")
    summaries = [r["summary"] for r in all_results]

    # Also add benchmark config
    output = {
        "benchmark_config": {
            "eval_pos_threshold": args.pos_threshold,
            "eval_min_displacement": args.min_displacement,
            "max_target_steps": args.max_target_steps,
            "min_target_distance": args.min_target_dist,
            "seed": args.seed,
            "steps_per_policy": args.steps,
            "num_envs": args.num_envs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "s6_checkpoint": args.s6_checkpoint,
            "s6u_checkpoint": args.s6u_checkpoint,
            "s7_checkpoint": args.s7_checkpoint,
        },
        "results": summaries,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[OUTPUT] summary.json -> {path}")


def generate_plots(all_results, output_dir):
    """Generate comparison plots. Supports 2 or 3 policies dynamically."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    # Group results by mode
    results_by_mode = {}
    for r in all_results:
        mode = r["summary"]["mode"]
        policy = r["summary"]["policy"]
        if mode not in results_by_mode:
            results_by_mode[mode] = {}
        results_by_mode[mode][policy] = r

    # Dynamic color map per policy
    POLICY_COLORS = {
        "S6u_gaming":      {"main": "#E63946", "light": "#F4A3A8", "label": "S6u (Gaming)"},
        "S6s_simplified":  {"main": "#FF9F1C", "light": "#FFD699", "label": "S6s (Simplified)"},
        "S6_gaming":       {"main": "#E63946", "light": "#F4A3A8", "label": "S6 (Gaming)"},
        "S7_antigaming":   {"main": "#457B9D", "light": "#A8DADC", "label": "S7 (Anti-Gaming)"},
    }

    # Get unique policy names from results
    policy_names = []
    for mode_data in results_by_mode.values():
        for pname in mode_data.keys():
            if pname not in policy_names:
                policy_names.append(pname)

    def _get_color(pname, variant="main"):
        if pname in POLICY_COLORS:
            return POLICY_COLORS[pname][variant]
        # Fallback
        fallback = {"main": "#2A9D8F", "light": "#B7E4C7", "label": pname}
        return fallback[variant]

    def _get_label(pname):
        if pname in POLICY_COLORS:
            return POLICY_COLORS[pname]["label"]
        return pname

    modes = list(results_by_mode.keys())
    n_policies = len(policy_names)

    # ---- PLOT 1: Reach Rate Comparison (Grouped Bar) ----
    fig, ax = plt.subplots(figsize=(max(10, 4 * len(modes)), 6))
    n_modes = len(modes)
    x = np.arange(n_modes)
    # 2 bars per policy (pos-only + validated), plus gaps
    total_bars = n_policies * 2
    width = 0.8 / total_bars
    offsets = np.linspace(-(total_bars - 1) * width / 2,
                          (total_bars - 1) * width / 2, total_bars)

    legend_added = set()
    for i, mode in enumerate(modes):
        data = results_by_mode[mode]
        bar_idx = 0
        for pname in policy_names:
            if pname not in data:
                bar_idx += 2
                continue
            pdata = data[pname]["summary"]
            color_main = _get_color(pname, "main")
            color_light = _get_color(pname, "light")
            plabel = _get_label(pname)

            # Position-only bar
            lbl_pos = f"{plabel} pos-only" if f"{plabel}_pos" not in legend_added else ""
            ax.bar(x[i] + offsets[bar_idx], pdata["position_reach_rate"],
                   width, color=color_light, edgecolor=color_main, linewidth=1.5,
                   label=lbl_pos)
            ax.text(x[i] + offsets[bar_idx], pdata["position_reach_rate"] + 1,
                    f'{pdata["position_reach_rate"]:.0f}%',
                    ha="center", va="bottom", fontsize=8)
            if lbl_pos:
                legend_added.add(f"{plabel}_pos")

            # Validated bar
            lbl_val = f"{plabel} validated" if f"{plabel}_val" not in legend_added else ""
            ax.bar(x[i] + offsets[bar_idx + 1], pdata["validated_reach_rate"],
                   width, color=color_main, edgecolor=color_main, linewidth=1.5,
                   label=lbl_val)
            ax.text(x[i] + offsets[bar_idx + 1], pdata["validated_reach_rate"] + 1,
                    f'{pdata["validated_reach_rate"]:.0f}%',
                    ha="center", va="bottom", fontsize=8)
            if lbl_val:
                legend_added.add(f"{plabel}_val")

            bar_idx += 2

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.set_ylabel("Reach Rate (%)")
    ax.set_title("Reach Rate Comparison: Position-Only vs Validated")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(plot_dir, f"reach_rate_comparison.{ext}"))
    plt.close(fig)
    print(f"[PLOT] reach_rate_comparison saved")

    # ---- PLOT 2: EE Displacement Distribution ----
    first_mode = modes[0]
    data = results_by_mode[first_mode]
    n_plots = len(data)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, pname in enumerate(policy_names):
        if pname not in data:
            continue
        val = data[pname]
        disps = [t["ee_displacement"] for t in val["per_target"]]
        if not disps:
            continue
        color = _get_color(pname, "main")
        label = _get_label(pname)
        ax = axes[ax_idx]
        ax.hist(disps, bins=25, range=(0, 0.5), color=color, alpha=0.7, edgecolor="white")
        ax.axvline(x=args.min_displacement, color="black", linestyle="--",
                  linewidth=2, label=f"Threshold ({args.min_displacement}m)")
        mean_d = np.mean(disps)
        ax.axvline(x=mean_d, color=color, linestyle=":", linewidth=1.5,
                  label=f"Mean: {mean_d:.3f}m")
        ax.set_title(f"{label} ({first_mode})")
        ax.set_xlabel("EE Displacement (m)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle("End-Effector Displacement Distribution", fontsize=16, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(plot_dir, f"displacement_distribution.{ext}"))
    plt.close(fig)
    print(f"[PLOT] displacement_distribution saved")

    # ---- PLOT 3: Cumulative Reaches Over Time ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode in modes:
        data = results_by_mode[mode]
        for pname in policy_names:
            if pname not in data:
                continue
            val = data[pname]
            color_main = _get_color(pname, "main")
            color_light = _get_color(pname, "light")
            label = _get_label(pname)
            steps_arr = np.arange(len(val["cumulative_pos"]))
            ax.plot(steps_arr, val["cumulative_pos"],
                   color=color_light, linestyle="--", linewidth=1.5,
                   label=f"{label} pos-only ({mode})")
            ax.plot(steps_arr, val["cumulative_val"],
                   color=color_main, linestyle="-", linewidth=2,
                   label=f"{label} validated ({mode})")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Reaches")
    ax.set_title("Cumulative Reaches Over Time")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(plot_dir, f"cumulative_reaches.{ext}"))
    plt.close(fig)
    print(f"[PLOT] cumulative_reaches saved")

    # ---- PLOT 4: Time-to-Reach Histogram ----
    first_mode = modes[0]
    data = results_by_mode[first_mode]
    n_plots = len(data)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax_idx, pname in enumerate(policy_names):
        if pname not in data:
            continue
        val = data[pname]
        times = [t["time_steps"] for t in val["per_target"] if t["validated_reached"]]
        color = _get_color(pname, "main")
        label = _get_label(pname)
        ax = axes[ax_idx]
        if times:
            ax.hist(times, bins=15, range=(0, args.max_target_steps),
                   color=color, alpha=0.7, edgecolor="white")
            mean_t = np.mean(times)
            ax.axvline(x=mean_t, color="black", linestyle="--",
                      linewidth=2, label=f"Mean: {mean_t:.0f} steps")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No validated reaches",
                   transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_title(f"{label} ({first_mode})")
        ax.set_xlabel("Steps to Reach")
    axes[0].set_ylabel("Count")
    fig.suptitle("Time to Validated Reach", fontsize=16, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(plot_dir, f"time_to_reach_histogram.{ext}"))
    plt.close(fig)
    print(f"[PLOT] time_to_reach_histogram saved")

    # ---- PLOT 5: Timeout Rate ----
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(policy_names) * len(modes)), 5))
    labels = []
    values = []
    colors = []
    for mode in modes:
        data = results_by_mode[mode]
        for pname in policy_names:
            if pname not in data:
                continue
            plabel = _get_label(pname)
            labels.append(f"{plabel}\n{mode}")
            values.append(data[pname]["summary"]["timeout_rate"])
            colors.append(_get_color(pname, "main"))

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{val:.0f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Timeout Rate (%)")
    ax.set_title("Target Timeout Rate Comparison")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(plot_dir, f"timeout_rate.{ext}"))
    plt.close(fig)
    print(f"[PLOT] timeout_rate saved")


def print_comparison_table(all_results):
    """Print final side-by-side comparison table."""
    n_cols = len(all_results)
    col_width = 22
    label_width = 28
    total_width = label_width + 3 + (col_width + 3) * n_cols

    print(f"\n{'=' * total_width}")
    print(f"  FINAL KARSILASTIRMA TABLOSU")
    print(f"{'=' * total_width}")
    print(f"{'Metrik':<{label_width}} | ", end="")

    # Headers
    for r in all_results:
        label = f"{r['summary']['policy']} ({r['summary']['mode']})"
        print(f"{label:>{col_width}} | ", end="")
    print()
    print("-" * total_width)

    # Rows
    metrics = [
        ("Toplam hedef", "total_targets"),
        ("Position-only reach", "position_reach_count"),
        ("Position-only rate (%)", "position_reach_rate"),
        ("Validated reach", "validated_reach_count"),
        ("Validated rate (%)", "validated_reach_rate"),
        ("Timeout", "timeout_count"),
        ("Timeout rate (%)", "timeout_rate"),
        ("Ort. displacement (m)", "mean_ee_displacement"),
        ("Ort. time-to-reach", "mean_time_to_reach"),
        ("Ort. initial dist (m)", "mean_initial_distance"),
        ("Ort. final dist (m)", "mean_final_distance"),
        ("Validated/1K steps", "validated_per_1k_steps"),
        ("Dusme", "falls"),
    ]

    for label, key in metrics:
        print(f"{label:<{label_width}} | ", end="")
        for r in all_results:
            val = r["summary"][key]
            if isinstance(val, float):
                print(f"{val:>{col_width}.4f} | ", end="")
            else:
                print(f"{val:>{col_width}} | ", end="")
        print()

    print(f"{'=' * total_width}")


# ============================================================================
# MAIN
# ============================================================================

def _load_checkpoint(path, device):
    """Load checkpoint and extract model state dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    return ckpt, state


def _load_into_net(net, state, label):
    """Load state dict into network with shape matching. Returns (loaded, skipped)."""
    net_state = net.state_dict()
    loaded = 0
    skipped = []
    for key in net_state.keys():
        if key in state and state[key].shape == net_state[key].shape:
            net_state[key] = state[key]
            loaded += 1
        else:
            skipped.append(key)
    net.load_state_dict(net_state)
    net.eval()
    print(f"  {label} loaded: {loaded}/{len(net_state)} keys")
    if skipped:
        print(f"  {label} skipped: {skipped}")
    return loaded, skipped


def main():
    device = "cuda:0"

    # Validate: at least one S6 checkpoint must be provided
    if args.s6_checkpoint is None and args.s6u_checkpoint is None:
        print("[ERROR] En az bir S6 checkpoint gerekli: --s6_checkpoint veya --s6u_checkpoint")
        simulation_app.close()
        return

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("benchmarks", f"s6_vs_s7_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[BENCHMARK] Cikti klasoru: {output_dir}")

    # Create environment
    cfg = BenchEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing
    env = BenchmarkEnv(cfg)

    # ---- Load Stage 7 network (always required, also provides reference loco) ----
    print(f"\n[CHECKPOINT] Stage 7 yukleniyor: {args.s7_checkpoint}")
    s7_ckpt, s7_state = _load_checkpoint(args.s7_checkpoint, device)
    s7_net = DualACStage7().to(device)
    _load_into_net(s7_net, s7_state, "S7")
    print(f"  S7 curriculum level: {s7_ckpt.get('curriculum_level', 'N/A')}")

    # Reference loco state from S7 (the frozen Stage 6 loco)
    ref_loco_state = s7_net.loco_actor.state_dict()

    # ---- Load Stage 6 SIMPLIFIED network (optional) ----
    s6s_net = None
    if args.s6_checkpoint:
        print(f"\n[CHECKPOINT] Stage 6 Simplified yukleniyor: {args.s6_checkpoint}")
        s6s_ckpt, s6s_state = _load_checkpoint(args.s6_checkpoint, device)

        # Remap if needed (.actor. -> .net.)
        needs_remap = any(k.startswith("loco_actor.actor.") for k in s6s_state.keys())
        if needs_remap:
            print("  [INFO] S6s uses .actor. naming, applying remap...")
            s6s_state = remap_s6_keys(s6s_state)

        s6s_net = DualACStage6().to(device)
        _load_into_net(s6s_net, s6s_state, "S6s")
        print(f"  S6s curriculum level: {s6s_ckpt.get('curriculum_level', 'N/A')}")

        # Verify loco match with S7
        s6s_loco = s6s_net.loco_actor.state_dict()
        loco_match = all(
            torch.allclose(s6s_loco[k], ref_loco_state[k], atol=1e-6)
            for k in s6s_loco if k in ref_loco_state
        )
        if loco_match:
            print("  [VERIFY] S6s loco == S7 loco: ESLESIYOR")
        else:
            print("  [WARNING] S6s loco != S7 loco! S7 loco kopyalaniyor.")
            s6s_net.loco_actor.load_state_dict(ref_loco_state)

    # ---- Load Stage 6 UNIFIED network (optional) ----
    s6u_net = None
    if args.s6u_checkpoint:
        print(f"\n[CHECKPOINT] Stage 6 Unified yukleniyor: {args.s6u_checkpoint}")
        s6u_ckpt, s6u_state = _load_checkpoint(args.s6u_checkpoint, device)

        # S6 unified ALWAYS uses .actor. naming
        needs_remap = any(k.startswith("loco_actor.actor.") for k in s6u_state.keys())
        if needs_remap:
            print("  [INFO] S6u uses .actor. naming, applying remap...")
            s6u_state = remap_s6_keys(s6u_state)

        s6u_net = DualACStage6Unified().to(device)
        _load_into_net(s6u_net, s6u_state, "S6u")
        print(f"  S6u curriculum level: {s6u_ckpt.get('curriculum_level', 'N/A')}")
        print(f"  S6u total_reaches: {s6u_ckpt.get('total_reaches', 'N/A')}")
        print(f"  S6u best_reward: {s6u_ckpt.get('best_reward', 'N/A')}")

        # Verify loco match with S7
        s6u_loco = s6u_net.loco_actor.state_dict()
        loco_match = all(
            torch.allclose(s6u_loco[k], ref_loco_state[k], atol=1e-6)
            for k in s6u_loco if k in ref_loco_state
        )
        if loco_match:
            print("  [VERIFY] S6u loco == S7 loco: ESLESIYOR")
        else:
            print("  [WARNING] S6u loco != S7 loco! S7 loco kopyalaniyor.")
            s6u_net.loco_actor.load_state_dict(ref_loco_state)

    # ---- Determine modes ----
    if args.mode == "both":
        modes = ["standing", "walking"]
    else:
        modes = [args.mode]

    # ---- Build policy list ----
    # Each entry: (net, label, obs_fn, obs_pos_th, obs_orient_th)
    # Native obs thresholds for each policy:
    #   S6 Level 12: pos=0.04, orient=1.0
    #   S6 Unified Level 0: pos=0.04, orient=1.0 (same training code)
    #   S7 Level 7: pos=0.04, orient=2.0
    policies = []
    if s6u_net is not None:
        policies.append((s6u_net, "S6u_gaming", env.get_arm_obs_s6, 0.04, 1.0))
    if s6s_net is not None:
        policies.append((s6s_net, "S6s_simplified", env.get_arm_obs_s6, 0.04, 1.0))
    policies.append((s7_net, "S7_antigaming", env.get_arm_obs_s7, 0.04, 2.0))

    print(f"\n[BENCHMARK] {len(policies)} policy test edilecek: "
          f"{[p[1] for p in policies]}")
    print(f"[BENCHMARK] Modlar: {modes}")
    print(f"[BENCHMARK] Adim sayisi: {args.steps} per policy per mode")

    # ---- Run benchmark ----
    all_results = []

    for mode_name in modes:
        env.set_mode(mode_name)

        for net, label, obs_fn, obs_pos_th, obs_orient_th in policies:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            env.full_reset()
            result = run_policy_session(
                env, net, label, obs_fn,
                obs_pos_th, obs_orient_th, args.steps, mode_name
            )
            all_results.append(result)

    # ---- Save outputs ----
    save_per_target_csv(all_results, output_dir)
    save_summary_json(all_results, output_dir)

    if not args.no_plots:
        try:
            generate_plots(all_results, output_dir)
        except Exception as e:
            print(f"[WARNING] Grafik olusturma hatasi: {e}")
            import traceback
            traceback.print_exc()

    # ---- Final comparison table ----
    print_comparison_table(all_results)

    print(f"\n[BENCHMARK] Tamamlandi. Ciktilar: {output_dir}")
    print(f"  per_target.csv: Her hedef icin detayli metrikler")
    print(f"  summary.json:   Toplam istatistikler + config")
    if not args.no_plots:
        print(f"  plots/:         Karsilastirma grafikleri (PNG + PDF)")

    simulation_app.close()


if __name__ == "__main__":
    main()
