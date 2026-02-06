#!/usr/bin/env python3
"""
ULC G1 Stage 7 Play Script
============================
Evaluation script for Stage 7 anti-gaming arm reaching.

KEY DIFFERENCES FROM STAGE 6 PLAY:
1. Arm obs = 55 dim (52 + 3 anti-gaming: steps_since_spawn, ee_displacement, initial_distance)
2. Validated reach counting (position + displacement + time)
3. Absolute-only target sampling with min distance enforcement
4. Reports validated vs total reaches, ee_displacement, ee_speed

Usage:
    # Default: standing mode with anti-gaming validation
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 4 --mode standing

    # Walking test
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 4 --mode walking

    # Single env for visual debugging
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 1 --mode standing --steps 2000
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description="ULC G1 Stage 7 Play - Anti-Gaming")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--mode", type=str, default="standing",
                    choices=["standing", "walking", "fast", "demo"])
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Use stochastic actions (default: deterministic)")
parser.add_argument("--reach_threshold", type=float, default=0.08,
                    help="Position reach threshold in meters")
parser.add_argument("--min_displacement", type=float, default=0.05,
                    help="Minimum EE displacement to count as valid reach (meters)")
parser.add_argument("--max_reach_steps", type=int, default=200,
                    help="Maximum steps to reach target before timeout")
parser.add_argument("--orient_check", action="store_true", default=False,
                    help="Enable orientation check for reaches")
parser.add_argument("--orient_threshold", type=float, default=2.0,
                    help="Orientation threshold in radians")
parser.add_argument("--env_spacing", type=float, default=5.0,
                    help="Spacing between environments")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

use_deterministic = not args.stochastic

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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

MODE_CONFIGS = {
    "standing": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.18, 0.35),
        "min_target_distance": 0.10,
        "description": "Standing still, arm reaching only"
    },
    "walking": {
        "vx_range": (0.1, 0.3),
        "vy_range": (-0.05, 0.05),
        "vyaw_range": (-0.1, 0.1),
        "workspace_radius": (0.18, 0.40),
        "min_target_distance": 0.12,
        "description": "Normal walking with arm reaching"
    },
    "fast": {
        "vx_range": (0.3, 0.5),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "min_target_distance": 0.15,
        "description": "Fast walking with arm reaching"
    },
    "demo": {
        "vx_range": (0.0, 0.4),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "min_target_distance": 0.12,
        "description": "Mixed demo mode"
    },
}


# ============================================================================
# NETWORK - MATCHES TRAINING (55 arm obs)
# ============================================================================

class LocoActor(nn.Module):
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
    """55 obs -> 5 actions (Stage 7: anti-gaming obs)"""
    def __init__(self, num_obs=55, num_act=5, hidden=[256, 256, 128]):
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
    def __init__(self, num_obs=55, hidden=[256, 256, 128]):
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


class DualActorCritic(nn.Module):
    def __init__(self, loco_obs=57, arm_obs=55, loco_act=12, arm_act=5):
        super().__init__()
        self.loco_actor = LocoActor(loco_obs, loco_act)
        self.loco_critic = LocoCritic(loco_obs)
        self.arm_actor = ArmActor(arm_obs, arm_act)
        self.arm_critic = ArmCritic(arm_obs)

    def get_actions(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        return (torch.distributions.Normal(loco_mean, loco_std).sample(),
                torch.distributions.Normal(arm_mean, arm_std).sample())


# ============================================================================
# HELPERS
# ============================================================================

def quat_to_euler_xyz(quat):
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


def get_palm_forward(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=-1)


def compute_orientation_error(palm_quat, target_dir=None):
    forward = get_palm_forward(palm_quat)
    if target_dir is None:
        target_dir = torch.zeros_like(forward)
        target_dir[:, 2] = -1.0
    dot = torch.clamp((forward * target_dir).sum(dim=-1), -1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# ENVIRONMENT
# ============================================================================

@configclass
class PlaySceneCfg(InteractiveSceneCfg):
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
                joint_names_expr=[".*zero.*", ".*one.*", ".*two.*", ".*three.*", ".*four.*", ".*five.*", ".*six.*"],
                stiffness=20.0, damping=2.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0, damping=10.0,
            ),
        },
    )


@configclass
class PlayEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_space = 17
    observation_space = 57
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=4, env_spacing=5.0)


class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

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

        joint_limits = self.robot.root_physx_view.get_dof_limits()
        self.finger_lower = torch.tensor(
            [joint_limits[0, i, 0].item() for i in self.finger_idx], device=self.device
        )
        self.finger_upper = torch.tensor(
            [joint_limits[0, i, 1].item() for i in self.finger_idx], device=self.device
        )

        body_names = self.robot.body_names
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right_palm" in name.lower():
                self.palm_idx = i
                break
        if self.palm_idx is None:
            self.palm_idx = len(body_names) - 1

        self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body[:, 2] = -1.0

        self.phase = torch.zeros(self.num_envs, device=self.device)

        # Action history
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Anti-gaming state buffers
        self.ee_pos_at_spawn = torch.zeros(self.num_envs, 3, device=self.device)
        self.steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.initial_dist = torch.zeros(self.num_envs, device=self.device)

        # Config
        self.reach_pos_threshold = args.reach_threshold
        self.min_displacement = args.min_displacement
        self.max_reach_steps = args.max_reach_steps
        self.use_orient_check = args.orient_check
        self.orient_threshold = args.orient_threshold

        # Stats
        self.total_reaches = 0
        self.validated_reaches = 0
        self.pos_only_reaches = 0
        self.timed_out_targets = 0
        self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reach_distances = []
        self.reach_displacements = []

        self.mode_cfg = MODE_CONFIGS[args.mode]
        self._markers_initialized = False

        print(f"\n[PlayEnv Stage 7] Configuration:")
        print(f"  Envs: {self.num_envs}, Spacing: {args.env_spacing}m")
        print(f"  Mode: {args.mode} ({self.mode_cfg['description']})")
        print(f"  Pos threshold: {self.reach_pos_threshold:.3f}m")
        print(f"  Min displacement: {self.min_displacement:.3f}m")
        print(f"  Max reach steps: {self.max_reach_steps}")
        print(f"  Orient check: {self.use_orient_check}")
        print(f"  Deterministic: {use_deterministic}")

    @property
    def robot(self):
        return self.scene["robot"]

    def _init_markers(self):
        if self._markers_initialized:
            return
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
        )
        self.ee_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/EEMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
        )
        self._markers_initialized = True

    def _compute_palm_ee(self):
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_forward = get_palm_forward(palm_quat)
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward
        return ee_pos, palm_quat

    def _sample_commands(self, env_ids):
        """Absolute-only sampling with min distance enforcement"""
        n = len(env_ids)
        cfg = self.mode_cfg

        self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*cfg["vx_range"])
        self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*cfg["vy_range"])
        self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*cfg["vyaw_range"])
        self.height_cmd[env_ids] = HEIGHT_DEFAULT

        # Get current EE in body frame
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        ee_world, _ = self._compute_palm_ee()
        ee_world = ee_world[env_ids]
        current_ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

        # Absolute sampling
        ws = cfg["workspace_radius"]
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

        # Enforce min distance from EE
        min_dist = cfg.get("min_target_distance", 0.10)
        dist_to_ee = torch.norm(target_body - current_ee_body, dim=-1)
        too_close = dist_to_ee < min_dist
        if too_close.any():
            direction = target_body - current_ee_body
            direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            direction = direction / direction_norm
            pushed = current_ee_body + min_dist * direction
            pushed[:, 0] = pushed[:, 0].clamp(0.05, 0.55)
            pushed[:, 1] = pushed[:, 1].clamp(-0.55, 0.10)
            pushed[:, 2] = pushed[:, 2].clamp(-0.25, 0.55)
            target_body = torch.where(too_close.unsqueeze(-1).expand_as(target_body), pushed, target_body)

        self.target_pos_body[env_ids] = target_body

        # Palm down orientation
        self.target_orient_body[env_ids, 0] = 0.0
        self.target_orient_body[env_ids, 1] = 0.0
        self.target_orient_body[env_ids, 2] = -1.0

        # Record spawn state
        self.ee_pos_at_spawn[env_ids] = current_ee_body
        self.steps_since_spawn[env_ids] = 0
        self.initial_dist[env_ids] = torch.norm(target_body - current_ee_body, dim=-1).clamp(min=0.01)
        self.already_reached[env_ids] = False

    def get_loco_obs(self):
        """57 dims - Same as training"""
        robot = self.robot
        quat = robot.data.root_quat_w
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        joint_pos = robot.data.joint_pos[:, self.leg_idx]
        joint_vel = robot.data.joint_vel[:, self.leg_idx]
        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.phase),
            torch.cos(2 * np.pi * self.phase)
        ], dim=-1)
        torso_euler = quat_to_euler_xyz(quat)
        obs = torch.cat([
            lin_vel_b, ang_vel_b, proj_gravity, joint_pos, joint_vel,
            self.height_cmd.unsqueeze(-1), self.vel_cmd, gait_phase,
            self.prev_leg_actions, self.torso_cmd, torso_euler,
        ], dim=-1)
        return obs.clamp(-10, 10).nan_to_num()

    def get_arm_obs(self):
        """55 dims = 52 (base) + 3 (anti-gaming) - MUST match training"""
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

        finger_normalized = (finger_pos - self.finger_lower) / (self.finger_upper - self.finger_lower + 1e-6)
        gripper_closed_ratio = finger_normalized.mean(dim=-1, keepdim=True)
        finger_vel = robot.data.joint_vel[:, self.finger_idx]
        grip_force = (finger_vel.abs().mean(dim=-1, keepdim=True) * gripper_closed_ratio).clamp(0, 1)

        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist_to_target = torch.norm(ee_world - target_world, dim=-1, keepdim=True)
        contact_detected = (dist_to_target < 0.08).float()

        target_body = self.target_pos_body
        pos_error = target_body - ee_body
        pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5
        orient_err = compute_orientation_error(palm_quat, self.target_orient_body).unsqueeze(-1) / np.pi

        orient_threshold = self.orient_threshold if self.use_orient_check else 1.0
        target_reached = ((dist_to_target < self.reach_pos_threshold) &
                         (orient_err * np.pi < orient_threshold)).float()

        current_height = root_pos[:, 2:3]
        height_cmd_obs = self.height_cmd.unsqueeze(-1)
        height_err = (height_cmd_obs - current_height) / 0.4

        estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
        object_in_hand_obs = torch.zeros(self.num_envs, 1, device=self.device)
        target_orient_obs = self.target_orient_body

        lin_vel_b = quat_apply_inverse(root_quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w)
        lin_vel_xy = lin_vel_b[:, :2]
        ang_vel_z = ang_vel_b[:, 2:3]

        # 3 anti-gaming observations
        max_steps = float(self.max_reach_steps)
        steps_norm = (self.steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)
        ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1, keepdim=True)
        initial_dist_obs = self.initial_dist.unsqueeze(-1) / 0.5

        obs = torch.cat([
            arm_pos, arm_vel, finger_pos,
            ee_body, ee_vel_body, palm_quat, grip_force, gripper_closed_ratio, contact_detected,
            target_body, pos_error, pos_dist, orient_err, target_reached,
            height_cmd_obs, current_height, height_err,
            estimated_load, object_in_hand_obs, target_orient_obs,
            lin_vel_xy, ang_vel_z,
            # 3 new
            steps_norm, ee_displacement, initial_dist_obs,
        ], dim=-1)
        return obs.clamp(-10, 10).nan_to_num()

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5
        target_pos[:, self.finger_idx] = self.finger_lower

        self.robot.set_joint_position_target(target_pos)
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

        # Increment step counter
        self.steps_since_spawn += 1

        # Reach check with anti-gaming validation
        ee_pos, palm_quat = self._compute_palm_ee()
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist = torch.norm(ee_pos - target_world, dim=-1)

        # Condition 1: Position close
        pos_close = dist < self.reach_pos_threshold

        # Condition 1b: Optional orientation
        if self.use_orient_check:
            orient_err = compute_orientation_error(palm_quat, self.target_orient_body)
            pos_close = pos_close & (orient_err < self.orient_threshold)

        # Track position-only reaches (for comparison)
        pos_only_new = pos_close & ~self.already_reached
        self.pos_only_reaches += pos_only_new.sum().item()

        # Condition 2: Displacement
        ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1)
        moved_enough = ee_displacement >= self.min_displacement

        # Condition 3: Within time limit
        within_time = self.steps_since_spawn <= self.max_reach_steps

        # Validated reach = all 3 conditions
        validated_reach = pos_close & moved_enough & within_time
        new_reaches = validated_reach & ~self.already_reached

        if new_reaches.any():
            reached_ids = torch.where(new_reaches)[0]
            self.total_reaches += len(reached_ids)
            self.validated_reaches += len(reached_ids)
            # Record stats
            for idx in reached_ids:
                self.reach_distances.append(dist[idx].item())
                self.reach_displacements.append(ee_displacement[idx].item())
            self.already_reached[reached_ids] = True
            self._sample_commands(reached_ids)

        # Handle timeouts
        timed_out = (self.steps_since_spawn > self.max_reach_steps) & ~self.already_reached
        if timed_out.any():
            timed_out_ids = torch.where(timed_out)[0]
            self.timed_out_targets += len(timed_out_ids)
            self._sample_commands(timed_out_ids)

        # Markers
        self._init_markers()
        default_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(self.num_envs, -1)
        self.target_markers.visualize(translations=target_world, orientations=default_quat)
        self.ee_markers.visualize(translations=ee_pos, orientations=default_quat)

        self.prev_ee_pos = ee_pos.clone()
        self._prev_leg_actions = self.prev_leg_actions.clone()
        self._prev_arm_actions = self.prev_arm_actions.clone()
        self.prev_leg_actions = leg_actions.clone()
        self.prev_arm_actions = arm_actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {"policy": self.get_loco_obs()}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        height = self.robot.data.root_pos_w[:, 2]
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
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
        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)
        self._sample_commands(env_ids)
        self.phase[env_ids] = torch.rand(n, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0
        self._prev_leg_actions[env_ids] = 0
        self._prev_arm_actions[env_ids] = 0
        self.prev_ee_pos[env_ids] = 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda:0"

    print(f"\n{'=' * 70}")
    print("ULC G1 STAGE 7 - ANTI-GAMING PLAY")
    print(f"{'=' * 70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {args.mode} | {MODE_CONFIGS[args.mode]['description']}")
    print(f"  Reach threshold: {args.reach_threshold}m")
    print(f"  Min displacement: {args.min_displacement}m")
    print(f"  Max reach steps: {args.max_reach_steps}")
    print(f"  Deterministic: {use_deterministic}")
    print(f"{'=' * 70}\n")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print(f"[Checkpoint]")
    for key in ["best_reward", "iteration", "curriculum_level", "total_reaches",
                "validated_reaches", "timed_out_targets"]:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")

    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing
    env = PlayEnv(cfg)

    net = DualActorCritic(loco_obs=57, arm_obs=55, loco_act=12, arm_act=5).to(device)

    state_dict = checkpoint.get("model", checkpoint)
    model_keys = set(net.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing or unexpected:
        if missing:
            print(f"\n  MISSING: {sorted(missing)[:5]}")
        if unexpected:
            print(f"\n  UNEXPECTED: {sorted(unexpected)[:5]}")
    else:
        print(f"\n  All {len(model_keys)} keys match!")

    net.load_state_dict(state_dict, strict=True)
    net.eval()

    arm_std = net.arm_actor.log_std.clamp(-2, 1).exp().mean().item()
    print(f"[Std] Arm: {arm_std:.4f}")

    obs, _ = env.reset()
    prev_reaches = 0

    print(f"\n[Play] {args.steps} steps | '{args.mode}' | {'DETERM' if use_deterministic else 'STOCH'}\n")

    with torch.no_grad():
        for step in range(args.steps):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            leg_act, arm_act = net.get_actions(loco_obs, arm_obs, deterministic=use_deterministic)
            actions = torch.cat([leg_act, arm_act], dim=-1)

            obs, reward, terminated, truncated, info = env.step(actions)

            if env.validated_reaches > prev_reaches:
                ee_pos, palm_quat = env._compute_palm_ee()
                root_pos = env.robot.data.root_pos_w
                root_quat = env.robot.data.root_quat_w
                ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
                target_world = root_pos + quat_apply(root_quat, env.target_pos_body)
                dist = torch.norm(ee_pos - target_world, dim=-1).min().item()
                disp = torch.norm(ee_body - env.ee_pos_at_spawn, dim=-1).max().item()
                new = env.validated_reaches - prev_reaches
                print(f"[Step {step:4d}] 🎯 +{new} VALIDATED REACH (total={env.validated_reaches}) "
                      f"dist={dist:.3f}m disp={disp:.3f}m")
                prev_reaches = env.validated_reaches

            if step > 0 and step % 200 == 0:
                h = env.robot.data.root_pos_w[:, 2].mean().item()
                root_quat = env.robot.data.root_quat_w
                vx_b = quat_apply_inverse(root_quat, env.robot.data.root_lin_vel_w)[:, 0].mean().item()
                cmd_vx = env.vel_cmd[:, 0].mean().item()

                ee_pos, pq = env._compute_palm_ee()
                root_pos = env.robot.data.root_pos_w
                tw = root_pos + quat_apply(root_quat, env.target_pos_body)
                ee_d = torch.norm(ee_pos - tw, dim=-1).mean().item()
                ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
                ee_disp = torch.norm(ee_body - env.ee_pos_at_spawn, dim=-1).mean().item()
                ee_spd = torch.norm((ee_pos - env.prev_ee_pos) / 0.02, dim=-1).mean().item()

                total_attempts = env.validated_reaches + env.timed_out_targets
                v_rate = env.validated_reaches / max(total_attempts, 1) * 100

                print(
                    f"[Step {step:4d}] "
                    f"H={h:.3f} Vx={vx_b:+.2f}(cmd={cmd_vx:.2f}) "
                    f"EE={ee_d:.3f}m Disp={ee_disp:.3f}m Spd={ee_spd:.3f} "
                    f"VR={env.validated_reaches} TO={env.timed_out_targets} "
                    f"Rate={v_rate:.0f}%"
                )

    total_attempts = env.validated_reaches + env.timed_out_targets
    print(f"\n{'=' * 70}")
    print("STAGE 7 PLAY COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Validated Reaches: {env.validated_reaches}")
    print(f"  Position-Only Reaches: {env.pos_only_reaches}")
    print(f"  Timed Out: {env.timed_out_targets}")
    print(f"  Total Attempts: {total_attempts}")
    if total_attempts > 0:
        print(f"  Validated Rate: {env.validated_reaches / total_attempts:.1%}")
    print(f"  Rate: {env.validated_reaches / args.steps * 1000:.1f} validated per 1K steps")
    if env.reach_distances:
        print(f"  Avg Reach Distance: {np.mean(env.reach_distances):.3f}m")
        print(f"  Avg Reach Displacement: {np.mean(env.reach_displacements):.3f}m")
    print(f"{'=' * 70}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
