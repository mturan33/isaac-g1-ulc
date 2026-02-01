#!/usr/bin/env python3
"""
ULC G1 Stage 6 Unified Play Script
===================================
Test the unified loco-manipulation policy.

Features:
- Locomotion + Arm reaching + Gripper control
- Target visualization (green sphere)
- EE visualization (red sphere)
- Multiple test modes

Usage:
    ./isaaclab.bat -p play_ulc_stage6_unified.py \
        --checkpoint logs/ulc/ulc_g1_stage6_complete_.../model_best.pt \
        --num_envs 4 \
        --mode walking
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Parse arguments before Isaac imports
parser = argparse.ArgumentParser(description="ULC G1 Stage 6 Unified Play")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--steps", type=int, default=3000, help="Steps to run")
parser.add_argument("--mode", type=str, default="walking",
                    choices=["standing", "walking", "fast", "demo"],
                    help="Test mode")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

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
REACH_THRESHOLD = 0.05

G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

# Mode configurations
MODE_CONFIGS = {
    "standing": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.18, 0.35),
        "description": "Standing still, arm reaching only"
    },
    "walking": {
        "vx_range": (0.2, 0.4),
        "vy_range": (-0.05, 0.05),
        "vyaw_range": (-0.1, 0.1),
        "workspace_radius": (0.18, 0.40),
        "description": "Normal walking with arm reaching"
    },
    "fast": {
        "vx_range": (0.4, 0.6),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.2, 0.2),
        "workspace_radius": (0.18, 0.40),
        "description": "Fast walking with arm reaching"
    },
    "demo": {
        "vx_range": (0.0, 0.5),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "description": "Mixed demo mode"
    },
}


# ============================================================================
# NETWORK ARCHITECTURE (must match training)
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion actor - matches Stage 3 architecture"""

    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ArmActor(nn.Module):
    """Arm actor for reaching + gripper - MUST match training architecture!"""

    def __init__(self, num_obs=52, num_act=12, hidden=[256, 256, 128]):  # FIXED: was [256, 128, 64]
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnifiedCritic(nn.Module):
    """Unified critic"""

    def __init__(self, num_obs=109, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnifiedActorCritic(nn.Module):
    """Complete unified network"""

    def __init__(self, loco_obs=57, arm_obs=52, loco_act=12, arm_act=12):
        super().__init__()
        self.loco_actor = LocoActor(loco_obs, loco_act)
        self.arm_actor = ArmActor(arm_obs, arm_act)
        self.critic = UnifiedCritic(loco_obs + arm_obs)
        self.loco_log_std = nn.Parameter(torch.zeros(loco_act))
        self.arm_log_std = nn.Parameter(torch.zeros(arm_act))

    def act(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_log_std.clamp(-2, 1).exp()
        arm_std = self.arm_log_std.clamp(-2, 1).exp()
        loco_act = torch.distributions.Normal(loco_mean, loco_std).sample()
        arm_act = torch.distributions.Normal(arm_mean, arm_std).sample()
        return loco_act, arm_act


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


# ============================================================================
# ENVIRONMENT CONFIG
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
            pos=(0.0, 0.0, 0.85),  # Higher initial position
            joint_pos={
                # Legs - slightly bent for stability
                "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
                # Arms - natural position
                "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
                "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.6, "right_elbow_pitch_joint": 0.6,
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
                stiffness=80.0, damping=8.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0, damping=10.0,
            ),
        },
    )
    # NO RigidObjectCfg markers - they break physics!
    # Using VisualizationMarkers instead (created in environment __init__)


@configclass
class PlayEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_space = 24  # 12 leg + 12 arm (5 arm joints + 7 finger)
    observation_space = 109  # 57 loco + 52 arm
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=4, env_spacing=2.5)


# ============================================================================
# PLAY ENVIRONMENT
# ============================================================================

class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        joint_names = self.robot.joint_names

        leg_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]

        arm_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ]

        self.leg_idx = torch.tensor([joint_names.index(n) for n in leg_names], device=self.device)
        self.arm_idx = torch.tensor([joint_names.index(n) for n in arm_names], device=self.device)

        # Finger joint indices (7 finger joints for right hand)
        finger_names = [
            "right_zero_joint", "right_one_joint", "right_two_joint",
            "right_three_joint", "right_four_joint", "right_five_joint", "right_six_joint"
        ]
        self.finger_idx = torch.tensor(
            [joint_names.index(n) for n in finger_names if n in joint_names],
            device=self.device
        )

        # Finger limits for gripper normalization
        joint_limits = self.robot.data.joint_limits
        if len(self.finger_idx) > 0:
            self.finger_lower = torch.tensor(
                [joint_limits[0, i, 0].item() for i in self.finger_idx],
                device=self.device
            )
            self.finger_upper = torch.tensor(
                [joint_limits[0, i, 1].item() for i in self.finger_idx],
                device=self.device
            )
        else:
            # Fallback if no finger joints found
            self.finger_lower = torch.zeros(7, device=self.device)
            self.finger_upper = torch.ones(7, device=self.device)

        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )
        self.default_arm = torch.tensor([0.3, -0.2, 0.0, 0.6, 0.0], device=self.device)

        # Find palm body for EE
        body_names = self.robot.body_names
        self.palm_idx = body_names.index("right_palm_link") if "right_palm_link" in body_names else None
        if self.palm_idx is None:
            print("[WARNING] right_palm_link not found, using right_elbow_roll_link")
            self.palm_idx = body_names.index("right_elbow_roll_link")

        # Shoulder offset for target sampling
        self.shoulder_offset = torch.tensor([0.0, -0.174, 0.259], device=self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # State - SEPARATE leg and arm actions like training!
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.total_reaches = 0
        self.reach_threshold = REACH_THRESHOLD

        # Mode config
        self.mode_cfg = MODE_CONFIGS[args.mode]

        print(f"[PlayEnv] Leg joints: {len(self.leg_idx)}")
        print(f"[PlayEnv] Arm joints: {len(self.arm_idx)}")
        print(f"[PlayEnv] Finger joints: {len(self.finger_idx)}")
        print(f"[PlayEnv] Palm body idx: {self.palm_idx}")

        # Create visualization markers (no physics impact!)
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),  # Green
                        ),
                    ),
                },
            )
        )

        self.ee_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/EEMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),  # Red
                        ),
                    ),
                },
            )
        )
        print("[PlayEnv] VisualizationMarkers created")

    @property
    def robot(self):
        return self.scene["robot"]

    def _compute_palm_ee(self):
        """Compute palm end-effector position and forward direction.

        MUST MATCH TRAINING:
        - Palm forward = LOCAL +X axis (not +Z!)
        - EE = palm_pos + 0.08 * palm_forward
        """
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]

        # Palm forward is LOCAL +X axis (matches training get_palm_forward function)
        # get_palm_forward extracts first column of rotation matrix = local +X in world
        w, x, y, z = palm_quat[:, 0], palm_quat[:, 1], palm_quat[:, 2], palm_quat[:, 3]
        fwd_x = 1 - 2 * (y * y + z * z)
        fwd_y = 2 * (x * y + w * z)
        fwd_z = 2 * (x * z - w * y)
        palm_forward = torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)

        # EE position = palm_pos + offset along forward direction (matches training)
        PALM_FORWARD_OFFSET = 0.08
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward

        return ee_pos, palm_forward

    def _sample_targets(self, env_ids):
        """Sample new arm targets in body frame."""
        n = len(env_ids)
        ws = self.mode_cfg["workspace_radius"]

        azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
        elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
        radius = torch.empty(n, device=self.device).uniform_(*ws)

        x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
        y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
        z = radius * torch.sin(elevation) + self.shoulder_offset[2]

        self.target_pos_body[env_ids, 0] = x.clamp(0.05, 0.55)
        self.target_pos_body[env_ids, 1] = y.clamp(-0.55, 0.10)
        self.target_pos_body[env_ids, 2] = z.clamp(-0.25, 0.55)

    def _sample_commands(self, env_ids):
        """Sample velocity commands."""
        n = len(env_ids)
        cfg = self.mode_cfg

        vx = cfg["vx_range"]
        vy = cfg["vy_range"]
        vyaw = cfg["vyaw_range"]

        self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*vx)
        self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*vy)
        self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*vyaw)

        self._sample_targets(env_ids)

    def _update_markers(self):
        """Update target and EE marker positions using VisualizationMarkers."""
        # Target marker: body frame -> world frame
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)

        # VisualizationMarkers use wxyz quaternion format
        target_quat = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self.target_markers.visualize(translations=target_world, orientations=target_quat)

        # EE marker
        ee_pos, _ = self._compute_palm_ee()
        ee_quat = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self.ee_markers.visualize(translations=ee_pos, orientations=ee_quat)

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()

        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]  # Only first 5 for arm joints

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5

        self.robot.set_joint_position_target(target_pos)

        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

        # Track separate action histories like training
        self.prev_leg_actions = leg_actions.clone()
        self.prev_arm_actions = arm_actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
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

        # Loco obs (57) - MUST MATCH TRAINING EXACTLY
        loco_obs = torch.cat([
            lin_vel_b,  # 3
            ang_vel_b,  # 3
            proj_gravity,  # 3
            joint_pos,  # 12
            joint_vel,  # 12
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,  # 3
            gait_phase,  # 2
            self.prev_leg_actions,  # 12 (FIXED: was prev_actions[:, :12])
            self.torso_cmd,  # 3
            torso_euler,  # 3
        ], dim=-1)

        # Arm obs (52) - MUST MATCH TRAINING EXACTLY
        arm_pos = robot.data.joint_pos[:, self.arm_idx]
        arm_vel = robot.data.joint_vel[:, self.arm_idx] * 0.1  # FIXED: scale by 0.1

        ee_pos_world, palm_forward = self._compute_palm_ee()
        root_pos = robot.data.root_pos_w
        ee_pos_body = quat_apply_inverse(quat, ee_pos_world - root_pos)

        # EE velocity from previous position (matches training)
        ee_vel_world = (ee_pos_world - self.prev_ee_pos) / 0.02  # dt = 0.02
        ee_vel_body = quat_apply_inverse(quat, ee_vel_world)

        palm_quat = robot.data.body_quat_w[:, self.palm_idx]

        pos_error = self.target_pos_body - ee_pos_body
        pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5  # FIXED: normalize by 0.5

        # Orientation error: angle between palm forward and world DOWN (-Z)
        # MUST MATCH TRAINING: acos(dot) / pi, NOT 1-dot!
        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        dot = (palm_forward * down_vec).sum(dim=-1)
        angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
        orient_error = (angle / np.pi).unsqueeze(-1)  # Normalized [0, 1]

        # Check reaches (use unnormalized distance for threshold check)
        actual_dist = pos_error.norm(dim=-1)
        reached = actual_dist < self.reach_threshold
        new_reaches = reached.sum().item()
        if new_reaches > 0:
            self.total_reaches += new_reaches
            self._sample_targets(reached.nonzero(as_tuple=True)[0])

        # Actual finger positions (FIXED: was zeros)
        if len(self.finger_idx) > 0:
            finger_pos = robot.data.joint_pos[:, self.finger_idx]
        else:
            finger_pos = torch.zeros(self.num_envs, 7, device=self.device)

        # Gripper state calculations
        if len(self.finger_idx) > 0:
            finger_normalized = (finger_pos - self.finger_lower) / (self.finger_upper - self.finger_lower + 1e-6)
            gripper_closed = finger_normalized.mean(dim=-1, keepdim=True)
            finger_vel = robot.data.joint_vel[:, self.finger_idx]
            grip_force = (finger_vel.abs().mean(dim=-1, keepdim=True) * gripper_closed).clamp(0, 1)
        else:
            grip_force = torch.zeros(self.num_envs, 1, device=self.device)
            gripper_closed = torch.zeros(self.num_envs, 1, device=self.device)

        # Contact detection
        contact = (actual_dist < 0.05).float().unsqueeze(-1)  # GRASP_THRESHOLD
        target_reached = reached.float().unsqueeze(-1)

        height_cmd = self.height_cmd.unsqueeze(-1)
        current_height = robot.data.root_pos_w[:, 2:3]
        height_err = (height_cmd - current_height) / 0.4  # FIXED: normalize by 0.4

        estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
        object_in_hand = torch.zeros(self.num_envs, 1, device=self.device)
        object_rel_ee = torch.zeros(self.num_envs, 3, device=self.device)

        base_lin_vel = lin_vel_b[:, :2]
        base_ang_vel = ang_vel_b[:, 2:3]

        arm_obs = torch.cat([
            arm_pos,  # 5
            arm_vel,  # 5
            finger_pos,  # 7
            ee_pos_body,  # 3
            ee_vel_body,  # 3
            palm_quat,  # 4
            grip_force,  # 1
            gripper_closed,  # 1
            contact,  # 1
            self.target_pos_body,  # 3
            pos_error,  # 3
            pos_dist,  # 1
            orient_error,  # 1
            target_reached,  # 1
            height_cmd,  # 1
            current_height,  # 1
            height_err,  # 1
            estimated_load,  # 3
            object_in_hand,  # 1
            object_rel_ee,  # 3
            base_lin_vel,  # 2
            base_ang_vel,  # 1
        ], dim=-1)

        self._update_markers()

        # Update prev_ee_pos for next frame velocity calculation
        self.prev_ee_pos = ee_pos_world.clone()

        return {
            "policy": torch.cat([loco_obs, arm_obs], dim=-1).clamp(-10, 10).nan_to_num(),
            "loco": loco_obs.clamp(-10, 10).nan_to_num(),
            "arm": arm_obs.clamp(-10, 10).nan_to_num(),
        }

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple:
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

        # Higher spawn position to prevent sinking
        default_pos = torch.tensor([[0.0, 0.0, 0.85]], device=self.device).expand(len(env_ids), -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(env_ids), -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)

        # Reset action histories (FIXED: separate leg/arm tracking)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0
        self.prev_ee_pos[env_ids] = 0

        self._sample_commands(env_ids)


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda:0"

    # Load checkpoint
    print(f"\n{'=' * 60}")
    print("ULC G1 STAGE 6 UNIFIED - PLAY")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {args.mode}")
    print(f"  Description: {MODE_CONFIGS[args.mode]['description']}")
    print(f"{'=' * 60}\n")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Show checkpoint info
    print(f"\n[Checkpoint Info]")
    if "best_reward" in checkpoint:
        print(f"  Best reward: {checkpoint['best_reward']:.2f}")
    if "iteration" in checkpoint:
        print(f"  Iteration: {checkpoint['iteration']}")
    if "curriculum_level" in checkpoint:
        print(f"  Curriculum level: {checkpoint['curriculum_level']}")

    # Warn if using early checkpoint
    if checkpoint.get("iteration", 0) < 1000:
        print(f"\n⚠️  WARNING: This checkpoint is from iteration {checkpoint.get('iteration', 'unknown')}!")
        print(f"   Consider using a later checkpoint like model_19998.pt or model_final.pt")
        print(f"   for better performance.\n")

    # Create environment
    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = PlayEnv(cfg)

    # Create network and load weights
    net = UnifiedActorCritic(57, 52, 12, 12).to(device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        net.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded model_state_dict with {len(state_dict)} parameters")
    elif "actor_critic" in checkpoint:
        # Old format compatibility
        state_dict = checkpoint["actor_critic"]
        net.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded actor_critic with {len(state_dict)} parameters")
    else:
        # Direct state dict
        net.load_state_dict(checkpoint, strict=False)
        print("[INFO] Loaded checkpoint directly")

    net.eval()

    # Run
    obs, _ = env.reset()
    total_reward = 0.0
    prev_reaches = 0

    print(f"\n[Play] Starting {args.steps} steps in '{args.mode}' mode...")
    print(f"       Press Ctrl+C to stop\n")

    with torch.no_grad():
        for step in range(args.steps):
            loco_obs = obs["loco"]
            arm_obs = obs["arm"]

            leg_actions, arm_actions = net.act(loco_obs, arm_obs, deterministic=True)
            actions = torch.cat([leg_actions, arm_actions], dim=-1)

            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward.mean().item()

            # Report reaches
            if env.total_reaches > prev_reaches:
                print(f"[Step {step:4d}] 🎯 REACH! Total: {env.total_reaches}")
                prev_reaches = env.total_reaches

            # Resample commands periodically
            if step > 0 and step % 500 == 0:
                env._sample_commands(torch.arange(env.num_envs, device=device))

            # Progress report
            if step > 0 and step % 200 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()
                ee_pos, _ = env._compute_palm_ee()
                root_pos = env.robot.data.root_pos_w
                root_quat = env.robot.data.root_quat_w
                ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
                ee_dist = (ee_body - env.target_pos_body).norm(dim=-1).mean().item()

                print(
                    f"[Step {step:4d}] "
                    f"H={height:.3f}m | "
                    f"Vx={vx:.2f}m/s | "
                    f"EE_dist={ee_dist:.3f}m | "
                    f"Reaches={env.total_reaches}"
                )

    print(f"\n{'=' * 60}")
    print("PLAY COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total reaches: {env.total_reaches}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"{'=' * 60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()