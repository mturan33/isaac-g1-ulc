#!/usr/bin/env python3
"""
ULC G1 Stage 6 SIMPLIFIED Play Script
=====================================
Matches the DUAL ACTOR-CRITIC architecture from training.

Architecture (MUST MATCH TRAINING):
- LocoActor (57→12) with self.net
- LocoCritic (57→1) with self.net
- ArmActor (52→5) with self.net  ← NO FINGERS!
- ArmCritic (52→1) with self.net

Usage:
    ./isaaclab.bat -p play_ulc_stage6_simplified.py \
        --checkpoint logs/ulc/ulc_g1_stage6_simplified_.../model_best.pt \
        --num_envs 4 \
        --mode walking
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import numpy as np

# Parse arguments before Isaac imports
parser = argparse.ArgumentParser(description="ULC G1 Stage 6 Simplified Play")
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
        "vx_range": (0.1, 0.3),
        "vy_range": (-0.05, 0.05),
        "vyaw_range": (-0.1, 0.1),
        "workspace_radius": (0.18, 0.40),
        "description": "Normal walking with arm reaching"
    },
    "fast": {
        "vx_range": (0.3, 0.5),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "description": "Fast walking with arm reaching"
    },
    "demo": {
        "vx_range": (0.0, 0.4),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "description": "Mixed demo mode"
    },
}


# ============================================================================
# NETWORK ARCHITECTURE - MUST MATCH TRAINING EXACTLY!
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion policy: 57 obs → 12 leg actions"""

    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)  # MUST BE self.net!
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, x):
        return self.net(x)


class LocoCritic(nn.Module):
    """Locomotion value function: 57 obs → 1 value"""

    def __init__(self, num_obs=57, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)  # MUST BE self.net!

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ArmActor(nn.Module):
    """Arm policy: 52 obs → 5 actions (NO FINGERS!)"""

    def __init__(self, num_obs=52, num_act=5, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]  # NO LayerNorm!
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)  # MUST BE self.net!
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, x):
        return self.net(x)


class ArmCritic(nn.Module):
    """Arm value function: 52 obs → 1 value"""

    def __init__(self, num_obs=52, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]  # NO LayerNorm!
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)  # MUST BE self.net!

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualActorCritic(nn.Module):
    """
    Dual Actor-Critic - MATCHES TRAINING EXACTLY

    Architecture:
    - LocoActor (57→12) + LocoCritic (57→1)
    - ArmActor (52→5)   + ArmCritic (52→1)  ← NO FINGERS!
    """

    def __init__(self, loco_obs=57, arm_obs=52, loco_act=12, arm_act=5):
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

        loco_dist = torch.distributions.Normal(loco_mean, loco_std)
        arm_dist = torch.distributions.Normal(arm_mean, arm_std)

        return loco_dist.sample(), arm_dist.sample()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
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


def get_palm_forward(quat: torch.Tensor) -> torch.Tensor:
    """Get palm forward direction from quaternion (local +X axis)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fwd_x = 1 - 2 * (y * y + z * z)
    fwd_y = 2 * (x * y + w * z)
    fwd_z = 2 * (x * z - w * y)
    return torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)


def compute_orientation_error(palm_quat: torch.Tensor, target_dir: torch.Tensor = None) -> torch.Tensor:
    """Compute orientation error between palm forward and target direction."""
    forward = get_palm_forward(palm_quat)
    if target_dir is None:
        target_dir = torch.zeros_like(forward)
        target_dir[:, 2] = -1.0  # Default: palm down
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.acos(dot)


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
    action_space = 17  # 12 leg + 5 arm (NO FINGERS!)
    observation_space = 57  # loco obs
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
            [joint_limits[0, i, 0].item() for i in self.finger_idx],
            device=self.device
        )
        self.finger_upper = torch.tensor(
            [joint_limits[0, i, 1].item() for i in self.finger_idx],
            device=self.device
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

        self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body[:, 2] = -1.0  # Default: palm down

        self.phase = torch.zeros(self.num_envs, device=self.device)

        # Action history
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Stats
        self.total_reaches = 0
        self.reach_threshold = 0.05  # Position threshold for reach
        self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Mode config
        self.mode_cfg = MODE_CONFIGS[args.mode]

        print(f"[PlayEnv] Leg joints: {len(self.leg_idx)}")
        print(f"[PlayEnv] Arm joints: {len(self.arm_idx)}")
        print(f"[PlayEnv] Finger joints: {len(self.finger_idx)} (FIXED OPEN)")
        print(f"[PlayEnv] Palm body idx: {self.palm_idx}")

        # Markers
        self._markers_initialized = False

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
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
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
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red
                    ),
                },
            )
        )
        self._markers_initialized = True

    def _compute_palm_ee(self) -> tuple:
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_forward = get_palm_forward(palm_quat)
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward
        return ee_pos, palm_quat

    def _sample_commands(self, env_ids):
        n = len(env_ids)
        cfg = self.mode_cfg

        # Velocity commands
        self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*cfg["vx_range"])
        self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*cfg["vy_range"])
        self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*cfg["vyaw_range"])
        self.height_cmd[env_ids] = HEIGHT_DEFAULT

        # Target position
        ws = cfg["workspace_radius"]
        azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
        elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
        radius = torch.empty(n, device=self.device).uniform_(*ws)

        x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
        y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
        z = radius * torch.sin(elevation) + self.shoulder_offset[2]

        self.target_pos_body[env_ids, 0] = x.clamp(0.05, 0.55)
        self.target_pos_body[env_ids, 1] = y.clamp(-0.55, 0.10)
        self.target_pos_body[env_ids, 2] = z.clamp(-0.25, 0.55)

        # Target orientation (variable - sample from cone around down)
        orient_range = 0.7  # ~40° cone
        theta = torch.empty(n, device=self.device).uniform_(0, orient_range)
        phi = torch.empty(n, device=self.device).uniform_(0, 2 * np.pi)

        dir_x = torch.sin(theta) * torch.cos(phi)
        dir_y = torch.sin(theta) * torch.sin(phi)
        dir_z = -torch.cos(theta)

        self.target_orient_body[env_ids, 0] = dir_x
        self.target_orient_body[env_ids, 1] = dir_y
        self.target_orient_body[env_ids, 2] = dir_z

        self.already_reached[env_ids] = False

    def get_loco_obs(self) -> torch.Tensor:
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

    def get_arm_obs(self) -> torch.Tensor:
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

        target_reached = (dist_to_target < self.reach_threshold).float()

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

        obs = torch.cat([
            arm_pos, arm_vel, finger_pos,
            ee_body, ee_vel_body, palm_quat, grip_force, gripper_closed_ratio, contact_detected,
            target_body, pos_error, pos_dist, orient_err, target_reached,
            height_cmd_obs, current_height, height_err,
            estimated_load, object_in_hand_obs, target_orient_obs,
            lin_vel_xy, ang_vel_z,
        ], dim=-1)
        return obs.clamp(-10, 10).nan_to_num()

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]  # Only 5 arm joints

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5

        # FINGERS ALWAYS OPEN
        target_pos[:, self.finger_idx] = self.finger_lower

        self.robot.set_joint_position_target(target_pos)
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

        # Check reaches
        ee_pos, palm_quat = self._compute_palm_ee()
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist = torch.norm(ee_pos - target_world, dim=-1)

        reached = dist < self.reach_threshold
        new_reaches = reached & ~self.already_reached
        if new_reaches.any():
            reached_ids = torch.where(new_reaches)[0]
            self.total_reaches += len(reached_ids)
            self.already_reached[reached_ids] = True
            self._sample_commands(reached_ids)

        # Update markers
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

    def _get_observations(self) -> dict:
        return {"policy": self.get_loco_obs()}

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

    print(f"\n{'=' * 60}")
    print("ULC G1 STAGE 6 SIMPLIFIED - PLAY")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {args.mode}")
    print(f"  Description: {MODE_CONFIGS[args.mode]['description']}")
    print(f"{'=' * 60}\n")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print(f"\n[Checkpoint Info]")
    if "best_reward" in checkpoint:
        print(f"  Best reward: {checkpoint['best_reward']:.2f}")
    if "iteration" in checkpoint:
        print(f"  Iteration: {checkpoint['iteration']}")
    if "curriculum_level" in checkpoint:
        print(f"  Curriculum level: {checkpoint['curriculum_level']}")
    if "total_reaches" in checkpoint:
        print(f"  Total reaches: {checkpoint['total_reaches']}")

    # Create environment
    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = PlayEnv(cfg)

    # Create network - MUST MATCH TRAINING!
    net = DualActorCritic(loco_obs=57, arm_obs=52, loco_act=12, arm_act=5).to(device)

    # Load weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Verify keys match
    model_keys = set(net.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"\n⚠️  MISSING KEYS: {len(missing)}")
        for k in sorted(missing)[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"\n⚠️  UNEXPECTED KEYS: {len(unexpected)}")
        for k in sorted(unexpected)[:5]:
            print(f"    {k}")

    if not missing and not unexpected:
        print(f"\n✅ All {len(model_keys)} keys match perfectly!")

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print(f"[INFO] Loaded checkpoint successfully\n")

    # Run
    obs, _ = env.reset()
    prev_reaches = 0

    print(f"[Play] Starting {args.steps} steps in '{args.mode}' mode...")
    print(f"       Press Ctrl+C to stop\n")

    with torch.no_grad():
        for step in range(args.steps):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            leg_actions, arm_actions = net.get_actions(loco_obs, arm_obs, deterministic=False)
            actions = torch.cat([leg_actions, arm_actions], dim=-1)

            obs, reward, terminated, truncated, info = env.step(actions)

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
    print(f"{'=' * 60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()