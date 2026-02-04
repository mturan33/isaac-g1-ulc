"""
G1 Decoupled Policy Demo - Option B Implementation
===================================================

İki AYRI policy'yi bir COORDINATOR ile birleştirir:
- Stage 3 Loco Policy (frozen): Yürüme kontrolü
- Stage 5 Arm Policy (frozen): Reaching kontrolü

Çalışma Mantığı:
1. IDLE → Target pozisyon verilir
2. WALKING → Robot hedefe yürür (arm sabit)
3. REACHING → Robot durur, arm uzanır
4. DONE → Başarılı

KULLANIM:
./isaaclab.bat -p <path>/g1_decoupled_demo.py --num_envs 1

Turan Özhan - VLM-RL G1 Humanoid Project
4 Şubat 2026
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

parser = argparse.ArgumentParser(description="G1 Decoupled Policy Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--loco_checkpoint", type=str,
                    default="logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt",
                    help="Path to Stage 3 loco checkpoint")
parser.add_argument("--arm_checkpoint", type=str,
                    default="logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt",
                    help="Path to Stage 5 arm checkpoint")
parser.add_argument("--target_x", type=float, default=1.5, help="Target X position")
parser.add_argument("--target_y", type=float, default=0.0, help="Target Y position")
parser.add_argument("--target_z", type=float, default=0.8, help="Target Z position")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply_inverse

# ============================================================================
# CONSTANTS
# ============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Stage 3 leg joint names (exact order)
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

# Stage 5 arm joint names (right arm only)
ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# Default positions
DEFAULT_LEG_POS = torch.tensor([-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0])
DEFAULT_ARM_POS = torch.tensor([-0.3, 0.0, 0.0, 0.5, 0.0])

# EE offset (from palm to fingertip approximation)
EE_OFFSET = 0.02

# Arm joint limits
ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}


# ============================================================================
# COORDINATOR STATE MACHINE
# ============================================================================

class CoordinatorState(Enum):
    IDLE = 0
    WALKING = 1
    APPROACHING = 2  # Slow down phase
    REACHING = 3
    DONE = 4
    ERROR = 5


@dataclass
class TargetInfo:
    """Target information"""
    position_world: torch.Tensor  # [3] - World frame position
    confidence: float = 1.0
    label: str = "target"


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class LocoActor(nn.Module):
    """Stage 3 Loco Actor - EXACT architecture match"""

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
    """Stage 5 Arm Actor - EXACT architecture match (RSL-RL default)"""

    def __init__(self, num_obs=29, num_act=5, hidden=[256, 128, 64]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# POLICY LOADER
# ============================================================================

def load_loco_policy(checkpoint_path: str, device: str = "cuda:0") -> LocoActor:
    """Load Stage 3 locomotion policy"""

    actor = LocoActor().to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Loco checkpoint not found: {checkpoint_path}")
        print("[WARNING] Using random initialization!")
        return actor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Stage 3 format: {"actor_critic": {"actor.0.weight": ..., ...}, ...}
    if "actor_critic" in ckpt:
        state_dict = {}
        for key, value in ckpt["actor_critic"].items():
            if key.startswith("actor."):
                # actor.0.weight -> net.0.weight
                new_key = "net." + key[6:]
                state_dict[new_key] = value
        actor.load_state_dict(state_dict)
        print(f"[OK] Loaded loco policy from {checkpoint_path}")
    elif "model" in ckpt:
        # RSL-RL format
        state_dict = {}
        for key, value in ckpt["model"].items():
            if "actor" in key:
                new_key = key.replace("actor.", "net.")
                state_dict[new_key] = value
        actor.load_state_dict(state_dict, strict=False)
        print(f"[OK] Loaded loco policy (RSL-RL format) from {checkpoint_path}")
    else:
        print(f"[WARNING] Unknown checkpoint format: {list(ckpt.keys())}")

    actor.eval()
    return actor


def load_arm_policy(checkpoint_path: str, device: str = "cuda:0") -> ArmActor:
    """Load Stage 5 arm policy"""

    actor = ArmActor().to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Arm checkpoint not found: {checkpoint_path}")
        print("[WARNING] Using random initialization!")
        return actor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # RSL-RL format: {"model": {"actor.0.weight": ..., ...}, ...}
    if "model" in ckpt:
        state_dict = {}
        for key, value in ckpt["model"].items():
            if "actor" in key and "critic" not in key:
                # actor.0.weight -> net.0.weight
                new_key = key.replace("actor.", "net.")
                state_dict[new_key] = value

        # Try to load
        try:
            actor.load_state_dict(state_dict, strict=True)
            print(f"[OK] Loaded arm policy from {checkpoint_path}")
        except Exception as e:
            print(f"[WARNING] Partial load: {e}")
            actor.load_state_dict(state_dict, strict=False)
    elif "actor_critic" in ckpt:
        # Custom format
        state_dict = {}
        for key, value in ckpt["actor_critic"].items():
            if key.startswith("actor."):
                new_key = "net." + key[6:]
                state_dict[new_key] = value
        actor.load_state_dict(state_dict)
        print(f"[OK] Loaded arm policy from {checkpoint_path}")
    else:
        print(f"[WARNING] Unknown checkpoint format: {list(ckpt.keys())}")

    actor.eval()
    return actor


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


def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz format)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular difference between two quaternions in radians."""
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


# ============================================================================
# DECOUPLED COORDINATOR
# ============================================================================

class DecoupledCoordinator:
    """
    State machine that coordinates loco and arm policies.

    Logic:
    1. IDLE: Waiting for target
    2. WALKING: Walk to target (arm fixed)
    3. APPROACHING: Slow down near target
    4. REACHING: Stop walking, extend arm
    5. DONE: Success
    """

    def __init__(
            self,
            loco_policy: LocoActor,
            arm_policy: ArmActor,
            device: str = "cuda:0",
            walk_threshold: float = 0.6,  # Start reaching when closer than this
            approach_threshold: float = 0.4,  # Start slowing down
            reach_threshold: float = 0.08,  # Success threshold
            max_walk_speed: float = 0.5,
    ):
        self.loco_policy = loco_policy
        self.arm_policy = arm_policy
        self.device = device

        # Thresholds
        self.walk_threshold = walk_threshold
        self.approach_threshold = approach_threshold
        self.reach_threshold = reach_threshold
        self.max_walk_speed = max_walk_speed

        # State
        self.state = CoordinatorState.IDLE
        self.target: Optional[TargetInfo] = None
        self.steps_in_state = 0

        # Default actions
        self.default_arm_actions = torch.zeros(5, device=device)
        self.standing_leg_actions = torch.zeros(12, device=device)

        # Gait phase (for loco policy)
        self.gait_phase = 0.0
        self.gait_frequency = 1.5

        print(f"[Coordinator] Initialized")
        print(f"  Walk threshold: {walk_threshold}m")
        print(f"  Reach threshold: {reach_threshold}m")

    def reset(self):
        """Reset coordinator state"""
        self.state = CoordinatorState.IDLE
        self.target = None
        self.steps_in_state = 0
        self.gait_phase = 0.0

    def set_target(self, target: TargetInfo):
        """Set target and start walking"""
        self.target = target
        self.state = CoordinatorState.WALKING
        self.steps_in_state = 0
        print(f"[Coordinator] Target set: {target.label} at {target.position_world}")

    def step(
            self,
            robot_pos: torch.Tensor,  # [N, 3] World position
            robot_quat: torch.Tensor,  # [N, 4] World quaternion (xyzw)
            robot_lin_vel: torch.Tensor,  # [N, 3] World linear velocity
            robot_ang_vel: torch.Tensor,  # [N, 3] World angular velocity
            leg_joint_pos: torch.Tensor,  # [N, 12] Leg joint positions
            leg_joint_vel: torch.Tensor,  # [N, 12] Leg joint velocities
            arm_joint_pos: torch.Tensor,  # [N, 5] Arm joint positions
            arm_joint_vel: torch.Tensor,  # [N, 5] Arm joint velocities
            ee_pos_world: torch.Tensor,  # [N, 3] End-effector world position
            dt: float = 0.02,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run coordinator step.

        Returns:
            actions: [N, 17] = [leg_actions (12) + arm_actions (5)]
            info: dict with state information
        """
        N = robot_pos.shape[0]

        if self.target is None:
            # No target, stay idle
            actions = torch.cat([
                self.standing_leg_actions.expand(N, -1),
                self.default_arm_actions.expand(N, -1)
            ], dim=-1)
            return actions, {"state": self.state.name}

        # Calculate distances
        target_pos = self.target.position_world.unsqueeze(0).expand(N, -1)
        target_distance = (robot_pos[:, :2] - target_pos[:, :2]).norm(dim=-1)  # XY only
        ee_distance = (ee_pos_world - target_pos).norm(dim=-1)

        # State transitions
        self._update_state(target_distance, ee_distance)

        # Generate actions based on state
        if self.state == CoordinatorState.WALKING:
            leg_actions, vel_cmd = self._walking_step(
                robot_pos, robot_quat, robot_lin_vel, robot_ang_vel,
                leg_joint_pos, leg_joint_vel, target_pos, target_distance, dt
            )
            arm_actions = self.default_arm_actions.expand(N, -1)

        elif self.state == CoordinatorState.APPROACHING:
            # Slow walking
            leg_actions, vel_cmd = self._walking_step(
                robot_pos, robot_quat, robot_lin_vel, robot_ang_vel,
                leg_joint_pos, leg_joint_vel, target_pos, target_distance, dt,
                speed_scale=0.3  # Slower
            )
            arm_actions = self.default_arm_actions.expand(N, -1)

        elif self.state == CoordinatorState.REACHING:
            # Standing still + arm reaching
            leg_actions = self.standing_leg_actions.expand(N, -1)
            arm_actions = self._reaching_step(
                robot_pos, robot_quat,
                arm_joint_pos, arm_joint_vel, target_pos, ee_pos_world
            )
            vel_cmd = torch.zeros(N, 3, device=self.device)

        else:  # IDLE or DONE
            leg_actions = self.standing_leg_actions.expand(N, -1)
            arm_actions = self.default_arm_actions.expand(N, -1)
            vel_cmd = torch.zeros(N, 3, device=self.device)

        actions = torch.cat([leg_actions, arm_actions], dim=-1)

        info = {
            "state": self.state.name,
            "target_distance": target_distance.mean().item(),
            "ee_distance": ee_distance.mean().item(),
            "steps_in_state": self.steps_in_state,
        }

        self.steps_in_state += 1

        return actions, info

    def _update_state(self, target_distance: torch.Tensor, ee_distance: torch.Tensor):
        """Update coordinator state based on distances"""

        mean_target_dist = target_distance.mean().item()
        mean_ee_dist = ee_distance.mean().item()

        if self.state == CoordinatorState.WALKING:
            if mean_target_dist < self.approach_threshold:
                print(f"[Coordinator] WALKING → APPROACHING (dist={mean_target_dist:.2f}m)")
                self.state = CoordinatorState.APPROACHING
                self.steps_in_state = 0

        elif self.state == CoordinatorState.APPROACHING:
            if mean_target_dist < self.walk_threshold:
                print(f"[Coordinator] APPROACHING → REACHING (dist={mean_target_dist:.2f}m)")
                self.state = CoordinatorState.REACHING
                self.steps_in_state = 0

        elif self.state == CoordinatorState.REACHING:
            if mean_ee_dist < self.reach_threshold:
                print(f"[Coordinator] REACHING → DONE! (ee_dist={mean_ee_dist:.3f}m)")
                self.state = CoordinatorState.DONE
                self.steps_in_state = 0

    def _walking_step(
            self,
            robot_pos: torch.Tensor,
            robot_quat: torch.Tensor,
            robot_lin_vel: torch.Tensor,
            robot_ang_vel: torch.Tensor,
            leg_joint_pos: torch.Tensor,
            leg_joint_vel: torch.Tensor,
            target_pos: torch.Tensor,
            target_distance: torch.Tensor,
            dt: float,
            speed_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate walking actions using loco policy"""

        N = robot_pos.shape[0]

        # Calculate velocity commands
        direction = target_pos[:, :2] - robot_pos[:, :2]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)

        # Convert to body frame
        euler = quat_to_euler_xyz(robot_quat)
        yaw = euler[:, 2]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx_world = direction[:, 0] * self.max_walk_speed * speed_scale
        vy_world = direction[:, 1] * self.max_walk_speed * speed_scale

        # Rotate to body frame
        vx_body = vx_world * cos_yaw + vy_world * sin_yaw
        vy_body = -vx_world * sin_yaw + vy_world * cos_yaw

        # Yaw rate to face target
        target_yaw = torch.atan2(direction[:, 1], direction[:, 0])
        yaw_error = target_yaw - yaw
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))  # Normalize
        vyaw = torch.clamp(yaw_error * 1.0, -0.5, 0.5)

        vel_cmd = torch.stack([vx_body, vy_body, vyaw], dim=-1)

        # Build loco observation (57 dims)
        loco_obs = self._build_loco_obs(
            robot_quat, robot_lin_vel, robot_ang_vel,
            leg_joint_pos, leg_joint_vel, vel_cmd, dt
        )

        # Get actions from policy
        with torch.no_grad():
            leg_actions = self.loco_policy(loco_obs)

        return leg_actions, vel_cmd

    def _build_loco_obs(
            self,
            robot_quat: torch.Tensor,
            robot_lin_vel: torch.Tensor,
            robot_ang_vel: torch.Tensor,
            leg_joint_pos: torch.Tensor,
            leg_joint_vel: torch.Tensor,
            vel_cmd: torch.Tensor,
            dt: float,
    ) -> torch.Tensor:
        """Build Stage 3 loco observation (57 dims)"""

        N = robot_quat.shape[0]

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(robot_quat, robot_lin_vel)
        ang_vel_b = quat_apply_inverse(robot_quat, robot_ang_vel)

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(N, -1)
        proj_gravity = quat_apply_inverse(robot_quat, gravity_vec)

        # Height command (fixed)
        height_cmd = torch.ones(N, 1, device=self.device) * 0.72

        # Gait phase
        self.gait_phase = (self.gait_phase + self.gait_frequency * dt) % 1.0
        gait_phase = torch.tensor([
            [math.sin(2 * math.pi * self.gait_phase),
             math.cos(2 * math.pi * self.gait_phase)]
        ], device=self.device).expand(N, -1)

        # Previous actions (use zeros for simplicity)
        prev_actions = torch.zeros(N, 12, device=self.device)

        # Stage 3 observation (51 dims originally, but training was 57)
        # Let me check: 3+3+3+12+12+1+3+2+12 = 51 for Stage 3
        # But config says 57... Stage 3 also has torso commands maybe?

        # Based on Document 1 (Stage 3): 57 obs = Stage 2 (51) + torso commands (6)
        # torso_cmd (3) + torso_euler (3) = 6
        torso_cmd = torch.zeros(N, 3, device=self.device)
        torso_euler = quat_to_euler_xyz(robot_quat)

        obs = torch.cat([
            lin_vel_b,  # 3
            ang_vel_b,  # 3
            proj_gravity,  # 3
            leg_joint_pos,  # 12
            leg_joint_vel,  # 12
            height_cmd,  # 1
            vel_cmd,  # 3
            gait_phase,  # 2
            prev_actions,  # 12
            torso_cmd,  # 3
            torso_euler,  # 3
        ], dim=-1)  # Total: 57

        return obs.clamp(-10, 10)

    def _reaching_step(
            self,
            robot_pos: torch.Tensor,
            robot_quat: torch.Tensor,
            arm_joint_pos: torch.Tensor,
            arm_joint_vel: torch.Tensor,
            target_pos: torch.Tensor,
            ee_pos_world: torch.Tensor,
    ) -> torch.Tensor:
        """Generate arm reaching actions using arm policy"""

        N = robot_pos.shape[0]

        # Convert target to body frame (relative to shoulder)
        shoulder_offset = torch.tensor([0.0, -0.174, 0.259], device=self.device)
        shoulder_world = robot_pos + shoulder_offset

        target_rel = target_pos - robot_pos
        ee_rel = ee_pos_world - robot_pos

        # Target quaternion (fixed orientation for now)
        target_quat = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=self.device).expand(N, -1)

        # Current EE orientation (from robot)
        ee_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(N, -1)

        # Position error
        pos_err = target_rel - ee_rel
        pos_dist = pos_err.norm(dim=-1, keepdim=True)

        # Orientation error
        ori_err = quat_diff_rad(ee_quat, target_quat).unsqueeze(-1)

        # Build arm observation (29 dims)
        arm_obs = torch.cat([
            arm_joint_pos,  # 5
            arm_joint_vel * 0.1,  # 5
            target_rel,  # 3
            target_quat,  # 4
            ee_rel,  # 3
            ee_quat,  # 4
            pos_err,  # 3
            ori_err,  # 1
            pos_dist / 0.5,  # 1
        ], dim=-1)  # Total: 29

        # Get actions from policy
        with torch.no_grad():
            arm_actions = self.arm_policy(arm_obs)

        return arm_actions


# ============================================================================
# DEMO ENVIRONMENT
# ============================================================================

@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Scene config for demo"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                "left_hip_pitch_joint": -0.2,
                "right_hip_pitch_joint": -0.2,
                "left_knee_joint": 0.4,
                "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2,
                "right_ankle_pitch_joint": -0.2,
                "right_shoulder_pitch_joint": -0.3,
                "right_elbow_pitch_joint": 0.5,
                "left_shoulder_pitch_joint": -0.3,
                "left_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=150.0,
                damping=15.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=100.0,
                damping=10.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                emissive_color=(0.5, 0.0, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.8)),
    )


@configclass
class DemoEnvCfg(DirectRLEnvCfg):
    """Environment config for demo"""

    decimation = 4
    episode_length_s = 60.0

    num_actions = 17  # 12 legs + 5 arm
    num_observations = 50  # Placeholder
    num_states = 0

    action_space = 17
    observation_space = 50
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        device="cuda:0",
    )

    scene: DemoSceneCfg = DemoSceneCfg(num_envs=1, env_spacing=5.0)


class DemoEnv(DirectRLEnv):
    """Demo environment for decoupled policy testing"""

    cfg: DemoEnvCfg

    def __init__(self, cfg: DemoEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]

        # Joint indices
        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        self.leg_indices = []
        for jn in LEG_JOINT_NAMES:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.leg_indices.append(i)
                    break
        self.leg_indices = torch.tensor(self.leg_indices, device=self.device, dtype=torch.long)

        self.arm_indices = []
        for jn in ARM_JOINT_NAMES:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.arm_indices.append(i)
                    break
        self.arm_indices = torch.tensor(self.arm_indices, device=self.device, dtype=torch.long)

        # Palm body index
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break

        if self.palm_idx is None:
            # Fallback
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "elbow" in name.lower():
                    self.palm_idx = i
                    break

        # Default positions
        self.default_leg = DEFAULT_LEG_POS.to(self.device)
        self.default_arm = DEFAULT_ARM_POS.to(self.device)

        # Arm joint limits
        self.arm_lower = torch.zeros(5, device=self.device)
        self.arm_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(ARM_JOINT_NAMES):
            self.arm_lower[i], self.arm_upper[i] = ARM_JOINT_LIMITS[jn]

        # Previous actions for smoothing
        self.prev_actions = torch.zeros(self.num_envs, 17, device=self.device)

        # Local forward vector
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print(f"\n[DemoEnv] Initialized")
        print(f"  Leg joints: {len(self.leg_indices)}")
        print(f"  Arm joints: {len(self.arm_indices)}")
        print(f"  Palm idx: {self.palm_idx}")
        print(f"  Joint names: {joint_names[:10]}...")

    def _setup_scene(self):
        """Setup scene"""
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]

    def get_ee_pos_world(self) -> torch.Tensor:
        """Get end-effector position in world frame"""
        if self.palm_idx is not None:
            palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
            palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]

            # Convert quat to wxyz for rotate function
            palm_quat_wxyz = torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)
            forward = rotate_vector_by_quat(self.local_forward, palm_quat_wxyz)
            return palm_pos + EE_OFFSET * forward
        else:
            return self.robot.data.root_pos_w

    def set_target_position(self, pos: torch.Tensor):
        """Set target position"""
        N = pos.shape[0] if len(pos.shape) > 1 else 1
        pos_expanded = pos.view(N, 3) if len(pos.shape) > 1 else pos.unsqueeze(0)

        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(N, -1)
        pose = torch.cat([pos_expanded, quat], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step"""
        self.actions = actions.clone()

        # Split actions
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:]

        # Compute targets
        target_pos = self.robot.data.default_joint_pos.clone()

        # Legs
        target_pos[:, self.leg_indices] = self.default_leg + leg_actions * 0.4

        # Arms (with limits)
        arm_target = self.default_arm + arm_actions * 0.15  # Smaller scale for arms
        arm_target = torch.clamp(arm_target, self.arm_lower, self.arm_upper)
        target_pos[:, self.arm_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)

        # Smooth actions
        alpha = 0.3
        self.prev_actions = alpha * actions + (1 - alpha) * self.prev_actions

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        """Placeholder observations"""
        obs = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        N = len(env_ids)

        # Reset pose
        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(N, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(N, -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=self.device), env_ids)

        # Reset joints
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(
            default_joint_pos,
            torch.zeros_like(default_joint_pos),
            None,
            env_ids
        )

        self.prev_actions[env_ids] = 0


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    G1 DECOUPLED POLICY DEMO - Option B")
    print("=" * 70)

    device = "cuda:0"

    # 1. Load policies
    print("\n[1/4] Loading policies...")
    loco_policy = load_loco_policy(args.loco_checkpoint, device)
    arm_policy = load_arm_policy(args.arm_checkpoint, device)

    # 2. Create coordinator
    print("\n[2/4] Creating coordinator...")
    coordinator = DecoupledCoordinator(
        loco_policy=loco_policy,
        arm_policy=arm_policy,
        device=device,
        walk_threshold=0.5,
        approach_threshold=0.8,
        reach_threshold=0.10,
        max_walk_speed=0.4,
    )

    # 3. Create environment
    print("\n[3/4] Creating environment...")
    env_cfg = DemoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DemoEnv(cfg=env_cfg)

    # 4. Set target
    print("\n[4/4] Setting target...")
    target_pos = torch.tensor([args.target_x, args.target_y, args.target_z], device=device)
    env.set_target_position(target_pos)

    target_info = TargetInfo(
        position_world=target_pos,
        confidence=1.0,
        label="red_ball"
    )
    coordinator.set_target(target_info)

    print("\n" + "=" * 70)
    print("    STARTING DEMO")
    print(f"    Target: ({args.target_x}, {args.target_y}, {args.target_z})")
    print("=" * 70 + "\n")

    # Reset environment
    obs, _ = env.reset()

    # Main loop
    step = 0
    max_steps = 3000  # 60 seconds at 50Hz
    dt = env.cfg.sim.dt * env.cfg.decimation

    try:
        while step < max_steps and coordinator.state != CoordinatorState.DONE:
            # Get robot state
            robot_pos = env.robot.data.root_pos_w
            robot_quat = env.robot.data.root_quat_w
            robot_lin_vel = env.robot.data.root_lin_vel_w
            robot_ang_vel = env.robot.data.root_ang_vel_w

            leg_joint_pos = env.robot.data.joint_pos[:, env.leg_indices]
            leg_joint_vel = env.robot.data.joint_vel[:, env.leg_indices]
            arm_joint_pos = env.robot.data.joint_pos[:, env.arm_indices]
            arm_joint_vel = env.robot.data.joint_vel[:, env.arm_indices]

            ee_pos = env.get_ee_pos_world()

            # Coordinator step
            actions, info = coordinator.step(
                robot_pos, robot_quat, robot_lin_vel, robot_ang_vel,
                leg_joint_pos, leg_joint_vel, arm_joint_pos, arm_joint_vel,
                ee_pos, dt
            )

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(actions)

            # Log
            if step % 50 == 0:
                height = robot_pos[0, 2].item()
                print(
                    f"Step {step:4d} | "
                    f"State={info['state']:12s} | "
                    f"H={height:.2f}m | "
                    f"TargetDist={info.get('target_distance', 0):.2f}m | "
                    f"EEDist={info.get('ee_distance', 0):.3f}m"
                )

            step += 1

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    # Final status
    print("\n" + "=" * 70)
    if coordinator.state == CoordinatorState.DONE:
        print("    ✅ SUCCESS! Target reached!")
    else:
        print(f"    ⏹️  Demo ended in state: {coordinator.state.name}")
    print(f"    Total steps: {step}")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()