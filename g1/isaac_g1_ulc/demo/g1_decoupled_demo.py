"""
G1 Decoupled Policy Demo V3 - Standing Balance + Reaching
==========================================================

- Loco policy (Stage 3) vx=0 ile çalışır → Active standing balance
- Arm policy (Stage 5) reaching yapar
- Gravity AÇIK → Gerçekçi simülasyon

KULLANIM:
./isaaclab.bat -p g1_decoupled_demo_v3.py --num_envs 1

Turan Özhan - VLM-RL G1 Humanoid Project
4 Şubat 2026
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from typing import Tuple

parser = argparse.ArgumentParser(description="G1 Decoupled Demo V3 - Standing + Reaching")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--loco_checkpoint", type=str,
                    default="logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt",
                    help="Path to Stage 3 loco checkpoint")
parser.add_argument("--arm_checkpoint", type=str,
                    default="logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt",
                    help="Path to Stage 5 arm checkpoint")
parser.add_argument("--spawn_radius", type=float, default=0.25, help="Target spawn radius (m)")
parser.add_argument("--steps", type=int, default=3000, help="Number of steps")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply_inverse


# ============================================================================
# CONSTANTS
# ============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Stage 3 leg joints (exact order)
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

# Stage 5 arm joints (right arm)
ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# Defaults
DEFAULT_LEG_POS = [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0]
DEFAULT_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.5,
    "right_elbow_roll_joint": 0.0,
}

ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

SHOULDER_CENTER_OFFSET = [0.0, -0.174, 0.259]
WORKSPACE_INNER_RADIUS = 0.18
WORKSPACE_OUTER_RADIUS = 0.45
EE_OFFSET = 0.02
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to euler angles (roll, pitch, yaw)."""
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
# NEURAL NETWORKS
# ============================================================================

class LocoActor(nn.Module):
    """Stage 3 Loco Actor - [512, 256, 128] + LayerNorm + ELU"""

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
    """Stage 5 Arm Actor - [256, 128, 64] + ELU (RSL-RL default)"""

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


def load_loco_policy(checkpoint_path: str, device: str) -> LocoActor:
    """Load Stage 3 locomotion policy"""

    actor = LocoActor().to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Loco checkpoint not found: {checkpoint_path}")
        return actor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Stage 3 format: {"actor_critic": {"actor.0.weight": ..., ...}, ...}
    if "actor_critic" in ckpt:
        state_dict = {}
        for key, value in ckpt["actor_critic"].items():
            if key.startswith("actor."):
                new_key = "net." + key[6:]
                state_dict[new_key] = value
        actor.load_state_dict(state_dict)
        print(f"[OK] Loaded loco policy from {checkpoint_path}")
    else:
        print(f"[WARNING] Unknown loco checkpoint format: {list(ckpt.keys())}")

    actor.eval()
    return actor


def load_arm_policy(checkpoint_path: str, device: str) -> ArmActor:
    """Load Stage 5 arm policy"""

    actor = ArmActor().to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Arm checkpoint not found: {checkpoint_path}")
        return actor

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # RSL-RL format
    if "model_state_dict" in ckpt:
        state_dict = {}
        for key, value in ckpt["model_state_dict"].items():
            if "actor" in key and "critic" not in key:
                new_key = key.replace("actor.", "net.")
                state_dict[new_key] = value

        try:
            actor.load_state_dict(state_dict, strict=True)
            print(f"[OK] Loaded arm policy from {checkpoint_path}")
        except Exception as e:
            print(f"[WARNING] Partial arm load: {e}")
            actor.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARNING] Unknown arm checkpoint format: {list(ckpt.keys())}")

    actor.eval()
    return actor


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Scene with gravity enabled"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
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
                disable_gravity=False,  # GRAVITY AÇIK!
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
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                emissive_color=(0.0, 0.5, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.0)),
    )

    ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
                emissive_color=(0.5, 0.25, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


@configclass
class DemoEnvCfg(DirectRLEnvCfg):
    """Environment config"""

    decimation = 4
    episode_length_s = 30.0

    num_actions = 17  # 12 legs + 5 arm
    num_observations = 50
    num_states = 0

    action_space = 17
    observation_space = 50
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        device="cuda:0",
    )

    scene: DemoSceneCfg = DemoSceneCfg(num_envs=1, env_spacing=2.5)

    # Arm parameters
    action_smoothing_alpha = 0.15
    arm_action_scale = 0.12
    reach_threshold = 0.08


class DemoEnv(DirectRLEnv):
    """Demo environment with active balance + reaching"""

    cfg: DemoEnvCfg

    def __init__(self, cfg: DemoEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        # Get joint indices
        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        # Leg indices (Stage 3 order)
        self.leg_indices = []
        for jn in LEG_JOINT_NAMES:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.leg_indices.append(i)
                    break
        self.leg_indices = torch.tensor(self.leg_indices, device=self.device, dtype=torch.long)

        # Arm indices (Stage 5 order)
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

        # Default positions
        self.default_leg = torch.tensor(DEFAULT_LEG_POS, device=self.device)
        self.default_arm = torch.tensor(
            [DEFAULT_ARM_POSE[jn] for jn in ARM_JOINT_NAMES], device=self.device
        )

        # Arm joint limits
        self.arm_lower = torch.zeros(5, device=self.device)
        self.arm_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(ARM_JOINT_NAMES):
            self.arm_lower[i], self.arm_upper[i] = ARM_JOINT_LIMITS[jn]

        # Shoulder center (body frame)
        self.shoulder_center = torch.tensor(
            SHOULDER_CENTER_OFFSET, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Target (body frame)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.tensor(
            [[0.707, 0.707, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1).clone()

        # Loco state
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)

        # Arm state
        self.smoothed_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

        # Commands (loco policy)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # vx=0, vy=0, vyaw=0
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # Local forward
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Stats
        self.total_reaches = 0

        print(f"\n[DemoEnv] Initialized with GRAVITY")
        print(f"  Leg joints: {len(self.leg_indices)}")
        print(f"  Arm joints: {len(self.arm_indices)}")
        print(f"  Palm idx: {self.palm_idx}")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    def _compute_ee_pos(self) -> torch.Tensor:
        """Get EE position in world frame"""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_quat_wxyz = torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)
        forward = rotate_vector_by_quat(self.local_forward, palm_quat_wxyz)
        return palm_pos + EE_OFFSET * forward

    def _compute_ee_quat(self) -> torch.Tensor:
        """Get EE orientation (wxyz)"""
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        return torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)

    def sample_target_in_workspace(self, env_ids: torch.Tensor = None):
        """Sample target in workspace (body frame)"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        num = len(env_ids)
        root_pos = self.robot.data.root_pos_w[env_ids]

        # Random direction (front-biased)
        direction = torch.randn((num, 3), device=self.device)
        direction[:, 0] = -torch.abs(direction[:, 0])  # -X = front
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Random distance
        inner = WORKSPACE_INNER_RADIUS
        outer = min(args.spawn_radius, WORKSPACE_OUTER_RADIUS)
        distance = inner + torch.rand((num, 1), device=self.device) * (outer - inner)

        # Target relative to shoulder
        target_rel = self.shoulder_center[env_ids] + direction * distance
        target_rel[:, 2] = torch.clamp(target_rel[:, 2], 0.05, 0.55)

        self.target_pos[env_ids] = target_rel

        # Update visual
        target_world = root_pos + target_rel
        pose = torch.cat([target_world, torch.tensor([[0, 0, 0, 1]], device=self.device).expand(num, -1)], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        print(f"[Target] Body frame: [{target_rel[0, 0]:.2f}, {target_rel[0, 1]:.2f}, {target_rel[0, 2]:.2f}]")

    def build_loco_obs(self) -> torch.Tensor:
        """Build Stage 3 loco observation (57 dims)"""
        robot = self.robot
        quat = robot.data.root_quat_w

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # Projected gravity
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity)

        # Leg joints
        leg_pos = robot.data.joint_pos[:, self.leg_indices]
        leg_vel = robot.data.joint_vel[:, self.leg_indices]

        # Gait phase
        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.gait_phase),
            torch.cos(2 * np.pi * self.gait_phase)
        ], dim=-1)

        # Torso euler
        torso_euler = quat_to_euler_xyz(quat)

        obs = torch.cat([
            lin_vel_b,                        # 3
            ang_vel_b,                        # 3
            proj_gravity,                     # 3
            leg_pos,                          # 12
            leg_vel,                          # 12
            self.height_cmd.unsqueeze(-1),    # 1
            self.vel_cmd,                     # 3 (vx=0, vy=0, vyaw=0)
            gait_phase,                       # 2
            self.prev_leg_actions,            # 12
            self.torso_cmd,                   # 3
            torso_euler,                      # 3
        ], dim=-1)  # Total: 57

        return obs.clamp(-10, 10).nan_to_num()

    def build_arm_obs(self) -> torch.Tensor:
        """Build Stage 5 arm observation (29 dims)"""
        root_pos = self.robot.data.root_pos_w

        arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
        arm_vel = self.robot.data.joint_vel[:, self.arm_indices]

        ee_pos = self._compute_ee_pos() - root_pos  # Body frame
        ee_quat = self._compute_ee_quat()

        pos_err = self.target_pos - ee_pos
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        ori_err = quat_diff_rad(ee_quat, self.target_quat).unsqueeze(-1)

        obs = torch.cat([
            arm_pos,                 # 5
            arm_vel * 0.1,           # 5
            self.target_pos,         # 3
            self.target_quat,        # 4
            ee_pos,                  # 3
            ee_quat,                 # 4
            pos_err,                 # 3
            ori_err,                 # 1
            pos_dist / 0.5,          # 1
        ], dim=-1)  # Total: 29

        return obs

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions for processing"""
        self.actions = actions.clone()

    def _apply_action(self):
        """Apply leg and arm actions"""
        leg_actions = self.actions[:, :12]
        arm_actions = self.actions[:, 12:]

        # Smooth arm actions
        alpha = self.cfg.action_smoothing_alpha
        self.prev_arm_actions = self.smoothed_arm_actions.clone()
        self.smoothed_arm_actions = alpha * arm_actions + (1 - alpha) * self.smoothed_arm_actions

        # Compute joint targets
        target_pos = self.robot.data.default_joint_pos.clone()

        # Legs: Stage 3 style
        target_pos[:, self.leg_indices] = self.default_leg + leg_actions * 0.4

        # Arms: Stage 5 style (incremental)
        cur_arm = self.robot.data.joint_pos[:, self.arm_indices]
        arm_target = torch.clamp(
            cur_arm + self.smoothed_arm_actions * self.cfg.arm_action_scale,
            self.arm_lower, self.arm_upper
        )
        target_pos[:, self.arm_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)

        # Update gait phase
        self.gait_phase = (self.gait_phase + GAIT_FREQUENCY * 0.02) % 1.0

        # Store prev leg actions
        self.prev_leg_actions = leg_actions.clone()

    def _get_observations(self) -> dict:
        """Placeholder - actual obs built in main loop"""
        # Update EE marker
        ee_pos = self._compute_ee_pos()
        ee_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(torch.cat([ee_pos, ee_quat], dim=-1))

        return {"policy": torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        height = self.robot.data.root_pos_w[:, 2]
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(self.robot.data.root_quat_w, gravity)

        fallen = (height < 0.3) | (height > 1.2)
        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

        terminated = fallen | bad_orientation
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
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, env_ids)

        # Reset state
        self.gait_phase[env_ids] = torch.rand(N, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.smoothed_arm_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0

        # Sample target
        self.sample_target_in_workspace(env_ids)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    G1 DECOUPLED DEMO V3 - Standing Balance + Reaching")
    print("    Loco Policy (vx=0) + Arm Policy + Gravity AÇIK")
    print("=" * 70)

    device = "cuda:0"

    # 1. Load policies
    print("\n[1/3] Loading policies...")
    loco_policy = load_loco_policy(args.loco_checkpoint, device)
    arm_policy = load_arm_policy(args.arm_checkpoint, device)

    # 2. Create environment
    print("\n[2/3] Creating environment...")
    env_cfg = DemoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DemoEnv(cfg=env_cfg)

    # 3. Reset and sample target
    print("\n[3/3] Starting demo...")
    obs, _ = env.reset()
    env.sample_target_in_workspace()

    print("\n" + "=" * 70)
    print("    DEMO RUNNING")
    print(f"    Loco: vx=0 (standing balance)")
    print(f"    Arm: reaching to target")
    print(f"    Gravity: ENABLED")
    print("=" * 70 + "\n")

    # Main loop
    total_reaches = 0
    min_distance = float('inf')

    with torch.no_grad():
        for step in range(args.steps):
            # Build observations
            loco_obs = env.build_loco_obs()
            arm_obs = env.build_arm_obs()

            # Get actions from policies
            leg_actions = loco_policy(loco_obs)
            arm_actions = arm_policy(arm_obs)

            # Combine
            actions = torch.cat([leg_actions, arm_actions], dim=-1)

            # Step
            obs, reward, terminated, truncated, _ = env.step(actions)

            # Calculate distance
            root_pos = env.robot.data.root_pos_w
            ee_pos = env._compute_ee_pos() - root_pos
            dist = (ee_pos - env.target_pos).norm(dim=-1).mean().item()

            if dist < min_distance:
                min_distance = dist

            # Check reach
            if dist < env.cfg.reach_threshold:
                total_reaches += 1
                print(f"[Step {step:4d}] 🎯 REACH #{total_reaches}! Distance: {dist:.3f}m")
                env.sample_target_in_workspace()
                min_distance = float('inf')

            # Check fallen
            if terminated.any():
                print(f"[Step {step:4d}] ⚠️ Robot fell! Resetting...")
                env.reset()
                env.sample_target_in_workspace()
                min_distance = float('inf')

            # Log
            if step % 100 == 0:
                height = env.robot.data.root_pos_w[0, 2].item()
                ee_quat = env._compute_ee_quat()
                ori_err = quat_diff_rad(ee_quat, env.target_quat).mean().item()
                ori_err_deg = ori_err * 180 / np.pi

                print(f"[Step {step:4d}] H={height:.2f}m | Dist: {dist:.3f}m (min: {min_distance:.3f}m) | "
                      f"Ori: {ori_err_deg:.1f}° | Reaches: {total_reaches}")

    print("\n" + "=" * 70)
    print(f"    DEMO COMPLETE")
    print(f"    Total reaches: {total_reaches}")
    print(f"    Best distance: {min_distance:.3f}m")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()