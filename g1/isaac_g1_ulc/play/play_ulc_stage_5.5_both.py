#!/usr/bin/env python3
"""
Stage 5.5 Dual Policy Play Script - V2 (FIXED COORDINATES)
===========================================================

Düzeltmeler:
1. Hedef koordinatları: +X (öne), -Y (sağa)
2. Body frame dönüşümü eklendi
3. Azimuth aralığı daraltıldı
4. Ground collision koruması

KULLANIM:
./isaaclab.bat -p .../play/play_ulc_stage_5_5_both_v2.py \
    --loco_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt \
    --arm_checkpoint logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt \
    --num_envs 4 --vx 0.0

Author: Turan
Date: January 2026
Version: 2.0 - Fixed coordinate system
"""

from __future__ import annotations

import argparse

# ==============================================================================
# ARGUMENT PARSING (MUST BE BEFORE ISAAC IMPORTS)
# ==============================================================================

parser = argparse.ArgumentParser(description="Stage 5.5 Dual Policy Play V2")
parser.add_argument("--loco_checkpoint", type=str, required=True,
                    help="Path to Stage 3 locomotion checkpoint")
parser.add_argument("--arm_checkpoint", type=str, required=True,
                    help="Path to Stage 5 arm checkpoint")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity")
parser.add_argument("--pitch", type=float, default=0.0, help="Torso pitch (rad)")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
# NOW WE CAN IMPORT ISAAC MODULES
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply_inverse, quat_apply

# ==============================================================================
# CONSTANTS
# ==============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

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

ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

DEFAULT_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.5,
    "right_elbow_roll_joint": 0.0,
}

EE_OFFSET = 0.02


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

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
    """Angular difference between quaternions in radians."""
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


# ==============================================================================
# SCENE CONFIG
# ==============================================================================

@configclass
class DualPlaySceneCfg(InteractiveSceneCfg):
    """Scene configuration."""

    # Stage 3 ile aynı terrain config - BU ÇALIŞIYOR
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.90),  # Daha da yüksek başlangıç
            joint_pos={
                "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                "right_shoulder_pitch_joint": -0.3,
                "right_elbow_pitch_joint": 0.5,
                "left_shoulder_pitch_joint": -0.3,
                "left_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=200.0,
                damping=20.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=150.0,
                damping=20.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=150.0,
                damping=15.0,
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
                emissive_color=(0.0, 0.5, 0.0),
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
                emissive_color=(0.5, 0.25, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


# ==============================================================================
# ENVIRONMENT CONFIG
# ==============================================================================

@configclass
class DualPlayEnvCfg(DirectRLEnvCfg):
    """Environment configuration."""

    decimation = 4
    episode_length_s = 20.0

    # Gymnasium spaces (required in Isaac Lab 2.3+)
    action_space = 17
    observation_space = 86
    state_space = 0

    sim = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.5,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.01,
            gpu_max_rigid_contact_count=512 * 1024,
            gpu_max_rigid_patch_count=80 * 1024,
        ),
    )

    scene: DualPlaySceneCfg = DualPlaySceneCfg(num_envs=1, env_spacing=2.5)

    # Workspace - FIXED: sağ omuz etrafında yarım küre
    workspace_radius = 0.40
    workspace_inner_radius = 0.15
    # Shoulder offset: sağ omuz robotun sağında (-Y)
    shoulder_offset = [0.0, -0.174, 0.259]

    # Commands
    height_target = 0.72
    gait_frequency = 1.5

    # Actions
    leg_action_scale = 0.4
    arm_action_scale = 0.12


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class DualPlayEnv(DirectRLEnv):
    """Stage 5.5 Dual Policy Environment."""

    cfg: DualPlayEnvCfg

    def __init__(self, cfg: DualPlayEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices
        all_names = self.robot.joint_names
        self.leg_indices = torch.tensor(
            [all_names.index(n) for n in LEG_JOINT_NAMES],
            device=self.device
        )
        self.arm_indices = torch.tensor(
            [all_names.index(n) for n in ARM_JOINT_NAMES],
            device=self.device
        )

        # Palm for EE
        self.palm_idx = self.robot.body_names.index("right_palm_link")

        # Arm limits
        arm_lims = torch.tensor(
            [[ARM_JOINT_LIMITS[n][0], ARM_JOINT_LIMITS[n][1]] for n in ARM_JOINT_NAMES],
            device=self.device
        )
        self.arm_lower = arm_lims[:, 0]
        self.arm_upper = arm_lims[:, 1]

        # Default positions
        self.default_leg = self.robot.data.default_joint_pos[:, self.leg_indices].clone()
        self.default_arm = self.robot.data.default_joint_pos[:, self.arm_indices].clone()

        # State buffers
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.full(
            (self.num_envs,), self.cfg.height_target, device=self.device
        )

        # Target - BODY FRAME coordinates
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.tensor(
            [[0.707, 0.707, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1).clone()

        self.shoulder_offset = torch.tensor(
            self.cfg.shoulder_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        self.local_forward = torch.tensor(
            [[1.0, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1)

        self.total_reaches = 0
        self.reach_threshold = 0.10

        print("\n" + "=" * 60)
        print("DUAL PLAY ENVIRONMENT INITIALIZED - V2")
        print("=" * 60)
        print(f"  Leg joints: {len(self.leg_indices)}")
        print(f"  Arm joints: {len(self.arm_indices)}")
        print(f"  Palm idx: {self.palm_idx}")
        print(f"  Workspace: {self.cfg.workspace_inner_radius:.2f}m - {self.cfg.workspace_radius:.2f}m")
        print(f"  Shoulder offset: {self.cfg.shoulder_offset}")
        print("=" * 60 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        # Clone environments properly
        self.scene.clone_environments(copy_from_source=False)

        # CRITICAL: Filter collisions - Stage 3 pattern
        self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])

    # =========================================================================
    # DUAL OBSERVATIONS
    # =========================================================================

    def get_loco_obs(self) -> torch.Tensor:
        """Stage 3 locomotion observation - 57 dim."""
        robot = self.robot
        quat = robot.data.root_quat_w

        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity)

        leg_pos = robot.data.joint_pos[:, self.leg_indices]
        leg_vel = robot.data.joint_vel[:, self.leg_indices]

        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.gait_phase),
            torch.cos(2 * np.pi * self.gait_phase),
        ], dim=-1)

        torso_euler = quat_to_euler_xyz(quat)

        obs = torch.cat([
            lin_vel_b,  # 3
            ang_vel_b,  # 3
            proj_gravity,  # 3
            leg_pos,  # 12
            leg_vel,  # 12
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,  # 3
            gait_phase,  # 2
            self.prev_leg_actions,  # 12
            self.torso_cmd,  # 3
            torso_euler,  # 3
        ], dim=-1)  # Total: 57

        return obs.clamp(-10, 10).nan_to_num()

    def get_arm_obs(self) -> torch.Tensor:
        """Stage 5 arm observation - 29 dim (BODY FRAME)."""
        robot = self.robot
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        arm_pos = robot.data.joint_pos[:, self.arm_indices]
        arm_vel = robot.data.joint_vel[:, self.arm_indices]

        # EE position in BODY FRAME
        ee_pos_world = self._compute_ee_pos()
        ee_pos_rel = ee_pos_world - root_pos
        ee_pos_body = quat_apply_inverse(root_quat, ee_pos_rel)

        # EE orientation in BODY FRAME
        ee_quat_world = self._compute_ee_quat()
        # For simplicity, just use world quat (could rotate to body frame too)
        ee_quat = ee_quat_world

        # Target is already in BODY FRAME
        target_body = self.target_pos_body
        target_quat = self.target_quat

        # Position error in BODY FRAME
        pos_err = target_body - ee_pos_body
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        ori_err = quat_diff_rad(ee_quat, target_quat).unsqueeze(-1)

        obs = torch.cat([
            arm_pos,  # 5
            arm_vel * 0.1,  # 5
            target_body,  # 3  (body frame)
            target_quat,  # 4
            ee_pos_body,  # 3  (body frame)
            ee_quat,  # 4
            pos_err,  # 3  (body frame)
            ori_err,  # 1
            pos_dist / 0.5,  # 1
        ], dim=-1)  # Total: 29

        return obs.clamp(-10, 10).nan_to_num()

    def _compute_ee_pos(self) -> torch.Tensor:
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_ee_quat(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.palm_idx]

    # =========================================================================
    # STANDARD INTERFACE
    # =========================================================================

    def _get_observations(self) -> dict:
        ee_pos = self._compute_ee_pos()
        ee_quat = self._compute_ee_quat()
        self.ee_marker.write_root_pose_to_sim(torch.cat([ee_pos, ee_quat], dim=-1))

        loco = self.get_loco_obs()
        arm = self.get_arm_obs()
        return {"policy": torch.cat([loco, arm], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # Compute distance in BODY FRAME
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        ee_pos_world = self._compute_ee_pos()
        ee_pos_rel = ee_pos_world - root_pos
        ee_pos_body = quat_apply_inverse(root_quat, ee_pos_rel)

        dist = (ee_pos_body - self.target_pos_body).norm(dim=-1)

        reached = dist < self.reach_threshold
        reached_ids = torch.where(reached)[0]

        if len(reached_ids) > 0:
            self.total_reaches += len(reached_ids)
            self._sample_target(reached_ids)

        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        height = self.robot.data.root_pos_w[:, 2]
        # Daha erken düşme tespiti (0.5m altında = düşmüş)
        fallen = (height < 0.5) | (height > 1.3)

        quat = self.robot.data.root_quat_w
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_grav = quat_apply_inverse(quat, gravity)
        # Daha hassas tilt kontrolü
        bad_orientation = proj_grav[:, :2].abs().max(dim=-1)[0] > 0.5

        terminated = fallen | bad_orientation
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Reset robot with higher starting position for stability
        default_pos = torch.tensor([[0.0, 0.0, 0.90]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        default_joint = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_joint, torch.zeros_like(default_joint), None, env_ids)

        self.gait_phase[env_ids] = torch.rand(n, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0

        self._sample_target(env_ids)

    def _sample_target(self, env_ids: torch.Tensor):
        """Sample target in BODY FRAME - front right of robot."""
        n = len(env_ids)

        # Azimuth: -30° to +60° (mostly forward and to the right)
        # Positive azimuth = towards robot's right (-Y direction)
        azimuth = torch.empty(n, device=self.device).uniform_(-0.5, 1.0)  # ~-30° to +60°

        radius = torch.empty(n, device=self.device).uniform_(
            self.cfg.workspace_inner_radius, self.cfg.workspace_radius
        )

        # Height relative to shoulder
        height = torch.empty(n, device=self.device).uniform_(-0.10, 0.20)

        # FIXED: +X (forward), -Y (right side for right arm)
        x = radius * torch.cos(azimuth)  # POSITIVE X = forward
        y = -radius * torch.sin(azimuth)  # NEGATIVE Y = right side
        z = height

        # Store in BODY FRAME (relative to shoulder)
        self.target_pos_body[env_ids, 0] = x + self.shoulder_offset[0, 0]
        self.target_pos_body[env_ids, 1] = y + self.shoulder_offset[0, 1]
        self.target_pos_body[env_ids, 2] = z + self.shoulder_offset[0, 2]

        # Update visual marker in WORLD FRAME
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]

        # Transform body frame target to world frame for visualization
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body[env_ids])

        identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

        self.target_obj.write_root_pose_to_sim(
            torch.cat([target_world, identity_quat], dim=-1), env_ids
        )

    def _update_target_visuals(self):
        """Update target marker position in world frame."""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Transform body frame target to world frame
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)

        identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)

        self.target_obj.write_root_pose_to_sim(
            torch.cat([target_world, identity_quat], dim=-1)
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        leg_actions = self.actions[:, :12]
        arm_actions = self.actions[:, 12:17]

        joint_targets = self.robot.data.default_joint_pos.clone()
        joint_targets[:, self.leg_indices] = self.default_leg + leg_actions * self.cfg.leg_action_scale

        cur_arm = self.robot.data.joint_pos[:, self.arm_indices]
        new_arm = torch.clamp(
            cur_arm + arm_actions * self.cfg.arm_action_scale,
            self.arm_lower, self.arm_upper
        )
        joint_targets[:, self.arm_indices] = new_arm

        self.robot.set_joint_position_target(joint_targets)

        self.gait_phase = (self.gait_phase + self.cfg.gait_frequency * 0.02) % 1.0
        self.prev_leg_actions = leg_actions.clone()
        self.prev_arm_actions = arm_actions.clone()

        # Update target visual to follow robot
        self._update_target_visuals()

    def set_velocity_command(self, vx: float, vy: float = 0.0, vyaw: float = 0.0):
        self.vel_cmd[:, 0] = vx
        self.vel_cmd[:, 1] = vy
        self.vel_cmd[:, 2] = vyaw

    def set_torso_command(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        self.torso_cmd[:, 0] = roll
        self.torso_cmd[:, 1] = pitch
        self.torso_cmd[:, 2] = yaw


# ==============================================================================
# POLICY NETWORKS
# ==============================================================================

class LocoActorCritic(nn.Module):
    """Stage 3 network: 57 obs → 12 actions"""

    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
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

    def act(self, x, deterministic=True):
        mean = self.actor(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


class ArmActor(nn.Module):
    """Stage 5 network: 29 obs → 5 actions"""

    def __init__(self, num_obs=29, num_act=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, num_act),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = "cuda:0"

    print("\n" + "=" * 70)
    print("STAGE 5.5 DUAL POLICY PLAY - V2 (FIXED COORDINATES)")
    print("=" * 70)
    print(f"Loco checkpoint: {args.loco_checkpoint}")
    print(f"Arm checkpoint: {args.arm_checkpoint}")
    print(f"Commands: vx={args.vx}, pitch={np.rad2deg(args.pitch):.1f}°")
    print("=" * 70 + "\n")

    # =========================================================================
    # LOAD LOCO POLICY
    # =========================================================================
    print("[1/4] Loading LOCO policy (Stage 3)...")

    loco_ckpt = torch.load(args.loco_checkpoint, map_location=device, weights_only=False)
    loco_net = LocoActorCritic(57, 12).to(device)

    if "actor_critic" in loco_ckpt:
        loco_net.load_state_dict(loco_ckpt["actor_critic"])
    else:
        loco_net.load_state_dict(loco_ckpt)

    loco_net.eval()
    print("      ✓ LOCO policy loaded!")

    # =========================================================================
    # LOAD ARM POLICY
    # =========================================================================
    print("[2/4] Loading ARM policy (Stage 5)...")

    arm_ckpt = torch.load(args.arm_checkpoint, map_location=device, weights_only=False)
    arm_net = ArmActor(29, 5).to(device)

    if "model_state_dict" in arm_ckpt:
        state_dict = arm_ckpt["model_state_dict"]
        actor_state = {}
        for key, value in state_dict.items():
            if "actor" in key:
                new_key = key.replace("actor.", "")
                actor_state[new_key] = value

        if actor_state:
            arm_net.net.load_state_dict(actor_state)
    else:
        arm_net.load_state_dict(arm_ckpt)

    arm_net.eval()
    print("      ✓ ARM policy loaded!")

    # =========================================================================
    # CREATE ENVIRONMENT
    # =========================================================================
    print("[3/4] Creating environment...")

    cfg = DualPlayEnvCfg()
    cfg.scene.num_envs = args.num_envs

    env = DualPlayEnv(cfg)

    env.set_velocity_command(vx=args.vx)
    env.set_torso_command(pitch=args.pitch)

    print("      ✓ Environment created!")

    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    print("[4/4] Running simulation...")
    print("-" * 70)

    obs, _ = env.reset()

    prev_reaches = 0

    with torch.no_grad():
        for step in range(args.steps):
            # Get separate observations
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            # Get actions
            leg_actions = loco_net.act(loco_obs, deterministic=True)
            arm_actions = arm_net(arm_obs)

            # Combine
            combined_actions = torch.cat([leg_actions, arm_actions], dim=-1)

            # Step
            obs, reward, terminated, truncated, info = env.step(combined_actions)

            # Check reaches
            if env.total_reaches > prev_reaches:
                print(f"[Step {step:5d}] 🎯 REACH #{env.total_reaches}!")
                prev_reaches = env.total_reaches

            # Progress
            if step > 0 and step % 200 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()

                # Distance in body frame
                root_pos = env.robot.data.root_pos_w
                root_quat = env.robot.data.root_quat_w
                ee_pos_world = env._compute_ee_pos()
                ee_pos_rel = ee_pos_world - root_pos
                ee_pos_body = quat_apply_inverse(root_quat, ee_pos_rel)
                dist = (ee_pos_body - env.target_pos_body).norm(dim=-1).mean().item()

                print(f"[Step {step:5d}] H={height:.3f}m | Vx={vx:.2f}m/s | "
                      f"EE dist={dist:.3f}m | Reaches={env.total_reaches}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DUAL POLICY PLAY COMPLETE - V2")
    print("=" * 70)
    print(f"  Total reaches: {env.total_reaches}")
    print(f"  Steps: {args.steps}")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()