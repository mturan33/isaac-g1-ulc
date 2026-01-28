"""
G1 29DoF + Dex1 Stage 6 Loco-Manipulation Environment
======================================================

Based on working g1_arm_dual_orient_env.py (V3.3 SMOOTH + STABLE)

CHANGES FROM ARM-ONLY ENV:
1. Added 12 leg joints for locomotion
2. Added 2 Dex1 gripper joints
3. Observation space: 29 → 85 dimensions
4. Action space: 5 → 21 dimensions
5. Curriculum for loco-manipulation stages
6. USD: G1 basic → G1 29DoF + Dex1

Author: Turan (VLM-RL Project)
Date: January 2026
"""

from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# ==============================================================================
# CONSTANTS
# ==============================================================================

# USD Paths
G1_DEX1_USD_PATH = "C:/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd"
G1_BASIC_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Use Dex1 if available, fallback to basic
G1_USD_PATH = G1_DEX1_USD_PATH

# Joint definitions
G1_LEG_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
]

# Dex1 gripper joints (2 per hand)
G1_RIGHT_GRIPPER_JOINTS = ["right_hand_Joint1_1", "right_hand_Joint2_1"]

# Joint limits
ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

# Default poses
DEFAULT_LEG_POSE = {
    "left_hip_yaw_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
}

DEFAULT_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.5,
    "right_elbow_roll_joint": 0.0,
}

DEFAULT_GRIPPER_POSE = {
    "right_hand_Joint1_1": 0.0,
    "right_hand_Joint2_1": 0.0,
}

EE_OFFSET = 0.02


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz format)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute angular difference between two quaternions in radians."""
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


# ==============================================================================
# CURRICULUM LEVELS
# ==============================================================================
@dataclass
class CurriculumLevel:
    """Configuration for a curriculum level."""
    name: str
    vx_range: Tuple[float, float] = (0.0, 0.0)
    vy_range: Tuple[float, float] = (0.0, 0.0)
    omega_range: Tuple[float, float] = (0.0, 0.0)
    grasp_enabled: bool = False
    reach_threshold: float = 0.10
    min_reaches_to_advance: int = 50
    success_rate_threshold: float = 0.6


CURRICULUM_LEVELS = [
    # Level 0: Standing reach only (like arm env)
    CurriculumLevel(
        name="standing_reach",
        grasp_enabled=False,
        reach_threshold=0.10,
        min_reaches_to_advance=50,
        success_rate_threshold=0.6,
    ),
    # Level 1: Standing + gripper
    CurriculumLevel(
        name="standing_grasp",
        grasp_enabled=True,
        reach_threshold=0.08,
        min_reaches_to_advance=40,
        success_rate_threshold=0.5,
    ),
    # Level 2: Slow walk + reach
    CurriculumLevel(
        name="walk_reach",
        vx_range=(0.0, 0.3),
        grasp_enabled=True,
        reach_threshold=0.10,
        min_reaches_to_advance=30,
        success_rate_threshold=0.4,
    ),
    # Level 3: Full locomotion + grasp
    CurriculumLevel(
        name="full_locomanip",
        vx_range=(-0.2, 0.5),
        vy_range=(-0.15, 0.15),
        omega_range=(-0.3, 0.3),
        grasp_enabled=True,
        reach_threshold=0.12,
        min_reaches_to_advance=25,
        success_rate_threshold=0.35,
    ),
]


# ==============================================================================
# SCENE CONFIGURATION
# ==============================================================================
@configclass
class G1LocoManipSceneCfg(InteractiveSceneCfg):
    """Scene configuration - same structure as working env."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
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
                disable_gravity=False,  # Enable gravity for locomotion
                linear_damping=0.1,
                angular_damping=0.1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),  # Standing height
            joint_pos={
                # Legs
                **DEFAULT_LEG_POSE,
                # Arms
                "right_shoulder_pitch_joint": -0.3,
                "right_elbow_pitch_joint": 0.5,
                "left_shoulder_pitch_joint": -0.3,
                "left_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                stiffness=150.0,
                damping=5.0,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*", "torso.*"],
                stiffness=500.0,
                damping=50.0,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder.*", "left_elbow.*"],
                stiffness=100.0,
                damping=10.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder.*", "right_elbow.*"],
                stiffness=150.0,
                damping=20.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_.*"],
                stiffness=30.0,
                damping=1.0,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.3, 1.0)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.0)),
    )


# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
@configclass
class G1Dex1Stage6EnvCfg(DirectRLEnvCfg):
    """Environment configuration for G1 Loco-Manipulation."""

    decimation = 4
    episode_length_s = 15.0

    # Dimensions
    # Obs: 9 (base) + 24 (legs) + 6 (commands) + 10 (arm) + 4 (gripper) + 9 (target/ee) + 21 (prev_actions) = 83
    # Simplified: base(9) + legs(24) + cmd(6) + arm(10) + grip(4) + target(6) + prev_act(21) = 80
    num_actions = 19  # 12 legs + 5 arm + 2 gripper
    num_observations = 80
    num_states = 0

    action_space = 19
    observation_space = 80
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,
        render_interval=4,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    scene: G1LocoManipSceneCfg = G1LocoManipSceneCfg(num_envs=4096, env_spacing=2.5)

    # Action settings
    action_scale_legs = 0.25
    action_scale_arm = 0.12
    action_scale_gripper = 0.5
    action_smoothing_alpha = 0.2

    # Curriculum
    curriculum_level = 0
    auto_curriculum = True
    min_steps_per_stage = 200

    # Rewards - Locomotion
    reward_lin_vel_tracking = 1.5
    reward_ang_vel_tracking = 0.8
    reward_base_height = 0.5
    reward_orientation = 1.0
    reward_feet_air_time = 0.5

    # Rewards - Arm reaching (from working env)
    use_potential_shaping = True
    potential_gamma = 0.99
    potential_sigma = 0.15
    potential_scale = 3.0

    reward_pos_tanh_std = 0.12
    reward_pos_tanh_weight = 1.5
    reward_reaching = 100.0

    # Proximity zones
    proximity_zone1_dist = 0.15
    proximity_zone1_bonus = 1.0
    proximity_zone2_dist = 0.10
    proximity_zone2_bonus = 2.0
    proximity_zone3_dist = 0.05
    proximity_zone3_bonus = 5.0

    # Grasp rewards
    reward_grasp_contact = 20.0
    reward_grasp_hold = 50.0

    # Regularization
    reward_action_rate = -0.01
    reward_joint_vel = -0.002
    reward_joint_acc = -0.005

    # Workspace
    shoulder_center_offset = [0.0, -0.2, 0.4]
    workspace_inner_radius = 0.18
    workspace_outer_radius = 0.45
    initial_spawn_radius = 0.22
    max_spawn_radius = 0.42

    # Base height
    target_base_height = 0.72

    # Episode settings
    terminate_on_fall = True
    fall_height_threshold = 0.4
    max_reaches_per_episode = 5


# ==============================================================================
# ENVIRONMENT CLASS
# ==============================================================================
class G1Dex1Stage6Env(DirectRLEnv):
    """G1 Loco-Manipulation Environment with Dex1 gripper."""

    cfg: G1Dex1Stage6EnvCfg

    def __init__(self, cfg: G1Dex1Stage6EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Scene entities
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        # Get joint indices
        joint_names = list(self.robot.data.joint_names)
        body_names = list(self.robot.data.body_names)

        # Leg indices
        self.leg_indices = []
        for jn in G1_LEG_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.leg_indices.append(i)
                    break
        self.leg_indices = torch.tensor(self.leg_indices, device=self.device, dtype=torch.long)

        # Arm indices
        self.arm_indices = []
        for jn in G1_RIGHT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.arm_indices.append(i)
                    break
        self.arm_indices = torch.tensor(self.arm_indices, device=self.device, dtype=torch.long)

        # Gripper indices
        self.gripper_indices = []
        for jn in G1_RIGHT_GRIPPER_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.gripper_indices.append(i)
                    break
        self.gripper_indices = torch.tensor(self.gripper_indices, device=self.device, dtype=torch.long)

        # Palm/EE index
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and ("palm" in name.lower() or "wrist" in name.lower()):
                self.palm_idx = i
                break
        if self.palm_idx is None:
            self.palm_idx = len(body_names) - 1  # Fallback

        # Joint limits for arm
        self.arm_joint_lower = torch.zeros(5, device=self.device)
        self.arm_joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS):
            if jn in ARM_JOINT_LIMITS:
                self.arm_joint_lower[i], self.arm_joint_upper[i] = ARM_JOINT_LIMITS[jn]
            else:
                self.arm_joint_lower[i], self.arm_joint_upper[i] = -3.14, 3.14

        # Buffers
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.smoothed_actions = torch.zeros_like(self.actions)

        self.commands = torch.zeros(self.num_envs, 6, device=self.device)  # vx, vy, omega, pitch, height, phase
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # For potential-based shaping
        self.prev_potential = torch.zeros(self.num_envs, device=self.device)
        self.prev_joint_vel = torch.zeros(self.num_envs, 5, device=self.device)

        # Tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.total_reaches = 0
        self.total_attempts = 0

        # Curriculum
        self.curriculum_stage = cfg.curriculum_level
        self.stage_reaches = 0
        self.stage_attempts = 0
        self.stage_step_count = 0
        self.current_spawn_radius = cfg.initial_spawn_radius

        # Shoulder offset
        self.shoulder_center = torch.tensor(
            self.cfg.shoulder_center_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print("\n" + "=" * 70)
        print("G1 LOCO-MANIPULATION ENVIRONMENT - STAGE 6")
        print("=" * 70)
        print(f"  Leg joints: {len(self.leg_indices)} (expected 12)")
        print(f"  Arm joints: {len(self.arm_indices)} (expected 5)")
        print(f"  Gripper joints: {len(self.gripper_indices)} (expected 2)")
        print(f"  Palm idx: {self.palm_idx}")
        print(f"  Actions: {self.cfg.num_actions}")
        print(f"  Observations: {self.cfg.num_observations}")
        print("-" * 70)
        print(f"  Curriculum: Level {self.curriculum_stage} ({self.current_curriculum.name})")
        print(f"  Grasp enabled: {self.current_curriculum.grasp_enabled}")
        print("=" * 70 + "\n")

    @property
    def current_curriculum(self) -> CurriculumLevel:
        return CURRICULUM_LEVELS[min(self.curriculum_stage, len(CURRICULUM_LEVELS) - 1)]

    def _setup_scene(self):
        """Scene setup - entities accessed via self.scene[]."""
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    def _compute_ee_pos(self) -> torch.Tensor:
        """Compute end-effector position."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_potential(self, distance: torch.Tensor) -> torch.Tensor:
        """Exponential potential function."""
        return torch.exp(-distance / self.cfg.potential_sigma)

    def _sample_target_in_workspace(self, env_ids: torch.Tensor):
        """Sample target positions."""
        num = len(env_ids)
        root_pos = self.robot.data.root_pos_w[env_ids]
        shoulder_rel = self.shoulder_center[env_ids]

        # Random direction (prefer forward)
        direction = torch.randn((num, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction[:, 0] = -torch.abs(direction[:, 0])  # Prefer forward (-X)

        # Random distance
        inner = self.cfg.workspace_inner_radius
        outer = min(self.current_spawn_radius, self.cfg.workspace_outer_radius)
        distance = inner + torch.rand((num, 1), device=self.device) * (outer - inner)

        targets = shoulder_rel + direction * distance
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.55)

        self.target_pos[env_ids] = targets

        # Update visual
        target_world = root_pos + targets
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num, -1)
        pose = torch.cat([target_world, identity_quat], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Initialize potential
        ee_pos = self._compute_ee_pos()[env_ids] - root_pos
        initial_dist = (ee_pos - targets).norm(dim=-1)
        self.prev_potential[env_ids] = self._compute_potential(initial_dist)

        self.stage_attempts += len(env_ids)
        self.total_attempts += len(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands for curriculum level."""
        level = self.current_curriculum
        num = len(env_ids)

        vx = torch.empty(num, device=self.device).uniform_(*level.vx_range)
        vy = torch.empty(num, device=self.device).uniform_(*level.vy_range)
        omega = torch.empty(num, device=self.device).uniform_(*level.omega_range)

        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        self.commands[env_ids, 2] = omega
        self.commands[env_ids, 3] = 0.0  # pitch
        self.commands[env_ids, 4] = self.cfg.target_base_height
        self.commands[env_ids, 5] = 0.0  # phase

    def _get_observations(self) -> dict:
        """Compute observations."""
        obs_list = []

        # Base state (9)
        obs_list.append(self.robot.data.root_lin_vel_b)  # 3
        obs_list.append(self.robot.data.root_ang_vel_b)  # 3
        obs_list.append(self.robot.data.projected_gravity_b)  # 3

        # Leg joints (24)
        if len(self.leg_indices) > 0:
            leg_pos = self.robot.data.joint_pos[:, self.leg_indices]
            leg_vel = self.robot.data.joint_vel[:, self.leg_indices]
            obs_list.append(leg_pos)  # 12
            obs_list.append(leg_vel * 0.1)  # 12
        else:
            obs_list.append(torch.zeros(self.num_envs, 24, device=self.device))

        # Commands (6)
        obs_list.append(self.commands)  # 6

        # Arm joints (10)
        if len(self.arm_indices) > 0:
            arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
            arm_vel = self.robot.data.joint_vel[:, self.arm_indices]
            obs_list.append(arm_pos)  # 5
            obs_list.append(arm_vel * 0.1)  # 5
        else:
            obs_list.append(torch.zeros(self.num_envs, 10, device=self.device))

        # Gripper joints (4)
        if len(self.gripper_indices) > 0:
            grip_pos = self.robot.data.joint_pos[:, self.gripper_indices]
            grip_vel = self.robot.data.joint_vel[:, self.gripper_indices]
            obs_list.append(grip_pos)  # 2
            obs_list.append(grip_vel * 0.1)  # 2
        else:
            obs_list.append(torch.zeros(self.num_envs, 4, device=self.device))

        # Target info (6)
        root_pos = self.robot.data.root_pos_w
        ee_pos = self._compute_ee_pos() - root_pos
        pos_err = self.target_pos - ee_pos
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        obs_list.append(pos_err)  # 3
        obs_list.append(pos_dist / 0.5)  # 1
        obs_list.append(self.target_pos[:, :2])  # 2 (XY only)

        # Previous actions (19)
        obs_list.append(self.prev_actions)

        # Update EE marker
        ee_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(
            torch.cat([self._compute_ee_pos(), ee_quat], dim=-1)
        )

        obs = torch.cat(obs_list, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        rewards = torch.zeros(self.num_envs, device=self.device)

        # === LOCOMOTION REWARDS ===
        # Velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rewards += torch.exp(-lin_vel_error / 0.25) * self.cfg.reward_lin_vel_tracking

        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        rewards += torch.exp(-ang_vel_error / 0.25) * self.cfg.reward_ang_vel_tracking

        # Base height
        height_error = torch.square(
            self.robot.data.root_pos_w[:, 2] - self.cfg.target_base_height
        )
        rewards += torch.exp(-height_error / 0.1) * self.cfg.reward_base_height

        # Upright orientation
        proj_grav = self.robot.data.projected_gravity_b
        orientation_error = torch.sum(torch.square(proj_grav[:, :2]), dim=1)
        rewards += torch.exp(-orientation_error / 0.5) * self.cfg.reward_orientation

        # === ARM REACHING REWARDS ===
        root_pos = self.robot.data.root_pos_w
        ee_pos = self._compute_ee_pos() - root_pos
        pos_dist = (ee_pos - self.target_pos).norm(dim=-1)

        # Potential-based shaping
        if self.cfg.use_potential_shaping:
            current_potential = self._compute_potential(pos_dist)
            potential_reward = self.cfg.potential_scale * (
                self.cfg.potential_gamma * current_potential - self.prev_potential
            )
            self.prev_potential = current_potential
            rewards += potential_reward

        # Tanh distance reward
        pos_reward = 1.0 - torch.tanh(pos_dist / self.cfg.reward_pos_tanh_std)
        rewards += self.cfg.reward_pos_tanh_weight * pos_reward

        # Proximity bonuses
        in_zone1 = pos_dist < self.cfg.proximity_zone1_dist
        in_zone2 = pos_dist < self.cfg.proximity_zone2_dist
        in_zone3 = pos_dist < self.cfg.proximity_zone3_dist
        rewards += in_zone1.float() * self.cfg.proximity_zone1_bonus
        rewards += in_zone2.float() * self.cfg.proximity_zone2_bonus
        rewards += in_zone3.float() * self.cfg.proximity_zone3_bonus

        # Reach detection
        reached = pos_dist < self.current_curriculum.reach_threshold
        rewards += reached.float() * self.cfg.reward_reaching

        reached_ids = torch.where(reached)[0]
        if len(reached_ids) > 0:
            self.reach_count[reached_ids] += 1
            self.episode_reach_count[reached_ids] += 1
            self.stage_reaches += len(reached_ids)
            self.total_reaches += len(reached_ids)
            self._sample_target_in_workspace(reached_ids)

        # === REGULARIZATION ===
        # Action rate
        action_diff = self.smoothed_actions - self.prev_actions
        rewards += action_diff.norm(dim=-1) * self.cfg.reward_action_rate

        # Joint velocity (arm only)
        if len(self.arm_indices) > 0:
            arm_vel = self.robot.data.joint_vel[:, self.arm_indices]
            rewards += arm_vel.norm(dim=-1) * self.cfg.reward_joint_vel

            # Acceleration
            joint_acc = (arm_vel - self.prev_joint_vel).norm(dim=-1)
            self.prev_joint_vel = arm_vel.clone()
            rewards += joint_acc * self.cfg.reward_joint_acc

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Fall detection
        fallen = self.robot.data.root_pos_w[:, 2] < self.cfg.fall_height_threshold

        # Bad orientation
        proj_grav = self.robot.data.projected_gravity_b
        bad_orientation = proj_grav[:, 2] > -0.5

        # Max reaches
        max_reached = self.episode_reach_count >= self.cfg.max_reaches_per_episode

        terminated = fallen | bad_orientation
        if self.cfg.terminate_on_fall:
            terminated = terminated | max_reached

        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # Reset robot pose
        default_state = self.robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        # Reset joints
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(jp)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        # Sample targets and commands
        self._sample_target_in_workspace(env_ids)
        self._sample_commands(env_ids)

        # Reset buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.smoothed_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.episode_reach_count[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions."""
        self.prev_actions = self.smoothed_actions.clone()
        self.actions = actions.clone()

    def _apply_action(self):
        """Apply actions to robot."""
        # Smooth actions
        alpha = self.cfg.action_smoothing_alpha
        self.smoothed_actions = alpha * self.actions + (1 - alpha) * self.smoothed_actions

        # Get current joint positions
        joint_targets = self.robot.data.joint_pos.clone()

        # Apply leg actions [0:12]
        if len(self.leg_indices) > 0:
            leg_actions = self.smoothed_actions[:, :12] * self.cfg.action_scale_legs
            for i, idx in enumerate(self.leg_indices):
                joint_targets[:, idx] += leg_actions[:, i]

        # Apply arm actions [12:17]
        if len(self.arm_indices) > 0:
            arm_actions = self.smoothed_actions[:, 12:17] * self.cfg.action_scale_arm
            cur_arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
            new_arm_pos = torch.clamp(
                cur_arm_pos + arm_actions,
                self.arm_joint_lower, self.arm_joint_upper
            )
            for i, idx in enumerate(self.arm_indices):
                joint_targets[:, idx] = new_arm_pos[:, i]

        # Apply gripper actions [17:19] - only if grasp enabled
        if self.current_curriculum.grasp_enabled and len(self.gripper_indices) > 0:
            grip_actions = self.smoothed_actions[:, 17:19] * self.cfg.action_scale_gripper
            for i, idx in enumerate(self.gripper_indices):
                joint_targets[:, idx] = torch.clamp(grip_actions[:, i] + 0.5, 0.0, 1.0)

        self.robot.set_joint_position_target(joint_targets)

    def update_curriculum(self, iteration: int) -> float:
        """Update curriculum based on success rate."""
        self.stage_step_count += 1

        if self.stage_attempts > 0:
            current_success_rate = self.stage_reaches / self.stage_attempts
        else:
            current_success_rate = 0.0

        level = self.current_curriculum

        should_advance = (
            self.cfg.auto_curriculum and
            current_success_rate >= level.success_rate_threshold and
            self.stage_reaches >= level.min_reaches_to_advance and
            self.stage_step_count >= self.cfg.min_steps_per_stage and
            self.curriculum_stage < len(CURRICULUM_LEVELS) - 1
        )

        if should_advance:
            self.curriculum_stage += 1

            # Update spawn radius
            radius_increment = (self.cfg.max_spawn_radius - self.cfg.initial_spawn_radius) / len(CURRICULUM_LEVELS)
            self.current_spawn_radius = self.cfg.initial_spawn_radius + self.curriculum_stage * radius_increment

            print(f"\n{'='*60}")
            print(f"🎯 CURRICULUM ADVANCED TO STAGE {self.curriculum_stage + 1}/{len(CURRICULUM_LEVELS)}")
            print(f"   Level: {self.current_curriculum.name}")
            print(f"   Success rate: {current_success_rate*100:.1f}%")
            print(f"   Grasp enabled: {self.current_curriculum.grasp_enabled}")
            print(f"{'='*60}\n")

            self.stage_reaches = 0
            self.stage_attempts = 0
            self.stage_step_count = 0

        return current_success_rate


# ==============================================================================
# ALIASES
# ==============================================================================
G1LocoManipEnv = G1Dex1Stage6Env
G1LocoManipEnvCfg = G1Dex1Stage6EnvCfg