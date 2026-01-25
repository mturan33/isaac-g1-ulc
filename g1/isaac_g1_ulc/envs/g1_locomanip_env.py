"""
G1 29DoF + Dex1 Stage 6 Loco-Manipulation Environment
======================================================

Based on working g1_arm_dual_orient_env.py structure.
Combines locomotion with arm reaching and grasping using Dex1 gripper.

Joint Structure:
- 12 leg joints (6 per leg)
- 3 waist joints (locked)
- 14 arm joints (7 per arm)
- 4 gripper joints (2 per hand - Dex1)

We control: 12 legs + 7 right arm + 2 right gripper = 21 actions

Author: Turan (VLM-RL Project)
Date: January 2026
"""

from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


# ==============================================================================
# CONSTANTS & JOINT DEFINITIONS
# ==============================================================================

# USD Path - Unitree G1 29DoF with Dex1 gripper
G1_DEX1_USD_PATH = "C:/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd"

# Alternative: Isaac Sim cloud path (backup)
G1_ISAAC_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Joint names
G1_LEG_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

G1_WAIST_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

G1_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]

# Dex1 gripper joints
G1_RIGHT_GRIPPER_JOINTS = ["right_hand_Joint1_1", "right_hand_Joint2_1"]
G1_LEFT_GRIPPER_JOINTS = ["left_hand_Joint1_1", "left_hand_Joint2_1"]

# End effector link
G1_RIGHT_EE_LINK = "right_wrist_yaw_link"

# Default joint positions
G1_DEFAULT_JOINT_POS = {
    # Legs - standing pose
    "left_hip_yaw_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    # Waist - locked
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    # Arms - relaxed
    "left_shoulder_pitch_joint": 0.3, "left_shoulder_roll_joint": 0.2, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3, "right_shoulder_roll_joint": -0.2, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
    # Grippers - open
    "left_hand_Joint1_1": 0.0, "left_hand_Joint2_1": 0.0,
    "right_hand_Joint1_1": 0.0, "right_hand_Joint2_1": 0.0,
}


# ==============================================================================
# CURRICULUM CONFIGURATION
# ==============================================================================
@dataclass
class CurriculumLevel:
    """Configuration for a curriculum level."""
    name: str
    vx_range: Tuple[float, float] = (0.0, 0.0)
    vy_range: Tuple[float, float] = (0.0, 0.0)
    omega_range: Tuple[float, float] = (0.0, 0.0)
    pitch_range: Tuple[float, float] = (0.0, 0.0)
    reach_radius_range: Tuple[float, float] = (0.20, 0.30)
    reach_height_range: Tuple[float, float] = (-0.10, 0.15)
    reach_azimuth_range: Tuple[float, float] = (-0.5, 0.5)
    grasp_enabled: bool = False
    reach_threshold: float = 0.05
    min_reach_success: int = 50
    success_rate_threshold: float = 0.7


CURRICULUM_LEVELS = [
    CurriculumLevel(name="standing_reach", grasp_enabled=False,
                    reach_radius_range=(0.20, 0.28), reach_height_range=(-0.05, 0.15)),
    CurriculumLevel(name="standing_grasp", grasp_enabled=True,
                    reach_radius_range=(0.20, 0.30), reach_height_range=(-0.10, 0.18)),
    CurriculumLevel(name="walk_grasp", grasp_enabled=True, vx_range=(0.0, 0.4),
                    reach_radius_range=(0.20, 0.35), reach_height_range=(-0.20, 0.22)),
    CurriculumLevel(name="squat_grasp", grasp_enabled=True, vx_range=(0.0, 0.3), pitch_range=(-0.25, 0.0),
                    reach_radius_range=(0.20, 0.40), reach_height_range=(-0.40, 0.20)),
    CurriculumLevel(name="full_locomanip", grasp_enabled=True, vx_range=(-0.2, 0.6), vy_range=(-0.2, 0.2),
                    reach_radius_range=(0.20, 0.45), reach_height_range=(-0.50, 0.30)),
]


# ==============================================================================
# SCENE CONFIGURATION (KEY DIFFERENCE FROM BEFORE)
# ==============================================================================
@configclass
class G1LocoManipSceneCfg(InteractiveSceneCfg):
    """Scene configuration with all entities defined here."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # Robot - G1 29DoF with Dex1
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_DEX1_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos=G1_DEFAULT_JOINT_POS,
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*"],
                stiffness=150.0,
                damping=5.0,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*"],
                stiffness=100.0,
                damping=5.0,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                stiffness=10000.0,
                damping=10000.0,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder.*", "left_elbow.*", "left_wrist.*"],
                stiffness=100.0,
                damping=10.0,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder.*", "right_elbow.*", "right_wrist.*"],
                stiffness=80.0,
                damping=8.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_.*"],
                stiffness=30.0,
                damping=1.0,
            ),
        },
    )

    # Target object for reaching/grasping
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.3, 0.9)),
    )

    # EE marker for visualization
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
    """Configuration for G1 + Dex1 Stage 6 Loco-Manipulation environment."""

    # Environment timing
    decimation = 4
    episode_length_s = 20.0

    # Action/observation dimensions
    num_actions = 21  # 12 legs + 7 right arm + 2 right gripper
    num_observations = 85
    num_states = 0

    # Spaces (required by DirectRLEnvCfg)
    action_space = 21
    observation_space = 85
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,
        render_interval=4,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    # Scene - CRITICAL: Use the scene config class
    scene: G1LocoManipSceneCfg = G1LocoManipSceneCfg(num_envs=4096, env_spacing=2.5)

    # Action scaling
    action_scale = 0.5
    action_smoothing_alpha = 0.3

    # Curriculum
    curriculum_level = 0
    auto_curriculum = True

    # Reward weights
    reward_lin_vel_tracking = 1.5
    reward_ang_vel_tracking = 0.8
    reward_height_tracking = 0.5
    reward_base_stability = 1.0
    reward_upright = 0.5
    reward_arm_reach = 5.0
    reward_arm_reach_bonus = 20.0
    reward_grasp_contact = 30.0
    reward_action_rate = -0.01
    reward_joint_torque = -0.0001

    # Reach thresholds
    reach_threshold = 0.05

    # Target workspace (relative to shoulder)
    shoulder_offset = (0.0, -0.2, 0.4)  # Right shoulder relative to base


# ==============================================================================
# ENVIRONMENT CLASS
# ==============================================================================
class G1Dex1Stage6Env(DirectRLEnv):
    """
    G1 + Dex1 Stage 6 Loco-Manipulation Environment.

    Combines locomotion with arm reaching and grasping using Dex1 gripper.
    """

    cfg: G1Dex1Stage6EnvCfg

    def __init__(self, cfg: G1Dex1Stage6EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Curriculum
        self._curriculum_level = cfg.curriculum_level
        self._reach_success_count = torch.zeros(self.num_envs, device=self.device)
        self._reach_attempt_count = torch.zeros(self.num_envs, device=self.device)

        # Buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._smoothed_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._commands = torch.zeros(self.num_envs, 6, device=self.device)  # vx, vy, omega, pitch, height, phase
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Joint indices
        self._setup_joint_indices()

        # EE body index
        self._setup_ee_index()

        # Fixed root pose (for stability in early stages)
        self._fixed_root_pose = self.robot.data.default_root_state[:, :7].clone()
        self._fixed_root_pose[:, :3] += self.scene.env_origins
        self._zero_root_vel = torch.zeros(self.num_envs, 6, device=self.device)

        print(f"\n{'='*60}")
        print(f"G1 + Dex1 Stage 6 Loco-Manipulation Environment")
        print(f"{'='*60}")
        print(f"Environments: {self.num_envs}")
        print(f"Observations: {self.cfg.num_observations}")
        print(f"Actions: {self.cfg.num_actions}")
        print(f"Curriculum Level: {self._curriculum_level} ({self.current_curriculum.name})")
        print(f"Leg joints: {len(self._leg_joint_ids)}")
        print(f"Right arm joints: {len(self._right_arm_joint_ids)}")
        print(f"Right gripper joints: {len(self._right_gripper_joint_ids)}")
        print(f"{'='*60}\n")

    @property
    def robot(self) -> Articulation:
        return self.scene["robot"]

    @property
    def target(self) -> RigidObject:
        return self.scene["target"]

    @property
    def ee_marker(self) -> RigidObject:
        return self.scene["ee_marker"]

    @property
    def current_curriculum(self) -> CurriculumLevel:
        return CURRICULUM_LEVELS[min(self._curriculum_level, len(CURRICULUM_LEVELS)-1)]

    def _setup_joint_indices(self):
        """Set up joint indices for different body parts."""
        joint_names = list(self.robot.data.joint_names)

        # Leg joints
        self._leg_joint_ids = []
        for name in G1_LEG_JOINTS:
            if name in joint_names:
                self._leg_joint_ids.append(joint_names.index(name))
        self._leg_joint_ids = torch.tensor(self._leg_joint_ids, device=self.device, dtype=torch.long)

        # Right arm joints
        self._right_arm_joint_ids = []
        for name in G1_RIGHT_ARM_JOINTS:
            if name in joint_names:
                self._right_arm_joint_ids.append(joint_names.index(name))
        self._right_arm_joint_ids = torch.tensor(self._right_arm_joint_ids, device=self.device, dtype=torch.long)

        # Right gripper joints
        self._right_gripper_joint_ids = []
        for name in G1_RIGHT_GRIPPER_JOINTS:
            if name in joint_names:
                self._right_gripper_joint_ids.append(joint_names.index(name))
        self._right_gripper_joint_ids = torch.tensor(self._right_gripper_joint_ids, device=self.device, dtype=torch.long)

        if len(self._right_gripper_joint_ids) == 0:
            print(f"[WARNING] No gripper joints found! Available joints:")
            for i, name in enumerate(joint_names):
                if "hand" in name.lower():
                    print(f"  {i}: {name}")

    def _setup_ee_index(self):
        """Set up end-effector body index."""
        body_names = list(self.robot.data.body_names)

        if G1_RIGHT_EE_LINK in body_names:
            self._ee_body_idx = body_names.index(G1_RIGHT_EE_LINK)
        else:
            # Fallback - find similar name
            self._ee_body_idx = -1
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "wrist" in name.lower():
                    self._ee_body_idx = i
                    break

            if self._ee_body_idx == -1:
                print(f"[WARNING] Could not find right EE link. Using last body.")
                self._ee_body_idx = len(body_names) - 1

    def _get_ee_position(self) -> torch.Tensor:
        """Get end-effector world position."""
        return self.robot.data.body_pos_w[:, self._ee_body_idx, :3]

    def _get_projected_gravity(self) -> torch.Tensor:
        """Get gravity vector in body frame."""
        return self.robot.data.projected_gravity_b

    def _get_observations(self) -> dict:
        """Compute observations."""
        obs_list = []

        # Base state (9): lin_vel, ang_vel, proj_gravity
        obs_list.append(self.robot.data.root_lin_vel_b)  # 3
        obs_list.append(self.robot.data.root_ang_vel_b)  # 3
        obs_list.append(self._get_projected_gravity())   # 3

        # Leg joints (24): pos + vel
        if len(self._leg_joint_ids) > 0:
            obs_list.append(self.robot.data.joint_pos[:, self._leg_joint_ids])  # 12
            obs_list.append(self.robot.data.joint_vel[:, self._leg_joint_ids])  # 12
        else:
            obs_list.append(torch.zeros(self.num_envs, 24, device=self.device))

        # Commands (6)
        obs_list.append(self._commands)  # 6

        # Right arm joints (14): pos + vel
        if len(self._right_arm_joint_ids) > 0:
            obs_list.append(self.robot.data.joint_pos[:, self._right_arm_joint_ids])  # 7
            obs_list.append(self.robot.data.joint_vel[:, self._right_arm_joint_ids])  # 7
        else:
            obs_list.append(torch.zeros(self.num_envs, 14, device=self.device))

        # Right gripper joints (4): pos + vel
        if len(self._right_gripper_joint_ids) > 0:
            obs_list.append(self.robot.data.joint_pos[:, self._right_gripper_joint_ids])  # 2
            obs_list.append(self.robot.data.joint_vel[:, self._right_gripper_joint_ids])  # 2
        else:
            obs_list.append(torch.zeros(self.num_envs, 4, device=self.device))

        # Target in body frame (3)
        target_body = self._get_target_in_body_frame()
        obs_list.append(target_body)  # 3

        # EE to target distance (3)
        ee_pos = self._get_ee_position()
        obs_list.append(self._target_pos - ee_pos)  # 3

        # Gripper contacts placeholder (2)
        obs_list.append(torch.zeros(self.num_envs, 2, device=self.device))  # 2

        # Previous actions (21)
        obs_list.append(self._prev_actions)  # 21

        obs = torch.cat(obs_list, dim=-1)

        # Clamp to reasonable range
        obs = torch.clamp(obs, -100.0, 100.0)

        return {"policy": obs}

    def _get_target_in_body_frame(self) -> torch.Tensor:
        """Transform target position to body frame."""
        base_pos = self.robot.data.root_pos_w[:, :3]
        base_quat = self.robot.data.root_quat_w
        rel_pos = self._target_pos - base_pos
        return self._quat_rotate_inverse(base_quat, rel_pos)

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse quaternion (wxyz format)."""
        w = q[:, 0:1]
        xyz = q[:, 1:4]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        rewards = torch.zeros(self.num_envs, device=self.device)

        # === Locomotion ===
        # Velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rewards += torch.exp(-lin_vel_error / 0.25) * self.cfg.reward_lin_vel_tracking

        # Angular velocity tracking
        ang_vel_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        rewards += torch.exp(-ang_vel_error / 0.25) * self.cfg.reward_ang_vel_tracking

        # === Balance ===
        # Base stability
        ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        rewards += torch.exp(-ang_vel_xy / 0.5) * self.cfg.reward_base_stability

        # Upright
        proj_grav = self._get_projected_gravity()
        upright = proj_grav[:, 2]  # Should be close to -1
        rewards += torch.clamp(upright + 1.0, 0.0, 1.0) * 0.5 * self.cfg.reward_upright

        # === Arm Reaching ===
        ee_pos = self._get_ee_position()
        distance = torch.norm(ee_pos - self._target_pos, dim=1)

        # Distance reward (tanh)
        rewards += (1.0 - torch.tanh(distance / 0.1)) * self.cfg.reward_arm_reach

        # Reach bonus
        reached = distance < self.current_curriculum.reach_threshold
        rewards += reached.float() * self.cfg.reward_arm_reach_bonus

        # Update tracking
        self._reach_success_count += reached.float()
        self._reach_attempt_count += 1

        # Resample target when reached
        reached_ids = torch.where(reached)[0]
        if len(reached_ids) > 0:
            self._sample_target(reached_ids)

        # === Regularization ===
        action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
        rewards += action_rate * self.cfg.reward_action_rate

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination."""
        # Fall detection
        base_height = self.robot.data.root_pos_w[:, 2]
        fallen = base_height < 0.3

        # Orientation check
        proj_grav = self._get_projected_gravity()
        bad_orientation = proj_grav[:, 2] > -0.5

        terminated = fallen | bad_orientation
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # Reset robot
        self.robot.reset(env_ids)

        # Write default pose
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        # Reset joints
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(jp)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        # Sample commands and targets
        self._sample_commands(env_ids)
        self._sample_target(env_ids)

        # Reset buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._smoothed_actions[env_ids] = 0.0
        self._reach_success_count[env_ids] = 0
        self._reach_attempt_count[env_ids] = 0

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands for curriculum level."""
        level = self.current_curriculum
        n = len(env_ids)

        vx = torch.empty(n, device=self.device).uniform_(*level.vx_range)
        vy = torch.empty(n, device=self.device).uniform_(*level.vy_range)
        omega = torch.empty(n, device=self.device).uniform_(*level.omega_range)
        pitch = torch.empty(n, device=self.device).uniform_(*level.pitch_range)

        self._commands[env_ids, 0] = vx
        self._commands[env_ids, 1] = vy
        self._commands[env_ids, 2] = omega
        self._commands[env_ids, 3] = pitch
        self._commands[env_ids, 4] = 0.75  # Target height
        self._commands[env_ids, 5] = 0.0   # Phase

    def _sample_target(self, env_ids: torch.Tensor):
        """Sample target positions in workspace."""
        level = self.current_curriculum
        n = len(env_ids)

        # Sample in cylindrical coordinates relative to shoulder
        radius = torch.empty(n, device=self.device).uniform_(*level.reach_radius_range)
        height = torch.empty(n, device=self.device).uniform_(*level.reach_height_range)
        azimuth = torch.empty(n, device=self.device).uniform_(*level.reach_azimuth_range)

        # Convert to Cartesian (relative to shoulder)
        x = radius * torch.cos(azimuth)
        y = radius * torch.sin(azimuth) + self.cfg.shoulder_offset[1]
        z = height + self.cfg.shoulder_offset[2] + 0.75  # Add base height

        # World frame
        base_pos = self.robot.data.root_pos_w[env_ids, :3]
        self._target_pos[env_ids, 0] = base_pos[:, 0] + x
        self._target_pos[env_ids, 1] = base_pos[:, 1] + y
        self._target_pos[env_ids, 2] = z

        # Update target visual
        target_pose = torch.zeros(n, 7, device=self.device)
        target_pose[:, :3] = self._target_pos[env_ids]
        target_pose[:, 3] = 1.0  # quat w
        self.target.write_root_pose_to_sim(target_pose, env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions before physics step."""
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()

    def _apply_action(self):
        """Apply actions to robot."""
        # Smooth actions
        alpha = self.cfg.action_smoothing_alpha
        self._smoothed_actions = alpha * self._actions + (1 - alpha) * self._smoothed_actions

        # Scale actions
        scaled = self._smoothed_actions * self.cfg.action_scale

        # Get current joint targets
        joint_targets = self.robot.data.joint_pos.clone()

        # Apply leg actions (indices 0-11)
        if len(self._leg_joint_ids) > 0:
            for i, idx in enumerate(self._leg_joint_ids):
                if i < 12:
                    joint_targets[:, idx] += scaled[:, i]

        # Apply arm actions (indices 12-18)
        if len(self._right_arm_joint_ids) > 0:
            for i, idx in enumerate(self._right_arm_joint_ids):
                if i < 7:
                    joint_targets[:, idx] += scaled[:, 12 + i]

        # Apply gripper actions (indices 19-20)
        if self.current_curriculum.grasp_enabled and len(self._right_gripper_joint_ids) > 0:
            for i, idx in enumerate(self._right_gripper_joint_ids):
                if i < 2:
                    joint_targets[:, idx] = torch.clamp(scaled[:, 19 + i] + 0.5, 0.0, 1.0)

        self.robot.set_joint_position_target(joint_targets)

        # Update EE marker
        ee_pos = self._get_ee_position()
        ee_pose = torch.zeros(self.num_envs, 7, device=self.device)
        ee_pose[:, :3] = ee_pos
        ee_pose[:, 3] = 1.0
        self.ee_marker.write_root_pose_to_sim(ee_pose)

    def advance_curriculum(self) -> bool:
        """Check and advance curriculum if ready."""
        if not self.cfg.auto_curriculum:
            return False
        if self._curriculum_level >= len(CURRICULUM_LEVELS) - 1:
            return False

        level = self.current_curriculum
        total = self._reach_attempt_count.sum().item()
        success = self._reach_success_count.sum().item()

        if total < level.min_reach_success * self.num_envs:
            return False

        rate = success / total if total > 0 else 0

        if rate >= level.success_rate_threshold:
            self._curriculum_level += 1
            print(f"\n[CURRICULUM] Advancing to level {self._curriculum_level}: {self.current_curriculum.name}")
            print(f"[CURRICULUM] Success rate: {rate:.2%}")
            return True

        return False


# ==============================================================================
# ALIASES FOR COMPATIBILITY
# ==============================================================================
G1LocoManipEnv = G1Dex1Stage6Env
G1LocoManipEnvCfg = G1Dex1Stage6EnvCfg