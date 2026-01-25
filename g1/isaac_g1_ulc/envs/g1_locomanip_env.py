"""
G1 29DoF + Dex1 Stage 6 Loco-Manipulation Environment
======================================================

Based on Unitree G1 29DoF with Dex1 gripper official specifications.

Joint Structure:
- 12 leg joints (6 per leg)
- 3 waist joints (yaw, roll, pitch) - LOCKED for stability
- 14 arm joints (7 per arm)
- 4 gripper joints (2 per hand - Dex1)

Total: 29 DoF (body) + 4 DoF (grippers) = 33 controlled joints
We control: 12 legs + 5 right arm + 2 right gripper = 19 actions

Author: Turan (VLM-RL Project)
Date: January 2026
"""

from __future__ import annotations

import math
import torch
from typing import Dict, Tuple, Sequence
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

# ==============================================================================
# G1 29DOF + DEX1 JOINT CONFIGURATION
# ==============================================================================
"""
G1 29DoF + Dex1 Joint Structure (from unitree_sim_isaaclab):
------------------------------------------------------------

LEGS (12 joints):
  left_hip_yaw_joint, left_hip_roll_joint, left_hip_pitch_joint
  left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint
  right_hip_yaw_joint, right_hip_roll_joint, right_hip_pitch_joint
  right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint

WAIST (3 joints) - HIGH STIFFNESS (locked):
  waist_yaw_joint, waist_roll_joint, waist_pitch_joint

ARMS (14 joints - 7 per arm):
  left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint
  left_elbow_joint
  left_wrist_roll_joint, left_wrist_pitch_joint, left_wrist_yaw_joint
  
  right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint
  right_elbow_joint
  right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint

DEX1 GRIPPER (4 joints - 2 per hand):
  left_hand_Joint1_1, left_hand_Joint2_1
  right_hand_Joint1_1, right_hand_Joint2_1

End Effectors:
  - left_wrist_yaw_link
  - right_wrist_yaw_link
"""

# Joint name lists
G1_LEG_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

G1_WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# Right arm only (7 joints) - we control this
G1_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Left arm (for symmetry, not controlled in Stage 6)
G1_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

# Dex1 Gripper joints (2 per hand)
G1_RIGHT_GRIPPER_JOINT_NAMES = [
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
]

G1_LEFT_GRIPPER_JOINT_NAMES = [
    "left_hand_Joint1_1",
    "left_hand_Joint2_1",
]

# End effector link names
G1_RIGHT_EE_LINK = "right_wrist_yaw_link"
G1_LEFT_EE_LINK = "left_wrist_yaw_link"

# Foot link names
G1_LEFT_FOOT_LINK = "left_ankle_roll_link"
G1_RIGHT_FOOT_LINK = "right_ankle_roll_link"

# ==============================================================================
# DEFAULT JOINT POSITIONS
# ==============================================================================
G1_DEFAULT_JOINT_POS = {
    # Legs - standing pose
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.05,
    "right_knee_joint": 0.2,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    # Waist - locked at zero
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # Left arm - relaxed position
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    # Right arm - ready position
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    # Grippers - open
    "left_hand_Joint1_1": 0.0,
    "left_hand_Joint2_1": 0.0,
    "right_hand_Joint1_1": 0.0,
    "right_hand_Joint2_1": 0.0,
}

# ==============================================================================
# CURRICULUM CONFIGURATION
# ==============================================================================
@dataclass
class CurriculumLevel:
    """Configuration for a curriculum level."""
    name: str
    # Velocity command ranges
    vx_range: Tuple[float, float] = (0.0, 0.0)
    vy_range: Tuple[float, float] = (0.0, 0.0)
    omega_range: Tuple[float, float] = (0.0, 0.0)
    # Body pitch (for squatting)
    pitch_range: Tuple[float, float] = (0.0, 0.0)
    # Reaching workspace (relative to right shoulder)
    reach_radius_range: Tuple[float, float] = (0.20, 0.30)
    reach_height_range: Tuple[float, float] = (-0.10, 0.15)
    reach_azimuth_range: Tuple[float, float] = (-0.5, 0.5)
    # Grasp settings
    grasp_enabled: bool = False
    # Success criteria
    reach_threshold: float = 0.05
    min_reach_success: int = 50
    success_rate_threshold: float = 0.7


CURRICULUM_LEVELS = [
    # Level 0: Standing reach (no grasp) - Robot ayakta, kol hedeflere uzanıyor
    CurriculumLevel(
        name="standing_reach",
        vx_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        reach_radius_range=(0.20, 0.28),
        reach_height_range=(-0.05, 0.15),
        reach_azimuth_range=(-0.3, 0.3),
        grasp_enabled=False,
        reach_threshold=0.05,
        min_reach_success=50,
        success_rate_threshold=0.7,
    ),
    # Level 1: Standing grasp - Ayakta + gripper aktif
    CurriculumLevel(
        name="standing_grasp",
        vx_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        reach_radius_range=(0.20, 0.30),
        reach_height_range=(-0.10, 0.18),
        reach_azimuth_range=(-0.4, 0.4),
        grasp_enabled=True,
        reach_threshold=0.05,
        min_reach_success=30,
        success_rate_threshold=0.6,
    ),
    # Level 2: Walk + grasp - Yürüyerek + gripper
    CurriculumLevel(
        name="walk_grasp",
        vx_range=(0.0, 0.4),
        vy_range=(-0.1, 0.1),
        omega_range=(-0.2, 0.2),
        pitch_range=(0.0, 0.0),
        reach_radius_range=(0.20, 0.35),
        reach_height_range=(-0.20, 0.22),
        reach_azimuth_range=(-0.5, 0.5),
        grasp_enabled=True,
        reach_threshold=0.06,
        min_reach_success=30,
        success_rate_threshold=0.5,
    ),
    # Level 3: Squat + grasp - Çömelme + gripper (düşük hedefler)
    CurriculumLevel(
        name="squat_grasp",
        vx_range=(0.0, 0.3),
        pitch_range=(-0.25, 0.0),
        reach_radius_range=(0.20, 0.40),
        reach_height_range=(-0.40, 0.20),
        reach_azimuth_range=(-0.6, 0.6),
        grasp_enabled=True,
        reach_threshold=0.07,
        min_reach_success=25,
        success_rate_threshold=0.4,
    ),
    # Level 4: Full loco-manipulation - Tam hareket + geniş workspace
    CurriculumLevel(
        name="full_locomanip",
        vx_range=(-0.2, 0.6),
        vy_range=(-0.2, 0.2),
        omega_range=(-0.3, 0.3),
        pitch_range=(-0.35, 0.0),
        reach_radius_range=(0.20, 0.45),
        reach_height_range=(-0.50, 0.30),
        reach_azimuth_range=(-0.8, 0.8),
        grasp_enabled=True,
        reach_threshold=0.08,
        min_reach_success=20,
        success_rate_threshold=0.35,
    ),
]

# ==============================================================================
# OBSERVATION & ACTION DIMENSIONS
# ==============================================================================
"""
Observation Space (85 dimensions):
----------------------------------
[0-2]     Base linear velocity (3)
[3-5]     Base angular velocity (3)
[6-8]     Projected gravity (3)
[9-20]    Leg joint positions (12)
[21-32]   Leg joint velocities (12)
[33-38]   Commands: vx, vy, omega, pitch, height, phase (6)
[39-45]   Right arm joint positions (7)
[46-52]   Right arm joint velocities (7)
[53-54]   Right gripper joint positions (2)
[55-56]   Right gripper joint velocities (2)
[57-59]   Target position in body frame (3)
[60-62]   EE to target distance (3)
[63-64]   Gripper contact state (2)
[65-84]   Previous actions (20)

Total: 85 dimensions

Action Space (21 dimensions):
-----------------------------
[0-11]    Leg actions (12)
[12-18]   Right arm actions (7)
[19-20]   Right gripper actions (2)

Total: 21 actions
"""

OBS_DIM = 85
ACT_DIM = 21

# ==============================================================================
# REWARD WEIGHTS
# ==============================================================================
@dataclass
class RewardWeights:
    """Reward weights for Stage 6."""
    # Locomotion rewards
    w_lin_vel_tracking: float = 1.5
    w_ang_vel_tracking: float = 0.8
    w_height_tracking: float = 0.5

    # Balance rewards
    w_base_stability: float = 1.0
    w_upright: float = 0.5
    w_foot_contact: float = 0.3

    # Arm reaching rewards
    w_arm_reach: float = 5.0
    w_arm_reach_bonus: float = 20.0
    w_ee_orientation: float = 0.5

    # Grasp rewards (when enabled)
    w_grasp_approach: float = 3.0
    w_grasp_contact: float = 30.0
    w_grasp_stable: float = 50.0

    # Regularization
    w_action_rate: float = -0.01
    w_joint_torque: float = -0.0001
    w_joint_accel: float = -0.0001
    w_joint_pos_limits: float = -1.0

    # Termination
    w_fall_penalty: float = -100.0


# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
@configclass
class G1Dex1Stage6EnvCfg(DirectRLEnvCfg):
    """Configuration for G1 + Dex1 Stage 6 Loco-Manipulation environment."""

    # Environment
    episode_length_s: float = 20.0
    decimation: int = 4
    action_scale: float = 0.5

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Robot - G1 29DoF with Dex1 (Wholebody version)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # Use Unitree's wholebody Dex1 USD
            usd_path="C:/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
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
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=150.0,
                damping=5.0,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness=100.0,
                damping=5.0,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                effort_limit=1000.0,
                velocity_limit=0.0,  # Locked
                stiffness=10000.0,
                damping=10000.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
                effort_limit=100.0,
                velocity_limit=100.0,
                stiffness={
                    ".*_shoulder_.*_joint": 80.0,
                    ".*_elbow_joint": 80.0,
                    ".*_wrist_.*_joint": 50.0,
                },
                damping={
                    ".*_shoulder_.*_joint": 4.0,
                    ".*_elbow_joint": 4.0,
                    ".*_wrist_.*_joint": 2.0,
                },
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_Joint.*"],
                effort_limit=50.0,
                velocity_limit=50.0,
                stiffness=30.0,
                damping=1.0,
            ),
        },
    )

    # Target object for grasping (cylinder)
    target_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetObject",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.2, 0.8),  # In front of right hand
        ),
    )

    # Reward weights
    rewards: RewardWeights = field(default_factory=RewardWeights)

    # Curriculum
    curriculum_level: int = 0
    auto_curriculum: bool = True

    # Dimensions
    num_observations: int = OBS_DIM
    num_actions: int = ACT_DIM

    # Noise
    add_noise: bool = True
    noise_level: float = 0.05


# ==============================================================================
# ENVIRONMENT CLASS
# ==============================================================================
class G1Dex1Stage6Env(DirectRLEnv):
    """
    G1 + Dex1 Stage 6 Loco-Manipulation Environment.

    Combines locomotion with arm reaching and grasping using Dex1 gripper.
    Uses curriculum learning to progressively increase task difficulty.
    """

    cfg: G1Dex1Stage6EnvCfg

    def __init__(self, cfg: G1Dex1Stage6EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Curriculum tracking
        self._curriculum_level = cfg.curriculum_level
        self._reach_success_count = torch.zeros(self.num_envs, device=self.device)
        self._reach_attempt_count = torch.zeros(self.num_envs, device=self.device)

        # Action/state buffers
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands = torch.zeros(self.num_envs, 6, device=self.device)
        self._gripper_contacts = torch.zeros(self.num_envs, 2, device=self.device)

        # Joint indices (will be set in _setup_scene)
        self.leg_joint_ids = []
        self.right_arm_joint_ids = []
        self.right_gripper_joint_ids = []

        print(f"\n{'='*60}")
        print(f"G1 + Dex1 Stage 6 Loco-Manipulation Environment")
        print(f"{'='*60}")
        print(f"Environments: {self.num_envs}")
        print(f"Observations: {self.cfg.num_observations}")
        print(f"Actions: {self.cfg.num_actions}")
        print(f"Curriculum Level: {self._curriculum_level} ({self.current_curriculum.name})")
        print(f"{'='*60}\n")

    @property
    def current_curriculum(self) -> CurriculumLevel:
        """Get current curriculum level configuration."""
        return CURRICULUM_LEVELS[min(self._curriculum_level, len(CURRICULUM_LEVELS)-1)]

    def _setup_scene(self):
        """Set up the simulation scene."""
        # Add robot
        self.robot = self.scene.articulations["robot"]

        # Add target object (only if grasping curriculum)
        if self.cfg.target_object_cfg is not None:
            self.target_object = self.scene.rigid_objects.get("target_object", None)
        else:
            self.target_object = None

        # Get joint indices
        self._setup_joint_indices()

        # Get body indices for EE
        body_names = self.robot.data.body_names
        if G1_RIGHT_EE_LINK in body_names:
            self.ee_body_idx = body_names.index(G1_RIGHT_EE_LINK)
        else:
            # Fallback - find similar name
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "wrist" in name.lower():
                    self.ee_body_idx = i
                    break
            else:
                self.ee_body_idx = -1
                print(f"[WARNING] Could not find right EE link. Available: {body_names}")

    def _setup_joint_indices(self):
        """Set up joint indices for different body parts."""
        joint_names = list(self.robot.data.joint_names)

        print(f"\n[Setup] Robot has {len(joint_names)} joints:")

        # Leg joints
        self.leg_joint_ids = []
        for name in G1_LEG_JOINT_NAMES:
            if name in joint_names:
                self.leg_joint_ids.append(joint_names.index(name))
        print(f"  - Leg joints: {len(self.leg_joint_ids)}")

        # Right arm joints
        self.right_arm_joint_ids = []
        for name in G1_RIGHT_ARM_JOINT_NAMES:
            if name in joint_names:
                self.right_arm_joint_ids.append(joint_names.index(name))
        print(f"  - Right arm joints: {len(self.right_arm_joint_ids)}")

        # Right gripper joints
        self.right_gripper_joint_ids = []
        for name in G1_RIGHT_GRIPPER_JOINT_NAMES:
            if name in joint_names:
                self.right_gripper_joint_ids.append(joint_names.index(name))
        print(f"  - Right gripper joints: {len(self.right_gripper_joint_ids)}")

        if len(self.right_gripper_joint_ids) == 0:
            print(f"  [WARNING] No gripper joints found! Available joints:")
            for i, name in enumerate(joint_names):
                if "hand" in name.lower() or "gripper" in name.lower():
                    print(f"    {i}: {name}")

    def _get_observations(self) -> dict:
        """Compute observations."""
        obs = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)

        # Base state (9)
        obs[:, 0:3] = self.robot.data.root_lin_vel_b
        obs[:, 3:6] = self.robot.data.root_ang_vel_b
        obs[:, 6:9] = self._get_projected_gravity()

        # Leg joints (24)
        if len(self.leg_joint_ids) > 0:
            obs[:, 9:9+len(self.leg_joint_ids)] = self.robot.data.joint_pos[:, self.leg_joint_ids]
            obs[:, 21:21+len(self.leg_joint_ids)] = self.robot.data.joint_vel[:, self.leg_joint_ids]

        # Commands (6)
        obs[:, 33:39] = self._commands

        # Right arm joints (14)
        if len(self.right_arm_joint_ids) > 0:
            obs[:, 39:39+len(self.right_arm_joint_ids)] = self.robot.data.joint_pos[:, self.right_arm_joint_ids]
            obs[:, 46:46+len(self.right_arm_joint_ids)] = self.robot.data.joint_vel[:, self.right_arm_joint_ids]

        # Right gripper joints (4)
        if len(self.right_gripper_joint_ids) > 0:
            obs[:, 53:53+len(self.right_gripper_joint_ids)] = self.robot.data.joint_pos[:, self.right_gripper_joint_ids]
            obs[:, 55:55+len(self.right_gripper_joint_ids)] = self.robot.data.joint_vel[:, self.right_gripper_joint_ids]

        # Target in body frame (3)
        target_body = self._get_target_in_body_frame()
        obs[:, 57:60] = target_body

        # EE to target distance (3)
        ee_pos = self._get_ee_position()
        obs[:, 60:63] = self._target_pos - ee_pos

        # Gripper contacts (2)
        obs[:, 63:65] = self._gripper_contacts

        # Previous actions (21 -> pad to 20)
        obs[:, 65:85] = self._prev_actions[:, :20]

        # Add noise
        if self.cfg.add_noise:
            obs += self.cfg.noise_level * torch.randn_like(obs)

        return {"policy": obs}

    def _get_projected_gravity(self) -> torch.Tensor:
        """Get gravity vector in body frame."""
        quat = self.robot.data.root_quat_w
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        return self._quat_rotate_inverse(quat, gravity)

    def _get_target_in_body_frame(self) -> torch.Tensor:
        """Transform target to body frame."""
        base_pos = self.robot.data.root_pos_w[:, :3]
        base_quat = self.robot.data.root_quat_w
        rel_pos = self._target_pos - base_pos
        return self._quat_rotate_inverse(base_quat, rel_pos)

    def _get_ee_position(self) -> torch.Tensor:
        """Get end-effector position."""
        if self.ee_body_idx >= 0:
            return self.robot.data.body_pos_w[:, self.ee_body_idx, :3]
        else:
            # Fallback: approximate from base
            base_pos = self.robot.data.root_pos_w[:, :3]
            offset = torch.tensor([0.3, -0.3, 0.3], device=self.device)
            return base_pos + offset

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse quaternion."""
        q_conj = q.clone()
        q_conj[:, :3] = -q_conj[:, :3]
        return self._quat_rotate(q_conj, v)

    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by quaternion."""
        qvec = q[:, :3]
        qw = q[:, 3:4]
        t = 2 * torch.cross(qvec, v, dim=-1)
        return v + qw * t + torch.cross(qvec, t, dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions."""
        self._prev_actions = actions.clone()

        # Scale actions
        scaled = actions * self.cfg.action_scale

        # Get current positions
        joint_targets = self.robot.data.joint_pos.clone()

        # Apply leg actions (12)
        for i, idx in enumerate(self.leg_joint_ids):
            if i < 12:
                joint_targets[:, idx] += scaled[:, i]

        # Apply arm actions (7)
        for i, idx in enumerate(self.right_arm_joint_ids):
            if i < 7:
                joint_targets[:, idx] += scaled[:, 12 + i]

        # Apply gripper actions (2)
        if self.current_curriculum.grasp_enabled:
            for i, idx in enumerate(self.right_gripper_joint_ids):
                if i < 2:
                    # Gripper: action maps to [0, 1] range (0=open, 1=closed)
                    joint_targets[:, idx] = torch.clamp(scaled[:, 19 + i] + 0.5, 0.0, 1.0)

        self.robot.set_joint_position_target(joint_targets)

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        w = self.cfg.rewards
        rewards = torch.zeros(self.num_envs, device=self.device)

        # === Locomotion Rewards ===
        # Velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1
        )
        rewards += torch.exp(-lin_vel_error / 0.25) * w.w_lin_vel_tracking

        # Angular velocity tracking
        ang_vel_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        rewards += torch.exp(-ang_vel_error / 0.25) * w.w_ang_vel_tracking

        # === Balance Rewards ===
        # Base stability (low roll/pitch velocity)
        ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        rewards += torch.exp(-ang_vel_xy / 0.5) * w.w_base_stability

        # Upright reward (gravity should point down in body frame)
        proj_grav = self._get_projected_gravity()
        upright = proj_grav[:, 2]  # Should be close to -1
        rewards += (upright + 1.0) * 0.5 * w.w_upright  # Normalize to [0, 1]

        # === Arm Reaching Rewards ===
        ee_pos = self._get_ee_position()
        distance = torch.norm(ee_pos - self._target_pos, dim=1)

        # Distance reward (exponential)
        rewards += torch.exp(-distance / 0.1) * w.w_arm_reach

        # Reach bonus
        reached = distance < self.current_curriculum.reach_threshold
        rewards += reached.float() * w.w_arm_reach_bonus

        # Update tracking
        self._reach_success_count += reached.float()
        self._reach_attempt_count += 1

        # === Grasp Rewards (if enabled) ===
        if self.current_curriculum.grasp_enabled:
            # Approach reward (closer to object = higher)
            rewards += torch.exp(-distance / 0.15) * w.w_grasp_approach

            # Contact reward (simulated)
            contact_reward = (self._gripper_contacts.sum(dim=1) / 2.0) * w.w_grasp_contact
            rewards += contact_reward

        # === Regularization ===
        # Action rate
        action_rate = torch.sum(torch.square(self._prev_actions), dim=1)
        rewards += action_rate * w.w_action_rate

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination."""
        # Fall detection
        base_height = self.robot.data.root_pos_w[:, 2]
        fallen = base_height < 0.3

        # Orientation check
        proj_grav = self._get_projected_gravity()
        bad_orientation = proj_grav[:, 2] > -0.5  # Tilted too much

        # Timeout
        timeout = self.episode_length_buf >= self.max_episode_length

        terminated = fallen | bad_orientation
        truncated = timeout

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        if len(env_ids) == 0:
            return

        # Reset robot
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        # Reset joints
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

        # Sample new commands and targets
        self._sample_commands(env_ids)
        self._sample_target(env_ids)

        # Reset buffers
        self._prev_actions[env_ids] = 0
        self._reach_success_count[env_ids] = 0
        self._reach_attempt_count[env_ids] = 0
        self._gripper_contacts[env_ids] = 0

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands."""
        level = self.current_curriculum
        n = len(env_ids)

        vx = torch.rand(n, device=self.device) * (level.vx_range[1] - level.vx_range[0]) + level.vx_range[0]
        vy = torch.rand(n, device=self.device) * (level.vy_range[1] - level.vy_range[0]) + level.vy_range[0]
        omega = torch.rand(n, device=self.device) * (level.omega_range[1] - level.omega_range[0]) + level.omega_range[0]
        pitch = torch.rand(n, device=self.device) * (level.pitch_range[1] - level.pitch_range[0]) + level.pitch_range[0]

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

        # Sample in cylindrical coordinates
        radius = torch.rand(n, device=self.device) * (level.reach_radius_range[1] - level.reach_radius_range[0]) + level.reach_radius_range[0]
        height = torch.rand(n, device=self.device) * (level.reach_height_range[1] - level.reach_height_range[0]) + level.reach_height_range[0]
        azimuth = torch.rand(n, device=self.device) * (level.reach_azimuth_range[1] - level.reach_azimuth_range[0]) + level.reach_azimuth_range[0]

        # Convert to Cartesian (relative to shoulder)
        x = radius * torch.cos(azimuth)
        y = radius * torch.sin(azimuth) - 0.25  # Offset for right shoulder
        z = height + 0.75  # Shoulder height

        # World frame
        base_pos = self.robot.data.root_pos_w[env_ids, :3]
        self._target_pos[env_ids, 0] = base_pos[:, 0] + x
        self._target_pos[env_ids, 1] = base_pos[:, 1] + y
        self._target_pos[env_ids, 2] = z

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
# REGISTRATION
# ==============================================================================
if __name__ == "__main__":
    print("G1 + Dex1 Stage 6 Environment Configuration")
    print("=" * 50)
    print(f"Observations: {OBS_DIM}")
    print(f"Actions: {ACT_DIM}")
    print(f"  - Legs: 12")
    print(f"  - Right Arm: 7")
    print(f"  - Right Gripper: 2")
    print()
    print("Curriculum Levels:")
    for i, level in enumerate(CURRICULUM_LEVELS):
        print(f"  {i}: {level.name}")
        print(f"     vx: {level.vx_range}, grasp: {level.grasp_enabled}")