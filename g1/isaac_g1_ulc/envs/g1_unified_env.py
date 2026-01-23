"""
G1 Unified Environment - Stage 6: Loco-Manipulation
====================================================

Stage 6: Walking + Arm Reaching birleşik environment.
Robot yürürken, squat yaparken, öne eğilirken kol kontrolü yapabilir.

HEDEFLER:
1. Yürüme sırasında denge koruma
2. Squat yapabilme (öne eğilme)
3. Kol ile hedeflere ulaşma
4. Yerden nesne alma hazırlığı

OBSERVATION SPACE (80 dims):
- Base velocities: 6 (lin_vel_b, ang_vel_b)
- Projected gravity: 3
- Leg joint pos/vel: 24 (12 + 12)
- Arm joint pos/vel: 10 (5 + 5)
- Commands: 10 (height, vel_cmd, torso_cmd, arm_target)
- Gait phase: 2
- Previous actions: 17
- Arm target error: 3
- EE position: 3
- Distance to target: 2

ACTION SPACE (17 dims):
- Legs: 12 joints
- Right arm: 5 joints

CHECKPOINT TRANSFER:
- Stage 3 checkpoint → leg controller weights
- Stage 5 checkpoint → arm controller weights
"""

from __future__ import annotations

import torch
import math
import numpy as np
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_apply_inverse


# ============================================================================
# CONSTANTS
# ============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Joint configurations
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

DEFAULT_LEG_POS = [-0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, -0.2, -0.2, 0.0, 0.0]

DEFAULT_ARM_POS = [-0.3, 0.0, 0.0, 0.5, 0.0]  # From Stage 5

ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

# End effector offset
EE_OFFSET = 0.02


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


# ============================================================================
# CURRICULUM LEVELS
# ============================================================================

CURRICULUM_LEVELS = [
    # Level 0: Standing + basic arm reach (warm-up)
    {
        "name": "standing_reach",
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "pitch_range": (0.0, 0.0),
        "arm_spawn_radius": 0.20,
        "arm_threshold": 0.12,
        "reward_threshold": 15.0,
        "min_steps": 500,
    },
    # Level 1: Slow walking + arm reach
    {
        "name": "slow_walk_reach",
        "vx_range": (0.0, 0.3),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.2, 0.2),
        "pitch_range": (0.0, 0.0),
        "arm_spawn_radius": 0.22,
        "arm_threshold": 0.11,
        "reward_threshold": 14.0,
        "min_steps": 800,
    },
    # Level 2: Walking + arm reach
    {
        "name": "walk_reach",
        "vx_range": (0.0, 0.5),
        "vy_range": (-0.15, 0.15),
        "vyaw_range": (-0.3, 0.3),
        "pitch_range": (-0.1, 0.0),
        "arm_spawn_radius": 0.25,
        "arm_threshold": 0.10,
        "reward_threshold": 13.0,
        "min_steps": 1000,
    },
    # Level 3: Walking with squat + arm reach
    {
        "name": "squat_reach",
        "vx_range": (0.0, 0.6),
        "vy_range": (-0.2, 0.2),
        "vyaw_range": (-0.4, 0.4),
        "pitch_range": (-0.2, 0.0),  # 12° forward lean
        "arm_spawn_radius": 0.28,
        "arm_threshold": 0.09,
        "reward_threshold": 12.0,
        "min_steps": 1200,
    },
    # Level 4: Deep squat + far reach (pick-up preparation)
    {
        "name": "deep_squat_reach",
        "vx_range": (-0.2, 0.7),
        "vy_range": (-0.25, 0.25),
        "vyaw_range": (-0.5, 0.5),
        "pitch_range": (-0.35, 0.0),  # 20° forward lean
        "arm_spawn_radius": 0.32,
        "arm_threshold": 0.08,
        "reward_threshold": 11.0,
        "min_steps": 1500,
    },
    # Level 5: Full loco-manipulation (final)
    {
        "name": "full_loco_manip",
        "vx_range": (-0.3, 0.8),
        "vy_range": (-0.3, 0.3),
        "vyaw_range": (-0.6, 0.6),
        "pitch_range": (-0.4, 0.0),  # 23° forward lean (max squat)
        "arm_spawn_radius": 0.35,
        "arm_threshold": 0.07,
        "reward_threshold": None,  # Final level
        "min_steps": 2000,
    },
]


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1UnifiedSceneCfg(InteractiveSceneCfg):
    """Scene configuration for unified loco-manipulation."""

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
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                # Legs
                "left_hip_pitch_joint": -0.2,
                "right_hip_pitch_joint": -0.2,
                "left_knee_joint": 0.4,
                "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2,
                "right_ankle_pitch_joint": -0.2,
                # Right arm (reaching position)
                "right_shoulder_pitch_joint": -0.3,
                "right_elbow_pitch_joint": 0.5,
                # Left arm (neutral)
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
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder.*", "right_elbow.*"],
                stiffness=150.0,
                damping=20.0,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_shoulder.*", "left_elbow.*"],
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

    # Target sphere for arm reaching
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

    # EE marker
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

@configclass
class G1UnifiedEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Unified Loco-Manipulation Environment."""

    decimation = 4
    episode_length_s = 15.0

    # Space dimensions
    num_actions = 17  # 12 legs + 5 right arm
    num_observations = 80
    num_states = 0

    action_space = 17
    observation_space = 80
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    scene: G1UnifiedSceneCfg = G1UnifiedSceneCfg(num_envs=1, env_spacing=2.5)

    # ============ LOCOMOTION PARAMETERS ============
    height_command = 0.72
    gait_frequency = 1.5

    # Action scales
    leg_action_scale = 0.4
    arm_action_scale = 0.12

    # Smoothing
    leg_smoothing_alpha = 0.3
    arm_smoothing_alpha = 0.15

    # ============ ARM WORKSPACE ============
    shoulder_center_offset = [0.0, 0.174, 0.259]
    workspace_inner_radius = 0.18
    workspace_outer_radius = 0.45

    # ============ REWARD WEIGHTS ============
    # Locomotion rewards
    w_vx = 2.5
    w_vy = 1.5
    w_vyaw = 1.5
    w_height = 2.0
    w_orientation = 2.0
    w_gait = 2.0
    w_torso_pitch = 3.0

    # Arm reaching rewards
    w_arm_pos = 3.0
    w_arm_reaching = 100.0
    w_arm_proximity = 1.5

    # Stability rewards
    w_com_stability = 4.0
    w_balance = 3.0

    # Penalties
    w_leg_action_rate = -0.01
    w_arm_action_rate = -0.02
    w_joint_vel = -0.002
    w_joint_acc = -0.005
    w_torque = -0.0003

    # Alive bonus
    w_alive = 0.5

    # ============ CURRICULUM SETTINGS ============
    min_success_rate = 0.60
    min_reaches_to_advance = 40
    min_steps_per_stage = 500

    # ============ ARM REACHING SETTINGS ============
    max_reaches_per_episode = 5
    terminate_on_max_reaches = True


# ============================================================================
# ENVIRONMENT
# ============================================================================

class G1UnifiedEnv(DirectRLEnv):
    """
    G1 Unified Loco-Manipulation Environment.

    Combines walking, torso control, and arm reaching into a single policy.
    """

    cfg: G1UnifiedEnvCfg

    def __init__(self, cfg: G1UnifiedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get scene objects
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        # Setup joint indices
        self._setup_joint_indices()

        # Default positions
        self.default_leg_pos = torch.tensor(DEFAULT_LEG_POS, device=self.device)
        self.default_arm_pos = torch.tensor(DEFAULT_ARM_POS, device=self.device)

        # Arm joint limits
        self.arm_joint_lower = torch.zeros(5, device=self.device)
        self.arm_joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(RIGHT_ARM_JOINT_NAMES):
            self.arm_joint_lower[i], self.arm_joint_upper[i] = ARM_JOINT_LIMITS[jn]

        # Commands
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * self.cfg.height_command
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # vx, vy, vyaw
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # roll, pitch, yaw
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)  # arm target

        # Gait phase
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        # Smoothed actions
        self.smoothed_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.smoothed_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_joint_vel = torch.zeros(self.num_envs, 17, device=self.device)

        # Shoulder center (for arm workspace)
        self.shoulder_center = torch.tensor(
            self.cfg.shoulder_center_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Arm reaching tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.total_reaches = 0
        self.total_attempts = 0
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        # Curriculum
        self.curriculum_level = 0
        self.stage_reaches = 0
        self.stage_attempts = 0
        self.stage_step_count = 0
        self.curr_history = []

        print("\n" + "=" * 70)
        print("G1 UNIFIED LOCO-MANIPULATION ENVIRONMENT - STAGE 6")
        print("=" * 70)
        print(f"  Leg joints: {self.leg_indices.tolist()}")
        print(f"  Arm joints: {self.arm_indices.tolist()}")
        print(f"  Palm index: {self.palm_idx}")
        print("-" * 70)
        print(f"  Observation space: {self.cfg.num_observations}")
        print(f"  Action space: {self.cfg.num_actions}")
        print(f"  Curriculum levels: {len(CURRICULUM_LEVELS)}")
        print("=" * 70 + "\n")

    def _setup_joint_indices(self):
        """Setup joint indices for legs and arm."""
        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        # Leg indices
        self.leg_indices = torch.tensor(
            [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
            device=self.device, dtype=torch.long
        )

        # Arm indices
        self.arm_indices = torch.tensor(
            [joint_names.index(n) for n in RIGHT_ARM_JOINT_NAMES if n in joint_names],
            device=self.device, dtype=torch.long
        )

        # Palm index for EE computation
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break

        if self.palm_idx is None:
            # Fallback to right_hand or similar
            for i, name in enumerate(body_names):
                if "right" in name.lower() and ("hand" in name.lower() or "wrist" in name.lower()):
                    self.palm_idx = i
                    break

    def _setup_scene(self):
        """Setup scene with robot."""
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    # =========================================================================
    # COMPUTE FUNCTIONS
    # =========================================================================

    def _compute_ee_pos(self) -> torch.Tensor:
        """Compute end-effector position."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def get_torso_euler(self) -> torch.Tensor:
        """Get torso orientation as euler angles."""
        quat = self.robot.data.root_quat_w
        return quat_to_euler_xyz(quat)

    # =========================================================================
    # CURRICULUM
    # =========================================================================

    def update_curriculum(self, iteration: int):
        """Update curriculum based on performance."""
        self.stage_step_count += 1
        level = CURRICULUM_LEVELS[self.curriculum_level]

        if self.stage_attempts > 0:
            success_rate = self.stage_reaches / self.stage_attempts
        else:
            success_rate = 0.0

        should_advance = (
            success_rate >= self.cfg.min_success_rate and
            self.stage_reaches >= self.cfg.min_reaches_to_advance and
            self.stage_step_count >= level["min_steps"] and
            self.curriculum_level < len(CURRICULUM_LEVELS) - 1 and
            level["reward_threshold"] is not None
        )

        if should_advance:
            self.curriculum_level += 1
            new_level = CURRICULUM_LEVELS[self.curriculum_level]

            print(f"\n{'='*60}")
            print(f"🎯 CURRICULUM ADVANCED TO LEVEL {self.curriculum_level + 1}/{len(CURRICULUM_LEVELS)}")
            print(f"   Name: {new_level['name']}")
            print(f"   Success rate: {success_rate*100:.1f}% ({self.stage_reaches}/{self.stage_attempts})")
            print(f"   Velocity range: {new_level['vx_range']}")
            print(f"   Pitch range: {new_level['pitch_range']}")
            print(f"   Arm radius: {new_level['arm_spawn_radius']*100:.0f}cm")
            print(f"   Arm threshold: {new_level['arm_threshold']*100:.0f}cm")
            print(f"{'='*60}\n")

            self.stage_reaches = 0
            self.stage_attempts = 0
            self.stage_step_count = 0

        return success_rate

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample new commands for given environments."""
        n = len(env_ids)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        # Velocity commands
        self.vel_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (level["vx_range"][1] - level["vx_range"][0]) + level["vx_range"][0]
        self.vel_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (level["vy_range"][1] - level["vy_range"][0]) + level["vy_range"][0]
        self.vel_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (level["vyaw_range"][1] - level["vyaw_range"][0]) + level["vyaw_range"][0]

        # Torso commands (pitch for squatting)
        self.torso_cmd[env_ids, 0] = 0.0  # roll
        self.torso_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (level["pitch_range"][1] - level["pitch_range"][0]) + level["pitch_range"][0]
        self.torso_cmd[env_ids, 2] = 0.0  # yaw

    def _sample_arm_target(self, env_ids: torch.Tensor):
        """Sample arm reaching targets."""
        n = len(env_ids)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        root_pos = self.robot.data.root_pos_w[env_ids]
        shoulder_rel = self.shoulder_center[env_ids]

        # Random direction (biased forward/down for reachability)
        direction = torch.randn((n, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction[:, 0] = -torch.abs(direction[:, 0])  # Bias forward

        inner = self.cfg.workspace_inner_radius
        outer = level["arm_spawn_radius"]
        outer = min(outer, self.cfg.workspace_outer_radius)

        distance = inner + torch.rand((n, 1), device=self.device) * (outer - inner)

        targets = shoulder_rel + direction * distance
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.55)

        self.target_pos[env_ids] = targets

        # Update target visualization
        target_world = root_pos + targets
        target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
        pose = torch.cat([target_world, target_quat], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Initialize distance tracking
        ee_pos = self._compute_ee_pos()[env_ids] - root_pos
        initial_dist = (ee_pos - targets).norm(dim=-1)
        self.prev_distance[env_ids] = initial_dist

        self.stage_attempts += len(env_ids)
        self.total_attempts += len(env_ids)

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    def _get_observations(self) -> dict:
        """
        Build observation vector (80 dims).

        Structure:
        - lin_vel_b: 3
        - ang_vel_b: 3
        - proj_gravity: 3
        - leg_joint_pos: 12
        - leg_joint_vel: 12
        - arm_joint_pos: 5
        - arm_joint_vel: 5
        - height_cmd: 1
        - vel_cmd: 3
        - torso_cmd: 3
        - gait_phase: 2 (sin, cos)
        - prev_leg_actions: 12
        - prev_arm_actions: 5
        - arm_target_pos: 3
        - ee_pos_rel: 3
        - pos_error: 3
        - pos_dist: 1
        - torso_euler: 3

        Total: 3+3+3+12+12+5+5+1+3+3+2+12+5+3+3+3+1+3 = 80
        """
        robot = self.robot
        quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        # Joint states
        leg_joint_pos = robot.data.joint_pos[:, self.leg_indices]
        leg_joint_vel = robot.data.joint_vel[:, self.leg_indices]
        arm_joint_pos = robot.data.joint_pos[:, self.arm_indices]
        arm_joint_vel = robot.data.joint_vel[:, self.arm_indices]

        # Gait phase encoding
        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.gait_phase),
            torch.cos(2 * np.pi * self.gait_phase)
        ], dim=-1)

        # Torso orientation
        torso_euler = self.get_torso_euler()

        # Arm reaching info
        ee_pos = self._compute_ee_pos() - root_pos
        pos_error = self.target_pos - ee_pos
        pos_dist = pos_error.norm(dim=-1, keepdim=True)

        # Update EE marker
        ee_quat = robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(
            torch.cat([self._compute_ee_pos(), ee_quat], dim=-1)
        )

        # Build observation
        obs = torch.cat([
            lin_vel_b,                          # 3
            ang_vel_b,                          # 3
            proj_gravity,                       # 3
            leg_joint_pos,                      # 12
            leg_joint_vel * 0.1,               # 12 (scaled)
            arm_joint_pos,                      # 5
            arm_joint_vel * 0.1,               # 5 (scaled)
            self.height_cmd.unsqueeze(-1),     # 1
            self.vel_cmd,                       # 3
            self.torso_cmd,                     # 3
            gait_phase,                         # 2
            self.prev_leg_actions,              # 12
            self.prev_arm_actions,              # 5
            self.target_pos,                    # 3
            ee_pos,                             # 3
            pos_error,                          # 3
            pos_dist / 0.5,                    # 1 (normalized)
            torso_euler,                        # 3
        ], dim=-1)

        return {"policy": obs.clamp(-10, 10).nan_to_num()}

    # =========================================================================
    # REWARDS
    # =========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for unified loco-manipulation."""
        robot = self.robot
        quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # Joint states
        leg_joint_pos = robot.data.joint_pos[:, self.leg_indices]
        leg_joint_vel = robot.data.joint_vel[:, self.leg_indices]
        arm_joint_vel = robot.data.joint_vel[:, self.arm_indices]

        # Torso
        torso_euler = self.get_torso_euler()

        # Arm reaching
        ee_pos = self._compute_ee_pos() - root_pos
        pos_dist = (ee_pos - self.target_pos).norm(dim=-1)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        # ==================== LOCOMOTION REWARDS ====================

        # Velocity tracking
        r_vx = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
        r_vy = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.vel_cmd[:, 1]) ** 2)
        r_vyaw = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.vel_cmd[:, 2]) ** 2)

        # Height tracking
        r_height = torch.exp(-10.0 * (root_pos[:, 2] - self.height_cmd) ** 2)

        # Base orientation (uprightness)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        base_tilt_error = proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2
        r_orientation = torch.exp(-3.0 * base_tilt_error)

        # Torso pitch tracking (for squat)
        pitch_err = (torso_euler[:, 1] - self.torso_cmd[:, 1]) ** 2
        r_torso_pitch = torch.exp(-5.0 * pitch_err)

        # Gait quality
        left_knee, right_knee = leg_joint_pos[:, 6], leg_joint_pos[:, 7]
        phase = self.gait_phase
        left_swing = (phase < 0.5).float()
        right_swing = (phase >= 0.5).float()

        knee_target_swing = 0.6
        knee_target_stance = 0.3
        knee_err = (
            (left_knee - (left_swing * knee_target_swing + (1 - left_swing) * knee_target_stance)) ** 2 +
            (right_knee - (right_swing * knee_target_swing + (1 - right_swing) * knee_target_stance)) ** 2
        )
        r_gait = torch.exp(-3.0 * knee_err)

        # ==================== ARM REACHING REWARDS ====================

        # Position reward (tanh kernel)
        r_arm_pos = 1.0 - torch.tanh(pos_dist / 0.12)

        # Proximity bonus
        proximity_bonus = torch.zeros_like(pos_dist)
        proximity_bonus = torch.where(pos_dist < 0.15, proximity_bonus + 1.0, proximity_bonus)
        proximity_bonus = torch.where(pos_dist < 0.10, proximity_bonus + 2.0, proximity_bonus)
        proximity_bonus = torch.where(pos_dist < 0.05, proximity_bonus + 5.0, proximity_bonus)

        # Sparse reaching reward
        reached = pos_dist < level["arm_threshold"]
        reached_ids = torch.where(reached)[0]

        if len(reached_ids) > 0:
            self.reach_count[reached_ids] += 1
            self.episode_reach_count[reached_ids] += 1
            self.stage_reaches += len(reached_ids)
            self.total_reaches += len(reached_ids)
            self._sample_arm_target(reached_ids)

        # ==================== STABILITY REWARDS ====================

        # CoM stability (simplified)
        xy_velocity = lin_vel_b[:, :2].norm(dim=-1)
        r_com_stability = torch.exp(-2.0 * xy_velocity ** 2) * (1.0 - torch.abs(self.vel_cmd[:, 0]).clamp(0, 1))

        # Balance (penalize excessive tilt when not commanded)
        r_balance = torch.exp(-5.0 * base_tilt_error)

        # ==================== PENALTIES ====================

        # Leg action smoothness
        leg_action_diff = self.smoothed_leg_actions - self.prev_leg_actions
        p_leg_action_rate = leg_action_diff.pow(2).sum(-1)

        # Arm action smoothness
        arm_action_diff = self.smoothed_arm_actions - self.prev_arm_actions
        p_arm_action_rate = arm_action_diff.pow(2).sum(-1)

        # Joint velocity
        all_vel = torch.cat([leg_joint_vel, arm_joint_vel], dim=-1)
        p_joint_vel = all_vel.pow(2).sum(-1)

        # Joint acceleration
        current_vel = torch.cat([leg_joint_vel, arm_joint_vel], dim=-1)
        p_joint_acc = (current_vel - self.prev_joint_vel).pow(2).sum(-1)
        self.prev_joint_vel = current_vel.clone()

        # Torque penalty
        p_torque = (leg_joint_vel.abs() * self.smoothed_leg_actions.abs()).sum(-1)

        # ==================== TOTAL REWARD ====================

        reward = (
            # Locomotion
            self.cfg.w_vx * r_vx +
            self.cfg.w_vy * r_vy +
            self.cfg.w_vyaw * r_vyaw +
            self.cfg.w_height * r_height +
            self.cfg.w_orientation * r_orientation +
            self.cfg.w_gait * r_gait +
            self.cfg.w_torso_pitch * r_torso_pitch +

            # Arm reaching
            self.cfg.w_arm_pos * r_arm_pos +
            self.cfg.w_arm_reaching * reached.float() +
            self.cfg.w_arm_proximity * proximity_bonus +

            # Stability
            self.cfg.w_com_stability * r_com_stability +
            self.cfg.w_balance * r_balance +

            # Penalties
            self.cfg.w_leg_action_rate * p_leg_action_rate +
            self.cfg.w_arm_action_rate * p_arm_action_rate +
            self.cfg.w_joint_vel * p_joint_vel +
            self.cfg.w_joint_acc * p_joint_acc +
            self.cfg.w_torque * p_torque +

            # Alive bonus
            self.cfg.w_alive
        )

        # Store extras
        self.extras = {
            "R/vx": r_vx.mean().item(),
            "R/height": r_height.mean().item(),
            "R/orientation": r_orientation.mean().item(),
            "R/gait": r_gait.mean().item(),
            "R/torso_pitch": r_torso_pitch.mean().item(),
            "R/arm_pos": r_arm_pos.mean().item(),
            "R/com_stability": r_com_stability.mean().item(),
            "M/height": root_pos[:, 2].mean().item(),
            "M/vx": lin_vel_b[:, 0].mean().item(),
            "M/pitch": torso_euler[:, 1].mean().item(),
            "M/arm_dist": pos_dist.mean().item(),
            "curriculum_level": self.curriculum_level,
            "total_reaches": self.total_reaches,
        }

        return reward.clamp(-10, 40)

    # =========================================================================
    # TERMINATION
    # =========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        robot = self.robot
        height = robot.data.root_pos_w[:, 2]
        quat = robot.data.root_quat_w

        # Orientation check
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        # Termination conditions
        fallen = (height < 0.3) | (height > 1.2)
        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

        terminated = fallen | bad_orientation

        # Max reaches termination
        if self.cfg.terminate_on_max_reaches:
            max_reached = self.episode_reach_count >= self.cfg.max_reaches_per_episode
            terminated = terminated | max_reached

        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    # =========================================================================
    # RESET
    # =========================================================================

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset selected environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)

        # Reset pose
        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        # Reset joint positions
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        # Sample new commands
        self._sample_commands(env_ids)
        self._sample_arm_target(env_ids)

        # Reset gait phase
        self.gait_phase[env_ids] = torch.rand(n, device=self.device)

        # Reset action history
        self.smoothed_leg_actions[env_ids] = 0.0
        self.smoothed_arm_actions[env_ids] = 0.0
        self.prev_leg_actions[env_ids] = 0.0
        self.prev_arm_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0

        # Reset reach tracking
        self.episode_reach_count[env_ids] = 0.0

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self.actions = actions.clone()

        # Split actions
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:]

        # Store previous for smoothing
        self.prev_leg_actions = self.smoothed_leg_actions.clone()
        self.prev_arm_actions = self.smoothed_arm_actions.clone()

        # Apply smoothing
        self.smoothed_leg_actions = (
            self.cfg.leg_smoothing_alpha * leg_actions +
            (1 - self.cfg.leg_smoothing_alpha) * self.smoothed_leg_actions
        )
        self.smoothed_arm_actions = (
            self.cfg.arm_smoothing_alpha * arm_actions +
            (1 - self.cfg.arm_smoothing_alpha) * self.smoothed_arm_actions
        )

        # Compute target positions
        target_pos = self.robot.data.default_joint_pos.clone()

        # Legs
        target_pos[:, self.leg_indices] = self.default_leg_pos + self.smoothed_leg_actions * self.cfg.leg_action_scale

        # Arm with joint limits
        arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
        arm_target = torch.clamp(
            arm_pos + self.smoothed_arm_actions * self.cfg.arm_action_scale,
            self.arm_joint_lower,
            self.arm_joint_upper
        )
        target_pos[:, self.arm_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)

        # Update gait phase
        self.gait_phase = (self.gait_phase + self.cfg.gait_frequency * self.cfg.sim.dt * self.cfg.decimation) % 1.0

    def _apply_action(self):
        """Apply actions - handled by set_joint_position_target."""
        pass

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_success_rate(self) -> float:
        """Get current success rate."""
        if self.total_attempts > 0:
            return self.total_reaches / self.total_attempts
        return 0.0


# Alias for backward compatibility
G1LocoManipEnv = G1UnifiedEnv
G1LocoManipEnvCfg = G1UnifiedEnvCfg