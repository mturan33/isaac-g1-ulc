"""
G1 Reactive Balance Environment
================================

REACTIVE BALANCE ARCHITECTURE:
- Arm Policy: FROZEN (Stage 5 checkpoint - reaching expert)
- Balance Loco Policy: TRAINABLE (learns to keep balance while arm moves)

KEY INSIGHT:
Kol zaten hedeflere nasıl ulaşacağını biliyor.
Bacaklar kolun hareketine ADAPTE olmayı öğrenecek:
- Kol uzanırken lean forward
- Kol aşağı inerken squat
- Kol hareket ederken CoM dengesi

OBSERVATION SPACE (75 dims):
────────────────────────────
Base state (9):
- lin_vel_b: 3
- ang_vel_b: 3
- proj_gravity: 3

Leg state (24):
- leg_joint_pos: 12
- leg_joint_vel: 12

Commands (10):
- height_cmd: 1
- vel_cmd: 3
- torso_cmd: 3
- gait_phase: 2
- (reserved): 1

🆕 Center of Mass (6):
- com_pos_b: 3 (CoM position in body frame)
- com_vel_b: 3 (CoM velocity in body frame)

🆕 Arm awareness (14):
- arm_joint_pos: 5 (where is arm now)
- arm_joint_vel: 5 (how fast is arm moving)
- arm_actions: 5 (what arm policy is commanding) - REMOVED, use target instead
- arm_target: 3 (where is arm going)
- ee_pos_body: 3 (current EE position)

Previous actions (12):
- prev_leg_actions: 12

Total: 9 + 24 + 10 + 6 + 11 + 12 = 72 (adjusted)

ACTION SPACE:
- Loco policy outputs: 12 leg actions
- Arm actions come from frozen policy: 5

CYLINDRICAL WORKSPACE:
- Radius: 20-45cm
- Height: -50cm to +35cm (relative to shoulder)
- Angle: -180° to +180° (full cylinder)
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_apply_inverse


# ============================================================================
# CONSTANTS
# ============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

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
DEFAULT_ARM_POS = [-0.3, 0.0, 0.0, 0.5, 0.0]

ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

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
    # Level 0: Standing + basic arm reach (warm-up for balance)
    {
        "name": "standing_balance",
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "pitch_range": (0.0, 0.0),
        "arm_radius_min": 0.20,
        "arm_radius_max": 0.28,
        "arm_height_min": -0.05,
        "arm_height_max": 0.15,
        "arm_angle_range": (-90, 90),
        "arm_threshold": 0.12,
        "min_steps": 500,
    },
    # Level 1: Slow walking + balance
    {
        "name": "walk_balance",
        "vx_range": (0.0, 0.3),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.2, 0.2),
        "pitch_range": (0.0, 0.0),
        "arm_radius_min": 0.20,
        "arm_radius_max": 0.32,
        "arm_height_min": -0.15,
        "arm_height_max": 0.20,
        "arm_angle_range": (-120, 120),
        "arm_threshold": 0.11,
        "min_steps": 800,
    },
    # Level 2: Walking + lean forward
    {
        "name": "lean_balance",
        "vx_range": (0.0, 0.5),
        "vy_range": (-0.15, 0.15),
        "vyaw_range": (-0.3, 0.3),
        "pitch_range": (-0.15, 0.0),  # Start learning to lean
        "arm_radius_min": 0.20,
        "arm_radius_max": 0.36,
        "arm_height_min": -0.25,
        "arm_height_max": 0.22,
        "arm_angle_range": (-150, 150),
        "arm_threshold": 0.10,
        "min_steps": 1000,
    },
    # Level 3: Squat for lower targets
    {
        "name": "squat_balance",
        "vx_range": (0.0, 0.5),
        "vy_range": (-0.2, 0.2),
        "vyaw_range": (-0.4, 0.4),
        "pitch_range": (-0.25, 0.0),  # More lean
        "arm_radius_min": 0.20,
        "arm_radius_max": 0.40,
        "arm_height_min": -0.38,  # Lower targets
        "arm_height_max": 0.25,
        "arm_angle_range": (-180, 180),
        "arm_threshold": 0.09,
        "min_steps": 1200,
    },
    # Level 4: Deep squat + far reach
    {
        "name": "deep_squat_balance",
        "vx_range": (-0.2, 0.6),
        "vy_range": (-0.25, 0.25),
        "vyaw_range": (-0.5, 0.5),
        "pitch_range": (-0.35, 0.0),
        "arm_radius_min": 0.20,
        "arm_radius_max": 0.43,
        "arm_height_min": -0.45,
        "arm_height_max": 0.30,
        "arm_angle_range": (-180, 180),
        "arm_threshold": 0.08,
        "min_steps": 1500,
    },
    # Level 5: Full range
    {
        "name": "full_balance",
        "vx_range": (-0.3, 0.7),
        "vy_range": (-0.3, 0.3),
        "vyaw_range": (-0.6, 0.6),
        "pitch_range": (-0.4, 0.0),
        "arm_radius_min": 0.20,
        "arm_radius_max": 0.45,
        "arm_height_min": -0.50,
        "arm_height_max": 0.35,
        "arm_angle_range": (-180, 180),
        "arm_threshold": 0.07,
        "min_steps": 2000,
    },
]


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1ReactiveBalanceSceneCfg(InteractiveSceneCfg):
    """Scene configuration."""

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

@configclass
class G1ReactiveBalanceEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Reactive Balance Environment."""

    decimation = 4
    episode_length_s = 15.0

    # Balance policy outputs only leg actions
    num_actions = 12  # Only legs!
    num_observations = 72  # Balance-focused observations
    num_states = 0

    action_space = 12
    observation_space = 72
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

    scene: G1ReactiveBalanceSceneCfg = G1ReactiveBalanceSceneCfg(num_envs=1, env_spacing=2.5)

    # Locomotion
    height_command = 0.72
    gait_frequency = 1.5
    leg_action_scale = 0.4
    leg_smoothing_alpha = 0.3

    # Arm (controlled by frozen policy)
    arm_action_scale = 0.12
    arm_smoothing_alpha = 0.15
    shoulder_center_offset = [0.0, -0.174, 0.259]

    # Cylindrical workspace
    workspace_radius_min = 0.20
    workspace_radius_max = 0.45
    workspace_height_min = -0.50
    workspace_height_max = 0.35

    # ============ REWARD WEIGHTS ============
    # Balance (main focus!)
    w_com_balance = 5.0      # Keep CoM over support polygon
    w_com_velocity = 3.0     # Smooth CoM movement
    w_upright = 4.0          # Stay upright
    w_height = 2.5           # Maintain height (but allow squat)

    # Locomotion
    w_vx = 2.0
    w_vy = 1.5
    w_vyaw = 1.5
    w_gait = 2.0

    # Arm reaching (reward for reaching, but loco learns to ENABLE it)
    w_arm_reaching = 80.0    # Reaching success
    w_arm_proximity = 1.0    # Approaching target

    # Adaptive posture (squat/lean rewards)
    w_adaptive_posture = 4.0  # Reward for appropriate squat/lean

    # Penalties
    w_leg_action_rate = -0.01
    w_joint_vel = -0.002
    w_joint_acc = -0.005
    w_torque = -0.0003
    w_base_acc = -0.02       # Penalize jerky movements

    w_alive = 0.5

    # Curriculum
    min_success_rate = 0.55
    min_reaches_to_advance = 35
    min_steps_per_stage = 500

    max_reaches_per_episode = 5
    terminate_on_max_reaches = True


# ============================================================================
# ENVIRONMENT
# ============================================================================

class G1ReactiveBalanceEnv(DirectRLEnv):
    """
    G1 Reactive Balance Environment.

    The arm policy is FROZEN (from Stage 5).
    The loco policy learns to BALANCE while the arm reaches.

    Key insight: Loco policy sees:
    - Where the arm is (joint positions)
    - Where the arm is going (target)
    - Center of Mass position/velocity

    And learns to:
    - Lean forward when arm reaches forward
    - Squat when arm reaches down
    - Compensate for CoM shifts
    """

    cfg: G1ReactiveBalanceEnvCfg

    def __init__(self, cfg: G1ReactiveBalanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        self._setup_joint_indices()

        # Default positions
        self.default_leg_pos = torch.tensor(DEFAULT_LEG_POS, device=self.device)
        self.default_arm_pos = torch.tensor(DEFAULT_ARM_POS, device=self.device)

        # Arm joint limits
        self.arm_joint_lower = torch.zeros(5, device=self.device)
        self.arm_joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(RIGHT_ARM_JOINT_NAMES):
            self.arm_joint_lower[i], self.arm_joint_upper[i] = ARM_JOINT_LIMITS[jn]

        # Locomotion commands
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * self.cfg.height_command
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # Arm target (body frame, relative to shoulder)
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)

        # Gait phase
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        # Smoothed actions
        self.smoothed_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.smoothed_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_joint_vel = torch.zeros(self.num_envs, 17, device=self.device)

        # CoM tracking
        self.prev_com_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.com_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_base_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Shoulder offset
        self.shoulder_offset = torch.tensor(
            self.cfg.shoulder_center_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.total_reaches = 0
        self.total_attempts = 0
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        # Arm actions from frozen policy (will be set externally)
        self.frozen_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

        # Curriculum
        self.curriculum_level = 0
        self.stage_reaches = 0
        self.stage_attempts = 0
        self.stage_step_count = 0

        print("\n" + "=" * 70)
        print("G1 REACTIVE BALANCE ENVIRONMENT")
        print("=" * 70)
        print(f"  Leg joints: {self.leg_indices.tolist()}")
        print(f"  Arm joints: {self.arm_indices.tolist()}")
        print(f"  Palm index: {self.palm_idx}")
        print("-" * 70)
        print("  ARCHITECTURE:")
        print("    Balance Policy: TRAINABLE (12 leg actions)")
        print("    Arm Policy: FROZEN (set externally)")
        print("-" * 70)
        print(f"  Observation: {self.cfg.num_observations} dims")
        print(f"    - Base state: 9")
        print(f"    - Leg joints: 24")
        print(f"    - Commands: 10")
        print(f"    - CoM (pos+vel): 6")
        print(f"    - Arm awareness: 11")
        print(f"    - Prev actions: 12")
        print("-" * 70)
        print("  CYLINDRICAL WORKSPACE:")
        print(f"    Radius: {self.cfg.workspace_radius_min*100:.0f}-{self.cfg.workspace_radius_max*100:.0f}cm")
        print(f"    Height: {self.cfg.workspace_height_min*100:.0f} to {self.cfg.workspace_height_max*100:.0f}cm")
        print("=" * 70 + "\n")

    def _setup_joint_indices(self):
        """Setup joint indices."""
        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        self.leg_indices = torch.tensor(
            [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
            device=self.device, dtype=torch.long
        )

        self.arm_indices = torch.tensor(
            [joint_names.index(n) for n in RIGHT_ARM_JOINT_NAMES if n in joint_names],
            device=self.device, dtype=torch.long
        )

        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break
        if self.palm_idx is None:
            for i, name in enumerate(body_names):
                if "right" in name.lower() and ("hand" in name.lower() or "wrist" in name.lower()):
                    self.palm_idx = i
                    break

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    # =========================================================================
    # COMPUTE FUNCTIONS
    # =========================================================================

    def _compute_ee_pos_world(self) -> torch.Tensor:
        """Compute end-effector position in WORLD frame."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_ee_pos_body(self) -> torch.Tensor:
        """Compute end-effector position in BODY frame (relative to shoulder)."""
        ee_world = self._compute_ee_pos_world()
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        ee_rel_world = ee_world - root_pos
        ee_body = quat_apply_inverse(root_quat, ee_rel_world)

        return ee_body - self.shoulder_offset

    def _compute_target_world(self) -> torch.Tensor:
        """Convert body-frame target to world frame."""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        target_body = self.target_pos_body + self.shoulder_offset
        target_world = root_pos + rotate_vector_by_quat(target_body, root_quat)

        return target_world

    def _compute_com_body(self) -> torch.Tensor:
        """
        Compute Center of Mass in body frame.
        Approximate using body positions weighted by mass.
        """
        # Simple approximation: use root position as CoM base
        # In reality, would integrate over all body masses
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Get key body positions for CoM estimate
        # Pelvis/torso contribute most to CoM
        pelvis_pos = self.robot.data.body_pos_w[:, 0]  # Root is pelvis

        # Arm position affects CoM
        if self.palm_idx is not None:
            arm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
            # Weighted average (arm is ~5% of body mass)
            com_world = 0.95 * pelvis_pos + 0.05 * arm_pos
        else:
            com_world = pelvis_pos

        # Transform to body frame
        com_rel = com_world - root_pos
        com_body = quat_apply_inverse(root_quat, com_rel)

        return com_body

    def get_torso_euler(self) -> torch.Tensor:
        """Get torso orientation as euler angles."""
        quat = self.robot.data.root_quat_w
        return quat_to_euler_xyz(quat)

    def set_frozen_arm_actions(self, arm_actions: torch.Tensor):
        """Set arm actions from frozen policy (called by training script)."""
        self.frozen_arm_actions = arm_actions.clone()

    # =========================================================================
    # TARGET SAMPLING
    # =========================================================================

    def _sample_cylindrical_target(self, env_ids: torch.Tensor):
        """Sample targets in cylindrical workspace."""
        n = len(env_ids)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        r_min = level["arm_radius_min"]
        r_max = level["arm_radius_max"]
        r = r_min + torch.rand(n, device=self.device) * (r_max - r_min)

        theta_min = math.radians(level["arm_angle_range"][0])
        theta_max = math.radians(level["arm_angle_range"][1])
        theta = theta_min + torch.rand(n, device=self.device) * (theta_max - theta_min)

        z_min = level["arm_height_min"]
        z_max = level["arm_height_max"]
        z = z_min + torch.rand(n, device=self.device) * (z_max - z_min)

        x = -r * torch.cos(theta)
        y = r * torch.sin(theta)

        targets = torch.stack([x, y, z], dim=-1)
        self.target_pos_body[env_ids] = targets

        # Update visualization
        target_world = self._compute_target_world()[env_ids]
        target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
        pose = torch.cat([target_world, target_quat], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Initialize distance
        ee_body = self._compute_ee_pos_body()[env_ids]
        self.prev_distance[env_ids] = (ee_body - targets).norm(dim=-1)

        self.stage_attempts += len(env_ids)
        self.total_attempts += len(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample locomotion commands."""
        n = len(env_ids)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        self.vel_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (level["vx_range"][1] - level["vx_range"][0]) + level["vx_range"][0]
        self.vel_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (level["vy_range"][1] - level["vy_range"][0]) + level["vy_range"][0]
        self.vel_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (level["vyaw_range"][1] - level["vyaw_range"][0]) + level["vyaw_range"][0]

        self.torso_cmd[env_ids, 0] = 0.0
        self.torso_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (level["pitch_range"][1] - level["pitch_range"][0]) + level["pitch_range"][0]
        self.torso_cmd[env_ids, 2] = 0.0

    # =========================================================================
    # CURRICULUM
    # =========================================================================

    def update_curriculum(self, iteration: int):
        """Update curriculum based on performance."""
        self.stage_step_count += 1
        level = CURRICULUM_LEVELS[self.curriculum_level]

        success_rate = self.stage_reaches / max(self.stage_attempts, 1)

        should_advance = (
            success_rate >= self.cfg.min_success_rate and
            self.stage_reaches >= self.cfg.min_reaches_to_advance and
            self.stage_step_count >= level["min_steps"] and
            self.curriculum_level < len(CURRICULUM_LEVELS) - 1
        )

        if should_advance:
            self.curriculum_level += 1
            new_level = CURRICULUM_LEVELS[self.curriculum_level]

            print(f"\n{'='*60}")
            print(f"🎯 CURRICULUM LEVEL {self.curriculum_level + 1}/{len(CURRICULUM_LEVELS)}: {new_level['name']}")
            print(f"   Success rate: {success_rate*100:.1f}%")
            print(f"   Arm workspace: r={new_level['arm_radius_min']*100:.0f}-{new_level['arm_radius_max']*100:.0f}cm")
            print(f"   Arm height: {new_level['arm_height_min']*100:.0f} to {new_level['arm_height_max']*100:.0f}cm")
            print(f"   Pitch (lean): {new_level['pitch_range']}")
            print(f"{'='*60}\n")

            self.stage_reaches = 0
            self.stage_attempts = 0
            self.stage_step_count = 0

        return success_rate

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    def _get_observations(self) -> dict:
        """
        Build observation vector (72 dims) for BALANCE policy.

        The policy needs to see:
        1. Where is the body (velocities, orientation)
        2. Where are the legs (joint states)
        3. What commands to follow (height, vel, gait)
        4. Where is the CoM (critical for balance!)
        5. What is the arm doing (position, velocity, target)
        """
        robot = self.robot
        quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w

        # Base state (9)
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        # Leg state (24)
        leg_joint_pos = robot.data.joint_pos[:, self.leg_indices]
        leg_joint_vel = robot.data.joint_vel[:, self.leg_indices]

        # Commands (10)
        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.gait_phase),
            torch.cos(2 * np.pi * self.gait_phase)
        ], dim=-1)
        torso_euler = self.get_torso_euler()
        reserved = torch.zeros(self.num_envs, 1, device=self.device)

        # CoM (6) - Critical for balance!
        com_body = self._compute_com_body()
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.com_velocity = (com_body - self.prev_com_pos) / dt
        self.prev_com_pos = com_body.clone()

        # Arm awareness (11)
        arm_joint_pos = robot.data.joint_pos[:, self.arm_indices]
        arm_joint_vel = robot.data.joint_vel[:, self.arm_indices]
        ee_pos_body = self._compute_ee_pos_body()

        # Update EE marker
        ee_world = self._compute_ee_pos_world()
        ee_quat = robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(torch.cat([ee_world, ee_quat], dim=-1))

        # Build observation
        obs = torch.cat([
            # === BASE STATE (9) ===
            lin_vel_b,                          # 3
            ang_vel_b,                          # 3
            proj_gravity,                       # 3

            # === LEG STATE (24) ===
            leg_joint_pos,                      # 12
            leg_joint_vel * 0.1,               # 12

            # === COMMANDS (10) ===
            self.height_cmd.unsqueeze(-1),     # 1
            self.vel_cmd,                       # 3
            self.torso_cmd,                     # 3
            gait_phase,                         # 2
            reserved,                           # 1

            # === CENTER OF MASS (6) ===
            com_body,                           # 3
            self.com_velocity,                  # 3

            # === ARM AWARENESS (11) ===
            arm_joint_pos,                      # 5
            arm_joint_vel * 0.1,               # 5 → Changed to make room
            self.target_pos_body[:, 2:3],      # 1 (just height for now)

            # === PREVIOUS ACTIONS (12) ===
            self.prev_leg_actions,              # 12
        ], dim=-1)

        return {"policy": obs.clamp(-10, 10).nan_to_num()}

    # =========================================================================
    # REWARDS
    # =========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards focused on BALANCE."""
        robot = self.robot
        quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w

        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        leg_joint_pos = robot.data.joint_pos[:, self.leg_indices]
        leg_joint_vel = robot.data.joint_vel[:, self.leg_indices]

        torso_euler = self.get_torso_euler()
        level = CURRICULUM_LEVELS[self.curriculum_level]

        # CoM and EE
        com_body = self._compute_com_body()
        ee_pos_body = self._compute_ee_pos_body()
        pos_dist = (ee_pos_body - self.target_pos_body).norm(dim=-1)

        # ==================== BALANCE REWARDS (Main!) ====================

        # CoM should stay over support polygon (feet)
        # In standing, CoM x should be near 0; y near 0
        com_xy_error = com_body[:, :2].pow(2).sum(-1)
        r_com_balance = torch.exp(-5.0 * com_xy_error)

        # CoM velocity should be smooth (no jerky movements)
        com_vel_mag = self.com_velocity.pow(2).sum(-1)
        r_com_velocity = torch.exp(-3.0 * com_vel_mag)

        # Stay upright (projected gravity should be [0, 0, -1])
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        upright_error = proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2
        r_upright = torch.exp(-5.0 * upright_error)

        # Height (allow squat when target is low)
        target_height = self.target_pos_body[:, 2]  # How low is target
        # If target is low (negative), allow lower height
        dynamic_height_cmd = self.cfg.height_command + 0.3 * target_height.clamp(-0.5, 0)
        height_error = (root_pos[:, 2] - dynamic_height_cmd) ** 2
        r_height = torch.exp(-8.0 * height_error)

        # ==================== LOCOMOTION REWARDS ====================
        r_vx = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
        r_vy = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.vel_cmd[:, 1]) ** 2)
        r_vyaw = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.vel_cmd[:, 2]) ** 2)

        # Gait
        left_knee, right_knee = leg_joint_pos[:, 6], leg_joint_pos[:, 7]
        phase = self.gait_phase
        left_swing = (phase < 0.5).float()
        right_swing = (phase >= 0.5).float()
        knee_target_swing, knee_target_stance = 0.6, 0.3
        knee_err = (
            (left_knee - (left_swing * knee_target_swing + (1 - left_swing) * knee_target_stance)) ** 2 +
            (right_knee - (right_swing * knee_target_swing + (1 - right_swing) * knee_target_stance)) ** 2
        )
        r_gait = torch.exp(-3.0 * knee_err)

        # ==================== ARM REACHING (Loco enables this!) ====================
        proximity_bonus = torch.zeros_like(pos_dist)
        proximity_bonus = torch.where(pos_dist < 0.15, proximity_bonus + 1.0, proximity_bonus)
        proximity_bonus = torch.where(pos_dist < 0.10, proximity_bonus + 2.0, proximity_bonus)
        proximity_bonus = torch.where(pos_dist < 0.05, proximity_bonus + 5.0, proximity_bonus)

        reached = pos_dist < level["arm_threshold"]
        reached_ids = torch.where(reached)[0]

        if len(reached_ids) > 0:
            self.reach_count[reached_ids] += 1
            self.episode_reach_count[reached_ids] += 1
            self.stage_reaches += len(reached_ids)
            self.total_reaches += len(reached_ids)
            self._sample_cylindrical_target(reached_ids)

        # ==================== ADAPTIVE POSTURE ====================
        # Reward appropriate squat/lean based on target position
        target_z = self.target_pos_body[:, 2]
        target_x = self.target_pos_body[:, 0]

        # If target is low, pitch should be forward (negative)
        desired_pitch = target_z.clamp(-0.5, 0) * 0.5  # Lower target → more lean
        pitch_match = torch.exp(-5.0 * (torso_euler[:, 1] - desired_pitch) ** 2)

        # If target is far forward, lean forward
        forward_lean_reward = torch.where(
            target_x < -0.25,  # Target is forward
            torch.exp(-3.0 * (torso_euler[:, 1] + 0.1) ** 2),  # Encourage slight lean
            torch.ones_like(target_x)
        )

        r_adaptive_posture = 0.5 * pitch_match + 0.5 * forward_lean_reward

        # ==================== PENALTIES ====================
        leg_action_diff = self.smoothed_leg_actions - self.prev_leg_actions
        p_leg_action_rate = leg_action_diff.pow(2).sum(-1)

        p_joint_vel = leg_joint_vel.pow(2).sum(-1)

        current_vel = leg_joint_vel
        p_joint_acc = (current_vel[:, :12] - self.prev_joint_vel[:, :12]).pow(2).sum(-1)
        self.prev_joint_vel[:, :12] = current_vel.clone()

        p_torque = (leg_joint_vel.abs() * self.smoothed_leg_actions.abs()).sum(-1)

        # Base acceleration penalty
        base_acc = (lin_vel_b - self.prev_base_vel).pow(2).sum(-1)
        self.prev_base_vel = lin_vel_b.clone()

        # ==================== TOTAL ====================
        reward = (
            self.cfg.w_com_balance * r_com_balance +
            self.cfg.w_com_velocity * r_com_velocity +
            self.cfg.w_upright * r_upright +
            self.cfg.w_height * r_height +
            self.cfg.w_vx * r_vx +
            self.cfg.w_vy * r_vy +
            self.cfg.w_vyaw * r_vyaw +
            self.cfg.w_gait * r_gait +
            self.cfg.w_arm_reaching * reached.float() +
            self.cfg.w_arm_proximity * proximity_bonus +
            self.cfg.w_adaptive_posture * r_adaptive_posture +
            self.cfg.w_leg_action_rate * p_leg_action_rate +
            self.cfg.w_joint_vel * p_joint_vel +
            self.cfg.w_joint_acc * p_joint_acc +
            self.cfg.w_torque * p_torque +
            self.cfg.w_base_acc * base_acc +
            self.cfg.w_alive
        )

        self.extras = {
            "R/com_balance": r_com_balance.mean().item(),
            "R/com_velocity": r_com_velocity.mean().item(),
            "R/upright": r_upright.mean().item(),
            "R/height": r_height.mean().item(),
            "R/adaptive_posture": r_adaptive_posture.mean().item(),
            "R/vx": r_vx.mean().item(),
            "M/height": root_pos[:, 2].mean().item(),
            "M/vx": lin_vel_b[:, 0].mean().item(),
            "M/pitch": torso_euler[:, 1].mean().item(),
            "M/com_x": com_body[:, 0].mean().item(),
            "M/com_y": com_body[:, 1].mean().item(),
            "M/arm_dist": pos_dist.mean().item(),
            "curriculum_level": self.curriculum_level,
            "total_reaches": self.total_reaches,
        }

        return reward.clamp(-10, 40)

    # =========================================================================
    # TERMINATION & RESET
    # =========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination."""
        height = self.robot.data.root_pos_w[:, 2]
        quat = self.robot.data.root_quat_w

        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        fallen = (height < 0.3) | (height > 1.2)
        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

        terminated = fallen | bad_orientation

        if self.cfg.terminate_on_max_reaches:
            max_reached = self.episode_reach_count >= self.cfg.max_reaches_per_episode
            terminated = terminated | max_reached

        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)

        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        self._sample_commands(env_ids)
        self._sample_cylindrical_target(env_ids)

        self.gait_phase[env_ids] = torch.rand(n, device=self.device)

        self.smoothed_leg_actions[env_ids] = 0.0
        self.smoothed_arm_actions[env_ids] = 0.0
        self.prev_leg_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.prev_com_pos[env_ids] = 0.0
        self.com_velocity[env_ids] = 0.0
        self.prev_base_vel[env_ids] = 0.0
        self.episode_reach_count[env_ids] = 0.0
        self.frozen_arm_actions[env_ids] = 0.0

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process actions.

        actions: 12 leg actions from balance policy
        frozen_arm_actions: 5 arm actions from frozen arm policy (set externally)
        """
        self.actions = actions.clone()

        # Leg actions from balance policy
        leg_actions = actions[:, :12]

        self.prev_leg_actions = self.smoothed_leg_actions.clone()

        self.smoothed_leg_actions = (
            self.cfg.leg_smoothing_alpha * leg_actions +
            (1 - self.cfg.leg_smoothing_alpha) * self.smoothed_leg_actions
        )

        # Arm actions from frozen policy
        self.smoothed_arm_actions = (
            self.cfg.arm_smoothing_alpha * self.frozen_arm_actions +
            (1 - self.cfg.arm_smoothing_alpha) * self.smoothed_arm_actions
        )

        # Apply to robot
        target_pos = self.robot.data.default_joint_pos.clone()

        target_pos[:, self.leg_indices] = self.default_leg_pos + self.smoothed_leg_actions * self.cfg.leg_action_scale

        arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
        arm_target = torch.clamp(
            arm_pos + self.smoothed_arm_actions * self.cfg.arm_action_scale,
            self.arm_joint_lower,
            self.arm_joint_upper
        )
        target_pos[:, self.arm_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)

        self.gait_phase = (self.gait_phase + self.cfg.gait_frequency * self.cfg.sim.dt * self.cfg.decimation) % 1.0

    def _apply_action(self):
        pass

    def get_success_rate(self) -> float:
        if self.total_attempts > 0:
            return self.total_reaches / self.total_attempts
        return 0.0


# Aliases
G1BalanceEnv = G1ReactiveBalanceEnv
G1BalanceEnvCfg = G1ReactiveBalanceEnvCfg