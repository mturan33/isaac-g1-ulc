"""
G1 Full Loco-Manipulation Environment
======================================

GRIPPER-READY Architecture:
- 12 Leg actions (balance/locomotion)
- 5 Arm actions (shoulder + elbow)
- 3 Gripper actions (3-finger dex1)

OBSERVATION includes:
- Body state, leg joints, commands
- Arm joints + gripper joints
- Target position + object position
- Contact states (for grasp verification)

REWARD includes:
- Balance rewards (CoM, upright)
- Locomotion rewards (vel tracking)
- Reaching rewards (arm to target)
- Grasp rewards (contact + stable hold)
- Manipulation rewards (object movement)

CURRICULUM:
1. Standing + reach (no grasp)
2. Standing + reach + grasp
3. Walk + reach + grasp
4. Squat + reach + grasp (low targets)
5. Full loco-manipulation

CONTACT-BASED GRASP VERIFICATION:
- Check if gripper fingers contact object
- Verify object is between fingers (not just touching)
- Monitor grasp stability over time
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
from isaaclab.sensors import ContactSensorCfg

# ============================================================================
# CONSTANTS
# ============================================================================

# G1 29 DoF with Dex1 gripper
# Standard G1 USD (we'll use base and add gripper separately if needed)
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Leg joints (12)
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

# Right arm joints (5)
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# Right gripper joints (3) - Dex1 style
# Note: These names may need adjustment based on actual USD
RIGHT_GRIPPER_JOINT_NAMES = [
    "right_one_joint",  # Finger 1
    "right_two_joint",  # Finger 2
    "right_three_joint",  # Finger 3 (thumb)
]

# Default positions
DEFAULT_LEG_POS = [-0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, -0.2, -0.2, 0.0, 0.0]
DEFAULT_ARM_POS = [-0.3, 0.0, 0.0, 0.5, 0.0]
DEFAULT_GRIPPER_POS = [0.0, 0.0, 0.0]  # Open gripper

# Joint limits
ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

# Gripper limits (estimated, adjust based on actual robot)
GRIPPER_JOINT_LIMITS = {
    "right_one_joint": (-0.5, 1.5),
    "right_two_joint": (-0.5, 1.5),
    "right_three_joint": (-0.5, 1.5),
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
    # Level 0: Standing + basic reach (no grasp)
    {
        "name": "standing_reach",
        "vx_range": (0.0, 0.0),
        "pitch_range": (0.0, 0.0),
        "arm_radius_range": (0.20, 0.28),
        "arm_height_range": (-0.05, 0.15),
        "grasp_enabled": False,
        "threshold": 0.12,
    },
    # Level 1: Standing + reach + grasp
    {
        "name": "standing_grasp",
        "vx_range": (0.0, 0.0),
        "pitch_range": (0.0, 0.0),
        "arm_radius_range": (0.20, 0.30),
        "arm_height_range": (-0.10, 0.18),
        "grasp_enabled": True,
        "threshold": 0.10,
    },
    # Level 2: Walk + reach + grasp
    {
        "name": "walk_grasp",
        "vx_range": (0.0, 0.4),
        "pitch_range": (0.0, 0.0),
        "arm_radius_range": (0.20, 0.35),
        "arm_height_range": (-0.20, 0.22),
        "grasp_enabled": True,
        "threshold": 0.09,
    },
    # Level 3: Squat + reach + grasp (low targets)
    {
        "name": "squat_grasp",
        "vx_range": (0.0, 0.3),
        "pitch_range": (-0.25, 0.0),
        "arm_radius_range": (0.20, 0.40),
        "arm_height_range": (-0.40, 0.20),
        "grasp_enabled": True,
        "threshold": 0.08,
    },
    # Level 4: Full loco-manipulation
    {
        "name": "full_locomanip",
        "vx_range": (-0.2, 0.6),
        "pitch_range": (-0.35, 0.0),
        "arm_radius_range": (0.20, 0.45),
        "arm_height_range": (-0.50, 0.30),
        "grasp_enabled": True,
        "threshold": 0.07,
    },
]


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1LocoManipSceneCfg(InteractiveSceneCfg):
    """Scene configuration with robot, target, and graspable object."""

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
            activate_contact_sensors=True,  # Enable for grasp detection
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
            # Gripper actuator (will be active if gripper exists)
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["right_one.*", "right_two.*", "right_three.*"],
                stiffness=50.0,
                damping=5.0,
            ),
        },
    )

    # Target marker (where to reach)
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

    # Graspable object (cylinder)
    grasp_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspObject",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g object
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red cylinder
                emissive_color=(0.3, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.2, 1.0)),
    )


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

@configclass
class G1LocoManipEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 Loco-Manipulation Environment."""

    decimation = 4
    episode_length_s = 20.0

    # Actions: 12 legs + 5 arm + 3 gripper = 20
    num_actions = 20

    # Observations:
    # Base (9) + Leg (24) + Commands (10) + CoM (6) +
    # Arm (10) + Gripper (6) + Target (3) + Object (6) +
    # Contact (3) + Prev (20) = 97
    num_observations = 97
    num_states = 0

    action_space = 20
    observation_space = 97
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2 ** 21,
            gpu_total_aggregate_pairs_capacity=2 ** 21,
        ),
    )

    scene: G1LocoManipSceneCfg = G1LocoManipSceneCfg(num_envs=1, env_spacing=2.5)

    # Locomotion
    height_command = 0.72
    gait_frequency = 1.5
    leg_action_scale = 0.4
    arm_action_scale = 0.12
    gripper_action_scale = 0.3

    smoothing_alpha = 0.25

    # Workspace
    shoulder_offset = [0.0, -0.174, 0.259]
    workspace_radius_min = 0.20
    workspace_radius_max = 0.45

    # ============ REWARD WEIGHTS ============
    # Balance
    w_com_balance = 5.0
    w_upright = 4.0
    w_height = 2.5

    # Locomotion
    w_vx = 2.0
    w_vy = 1.5
    w_vyaw = 1.5
    w_gait = 2.0

    # Reaching
    w_reaching = 60.0
    w_proximity = 2.0

    # Grasping
    w_grasp_approach = 3.0  # Bonus for approaching with open gripper
    w_grasp_contact = 20.0  # Bonus for finger contact with object
    w_grasp_stable = 50.0  # Bonus for stable grasp (all fingers + object held)
    w_grasp_lift = 100.0  # Bonus for lifting object

    # Penalties
    w_action_rate = -0.01
    w_joint_vel = -0.002
    w_torque = -0.0003
    w_drop_penalty = -50.0  # Penalty for dropping object

    w_alive = 0.5

    # Grasp parameters
    grasp_contact_threshold = 0.02  # Distance for contact detection
    grasp_stable_time = 0.5  # Seconds to hold for stable grasp
    lift_height_threshold = 0.05  # How much to lift for success


# ============================================================================
# ENVIRONMENT
# ============================================================================

class G1LocoManipEnv(DirectRLEnv):
    """
    G1 Full Loco-Manipulation Environment.

    Features:
    - Dual trainable arm + balance policies
    - Gripper control for grasping
    - Contact-based grasp verification
    - Object manipulation
    """

    cfg: G1LocoManipEnvCfg

    def __init__(self, cfg: G1LocoManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]
        self.grasp_object = self.scene["grasp_object"]

        self._setup_joint_indices()

        # Default positions
        self.default_leg_pos = torch.tensor(DEFAULT_LEG_POS, device=self.device)
        self.default_arm_pos = torch.tensor(DEFAULT_ARM_POS, device=self.device)
        self.default_gripper_pos = torch.tensor(DEFAULT_GRIPPER_POS, device=self.device)

        # Check if gripper exists
        self.has_gripper = len(self.gripper_indices) > 0
        print(f"[INFO] Gripper detected: {self.has_gripper} ({len(self.gripper_indices)} joints)")

        # Locomotion commands
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * self.cfg.height_command
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # Target and object
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_initial_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Gait phase
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        # Smoothed actions
        self.smoothed_actions = torch.zeros(self.num_envs, 20, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 20, device=self.device)

        # CoM tracking
        self.prev_com_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.com_velocity = torch.zeros(self.num_envs, 3, device=self.device)

        # Shoulder offset
        self.shoulder_offset = torch.tensor(
            self.cfg.shoulder_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # ============ GRASP STATE TRACKING ============
        self.grasp_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 0: Approaching, 1: Contacting, 2: Grasping, 3: Lifting

        self.grasp_contact_count = torch.zeros(self.num_envs, 3, device=self.device)  # Per finger
        self.grasp_hold_time = torch.zeros(self.num_envs, device=self.device)
        self.object_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)
        self.grasp_count = torch.zeros(self.num_envs, device=self.device)
        self.total_reaches = 0
        self.total_grasps = 0

        # Curriculum
        self.curriculum_level = 0
        self.stage_successes = 0
        self.stage_attempts = 0

        print("\n" + "=" * 70)
        print("G1 LOCO-MANIPULATION ENVIRONMENT")
        print("=" * 70)
        print(f"  Leg joints: {self.leg_indices.tolist()}")
        print(f"  Arm joints: {self.arm_indices.tolist()}")
        print(f"  Gripper joints: {self.gripper_indices.tolist() if self.has_gripper else 'N/A'}")
        print(f"  Palm index: {self.palm_idx}")
        print("-" * 70)
        print(f"  Actions: {self.cfg.num_actions} (12 leg + 5 arm + 3 gripper)")
        print(f"  Observations: {self.cfg.num_observations}")
        print("-" * 70)
        print("  GRASP REWARDS:")
        print(f"    Contact: +{self.cfg.w_grasp_contact}")
        print(f"    Stable:  +{self.cfg.w_grasp_stable}")
        print(f"    Lift:    +{self.cfg.w_grasp_lift}")
        print("=" * 70 + "\n")

    def _setup_joint_indices(self):
        """Setup joint indices for legs, arm, and gripper."""
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

        # Gripper indices (may not exist in base G1)
        gripper_idx_list = []
        for n in RIGHT_GRIPPER_JOINT_NAMES:
            if n in joint_names:
                gripper_idx_list.append(joint_names.index(n))
        self.gripper_indices = torch.tensor(
            gripper_idx_list, device=self.device, dtype=torch.long
        )

        # Palm index for EE
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

        # Finger body indices for contact detection
        self.finger_body_indices = []
        for i, name in enumerate(body_names):
            if "right" in name.lower() and any(x in name.lower() for x in ["finger", "one", "two", "three"]):
                self.finger_body_indices.append(i)

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]
        self.grasp_object = self.scene["grasp_object"]

    # =========================================================================
    # COMPUTE FUNCTIONS
    # =========================================================================

    def _compute_ee_pos_world(self) -> torch.Tensor:
        """Compute end-effector position in world frame."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_ee_pos_body(self) -> torch.Tensor:
        """Compute EE position in body frame (relative to shoulder)."""
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
        """Compute Center of Mass in body frame."""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        pelvis_pos = self.robot.data.body_pos_w[:, 0]

        if self.palm_idx is not None:
            arm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
            com_world = 0.95 * pelvis_pos + 0.05 * arm_pos
        else:
            com_world = pelvis_pos

        com_rel = com_world - root_pos
        com_body = quat_apply_inverse(root_quat, com_rel)

        return com_body

    def _compute_finger_object_contacts(self) -> torch.Tensor:
        """
        Compute contact between gripper fingers and object.

        Returns: [num_envs, 3] tensor with 1.0 if finger i contacts object

        CONTACT-BASED GRASP VERIFICATION:
        - Each finger has collision bodies
        - Check if finger collision intersects with object
        - True grasp = all 3 fingers contacting object
        """
        contacts = torch.zeros(self.num_envs, 3, device=self.device)

        # Get object position
        object_pos = self.grasp_object.data.root_pos_w

        # Get EE position (palm)
        ee_pos = self._compute_ee_pos_world()

        # Simple distance-based contact (replace with actual contact sensor if available)
        # Check distance from EE to object
        ee_to_object = object_pos - ee_pos
        distance = ee_to_object.norm(dim=-1)

        # Get gripper state (how closed are the fingers)
        if self.has_gripper:
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_indices]
            gripper_closed = (gripper_pos > 0.5).float()  # Threshold for "closed"

            # Contact = close enough AND gripper closing
            in_range = (distance < self.cfg.grasp_contact_threshold + 0.05).unsqueeze(-1)
            contacts = gripper_closed * in_range
        else:
            # Without gripper, use distance-based heuristic
            in_range = distance < self.cfg.grasp_contact_threshold
            contacts[:, :] = in_range.unsqueeze(-1)

        return contacts

    def _check_grasp_stable(self) -> torch.Tensor:
        """
        Check if grasp is stable (object held firmly).

        STABLE GRASP CRITERIA:
        1. Multiple fingers contacting object
        2. Object not moving relative to gripper
        3. Held for minimum time
        """
        contacts = self._compute_finger_object_contacts()
        num_contacts = contacts.sum(dim=-1)

        # Need at least 2 fingers for stable grasp
        enough_contacts = num_contacts >= 2

        # Check object velocity (should be low if held)
        object_vel = self.grasp_object.data.root_lin_vel_w
        object_speed = object_vel.norm(dim=-1)
        low_velocity = object_speed < 0.1

        # Update hold time
        stable_now = enough_contacts & low_velocity
        self.grasp_hold_time = torch.where(
            stable_now,
            self.grasp_hold_time + self.cfg.sim.dt * self.cfg.decimation,
            torch.zeros_like(self.grasp_hold_time)
        )

        # Stable if held for minimum time
        is_stable = self.grasp_hold_time >= self.cfg.grasp_stable_time

        return is_stable

    def _check_object_lifted(self) -> torch.Tensor:
        """Check if object has been lifted above initial position."""
        object_pos = self.grasp_object.data.root_pos_w
        height_gain = object_pos[:, 2] - self.object_initial_pos[:, 2]
        return height_gain > self.cfg.lift_height_threshold

    # =========================================================================
    # TARGET AND OBJECT SAMPLING
    # =========================================================================

    def _sample_target_and_object(self, env_ids: torch.Tensor):
        """Sample target position and place object there."""
        n = len(env_ids)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        # Cylindrical sampling for target
        r_range = level["arm_radius_range"]
        r = r_range[0] + torch.rand(n, device=self.device) * (r_range[1] - r_range[0])
        theta = torch.rand(n, device=self.device) * 2 * np.pi - np.pi

        h_range = level["arm_height_range"]
        z = h_range[0] + torch.rand(n, device=self.device) * (h_range[1] - h_range[0])

        x = -r * torch.cos(theta)
        y = r * torch.sin(theta)

        targets = torch.stack([x, y, z], dim=-1)
        self.target_pos_body[env_ids] = targets

        # Update target visualization
        target_world = self._compute_target_world()[env_ids]
        target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
        self.target_obj.write_root_pose_to_sim(torch.cat([target_world, target_quat], dim=-1), env_ids=env_ids)

        # Place object at target (if grasp enabled)
        if level["grasp_enabled"]:
            # Object slightly above target
            object_pos = target_world.clone()
            object_pos[:, 2] += 0.04  # Object center above target point
            object_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
            self.grasp_object.write_root_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1), env_ids=env_ids)
            self.grasp_object.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
            self.object_initial_pos[env_ids] = object_pos
        else:
            # Move object far away
            far_pos = torch.tensor([[10.0, 10.0, 0.5]], device=self.device).expand(n, -1)
            far_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
            self.grasp_object.write_root_pose_to_sim(torch.cat([far_pos, far_quat], dim=-1), env_ids=env_ids)

        self.stage_attempts += len(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample locomotion commands based on curriculum."""
        n = len(env_ids)
        level = CURRICULUM_LEVELS[self.curriculum_level]

        vx_range = level["vx_range"]
        self.vel_cmd[env_ids, 0] = vx_range[0] + torch.rand(n, device=self.device) * (vx_range[1] - vx_range[0])
        self.vel_cmd[env_ids, 1] = torch.rand(n, device=self.device) * 0.2 - 0.1
        self.vel_cmd[env_ids, 2] = torch.rand(n, device=self.device) * 0.4 - 0.2

        pitch_range = level["pitch_range"]
        self.torso_cmd[env_ids, 1] = pitch_range[0] + torch.rand(n, device=self.device) * (
                    pitch_range[1] - pitch_range[0])

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    def _get_observations(self) -> dict:
        """
        Build observation vector (97 dims).

        Layout:
        [0-8]    Base state (9): lin_vel, ang_vel, proj_gravity
        [9-32]   Leg state (24): joint_pos, joint_vel
        [33-42]  Commands (10): height, vel, torso, gait
        [43-48]  CoM (6): position, velocity
        [49-58]  Arm state (10): joint_pos, joint_vel
        [59-64]  Gripper state (6): joint_pos, joint_vel
        [65-67]  Target position (3): body frame
        [68-73]  Object state (6): position, velocity (body frame)
        [74-76]  Contact state (3): per-finger contact
        [77-96]  Previous actions (20)
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
        reserved = torch.zeros(self.num_envs, 1, device=self.device)

        # CoM (6)
        com_body = self._compute_com_body()
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.com_velocity = (com_body - self.prev_com_pos) / dt
        self.prev_com_pos = com_body.clone()

        # Arm state (10)
        arm_joint_pos = robot.data.joint_pos[:, self.arm_indices]
        arm_joint_vel = robot.data.joint_vel[:, self.arm_indices]

        # Gripper state (6)
        if self.has_gripper:
            gripper_joint_pos = robot.data.joint_pos[:, self.gripper_indices]
            gripper_joint_vel = robot.data.joint_vel[:, self.gripper_indices]
        else:
            gripper_joint_pos = torch.zeros(self.num_envs, 3, device=self.device)
            gripper_joint_vel = torch.zeros(self.num_envs, 3, device=self.device)

        # Target position (3) - body frame
        target_body = self.target_pos_body

        # Object state (6) - body frame
        object_pos_world = self.grasp_object.data.root_pos_w
        object_vel_world = self.grasp_object.data.root_lin_vel_w
        object_rel_world = object_pos_world - root_pos
        object_pos_body = quat_apply_inverse(quat, object_rel_world) - self.shoulder_offset
        object_vel_body = quat_apply_inverse(quat, object_vel_world)

        # Contact state (3)
        contacts = self._compute_finger_object_contacts()

        # Update EE marker
        ee_world = self._compute_ee_pos_world()
        ee_quat = robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(torch.cat([ee_world, ee_quat], dim=-1))

        # Build observation
        obs = torch.cat([
            # Base state (9)
            lin_vel_b,  # 3
            ang_vel_b,  # 3
            proj_gravity,  # 3

            # Leg state (24)
            leg_joint_pos,  # 12
            leg_joint_vel * 0.1,  # 12

            # Commands (10)
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,  # 3
            self.torso_cmd,  # 3
            gait_phase,  # 2
            reserved,  # 1

            # CoM (6)
            com_body,  # 3
            self.com_velocity,  # 3

            # Arm state (10)
            arm_joint_pos,  # 5
            arm_joint_vel * 0.1,  # 5

            # Gripper state (6)
            gripper_joint_pos,  # 3
            gripper_joint_vel * 0.1,  # 3

            # Target (3)
            target_body,  # 3

            # Object state (6)
            object_pos_body,  # 3
            object_vel_body * 0.1,  # 3

            # Contact state (3)
            contacts,  # 3

            # Previous actions (20)
            self.prev_actions,  # 20
        ], dim=-1)

        return {"policy": obs.clamp(-10, 10).nan_to_num()}

    # =========================================================================
    # REWARDS
    # =========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards including grasp rewards."""
        robot = self.robot
        quat = robot.data.root_quat_w
        root_pos = robot.data.root_pos_w
        level = CURRICULUM_LEVELS[self.curriculum_level]

        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # ==================== BALANCE REWARDS ====================
        com_body = self._compute_com_body()
        com_xy_error = com_body[:, :2].pow(2).sum(-1)
        r_com_balance = torch.exp(-5.0 * com_xy_error)

        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        upright_error = proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2
        r_upright = torch.exp(-5.0 * upright_error)

        height_error = (root_pos[:, 2] - self.cfg.height_command) ** 2
        r_height = torch.exp(-8.0 * height_error)

        # ==================== LOCOMOTION REWARDS ====================
        r_vx = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
        r_vy = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.vel_cmd[:, 1]) ** 2)
        r_vyaw = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.vel_cmd[:, 2]) ** 2)

        leg_joint_pos = robot.data.joint_pos[:, self.leg_indices]
        left_knee, right_knee = leg_joint_pos[:, 6], leg_joint_pos[:, 7]
        phase = self.gait_phase
        knee_err = (
                (left_knee - (0.6 * (phase < 0.5).float() + 0.3 * (phase >= 0.5).float())) ** 2 +
                (right_knee - (0.6 * (phase >= 0.5).float() + 0.3 * (phase < 0.5).float())) ** 2
        )
        r_gait = torch.exp(-3.0 * knee_err)

        # ==================== REACHING REWARDS ====================
        ee_body = self._compute_ee_pos_body()
        pos_dist = (ee_body - self.target_pos_body).norm(dim=-1)

        proximity_bonus = torch.zeros_like(pos_dist)
        proximity_bonus = torch.where(pos_dist < 0.15, proximity_bonus + 1.0, proximity_bonus)
        proximity_bonus = torch.where(pos_dist < 0.10, proximity_bonus + 2.0, proximity_bonus)
        proximity_bonus = torch.where(pos_dist < 0.05, proximity_bonus + 3.0, proximity_bonus)

        reached = pos_dist < level["threshold"]

        # ==================== GRASP REWARDS ====================
        r_grasp_approach = torch.zeros_like(pos_dist)
        r_grasp_contact = torch.zeros_like(pos_dist)
        r_grasp_stable = torch.zeros_like(pos_dist)
        r_grasp_lift = torch.zeros_like(pos_dist)

        if level["grasp_enabled"]:
            contacts = self._compute_finger_object_contacts()
            num_contacts = contacts.sum(dim=-1)

            # Approach bonus: gripper open + near object
            if self.has_gripper:
                gripper_pos = robot.data.joint_pos[:, self.gripper_indices]
                gripper_open = (gripper_pos < 0.3).all(dim=-1).float()
            else:
                gripper_open = torch.ones(self.num_envs, device=self.device)
            r_grasp_approach = gripper_open * (pos_dist < 0.10).float()

            # Contact bonus
            r_grasp_contact = num_contacts / 3.0  # 0 to 1

            # Stable grasp bonus
            is_stable = self._check_grasp_stable()
            r_grasp_stable = is_stable.float()

            # Lift bonus
            is_lifted = self._check_object_lifted()
            r_grasp_lift = is_lifted.float()

            # Update grasp count
            newly_grasped = is_stable & ~self.object_lifted
            self.grasp_count[newly_grasped] += 1
            self.total_grasps += newly_grasped.sum().item()
            self.object_lifted = self.object_lifted | is_lifted

        # Update reach count
        reach_ids = torch.where(reached)[0]
        if len(reach_ids) > 0 and not level["grasp_enabled"]:
            self.reach_count[reach_ids] += 1
            self.total_reaches += len(reach_ids)
            self.stage_successes += len(reach_ids)
            self._sample_target_and_object(reach_ids)

        # ==================== PENALTIES ====================
        action_diff = self.smoothed_actions - self.prev_actions
        p_action_rate = action_diff.pow(2).sum(-1)

        leg_joint_vel = robot.data.joint_vel[:, self.leg_indices]
        p_joint_vel = leg_joint_vel.pow(2).sum(-1)

        p_torque = (leg_joint_vel.abs() * self.smoothed_actions[:, :12].abs()).sum(-1)

        # Drop penalty
        p_drop = torch.zeros_like(pos_dist)
        if level["grasp_enabled"]:
            object_pos = self.grasp_object.data.root_pos_w
            object_dropped = object_pos[:, 2] < 0.1  # Object fell to ground
            p_drop = object_dropped.float()

        # ==================== TOTAL ====================
        reward = (
                self.cfg.w_com_balance * r_com_balance +
                self.cfg.w_upright * r_upright +
                self.cfg.w_height * r_height +
                self.cfg.w_vx * r_vx +
                self.cfg.w_vy * r_vy +
                self.cfg.w_vyaw * r_vyaw +
                self.cfg.w_gait * r_gait +
                self.cfg.w_reaching * reached.float() +
                self.cfg.w_proximity * proximity_bonus +
                self.cfg.w_grasp_approach * r_grasp_approach +
                self.cfg.w_grasp_contact * r_grasp_contact +
                self.cfg.w_grasp_stable * r_grasp_stable +
                self.cfg.w_grasp_lift * r_grasp_lift +
                self.cfg.w_action_rate * p_action_rate +
                self.cfg.w_joint_vel * p_joint_vel +
                self.cfg.w_torque * p_torque +
                self.cfg.w_drop_penalty * p_drop +
                self.cfg.w_alive
        )

        self.extras = {
            "R/balance": r_com_balance.mean().item(),
            "R/upright": r_upright.mean().item(),
            "R/grasp_contact": r_grasp_contact.mean().item(),
            "R/grasp_stable": r_grasp_stable.mean().item(),
            "M/height": root_pos[:, 2].mean().item(),
            "M/arm_dist": pos_dist.mean().item(),
            "M/num_contacts": self._compute_finger_object_contacts().sum(-1).mean().item() if level[
                "grasp_enabled"] else 0,
            "curriculum_level": self.curriculum_level,
            "total_reaches": self.total_reaches,
            "total_grasps": self.total_grasps,
        }

        return reward.clamp(-20, 100)

    # =========================================================================
    # TERMINATION & RESET
    # =========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        height = self.robot.data.root_pos_w[:, 2]
        quat = self.robot.data.root_quat_w

        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        fallen = (height < 0.3) | (height > 1.2)
        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

        terminated = fallen | bad_orientation
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)

        # Reset robot
        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        # Reset commands and targets
        self._sample_commands(env_ids)
        self._sample_target_and_object(env_ids)

        # Reset states
        self.gait_phase[env_ids] = torch.rand(n, device=self.device)
        self.smoothed_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_com_pos[env_ids] = 0.0
        self.com_velocity[env_ids] = 0.0

        # Reset grasp tracking
        self.grasp_state[env_ids] = 0
        self.grasp_hold_time[env_ids] = 0.0
        self.object_lifted[env_ids] = False
        self.object_dropped[env_ids] = False

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process actions.

        actions layout:
        [0:12]  Leg actions
        [12:17] Arm actions
        [17:20] Gripper actions
        """
        self.actions = actions.clone()
        self.prev_actions = self.smoothed_actions.clone()

        self.smoothed_actions = (
                self.cfg.smoothing_alpha * actions +
                (1 - self.cfg.smoothing_alpha) * self.smoothed_actions
        )

        target_pos = self.robot.data.default_joint_pos.clone()

        # Leg actions
        leg_actions = self.smoothed_actions[:, :12]
        target_pos[:, self.leg_indices] = self.default_leg_pos + leg_actions * self.cfg.leg_action_scale

        # Arm actions
        arm_actions = self.smoothed_actions[:, 12:17]
        arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
        arm_target = arm_pos + arm_actions * self.cfg.arm_action_scale
        # Clamp to limits
        for i, jn in enumerate(RIGHT_ARM_JOINT_NAMES):
            low, high = ARM_JOINT_LIMITS[jn]
            arm_target[:, i] = arm_target[:, i].clamp(low, high)
        target_pos[:, self.arm_indices] = arm_target

        # Gripper actions
        if self.has_gripper:
            gripper_actions = self.smoothed_actions[:, 17:20]
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_indices]
            gripper_target = gripper_pos + gripper_actions * self.cfg.gripper_action_scale
            # Clamp
            for i, jn in enumerate(RIGHT_GRIPPER_JOINT_NAMES):
                if jn in GRIPPER_JOINT_LIMITS:
                    low, high = GRIPPER_JOINT_LIMITS[jn]
                    gripper_target[:, i] = gripper_target[:, i].clamp(low, high)
            target_pos[:, self.gripper_indices] = gripper_target

        self.robot.set_joint_position_target(target_pos)

        # Update gait phase
        self.gait_phase = (self.gait_phase + self.cfg.gait_frequency * self.cfg.sim.dt * self.cfg.decimation) % 1.0

    def _apply_action(self):
        pass

    # =========================================================================
    # CURRICULUM
    # =========================================================================

    def update_curriculum(self, iteration: int):
        """Update curriculum based on performance."""
        if self.stage_attempts < 100:
            return 0.0

        success_rate = self.stage_successes / max(self.stage_attempts, 1)

        should_advance = (
                success_rate >= 0.5 and
                self.stage_successes >= 30 and
                self.curriculum_level < len(CURRICULUM_LEVELS) - 1
        )

        if should_advance:
            self.curriculum_level += 1
            level = CURRICULUM_LEVELS[self.curriculum_level]

            print(f"\n{'=' * 60}")
            print(f"🎯 CURRICULUM LEVEL {self.curriculum_level + 1}/{len(CURRICULUM_LEVELS)}: {level['name']}")
            print(f"   Success rate: {success_rate * 100:.1f}%")
            print(f"   Grasp enabled: {level['grasp_enabled']}")
            print(f"{'=' * 60}\n")

            self.stage_successes = 0
            self.stage_attempts = 0

        return success_rate


# Aliases
G1GraspEnv = G1LocoManipEnv
G1GraspEnvCfg = G1LocoManipEnvCfg