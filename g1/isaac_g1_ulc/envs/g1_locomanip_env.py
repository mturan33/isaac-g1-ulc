"""
G1 29DoF + Dex1 Stage 6 Loco-Manipulation Environment V3
=========================================================

WORKSPACE: Yarım silindir (sağ omuz merkezli)
- Yarıçap: 40cm (sabit)
- Yükseklik: Omuzdan yere kadar (dinamik)
- Açı: Sadece öne bakan yarım (-90° to +90°)
- Ground clipping: Yerin altına hedef oluşturmaz

CURRICULUM: Yukarıdan aşağıya genişleme
- Stage 0: Omuz ±15cm (ayakta reach)
- Stage 1: Omuz -30cm (eğilme başlangıcı)
- Stage 2: Omuz -50cm (derin eğilme)
- Stage 3: Yere kadar (çömelme)

Based on working g1_arm_dual_orient_env.py

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

# USD - Use cloud path (working), local Dex1 path as backup
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"
# G1_DEX1_USD_PATH = "C:/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd"

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

# Dex1 gripper (will be empty if using basic G1)
G1_RIGHT_GRIPPER_JOINTS = ["right_hand_Joint1_1", "right_hand_Joint2_1"]

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
def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz format)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


# ==============================================================================
# CURRICULUM LEVELS - Yukarıdan Aşağıya
# ==============================================================================
@dataclass
class CurriculumLevel:
    """Curriculum level configuration."""
    name: str
    # Workspace height range (relative to shoulder, negative = below)
    height_min: float  # Minimum height (negative = below shoulder)
    height_max: float  # Maximum height (positive = above shoulder)
    # Locomotion
    vx_range: Tuple[float, float] = (0.0, 0.0)
    vy_range: Tuple[float, float] = (0.0, 0.0)
    omega_range: Tuple[float, float] = (0.0, 0.0)
    # Settings
    grasp_enabled: bool = False
    reach_threshold: float = 0.10
    min_reaches_to_advance: int = 50
    success_rate_threshold: float = 0.6


CURRICULUM_LEVELS = [
    # Stage 0: Omuz seviyesi ±15cm (ayakta kolay reach)
    CurriculumLevel(
        name="shoulder_level",
        height_min=-0.15,
        height_max=0.20,
        grasp_enabled=False,
        reach_threshold=0.10,
        min_reaches_to_advance=60,
        success_rate_threshold=0.65,
    ),
    # Stage 1: Omuz -30cm (hafif eğilme)
    CurriculumLevel(
        name="upper_body_lean",
        height_min=-0.30,
        height_max=0.25,
        vx_range=(0.0, 0.15),
        grasp_enabled=False,
        reach_threshold=0.10,
        min_reaches_to_advance=50,
        success_rate_threshold=0.55,
    ),
    # Stage 2: Omuz -50cm (derin eğilme)
    CurriculumLevel(
        name="deep_lean",
        height_min=-0.50,
        height_max=0.30,
        vx_range=(0.0, 0.25),
        grasp_enabled=True,
        reach_threshold=0.12,
        min_reaches_to_advance=40,
        success_rate_threshold=0.50,
    ),
    # Stage 3: Yere kadar (çömelme)
    CurriculumLevel(
        name="squat_reach",
        height_min=-0.70,  # Omuzdan ~70cm aşağı = yer seviyesi
        height_max=0.30,
        vx_range=(0.0, 0.30),
        grasp_enabled=True,
        reach_threshold=0.12,
        min_reaches_to_advance=35,
        success_rate_threshold=0.45,
    ),
    # Stage 4: Tam loco-manipulation
    CurriculumLevel(
        name="full_locomanip",
        height_min=-0.70,
        height_max=0.35,
        vx_range=(-0.2, 0.4),
        vy_range=(-0.15, 0.15),
        omega_range=(-0.3, 0.3),
        grasp_enabled=True,
        reach_threshold=0.12,
        min_reaches_to_advance=30,
        success_rate_threshold=0.40,
    ),
]


# ==============================================================================
# SCENE CONFIGURATION
# ==============================================================================
@configclass
class G1LocoManipSceneCfg(InteractiveSceneCfg):
    """Scene configuration - identical structure to working env."""

    # Ground - same as working env
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # Locomotion needs gravity
                linear_damping=0.1,
                angular_damping=0.1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos={
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
            "body": ImplicitActuatorCfg(
                joint_names_expr=["torso.*"],
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
        },
    )

    # Target sphere
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

    # Shoulder marker
    shoulder_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShoulderMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                emissive_color=(0.5, 0.5, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.2, 1.1)),
    )


# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================
@configclass
class G1Dex1Stage6EnvCfg(DirectRLEnvCfg):
    """Environment configuration."""

    decimation = 4
    episode_length_s = 12.0

    # 12 legs + 5 arm + 2 gripper = 19 (gripper optional)
    num_actions = 17  # Start without gripper
    num_observations = 65
    num_states = 0

    action_space = 17
    observation_space = 65
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    scene: G1LocoManipSceneCfg = G1LocoManipSceneCfg(num_envs=4096, env_spacing=2.5)

    # Action scaling
    action_scale_legs = 0.25
    action_scale_arm = 0.12
    action_scale_gripper = 0.5
    action_smoothing_alpha = 0.2

    # Workspace - Yarım silindir
    workspace_radius = 0.40  # 40cm sabit yarıçap
    workspace_azimuth_min = -1.57  # -90° (sağ)
    workspace_azimuth_max = 1.57   # +90° (sol) - sadece öne bakan yarım
    ground_clearance = 0.05  # Yerden 5cm yukarıda kal

    # Shoulder offset (relative to base)
    shoulder_offset_x = 0.0
    shoulder_offset_y = -0.2  # Sağ omuz
    shoulder_offset_z = 0.35  # Omuz yüksekliği (base'den)

    # Curriculum
    curriculum_level = 0
    auto_curriculum = True
    min_steps_per_stage = 250

    # Rewards - Locomotion
    reward_lin_vel_tracking = 1.2
    reward_ang_vel_tracking = 0.6
    reward_base_height = 0.8
    reward_orientation = 1.5

    # Rewards - Arm reaching
    use_potential_shaping = True
    potential_gamma = 0.99
    potential_sigma = 0.15
    potential_scale = 3.0

    reward_pos_tanh_std = 0.12
    reward_pos_tanh_weight = 2.0
    reward_reaching = 100.0

    # Proximity zones
    proximity_zone1_dist = 0.15
    proximity_zone1_bonus = 1.0
    proximity_zone2_dist = 0.10
    proximity_zone2_bonus = 2.0
    proximity_zone3_dist = 0.05
    proximity_zone3_bonus = 5.0

    # Regularization
    reward_action_rate = -0.015
    reward_joint_vel = -0.002
    reward_joint_acc = -0.005

    # Termination
    fall_height_threshold = 0.35
    max_reaches_per_episode = 5
    target_base_height = 0.72


# ==============================================================================
# ENVIRONMENT CLASS
# ==============================================================================
class G1Dex1Stage6Env(DirectRLEnv):
    """G1 Loco-Manipulation Environment with cylindrical workspace."""

    cfg: G1Dex1Stage6EnvCfg

    def __init__(self, cfg: G1Dex1Stage6EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Scene entities
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]
        self.shoulder_marker = self.scene["shoulder_marker"]

        # Joint indices
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

        # Gripper indices (may be empty)
        self.gripper_indices = []
        for jn in G1_RIGHT_GRIPPER_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.gripper_indices.append(i)
                    break
        self.gripper_indices = torch.tensor(self.gripper_indices, device=self.device, dtype=torch.long)
        self.has_gripper = len(self.gripper_indices) > 0

        # Palm index
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and ("palm" in name.lower() or "wrist" in name.lower()):
                self.palm_idx = i
                break
        if self.palm_idx is None:
            # Fallback to elbow
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "elbow" in name.lower():
                    self.palm_idx = i
                    break
        if self.palm_idx is None:
            self.palm_idx = len(body_names) - 1

        # Joint limits
        self.arm_joint_lower = torch.zeros(len(self.arm_indices), device=self.device)
        self.arm_joint_upper = torch.zeros(len(self.arm_indices), device=self.device)
        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS[:len(self.arm_indices)]):
            if jn in ARM_JOINT_LIMITS:
                self.arm_joint_lower[i], self.arm_joint_upper[i] = ARM_JOINT_LIMITS[jn]
            else:
                self.arm_joint_lower[i], self.arm_joint_upper[i] = -3.14, 3.14

        # Buffers
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.smoothed_actions = torch.zeros_like(self.actions)

        self.commands = torch.zeros(self.num_envs, 6, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)  # World frame

        self.prev_potential = torch.zeros(self.num_envs, device=self.device)
        self.prev_arm_vel = torch.zeros(self.num_envs, len(self.arm_indices), device=self.device)

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

        # Shoulder offset
        self.shoulder_offset = torch.tensor([
            cfg.shoulder_offset_x,
            cfg.shoulder_offset_y,
            cfg.shoulder_offset_z,
        ], device=self.device)

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Workspace visualization markers
        self._setup_workspace_visualization()

        print("\n" + "=" * 70)
        print("G1 LOCO-MANIPULATION ENV V3 - CYLINDRICAL WORKSPACE")
        print("=" * 70)
        print(f"  Leg joints: {len(self.leg_indices)}")
        print(f"  Arm joints: {len(self.arm_indices)}")
        print(f"  Gripper: {'YES' if self.has_gripper else 'NO'}")
        print(f"  Palm idx: {self.palm_idx}")
        print("-" * 70)
        print(f"  WORKSPACE:")
        print(f"    Radius: {self.cfg.workspace_radius*100:.0f}cm")
        print(f"    Azimuth: {math.degrees(self.cfg.workspace_azimuth_min):.0f}° to {math.degrees(self.cfg.workspace_azimuth_max):.0f}°")
        print(f"    Ground clearance: {self.cfg.ground_clearance*100:.0f}cm")
        print("-" * 70)
        print(f"  CURRICULUM: Stage {self.curriculum_stage} ({self.current_curriculum.name})")
        print(f"    Height range: {self.current_curriculum.height_min*100:.0f}cm to {self.current_curriculum.height_max*100:.0f}cm")
        print("=" * 70 + "\n")

    @property
    def current_curriculum(self) -> CurriculumLevel:
        return CURRICULUM_LEVELS[min(self.curriculum_stage, len(CURRICULUM_LEVELS) - 1)]

    def _setup_workspace_visualization(self):
        """Setup markers for cylindrical workspace visualization."""
        # Outer cylinder wireframe (yeşil)
        self.workspace_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/WorkspaceCylinder",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),  # Yeşil
                            emissive_color=(0.0, 0.5, 0.0),
                        ),
                    ),
                },
            )
        )

        # Current level indicator (sarı)
        self.level_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/LevelIndicator",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 1.0, 0.0),  # Sarı
                            emissive_color=(0.5, 0.5, 0.0),
                        ),
                    ),
                },
            )
        )

    def _setup_scene(self):
        """Scene setup."""
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]
        self.shoulder_marker = self.scene["shoulder_marker"]

    def _get_shoulder_world_pos(self) -> torch.Tensor:
        """Get shoulder position in world frame."""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Rotate offset by root orientation
        shoulder_local = self.shoulder_offset.unsqueeze(0).expand(self.num_envs, -1)
        shoulder_rotated = rotate_vector_by_quat(shoulder_local, root_quat)

        return root_pos + shoulder_rotated

    def _compute_ee_pos(self) -> torch.Tensor:
        """Get EE position in world frame."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_potential(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-distance / self.cfg.potential_sigma)

    def _sample_target_in_cylinder(self, env_ids: torch.Tensor):
        """
        Sample targets in half-cylinder workspace.

        Cylinder parameters:
        - Center: Right shoulder (dynamic)
        - Radius: workspace_radius (fixed)
        - Height: curriculum height_min to height_max (relative to shoulder)
        - Azimuth: -90° to +90° (front half only)
        - Ground clipping: Never below ground_clearance
        """
        num = len(env_ids)
        level = self.current_curriculum

        # Get shoulder positions
        shoulder_pos = self._get_shoulder_world_pos()[env_ids]  # [N, 3]

        # Sample cylindrical coordinates
        # Radius: fixed
        radius = torch.empty(num, device=self.device).uniform_(
            self.cfg.workspace_radius * 0.5,  # Min 50% of max radius
            self.cfg.workspace_radius
        )

        # Azimuth: front half only (-90° to +90°)
        azimuth = torch.empty(num, device=self.device).uniform_(
            self.cfg.workspace_azimuth_min,
            self.cfg.workspace_azimuth_max
        )

        # Height: curriculum-based (relative to shoulder)
        height_rel = torch.empty(num, device=self.device).uniform_(
            level.height_min,
            level.height_max
        )

        # Convert to Cartesian (relative to shoulder)
        # X = forward (robot facing -X), Y = left/right, Z = up/down
        local_x = -radius * torch.cos(azimuth)  # Forward is -X
        local_y = radius * torch.sin(azimuth)   # Y is sideways
        local_z = height_rel

        # World position
        target_world = torch.stack([
            shoulder_pos[:, 0] + local_x,
            shoulder_pos[:, 1] + local_y,
            shoulder_pos[:, 2] + local_z,
        ], dim=-1)

        # Ground clipping: ensure Z >= ground_clearance
        target_world[:, 2] = torch.clamp(
            target_world[:, 2],
            min=self.cfg.ground_clearance
        )

        self.target_pos[env_ids] = target_world

        # Update target visual
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num, -1)
        pose = torch.cat([target_world, identity_quat], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Initialize potential
        ee_pos = self._compute_ee_pos()[env_ids]
        initial_dist = (ee_pos - target_world).norm(dim=-1)
        self.prev_potential[env_ids] = self._compute_potential(initial_dist)

        self.stage_attempts += len(env_ids)
        self.total_attempts += len(env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands."""
        level = self.current_curriculum
        num = len(env_ids)

        vx = torch.empty(num, device=self.device).uniform_(*level.vx_range)
        vy = torch.empty(num, device=self.device).uniform_(*level.vy_range)
        omega = torch.empty(num, device=self.device).uniform_(*level.omega_range)

        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        self.commands[env_ids, 2] = omega
        self.commands[env_ids, 3] = 0.0
        self.commands[env_ids, 4] = self.cfg.target_base_height
        self.commands[env_ids, 5] = 0.0

    def _update_workspace_visualization(self):
        """Draw cylindrical workspace as green wireframe."""
        if self.num_envs == 0:
            return

        # Only visualize for env 0
        shoulder_pos = self._get_shoulder_world_pos()[0].cpu()
        level = self.current_curriculum

        points = []
        n_azimuth = 16
        n_height = 8

        radius = self.cfg.workspace_radius

        # Generate cylinder points
        azimuths = torch.linspace(
            self.cfg.workspace_azimuth_min,
            self.cfg.workspace_azimuth_max,
            n_azimuth
        )

        heights = torch.linspace(level.height_min, level.height_max, n_height)

        for h in heights:
            z = shoulder_pos[2] + h
            z = max(z, self.cfg.ground_clearance)  # Ground clip

            for az in azimuths:
                x = shoulder_pos[0] - radius * torch.cos(az)
                y = shoulder_pos[1] + radius * torch.sin(az)
                points.append([x.item(), y.item(), z.item()])

        # Vertical lines
        for az in azimuths[::4]:  # Every 4th azimuth
            for h in heights:
                z = shoulder_pos[2] + h
                z = max(z, self.cfg.ground_clearance)
                x = shoulder_pos[0] - radius * torch.cos(az)
                y = shoulder_pos[1] + radius * torch.sin(az)
                points.append([x.item(), y.item(), z.item()])

        if len(points) > 0:
            pos = torch.tensor(points, device=self.device)
            quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(points), -1)
            self.workspace_markers.visualize(translations=pos, orientations=quat)

        # Update shoulder marker
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        shoulder_world = self._get_shoulder_world_pos()[0:1]
        self.shoulder_marker.write_root_pose_to_sim(
            torch.cat([shoulder_world, identity_quat], dim=-1)
        )

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
            leg_vel = self.robot.data.joint_vel[:, self.leg_indices] * 0.1
            obs_list.append(leg_pos)
            obs_list.append(leg_vel)
        else:
            obs_list.append(torch.zeros(self.num_envs, 24, device=self.device))

        # Commands (6)
        obs_list.append(self.commands)

        # Arm joints (10)
        if len(self.arm_indices) > 0:
            arm_pos = self.robot.data.joint_pos[:, self.arm_indices]
            arm_vel = self.robot.data.joint_vel[:, self.arm_indices] * 0.1
            obs_list.append(arm_pos)
            obs_list.append(arm_vel)
        else:
            obs_list.append(torch.zeros(self.num_envs, 10, device=self.device))

        # Target info (6)
        ee_pos = self._compute_ee_pos()
        pos_err = self.target_pos - ee_pos
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        obs_list.append(pos_err)  # 3
        obs_list.append(pos_dist / 0.5)  # 1

        # Target relative to shoulder (2) - useful for learning body pose
        shoulder_pos = self._get_shoulder_world_pos()
        target_rel_shoulder = self.target_pos - shoulder_pos
        obs_list.append(target_rel_shoulder[:, :2])  # XY only

        # Previous actions
        obs_list.append(self.prev_actions)

        # Update visuals
        ee_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(
            torch.cat([ee_pos, ee_quat], dim=-1)
        )
        self._update_workspace_visualization()

        obs = torch.cat(obs_list, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        rewards = torch.zeros(self.num_envs, device=self.device)

        # === LOCOMOTION ===
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rewards += torch.exp(-lin_vel_error / 0.25) * self.cfg.reward_lin_vel_tracking

        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        rewards += torch.exp(-ang_vel_error / 0.25) * self.cfg.reward_ang_vel_tracking

        height_error = torch.square(
            self.robot.data.root_pos_w[:, 2] - self.cfg.target_base_height
        )
        rewards += torch.exp(-height_error / 0.1) * self.cfg.reward_base_height

        proj_grav = self.robot.data.projected_gravity_b
        orientation_error = torch.sum(torch.square(proj_grav[:, :2]), dim=1)
        rewards += torch.exp(-orientation_error / 0.5) * self.cfg.reward_orientation

        # === ARM REACHING ===
        ee_pos = self._compute_ee_pos()
        pos_dist = (ee_pos - self.target_pos).norm(dim=-1)

        # Potential shaping
        if self.cfg.use_potential_shaping:
            current_potential = self._compute_potential(pos_dist)
            potential_reward = self.cfg.potential_scale * (
                self.cfg.potential_gamma * current_potential - self.prev_potential
            )
            self.prev_potential = current_potential
            rewards += potential_reward

        # Distance reward
        pos_reward = 1.0 - torch.tanh(pos_dist / self.cfg.reward_pos_tanh_std)
        rewards += self.cfg.reward_pos_tanh_weight * pos_reward

        # Proximity bonuses
        rewards += (pos_dist < self.cfg.proximity_zone1_dist).float() * self.cfg.proximity_zone1_bonus
        rewards += (pos_dist < self.cfg.proximity_zone2_dist).float() * self.cfg.proximity_zone2_bonus
        rewards += (pos_dist < self.cfg.proximity_zone3_dist).float() * self.cfg.proximity_zone3_bonus

        # Reach detection
        reached = pos_dist < self.current_curriculum.reach_threshold
        rewards += reached.float() * self.cfg.reward_reaching

        reached_ids = torch.where(reached)[0]
        if len(reached_ids) > 0:
            self.reach_count[reached_ids] += 1
            self.episode_reach_count[reached_ids] += 1
            self.stage_reaches += len(reached_ids)
            self.total_reaches += len(reached_ids)
            self._sample_target_in_cylinder(reached_ids)

        # === REGULARIZATION ===
        action_diff = self.smoothed_actions - self.prev_actions
        rewards += action_diff.norm(dim=-1) * self.cfg.reward_action_rate

        if len(self.arm_indices) > 0:
            arm_vel = self.robot.data.joint_vel[:, self.arm_indices]
            rewards += arm_vel.norm(dim=-1) * self.cfg.reward_joint_vel

            arm_acc = (arm_vel - self.prev_arm_vel).norm(dim=-1)
            self.prev_arm_vel = arm_vel.clone()
            rewards += arm_acc * self.cfg.reward_joint_acc

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fallen = self.robot.data.root_pos_w[:, 2] < self.cfg.fall_height_threshold

        proj_grav = self.robot.data.projected_gravity_b
        bad_orientation = proj_grav[:, 2] > -0.5

        max_reached = self.episode_reach_count >= self.cfg.max_reaches_per_episode

        terminated = fallen | bad_orientation | max_reached
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # Reset robot
        default_state = self.robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(jp)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        # Sample
        self._sample_target_in_cylinder(env_ids)
        self._sample_commands(env_ids)

        # Reset buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.smoothed_actions[env_ids] = 0.0
        self.prev_arm_vel[env_ids] = 0.0
        self.episode_reach_count[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.smoothed_actions.clone()
        self.actions = actions.clone()

    def _apply_action(self):
        alpha = self.cfg.action_smoothing_alpha
        self.smoothed_actions = alpha * self.actions + (1 - alpha) * self.smoothed_actions

        joint_targets = self.robot.data.joint_pos.clone()

        # Legs [0:12]
        if len(self.leg_indices) > 0:
            leg_actions = self.smoothed_actions[:, :len(self.leg_indices)] * self.cfg.action_scale_legs
            for i, idx in enumerate(self.leg_indices):
                joint_targets[:, idx] += leg_actions[:, i]

        # Arm [12:17]
        if len(self.arm_indices) > 0:
            arm_start = len(self.leg_indices)
            arm_end = arm_start + len(self.arm_indices)
            arm_actions = self.smoothed_actions[:, arm_start:arm_end] * self.cfg.action_scale_arm

            cur_arm = self.robot.data.joint_pos[:, self.arm_indices]
            new_arm = torch.clamp(cur_arm + arm_actions, self.arm_joint_lower, self.arm_joint_upper)
            for i, idx in enumerate(self.arm_indices):
                joint_targets[:, idx] = new_arm[:, i]

        self.robot.set_joint_position_target(joint_targets)

    def update_curriculum(self, iteration: int) -> float:
        self.stage_step_count += 1

        if self.stage_attempts > 0:
            success_rate = self.stage_reaches / self.stage_attempts
        else:
            success_rate = 0.0

        level = self.current_curriculum

        should_advance = (
            self.cfg.auto_curriculum and
            success_rate >= level.success_rate_threshold and
            self.stage_reaches >= level.min_reaches_to_advance and
            self.stage_step_count >= self.cfg.min_steps_per_stage and
            self.curriculum_stage < len(CURRICULUM_LEVELS) - 1
        )

        if should_advance:
            self.curriculum_stage += 1
            new_level = self.current_curriculum

            print(f"\n{'='*60}")
            print(f"🎯 CURRICULUM → Stage {self.curriculum_stage + 1}: {new_level.name}")
            print(f"   Success: {success_rate*100:.1f}% | Height: {new_level.height_min*100:.0f}cm to {new_level.height_max*100:.0f}cm")
            print(f"{'='*60}\n")

            self.stage_reaches = 0
            self.stage_attempts = 0
            self.stage_step_count = 0

        return success_rate


# Aliases
G1LocoManipEnv = G1Dex1Stage6Env
G1LocoManipEnvCfg = G1Dex1Stage6EnvCfg