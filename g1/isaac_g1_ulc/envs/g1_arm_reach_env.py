"""
G1 Fixed-Base Arm Reaching Environment v2
==========================================

Stage 4: Robot gövdesi SABİT, sadece kol joint'leri eğitiliyor.
Hedef: Palm'ın 2cm önündeki noktayı (EE) rastgele target pozisyon VE oryantasyonuna götürmek.

WORKSPACE (v3 test sonuçlarından):
- MAX_REACH = 0.458m
- SAFE_RADIUS = 0.321m (training için)
- RIGHT_ARM_CENTER = [0.000, 0.174, 0.259] (root-relative)

OBSERVATION SPACE (30 dim):
- Arm joint positions (5)
- Arm joint velocities (5)
- Target position, root-relative (3)
- Target orientation, quaternion (4)
- Current EE position, root-relative (3) - palm'ın 2cm önü
- Current EE orientation, quaternion (4)
- EE-to-target position error (3)
- EE-to-target orientation error (3) - axis-angle representation

ACTION SPACE (5 dim):
- Delta joint positions for arm joints

REWARD:
- Position distance reward
- Orientation distance reward
- Reaching bonus (within 5cm position + 15° orientation)
- Smooth motion reward
- Joint limit penalty
"""

from __future__ import annotations

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_conjugate
from isaaclab.actuators import ImplicitActuatorCfg

# =============================================================================
# WORKSPACE PARAMETERS (from v3 discovery)
# =============================================================================

MAX_REACH = 0.458
SAFE_RADIUS = 0.321  # 70% of max for training

# Root-relative centers
LEFT_ARM_CENTER = torch.tensor([-0.013, -0.082, 0.233])
RIGHT_ARM_CENTER = torch.tensor([0.000, 0.174, 0.259])

# Shoulder offsets from root
LEFT_SHOULDER_OFFSET = torch.tensor([0.002, -0.104, 0.259])
RIGHT_SHOULDER_OFFSET = torch.tensor([0.002, 0.104, 0.259])

# End effector offset (2cm in front of palm)
EE_OFFSET = 0.02  # meters

# =============================================================================
# G1 ARM CONFIGURATION
# =============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Right arm joint names
G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# Joint limits (radians)
ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

# Default arm pose (slightly forward)
DEFAULT_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.5,
    "right_elbow_roll_joint": 0.0,
}


# =============================================================================
# SCENE CONFIGURATION
# =============================================================================

@configclass
class G1ArmReachSceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed-base arm reaching."""

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # G1 Robot - Fixed base, only arm moves
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                ".*": 0.0,
                "right_shoulder_pitch_joint": -0.3,
                "right_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            # High stiffness for non-arm joints (keep them fixed)
            "body_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*hip.*", ".*knee.*", ".*ankle.*", ".*waist.*",
                    "left_shoulder.*", "left_elbow.*",
                ],
                stiffness=1000.0,
                damping=100.0,
            ),
            # Lower stiffness for arm joints (controllable)
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder.*", "right_elbow.*"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    # Target sphere (visual indicator - position)
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                emissive_color=(0.0, 0.5, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.3)),
    )

    # Target orientation indicator (small arrow/cone)
    target_arrow: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetArrow",
        spawn=sim_utils.ConeCfg(
            radius=0.015,
            height=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 0.0),
                emissive_color=(0.0, 0.4, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.35)),
    )

    # End effector marker (2cm in front of palm)
    ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.015,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
                emissive_color=(0.5, 0.25, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.2)),
    )

    # Palm marker
    palm_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PalmMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.0, 0.8),
                emissive_color=(0.4, 0.0, 0.4),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.2)),
    )


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@configclass
class G1ArmReachEnvCfg(DirectRLEnvCfg):
    """Configuration for fixed-base arm reaching environment."""

    # Environment settings
    decimation = 4
    episode_length_s = 10.0

    # Spaces
    num_actions = 5  # 5 arm joints
    num_observations = 30  # Updated for orientation
    num_states = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    # Scene
    scene: G1ArmReachSceneCfg = G1ArmReachSceneCfg(num_envs=1024, env_spacing=2.0)

    # Reward scales
    reward_reaching = 15.0        # Bonus for reaching target (pos + ori)
    reward_pos_distance = -2.0    # Penalty proportional to position distance
    reward_ori_distance = -0.5    # Penalty proportional to orientation distance
    reward_smooth = -0.01         # Penalty for jerky motion
    reward_joint_limit = -1.0     # Penalty for hitting joint limits

    # Task parameters
    pos_threshold = 0.05          # Position success threshold (5cm)
    ori_threshold = 0.26          # Orientation success threshold (~15 degrees in radians)
    action_scale = 0.1            # Scale for delta actions (radians)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def random_quaternions(n: int, device: torch.device) -> torch.Tensor:
    """Generate random unit quaternions (w, x, y, z format)."""
    # Random axis
    axis = torch.randn((n, 3), device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

    # Random angle [0, pi]
    angle = torch.rand((n, 1), device=device) * math.pi

    # Axis-angle to quaternion
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)

    return torch.cat([w, xyz], dim=-1)


def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to axis-angle representation (3D vector)."""
    # q = (w, x, y, z)
    w = q[:, 0:1]
    xyz = q[:, 1:4]

    # Clamp w to avoid numerical issues
    w = torch.clamp(w, -1.0, 1.0)

    # Angle
    angle = 2.0 * torch.acos(torch.abs(w))

    # Axis (normalized xyz)
    norm = xyz.norm(dim=-1, keepdim=True)
    axis = xyz / (norm + 1e-8)

    # Handle small angles
    small_angle = angle < 1e-6
    axis = torch.where(small_angle, torch.zeros_like(axis), axis)

    # Sign correction
    sign = torch.sign(w)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    return axis * angle * sign


def quat_error_axis_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute orientation error as axis-angle vector (3D)."""
    # q_error = q2 * q1^(-1)
    q1_conj = quat_conjugate(q1)
    q_error = quat_mul(q2, q1_conj)

    # Ensure positive w (shortest path)
    q_error = torch.where(
        q_error[:, 0:1] < 0,
        -q_error,
        q_error
    )

    return quat_to_axis_angle(q_error)


def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q. q = (w, x, y, z)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]

    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class G1ArmReachEnv(DirectRLEnv):
    """Fixed-base arm reaching environment with orientation."""

    cfg: G1ArmReachEnvCfg

    def __init__(self, cfg: G1ArmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get scene objects
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.target_arrow = self.scene["target_arrow"]
        self.ee_marker = self.scene["ee_marker"]
        self.palm_marker = self.scene["palm_marker"]

        # Find joint indices for right arm
        self.arm_joint_indices = []
        joint_names = self.robot.data.joint_names
        for joint_name in G1_RIGHT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == joint_name:
                    self.arm_joint_indices.append(i)
                    break

        self.arm_joint_indices = torch.tensor(
            self.arm_joint_indices, device=self.device, dtype=torch.long
        )

        # Find right palm body index
        body_names = self.robot.data.body_names
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break

        if self.palm_idx is None:
            raise ValueError("Could not find right_palm body!")

        # Joint limits tensor
        self.joint_lower = torch.zeros(5, device=self.device)
        self.joint_upper = torch.zeros(5, device=self.device)
        for i, joint_name in enumerate(G1_RIGHT_ARM_JOINTS):
            low, high = ARM_JOINT_LIMITS[joint_name]
            self.joint_lower[i] = low
            self.joint_upper[i] = high

        # Fixed root pose
        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Target storage (root-relative)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 1.0  # Identity quaternion (w=1)

        # Previous actions for smooth reward
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)

        # Local forward direction for EE offset
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Arm center
        self.arm_center = RIGHT_ARM_CENTER.to(self.device)

        print("\n" + "=" * 60)
        print("G1 FIXED-BASE ARM REACHING v2 (with Orientation)")
        print("=" * 60)
        print(f"  Arm joint indices: {self.arm_joint_indices.tolist()}")
        print(f"  Palm body index: {self.palm_idx}")
        print(f"  Arm center: {self.arm_center.tolist()}")
        print(f"  EE offset: {EE_OFFSET}m (in front of palm)")
        print(f"  Safe radius: {SAFE_RADIUS:.3f}m")
        print(f"  Position threshold: {self.cfg.pos_threshold:.3f}m")
        print(f"  Orientation threshold: {math.degrees(self.cfg.ori_threshold):.1f}°")
        print(f"  Observations: {self.cfg.num_observations}")
        print(f"  Actions: {self.cfg.num_actions}")
        print("=" * 60 + "\n")

    def _setup_scene(self):
        """Setup the scene."""
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.target_arrow = self.scene["target_arrow"]
        self.ee_marker = self.scene["ee_marker"]
        self.palm_marker = self.scene["palm_marker"]

    def _compute_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute end effector pose (2cm in front of palm)."""
        # Palm pose
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat_world = self.robot.data.body_quat_w[:, self.palm_idx]  # (w, x, y, z)

        # Palm forward direction (local +X axis in world frame)
        forward_world = rotate_vector_by_quat(self.local_forward, palm_quat_world)

        # EE position = palm + offset * forward
        ee_pos_world = palm_pos_world + EE_OFFSET * forward_world
        ee_quat_world = palm_quat_world

        return ee_pos_world, ee_quat_world

    def _sample_target(self, env_ids: torch.Tensor):
        """Sample random target positions and orientations within workspace."""
        num_samples = len(env_ids)

        # ===== POSITION =====
        # Random direction (unit sphere)
        direction = torch.randn((num_samples, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Random distance (within safe radius)
        distance = torch.rand((num_samples, 1), device=self.device) * SAFE_RADIUS

        # Target position (root-relative)
        targets = self.arm_center + direction * distance

        # Clamp to reasonable bounds
        targets[:, 0] = torch.clamp(targets[:, 0], -0.35, 0.40)  # X
        targets[:, 1] = torch.clamp(targets[:, 1], -0.25, 0.55)  # Y (right side)
        targets[:, 2] = torch.clamp(targets[:, 2], -0.10, 0.60)  # Z

        self.target_pos[env_ids] = targets

        # ===== ORIENTATION =====
        # Random quaternion
        random_quats = random_quaternions(num_samples, self.device)
        self.target_quat[env_ids] = random_quats

        # ===== UPDATE VISUALS =====
        root_pos = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos + targets

        # Target sphere
        target_pose = torch.cat([target_world, random_quats], dim=-1)
        self.target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

        # Target arrow (shows orientation)
        arrow_offset = torch.tensor([[0.0, 0.0, 0.03]], device=self.device).expand(num_samples, -1)
        arrow_pos = target_world + arrow_offset
        arrow_pose = torch.cat([arrow_pos, random_quats], dim=-1)
        self.target_arrow.write_root_pose_to_sim(arrow_pose, env_ids=env_ids)

    def _get_observations(self) -> dict:
        """Compute observations (30 dim)."""
        # Arm joint positions (5)
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]

        # Arm joint velocities (5)
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]

        # EE pose (world)
        ee_pos_world, ee_quat_world = self._compute_ee_pose()

        # Palm pose (for marker)
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat_world = self.robot.data.body_quat_w[:, self.palm_idx]

        # Root position
        root_pos = self.robot.data.root_pos_w

        # EE position (root-relative) (3)
        ee_pos_rel = ee_pos_world - root_pos

        # Target position (root-relative) (3)
        target_pos = self.target_pos

        # Target orientation (4)
        target_quat = self.target_quat

        # EE-to-target position error (3)
        pos_error = target_pos - ee_pos_rel

        # EE-to-target orientation error as axis-angle (3)
        ori_error = quat_error_axis_angle(ee_quat_world, target_quat)

        # Update visual markers
        # EE marker
        ee_marker_pose = torch.cat([ee_pos_world, ee_quat_world], dim=-1)
        self.ee_marker.write_root_pose_to_sim(ee_marker_pose)

        # Palm marker
        palm_marker_pose = torch.cat([palm_pos_world, palm_quat_world], dim=-1)
        self.palm_marker.write_root_pose_to_sim(palm_marker_pose)

        # Concatenate observations (30 total)
        obs = torch.cat([
            arm_joint_pos,      # 5
            arm_joint_vel,      # 5
            target_pos,         # 3
            target_quat,        # 4
            ee_pos_rel,         # 3
            ee_quat_world,      # 4
            pos_error,          # 3
            ori_error,          # 3
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # EE pose
        ee_pos_world, ee_quat_world = self._compute_ee_pose()
        root_pos = self.robot.data.root_pos_w
        ee_pos_rel = ee_pos_world - root_pos

        # Position distance
        pos_distance = torch.norm(ee_pos_rel - self.target_pos, dim=-1)

        # Orientation distance (quaternion dot product -> angle)
        quat_dot = torch.abs(torch.sum(ee_quat_world * self.target_quat, dim=-1))
        quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
        ori_distance = 2.0 * torch.acos(quat_dot)  # angle in radians

        # Position reward
        reward_pos = self.cfg.reward_pos_distance * pos_distance

        # Orientation reward
        reward_ori = self.cfg.reward_ori_distance * ori_distance

        # Reaching bonus (both position AND orientation)
        pos_reached = pos_distance < self.cfg.pos_threshold
        ori_reached = ori_distance < self.cfg.ori_threshold
        fully_reached = pos_reached & ori_reached
        reward_reach = self.cfg.reward_reaching * fully_reached.float()

        # Smooth motion reward
        action_diff = torch.norm(self.actions - self.prev_actions, dim=-1)
        reward_smooth = self.cfg.reward_smooth * action_diff

        # Joint limit penalty
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        lower_violation = torch.clamp(self.joint_lower - arm_joint_pos, min=0).sum(dim=-1)
        upper_violation = torch.clamp(arm_joint_pos - self.joint_upper, min=0).sum(dim=-1)
        reward_limits = self.cfg.reward_joint_limit * (lower_violation + upper_violation)

        # Total reward
        reward = reward_pos + reward_ori + reward_reach + reward_smooth + reward_limits

        # Store actions for next step
        self.prev_actions = self.actions.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # Reset robot to fixed pose
        self.robot.write_root_pose_to_sim(self.fixed_root_pose[env_ids], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel[env_ids], env_ids=env_ids)

        # Reset joints to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])

        for i, joint_name in enumerate(G1_RIGHT_ARM_JOINTS):
            joint_idx = self.arm_joint_indices[i]
            joint_pos[:, joint_idx] = DEFAULT_ARM_POSE.get(joint_name, 0.0)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Sample new target
        self._sample_target(env_ids)

        # Reset previous actions
        self.prev_actions[env_ids] = 0.0

    def _apply_action(self):
        """Apply actions to the robot."""
        # Keep root fixed
        self.robot.write_root_pose_to_sim(self.fixed_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel)

        # Get current arm positions
        current_arm_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]

        # Apply delta action
        target_arm_pos = current_arm_pos + self.actions * self.cfg.action_scale

        # Clamp to joint limits
        target_arm_pos = torch.clamp(target_arm_pos, self.joint_lower, self.joint_upper)

        # Write to simulation
        joint_pos_targets = self.robot.data.joint_pos.clone()
        joint_pos_targets[:, self.arm_joint_indices] = target_arm_pos

        self.robot.set_joint_position_target(joint_pos_targets)