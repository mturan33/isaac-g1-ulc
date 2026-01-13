"""
G1 Fixed-Base Arm Reaching Environment
=======================================

Stage 4: Robot gövdesi SABİT, sadece kol joint'leri eğitiliyor.
Hedef: Palm'ı rastgele target pozisyonuna götürmek.

WORKSPACE (v3 test sonuçlarından):
- MAX_REACH = 0.458m
- SAFE_RADIUS = 0.321m (training için)
- RIGHT_ARM_CENTER = [0.000, 0.174, 0.259] (root-relative)
- LEFT_ARM_CENTER = [-0.013, -0.082, 0.233] (root-relative)

OBSERVATION SPACE (19 dim):
- Arm joint positions (5)
- Arm joint velocities (5)
- Target position, root-relative (3)
- Current palm position, root-relative (3)
- Palm-to-target vector (3)

ACTION SPACE (5 dim):
- Delta joint positions for arm joints

REWARD:
- Distance reward (closer = better)
- Reaching bonus (within 5cm)
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

# =============================================================================
# G1 ARM CONFIGURATION
# =============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Right arm joint names (we'll train right arm first)
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
                linear_damping=1000.0,  # Very high to keep body fixed
                angular_damping=1000.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                ".*": 0.0,
                # Set default arm pose
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

    # Target sphere (visual indicator)
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
                diffuse_color=(0.0, 1.0, 0.0),  # Green
                emissive_color=(0.0, 0.5, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.3)),
    )

    # Palm marker (visual indicator)
    palm_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PalmMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # Orange
                emissive_color=(0.5, 0.25, 0.0),
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
    num_observations = 19  # See docstring
    num_states = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2 ** 21,
            gpu_total_aggregate_pairs_capacity=2 ** 21,
        ),
    )

    # Scene
    scene: G1ArmReachSceneCfg = G1ArmReachSceneCfg(num_envs=1024, env_spacing=2.0)

    # Reward scales
    reward_reaching = 10.0  # Bonus for reaching target
    reward_distance = -1.0  # Penalty proportional to distance
    reward_smooth = -0.01  # Penalty for jerky motion
    reward_joint_limit = -1.0  # Penalty for hitting joint limits

    # Task parameters
    target_radius = 0.05  # Success threshold (5cm)
    target_change_prob = 0.0  # Probability of changing target mid-episode
    action_scale = 0.1  # Scale for delta actions (radians)


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class G1ArmReachEnv(DirectRLEnv):
    """Fixed-base arm reaching environment."""

    cfg: G1ArmReachEnvCfg

    def __init__(self, cfg: G1ArmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get robot
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
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

        # Fixed root pose (for resetting)
        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            device=self.device
        ).expand(self.num_envs, -1)
        self.zero_root_vel = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        # Target positions (root-relative)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Previous actions for smooth reward
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)

        # Default quaternion for marker updates
        self.default_quat = torch.tensor(
            [[0, 0, 0, 1]], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)

        # Arm center (root-relative)
        self.arm_center = RIGHT_ARM_CENTER.to(self.device)

        print("\n" + "=" * 60)
        print("G1 FIXED-BASE ARM REACHING ENVIRONMENT")
        print("=" * 60)
        print(f"  Arm joint indices: {self.arm_joint_indices.tolist()}")
        print(f"  Palm body index: {self.palm_idx}")
        print(f"  Arm center: {self.arm_center.tolist()}")
        print(f"  Safe radius: {SAFE_RADIUS:.3f}m")
        print(f"  Target radius: {self.cfg.target_radius:.3f}m")
        print("=" * 60 + "\n")

    def _setup_scene(self):
        """Setup the scene."""
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.palm_marker = self.scene["palm_marker"]

    def _sample_target(self, env_ids: torch.Tensor):
        """Sample random target positions within workspace."""
        num_samples = len(env_ids)

        # Random direction (unit sphere)
        direction = torch.randn((num_samples, 3), device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True)

        # Random distance (within safe radius)
        distance = torch.rand((num_samples, 1), device=self.device) * SAFE_RADIUS

        # Target position (root-relative)
        targets = self.arm_center + direction * distance

        # Clamp to reasonable bounds
        targets[:, 0] = torch.clamp(targets[:, 0], -0.35, 0.40)  # X
        targets[:, 1] = torch.clamp(targets[:, 1], -0.25, 0.55)  # Y (right side)
        targets[:, 2] = torch.clamp(targets[:, 2], -0.10, 0.60)  # Z

        self.target_pos[env_ids] = targets

        # Update visual target (world coordinates = root + target_pos)
        root_pos = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos + targets

        target_pose = torch.cat([target_world, self.default_quat[env_ids]], dim=-1)
        self.target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    def _get_observations(self) -> dict:
        """Compute observations."""
        # Arm joint positions (5)
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]

        # Arm joint velocities (5)
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]

        # Current palm position (world)
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]

        # Root position
        root_pos = self.robot.data.root_pos_w

        # Palm position (root-relative)
        palm_pos_rel = palm_pos_world - root_pos

        # Target position (already root-relative)
        target_pos = self.target_pos

        # Palm-to-target vector
        palm_to_target = target_pos - palm_pos_rel

        # Update palm marker visual
        palm_marker_pose = torch.cat([palm_pos_world, self.default_quat], dim=-1)
        self.palm_marker.write_root_pose_to_sim(palm_marker_pose)

        # Concatenate observations
        obs = torch.cat([
            arm_joint_pos,  # 5
            arm_joint_vel,  # 5
            target_pos,  # 3
            palm_pos_rel,  # 3
            palm_to_target,  # 3
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Palm position (world)
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]
        root_pos = self.robot.data.root_pos_w
        palm_pos_rel = palm_pos_world - root_pos

        # Distance to target
        distance = torch.norm(palm_pos_rel - self.target_pos, dim=-1)

        # Distance reward (closer = better)
        reward_dist = self.cfg.reward_distance * distance

        # Reaching bonus (within threshold)
        reached = distance < self.cfg.target_radius
        reward_reach = self.cfg.reward_reaching * reached.float()

        # Smooth motion reward
        action_diff = torch.norm(self.actions - self.prev_actions, dim=-1)
        reward_smooth = self.cfg.reward_smooth * action_diff

        # Joint limit penalty
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        lower_violation = torch.clamp(self.joint_lower - arm_joint_pos, min=0).sum(dim=-1)
        upper_violation = torch.clamp(arm_joint_pos - self.joint_upper, min=0).sum(dim=-1)
        reward_limits = self.cfg.reward_joint_limit * (lower_violation + upper_violation)

        # Total reward
        reward = reward_dist + reward_reach + reward_smooth + reward_limits

        # Store actions for next step
        self.prev_actions = self.actions.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length

        # No early termination for arm reaching
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

        # Reset arm joints to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])

        # Set default arm pose
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