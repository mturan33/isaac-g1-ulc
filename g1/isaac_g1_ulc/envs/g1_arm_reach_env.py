"""
G1 Fixed-Base Arm Reaching Environment v4 (Stage 4a)
=====================================================

SIMPLIFIED VERSION - Position Only, No Orientation

Stage 4a: Robot gövdesi SABİT, sadece kol joint'leri eğitiliyor.
Hedef: Palm'ın 2cm önündeki noktayı (EE) rastgele target pozisyonuna götürmek.

KEY SIMPLIFICATIONS from v3:
============================
1. NO ORIENTATION TARGET - sadece pozisyon
2. VERY CLOSE START - 0.05m radius (5cm)
3. HIGHER TOLERANCE - 0.08m threshold
4. BIGGER BONUS - +50 reaching reward
5. REDUCED SMOOTHNESS PENALTIES - 50% of v3

OBSERVATION SPACE (18 dim):
- Arm joint positions (5)
- Arm joint velocities (5)
- Target position, root-relative (3)
- Current EE position, root-relative (3)
- EE-to-target position error (3) - NORMALIZED

ACTION SPACE (5 dim):
- Delta joint positions for arm joints
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
# WORKSPACE PARAMETERS
# =============================================================================

RIGHT_ARM_CENTER = torch.tensor([0.000, 0.174, 0.259])
EE_OFFSET = 0.02

# =============================================================================
# G1 ARM CONFIGURATION
# =============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

G1_RIGHT_ARM_JOINTS = [
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


# =============================================================================
# SCENE CONFIGURATION
# =============================================================================

@configclass
class G1ArmReachSceneCfg(InteractiveSceneCfg):
    """Scene configuration for fixed-base arm reaching."""

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
                "right_shoulder_pitch_joint": -0.3,
                "right_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            "body_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*hip.*", ".*knee.*", ".*ankle.*", "torso.*",
                    "left_shoulder.*", "left_elbow.*",
                ],
                stiffness=1000.0,
                damping=100.0,
            ),
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder.*", "right_elbow.*"],
                stiffness=150.0,  # Slightly higher for better tracking
                damping=15.0,
            ),
        },
    )

    # Green sphere for target
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.04,  # Bigger for visibility
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.5, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.3)),
    )

    # Orange sphere for EE
    ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0), emissive_color=(0.5, 0.25, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.2)),
    )


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@configclass
class G1ArmReachEnvCfg(DirectRLEnvCfg):
    """Configuration for SIMPLIFIED fixed-base arm reaching."""

    decimation = 4
    episode_length_s = 10.0  # Shorter episodes for faster learning

    # Simplified observation space (no orientation)
    num_actions = 5
    num_observations = 18
    num_states = 0

    action_space = 5
    observation_space = 18
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    scene: G1ArmReachSceneCfg = G1ArmReachSceneCfg(num_envs=1024, env_spacing=2.0)

    # =========================================================================
    # REWARD SCALES - SIMPLIFIED & STRONGER
    # =========================================================================

    # Primary task - HIGH REWARDS
    reward_reaching = 50.0           # +50 for reaching (was +20)
    reward_pos_distance = -2.0       # Position tracking (slightly stronger)

    # Dense shaping rewards
    reward_approach = 5.0            # Bonus for getting closer
    reward_stay_near = 10.0          # Bonus for staying near target

    # Smoothness - REDUCED (50% of v3)
    reward_action_rate = -0.025      # Was -0.05
    reward_joint_vel = -0.01         # Was -0.02

    # Safety
    reward_joint_limit = -2.0

    # =========================================================================
    # MOTION PARAMETERS - MORE RESPONSIVE
    # =========================================================================

    action_smoothing_alpha = 0.5     # Less smoothing (was 0.3)
    action_scale = 0.08              # Bigger steps (was 0.05)
    max_joint_vel = 2.0              # Higher velocity allowed

    # =========================================================================
    # SIMPLIFIED TASK PARAMETERS
    # =========================================================================

    pos_threshold = 0.08             # 8cm (was 5cm)

    # Curriculum - START VERY CLOSE
    initial_target_radius = 0.05     # 5cm (was 0.15m)
    max_target_radius = 0.25         # 25cm max (was 0.30m)
    curriculum_steps = 3000          # Slower curriculum


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class G1ArmReachEnv(DirectRLEnv):
    """SIMPLIFIED fixed-base arm reaching - Position Only."""

    cfg: G1ArmReachEnvCfg

    def __init__(self, cfg: G1ArmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        # Find joint indices
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

        # Find palm index
        body_names = self.robot.data.body_names
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break

        if self.palm_idx is None:
            raise ValueError("Could not find right_palm body!")

        # Joint limits
        self.joint_lower = torch.zeros(5, device=self.device)
        self.joint_upper = torch.zeros(5, device=self.device)
        for i, joint_name in enumerate(G1_RIGHT_ARM_JOINTS):
            low, high = ARM_JOINT_LIMITS[joint_name]
            self.joint_lower[i] = low
            self.joint_upper[i] = high

        # Fixed root
        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Target storage (POSITION ONLY)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # History buffers
        self.smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_ee_distance = torch.zeros(self.num_envs, device=self.device)

        # Curriculum
        self.curriculum_progress = 0.0
        self.current_target_radius = self.cfg.initial_target_radius

        # Counters
        self.reach_count = torch.zeros(self.num_envs, device=self.device)

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        self.arm_center = RIGHT_ARM_CENTER.to(self.device)

        print("\n" + "=" * 70)
        print("G1 FIXED-BASE ARM REACHING v4 (SIMPLIFIED - Position Only)")
        print("=" * 70)
        print(f"  Arm joint indices: {self.arm_joint_indices.tolist()}")
        print(f"  Palm body index: {self.palm_idx}")
        print(f"  Observation dim: {self.cfg.num_observations}")
        print("-" * 70)
        print("  SIMPLIFIED PARAMETERS:")
        print(f"    NO ORIENTATION TARGET")
        print(f"    Position threshold: {self.cfg.pos_threshold}m")
        print(f"    Action scale: {self.cfg.action_scale} rad")
        print(f"    Reaching bonus: +{self.cfg.reward_reaching}")
        print("-" * 70)
        print("  CURRICULUM:")
        print(f"    Initial radius: {self.cfg.initial_target_radius}m")
        print(f"    Max radius: {self.cfg.max_target_radius}m")
        print("=" * 70 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    def _compute_ee_pos(self) -> torch.Tensor:
        """Compute end effector position (2cm in front of palm)."""
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat_world = self.robot.data.body_quat_w[:, self.palm_idx]
        forward_world = rotate_vector_by_quat(self.local_forward, palm_quat_world)
        ee_pos_world = palm_pos_world + EE_OFFSET * forward_world
        return ee_pos_world

    def _sample_target(self, env_ids: torch.Tensor):
        """Sample random target position within current curriculum radius."""
        num_samples = len(env_ids)

        # Random direction
        direction = torch.randn((num_samples, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Random distance within curriculum radius
        distance = torch.rand((num_samples, 1), device=self.device) * self.current_target_radius

        # Target relative to arm center
        targets = self.arm_center + direction * distance

        # Clamp to reachable workspace
        targets[:, 0] = torch.clamp(targets[:, 0], -0.25, 0.30)  # X: front-back
        targets[:, 1] = torch.clamp(targets[:, 1], -0.10, 0.45)  # Y: left-right
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.50)   # Z: up-down

        self.target_pos[env_ids] = targets

        # Update target visual
        root_pos = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos + targets

        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num_samples, -1)
        target_pose = torch.cat([target_world, default_quat], dim=-1)
        self.target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    def _get_observations(self) -> dict:
        """Get observations - SIMPLIFIED (no orientation)."""

        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]

        ee_pos_world = self._compute_ee_pos()
        root_pos = self.robot.data.root_pos_w
        ee_pos_rel = ee_pos_world - root_pos

        # Position error (normalized for stable learning)
        pos_error = self.target_pos - ee_pos_rel
        pos_error_normalized = pos_error / (self.cfg.max_target_radius + 0.01)

        # Update EE marker
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        ee_marker_pose = torch.cat([ee_pos_world, palm_quat], dim=-1)
        self.ee_marker.write_root_pose_to_sim(ee_marker_pose)

        # Observation: 18 dim
        obs = torch.cat([
            arm_joint_pos,           # 5
            arm_joint_vel * 0.1,     # 5 (scaled down)
            self.target_pos,         # 3
            ee_pos_rel,              # 3
            pos_error_normalized,    # 3 (normalized)
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards - SIMPLIFIED with dense shaping."""

        ee_pos_world = self._compute_ee_pos()
        root_pos = self.robot.data.root_pos_w
        ee_pos_rel = ee_pos_world - root_pos

        # Distance to target
        pos_distance = torch.norm(ee_pos_rel - self.target_pos, dim=-1)

        # =====================================================================
        # TASK REWARDS
        # =====================================================================

        # 1. Position distance penalty (main signal)
        reward_pos = self.cfg.reward_pos_distance * pos_distance

        # 2. REACHING BONUS - BIG REWARD
        reached = pos_distance < self.cfg.pos_threshold
        reward_reach = self.cfg.reward_reaching * reached.float()

        # 3. Approach reward (getting closer)
        distance_improvement = self.prev_ee_distance - pos_distance
        reward_approach = self.cfg.reward_approach * torch.clamp(distance_improvement, 0, 0.1)

        # 4. Stay near bonus (encourage staying at target)
        near_target = pos_distance < (self.cfg.pos_threshold * 2)
        reward_stay = self.cfg.reward_stay_near * near_target.float()

        # =====================================================================
        # SMOOTHNESS PENALTIES (REDUCED)
        # =====================================================================

        # Action rate penalty
        action_rate = torch.norm(self.smoothed_actions - self.prev_actions, dim=-1)
        reward_action_rate = self.cfg.reward_action_rate * action_rate

        # Joint velocity penalty
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]
        vel_magnitude = torch.norm(arm_joint_vel, dim=-1)
        vel_excess = torch.clamp(vel_magnitude - self.cfg.max_joint_vel, min=0)
        reward_joint_vel = self.cfg.reward_joint_vel * vel_excess

        # =====================================================================
        # SAFETY PENALTY
        # =====================================================================

        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        lower_violation = torch.clamp(self.joint_lower - arm_joint_pos, min=0).sum(dim=-1)
        upper_violation = torch.clamp(arm_joint_pos - self.joint_upper, min=0).sum(dim=-1)
        reward_joint_limit = self.cfg.reward_joint_limit * (lower_violation + upper_violation)

        # =====================================================================
        # TOTAL REWARD
        # =====================================================================

        reward = (
            reward_pos +
            reward_reach +
            reward_approach +
            reward_stay +
            reward_action_rate +
            reward_joint_vel +
            reward_joint_limit
        )

        # Update history
        self.prev_actions = self.smoothed_actions.clone()
        self.prev_ee_distance = pos_distance.clone()

        # Track reaching
        self.reach_count += reached.float()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # Fix root
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

        # Reset history
        self.smoothed_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_ee_distance[env_ids] = 0.3  # Start with reasonable distance
        self.reach_count[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions before physics step."""
        self.actions = actions

    def _apply_action(self):
        """Apply actions with smoothing filter."""

        # Action smoothing (EMA)
        alpha = self.cfg.action_smoothing_alpha
        self.smoothed_actions = alpha * self.actions + (1 - alpha) * self.smoothed_actions

        # Keep root fixed
        self.robot.write_root_pose_to_sim(self.fixed_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel)

        # Apply smoothed delta
        current_arm_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        target_arm_pos = current_arm_pos + self.smoothed_actions * self.cfg.action_scale
        target_arm_pos = torch.clamp(target_arm_pos, self.joint_lower, self.joint_upper)

        # Set targets for all joints
        joint_pos_targets = self.robot.data.joint_pos.clone()
        joint_pos_targets[:, self.arm_joint_indices] = target_arm_pos

        self.robot.set_joint_position_target(joint_pos_targets)

    def update_curriculum(self, iteration: int):
        """Update curriculum based on training progress."""
        progress = min(iteration / self.cfg.curriculum_steps, 1.0)
        self.curriculum_progress = progress
        self.current_target_radius = (
            self.cfg.initial_target_radius +
            progress * (self.cfg.max_target_radius - self.cfg.initial_target_radius)
        )