"""
G1 Fixed-Base Arm Reaching Environment v3
==========================================

Stage 4: Robot gövdesi SABİT, sadece kol joint'leri eğitiliyor.
Hedef: Palm'ın 2cm önündeki noktayı (EE) rastgele target pozisyon VE oryantasyonuna
SMOOTH ve TİTREŞİMSİZ bir şekilde götürmek.

ULC-INSPIRED SMOOTH MOTION TECHNIQUES:
======================================
1. Action Smoothing (Exponential Moving Average)
   - smoothed_action = α * raw_action + (1-α) * prev_action
   - Ani action değişimlerini filtreler

2. Joint Velocity Penalty
   - Yüksek eklem hızlarında ceza
   - Kontrollü, yavaş hareketler

3. Joint Acceleration Penalty (Jerk Minimization)
   - Hız değişimlerinde ceza
   - Titreşimi önler

4. Action Rate Penalty (Double Derivative)
   - Action'ın ikinci türevinde ceza
   - Ultra-smooth trajectories

5. Progressive Target Distance
   - Başlangıçta yakın hedefler
   - Curriculum ile uzak hedeflere geçiş

6. Reaching Smoothness Bonus
   - Hedefe smooth yaklaşmada bonus
   - Ani frenleme yerine yavaş durma

OBSERVATION SPACE (30 dim):
- Arm joint positions (5)
- Arm joint velocities (5)
- Target position, root-relative (3)
- Target orientation, quaternion (4)
- Current EE position, root-relative (3)
- Current EE orientation, quaternion (4)
- EE-to-target position error (3)
- EE-to-target orientation error (3)

ACTION SPACE (5 dim):
- Delta joint positions for arm joints (smoothed internally)
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
SAFE_RADIUS = 0.321

LEFT_ARM_CENTER = torch.tensor([-0.013, -0.082, 0.233])
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
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.5, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.3)),
    )

    target_arrow: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetArrow",
        spawn=sim_utils.ConeCfg(
            radius=0.015,
            height=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0), emissive_color=(0.0, 0.4, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.35)),
    )

    ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.015,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0), emissive_color=(0.5, 0.25, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.2)),
    )

    palm_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PalmMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8), emissive_color=(0.4, 0.0, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.2)),
    )


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@configclass
class G1ArmReachEnvCfg(DirectRLEnvCfg):
    """Configuration for fixed-base arm reaching with smooth motion."""

    decimation = 4
    episode_length_s = 12.0

    num_actions = 5
    num_observations = 30
    num_states = 0

    # Action and observation spaces (required by Isaac Lab 2.x)
    action_space = 5
    observation_space = 30
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
    # REWARD SCALES (ULC-inspired smooth motion)
    # =========================================================================

    # Primary task rewards
    reward_reaching = 20.0
    reward_pos_distance = -1.5
    reward_ori_distance = -0.3

    # Smoothness rewards (CRITICAL for no jitter)
    reward_action_rate = -0.05
    reward_action_accel = -0.02
    reward_joint_vel = -0.02
    reward_joint_accel = -0.01
    reward_smooth_approach = 2.0

    # Safety rewards
    reward_joint_limit = -2.0
    reward_joint_limit_soft = -0.1

    # =========================================================================
    # SMOOTH MOTION PARAMETERS
    # =========================================================================

    action_smoothing_alpha = 0.3
    action_scale = 0.05
    max_joint_vel = 1.5

    pos_threshold = 0.05
    ori_threshold = 0.26

    initial_target_radius = 0.15
    max_target_radius = 0.30
    curriculum_steps = 2000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def random_quaternions(n: int, device: torch.device) -> torch.Tensor:
    axis = torch.randn((n, 3), device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    angle = torch.rand((n, 1), device=device) * math.pi
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    return torch.cat([w, xyz], dim=-1)


def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    w = torch.clamp(w, -1.0, 1.0)
    angle = 2.0 * torch.acos(torch.abs(w))
    norm = xyz.norm(dim=-1, keepdim=True)
    axis = xyz / (norm + 1e-8)
    small_angle = angle < 1e-6
    axis = torch.where(small_angle, torch.zeros_like(axis), axis)
    sign = torch.sign(w)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return axis * angle * sign


def quat_error_axis_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1_conj = quat_conjugate(q1)
    q_error = quat_mul(q2, q1_conj)
    q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
    return quat_to_axis_angle(q_error)


def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class G1ArmReachEnv(DirectRLEnv):
    """Fixed-base arm reaching environment with ULC-inspired smooth motion."""

    cfg: G1ArmReachEnvCfg

    def __init__(self, cfg: G1ArmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.target_arrow = self.scene["target_arrow"]
        self.ee_marker = self.scene["ee_marker"]
        self.palm_marker = self.scene["palm_marker"]

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

        # Soft limits (90% of range)
        joint_range = self.joint_upper - self.joint_lower
        self.joint_lower_soft = self.joint_lower + 0.1 * joint_range
        self.joint_upper_soft = self.joint_upper - 0.1 * joint_range

        # Fixed root
        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Target storage
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat[:, 0] = 1.0

        # =====================================================================
        # SMOOTH MOTION HISTORY BUFFERS
        # =====================================================================

        self.smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_prev_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_ee_distance = torch.zeros(self.num_envs, device=self.device)

        # Curriculum
        self.curriculum_progress = 0.0
        self.current_target_radius = self.cfg.initial_target_radius

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        self.arm_center = RIGHT_ARM_CENTER.to(self.device)

        print("\n" + "=" * 70)
        print("G1 FIXED-BASE ARM REACHING v3 (ULC Smooth Motion)")
        print("=" * 70)
        print(f"  Arm joint indices: {self.arm_joint_indices.tolist()}")
        print(f"  Palm body index: {self.palm_idx}")
        print(f"  EE offset: {EE_OFFSET}m")
        print("-" * 70)
        print("  SMOOTH MOTION PARAMETERS:")
        print(f"    Action smoothing α: {self.cfg.action_smoothing_alpha}")
        print(f"    Action scale: {self.cfg.action_scale} rad")
        print(f"    Max joint velocity: {self.cfg.max_joint_vel} rad/s")
        print("-" * 70)
        print("  CURRICULUM:")
        print(f"    Initial target radius: {self.cfg.initial_target_radius}m")
        print(f"    Max target radius: {self.cfg.max_target_radius}m")
        print(f"    Curriculum steps: {self.cfg.curriculum_steps}")
        print("=" * 70 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.target_arrow = self.scene["target_arrow"]
        self.ee_marker = self.scene["ee_marker"]
        self.palm_marker = self.scene["palm_marker"]

    def _compute_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat_world = self.robot.data.body_quat_w[:, self.palm_idx]
        forward_world = rotate_vector_by_quat(self.local_forward, palm_quat_world)
        ee_pos_world = palm_pos_world + EE_OFFSET * forward_world
        return ee_pos_world, palm_quat_world

    def _sample_target(self, env_ids: torch.Tensor):
        num_samples = len(env_ids)

        # Position with curriculum
        direction = torch.randn((num_samples, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        distance = torch.rand((num_samples, 1), device=self.device) * self.current_target_radius
        targets = self.arm_center + direction * distance

        targets[:, 0] = torch.clamp(targets[:, 0], -0.30, 0.35)
        targets[:, 1] = torch.clamp(targets[:, 1], -0.20, 0.50)
        targets[:, 2] = torch.clamp(targets[:, 2], 0.00, 0.55)

        self.target_pos[env_ids] = targets

        # Orientation (constrained to ±90°)
        axis = torch.randn((num_samples, 3), device=self.device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        angle = (torch.rand((num_samples, 1), device=self.device) - 0.5) * math.pi
        half_angle = angle / 2
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle)
        random_quats = torch.cat([w, xyz], dim=-1)
        self.target_quat[env_ids] = random_quats

        # Update visuals
        root_pos = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos + targets

        target_pose = torch.cat([target_world, random_quats], dim=-1)
        self.target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

        arrow_offset = torch.tensor([[0.0, 0.0, 0.03]], device=self.device).expand(num_samples, -1)
        arrow_pos = target_world + arrow_offset
        arrow_pose = torch.cat([arrow_pos, random_quats], dim=-1)
        self.target_arrow.write_root_pose_to_sim(arrow_pose, env_ids=env_ids)

    def _get_observations(self) -> dict:
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]

        ee_pos_world, ee_quat_world = self._compute_ee_pose()
        palm_pos_world = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat_world = self.robot.data.body_quat_w[:, self.palm_idx]

        root_pos = self.robot.data.root_pos_w
        ee_pos_rel = ee_pos_world - root_pos

        pos_error = self.target_pos - ee_pos_rel
        ori_error = quat_error_axis_angle(ee_quat_world, self.target_quat)

        # Update markers
        ee_marker_pose = torch.cat([ee_pos_world, ee_quat_world], dim=-1)
        self.ee_marker.write_root_pose_to_sim(ee_marker_pose)

        palm_marker_pose = torch.cat([palm_pos_world, palm_quat_world], dim=-1)
        self.palm_marker.write_root_pose_to_sim(palm_marker_pose)

        obs = torch.cat([
            arm_joint_pos,
            arm_joint_vel,
            self.target_pos,
            self.target_quat,
            ee_pos_rel,
            ee_quat_world,
            pos_error,
            ori_error,
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with ULC-style smooth motion incentives."""

        # Task rewards
        ee_pos_world, ee_quat_world = self._compute_ee_pose()
        root_pos = self.robot.data.root_pos_w
        ee_pos_rel = ee_pos_world - root_pos

        pos_distance = torch.norm(ee_pos_rel - self.target_pos, dim=-1)

        quat_dot = torch.abs(torch.sum(ee_quat_world * self.target_quat, dim=-1))
        quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
        ori_distance = 2.0 * torch.acos(quat_dot)

        reward_pos = self.cfg.reward_pos_distance * pos_distance
        reward_ori = self.cfg.reward_ori_distance * ori_distance

        pos_reached = pos_distance < self.cfg.pos_threshold
        ori_reached = ori_distance < self.cfg.ori_threshold
        fully_reached = pos_reached & ori_reached
        reward_reach = self.cfg.reward_reaching * fully_reached.float()

        # =====================================================================
        # SMOOTH MOTION REWARDS
        # =====================================================================

        # 1. Action Rate Penalty
        action_rate = torch.norm(self.smoothed_actions - self.prev_actions, dim=-1)
        reward_action_rate = self.cfg.reward_action_rate * action_rate

        # 2. Action Acceleration Penalty
        action_accel = torch.norm(
            (self.smoothed_actions - self.prev_actions) -
            (self.prev_actions - self.prev_prev_actions),
            dim=-1
        )
        reward_action_accel = self.cfg.reward_action_accel * action_accel

        # 3. Joint Velocity Penalty
        arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]
        joint_vel_magnitude = torch.norm(arm_joint_vel, dim=-1)
        vel_excess = torch.clamp(joint_vel_magnitude - self.cfg.max_joint_vel * 0.5, min=0)
        reward_joint_vel = self.cfg.reward_joint_vel * (vel_excess ** 2)

        # 4. Joint Acceleration Penalty
        joint_accel = torch.norm(arm_joint_vel - self.prev_joint_vel, dim=-1)
        reward_joint_accel = self.cfg.reward_joint_accel * joint_accel

        # 5. Smooth Approach Bonus
        distance_reduction = self.prev_ee_distance - pos_distance
        approaching = distance_reduction > 0
        close_to_target = pos_distance < 0.15
        slowing_down = joint_vel_magnitude < self.cfg.max_joint_vel * 0.3
        smooth_approach = approaching & close_to_target & slowing_down
        reward_smooth = self.cfg.reward_smooth_approach * smooth_approach.float()

        # Safety rewards
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        lower_violation = torch.clamp(self.joint_lower - arm_joint_pos, min=0).sum(dim=-1)
        upper_violation = torch.clamp(arm_joint_pos - self.joint_upper, min=0).sum(dim=-1)
        reward_joint_limit = self.cfg.reward_joint_limit * (lower_violation + upper_violation)

        lower_soft_violation = torch.clamp(self.joint_lower_soft - arm_joint_pos, min=0).sum(dim=-1)
        upper_soft_violation = torch.clamp(arm_joint_pos - self.joint_upper_soft, min=0).sum(dim=-1)
        reward_joint_soft = self.cfg.reward_joint_limit_soft * (lower_soft_violation + upper_soft_violation)

        # Total
        reward = (
            reward_pos +
            reward_ori +
            reward_reach +
            reward_action_rate +
            reward_action_accel +
            reward_joint_vel +
            reward_joint_accel +
            reward_smooth +
            reward_joint_limit +
            reward_joint_soft
        )

        # Update history
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.smoothed_actions.clone()
        self.prev_joint_vel = arm_joint_vel.clone()
        self.prev_ee_distance = pos_distance.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        self.robot.write_root_pose_to_sim(self.fixed_root_pose[env_ids], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel[env_ids], env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])

        for i, joint_name in enumerate(G1_RIGHT_ARM_JOINTS):
            joint_idx = self.arm_joint_indices[i]
            joint_pos[:, joint_idx] = DEFAULT_ARM_POSE.get(joint_name, 0.0)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._sample_target(env_ids)

        # Reset history
        self.smoothed_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.prev_ee_distance[env_ids] = 0.5

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions with smoothing filter (called before physics step)."""

        # Store raw actions
        self.actions = actions

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