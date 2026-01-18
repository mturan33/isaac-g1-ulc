"""
G1 Arm Reach with Orientation - Stage 5
=========================================

Sadece SAĞ KOL - Position + Orientation (Palm Down) reaching.

HEDEF:
- Eli hedef pozisyona götür
- Avuç içi yere baksın (palm down)
- Threshold'a ulaşınca yeni hedef

VISUAL MARKERS:
- 🟢 Yeşil küre  = Target pozisyon
- 🟠 Turuncu    = End effector
- 🔴 Kırmızı ok = Hedef orientation (aşağı)
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

EE_OFFSET = 0.02

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


def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz format)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute angular difference between two quaternions in radians."""
    # q1, q2: (N, 4) wxyz format
    # Returns: (N,) angle in radians
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Create quaternion from axis-angle representation."""
    # axis: (N, 3), angle: (N,)
    # Returns: (N, 4) wxyz format
    half_angle = angle / 2.0
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


@configclass
class G1ArmOrientSceneCfg(InteractiveSceneCfg):
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
                "left_shoulder_pitch_joint": -0.3,
                "left_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            "body_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*", "torso.*", "left_shoulder.*", "left_elbow.*"],
                stiffness=1000.0,
                damping=100.0,
            ),
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["right_shoulder.*", "right_elbow.*"],
                stiffness=150.0,
                damping=15.0,
            ),
        },
    )

    # Yeşil küre - Target pozisyon
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                emissive_color=(0.0, 0.5, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.2, 1.0)),
    )

    # Turuncu küre - End effector marker
    ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
                emissive_color=(0.5, 0.25, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


@configclass
class G1ArmOrientEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 300.0

    # Sadece sağ kol: 5 joint
    num_actions = 5
    # Observation: joint_pos(5) + joint_vel(5) + target_pos(3) + target_ori(4) + ee_pos(3) + ee_ori(4) + pos_err(3) + ori_err(1) = 28
    num_observations = 28
    num_states = 0

    action_space = 5
    observation_space = 28
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity=2 ** 21,
            gpu_total_aggregate_pairs_capacity=2 ** 21,
        ),
    )

    scene: G1ArmOrientSceneCfg = G1ArmOrientSceneCfg(num_envs=1, env_spacing=2.0)

    # Rewards
    reward_reaching = 50.0  # Bonus for reaching (pos + ori)
    reward_pos_distance = -1.5  # Position distance penalty
    reward_ori_distance = -0.5  # Orientation distance penalty
    reward_action_rate = -0.01  # Action smoothness

    # Thresholds
    pos_threshold = 0.07  # 7cm position threshold
    ori_threshold = 0.26  # ~15 degrees orientation threshold (in radians)

    # Action
    action_smoothing_alpha = 0.5
    action_scale = 0.08

    # Curriculum
    initial_target_radius = 0.10
    max_target_radius = 0.25
    curriculum_steps = 3000


class G1ArmOrientEnv(DirectRLEnv):
    cfg: G1ArmOrientEnvCfg

    def __init__(self, cfg: G1ArmOrientEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        # Find arm joint indices
        self.arm_indices = []
        for jn in G1_RIGHT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.arm_indices.append(i)
                    break
        self.arm_indices = torch.tensor(self.arm_indices, device=self.device, dtype=torch.long)

        # Find palm body index
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break

        # Joint limits
        self.joint_lower = torch.zeros(5, device=self.device)
        self.joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS):
            self.joint_lower[i], self.joint_upper[i] = ARM_JOINT_LIMITS[jn]

        # Fixed root pose
        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Target state
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Palm down quaternion: rotation around X-axis by 90 degrees
        # wxyz format: [cos(45°), sin(45°), 0, 0] = [0.707, 0.707, 0, 0]
        self.target_quat = torch.tensor(
            [[0.707, 0.707, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1).clone()

        # Action smoothing
        self.smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)

        # Tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)

        # Curriculum
        self.current_target_radius = self.cfg.initial_target_radius
        self.curriculum_progress = 0.0

        # Timeout
        self.timeout_steps = 90  # 3 seconds
        self.target_timer = torch.zeros(self.num_envs, device=self.device)

        # Forward vector for EE offset
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print("\n" + "=" * 70)
        print("G1 ARM ORIENT ENVIRONMENT - STAGE 5")
        print("=" * 70)
        print(f"  Arm joints: {self.arm_indices.tolist()}")
        print(f"  Palm idx: {self.palm_idx}")
        print("-" * 70)
        print("  HEDEF: Position + Orientation (Palm Down)")
        print(f"    Pos threshold: {self.cfg.pos_threshold}m")
        print(f"    Ori threshold: {self.cfg.ori_threshold:.2f} rad (~{math.degrees(self.cfg.ori_threshold):.0f}°)")
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
        """Compute end-effector position (palm + offset)."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_ee_quat(self) -> torch.Tensor:
        """Get end-effector orientation (palm quaternion)."""
        return self.robot.data.body_quat_w[:, self.palm_idx]

    def _sample_target(self, env_ids: torch.Tensor):
        """Sample target position around current EE, with palm-down orientation."""
        num = len(env_ids)

        # Get current EE position (relative to root)
        ee_pos_world = self._compute_ee_pos()
        root_pos = self.robot.data.root_pos_w
        current_ee_rel = (ee_pos_world - root_pos)[env_ids]

        # Random direction
        direction = torch.randn((num, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Distance
        min_dist = self.cfg.pos_threshold + 0.02
        max_dist = self.current_target_radius
        max_dist = max(max_dist, min_dist + 0.01)
        distance = min_dist + torch.rand((num, 1), device=self.device) * (max_dist - min_dist)

        # Target position
        targets = current_ee_rel + direction * distance

        # Clamp to reachable workspace
        targets[:, 0] = torch.clamp(targets[:, 0], -0.15, 0.35)  # X: forward
        targets[:, 1] = torch.clamp(targets[:, 1], -0.10, 0.40)  # Y: right side
        targets[:, 2] = torch.clamp(targets[:, 2], 0.00, 0.50)  # Z: chest height

        self.target_pos[env_ids] = targets

        # Target orientation: palm down (fixed)
        # Already set in __init__, no need to change

        # Update visual marker
        root_pos_ids = root_pos[env_ids]
        target_world = root_pos_ids + targets
        pose = torch.cat([target_world, self.target_quat[env_ids]], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Reset timer
        self.target_timer[env_ids] = 0

    def update_curriculum(self, iteration: int):
        """Update curriculum based on iteration."""
        progress = min(1.0, iteration / self.cfg.curriculum_steps)
        self.curriculum_progress = progress
        self.current_target_radius = self.cfg.initial_target_radius + \
                                     progress * (self.cfg.max_target_radius - self.cfg.initial_target_radius)

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w

        # Joint state
        joint_pos = self.robot.data.joint_pos[:, self.arm_indices]
        joint_vel = self.robot.data.joint_vel[:, self.arm_indices]

        # EE state (relative to root)
        ee_pos = self._compute_ee_pos() - root_pos
        ee_quat = self._compute_ee_quat()

        # Errors
        pos_err = self.target_pos - ee_pos
        ori_err = quat_diff_rad(ee_quat, self.target_quat).unsqueeze(-1)

        # Update EE marker
        self.ee_marker.write_root_pose_to_sim(
            torch.cat([self._compute_ee_pos(), ee_quat], dim=-1)
        )

        obs = torch.cat([
            joint_pos,  # 5
            joint_vel * 0.1,  # 5
            self.target_pos,  # 3
            self.target_quat,  # 4
            ee_pos,  # 3
            ee_quat,  # 4
            pos_err,  # 3
            ori_err,  # 1
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w

        # Timer
        self.target_timer += 1

        # Position error
        ee_pos = self._compute_ee_pos() - root_pos
        pos_dist = (ee_pos - self.target_pos).norm(dim=-1)

        # Orientation error
        ee_quat = self._compute_ee_quat()
        ori_dist = quat_diff_rad(ee_quat, self.target_quat)

        # Check reaching (both position AND orientation)
        pos_reached = pos_dist < self.cfg.pos_threshold
        ori_reached = ori_dist < self.cfg.ori_threshold
        fully_reached = pos_reached & ori_reached

        reached_ids = torch.where(fully_reached)[0]
        if len(reached_ids) > 0:
            self._sample_target(reached_ids)
            self.reach_count[reached_ids] += 1

        # Timeout
        timeout_ids = torch.where(self.target_timer >= self.timeout_steps)[0]
        if len(timeout_ids) > 0:
            self._sample_target(timeout_ids)

        # Action rate penalty
        action_rate = (self.smoothed_actions - self.prev_actions).norm(dim=-1)

        # Reward
        reward = (
                self.cfg.reward_reaching * fully_reached.float() +
                self.cfg.reward_pos_distance * pos_dist +
                self.cfg.reward_ori_distance * ori_dist +
                self.cfg.reward_action_rate * action_rate
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            self.episode_length_buf >= self.max_episode_length
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        # Reset robot
        self.robot.write_root_pose_to_sim(self.fixed_root_pose[env_ids], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel[env_ids], env_ids=env_ids)

        # Reset joints to default pose
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.robot.data.joint_vel[env_ids])

        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS):
            jp[:, self.arm_indices[i]] = DEFAULT_ARM_POSE[jn]

        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        # Sample new target
        self._sample_target(env_ids)

        # Reset tracking
        self.smoothed_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.reach_count[env_ids] = 0.0
        self.target_timer[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions

    def _apply_action(self):
        alpha = self.cfg.action_smoothing_alpha

        # Store previous for action rate
        self.prev_actions = self.smoothed_actions.clone()

        # Smooth actions
        self.smoothed_actions = alpha * self.actions + (1 - alpha) * self.smoothed_actions

        # Keep root fixed
        self.robot.write_root_pose_to_sim(self.fixed_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel)

        # Apply to arm joints
        cur_pos = self.robot.data.joint_pos[:, self.arm_indices]
        tgt_pos = torch.clamp(
            cur_pos + self.smoothed_actions * self.cfg.action_scale,
            self.joint_lower, self.joint_upper
        )

        jt = self.robot.data.joint_pos.clone()
        jt[:, self.arm_indices] = tgt_pos
        self.robot.set_joint_position_target(jt)