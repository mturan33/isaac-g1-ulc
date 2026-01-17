"""
G1 Dual Arm Environment (Play Only)
====================================

Stage 4 Dual Arm: Her iki kol da aktif, 4 visual marker.
Training env'deki spawn sistemi kullanılıyor.
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
# WORKSPACE PARAMETERS (Training env'den)
# =============================================================================

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

G1_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
]

ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
    "left_shoulder_pitch_joint": (-2.97, 2.79),
    "left_shoulder_roll_joint": (-1.59, 2.25),
    "left_shoulder_yaw_joint": (-2.62, 2.62),
    "left_elbow_pitch_joint": (-0.23, 3.42),
    "left_elbow_roll_joint": (-2.09, 2.09),
}

DEFAULT_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.5,
    "right_elbow_roll_joint": 0.0,
    "left_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 0.5,
    "left_elbow_roll_joint": 0.0,
}


# =============================================================================
# SCENE CONFIGURATION
# =============================================================================

@configclass
class G1DualArmSceneCfg(InteractiveSceneCfg):
    """Scene configuration for dual arm."""

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
                joint_names_expr=[
                    ".*hip.*", ".*knee.*", ".*ankle.*", "torso.*",
                ],
                stiffness=1000.0,
                damping=100.0,
            ),
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder.*", "right_elbow.*",
                    "left_shoulder.*", "left_elbow.*",
                ],
                stiffness=150.0,
                damping=15.0,
            ),
        },
    )

    # Green sphere - RIGHT arm target
    right_target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightTarget",
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

    # Blue sphere - LEFT arm target
    left_target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftTarget",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.5, 1.0),
                emissive_color=(0.0, 0.25, 0.5)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.2, 1.0)),
    )

    # Orange sphere - RIGHT arm EE marker
    right_ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightEE",
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

    # Purple sphere - LEFT arm EE marker
    left_ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftEE",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.0, 0.8),
                emissive_color=(0.4, 0.0, 0.4)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.2, 1.0)),
    )


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@configclass
class G1DualArmEnvCfg(DirectRLEnvCfg):
    """Configuration for dual arm play environment."""

    decimation = 4
    episode_length_s = 300.0

    num_actions = 10
    num_observations = 36
    num_states = 0

    action_space = 10
    observation_space = 36
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

    scene: G1DualArmSceneCfg = G1DualArmSceneCfg(num_envs=1, env_spacing=2.0)

    # Reward scales (training env'den)
    reward_reaching = 20.0
    reward_pos_distance = -1.0
    reward_approach = 2.0
    reward_action_rate = -0.02
    reward_joint_vel = -0.01
    reward_joint_limit = -2.0

    # Motion parameters (training env'den)
    action_smoothing_alpha = 0.5
    action_scale = 0.08
    max_joint_vel = 2.0

    # Task parameters (training env'den)
    pos_threshold = 0.05
    initial_target_radius = 0.10
    max_target_radius = 0.30


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

class G1DualArmEnv(DirectRLEnv):
    """Dual arm environment for play."""

    cfg: G1DualArmEnvCfg

    def __init__(self, cfg: G1DualArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.right_target_obj = self.scene["right_target"]
        self.left_target_obj = self.scene["left_target"]
        self.right_ee_marker = self.scene["right_ee_marker"]
        self.left_ee_marker = self.scene["left_ee_marker"]

        joint_names = self.robot.data.joint_names

        # Right arm joint indices
        self.right_arm_indices = []
        for joint_name in G1_RIGHT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == joint_name:
                    self.right_arm_indices.append(i)
                    break
        self.right_arm_indices = torch.tensor(
            self.right_arm_indices, device=self.device, dtype=torch.long
        )

        # Left arm joint indices
        self.left_arm_indices = []
        for joint_name in G1_LEFT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == joint_name:
                    self.left_arm_indices.append(i)
                    break
        self.left_arm_indices = torch.tensor(
            self.left_arm_indices, device=self.device, dtype=torch.long
        )

        # Palm indices
        body_names = self.robot.data.body_names
        self.right_palm_idx = None
        self.left_palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.right_palm_idx = i
            if "left" in name.lower() and "palm" in name.lower():
                self.left_palm_idx = i

        if self.right_palm_idx is None or self.left_palm_idx is None:
            raise ValueError("Could not find palm bodies!")

        # Joint limits - RIGHT
        self.right_joint_lower = torch.zeros(5, device=self.device)
        self.right_joint_upper = torch.zeros(5, device=self.device)
        for i, joint_name in enumerate(G1_RIGHT_ARM_JOINTS):
            low, high = ARM_JOINT_LIMITS[joint_name]
            self.right_joint_lower[i] = low
            self.right_joint_upper[i] = high

        # Joint limits - LEFT
        self.left_joint_lower = torch.zeros(5, device=self.device)
        self.left_joint_upper = torch.zeros(5, device=self.device)
        for i, joint_name in enumerate(G1_LEFT_ARM_JOINTS):
            low, high = ARM_JOINT_LIMITS[joint_name]
            self.left_joint_lower[i] = low
            self.left_joint_upper[i] = high

        # Fixed root
        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Target storage
        self.right_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # History buffers
        self.right_smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.left_smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_right_distance = torch.zeros(self.num_envs, device=self.device)
        self.prev_left_distance = torch.zeros(self.num_envs, device=self.device)

        # Reach counters
        self.right_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.left_reach_count = torch.zeros(self.num_envs, device=self.device)

        # Curriculum (training env'den)
        self.current_target_radius = self.cfg.initial_target_radius

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print("\n" + "=" * 70)
        print("G1 DUAL ARM ENVIRONMENT (Play Only)")
        print("=" * 70)
        print(f"  Right arm indices: {self.right_arm_indices.tolist()}")
        print(f"  Left arm indices:  {self.left_arm_indices.tolist()}")
        print(f"  Right palm idx:    {self.right_palm_idx}")
        print(f"  Left palm idx:     {self.left_palm_idx}")
        print("-" * 70)
        print("  VISUAL MARKERS:")
        print("    🟢 Yeşil   = Sağ kol target")
        print("    🔵 Mavi    = Sol kol target")
        print("    🟠 Turuncu = Sağ el (EE)")
        print("    🟣 Mor     = Sol el (EE)")
        print("=" * 70 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.right_target_obj = self.scene["right_target"]
        self.left_target_obj = self.scene["left_target"]
        self.right_ee_marker = self.scene["right_ee_marker"]
        self.left_ee_marker = self.scene["left_ee_marker"]

    def _compute_right_ee_pos(self) -> torch.Tensor:
        """Compute right end effector position (2cm in front of palm)."""
        palm_pos = self.robot.data.body_pos_w[:, self.right_palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.right_palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_left_ee_pos(self) -> torch.Tensor:
        """Compute left end effector position (2cm in front of palm)."""
        palm_pos = self.robot.data.body_pos_w[:, self.left_palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.left_palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _sample_right_target(self, env_ids: torch.Tensor):
        """Sample right arm target - TRAINING ENV İLE AYNI SİSTEM."""
        num_samples = len(env_ids)

        # Mevcut EE pozisyonunu al (training env'deki gibi)
        ee_pos_world = self._compute_right_ee_pos()
        root_pos = self.robot.data.root_pos_w
        current_ee_rel = (ee_pos_world - root_pos)[env_ids]

        # Rastgele yön (training env'deki gibi)
        direction = torch.randn((num_samples, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Mesafe hesapla (training env'deki gibi)
        min_dist = self.cfg.pos_threshold + 0.02
        max_dist = self.current_target_radius
        max_dist = max(max_dist, min_dist + 0.01)
        distance = min_dist + torch.rand((num_samples, 1), device=self.device) * (max_dist - min_dist)

        # Hedef = mevcut EE + offset
        targets = current_ee_rel + direction * distance

        # SAĞ KOL WORKSPACE - Training env'den farklı olarak ÖNDE olmasını garanti et
        targets[:, 0] = torch.clamp(targets[:, 0], 0.15, 0.40)   # X: KESİNLİKLE ÖNDE!
        targets[:, 1] = torch.clamp(targets[:, 1], -0.35, 0.00)  # Y: sağ taraf (negatif veya 0)
        targets[:, 2] = torch.clamp(targets[:, 2], -0.15, 0.35)  # Z: el seviyesi

        self.right_target_pos[env_ids] = targets

        # Marker güncelle
        root_pos_ids = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos_ids + targets
        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num_samples, -1)
        target_pose = torch.cat([target_world, default_quat], dim=-1)
        self.right_target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    def _sample_left_target(self, env_ids: torch.Tensor):
        """Sample left arm target - TRAINING ENV İLE AYNI SİSTEM."""
        num_samples = len(env_ids)

        # Mevcut EE pozisyonunu al
        ee_pos_world = self._compute_left_ee_pos()
        root_pos = self.robot.data.root_pos_w
        current_ee_rel = (ee_pos_world - root_pos)[env_ids]

        # Rastgele yön
        direction = torch.randn((num_samples, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Mesafe hesapla
        min_dist = self.cfg.pos_threshold + 0.02
        max_dist = self.current_target_radius
        max_dist = max(max_dist, min_dist + 0.01)
        distance = min_dist + torch.rand((num_samples, 1), device=self.device) * (max_dist - min_dist)

        # Hedef = mevcut EE + offset
        targets = current_ee_rel + direction * distance

        # SOL KOL WORKSPACE - ÖNDE olmasını garanti et
        targets[:, 0] = torch.clamp(targets[:, 0], 0.15, 0.40)   # X: KESİNLİKLE ÖNDE!
        targets[:, 1] = torch.clamp(targets[:, 1], 0.00, 0.35)   # Y: sol taraf (pozitif veya 0)
        targets[:, 2] = torch.clamp(targets[:, 2], -0.15, 0.35)  # Z: el seviyesi

        self.left_target_pos[env_ids] = targets

        # Marker güncelle
        root_pos_ids = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos_ids + targets
        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num_samples, -1)
        target_pose = torch.cat([target_world, default_quat], dim=-1)
        self.left_target_obj.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    def _get_observations(self) -> dict:
        """Get observations for both arms."""
        root_pos = self.robot.data.root_pos_w

        # Right arm
        right_joint_pos = self.robot.data.joint_pos[:, self.right_arm_indices]
        right_joint_vel = self.robot.data.joint_vel[:, self.right_arm_indices]
        right_ee_pos = self._compute_right_ee_pos()
        right_ee_rel = right_ee_pos - root_pos
        right_error = self.right_target_pos - right_ee_rel

        # Left arm
        left_joint_pos = self.robot.data.joint_pos[:, self.left_arm_indices]
        left_joint_vel = self.robot.data.joint_vel[:, self.left_arm_indices]
        left_ee_pos = self._compute_left_ee_pos()
        left_ee_rel = left_ee_pos - root_pos
        left_error = self.left_target_pos - left_ee_rel

        # Update EE markers
        right_quat = self.robot.data.body_quat_w[:, self.right_palm_idx]
        right_marker_pose = torch.cat([right_ee_pos, right_quat], dim=-1)
        self.right_ee_marker.write_root_pose_to_sim(right_marker_pose)

        left_quat = self.robot.data.body_quat_w[:, self.left_palm_idx]
        left_marker_pose = torch.cat([left_ee_pos, left_quat], dim=-1)
        self.left_ee_marker.write_root_pose_to_sim(left_marker_pose)

        obs = torch.cat([
            right_joint_pos, right_joint_vel * 0.1, self.right_target_pos, right_ee_rel, right_error,
            left_joint_pos, left_joint_vel * 0.1, self.left_target_pos, left_ee_rel, left_error,
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for both arms."""
        root_pos = self.robot.data.root_pos_w

        # Right arm
        right_ee_rel = self._compute_right_ee_pos() - root_pos
        right_dist = (right_ee_rel - self.right_target_pos).norm(dim=-1)

        right_reached = right_dist < self.cfg.pos_threshold
        reached_right_envs = torch.where(right_reached)[0]
        if len(reached_right_envs) > 0:
            self._sample_right_target(reached_right_envs)
            self.right_reach_count[reached_right_envs] += 1

        # Left arm
        left_ee_rel = self._compute_left_ee_pos() - root_pos
        left_dist = (left_ee_rel - self.left_target_pos).norm(dim=-1)

        left_reached = left_dist < self.cfg.pos_threshold
        reached_left_envs = torch.where(left_reached)[0]
        if len(reached_left_envs) > 0:
            self._sample_left_target(reached_left_envs)
            self.left_reach_count[reached_left_envs] += 1

        # Combined reward
        reward_right = self.cfg.reward_reaching * right_reached.float() - right_dist
        reward_left = self.cfg.reward_reaching * left_reached.float() - left_dist

        self.prev_right_distance = right_dist
        self.prev_left_distance = left_dist

        return reward_right + reward_left

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

        # Set default poses for both arms
        for i, joint_name in enumerate(G1_RIGHT_ARM_JOINTS):
            idx = self.right_arm_indices[i]
            joint_pos[:, idx] = DEFAULT_ARM_POSE.get(joint_name, 0.0)

        for i, joint_name in enumerate(G1_LEFT_ARM_JOINTS):
            idx = self.left_arm_indices[i]
            joint_pos[:, idx] = DEFAULT_ARM_POSE.get(joint_name, 0.0)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._sample_right_target(env_ids)
        self._sample_left_target(env_ids)

        self.right_smoothed_actions[env_ids] = 0.0
        self.left_smoothed_actions[env_ids] = 0.0
        self.right_reach_count[env_ids] = 0.0
        self.left_reach_count[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions (10 dim: 5 right + 5 left)."""
        self.actions = actions

    def _apply_action(self):
        """Apply actions to both arms."""
        alpha = self.cfg.action_smoothing_alpha

        # Split actions
        right_actions = self.actions[:, :5]
        left_actions = self.actions[:, 5:]

        self.right_smoothed_actions = alpha * right_actions + (1 - alpha) * self.right_smoothed_actions
        self.left_smoothed_actions = alpha * left_actions + (1 - alpha) * self.left_smoothed_actions

        # Fix root
        self.robot.write_root_pose_to_sim(self.fixed_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel)

        # Right arm
        right_current = self.robot.data.joint_pos[:, self.right_arm_indices]
        right_target = right_current + self.right_smoothed_actions * self.cfg.action_scale
        right_target = torch.clamp(right_target, self.right_joint_lower, self.right_joint_upper)

        # Left arm
        left_current = self.robot.data.joint_pos[:, self.left_arm_indices]
        left_target = left_current + self.left_smoothed_actions * self.cfg.action_scale
        left_target = torch.clamp(left_target, self.left_joint_lower, self.left_joint_upper)

        # Combine
        joint_targets = self.robot.data.joint_pos.clone()
        joint_targets[:, self.right_arm_indices] = right_target
        joint_targets[:, self.left_arm_indices] = left_target

        self.robot.set_joint_position_target(joint_targets)