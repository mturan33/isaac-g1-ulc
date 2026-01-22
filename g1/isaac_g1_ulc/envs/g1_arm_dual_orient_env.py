"""
G1 Arm Reach with Orientation - Stage 5 V3.3 (SMOOTH + STABLE)
==============================================================

V3.2 → V3.3 DEĞİŞİKLİKLER:

1. SMOOTHNESS İYİLEŞTİRMELERİ:
   - action_smoothing_alpha: 0.3 → 0.15 (daha smooth geçişler)
   - reward_action_rate: -0.005 → -0.02 (ani hareketleri cezalandır)
   - reward_joint_vel: -0.0005 → -0.003 (hız cezası artır)
   - reward_joint_acc: YENİ - ivme cezası (titreme önleme)

2. CURRICULUM YAVAŞLATMA:
   - min_success_rate: 0.50 → 0.65 (daha yüksek başarı gerekli)
   - min_reaches_to_advance: 20 → 60 (daha fazla reach)
   - min_steps_per_stage: 100 → 300 (daha uzun stage'ler)

3. STABILITY:
   - Daha düşük action_scale ile başla
   - Smooth ramp-up for actions
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


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
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


@configclass
class G1ArmReachSceneCfg(InteractiveSceneCfg):

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
                damping=20.0,  # Increased damping for smoother motion
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
                emissive_color=(0.0, 0.5, 0.0)
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
                emissive_color=(0.5, 0.25, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )

    shoulder_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShoulderMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                emissive_color=(0.5, 0.5, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.174, 1.259)),
    )


@configclass
class G1ArmReachEnvCfg(DirectRLEnvCfg):

    decimation = 4
    episode_length_s = 10.0

    num_actions = 5
    num_observations = 29
    num_states = 0

    action_space = 5
    observation_space = 29
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

    scene: G1ArmReachSceneCfg = G1ArmReachSceneCfg(num_envs=1, env_spacing=2.0)

    # ============ POTENTIAL-BASED REWARD SHAPING ============
    use_potential_shaping = True
    potential_gamma = 0.99
    potential_sigma = 0.15
    potential_scale = 3.0

    # ============ PROXIMITY BONUS ZONES ============
    proximity_zone1_dist = 0.15
    proximity_zone1_bonus = 1.0
    proximity_zone2_dist = 0.10
    proximity_zone2_bonus = 2.0
    proximity_zone3_dist = 0.05
    proximity_zone3_bonus = 5.0

    # ============ REWARDS ============
    reward_pos_tanh_std = 0.12
    reward_pos_tanh_weight = 1.5

    reward_ori_tanh_std = 0.50
    reward_ori_tanh_weight = 0.3

    reward_reaching = 150.0

    # ============ V3.3: SMOOTHNESS PENALTIES (ARTTIRILDI) ============
    reward_action_rate = -0.02      # -0.005 → -0.02 (4x artış)
    reward_joint_vel = -0.003       # -0.0005 → -0.003 (6x artış)
    reward_joint_acc = -0.01        # YENİ: İvme cezası (titreme önleme)

    # ============ PCCL THRESHOLDS ============
    stage_thresholds = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06]
    ori_threshold = 0.50

    # Backward compatibility
    pos_threshold = 0.15  # Initial value, will be updated by curriculum

    # Episode settings
    terminate_on_success = True
    max_reaches_per_episode = 5

    # ============ V3.3: SMOOTHNESS (ACTION) ============
    action_smoothing_alpha = 0.15   # 0.3 → 0.15 (daha smooth)
    action_scale = 0.12             # 0.15 → 0.12 (daha küçük adımlar)

    # ============ WORKSPACE ============
    shoulder_center_offset = [0.0, 0.174, 0.259]
    workspace_inner_radius = 0.08
    workspace_outer_radius = 0.45

    # ============ V3.3: SLOWER CURRICULUM ============
    initial_spawn_radius = 0.15
    max_spawn_radius = 0.40
    curriculum_stages = 10

    min_success_rate = 0.65         # 0.50 → 0.65 (daha yüksek başarı)
    min_reaches_to_advance = 60     # 20 → 60 (3x fazla reach)
    min_steps_per_stage = 300       # 100 → 300 (3x uzun stage)

    # ============ ARM POSITION PERSISTENCE ============
    persist_arm_position = True
    random_start_probability = 0.15

    # ============ HINDSIGHT ============
    use_hindsight_bonus = True
    hindsight_threshold = 0.20
    hindsight_bonus = 10.0


class G1ArmReachEnv(DirectRLEnv):

    cfg: G1ArmReachEnvCfg

    def __init__(self, cfg: G1ArmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]
        self.shoulder_marker = self.scene["shoulder_marker"]

        self.outer_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/OuterWorkspace",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.3, 1.0),
                            emissive_color=(0.0, 0.2, 0.5),
                        ),
                    ),
                },
            )
        )

        self.inner_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/InnerExclusion",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),
                            emissive_color=(0.8, 0.0, 0.0),
                        ),
                    ),
                },
            )
        )

        self.num_wireframe_points = 24

        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        self.arm_indices = []
        for jn in G1_RIGHT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.arm_indices.append(i)
                    break
        self.arm_indices = torch.tensor(self.arm_indices, device=self.device, dtype=torch.long)

        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.palm_idx = i
                break

        self.joint_lower = torch.zeros(5, device=self.device)
        self.joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS):
            self.joint_lower[i], self.joint_upper[i] = ARM_JOINT_LIMITS[jn]

        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.tensor(
            [[0.707, 0.707, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1).clone()

        self.smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)

        # V3.3: Previous velocity for acceleration calculation
        self.prev_joint_vel = torch.zeros((self.num_envs, 5), device=self.device)

        self.shoulder_center = torch.tensor(
            self.cfg.shoulder_center_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # ARM POSITION PERSISTENCE
        self.last_reached_joint_pos = torch.zeros((self.num_envs, 5), device=self.device)
        self.default_arm_pos = torch.tensor(
            [DEFAULT_ARM_POSE[jn] for jn in G1_RIGHT_ARM_JOINTS],
            device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()
        self.last_reached_joint_pos[:] = self.default_arm_pos
        self.reached_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # POTENTIAL-BASED SHAPING STATE
        self.prev_potential = torch.zeros(self.num_envs, device=self.device)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)

        # HINDSIGHT TRACKING
        self.min_distance_in_episode = torch.full((self.num_envs,), 1.0, device=self.device)

        # Success tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.attempt_count = torch.zeros(self.num_envs, device=self.device)

        self.total_reaches = 0
        self.total_attempts = 0

        # Curriculum
        self.current_spawn_radius = self.cfg.initial_spawn_radius
        self.curriculum_stage = 0
        self.stage_reaches = 0
        self.stage_attempts = 0
        self.stage_step_count = 0
        self.orientation_enabled = False

        # Current threshold (changes with curriculum)
        self.current_pos_threshold = self.cfg.stage_thresholds[0]

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Validation
        assert self.cfg.initial_spawn_radius > self.cfg.workspace_inner_radius, \
            f"BUG: initial_spawn_radius ({self.cfg.initial_spawn_radius}) must be > workspace_inner_radius ({self.cfg.workspace_inner_radius})"

        print("\n" + "=" * 70)
        print("G1 ARM REACH ENVIRONMENT - V3.3 (SMOOTH + STABLE)")
        print("=" * 70)
        print(f"  Arm joints: {self.arm_indices.tolist()}")
        print(f"  Palm idx: {self.palm_idx}")
        print("-" * 70)
        print("  🆕 V3.3 SMOOTHNESS IMPROVEMENTS:")
        print(f"    ✓ action_smoothing_alpha: {self.cfg.action_smoothing_alpha}")
        print(f"    ✓ action_scale: {self.cfg.action_scale}")
        print(f"    ✓ reward_action_rate: {self.cfg.reward_action_rate}")
        print(f"    ✓ reward_joint_vel: {self.cfg.reward_joint_vel}")
        print(f"    ✓ reward_joint_acc: {self.cfg.reward_joint_acc} (YENİ)")
        print("-" * 70)
        print("  🆕 V3.3 SLOWER CURRICULUM:")
        print(f"    ✓ min_success_rate: {self.cfg.min_success_rate*100:.0f}%")
        print(f"    ✓ min_reaches_to_advance: {self.cfg.min_reaches_to_advance}")
        print(f"    ✓ min_steps_per_stage: {self.cfg.min_steps_per_stage}")
        print("-" * 70)
        print(f"  PCCL THRESHOLDS: {[f'{t*100:.0f}cm' for t in self.cfg.stage_thresholds]}")
        print("=" * 70 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    def _compute_ee_pos(self) -> torch.Tensor:
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_ee_quat(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.palm_idx]

    def _compute_potential(self, distance: torch.Tensor) -> torch.Tensor:
        """Exponential potential function: Φ(s) = exp(-d/σ)"""
        return torch.exp(-distance / self.cfg.potential_sigma)

    def _sample_target_in_workspace(self, env_ids: torch.Tensor):
        """Sample targets in valid workspace region."""
        num = len(env_ids)
        root_pos = self.robot.data.root_pos_w[env_ids]
        shoulder_rel = self.shoulder_center[env_ids]

        direction = torch.randn((num, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction[:, 0] = -torch.abs(direction[:, 0])

        inner = self.cfg.workspace_inner_radius
        outer = max(self.current_spawn_radius, inner + 0.05)
        outer = min(outer, self.cfg.workspace_outer_radius)

        distance = inner + torch.rand((num, 1), device=self.device) * (outer - inner)

        targets = shoulder_rel + direction * distance
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.55)

        self.target_pos[env_ids] = targets

        target_world = root_pos + targets
        pose = torch.cat([target_world, self.target_quat[env_ids]], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Initialize potential for new targets
        ee_pos = self._compute_ee_pos()[env_ids] - root_pos
        initial_dist = (ee_pos - targets).norm(dim=-1)
        self.prev_distance[env_ids] = initial_dist
        self.prev_potential[env_ids] = self._compute_potential(initial_dist)
        self.min_distance_in_episode[env_ids] = initial_dist

        self.attempt_count[env_ids] += 1
        self.stage_attempts += len(env_ids)
        self.total_attempts += len(env_ids)

    def update_curriculum(self, iteration: int):
        self.stage_step_count += 1

        if self.stage_attempts > 0:
            current_success_rate = self.stage_reaches / self.stage_attempts
        else:
            current_success_rate = 0.0

        should_advance = (
            current_success_rate >= self.cfg.min_success_rate and
            self.stage_reaches >= self.cfg.min_reaches_to_advance and
            self.stage_step_count >= self.cfg.min_steps_per_stage and
            self.curriculum_stage < self.cfg.curriculum_stages - 1
        )

        if should_advance:
            self.curriculum_stage += 1

            # Update spawn radius
            radius_increment = (self.cfg.max_spawn_radius - self.cfg.initial_spawn_radius) / (self.cfg.curriculum_stages - 1)
            self.current_spawn_radius = self.cfg.initial_spawn_radius + self.curriculum_stage * radius_increment

            # Update threshold (PCCL)
            self.current_pos_threshold = self.cfg.stage_thresholds[min(self.curriculum_stage, len(self.cfg.stage_thresholds) - 1)]

            if self.curriculum_stage >= 5:
                self.orientation_enabled = True

            print(f"\n{'='*60}")
            print(f"🎯 CURRICULUM ADVANCED TO STAGE {self.curriculum_stage + 1}/{self.cfg.curriculum_stages}")
            print(f"   Success rate: {current_success_rate*100:.1f}% ({self.stage_reaches}/{self.stage_attempts})")
            print(f"   Spawn radius: {self.current_spawn_radius*100:.0f}cm")
            print(f"   Threshold: {self.current_pos_threshold*100:.0f}cm (PCCL)")
            print(f"   Orientation: {'ENABLED' if self.orientation_enabled else 'disabled'}")
            print(f"{'='*60}\n")

            self.stage_reaches = 0
            self.stage_attempts = 0
            self.stage_step_count = 0

        return current_success_rate

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w

        joint_pos = self.robot.data.joint_pos[:, self.arm_indices]
        joint_vel = self.robot.data.joint_vel[:, self.arm_indices]

        ee_pos = self._compute_ee_pos() - root_pos
        ee_quat = self._compute_ee_quat()

        pos_err = self.target_pos - ee_pos
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        ori_err = quat_diff_rad(ee_quat, self.target_quat).unsqueeze(-1)

        self.ee_marker.write_root_pose_to_sim(
            torch.cat([self._compute_ee_pos(), ee_quat], dim=-1)
        )

        self._update_workspace_spheres()

        obs = torch.cat([
            joint_pos,
            joint_vel * 0.1,
            self.target_pos,
            self.target_quat,
            ee_pos,
            ee_quat,
            pos_err,
            ori_err,
            pos_dist / 0.5,
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w

        ee_pos = self._compute_ee_pos() - root_pos
        pos_dist = (ee_pos - self.target_pos).norm(dim=-1)

        ee_quat = self._compute_ee_quat()
        ori_dist = quat_diff_rad(ee_quat, self.target_quat)

        # POTENTIAL-BASED REWARD SHAPING
        if self.cfg.use_potential_shaping:
            current_potential = self._compute_potential(pos_dist)
            potential_reward = self.cfg.potential_scale * (
                self.cfg.potential_gamma * current_potential - self.prev_potential
            )
            self.prev_potential = current_potential
        else:
            potential_reward = torch.zeros_like(pos_dist)

        # DENSE: TANH KERNEL
        pos_reward = 1.0 - torch.tanh(pos_dist / self.cfg.reward_pos_tanh_std)

        if self.orientation_enabled:
            ori_reward = 1.0 - torch.tanh(ori_dist / self.cfg.reward_ori_tanh_std)
        else:
            ori_reward = torch.zeros_like(pos_dist)

        # PROXIMITY BONUS ZONES
        proximity_bonus = torch.zeros_like(pos_dist)
        in_zone1 = pos_dist < self.cfg.proximity_zone1_dist
        proximity_bonus = torch.where(in_zone1, proximity_bonus + self.cfg.proximity_zone1_bonus, proximity_bonus)
        in_zone2 = pos_dist < self.cfg.proximity_zone2_dist
        proximity_bonus = torch.where(in_zone2, proximity_bonus + self.cfg.proximity_zone2_bonus, proximity_bonus)
        in_zone3 = pos_dist < self.cfg.proximity_zone3_dist
        proximity_bonus = torch.where(in_zone3, proximity_bonus + self.cfg.proximity_zone3_bonus, proximity_bonus)

        # SPARSE: REACH DETECTION
        pos_reached = pos_dist < self.current_pos_threshold
        if self.orientation_enabled:
            ori_reached = ori_dist < self.cfg.ori_threshold
            fully_reached = pos_reached & ori_reached
        else:
            fully_reached = pos_reached

        reached_ids = torch.where(fully_reached)[0]
        if len(reached_ids) > 0:
            self.reach_count[reached_ids] += 1
            self.episode_reach_count[reached_ids] += 1
            self.stage_reaches += len(reached_ids)
            self.total_reaches += len(reached_ids)

            current_arm_pos = self.robot.data.joint_pos[reached_ids][:, self.arm_indices]
            self.last_reached_joint_pos[reached_ids] = current_arm_pos
            self.reached_success[reached_ids] = True

            self._sample_target_in_workspace(reached_ids)

        # HINDSIGHT TRACKING
        self.min_distance_in_episode = torch.minimum(self.min_distance_in_episode, pos_dist)
        self.prev_distance = pos_dist

        # ============ V3.3: SMOOTHNESS PENALTIES ============
        action_rate = (self.smoothed_actions - self.prev_actions).norm(dim=-1)

        joint_vel = self.robot.data.joint_vel[:, self.arm_indices]
        joint_vel_norm = joint_vel.norm(dim=-1)

        # V3.3: Acceleration penalty (change in velocity = jerk indicator)
        joint_acc = (joint_vel - self.prev_joint_vel).norm(dim=-1)
        self.prev_joint_vel = joint_vel.clone()

        # TOTAL REWARD
        reward = (
            potential_reward +
            self.cfg.reward_pos_tanh_weight * pos_reward +
            self.cfg.reward_ori_tanh_weight * ori_reward +
            proximity_bonus +
            self.cfg.reward_reaching * fully_reached.float() +
            self.cfg.reward_action_rate * action_rate +
            self.cfg.reward_joint_vel * joint_vel_norm +
            self.cfg.reward_joint_acc * joint_acc  # V3.3: Acceleration penalty
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.terminate_on_success:
            terminated = self.episode_reach_count >= self.cfg.max_reaches_per_episode
        else:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        self.robot.write_root_pose_to_sim(self.fixed_root_pose[env_ids], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel[env_ids], env_ids=env_ids)

        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.robot.data.joint_vel[env_ids])

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()

            use_random_start = torch.rand(1).item() < self.cfg.random_start_probability

            if self.cfg.persist_arm_position and self.reached_success[env_id_item] and not use_random_start:
                jp[i, self.arm_indices] = self.last_reached_joint_pos[env_id_item]
            else:
                for j, jn in enumerate(G1_RIGHT_ARM_JOINTS):
                    jp[i, self.arm_indices[j]] = DEFAULT_ARM_POSE[jn]
                self.last_reached_joint_pos[env_id_item] = self.default_arm_pos[0]

        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        self._sample_target_in_workspace(env_ids)

        self.smoothed_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0  # V3.3: Reset velocity tracking
        self.episode_reach_count[env_ids] = 0.0
        self.reached_success[env_ids] = False
        self.min_distance_in_episode[env_ids] = 1.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions

    def _apply_action(self):
        alpha = self.cfg.action_smoothing_alpha

        self.prev_actions = self.smoothed_actions.clone()
        self.smoothed_actions = alpha * self.actions + (1 - alpha) * self.smoothed_actions

        self.robot.write_root_pose_to_sim(self.fixed_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel)

        cur_pos = self.robot.data.joint_pos[:, self.arm_indices]
        tgt_pos = torch.clamp(
            cur_pos + self.smoothed_actions * self.cfg.action_scale,
            self.joint_lower, self.joint_upper
        )

        jt = self.robot.data.joint_pos.clone()
        jt[:, self.arm_indices] = tgt_pos
        self.robot.set_joint_position_target(jt)

    def _update_workspace_spheres(self):
        root_pos = self.robot.data.root_pos_w
        shoulder_world = root_pos + self.shoulder_center

        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        shoulder_pose = torch.cat([shoulder_world, identity_quat], dim=-1)
        self.shoulder_marker.write_root_pose_to_sim(shoulder_pose)

        shoulder_0 = shoulder_world[0].cpu()
        n = self.num_wireframe_points
        angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]

        outer_points = []
        radius = self.cfg.workspace_outer_radius
        for angle in angles:
            x = shoulder_0[0] - radius * torch.cos(angle)
            y = shoulder_0[1] + radius * torch.sin(angle)
            z = shoulder_0[2]
            if x <= shoulder_0[0] + 0.05:
                outer_points.append([x.item(), y.item(), z.item()])

        for angle in angles:
            x = shoulder_0[0] - radius * torch.sin(angle)
            y = shoulder_0[1]
            z = shoulder_0[2] + radius * torch.cos(angle)
            if x <= shoulder_0[0] + 0.05:
                outer_points.append([x.item(), y.item(), z.item()])

        if outer_points:
            outer_pos = torch.tensor(outer_points, device=self.device)
            outer_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(outer_points), -1)
            self.outer_markers.visualize(translations=outer_pos, orientations=outer_quat)

        inner_points = []
        radius = self.cfg.workspace_inner_radius
        for angle in angles:
            x = shoulder_0[0] - radius * torch.cos(angle)
            y = shoulder_0[1] + radius * torch.sin(angle)
            z = shoulder_0[2]
            inner_points.append([x.item(), y.item(), z.item()])

        for angle in angles:
            x = shoulder_0[0] - radius * torch.sin(angle)
            y = shoulder_0[1]
            z = shoulder_0[2] + radius * torch.cos(angle)
            inner_points.append([x.item(), y.item(), z.item()])

        inner_pos = torch.tensor(inner_points, device=self.device)
        inner_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(inner_points), -1)
        self.inner_markers.visualize(translations=inner_pos, orientations=inner_quat)

    def get_success_rate(self) -> float:
        if self.total_attempts > 0:
            return self.total_reaches / self.total_attempts
        return 0.0


# ============ BACKWARD COMPATIBILITY ALIASES ============
G1ArmOrientEnv = G1ArmReachEnv
G1ArmOrientEnvCfg = G1ArmReachEnvCfg