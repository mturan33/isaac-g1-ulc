"""
G1 Arm Reach with Orientation - Stage 5 (V2 - Dense Shaping)
=============================================================

Sadece SAĞ KOL - Position + Orientation (Palm Down) reaching.

V2 DEĞİŞİKLİKLER:
- Dense progress reward (hedefe yaklaştıkça bonus)
- Daha yüksek reaching bonus (500)
- Daha düşük distance penalty (-0.5)
- Daha kolay curriculum threshold (10.0)
- Daha yakın başlangıç (3cm)
- Daha geniş pos tolerance (10cm)
- Daha uzun timeout (6 saniye)

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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


EE_OFFSET = 0.02

# Sağ omuz merkezi (root'a göre relatif) - workspace'in merkezi
RIGHT_SHOULDER_CENTER = torch.tensor([0.0, 0.174, 0.259])

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


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Create quaternion from axis-angle representation."""
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

    # ⚪ Beyaz küre - Omuz merkezi marker (solid)
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
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    scene: G1ArmOrientSceneCfg = G1ArmOrientSceneCfg(num_envs=1, env_spacing=2.0)

    # ============ V2: UPDATED REWARDS ============
    reward_reaching = 500.0          # ⬆️ 100 → 500 (büyük bonus)
    reward_pos_distance = -0.5       # ⬆️ -2.0 → -0.5 (düşük penalty)
    reward_ori_distance = -0.1       # ⬆️ -0.3 → -0.1 (düşük penalty)
    reward_action_rate = -0.001      # ⬆️ -0.005 → -0.001 (toleranslı)
    reward_progress = 15.0           # 🆕 Dense progress reward coefficient

    # ============ V2: UPDATED THRESHOLDS ============
    pos_threshold = 0.10             # ⬆️ 7cm → 10cm (daha kolay)
    ori_threshold = 0.50             # ⬆️ 0.40 → 0.50 rad (~29°, toleranslı)

    # Action
    action_smoothing_alpha = 0.5
    action_scale = 0.08

    # Workspace - Final hedef (omuz merkezi etrafında yarım küre)
    shoulder_center_offset = [0.0, 0.174, 0.259]
    final_workspace_radius = 0.40

    # ============ V2: UPDATED CURRICULUM ============
    initial_spawn_radius = 0.03       # ⬆️ 5cm → 3cm (daha kolay başlangıç)
    max_spawn_radius = 1.00
    curriculum_stages = 20

    # Reward-based curriculum progression
    reward_threshold_to_advance = 10.0  # ⬆️ 30 → 10 (daha kolay ilerleme)
    min_steps_per_stage = 100           # ⬆️ 150 → 100 (daha hızlı ilerleme)


class G1ArmOrientEnv(DirectRLEnv):

    cfg: G1ArmOrientEnvCfg

    def __init__(self, cfg: G1ArmOrientEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

        # Workspace visualization spheres
        self.shoulder_marker = self.scene["shoulder_marker"]

        # Wireframe visualization with VisualizationMarkers
        # Outer workspace (40cm) - mavi noktalar
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

        # Inner exclusion (10cm) - kırmızı noktalar
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

        # Curriculum radius (EE etrafında) - sarı noktalar
        self.curriculum_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CurriculumSphere",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 1.0, 0.0),
                            emissive_color=(0.5, 0.5, 0.0),
                        ),
                    ),
                },
            )
        )

        # Wireframe noktalarının sayısı (çember başına)
        self.num_wireframe_points = 24

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
        self.target_quat = torch.tensor(
            [[0.707, 0.707, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1).clone()

        # Action smoothing
        self.smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 5), device=self.device)

        # Tracking
        self.reach_count = torch.zeros(self.num_envs, device=self.device)

        # 🆕 V2: Previous distance for progress reward
        self.prev_pos_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_ori_dist = torch.zeros(self.num_envs, device=self.device)

        # Workspace - omuz merkezi (root'a göre)
        self.shoulder_center = torch.tensor(
            self.cfg.shoulder_center_offset, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Curriculum - Reward-based progression
        self.current_spawn_radius = self.cfg.initial_spawn_radius
        self.curriculum_stage = 0
        self.stage_step_count = 0
        self.recent_rewards = []
        self.orientation_weight = 0.0

        # ============ V2: UPDATED TIMEOUT ============
        self.timeout_steps = 180  # ⬆️ 90 → 180 (6 saniye)
        self.target_timer = torch.zeros(self.num_envs, device=self.device)

        # Forward vector for EE offset
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print("\n" + "=" * 70)
        print("G1 ARM ORIENT ENVIRONMENT - STAGE 5 V2 (DENSE SHAPING)")
        print("=" * 70)
        print(f"  Arm joints: {self.arm_indices.tolist()}")
        print(f"  Palm idx: {self.palm_idx}")
        print("-" * 70)
        print("  V2 CHANGES:")
        print(f"    ✓ Dense progress reward: {self.cfg.reward_progress}")
        print(f"    ✓ Reaching bonus: {self.cfg.reward_reaching} (was 100)")
        print(f"    ✓ Distance penalty: {self.cfg.reward_pos_distance} (was -2.0)")
        print(f"    ✓ Curriculum threshold: {self.cfg.reward_threshold_to_advance} (was 30)")
        print(f"    ✓ Initial spawn: {self.cfg.initial_spawn_radius*100:.0f}cm (was 5cm)")
        print(f"    ✓ Pos threshold: {self.cfg.pos_threshold*100:.0f}cm (was 7cm)")
        print(f"    ✓ Timeout: {self.timeout_steps} steps (was 90)")
        print("-" * 70)
        print("  WORKSPACE (Omuz Merkezi Etrafında Yarım Küre):")
        print(f"    Inner radius: 10cm (exclusion zone)")
        print(f"    Outer radius: {self.cfg.final_workspace_radius*100:.0f}cm")
        print("-" * 70)
        print("  CURRICULUM (20 Aşama - Reward-based):")
        print(f"    EE-relative spawn: 3cm → 100cm")
        print(f"    Reward > {self.cfg.reward_threshold_to_advance} olunca sonraki stage")
        print("-" * 70)
        print(f"  Stage 1-10: Sadece POSITION")
        print(f"  Stage 11-20: Position + ORIENTATION")
        print(f"    Pos threshold: {self.cfg.pos_threshold}m")
        print(f"    Ori threshold: {self.cfg.ori_threshold:.2f} rad (~{math.degrees(self.cfg.ori_threshold):.0f}°)")
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
        """
        Sample target - EE-relative spawn + YARIM KÜRE CLAMP.
        """
        num = len(env_ids)
        root_pos = self.robot.data.root_pos_w[env_ids]

        # Mevcut EE pozisyonu (root'a göre relatif)
        ee_pos_world = self._compute_ee_pos()
        current_ee_rel = (ee_pos_world - self.robot.data.root_pos_w)[env_ids]

        # Random yön
        direction = torch.randn((num, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Mesafe: pos_threshold+2cm ile current_spawn_radius arası
        min_dist = self.cfg.pos_threshold + 0.02
        max_dist = max(self.current_spawn_radius, min_dist + 0.01)
        distance = min_dist + torch.rand((num, 1), device=self.device) * (max_dist - min_dist)

        # Target = current EE + offset
        targets = current_ee_rel + direction * distance

        # ========== YARIM KÜRE CLAMP (Omuz merkezi etrafında) ==========
        shoulder_rel = self.shoulder_center[env_ids]

        # Hedefin omuz merkezine göre pozisyonu
        target_from_shoulder = targets - shoulder_rel

        # Mesafe hesapla
        dist_from_shoulder = target_from_shoulder.norm(dim=-1, keepdim=True)

        # Yön (normalize)
        target_dir = target_from_shoulder / (dist_from_shoulder + 1e-8)

        # Inner radius (10cm) ve outer radius (40cm) arasına clamp
        inner_radius = 0.10
        outer_radius = self.cfg.final_workspace_radius
        clamped_dist = torch.clamp(dist_from_shoulder, inner_radius, outer_radius)

        # Yarım küre: X <= 0 (robotun önünde)
        target_dir_clamped = target_dir.clone()
        target_dir_clamped[:, 0] = -torch.abs(target_dir[:, 0])

        # Yeni hedef pozisyonu
        targets = shoulder_rel + target_dir_clamped * clamped_dist

        # Z sınırlaması (yerden en az 5cm, max 55cm)
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.55)
        # =================================================================

        self.target_pos[env_ids] = targets

        # Update visual marker
        target_world = root_pos + targets
        pose = torch.cat([target_world, self.target_quat[env_ids]], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

        # Reset timer
        self.target_timer[env_ids] = 0

        # 🆕 V2: Initialize previous distances for new targets
        ee_pos = self._compute_ee_pos() - self.robot.data.root_pos_w
        self.prev_pos_dist[env_ids] = (ee_pos[env_ids] - targets).norm(dim=-1)
        ee_quat = self._compute_ee_quat()
        self.prev_ori_dist[env_ids] = quat_diff_rad(ee_quat[env_ids], self.target_quat[env_ids])

    def update_curriculum(self, iteration: int, mean_reward: float = None):
        """
        Update curriculum - REWARD-BASED progression.
        """
        self.stage_step_count += 1

        # Mean reward tracking
        if mean_reward is not None:
            self.recent_rewards.append(mean_reward)
            if len(self.recent_rewards) > 100:
                self.recent_rewards.pop(0)

        # Check if should advance to next stage
        if len(self.recent_rewards) >= 50:
            avg_recent_reward = sum(self.recent_rewards[-50:]) / 50

            if (avg_recent_reward > self.cfg.reward_threshold_to_advance and
                self.stage_step_count >= self.cfg.min_steps_per_stage and
                self.curriculum_stage < self.cfg.curriculum_stages - 1):

                # Advance to next stage!
                self.curriculum_stage += 1
                self.stage_step_count = 0
                self.recent_rewards = []

                # Update spawn radius (+5cm per stage)
                radius_increment = (self.cfg.max_spawn_radius - self.cfg.initial_spawn_radius) / (self.cfg.curriculum_stages - 1)
                self.current_spawn_radius = self.cfg.initial_spawn_radius + self.curriculum_stage * radius_increment

                # Update orientation weight (stage 11+ için)
                if self.curriculum_stage >= 10:
                    self.orientation_weight = (self.curriculum_stage - 10) / 9.0
                else:
                    self.orientation_weight = 0.0

                print(f"\n{'='*60}")
                print(f"🎯 CURRICULUM ADVANCED TO STAGE {self.curriculum_stage + 1}/{self.cfg.curriculum_stages}")
                print(f"   Spawn radius: {self.current_spawn_radius:.2f}m")
                print(f"   Orientation weight: {self.orientation_weight:.2f}")
                print(f"{'='*60}\n")

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

        # Update workspace visualization spheres
        self._update_workspace_spheres()

        obs = torch.cat([
            joint_pos,                    # 5
            joint_vel * 0.1,              # 5
            self.target_pos,              # 3
            self.target_quat,             # 4
            ee_pos,                       # 3
            ee_quat,                      # 4
            pos_err,                      # 3
            ori_err,                      # 1
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

        # ============ V2: DENSE PROGRESS REWARD ============
        # Hedefe yaklaştıkça pozitif reward
        pos_progress = self.prev_pos_dist - pos_dist  # Pozitif = yaklaşıyor
        ori_progress = self.prev_ori_dist - ori_dist

        # Update previous distances
        self.prev_pos_dist = pos_dist.clone()
        self.prev_ori_dist = ori_dist.clone()
        # ===================================================

        # Check reaching - Stage 1-10: sadece position, Stage 11+: position + orientation
        pos_reached = pos_dist < self.cfg.pos_threshold
        ori_reached = ori_dist < self.cfg.ori_threshold

        if self.curriculum_stage < 10:
            fully_reached = pos_reached
        else:
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

        # ============ V2: UPDATED REWARD CALCULATION ============
        reward = (
            # Sparse reaching bonus
            self.cfg.reward_reaching * fully_reached.float() +

            # Dense progress reward (main learning signal)
            self.cfg.reward_progress * pos_progress +
            self.cfg.reward_progress * ori_progress * self.orientation_weight * 0.5 +

            # Distance penalties (soft)
            self.cfg.reward_pos_distance * pos_dist +
            self.cfg.reward_ori_distance * ori_dist * self.orientation_weight +

            # Action smoothness
            self.cfg.reward_action_rate * action_rate
        )
        # ========================================================

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

    def _update_workspace_spheres(self):
        """Update workspace wireframe visualization."""
        root_pos = self.robot.data.root_pos_w

        # Omuz merkezi world koordinatları
        shoulder_world = root_pos + self.shoulder_center

        # Identity quaternion
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Shoulder marker güncelle
        shoulder_pose = torch.cat([shoulder_world, identity_quat], dim=-1)
        self.shoulder_marker.write_root_pose_to_sim(shoulder_pose)

        # Wireframe çemberler için noktalar (sadece env 0)
        shoulder_0 = shoulder_world[0].cpu()
        ee_0 = self._compute_ee_pos()[0].cpu()

        n = self.num_wireframe_points
        angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]

        # 🔵 Outer workspace (40cm)
        outer_points = []
        radius = 0.40

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

        # 🔴 Inner exclusion (10cm)
        inner_points = []
        radius = 0.10

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

        # 🟡 Curriculum sphere (EE etrafında)
        curr_radius = self.current_spawn_radius
        curr_points = []
        for angle in angles:
            x = ee_0[0] + curr_radius * torch.cos(angle)
            y = ee_0[1] + curr_radius * torch.sin(angle)
            z = ee_0[2]
            curr_points.append([x.item(), y.item(), z.item()])

        curr_pos = torch.tensor(curr_points, device=self.device)
        curr_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(curr_points), -1)
        self.curriculum_markers.visualize(translations=curr_pos, orientations=curr_quat)