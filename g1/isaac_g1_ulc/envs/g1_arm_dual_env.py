"""
G1 Dual Arm Environment (Play Only)
====================================

ÇALIŞAN VERSİYON - 330+ reach

KOORDİNAT SİSTEMİ:
- X- = İleri (önde)
- Y+ = Sağ taraf
- Y- = Sol taraf
- Z+ = Yukarı

VISUAL MARKERS:
- 🟢 Yeşil küre  = Sağ kol target
- 🔵 Mavi küre   = Sol kol target
- 🟠 Turuncu    = Sağ el (EE)
- 🟣 Mor        = Sol el (EE)
- 🟩 Yeşil kutu = Sağ workspace
- 🟦 Mavi kutu  = Sol workspace
"""

from __future__ import annotations

import torch

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
# TRAINING WORKSPACE PARAMETRELERİ (g1_arm_reach_env.py'den)
# =============================================================================
# Training'deki _sample_target clamp değerleri:
# X: -0.30 to 0.40, Y: -0.30 to 0.50, Z: -0.10 to 0.60

# Sağ kol workspace (training clamp değerleri, root'a göre relatif)
RIGHT_WS_X_MIN, RIGHT_WS_X_MAX = -0.30, 0.40
RIGHT_WS_Y_MIN, RIGHT_WS_Y_MAX = -0.30, 0.50
RIGHT_WS_Z_MIN, RIGHT_WS_Z_MAX = -0.10, 0.60

# Sol kol workspace (Y ekseni mirrored)
LEFT_WS_X_MIN, LEFT_WS_X_MAX = -0.30, 0.40
LEFT_WS_Y_MIN, LEFT_WS_Y_MAX = -0.50, 0.30   # Y mirrored for left arm
LEFT_WS_Z_MIN, LEFT_WS_Z_MAX = -0.10, 0.60

# Kutu boyutları (görselleştirme için)
WS_SIZE_X = RIGHT_WS_X_MAX - RIGHT_WS_X_MIN  # 0.70
WS_SIZE_Y = RIGHT_WS_Y_MAX - RIGHT_WS_Y_MIN  # 0.80
WS_SIZE_Z = RIGHT_WS_Z_MAX - RIGHT_WS_Z_MIN  # 0.70

# Kutu merkezleri (world koordinatları, root z=1.0)
WS_RIGHT_CENTER_X = (RIGHT_WS_X_MIN + RIGHT_WS_X_MAX) / 2  # 0.05
WS_RIGHT_CENTER_Y = (RIGHT_WS_Y_MIN + RIGHT_WS_Y_MAX) / 2  # 0.10
WS_RIGHT_CENTER_Z = 1.0 + (RIGHT_WS_Z_MIN + RIGHT_WS_Z_MAX) / 2  # 1.25

WS_LEFT_CENTER_X = (LEFT_WS_X_MIN + LEFT_WS_X_MAX) / 2  # 0.05
WS_LEFT_CENTER_Y = (LEFT_WS_Y_MIN + LEFT_WS_Y_MAX) / 2  # -0.10
WS_LEFT_CENTER_Z = 1.0 + (LEFT_WS_Z_MIN + LEFT_WS_Z_MAX) / 2  # 1.25


@configclass
class G1DualArmSceneCfg(InteractiveSceneCfg):

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
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*", "torso.*"],
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

    # Yeşil küre - SAĞ KOL target
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

    # Mavi küre - SOL KOL target
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

    # Turuncu küre - SAĞ EL (EE)
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

    # Mor küre - SOL EL (EE)
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

    # Yeşil kutu - SAĞ KOL workspace (opacity yok, emissive ile parlak)
    right_workspace: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightWorkspace",
        spawn=sim_utils.CuboidCfg(
            size=(WS_SIZE_X, WS_SIZE_Y, WS_SIZE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.3, 0.0),
                emissive_color=(0.0, 0.5, 0.0),
                metallic=0.0,
                roughness=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(WS_RIGHT_CENTER_X, WS_RIGHT_CENTER_Y, WS_RIGHT_CENTER_Z)
        ),
    )

    # Mavi kutu - SOL KOL workspace
    left_workspace: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftWorkspace",
        spawn=sim_utils.CuboidCfg(
            size=(WS_SIZE_X, WS_SIZE_Y, WS_SIZE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.3),
                emissive_color=(0.0, 0.3, 0.5),
                metallic=0.0,
                roughness=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(WS_LEFT_CENTER_X, WS_LEFT_CENTER_Y, WS_LEFT_CENTER_Z)
        ),
    )


@configclass
class G1DualArmEnvCfg(DirectRLEnvCfg):

    decimation = 4
    episode_length_s = 300.0

    num_actions = 10
    num_observations = 38
    num_states = 0

    action_space = 10
    observation_space = 38
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

    reward_reaching = 20.0
    reward_pos_distance = -1.0
    action_smoothing_alpha = 0.5
    action_scale = 0.08
    pos_threshold = 0.05


def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


class G1DualArmEnv(DirectRLEnv):

    cfg: G1DualArmEnvCfg

    def __init__(self, cfg: G1DualArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.right_target_obj = self.scene["right_target"]
        self.left_target_obj = self.scene["left_target"]
        self.right_ee_marker = self.scene["right_ee_marker"]
        self.left_ee_marker = self.scene["left_ee_marker"]
        self.right_workspace = self.scene["right_workspace"]
        self.left_workspace = self.scene["left_workspace"]

        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        self.right_arm_indices = []
        for jn in G1_RIGHT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.right_arm_indices.append(i)
                    break
        self.right_arm_indices = torch.tensor(self.right_arm_indices, device=self.device, dtype=torch.long)

        self.left_arm_indices = []
        for jn in G1_LEFT_ARM_JOINTS:
            for i, name in enumerate(joint_names):
                if name == jn:
                    self.left_arm_indices.append(i)
                    break
        self.left_arm_indices = torch.tensor(self.left_arm_indices, device=self.device, dtype=torch.long)

        self.right_palm_idx = None
        self.left_palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "palm" in name.lower():
                self.right_palm_idx = i
            if "left" in name.lower() and "palm" in name.lower():
                self.left_palm_idx = i

        self.right_joint_lower = torch.zeros(5, device=self.device)
        self.right_joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS):
            self.right_joint_lower[i], self.right_joint_upper[i] = ARM_JOINT_LIMITS[jn]

        self.left_joint_lower = torch.zeros(5, device=self.device)
        self.left_joint_upper = torch.zeros(5, device=self.device)
        for i, jn in enumerate(G1_LEFT_ARM_JOINTS):
            self.left_joint_lower[i], self.left_joint_upper[i] = ARM_JOINT_LIMITS[jn]

        self.fixed_root_pose = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], device=self.device
        ).expand(self.num_envs, -1).clone()
        self.zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        self.right_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.right_smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.left_smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)

        self.right_reach_count = torch.zeros(self.num_envs, device=self.device)
        self.left_reach_count = torch.zeros(self.num_envs, device=self.device)

        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print("\n" + "=" * 70)
        print("G1 DUAL ARM ENVIRONMENT - ÇALIŞAN VERSİYON + WORKSPACE VIS")
        print("=" * 70)
        print(f"  Right arm: {self.right_arm_indices.tolist()}")
        print(f"  Left arm:  {self.left_arm_indices.tolist()}")
        print("-" * 70)
        print("  KOORDİNAT SİSTEMİ:")
        print("    X- = İleri (önde)")
        print("    Y+ = Sağ taraf")
        print("    Y- = Sol taraf")
        print("    Z+ = Yukarı")
        print("-" * 70)
        print("  SAĞ KOL WORKSPACE (yeşil kutu):")
        print(f"    X: {RIGHT_WS_X_MIN} to {RIGHT_WS_X_MAX} (önde)")
        print(f"    Y: {RIGHT_WS_Y_MIN} to {RIGHT_WS_Y_MAX} (sağ)")
        print(f"    Z: {RIGHT_WS_Z_MIN} to {RIGHT_WS_Z_MAX} (göğüs)")
        print("  SOL KOL WORKSPACE (mavi kutu):")
        print(f"    X: {LEFT_WS_X_MIN} to {LEFT_WS_X_MAX} (önde)")
        print(f"    Y: {LEFT_WS_Y_MIN} to {LEFT_WS_Y_MAX} (sol)")
        print(f"    Z: {LEFT_WS_Z_MIN} to {LEFT_WS_Z_MAX} (göğüs)")
        print("=" * 70 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.right_target_obj = self.scene["right_target"]
        self.left_target_obj = self.scene["left_target"]
        self.right_ee_marker = self.scene["right_ee_marker"]
        self.left_ee_marker = self.scene["left_ee_marker"]
        self.right_workspace = self.scene["right_workspace"]
        self.left_workspace = self.scene["left_workspace"]

    def _compute_right_ee_pos(self) -> torch.Tensor:
        palm_pos = self.robot.data.body_pos_w[:, self.right_palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.right_palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _compute_left_ee_pos(self) -> torch.Tensor:
        palm_pos = self.robot.data.body_pos_w[:, self.left_palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.left_palm_idx]
        forward = rotate_vector_by_quat(self.local_forward, palm_quat)
        return palm_pos + EE_OFFSET * forward

    def _sample_right_target(self, env_ids: torch.Tensor):
        """SAĞ KOL - TRAINING MANTIĞI: EE etrafında spawn, sıkı clamp."""
        num = len(env_ids)

        # Training parametreleri - INITIAL radius kullan (curriculum yok play'de)
        pos_threshold = 0.05  # 5cm
        min_dist = pos_threshold + 0.02  # 7cm minimum
        max_dist = 0.10  # 10cm maximum (training initial_target_radius)

        # Mevcut EE pozisyonu (root'a göre relatif)
        ee_pos_world = self._compute_right_ee_pos()
        root_pos = self.robot.data.root_pos_w
        current_ee_rel = (ee_pos_world - root_pos)[env_ids]

        # Random yön
        direction = torch.randn((num, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Random mesafe (threshold'un ötesinde)
        distance = min_dist + torch.rand((num, 1), device=self.device) * (max_dist - min_dist)

        # Target = current EE + offset
        targets = current_ee_rel + direction * distance

        # SAĞ KOL İÇİN SIKI CLAMP - kolun erişebileceği alan
        # X: önde (negatif veya hafif pozitif)
        # Y: sağda (pozitif)
        # Z: göğüs-omuz arası
        targets[:, 0] = torch.clamp(targets[:, 0], -0.15, 0.30)   # Önde
        targets[:, 1] = torch.clamp(targets[:, 1], 0.00, 0.35)    # Sağda
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.45)    # Göğüs-omuz

        self.right_target_pos[env_ids] = targets

        root_pos_ids = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos_ids + targets
        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num, -1)
        pose = torch.cat([target_world, default_quat], dim=-1)
        self.right_target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

    def _sample_left_target(self, env_ids: torch.Tensor):
        """SOL KOL - TRAINING MANTIĞI: EE etrafında spawn, sıkı clamp."""
        num = len(env_ids)

        # Training parametreleri - INITIAL radius kullan (curriculum yok play'de)
        pos_threshold = 0.05  # 5cm
        min_dist = pos_threshold + 0.02  # 7cm minimum
        max_dist = 0.10  # 10cm maximum (training initial_target_radius)

        # Mevcut EE pozisyonu (root'a göre relatif)
        ee_pos_world = self._compute_left_ee_pos()
        root_pos = self.robot.data.root_pos_w
        current_ee_rel = (ee_pos_world - root_pos)[env_ids]

        # Random yön
        direction = torch.randn((num, 3), device=self.device)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Random mesafe (threshold'un ötesinde)
        distance = min_dist + torch.rand((num, 1), device=self.device) * (max_dist - min_dist)

        # Target = current EE + offset
        targets = current_ee_rel + direction * distance

        # SOL KOL İÇİN SIKI CLAMP - Y mirrored
        # X: önde (negatif veya hafif pozitif)
        # Y: solda (negatif)
        # Z: göğüs-omuz arası
        targets[:, 0] = torch.clamp(targets[:, 0], -0.15, 0.30)   # Önde
        targets[:, 1] = torch.clamp(targets[:, 1], -0.35, 0.00)   # Solda (mirrored)
        targets[:, 2] = torch.clamp(targets[:, 2], 0.05, 0.45)    # Göğüs-omuz

        self.left_target_pos[env_ids] = targets

        root_pos_ids = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos_ids + targets
        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num, -1)
        pose = torch.cat([target_world, default_quat], dim=-1)
        self.left_target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w

        r_jp = self.robot.data.joint_pos[:, self.right_arm_indices]
        r_jv = self.robot.data.joint_vel[:, self.right_arm_indices]
        r_ee = self._compute_right_ee_pos() - root_pos
        r_err = self.right_target_pos - r_ee

        l_jp = self.robot.data.joint_pos[:, self.left_arm_indices]
        l_jv = self.robot.data.joint_vel[:, self.left_arm_indices]
        l_ee = self._compute_left_ee_pos() - root_pos
        l_err = self.left_target_pos - l_ee

        # EE marker güncelle
        r_quat = self.robot.data.body_quat_w[:, self.right_palm_idx]
        self.right_ee_marker.write_root_pose_to_sim(torch.cat([self._compute_right_ee_pos(), r_quat], dim=-1))
        l_quat = self.robot.data.body_quat_w[:, self.left_palm_idx]
        self.left_ee_marker.write_root_pose_to_sim(torch.cat([self._compute_left_ee_pos(), l_quat], dim=-1))

        obs = torch.cat([
            r_jp, r_jv * 0.1, self.right_target_pos, r_ee, r_err,
            l_jp, l_jv * 0.1, self.left_target_pos, l_ee, l_err,
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w

        r_ee = self._compute_right_ee_pos() - root_pos
        r_dist = (r_ee - self.right_target_pos).norm(dim=-1)
        r_reached = r_dist < self.cfg.pos_threshold
        reached_r = torch.where(r_reached)[0]
        if len(reached_r) > 0:
            self._sample_right_target(reached_r)
            self.right_reach_count[reached_r] += 1

        l_ee = self._compute_left_ee_pos() - root_pos
        l_dist = (l_ee - self.left_target_pos).norm(dim=-1)
        l_reached = l_dist < self.cfg.pos_threshold
        reached_l = torch.where(l_reached)[0]
        if len(reached_l) > 0:
            self._sample_left_target(reached_l)
            self.left_reach_count[reached_l] += 1

        return self.cfg.reward_reaching * (r_reached.float() + l_reached.float()) - r_dist - l_dist

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        self.robot.write_root_pose_to_sim(self.fixed_root_pose[env_ids], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel[env_ids], env_ids=env_ids)

        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.robot.data.joint_vel[env_ids])

        for i, jn in enumerate(G1_RIGHT_ARM_JOINTS):
            jp[:, self.right_arm_indices[i]] = DEFAULT_ARM_POSE[jn]
        for i, jn in enumerate(G1_LEFT_ARM_JOINTS):
            jp[:, self.left_arm_indices[i]] = DEFAULT_ARM_POSE[jn]

        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        self._sample_right_target(env_ids)
        self._sample_left_target(env_ids)

        self.right_smoothed_actions[env_ids] = 0.0
        self.left_smoothed_actions[env_ids] = 0.0
        self.right_reach_count[env_ids] = 0.0
        self.left_reach_count[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions

    def _apply_action(self):
        alpha = self.cfg.action_smoothing_alpha

        r_act = self.actions[:, :5]
        l_act = self.actions[:, 5:]

        self.right_smoothed_actions = alpha * r_act + (1 - alpha) * self.right_smoothed_actions
        self.left_smoothed_actions = alpha * l_act + (1 - alpha) * self.left_smoothed_actions

        self.robot.write_root_pose_to_sim(self.fixed_root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_vel)

        r_cur = self.robot.data.joint_pos[:, self.right_arm_indices]
        r_tgt = torch.clamp(r_cur + self.right_smoothed_actions * self.cfg.action_scale,
                           self.right_joint_lower, self.right_joint_upper)

        l_cur = self.robot.data.joint_pos[:, self.left_arm_indices]
        l_tgt = torch.clamp(l_cur + self.left_smoothed_actions * self.cfg.action_scale,
                           self.left_joint_lower, self.left_joint_upper)

        jt = self.robot.data.joint_pos.clone()
        jt[:, self.right_arm_indices] = r_tgt
        jt[:, self.left_arm_indices] = l_tgt

        self.robot.set_joint_position_target(jt)