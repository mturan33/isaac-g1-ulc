"""
G1 Dual Arm Environment (Play Only)
====================================

Stage 4 Dual Arm: Her iki kol da aktif, 4 visual marker.
Workspace: Omuz merkezli yarım küre (10-30cm yarıçap)

VISUAL MARKERS:
- Yeşil küre  = Sağ kol target
- Mavi küre   = Sol kol target
- Turuncu    = Sağ el (EE)
- Mor        = Sol el (EE)
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

    # Yarım küre workspace
    workspace_radius_min = 0.10
    workspace_radius_max = 0.30


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

        # Omuz indeksleri
        self.right_shoulder_idx = None
        self.left_shoulder_idx = None
        for i, name in enumerate(body_names):
            if "right_shoulder_pitch_link" in name:
                self.right_shoulder_idx = i
            if "left_shoulder_pitch_link" in name:
                self.left_shoulder_idx = i

        self.workspace_radius_min = self.cfg.workspace_radius_min
        self.workspace_radius_max = self.cfg.workspace_radius_max

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
        print("G1 DUAL ARM ENVIRONMENT - YARIM KÜRE WORKSPACE")
        print("=" * 70)
        print(f"  Right arm: {self.right_arm_indices.tolist()}")
        print(f"  Left arm:  {self.left_arm_indices.tolist()}")
        print(f"  Workspace: {self.workspace_radius_min}m - {self.workspace_radius_max}m (yarım küre)")
        print("=" * 70 + "\n")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.right_target_obj = self.scene["right_target"]
        self.left_target_obj = self.scene["left_target"]
        self.right_ee_marker = self.scene["right_ee_marker"]
        self.left_ee_marker = self.scene["left_ee_marker"]

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

    def _get_shoulder_pos(self, env_ids: torch.Tensor, is_right: bool) -> torch.Tensor:
        """Omuz pozisyonunu al (root-relative)."""
        root_pos = self.robot.data.root_pos_w[env_ids]

        if is_right and self.right_shoulder_idx is not None:
            shoulder_world = self.robot.data.body_pos_w[env_ids, self.right_shoulder_idx]
            return shoulder_world - root_pos
        elif not is_right and self.left_shoulder_idx is not None:
            shoulder_world = self.robot.data.body_pos_w[env_ids, self.left_shoulder_idx]
            return shoulder_world - root_pos
        else:
            # Fallback: sabit offset
            if is_right:
                return torch.tensor([[0.0, -0.17, 0.35]], device=self.device).expand(len(env_ids), -1)
            else:
                return torch.tensor([[0.0, 0.17, 0.35]], device=self.device).expand(len(env_ids), -1)

    def _sample_right_target(self, env_ids: torch.Tensor):
        """SAĞ KOL - ÇALIŞAN KOD!"""
        num = len(env_ids)

        targets = torch.zeros((num, 3), device=self.device)
        targets[:, 0] = torch.empty(num, device=self.device).uniform_(-0.45, -0.30)  # X: önde
        targets[:, 1] = torch.empty(num, device=self.device).uniform_(0.00, 0.20)  # Y: sağ taraf (POZİTİF DEĞİL!)
        targets[:, 2] = torch.empty(num, device=self.device).uniform_(-0.05, 0.15)  # Z: göğüs hizası

        self.right_target_pos[env_ids] = targets

        root_pos = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos + targets
        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(num, -1)
        pose = torch.cat([target_world, default_quat], dim=-1)
        self.right_target_obj.write_root_pose_to_sim(pose, env_ids=env_ids)

    def _sample_left_target(self, env_ids: torch.Tensor):
        """SOL KOL - ÇALIŞAN KOD!"""
        num = len(env_ids)

        targets = torch.zeros((num, 3), device=self.device)
        targets[:, 0] = torch.empty(num, device=self.device).uniform_(-0.45, -0.30)  # X: önde
        targets[:, 1] = torch.empty(num, device=self.device).uniform_(-0.20, 0.00)  # Y: sol taraf (NEGATİF!)
        targets[:, 2] = torch.empty(num, device=self.device).uniform_(-0.05, 0.15)  # Z: göğüs hizası

        self.left_target_pos[env_ids] = targets

        root_pos = self.robot.data.root_pos_w[env_ids]
        target_world = root_pos + targets
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