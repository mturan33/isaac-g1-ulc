"""
G1 Decoupled Policy Demo V5 - Full State Machine (FIXED)
=========================================================

Düzeltmeler:
1. VisualizationMarkers kullanıyor (fizik bozulmuyor)
2. Target BODY FRAME'de (omuzun önünde spawn)
3. Walking → Standing → Reaching pipeline

KULLANIM:
./isaaclab.bat -p g1_decoupled_demo_v5.py --num_envs 1

Turan Özhan - VLM-RL G1 Humanoid Project
4 Şubat 2026
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

parser = argparse.ArgumentParser(description="G1 Decoupled Demo V5")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--loco_checkpoint", type=str,
                    default="logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt")
parser.add_argument("--arm_checkpoint", type=str,
                    default="logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt")
parser.add_argument("--walk_distance", type=float, default=1.5, help="How far to walk before reaching (m)")
parser.add_argument("--spawn_radius", type=float, default=0.25, help="Arm target spawn radius (m)")
parser.add_argument("--steps", type=int, default=5000)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# ============================================================================
# CONSTANTS
# ============================================================================

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_elbow_roll_joint",
]

DEFAULT_LEG_POS = [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0]
DEFAULT_ARM_POS = [-0.3, 0.0, 0.0, 0.5, 0.0]

ARM_JOINT_LIMITS = {
    "right_shoulder_pitch_joint": (-2.97, 2.79),
    "right_shoulder_roll_joint": (-2.25, 1.59),
    "right_shoulder_yaw_joint": (-2.62, 2.62),
    "right_elbow_pitch_joint": (-0.23, 3.42),
    "right_elbow_roll_joint": (-2.09, 2.09),
}

# Shoulder offset from robot root (body frame) - G1 specific
SHOULDER_CENTER_OFFSET = [0.0, -0.174, 0.259]
WORKSPACE_INNER_RADIUS = 0.18
WORKSPACE_OUTER_RADIUS = 0.45

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
EE_OFFSET = 0.02


# ============================================================================
# STATE MACHINE
# ============================================================================

class State(Enum):
    IDLE = 0
    WALKING = 1
    REACHING = 2
    DONE = 3


@dataclass
class CoordinatorConfig:
    walk_speed: float = 0.4
    reach_threshold: float = 0.10
    yaw_gain: float = 1.5


# ============================================================================
# HELPERS
# ============================================================================

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def rotate_vector_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion (wxyz format)"""
    w, xyz = q[:, 0:1], q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = torch.clamp(torch.sum(q1 * q2, dim=-1).abs(), -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# ============================================================================
# NETWORKS
# ============================================================================

class LocoActor(nn.Module):
    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ArmActor(nn.Module):
    def __init__(self, num_obs=29, num_act=5, hidden=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_loco_policy(path: str, device: str) -> LocoActor:
    actor = LocoActor().to(device)
    if not os.path.exists(path):
        print(f"[WARN] Loco checkpoint not found: {path}")
        return actor

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "actor_critic" in ckpt:
        state_dict = {f"net.{k[6:]}": v for k, v in ckpt["actor_critic"].items() if k.startswith("actor.")}
        actor.load_state_dict(state_dict)
        print(f"[OK] Loaded loco policy")
    actor.eval()
    return actor


def load_arm_policy(path: str, device: str) -> ArmActor:
    actor = ArmActor().to(device)
    if not os.path.exists(path):
        print(f"[WARN] Arm checkpoint not found: {path}")
        return actor

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = {k.replace("actor.", "net."): v for k, v in ckpt["model_state_dict"].items()
                      if "actor" in k and "critic" not in k}
        try:
            actor.load_state_dict(state_dict)
            print(f"[OK] Loaded arm policy")
        except:
            actor.load_state_dict(state_dict, strict=False)
            print(f"[OK] Loaded arm policy (partial)")
    actor.eval()
    return actor


# ============================================================================
# ENVIRONMENT (No RigidObjects for markers!)
# ============================================================================

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene WITHOUT RigidObject markers - using VisualizationMarkers instead"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                "right_shoulder_pitch_joint": -0.3, "right_elbow_pitch_joint": 0.5,
                "left_shoulder_pitch_joint": -0.3, "left_elbow_pitch_joint": 0.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"], stiffness=150.0, damping=15.0),
            "arms": ImplicitActuatorCfg(joint_names_expr=[".*shoulder.*", ".*elbow.*"], stiffness=100.0, damping=10.0),
            "torso": ImplicitActuatorCfg(joint_names_expr=["torso_joint"], stiffness=100.0, damping=10.0),
        },
    )


@configclass
class EnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 60.0
    num_actions = 17
    num_observations = 50
    num_states = 0
    action_space = 17
    observation_space = 50
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1/200, render_interval=4, device="cuda:0")
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=4.0)


class DemoEnv(DirectRLEnv):
    cfg: EnvCfg

    def __init__(self, cfg: EnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]

        joint_names = self.robot.data.joint_names
        body_names = self.robot.data.body_names

        # Joint indices
        self.leg_idx = torch.tensor([joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names], device=self.device)
        self.arm_idx = torch.tensor([joint_names.index(n) for n in ARM_JOINT_NAMES if n in joint_names], device=self.device)

        # Palm index
        self.palm_idx = next((i for i, n in enumerate(body_names) if "right" in n.lower() and "palm" in n.lower()), None)

        # Defaults
        self.default_leg = torch.tensor(DEFAULT_LEG_POS, device=self.device)
        self.default_arm = torch.tensor(DEFAULT_ARM_POS, device=self.device)
        self.arm_lower = torch.tensor([ARM_JOINT_LIMITS[n][0] for n in ARM_JOINT_NAMES], device=self.device)
        self.arm_upper = torch.tensor([ARM_JOINT_LIMITS[n][1] for n in ARM_JOINT_NAMES], device=self.device)
        self.shoulder_offset = torch.tensor(SHOULDER_CENTER_OFFSET, device=self.device)

        # State
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.smoothed_arm = torch.zeros(self.num_envs, 5, device=self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT

        # Target (BODY FRAME - relative to shoulder)
        self.target_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1).clone()

        # Walk target (world frame - where to walk to)
        self.walk_target_world = torch.zeros(self.num_envs, 3, device=self.device)

        # Helper
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # ============ VISUALIZATION MARKERS (no physics!) ============
        self.target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),
                            emissive_color=(0.0, 0.5, 0.0),
                        ),
                    ),
                },
            )
        )

        self.ee_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/EEMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.5, 0.0),
                            emissive_color=(0.5, 0.25, 0.0),
                        ),
                    ),
                },
            )
        )

        self.walk_target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/WalkTargetMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.08,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),
                            emissive_color=(0.5, 0.0, 0.0),
                        ),
                    ),
                },
            )
        )

        self.shoulder_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/ShoulderMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 1.0, 1.0),
                            emissive_color=(0.5, 0.5, 0.5),
                        ),
                    ),
                },
            )
        )

        print(f"\n[Env] Initialized with VisualizationMarkers (no physics interference)")
        print(f"  Leg joints: {len(self.leg_idx)}, Arm joints: {len(self.arm_idx)}, Palm: {self.palm_idx}")

    def _setup_scene(self):
        self.robot = self.scene["robot"]

    def set_walk_target(self, distance: float):
        """Set walk target distance ahead (world frame)"""
        root_pos = self.robot.data.root_pos_w[0]
        self.walk_target_world[0, 0] = root_pos[0] + distance
        self.walk_target_world[0, 1] = root_pos[1]
        self.walk_target_world[0, 2] = 0.0  # Ground level marker

        # Update marker
        self._update_walk_marker()
        print(f"[Env] Walk target set: {distance}m ahead")

    def sample_arm_target(self):
        """Sample arm target in BODY FRAME (relative to shoulder)"""
        # Random direction (front-biased)
        direction = torch.randn(3, device=self.device)
        direction[0] = -torch.abs(direction[0])  # -X = robot's front
        direction = direction / (direction.norm() + 1e-8)

        # Random distance in workspace
        inner = WORKSPACE_INNER_RADIUS
        outer = min(args.spawn_radius, WORKSPACE_OUTER_RADIUS)
        dist = inner + torch.rand(1, device=self.device).item() * (outer - inner)

        # Target relative to shoulder (body frame)
        target_rel = self.shoulder_offset + direction * dist
        target_rel[2] = torch.clamp(target_rel[2], 0.05, 0.55)

        self.target_body[0] = target_rel
        print(f"[Env] Arm target (body frame): [{target_rel[0]:.2f}, {target_rel[1]:.2f}, {target_rel[2]:.2f}]")

    def get_ee_pos(self) -> torch.Tensor:
        """Get EE position in world frame"""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_quat_wxyz = torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)
        forward = rotate_vector_by_quat(self.local_forward, palm_quat_wxyz)
        return palm_pos + EE_OFFSET * forward

    def get_ee_quat(self) -> torch.Tensor:
        """Get EE orientation (wxyz)"""
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        return torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)

    def get_target_world(self) -> torch.Tensor:
        """Convert body frame target to world frame"""
        root_pos = self.robot.data.root_pos_w
        return root_pos + self.target_body

    def _update_markers(self):
        """Update all visualization markers"""
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)

        # Target marker (world position)
        target_world = self.get_target_world()
        self.target_marker.visualize(translations=target_world, orientations=identity_quat.expand(self.num_envs, -1))

        # EE marker
        ee_pos = self.get_ee_pos()
        self.ee_marker.visualize(translations=ee_pos, orientations=identity_quat.expand(self.num_envs, -1))

        # Shoulder marker
        root_pos = self.robot.data.root_pos_w
        shoulder_world = root_pos + self.shoulder_offset.unsqueeze(0)
        self.shoulder_marker.visualize(translations=shoulder_world, orientations=identity_quat.expand(self.num_envs, -1))

    def _update_walk_marker(self):
        """Update walk target marker"""
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        self.walk_target_marker.visualize(
            translations=self.walk_target_world,
            orientations=identity_quat.expand(self.num_envs, -1)
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        leg_act = self.actions[:, :12]
        arm_act = self.actions[:, 12:]

        # Smooth arm
        self.smoothed_arm = 0.15 * arm_act + 0.85 * self.smoothed_arm

        # Targets
        tgt = self.robot.data.default_joint_pos.clone()
        tgt[:, self.leg_idx] = self.default_leg + leg_act * 0.4

        cur_arm = self.robot.data.joint_pos[:, self.arm_idx]
        arm_tgt = torch.clamp(cur_arm + self.smoothed_arm * 0.12, self.arm_lower, self.arm_upper)
        tgt[:, self.arm_idx] = arm_tgt

        self.robot.set_joint_position_target(tgt)
        self.gait_phase = (self.gait_phase + GAIT_FREQUENCY * 0.02) % 1.0
        self.prev_leg_actions = leg_act.clone()

    def build_loco_obs(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> torch.Tensor:
        """Build Stage 3 observation (57 dims)"""
        quat = self.robot.data.root_quat_w
        lin_vel_b = quat_apply_inverse(quat, self.robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, self.robot.data.root_ang_vel_w)

        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_grav = quat_apply_inverse(quat, gravity)

        leg_pos = self.robot.data.joint_pos[:, self.leg_idx]
        leg_vel = self.robot.data.joint_vel[:, self.leg_idx]

        gait = torch.stack([torch.sin(2*np.pi*self.gait_phase), torch.cos(2*np.pi*self.gait_phase)], dim=-1)
        torso_euler = quat_to_euler_xyz(quat)

        self.vel_cmd[:, 0] = vx
        self.vel_cmd[:, 1] = vy
        self.vel_cmd[:, 2] = vyaw

        obs = torch.cat([
            lin_vel_b, ang_vel_b, proj_grav,
            leg_pos, leg_vel,
            self.height_cmd.unsqueeze(-1),
            self.vel_cmd,
            gait,
            self.prev_leg_actions,
            self.torso_cmd,
            torso_euler,
        ], dim=-1)

        return obs.clamp(-10, 10).nan_to_num()

    def build_arm_obs(self) -> torch.Tensor:
        """Build Stage 5 arm observation (29 dims) - target in BODY FRAME"""
        root_pos = self.robot.data.root_pos_w

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        arm_vel = self.robot.data.joint_vel[:, self.arm_idx]

        # Target in body frame (already stored this way!)
        target_rel = self.target_body

        # EE relative to root (body frame)
        ee_pos_world = self.get_ee_pos()
        ee_rel = ee_pos_world - root_pos
        ee_quat = self.get_ee_quat()

        pos_err = target_rel - ee_rel
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        ori_err = quat_diff_rad(ee_quat, self.target_quat).unsqueeze(-1)

        obs = torch.cat([
            arm_pos, arm_vel * 0.1,
            target_rel, self.target_quat,
            ee_rel, ee_quat,
            pos_err, ori_err, pos_dist / 0.5,
        ], dim=-1)

        return obs

    def _get_observations(self) -> dict:
        self._update_markers()
        return {"policy": torch.zeros(self.num_envs, 50, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.robot.data.root_pos_w[:, 2]
        grav = torch.tensor([0,0,-1.0], device=self.device).expand(self.num_envs, -1)
        proj = quat_apply_inverse(self.robot.data.root_quat_w, grav)

        fallen = (h < 0.3) | (h > 1.2) | (proj[:, :2].abs().max(-1)[0] > 0.7)
        truncated = self.episode_length_buf >= self.max_episode_length
        return fallen, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        N = len(env_ids)
        pos = torch.tensor([[0,0,0.8]], device=self.device).expand(N,-1).clone()
        quat = torch.tensor([[0,0,0,1.0]], device=self.device).expand(N,-1)

        self.robot.write_root_pose_to_sim(torch.cat([pos, quat], -1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=self.device), env_ids)

        jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, env_ids)

        self.gait_phase[env_ids] = torch.rand(N, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.smoothed_arm[env_ids] = 0


# ============================================================================
# COORDINATOR
# ============================================================================

class Coordinator:
    def __init__(self, loco_policy, arm_policy, device, cfg: CoordinatorConfig = None):
        self.loco = loco_policy
        self.arm = arm_policy
        self.device = device
        self.cfg = cfg or CoordinatorConfig()

        self.state = State.IDLE
        self.steps_in_state = 0
        self.default_arm_actions = torch.zeros(5, device=device)

    def reset(self):
        self.state = State.IDLE
        self.steps_in_state = 0

    def start_walking(self):
        self.state = State.WALKING
        self.steps_in_state = 0
        print(f"\n[Coordinator] State: IDLE → WALKING\n")

    def step(self, env: DemoEnv) -> Tuple[torch.Tensor, dict]:
        robot_pos = env.robot.data.root_pos_w[0]
        robot_quat = env.robot.data.root_quat_w[0]
        euler = quat_to_euler_xyz(robot_quat.unsqueeze(0))[0]
        robot_yaw = euler[2].item()

        # Distances
        walk_target = env.walk_target_world[0]
        robot_to_walk_target = (walk_target[:2] - robot_pos[:2]).norm().item()

        ee_pos = env.get_ee_pos()[0]
        target_world = env.get_target_world()[0]
        ee_to_target = (ee_pos - target_world).norm().item()

        # State transitions
        if self.state == State.WALKING:
            if robot_to_walk_target < 0.3:  # Arrived at walk position
                self.state = State.REACHING
                self.steps_in_state = 0
                print(f"\n[Coordinator] Arrived! State: WALKING → REACHING")
                print(f"              Distance to walk target: {robot_to_walk_target:.2f}m\n")

        elif self.state == State.REACHING:
            if ee_to_target < self.cfg.reach_threshold:
                self.state = State.DONE
                self.steps_in_state = 0
                print(f"\n[Coordinator] 🎯 SUCCESS! State: REACHING → DONE")
                print(f"              EE-Target: {ee_to_target:.3f}m\n")

        # Generate actions
        if self.state == State.WALKING:
            dx = walk_target[0].item() - robot_pos[0].item()
            dy = walk_target[1].item() - robot_pos[1].item()
            target_yaw = math.atan2(dy, dx)
            yaw_error = normalize_angle(torch.tensor(target_yaw - robot_yaw)).item()

            vx = self.cfg.walk_speed
            vyaw = np.clip(yaw_error * self.cfg.yaw_gain, -0.5, 0.5)

            loco_obs = env.build_loco_obs(vx=vx, vy=0.0, vyaw=vyaw)
            with torch.no_grad():
                leg_actions = self.loco(loco_obs)
            arm_actions = self.default_arm_actions.unsqueeze(0)

        elif self.state == State.REACHING:
            loco_obs = env.build_loco_obs(vx=0.0, vy=0.0, vyaw=0.0)
            with torch.no_grad():
                leg_actions = self.loco(loco_obs)

            arm_obs = env.build_arm_obs()
            with torch.no_grad():
                arm_actions = self.arm(arm_obs)

        else:  # IDLE or DONE
            loco_obs = env.build_loco_obs(vx=0.0, vy=0.0, vyaw=0.0)
            with torch.no_grad():
                leg_actions = self.loco(loco_obs)
            arm_actions = self.default_arm_actions.unsqueeze(0)

        actions = torch.cat([leg_actions, arm_actions], dim=-1)
        self.steps_in_state += 1

        info = {
            "state": self.state.name,
            "robot_to_walk_target": robot_to_walk_target,
            "ee_to_target": ee_to_target,
        }

        return actions, info


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    G1 DECOUPLED DEMO V5 - FIXED")
    print("    VisualizationMarkers + Body Frame Target")
    print("=" * 70)

    device = "cuda:0"

    # Load policies
    print("\n[1/3] Loading policies...")
    loco_policy = load_loco_policy(args.loco_checkpoint, device)
    arm_policy = load_arm_policy(args.arm_checkpoint, device)

    # Create environment
    print("\n[2/3] Creating environment...")
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = DemoEnv(cfg=env_cfg)

    # Create coordinator
    coord_cfg = CoordinatorConfig(walk_speed=0.4, reach_threshold=0.10)
    coordinator = Coordinator(loco_policy, arm_policy, device, coord_cfg)

    # Reset
    print("\n[3/3] Setting up demo...")
    obs, _ = env.reset()

    # Set walk target (where robot should walk to)
    env.set_walk_target(args.walk_distance)

    # Sample arm target (body frame - relative to shoulder)
    env.sample_arm_target()

    # Start
    coordinator.start_walking()

    print("=" * 70)
    print(f"    Walk distance: {args.walk_distance}m")
    print(f"    Arm target radius: {args.spawn_radius}m")
    print(f"    Pipeline: WALKING → REACHING → DONE")
    print("=" * 70 + "\n")

    # Main loop
    for step in range(args.steps):
        actions, info = coordinator.step(env)
        obs, reward, terminated, truncated, _ = env.step(actions)

        if terminated.any():
            print(f"[Step {step}] ⚠️ Robot fell! Resetting...")
            env.reset()
            env.set_walk_target(args.walk_distance)
            env.sample_arm_target()
            coordinator.reset()
            coordinator.start_walking()
            continue

        if step % 50 == 0:
            h = env.robot.data.root_pos_w[0, 2].item()
            vx = env.robot.data.root_lin_vel_w[0, 0].item()

            print(f"[Step {step:4d}] {info['state']:10s} | H={h:.2f}m | Vx={vx:+.2f}m/s | "
                  f"Walk: {info['robot_to_walk_target']:.2f}m | EE: {info['ee_to_target']:.3f}m")

        if coordinator.state == State.DONE:
            print(f"\n{'='*70}")
            print(f"    ✅ SUCCESS! Completed in {step} steps!")
            print(f"{'='*70}\n")

            for _ in range(100):
                actions, _ = coordinator.step(env)
                env.step(actions)
            break

    if coordinator.state != State.DONE:
        print(f"\n[Timeout] Ended in state: {coordinator.state.name}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()