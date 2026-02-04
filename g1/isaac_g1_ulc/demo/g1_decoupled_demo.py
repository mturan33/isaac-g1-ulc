"""
G1 Decoupled Policy Demo V4 - Full State Machine
=================================================

Option B: Decoupled Policies with Coordinator

State Machine:
1. IDLE → Target pozisyon verilir
2. WALKING → Robot hedefe yürür (arm sabit, loco policy vx>0)
3. REACHING → Robot durur (loco policy vx=0), arm uzanır
4. DONE → Başarılı!

KULLANIM:
./isaaclab.bat -p g1_decoupled_demo_v4.py --num_envs 1 --target_x 2.0

Mehmet Turan YARDIMCI - VLM-RL G1 Humanoid Project
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

parser = argparse.ArgumentParser(description="G1 Decoupled Demo V4 - Full State Machine")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--loco_checkpoint", type=str,
                    default="logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt")
parser.add_argument("--arm_checkpoint", type=str,
                    default="logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt")
parser.add_argument("--target_x", type=float, default=2.0, help="Target X (world frame)")
parser.add_argument("--target_y", type=float, default=0.0, help="Target Y (world frame)")
parser.add_argument("--target_z", type=float, default=0.9, help="Target Z (world frame)")
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply_inverse


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

SHOULDER_CENTER_OFFSET = torch.tensor([0.0, -0.174, 0.259])
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
    """Coordinator thresholds"""
    walk_speed: float = 0.4           # m/s
    reach_start_distance: float = 0.6  # Start reaching when robot is this close to target XY
    reach_threshold: float = 0.08      # EE distance for success
    yaw_gain: float = 1.5              # Yaw correction gain


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
    w, xyz = q[:, 0:1], q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = torch.clamp(torch.sum(q1 * q2, dim=-1).abs(), -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]"""
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
# ENVIRONMENT
# ============================================================================

@configclass
class SceneCfg(InteractiveSceneCfg):
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

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.9)),
    )

    ee_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0), emissive_color=(0.5, 0.25, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
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
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

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
        self.shoulder_offset = SHOULDER_CENTER_OFFSET.to(self.device)

        # State
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.smoothed_arm = torch.zeros(self.num_envs, 5, device=self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT

        # Target (world frame)
        self.target_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1).clone()

        # Helper
        self.local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        print(f"\n[Env] Leg joints: {len(self.leg_idx)}, Arm joints: {len(self.arm_idx)}, Palm: {self.palm_idx}")

    def _setup_scene(self):
        self.robot = self.scene["robot"]
        self.target_obj = self.scene["target"]
        self.ee_marker = self.scene["ee_marker"]

    def set_target(self, pos: torch.Tensor):
        """Set target position (world frame)"""
        self.target_world[:] = pos
        pose = torch.cat([pos.unsqueeze(0), torch.tensor([[0,0,0,1]], device=self.device)], dim=-1)
        self.target_obj.write_root_pose_to_sim(pose)

    def get_ee_pos(self) -> torch.Tensor:
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_quat_wxyz = torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)
        forward = rotate_vector_by_quat(self.local_forward, palm_quat_wxyz)
        return palm_pos + EE_OFFSET * forward

    def get_ee_quat(self) -> torch.Tensor:
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        return torch.cat([palm_quat[:, 3:4], palm_quat[:, :3]], dim=-1)

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
        """Build Stage 3 observation with given velocity commands"""
        quat = self.robot.data.root_quat_w
        lin_vel_b = quat_apply_inverse(quat, self.robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, self.robot.data.root_ang_vel_w)

        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_grav = quat_apply_inverse(quat, gravity)

        leg_pos = self.robot.data.joint_pos[:, self.leg_idx]
        leg_vel = self.robot.data.joint_vel[:, self.leg_idx]

        gait = torch.stack([torch.sin(2*np.pi*self.gait_phase), torch.cos(2*np.pi*self.gait_phase)], dim=-1)
        torso_euler = quat_to_euler_xyz(quat)

        # Set velocity command
        self.vel_cmd[:, 0] = vx
        self.vel_cmd[:, 1] = vy
        self.vel_cmd[:, 2] = vyaw

        obs = torch.cat([
            lin_vel_b, ang_vel_b, proj_grav,     # 9
            leg_pos, leg_vel,                     # 24
            self.height_cmd.unsqueeze(-1),        # 1
            self.vel_cmd,                         # 3
            gait,                                 # 2
            self.prev_leg_actions,                # 12
            self.torso_cmd,                       # 3
            torso_euler,                          # 3
        ], dim=-1)  # 57

        return obs.clamp(-10, 10).nan_to_num()

    def build_arm_obs(self) -> torch.Tensor:
        """Build Stage 5 arm observation"""
        root_pos = self.robot.data.root_pos_w

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        arm_vel = self.robot.data.joint_vel[:, self.arm_idx]

        # Target relative to robot (body frame approximation)
        target_rel = self.target_world - root_pos

        ee_pos = self.get_ee_pos()
        ee_rel = ee_pos - root_pos
        ee_quat = self.get_ee_quat()

        pos_err = target_rel - ee_rel
        pos_dist = pos_err.norm(dim=-1, keepdim=True)
        ori_err = quat_diff_rad(ee_quat, self.target_quat).unsqueeze(-1)

        obs = torch.cat([
            arm_pos, arm_vel * 0.1,              # 10
            target_rel, self.target_quat,         # 7
            ee_rel, ee_quat,                      # 7
            pos_err, ori_err, pos_dist / 0.5,     # 5
        ], dim=-1)  # 29

        return obs

    def _get_observations(self) -> dict:
        # Update EE marker
        ee_pos = self.get_ee_pos()
        ee_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        self.ee_marker.write_root_pose_to_sim(torch.cat([ee_pos, ee_quat], dim=-1))
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
    """State machine coordinator"""

    def __init__(self, loco_policy, arm_policy, device, cfg: CoordinatorConfig = None):
        self.loco = loco_policy
        self.arm = arm_policy
        self.device = device
        self.cfg = cfg or CoordinatorConfig()

        self.state = State.IDLE
        self.target_world = None
        self.steps_in_state = 0

        self.default_arm_actions = torch.zeros(5, device=device)

    def reset(self):
        self.state = State.IDLE
        self.target_world = None
        self.steps_in_state = 0

    def set_target(self, target: torch.Tensor):
        self.target_world = target.clone()
        self.state = State.WALKING
        self.steps_in_state = 0
        print(f"\n[Coordinator] Target set: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
        print(f"[Coordinator] State: IDLE → WALKING\n")

    def step(self, env: DemoEnv) -> Tuple[torch.Tensor, dict]:
        """Run one coordinator step"""

        if self.target_world is None:
            # IDLE - return zeros
            actions = torch.zeros(env.num_envs, 17, device=self.device)
            return actions, {"state": "IDLE"}

        # Get robot state
        robot_pos = env.robot.data.root_pos_w[0]  # [3]
        robot_quat = env.robot.data.root_quat_w[0]  # [4]
        euler = quat_to_euler_xyz(robot_quat.unsqueeze(0))[0]
        robot_yaw = euler[2].item()

        ee_pos = env.get_ee_pos()[0]

        # Distances
        robot_to_target_xy = (self.target_world[:2] - robot_pos[:2]).norm().item()
        ee_to_target = (ee_pos - self.target_world).norm().item()

        # State transitions
        prev_state = self.state

        if self.state == State.WALKING:
            if robot_to_target_xy < self.cfg.reach_start_distance:
                self.state = State.REACHING
                self.steps_in_state = 0
                print(f"\n[Coordinator] Close enough! State: WALKING → REACHING")
                print(f"              Robot-Target XY: {robot_to_target_xy:.2f}m\n")

        elif self.state == State.REACHING:
            if ee_to_target < self.cfg.reach_threshold:
                self.state = State.DONE
                self.steps_in_state = 0
                print(f"\n[Coordinator] 🎯 SUCCESS! State: REACHING → DONE")
                print(f"              EE-Target: {ee_to_target:.3f}m\n")

        # Generate actions based on state
        if self.state == State.WALKING:
            # Calculate velocity command to walk toward target
            dx = self.target_world[0].item() - robot_pos[0].item()
            dy = self.target_world[1].item() - robot_pos[1].item()

            # Target yaw (world frame)
            target_yaw = math.atan2(dy, dx)
            yaw_error = normalize_angle(torch.tensor(target_yaw - robot_yaw)).item()

            # Body frame velocity
            vx = self.cfg.walk_speed
            vy = 0.0
            vyaw = np.clip(yaw_error * self.cfg.yaw_gain, -0.5, 0.5)

            # Get loco action
            loco_obs = env.build_loco_obs(vx=vx, vy=vy, vyaw=vyaw)
            with torch.no_grad():
                leg_actions = self.loco(loco_obs)

            # Arm stays at default
            arm_actions = self.default_arm_actions.unsqueeze(0).expand(env.num_envs, -1)

        elif self.state == State.REACHING:
            # Stand still (vx=0) + arm reaching
            loco_obs = env.build_loco_obs(vx=0.0, vy=0.0, vyaw=0.0)
            with torch.no_grad():
                leg_actions = self.loco(loco_obs)

            # Arm policy
            arm_obs = env.build_arm_obs()
            with torch.no_grad():
                arm_actions = self.arm(arm_obs)

        else:  # IDLE or DONE
            loco_obs = env.build_loco_obs(vx=0.0, vy=0.0, vyaw=0.0)
            with torch.no_grad():
                leg_actions = self.loco(loco_obs)
            arm_actions = self.default_arm_actions.unsqueeze(0).expand(env.num_envs, -1)

        actions = torch.cat([leg_actions, arm_actions], dim=-1)

        self.steps_in_state += 1

        info = {
            "state": self.state.name,
            "robot_to_target_xy": robot_to_target_xy,
            "ee_to_target": ee_to_target,
            "steps_in_state": self.steps_in_state,
        }

        return actions, info


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    G1 DECOUPLED DEMO V4 - Full State Machine")
    print("    WALKING → REACHING Pipeline")
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
    coord_cfg = CoordinatorConfig(
        walk_speed=0.4,
        reach_start_distance=0.6,
        reach_threshold=0.08,
    )
    coordinator = Coordinator(loco_policy, arm_policy, device, coord_cfg)

    # Set target
    print("\n[3/3] Setting target...")
    target = torch.tensor([args.target_x, args.target_y, args.target_z], device=device)

    # Reset
    obs, _ = env.reset()
    env.set_target(target)
    coordinator.set_target(target)

    print("=" * 70)
    print(f"    Target: ({args.target_x}, {args.target_y}, {args.target_z})")
    print(f"    Walk speed: {coord_cfg.walk_speed} m/s")
    print(f"    Reach start: {coord_cfg.reach_start_distance}m")
    print(f"    Reach threshold: {coord_cfg.reach_threshold}m")
    print("=" * 70 + "\n")

    # Main loop
    for step in range(args.steps):
        # Coordinator step
        actions, info = coordinator.step(env)

        # Environment step
        obs, reward, terminated, truncated, _ = env.step(actions)

        # Check fallen
        if terminated.any():
            print(f"[Step {step}] ⚠️ Robot fell! Resetting...")
            env.reset()
            env.set_target(target)
            coordinator.reset()
            coordinator.set_target(target)
            continue

        # Log
        if step % 50 == 0:
            h = env.robot.data.root_pos_w[0, 2].item()
            vx = env.robot.data.root_lin_vel_w[0, 0].item()

            print(f"[Step {step:4d}] State={info['state']:10s} | "
                  f"H={h:.2f}m | Vx={vx:+.2f}m/s | "
                  f"Robot→Target: {info['robot_to_target_xy']:.2f}m | "
                  f"EE→Target: {info['ee_to_target']:.3f}m")

        # Done?
        if coordinator.state == State.DONE:
            print(f"\n{'='*70}")
            print(f"    ✅ SUCCESS! Target reached in {step} steps!")
            print(f"{'='*70}\n")

            # Wait a bit then exit
            for _ in range(100):
                actions, _ = coordinator.step(env)
                env.step(actions)
            break

    if coordinator.state != State.DONE:
        print(f"\n[Timeout] Demo ended in state: {coordinator.state.name}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()