#!/usr/bin/env python3
"""
Stage 5.5 Dual Policy Play Script - V3
======================================
Stage 3 play script'i base alınarak oluşturuldu.
Stage 3 collision/terrain aynen korunuyor, sadece arm policy eklendi.

KULLANIM:
./isaaclab.bat -p .../play_ulc_stage_5_5_both_v3.py \
    --loco_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt \
    --arm_checkpoint logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt \
    --num_envs 1 --vx 0.0
"""

import torch
import torch.nn as nn
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 5.5 Dual Policy Play V3")
    parser.add_argument("--loco_checkpoint", type=str, required=True)
    parser.add_argument("--arm_checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command")
    parser.add_argument("--pitch", type=float, default=0.0, help="Torso pitch command (rad)")
    parser.add_argument("--roll", type=float, default=0.0, help="Torso roll command (rad)")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


args_cli = parse_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


print("=" * 70)
print("STAGE 5.5 DUAL POLICY PLAY - V3 (Stage 3 Based)")
print("=" * 70)
print(f"Loco checkpoint: {args_cli.loco_checkpoint}")
print(f"Arm checkpoint: {args_cli.arm_checkpoint}")
print(f"Commands: vx={args_cli.vx}, pitch={args_cli.pitch}°")
print("=" * 70)

# ==============================================================================
# CONSTANTS - Stage 3 ile aynı
# ==============================================================================

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# Shoulder offset for workspace (right shoulder in local frame)
# Stage 5 config: shoulder_center_offset = [0.0, 0.174, 0.259]
# Note: Stage 5 env uses this directly for target sampling
SHOULDER_OFFSET = torch.tensor([0.0, 0.174, 0.259])


# ==============================================================================
# NEURAL NETWORK DEFINITIONS
# ==============================================================================

class LocoActorCritic(nn.Module):
    """Stage 3 Locomotion Policy - 57 obs -> 12 actions"""

    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def act(self, x, deterministic=True):
        mean = self.actor(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


class ArmActorCritic(nn.Module):
    """Stage 5 Arm Policy - 29 obs -> 5 actions
    Checkpoint yapısına uygun: Linear + ELU (LayerNorm yok)
    """

    def __init__(self, num_obs=29, num_act=5, hidden=[256, 128, 64]):
        super().__init__()

        # Actor: Linear -> ELU -> Linear -> ELU -> Linear -> ELU -> Linear
        self.actor = nn.Sequential(
            nn.Linear(num_obs, hidden[0]),  # 0
            nn.ELU(),  # 1
            nn.Linear(hidden[0], hidden[1]),  # 2
            nn.ELU(),  # 3
            nn.Linear(hidden[1], hidden[2]),  # 4
            nn.ELU(),  # 5
            nn.Linear(hidden[2], num_act),  # 6
        )

        # Critic: same structure -> 1 output
        self.critic = nn.Sequential(
            nn.Linear(num_obs, hidden[0]),
            nn.ELU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.Linear(hidden[2], 1),
        )

        # std instead of log_std
        self.std = nn.Parameter(torch.zeros(num_act))

    def act(self, x, deterministic=True):
        mean = self.actor(x)
        if deterministic:
            return mean
        std = self.std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


# ==============================================================================
# SCENE CONFIG - Stage 3 ile BİREBİR AYNI
# ==============================================================================

@configclass
class PlaySceneCfg(InteractiveSceneCfg):
    """Stage 3 ile AYNI scene config - marker yok"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0, "right_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.0, "right_elbow_pitch_joint": 0.0,
                "left_elbow_roll_joint": 0.0, "right_elbow_roll_joint": 0.0,
                "torso_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=150.0,
                damping=15.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=50.0,
                damping=5.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )
    # NO target or ee_marker - they break physics!


@configclass
class PlayEnvCfg(DirectRLEnvCfg):
    """Stage 3 ile aynı env config"""
    decimation = 4
    episode_length_s = 30.0
    action_space = 17  # 12 legs + 5 arm
    observation_space = 57  # Stage 3 loco obs
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=1, env_spacing=2.5)


# ==============================================================================
# ENVIRONMENT - Stage 3 tabanlı, arm eklendi
# ==============================================================================

class DualPlayEnv(DirectRLEnv):
    """Stage 3 environment + arm control"""

    cfg: PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices - Stage 3 ile aynı
        joint_names = self.robot.joint_names

        self.leg_idx = torch.tensor(
            [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
            device=self.device
        )

        self.arm_idx = torch.tensor(
            [joint_names.index(n) for n in ARM_JOINT_NAMES if n in joint_names],
            device=self.device
        )

        # Find palm body index for EE tracking
        body_names = self.robot.body_names
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right" in name.lower() and ("palm" in name.lower() or "hand" in name.lower()):
                self.palm_idx = i
                break
        if self.palm_idx is None:
            # Fallback to right elbow roll link
            for i, name in enumerate(body_names):
                if "right_elbow_roll" in name.lower():
                    self.palm_idx = i
                    break

        # Default positions
        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )
        self.default_arm = torch.tensor(
            [-0.3, 0.2, 0.0, 0.8, 0.0],  # Right arm neutral
            device=self.device
        )

        # Commands from CLI - Stage 3 format
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_cmd[:, 0] = args_cli.vx
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd[:, 0] = args_cli.roll
        self.torso_cmd[:, 1] = args_cli.pitch

        # Gait phase
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

        # Arm target (body frame) - will be initialized on first reset
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        # Default target position (in front of robot)
        self.target_pos_body[:, 0] = 0.3  # forward
        self.target_pos_body[:, 1] = -0.2  # right
        self.target_pos_body[:, 2] = 0.2  # up

        # Reach tracking
        self.reach_count = 0
        self.reach_threshold = 0.08

        print(f"[DualPlayEnv] Leg joints: {len(self.leg_idx)}")
        print(f"[DualPlayEnv] Arm joints: {len(self.arm_idx)}")
        print(f"[DualPlayEnv] Palm idx: {self.palm_idx}")

        # Markers will be created lazily after simulation starts
        self._markers_initialized = False
        self.target_markers = None
        self.ee_markers = None

    @property
    def robot(self):
        return self.scene["robot"]

    def get_torso_euler(self):
        quat = self.robot.data.root_quat_w
        return quat_to_euler_xyz(quat)

    def _sample_targets(self, env_ids):
        """Sample targets and update markers (called during simulation)"""
        self._sample_target_positions(env_ids)

    def _compute_ee_pos(self):
        """Get end-effector world position"""
        return self.robot.data.body_pos_w[:, self.palm_idx, :]

    def _get_loco_obs(self) -> torch.Tensor:
        """Stage 3 locomotion observations (57 dim)"""
        robot = self.robot
        quat = robot.data.root_quat_w

        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        joint_pos = robot.data.joint_pos[:, self.leg_idx]
        joint_vel = robot.data.joint_vel[:, self.leg_idx]

        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.phase),
            torch.cos(2 * np.pi * self.phase)
        ], dim=-1)

        torso_euler = self.get_torso_euler()

        obs = torch.cat([
            lin_vel_b,  # 3
            ang_vel_b,  # 3
            proj_gravity,  # 3
            joint_pos,  # 12
            joint_vel,  # 12
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,  # 3
            gait_phase,  # 2
            self.prev_leg_actions,  # 12
            self.torso_cmd,  # 3
            torso_euler,  # 3
        ], dim=-1)  # Total: 57

        return obs.clamp(-10, 10).nan_to_num()

    def _get_arm_obs(self) -> torch.Tensor:
        """Stage 5 arm observations (29 dim)

        Stage 5 robot havada asılı, -X forward.
        Locomotion robot yerde, +X forward.

        Observation'ı Stage 5 formatında vermemiz lazım ki trained policy çalışsın.
        Bu yüzden koordinatları Stage 5 frame'ine dönüştürüyoruz.
        """
        robot = self.robot
        root_pos = robot.data.root_pos_w

        # Arm joint states (same in both frames)
        joint_pos = robot.data.joint_pos[:, self.arm_idx]
        joint_vel = robot.data.joint_vel[:, self.arm_idx]

        # EE position relative to root
        ee_pos_world = self._compute_ee_pos()
        ee_pos_local = ee_pos_world - root_pos

        # Convert to Stage 5 frame: flip X axis (-X is forward in Stage 5)
        ee_pos_stage5 = ee_pos_local.clone()
        ee_pos_stage5[:, 0] = -ee_pos_local[:, 0]  # Flip X
        ee_pos_stage5[:, 1] = -ee_pos_local[:, 1]  # Flip Y (right side is +Y in Stage 5)

        # EE quaternion
        ee_quat = robot.data.body_quat_w[:, self.palm_idx]

        # Target in Stage 5 frame
        target_local = self.target_pos_body.clone()
        target_stage5 = target_local.clone()
        target_stage5[:, 0] = -target_local[:, 0]  # Flip X
        target_stage5[:, 1] = -target_local[:, 1]  # Flip Y

        # Target orientation (fixed for now)
        target_quat = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

        # Position error in Stage 5 frame
        pos_err = target_stage5 - ee_pos_stage5
        pos_dist = pos_err.norm(dim=-1, keepdim=True)

        # Orientation error
        dot = torch.sum(ee_quat * target_quat, dim=-1).abs().clamp(-1, 1)
        ori_err = (2.0 * torch.acos(dot)).unsqueeze(-1)

        # BUILD OBS - EXACT Stage 5 ORDER
        obs = torch.cat([
            joint_pos,  # 5 - arm joint positions
            joint_vel * 0.1,  # 5 - arm joint velocities (scaled!)
            target_stage5,  # 3 - target position (Stage 5 frame)
            target_quat,  # 4 - target orientation
            ee_pos_stage5,  # 3 - EE position (Stage 5 frame)
            ee_quat,  # 4 - EE orientation
            pos_err,  # 3 - position error (Stage 5 frame)
            ori_err,  # 1 - orientation error
            pos_dist / 0.5,  # 1 - normalized distance
        ], dim=-1)  # Total: 5+5+3+4+3+4+3+1+1 = 29

        return obs.clamp(-10, 10).nan_to_num()

    def _pre_physics_step(self, actions):
        """Process combined actions"""
        self.actions = actions.clone()

        # Split: legs (0:12), arm (12:17)
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]

        # Compute joint targets
        target_pos = self.robot.data.default_joint_pos.clone()

        # Legs: Stage 3 formula
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4

        # Arm: default + scaled action
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.3

        self.robot.set_joint_position_target(target_pos)

        # Update gait phase
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

        # Store previous actions
        self.prev_leg_actions = leg_actions.clone()
        self.prev_arm_actions = arm_actions.clone()

        # Check reach and resample
        ee_pos = self._compute_ee_pos()
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Transform target to world for distance check
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist = torch.norm(ee_pos - target_world, dim=-1)

        # Update visual markers
        self._update_markers(ee_pos, target_world)

        reached = dist < self.reach_threshold
        if reached.any():
            reached_ids = torch.where(reached)[0]
            self.reach_count += len(reached_ids)
            self._sample_targets(reached_ids)
            print(f"[Step {self.episode_length_buf[0].item():5d}] 🎯 REACH #{self.reach_count}!")

    def _init_markers(self):
        """Create markers AFTER simulation has started (lazy initialization)"""
        if self._markers_initialized:
            return

        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 1.0, 0.0),  # Yellow
                        ),
                    ),
                },
            )
        )

        self.ee_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/EEMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),  # Green
                        ),
                    ),
                },
            )
        )

        self._markers_initialized = True
        print("[DualPlayEnv] ✓ Markers initialized")

    def _update_markers(self, ee_pos, target_world):
        """Update visual marker positions - creates markers on first call"""
        # Lazy init - create markers after simulation is stable
        self._init_markers()

        # Update positions
        default_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(self.num_envs, -1)
        self.target_markers.visualize(translations=target_world, orientations=default_quat)
        self.ee_markers.visualize(translations=ee_pos, orientations=default_quat)

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        """Return loco obs for DirectRLEnv compatibility"""
        return {"policy": self._get_loco_obs()}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple:
        height = self.robot.data.root_pos_w[:, 2]
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(self.robot.data.root_quat_w, gravity_vec)

        fallen = (height < 0.3) | (height > 1.2)
        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

        terminated = fallen | bad_orientation
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

        self.phase[env_ids] = torch.rand(n, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0

        # Sample new target positions (no marker update here - done in _pre_physics_step)
        self._sample_target_positions(env_ids)

    def _sample_target_positions(self, env_ids):
        """Sample target positions without updating markers"""
        n = len(env_ids)
        if n == 0:
            return

        # Shoulder offset in body frame (right shoulder)
        shoulder_offset = torch.tensor([0.0, -0.174, 0.259], device=self.device)

        # Sample in front-right hemisphere
        azimuth = torch.empty(n, device=self.device).uniform_(-0.5, 1.0)
        radius = torch.empty(n, device=self.device).uniform_(0.20, 0.35)
        height = torch.empty(n, device=self.device).uniform_(-0.10, 0.15)

        x = radius * torch.cos(azimuth)
        y = -radius * torch.sin(azimuth)
        z = height

        self.target_pos_body[env_ids, 0] = x + shoulder_offset[0]
        self.target_pos_body[env_ids, 1] = y + shoulder_offset[1]
        self.target_pos_body[env_ids, 2] = z + shoulder_offset[2]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = "cuda:0"

    # Load policies
    print("\n[1/4] Loading LOCO policy (Stage 3)...")
    loco_ckpt = torch.load(args_cli.loco_checkpoint, map_location=device, weights_only=False)
    loco_net = LocoActorCritic(57, 12).to(device)
    loco_net.load_state_dict(loco_ckpt["actor_critic"])
    loco_net.eval()
    print("      ✓ LOCO policy loaded!")

    print("[2/4] Loading ARM policy (Stage 5)...")
    arm_ckpt = torch.load(args_cli.arm_checkpoint, map_location=device, weights_only=False)
    arm_net = ArmActorCritic(29, 5).to(device)

    # Try different checkpoint key formats
    if "actor_critic" in arm_ckpt:
        arm_net.load_state_dict(arm_ckpt["actor_critic"])
    elif "model" in arm_ckpt:
        arm_net.load_state_dict(arm_ckpt["model"])
    elif "model_state_dict" in arm_ckpt:
        arm_net.load_state_dict(arm_ckpt["model_state_dict"])
    else:
        # Print available keys for debugging
        print(f"      Available keys: {list(arm_ckpt.keys())}")
        raise KeyError("Could not find model weights in checkpoint")

    arm_net.eval()
    print("      ✓ ARM policy loaded!")

    # Create environment
    print("[3/4] Creating environment...")
    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = DualPlayEnv(cfg)
    print("      ✓ Environment created!")

    # Run simulation
    print("[4/4] Running simulation...")
    print("-" * 70)

    obs_dict, _ = env.reset()

    for step in range(args_cli.steps):
        with torch.no_grad():
            # Get observations
            loco_obs = env._get_loco_obs()
            arm_obs = env._get_arm_obs()

            # Run policies
            leg_actions = loco_net.act(loco_obs, deterministic=True)
            arm_actions = arm_net.act(arm_obs, deterministic=True)

            # Combine
            actions = torch.cat([leg_actions, arm_actions], dim=-1)

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        # Logging
        if (step + 1) % 200 == 0:
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()

            ee_pos = env._compute_ee_pos()
            root_pos = env.robot.data.root_pos_w
            root_quat = env.robot.data.root_quat_w
            target_world = root_pos + quat_apply(root_quat, env.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1).mean().item()

            # Debug coordinates
            ee_local = ee_pos[0] - root_pos[0]
            target_local = env.target_pos_body[0]

            print(
                f"[Step {step + 1:5d}] H={height:.3f}m | Vx={vx:.2f}m/s | EE dist={dist:.3f}m | Reaches={env.reach_count}")
            print(f"    Target (body): [{target_local[0]:.3f}, {target_local[1]:.3f}, {target_local[2]:.3f}]")
            print(f"    EE (body):     [{ee_local[0]:.3f}, {ee_local[1]:.3f}, {ee_local[2]:.3f}]")
            print(
                f"    Diff:          [{(target_local[0] - ee_local[0]):.3f}, {(target_local[1] - ee_local[1]):.3f}, {(target_local[2] - ee_local[2]):.3f}]")

    print("=" * 70)
    print("DUAL POLICY PLAY COMPLETE - V3")
    print("=" * 70)
    print(f"  Total reaches: {env.reach_count}")
    print(f"  Steps: {args_cli.steps}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()