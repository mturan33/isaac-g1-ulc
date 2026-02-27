#!/usr/bin/env python3
"""
ULC G1 Stage 6 Unified Play Script
===================================
Test the unified loco-manipulation policy (gaming demonstration for paper).

Features:
- Locomotion + Arm reaching + Gripper control
- Target visualization (green sphere) + EE visualization (red sphere)
- Hybrid checkpoint loading: loco from Stage 7, arm from Stage 6
- Paper mode: 30s video with rotating camera angles
- Lighting and robot material matching Stage 7 look

Usage:
    # Default play (Stage 6 only)
    ./isaaclab.bat -p play_ulc_stage6_unified.py \
        --checkpoint logs/ulc/.../model_final.pt \
        --num_envs 1 --mode standing

    # Hybrid: good loco (Stage 7) + gaming arm (Stage 6)
    ./isaaclab.bat -p play_ulc_stage6_unified.py \
        --checkpoint logs/ulc/.../model_final.pt \
        --loco_checkpoint logs/ulc/ulc_g1_stage7_.../model_best.pt \
        --num_envs 1 --mode standing

    # Paper mode: 30s video, 3 camera angles, hybrid checkpoint
    ./isaaclab.bat -p play_ulc_stage6_unified.py \
        --checkpoint logs/ulc/.../model_final.pt \
        --loco_checkpoint logs/ulc/ulc_g1_stage7_.../model_best.pt \
        --mode paper

V1 (2026-01-31): Initial play script.
V2 (2026-02-15): Hybrid checkpoint, paper mode, lighting, robot material.
"""

from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Parse arguments before Isaac imports
parser = argparse.ArgumentParser(description="ULC G1 Stage 6 Unified Play")
parser.add_argument("--checkpoint", type=str, required=True, help="Stage 6 checkpoint path")
parser.add_argument("--loco_checkpoint", type=str, default=None,
                    help="Optional: Stage 7 checkpoint for better loco weights")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--steps", type=int, default=3000, help="Steps to run")
parser.add_argument("--mode", type=str, default="standing",
                    choices=["standing", "walking", "fast", "demo", "paper"],
                    help="Test mode")
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Use stochastic actions (default: deterministic)")
# Video recording
parser.add_argument("--record", action="store_true", default=False,
                    help="Enable video recording")
parser.add_argument("--record_duration", type=float, default=30.0,
                    help="Recording duration in seconds (default: 30)")
parser.add_argument("--fps", type=int, default=30,
                    help="Video FPS (default: 30)")
parser.add_argument("--camera_angle", type=str, default="front_right",
                    choices=["front_right", "front_left", "right_side", "front", "top"],
                    help="Camera viewing angle")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

use_deterministic = not args.stochastic

# Paper mode: force record, 1 env, 30s
if args.mode == "paper":
    args.num_envs = 1
    args.record = True
    args.record_duration = 30.0
    args.fps = 30

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# ============================================================================
# CONSTANTS
# ============================================================================
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
REACH_THRESHOLD = 0.06

G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

# Mode configurations
MODE_CONFIGS = {
    "standing": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.18, 0.35),
        "description": "Standing still, arm reaching (gaming demo)"
    },
    "walking": {
        "vx_range": (0.2, 0.4),
        "vy_range": (-0.05, 0.05),
        "vyaw_range": (-0.1, 0.1),
        "workspace_radius": (0.18, 0.40),
        "description": "Normal walking with arm reaching"
    },
    "fast": {
        "vx_range": (0.4, 0.6),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.2, 0.2),
        "workspace_radius": (0.18, 0.40),
        "description": "Fast walking with arm reaching"
    },
    "demo": {
        "vx_range": (0.0, 0.5),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "description": "Mixed demo mode"
    },
    "paper": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.18, 0.40),
        "description": "Paper video: 30s standing gaming demo"
    },
}

# Camera presets: (eye_position, target_position)
# NOTE: 23DoF model uses -X as forward direction (opposite of 29DoF)
# Robot's right side is -Y in this model
CAMERA_PRESETS = {
    "front_right": ((1.8, -1.3, 1.3), (0.0, 0.0, 0.65)),
    "front_left":  ((1.8, 1.3, 1.3), (0.0, 0.0, 0.65)),
    "right_side":  ((0.0, -2.0, 1.1), (0.0, 0.0, 0.65)),
    "front":       ((2.5, 0.0, 1.3), (0.0, 0.0, 0.65)),
    "top":         ((0.0, 0.0, 3.0), (0.0, 0.0, 0.65)),
}


# ============================================================================
# VIDEO RECORDING
# ============================================================================

class FrameRecorder:
    """Captures viewport frames and converts to MP4 via ffmpeg.

    Slow-motion video: captures every simulation step and encodes at output_fps.
    Each step = 0.02s simulation time but displayed at 1/30s = 0.033s.
    This gives 1.67x slow-motion effect which makes robot movements look natural.
    For 30s video: 900 frames / 30fps = 30s.
    """

    def __init__(self, output_dir: str, output_fps: int = 30):
        self.output_dir = output_dir
        self.output_fps = output_fps
        self.frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0

        from omni.kit.viewport.utility import get_active_viewport
        self.viewport = get_active_viewport()

    def capture_frame(self):
        from omni.kit.viewport.utility import capture_viewport_to_file
        frame_path = os.path.join(self.frame_dir, f"frame_{self.frame_count:06d}.png")
        capture_viewport_to_file(self.viewport, frame_path)
        self.frame_count += 1

    def finalize_video(self, output_name: str = "stage6_gaming_demo.mp4"):
        import subprocess
        import shutil

        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(self.frame_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.output_fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path
        ]

        expected_duration = self.frame_count / self.output_fps
        print(f"\n[VIDEO] {self.frame_count} frames -> MP4 @ {self.output_fps}fps")
        print(f"  Expected duration: {expected_duration:.1f}s (1.67x slow-motion)")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[VIDEO] Kaydedildi: {output_path}")
            shutil.rmtree(self.frame_dir)
            print(f"[VIDEO] Frame'ler temizlendi")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[VIDEO] ffmpeg hatasi: {e}")
            print(f"[VIDEO] Frame'ler: {self.frame_dir}")
            return None
        except FileNotFoundError:
            print(f"[VIDEO] ffmpeg bulunamadi! Frame'ler: {self.frame_dir}")
            return None


# ============================================================================
# NETWORK ARCHITECTURE (must match Stage 6 training)
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion actor: 57 obs -> 12 leg actions. [512,256,128]+LN+ELU."""
    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, x):
        return self.net(x)


class LocoCritic(nn.Module):
    def __init__(self, num_obs=57, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ArmActor(nn.Module):
    """Arm actor: 52 obs -> 12 actions (5 arm + 7 finger). [256,256,128]+ELU."""
    def __init__(self, num_obs=52, num_act=12, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, x):
        return self.net(x)


class ArmCritic(nn.Module):
    def __init__(self, num_obs=52, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualActorCritic(nn.Module):
    """Stage 6 Dual AC: LocoActor(57->12) + ArmActor(52->12)"""
    def __init__(self, loco_obs=57, arm_obs=52, loco_act=12, arm_act=12):
        super().__init__()
        self.loco_actor = LocoActor(loco_obs, loco_act)
        self.loco_critic = LocoCritic(loco_obs)
        self.arm_actor = ArmActor(arm_obs, arm_act)
        self.arm_critic = ArmCritic(arm_obs)

    def act(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        loco_act = torch.distributions.Normal(loco_mean, loco_std).sample()
        arm_act = torch.distributions.Normal(arm_mean, arm_std).sample()
        return loco_act, arm_act


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
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


# ============================================================================
# ENVIRONMENT CONFIG
# ============================================================================

@configclass
class PlaySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    # Dome light: ambient fill from all directions
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )
    # Distant light: directional sun-like light for metallic reflections
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 0.95), intensity=3000.0),
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=10.0,
            ),
            # Brushed steel material (same as Stage 7)
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.45, 0.48),
                metallic=0.85,
                roughness=0.4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85),
            joint_pos={
                "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
                "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
                "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.6, "right_elbow_pitch_joint": 0.6,
                "left_elbow_roll_joint": 0.0, "right_elbow_roll_joint": 0.0,
                "torso_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                stiffness=150.0, damping=15.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=80.0, damping=8.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0, damping=10.0,
            ),
        },
    )


@configclass
class PlayEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_space = 24  # 12 leg + 12 arm (5 arm joints + 7 finger)
    observation_space = 109  # 57 loco + 52 arm
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=4, env_spacing=2.5)


# ============================================================================
# PLAY ENVIRONMENT
# ============================================================================

class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_names = self.robot.joint_names

        leg_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]

        arm_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ]

        finger_names = [
            "right_zero_joint", "right_one_joint", "right_two_joint",
            "right_three_joint", "right_four_joint", "right_five_joint",
            "right_six_joint",
        ]

        self.leg_idx = torch.tensor([joint_names.index(n) for n in leg_names], device=self.device)
        self.arm_idx = torch.tensor([joint_names.index(n) for n in arm_names], device=self.device)
        self.finger_idx = torch.tensor(
            [joint_names.index(n) for n in finger_names if n in joint_names],
            device=self.device
        )

        # Finger limits for gripper normalization
        joint_limits = self.robot.data.joint_limits
        if len(self.finger_idx) > 0:
            self.finger_lower = torch.tensor(
                [joint_limits[0, i, 0].item() for i in self.finger_idx], device=self.device
            )
            self.finger_upper = torch.tensor(
                [joint_limits[0, i, 1].item() for i in self.finger_idx], device=self.device
            )
        else:
            self.finger_lower = torch.zeros(7, device=self.device)
            self.finger_upper = torch.ones(7, device=self.device)

        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )
        self.default_arm = self.robot.data.default_joint_pos[0, self.arm_idx].clone()
        if len(self.finger_idx) > 0:
            self.default_finger = self.robot.data.default_joint_pos[0, self.finger_idx].clone()
        else:
            self.default_finger = torch.zeros(7, device=self.device)

        # Find palm body for EE
        body_names = self.robot.body_names
        self.palm_idx = body_names.index("right_palm_link") if "right_palm_link" in body_names else None
        if self.palm_idx is None:
            print("[WARNING] right_palm_link not found, using right_elbow_roll_link")
            self.palm_idx = body_names.index("right_elbow_roll_link")

        self.shoulder_offset = torch.tensor([0.0, -0.174, 0.259], device=self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        # State
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.total_reaches = 0
        self.timed_out_targets = 0
        self.reach_threshold = REACH_THRESHOLD
        self.max_reach_steps = 150  # ~3s timeout per target
        self.steps_since_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.mode_cfg = MODE_CONFIGS[args.mode]

        print(f"[PlayEnv] Leg: {len(self.leg_idx)}, Arm: {len(self.arm_idx)}, "
              f"Finger: {len(self.finger_idx)}, Palm: {self.palm_idx}")
        print(f"[PlayEnv] default_arm: {self.default_arm.tolist()}")

        # Visualization markers
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
        )
        self.ee_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/EEMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
        )

    @property
    def robot(self):
        return self.scene["robot"]

    def _compute_palm_ee(self):
        """Compute palm EE position. Palm forward = LOCAL +X axis."""
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]

        w, x, y, z = palm_quat[:, 0], palm_quat[:, 1], palm_quat[:, 2], palm_quat[:, 3]
        fwd_x = 1 - 2 * (y * y + z * z)
        fwd_y = 2 * (x * y + w * z)
        fwd_z = 2 * (x * z - w * y)
        palm_forward = torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)

        PALM_FORWARD_OFFSET = 0.08
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward
        return ee_pos, palm_forward

    def _sample_targets(self, env_ids):
        """Sample new arm targets in body frame."""
        n = len(env_ids)
        ws = self.mode_cfg["workspace_radius"]

        azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
        elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
        radius = torch.empty(n, device=self.device).uniform_(*ws)

        x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
        y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
        z = radius * torch.sin(elevation) + self.shoulder_offset[2]

        self.target_pos_body[env_ids, 0] = x.clamp(0.05, 0.55)
        self.target_pos_body[env_ids, 1] = y.clamp(-0.55, 0.10)
        self.target_pos_body[env_ids, 2] = z.clamp(-0.25, 0.55)

    def _sample_commands(self, env_ids):
        """Sample velocity commands."""
        n = len(env_ids)
        cfg = self.mode_cfg

        self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*cfg["vx_range"])
        self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*cfg["vy_range"])
        self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*cfg["vyaw_range"])
        self._sample_targets(env_ids)

    def _update_markers(self):
        """Update target and EE marker positions."""
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        quat_id = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self.target_markers.visualize(translations=target_world, orientations=quat_id)

        ee_pos, _ = self._compute_palm_ee()
        self.ee_markers.visualize(translations=ee_pos, orientations=quat_id)

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()

        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]
        finger_actions = actions[:, 17:24]

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5

        if len(self.finger_idx) > 0:
            finger_normalized = (finger_actions + 1.0) / 2.0
            finger_targets = self.finger_lower + finger_normalized * (self.finger_upper - self.finger_lower)
            target_pos[:, self.finger_idx] = finger_targets

        self.robot.set_joint_position_target(target_pos)
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
        self.prev_leg_actions = leg_actions.clone()
        self.prev_arm_actions = arm_actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
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

        torso_euler = quat_to_euler_xyz(quat)

        # Loco obs (57) - matches training exactly
        loco_obs = torch.cat([
            lin_vel_b,              # 3
            ang_vel_b,              # 3
            proj_gravity,           # 3
            joint_pos,              # 12
            joint_vel,              # 12
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,           # 3
            gait_phase,             # 2
            self.prev_leg_actions,  # 12
            self.torso_cmd,         # 3
            torso_euler,            # 3
        ], dim=-1)

        # Arm obs (52) - matches training exactly
        arm_pos = robot.data.joint_pos[:, self.arm_idx]
        arm_vel = robot.data.joint_vel[:, self.arm_idx] * 0.1

        ee_pos_world, palm_forward = self._compute_palm_ee()
        root_pos = robot.data.root_pos_w
        ee_pos_body = quat_apply_inverse(quat, ee_pos_world - root_pos)

        ee_vel_world = (ee_pos_world - self.prev_ee_pos) / 0.02
        ee_vel_body = quat_apply_inverse(quat, ee_vel_world)

        palm_quat = robot.data.body_quat_w[:, self.palm_idx]

        pos_error = self.target_pos_body - ee_pos_body
        pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5

        down_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        dot = (palm_forward * down_vec).sum(dim=-1)
        angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
        orient_error = (angle / np.pi).unsqueeze(-1)

        # Check reaches + timeout
        actual_dist = pos_error.norm(dim=-1)
        reached = actual_dist < self.reach_threshold
        self.steps_since_target += 1

        # Timeout: resample if target not reached within max_reach_steps
        timed_out = self.steps_since_target >= self.max_reach_steps
        timed_out_ids = timed_out.nonzero(as_tuple=True)[0]
        if len(timed_out_ids) > 0:
            self.timed_out_targets += len(timed_out_ids)
            self._sample_targets(timed_out_ids)
            self.steps_since_target[timed_out_ids] = 0

        new_reaches = reached.sum().item()
        if new_reaches > 0:
            self.total_reaches += new_reaches
            reached_ids = reached.nonzero(as_tuple=True)[0]
            self._sample_targets(reached_ids)
            self.steps_since_target[reached_ids] = 0

        if len(self.finger_idx) > 0:
            finger_pos = robot.data.joint_pos[:, self.finger_idx]
            finger_normalized = (finger_pos - self.finger_lower) / (self.finger_upper - self.finger_lower + 1e-6)
            gripper_closed = finger_normalized.mean(dim=-1, keepdim=True)
            finger_vel = robot.data.joint_vel[:, self.finger_idx]
            grip_force = (finger_vel.abs().mean(dim=-1, keepdim=True) * gripper_closed).clamp(0, 1)
        else:
            finger_pos = torch.zeros(self.num_envs, 7, device=self.device)
            grip_force = torch.zeros(self.num_envs, 1, device=self.device)
            gripper_closed = torch.zeros(self.num_envs, 1, device=self.device)

        contact = (actual_dist < 0.05).float().unsqueeze(-1)
        target_reached = reached.float().unsqueeze(-1)

        height_cmd = self.height_cmd.unsqueeze(-1)
        current_height = robot.data.root_pos_w[:, 2:3]
        height_err = (height_cmd - current_height) / 0.4

        estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
        object_in_hand = torch.zeros(self.num_envs, 1, device=self.device)
        object_rel_ee = torch.zeros(self.num_envs, 3, device=self.device)

        base_lin_vel = lin_vel_b[:, :2]
        base_ang_vel = ang_vel_b[:, 2:3]

        arm_obs = torch.cat([
            arm_pos,            # 5
            arm_vel,            # 5
            finger_pos,         # 7
            ee_pos_body,        # 3
            ee_vel_body,        # 3
            palm_quat,          # 4
            grip_force,         # 1
            gripper_closed,     # 1
            contact,            # 1
            self.target_pos_body,  # 3
            pos_error,          # 3
            pos_dist,           # 1
            orient_error,       # 1
            target_reached,     # 1
            height_cmd,         # 1
            current_height,     # 1
            height_err,         # 1
            estimated_load,     # 3
            object_in_hand,     # 1
            object_rel_ee,      # 3
            base_lin_vel,       # 2
            base_ang_vel,       # 1
        ], dim=-1)

        self._update_markers()
        self.prev_ee_pos = ee_pos_world.clone()

        return {
            "policy": torch.cat([loco_obs, arm_obs], dim=-1).clamp(-10, 10).nan_to_num(),
            "loco": loco_obs.clamp(-10, 10).nan_to_num(),
            "arm": arm_obs.clamp(-10, 10).nan_to_num(),
        }

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

        # wxyz quaternion (Isaac Lab convention)
        default_pos = torch.tensor([[0.0, 0.0, 0.85]], device=self.device).expand(len(env_ids), -1).clone()
        default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(len(env_ids), -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0
        self.prev_ee_pos[env_ids] = 0
        self.steps_since_target[env_ids] = 0
        self._sample_commands(env_ids)


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda:0"

    print(f"\n{'=' * 60}")
    print("ULC G1 STAGE 6 UNIFIED - PLAY (GAMING DEMO)")
    print(f"{'=' * 60}")
    print(f"  Stage 6 checkpoint: {args.checkpoint}")
    if args.loco_checkpoint:
        print(f"  Loco checkpoint (Stage 7): {args.loco_checkpoint}")
    print(f"  Mode: {args.mode} | {MODE_CONFIGS[args.mode]['description']}")
    if args.record:
        print(f"  RECORDING: {args.record_duration}s @ {args.fps}fps")
    print(f"{'=' * 60}\n")

    # Load Stage 6 checkpoint
    s6_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"[Stage 6 Checkpoint]")
    for key in ["best_reward", "iteration", "curriculum_level"]:
        if key in s6_ckpt:
            print(f"  {key}: {s6_ckpt[key]}")

    s6_state_raw = s6_ckpt.get("model", s6_ckpt)

    # Remap Stage 6 checkpoint keys to match play model
    # Stage 6 training used: loco_actor.actor.* / arm_actor.actor.* / critic.critic.*
    # Play model uses:       loco_actor.net.*   / arm_actor.net.*   / loco_critic.net.* + arm_critic.net.*
    def remap_s6_keys(state_dict):
        """Remap Stage 6 checkpoint keys: .actor. -> .net., critic.critic -> loco_critic.net"""
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            # Actor Sequential name: .actor. -> .net.
            new_k = new_k.replace("loco_actor.actor.", "loco_actor.net.")
            new_k = new_k.replace("arm_actor.actor.", "arm_actor.net.")
            # Shared critic -> loco_critic (arm_critic ignored, not used in inference)
            new_k = new_k.replace("critic.critic.", "loco_critic.net.")
            remapped[new_k] = v
        return remapped

    s6_state = remap_s6_keys(s6_state_raw)
    remap_count = sum(1 for old_k, new_k in zip(s6_state_raw.keys(), s6_state.keys()) if old_k != new_k)
    print(f"[Key Remap] {remap_count}/{len(s6_state_raw)} keys remapped (.actor. -> .net.)")
    if remap_count > 0:
        # Show first few remaps
        for old_k, new_k in zip(s6_state_raw.keys(), s6_state.keys()):
            if old_k != new_k:
                print(f"  {old_k} -> {new_k}")
                break  # Just show first one as example

    # Create environment
    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = PlayEnv(cfg)

    # Create network (Stage 6 architecture)
    net = DualActorCritic(57, 52, 12, 12).to(device)

    if args.loco_checkpoint:
        # Hybrid loading: loco from Stage 7, arm from Stage 6
        s7_ckpt = torch.load(args.loco_checkpoint, map_location=device, weights_only=False)
        print(f"\n[Stage 7 Checkpoint (loco only)]")
        for key in ["best_reward", "iteration", "curriculum_level"]:
            if key in s7_ckpt:
                print(f"  {key}: {s7_ckpt[key]}")

        s7_state = s7_ckpt.get("model", s7_ckpt)

        # Load weights: loco from Stage 7, arm from Stage 6
        loco_loaded = 0
        arm_loaded = 0
        net_state = net.state_dict()

        for key in net_state.keys():
            if key.startswith("loco_actor.") or key.startswith("loco_critic."):
                # Try loading from Stage 7 first (better loco policy)
                if key in s7_state and s7_state[key].shape == net_state[key].shape:
                    net_state[key] = s7_state[key]
                    loco_loaded += 1
                elif key in s6_state and s6_state[key].shape == net_state[key].shape:
                    net_state[key] = s6_state[key]
                    loco_loaded += 1
                    print(f"  [FALLBACK] {key} from Stage 6")
            elif key.startswith("arm_actor.") or key.startswith("arm_critic."):
                # Load from Stage 6 (gaming arm policy)
                if key in s6_state and s6_state[key].shape == net_state[key].shape:
                    net_state[key] = s6_state[key]
                    arm_loaded += 1

        net.load_state_dict(net_state)
        print(f"\n[Hybrid Loading] Loco: {loco_loaded} keys from Stage 7, Arm: {arm_loaded} keys from Stage 6")
    else:
        # Load everything from Stage 6
        model_keys = set(net.state_dict().keys())
        ckpt_keys = set(s6_state.keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        if missing:
            print(f"\n  MISSING keys: {sorted(missing)[:10]}")
        if unexpected:
            print(f"\n  UNEXPECTED keys: {sorted(unexpected)[:10]}")
        if not missing and not unexpected:
            print(f"  All {len(model_keys)} keys match!")

        try:
            net.load_state_dict(s6_state, strict=True)
            print(f"[INFO] Loaded Stage 6 checkpoint ({len(model_keys)} keys)")
        except RuntimeError as e:
            print(f"  STRICT FAILED: {e}")
            # Shape-aware loading: skip keys with shape mismatch
            # This handles UnifiedCritic(109) -> LocoCritic(57) mismatch
            # Critics are not used during inference, so skipping is safe
            net_state = net.state_dict()
            loaded_count = 0
            skipped_keys = []
            for key in net_state.keys():
                if key in s6_state and s6_state[key].shape == net_state[key].shape:
                    net_state[key] = s6_state[key]
                    loaded_count += 1
                else:
                    skipped_keys.append(key)
            net.load_state_dict(net_state)
            print(f"  Shape-aware loading: {loaded_count}/{len(net_state)} keys loaded")
            if skipped_keys:
                print(f"  Skipped (shape mismatch or missing): {skipped_keys}")

    net.eval()

    # Camera setup
    from isaacsim.core.utils.viewports import set_camera_view
    eye, target_cam = CAMERA_PRESETS[args.camera_angle]
    set_camera_view(eye=eye, target=target_cam)

    # Video recorder setup
    recorder = None
    env_dt = 0.02
    if args.record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = os.path.join(os.getcwd(), "recordings", f"stage6_{timestamp}")
        recorder = FrameRecorder(record_dir, output_fps=args.fps)
        total_frames = int(args.record_duration * args.fps)  # 30s * 30fps = 900
        args.steps = total_frames
        print(f"[Record] {args.record_duration}s video = {total_frames} frames @ {args.fps}fps, "
              f"1.67x slow-motion ({total_frames * env_dt:.1f}s sim)")

    obs, _ = env.reset()
    set_camera_view(eye=eye, target=target_cam)

    prev_reaches = 0

    # Paper mode: relaxed thresholds for better demo
    if args.mode == "paper":
        env.reach_threshold = 0.08
        env.max_reach_steps = 80  # ~1.6s timeout (matches Stage 7 paper mode)
        env._sample_commands(torch.arange(env.num_envs, device=device))
        print(f"\n[PAPER MODE] {args.steps} steps -> {args.record_duration:.0f}s video (1.67x slow-motion)")
        print(f"  Standing only, reach_threshold={env.reach_threshold}m, timeout={env.max_reach_steps} steps")

    # Camera rotation for paper mode
    paper_cam_angles = [
        ("front_right", CAMERA_PRESETS["front_right"]),
        ("front",       CAMERA_PRESETS["front"]),
        ("top",         CAMERA_PRESETS["top"]),
    ]
    paper_cam_interval = int(10.0 * args.fps)  # 300 steps = 10s video
    paper_cam_current_idx = -1

    # Camera tracking with EMA
    cam_smooth_x = 0.0
    cam_smooth_y = 0.0
    cam_ema_alpha = 0.3
    cam_eye_offset = torch.tensor(list(eye), dtype=torch.float32)
    cam_target_offset = torch.tensor(list(target_cam), dtype=torch.float32)

    print(f"\n[Play] {args.steps} steps | '{args.mode}' | {'DETERM' if use_deterministic else 'STOCH'}\n")

    with torch.no_grad():
        for step in range(args.steps):
            # Paper mode: rotate camera angle every 10s video time
            if args.mode == "paper":
                angle_idx = (step // paper_cam_interval) % len(paper_cam_angles)
                if angle_idx != paper_cam_current_idx:
                    paper_cam_current_idx = angle_idx
                    angle_name, (new_eye, new_target) = paper_cam_angles[angle_idx]
                    cam_eye_offset = torch.tensor(list(new_eye), dtype=torch.float32)
                    cam_target_offset = torch.tensor(list(new_target), dtype=torch.float32)
                    vid_time = step / args.fps
                    print(f"[CAMERA] {vid_time:.0f}s -> {angle_name}")

            # Camera tracking: responsive EMA follow, update every 2 steps
            if args.mode == "paper" and step % 2 == 0:
                robot_pos = env.robot.data.root_pos_w[0].cpu()
                rx = robot_pos[0].item()
                ry = robot_pos[1].item()
                cam_smooth_x += cam_ema_alpha * (rx - cam_smooth_x)
                cam_smooth_y += cam_ema_alpha * (ry - cam_smooth_y)
                cam_eye_world = (cam_smooth_x + cam_eye_offset[0].item(),
                                 cam_smooth_y + cam_eye_offset[1].item(),
                                 cam_eye_offset[2].item())
                cam_target_world = (cam_smooth_x + cam_target_offset[0].item(),
                                    cam_smooth_y + cam_target_offset[1].item(),
                                    cam_target_offset[2].item())
                set_camera_view(eye=cam_eye_world, target=cam_target_world)

            loco_obs = obs["loco"]
            arm_obs = obs["arm"]

            leg_actions, arm_actions = net.act(loco_obs, arm_obs, deterministic=use_deterministic)
            actions = torch.cat([leg_actions, arm_actions], dim=-1)

            obs, reward, terminated, truncated, info = env.step(actions)

            # Capture every step for slow-motion video
            if recorder:
                recorder.capture_frame()

            # Report reaches
            if env.total_reaches > prev_reaches:
                new = env.total_reaches - prev_reaches
                elapsed = step * env_dt
                print(f"[{elapsed:5.1f}s Step {step:4d}] +{new} REACH (total={env.total_reaches})")
                prev_reaches = env.total_reaches

            # Resample commands periodically
            if step > 0 and step % 500 == 0:
                env._sample_commands(torch.arange(env.num_envs, device=device))

            # Progress report
            if step > 0 and step % 200 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()
                ee_pos, _ = env._compute_palm_ee()
                root_pos = env.robot.data.root_pos_w
                root_quat = env.robot.data.root_quat_w
                ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
                ee_dist = (ee_body - env.target_pos_body).norm(dim=-1).mean().item()

                print(
                    f"[Step {step:4d}] "
                    f"H={height:.3f}m | Vx={vx:.2f}m/s | "
                    f"EE_dist={ee_dist:.3f}m | Reaches={env.total_reaches}"
                )

    # Finalize video
    if recorder:
        if args.mode == "paper":
            video_name = "g1_stage6_gaming_30s.mp4"
        else:
            video_name = f"g1_stage6_{args.mode}.mp4"
        video_path = recorder.finalize_video(video_name)
        if video_path:
            video_duration = recorder.frame_count / args.fps
            sim_duration = recorder.frame_count * env_dt
            print(f"\n[VIDEO] Saved: {video_path}")
            print(f"  Frames: {recorder.frame_count}, FPS: {args.fps}")
            print(f"  Video duration: {video_duration:.1f}s (sim: {sim_duration:.1f}s, 1.67x slow-motion)")

    print(f"\n{'=' * 60}")
    print("STAGE 6 PLAY COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total reaches: {env.total_reaches}")
    print(f"  Timed out: {env.timed_out_targets}")
    total_attempts = env.total_reaches + env.timed_out_targets
    if total_attempts > 0:
        print(f"  Reach rate: {env.total_reaches / total_attempts:.1%}")
    if args.loco_checkpoint:
        print(f"  (Hybrid: Stage 7 loco + Stage 6 arm)")
    print(f"{'=' * 60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
