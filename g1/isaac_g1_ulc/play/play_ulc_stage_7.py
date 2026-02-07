#!/usr/bin/env python3
"""
ULC G1 Stage 7 Play Script
============================
Evaluation script for Stage 7 anti-gaming arm reaching.

KEY DIFFERENCES FROM STAGE 6 PLAY:
1. Arm obs = 55 dim (52 + 3 anti-gaming: steps_since_spawn, ee_displacement, initial_distance)
2. Validated reach counting (position + displacement + time)
3. Absolute-only target sampling with min distance enforcement
4. Reports validated vs total reaches, ee_displacement, ee_speed

Usage:
    # Default: standing mode with anti-gaming validation
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 1 --mode standing --no_orient

    # Showcase mode for video recording
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 1 --mode showcase --no_orient

    # Record 10-second demo video
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 1 --mode showcase --no_orient --record --record_duration 10

    # Walking test
    ./isaaclab.bat -p play_ulc_stage7.py \
        --checkpoint logs/ulc/.../model_best.pt \
        --num_envs 1 --mode walking --no_orient
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="ULC G1 Stage 7 Play - Anti-Gaming")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--mode", type=str, default="standing",
                    choices=["standing", "walking", "fast", "demo", "showcase"])
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Use stochastic actions (default: deterministic)")
parser.add_argument("--reach_threshold", type=float, default=0.08,
                    help="Position reach threshold in meters")
parser.add_argument("--min_displacement", type=float, default=0.05,
                    help="Minimum EE displacement to count as valid reach (meters)")
parser.add_argument("--max_reach_steps", type=int, default=200,
                    help="Maximum steps to reach target before timeout")
parser.add_argument("--orient_check", action="store_true", default=False,
                    help="Enable orientation check for reaches (overrides curriculum)")
parser.add_argument("--no_orient", action="store_true", default=False,
                    help="Force disable orientation check (overrides curriculum level)")
parser.add_argument("--orient_threshold", type=float, default=2.0,
                    help="Orientation threshold in radians")
parser.add_argument("--env_spacing", type=float, default=5.0,
                    help="Spacing between environments")
# Video recording
parser.add_argument("--record", action="store_true", default=False,
                    help="Enable video recording")
parser.add_argument("--record_duration", type=float, default=10.0,
                    help="Recording duration in seconds (default: 10)")
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

# Showcase mode: force no_orient and 1 env
if args.mode == "showcase":
    args.no_orient = True
    args.num_envs = 1

app_launcher = AppLauncher(args)
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


# ============================================================================
# CONSTANTS
# ============================================================================
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
PALM_FORWARD_OFFSET = 0.08
SHOULDER_OFFSET = torch.tensor([0.0, -0.174, 0.259])

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

FINGER_JOINT_NAMES = [
    "right_zero_joint", "right_one_joint", "right_two_joint",
    "right_three_joint", "right_four_joint", "right_five_joint",
    "right_six_joint",
]

MODE_CONFIGS = {
    "standing": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.18, 0.35),
        "min_target_distance": 0.10,
        "description": "Standing still, arm reaching only"
    },
    "walking": {
        "vx_range": (0.1, 0.3),
        "vy_range": (-0.05, 0.05),
        "vyaw_range": (-0.1, 0.1),
        "workspace_radius": (0.18, 0.40),
        "min_target_distance": 0.12,
        "description": "Normal walking with arm reaching"
    },
    "fast": {
        "vx_range": (0.3, 0.5),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "min_target_distance": 0.15,
        "description": "Fast walking with arm reaching"
    },
    "demo": {
        "vx_range": (0.0, 0.4),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.15, 0.15),
        "workspace_radius": (0.18, 0.40),
        "min_target_distance": 0.12,
        "description": "Mixed demo mode"
    },
    "showcase": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "workspace_radius": (0.20, 0.35),
        "min_target_distance": 0.08,
        "description": "Showcase: close targets, fast retry, no timeout stalls"
    },
}

# Camera presets: (eye_position, target_position)
CAMERA_PRESETS = {
    "front_right": ((-1.2, 1.0, 1.2), (0.0, 0.0, 0.75)),
    "front_left":  ((-1.2, -1.0, 1.2), (0.0, 0.0, 0.75)),
    "right_side":  ((0.0, 1.5, 1.0), (0.0, 0.0, 0.75)),
    "front":       ((-2.0, 0.0, 1.2), (0.0, 0.0, 0.75)),
    "top":         ((0.0, 0.0, 2.5), (0.0, 0.0, 0.75)),
}


# ============================================================================
# VIDEO RECORDING
# ============================================================================

class FrameRecorder:
    """Captures viewport frames and converts to MP4 via ffmpeg."""

    def __init__(self, output_dir: str, fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
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

    def finalize_video(self, output_name: str = "stage7_reaching_demo.mp4"):
        import subprocess
        import shutil

        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(self.frame_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path
        ]

        print(f"\n[VIDEO] {self.frame_count} frame -> MP4 donusturuluyor...")
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
# TRAINING CURRICULUM LEVELS (must match train_ulc_stage_7.py EXACTLY)
# Used to set correct thresholds for observation computation
# ============================================================================
TRAINING_CURRICULUM = [
    # Level 0: Stand+Reach easy
    {"pos_threshold": 0.10, "min_target_distance": 0.10, "min_displacement": 0.05,
     "max_reach_steps": 200, "use_orientation": False, "workspace_radius": (0.18, 0.30)},
    # Level 1
    {"pos_threshold": 0.08, "min_target_distance": 0.12, "min_displacement": 0.07,
     "max_reach_steps": 180, "use_orientation": False, "workspace_radius": (0.18, 0.33)},
    # Level 2
    {"pos_threshold": 0.07, "min_target_distance": 0.14, "min_displacement": 0.08,
     "max_reach_steps": 170, "use_orientation": False, "workspace_radius": (0.18, 0.36)},
    # Level 3
    {"pos_threshold": 0.06, "min_target_distance": 0.15, "min_displacement": 0.10,
     "max_reach_steps": 160, "use_orientation": False, "workspace_radius": (0.18, 0.40)},
    # Level 4: Walk+Reach slow
    {"pos_threshold": 0.06, "min_target_distance": 0.15, "min_displacement": 0.10,
     "max_reach_steps": 160, "use_orientation": False, "workspace_radius": (0.18, 0.40)},
    # Level 5: Walk+Reach normal
    {"pos_threshold": 0.05, "min_target_distance": 0.16, "min_displacement": 0.11,
     "max_reach_steps": 150, "use_orientation": False, "workspace_radius": (0.18, 0.40)},
    # Level 6: Walk+Orient
    {"pos_threshold": 0.05, "min_target_distance": 0.16, "min_displacement": 0.11,
     "max_reach_steps": 150, "orient_threshold": 2.5, "use_orientation": True, "workspace_radius": (0.18, 0.40)},
    # Level 7: FINAL
    {"pos_threshold": 0.04, "min_target_distance": 0.18, "min_displacement": 0.12,
     "max_reach_steps": 150, "orient_threshold": 2.0, "use_orientation": True, "workspace_radius": (0.18, 0.40)},
]


# ============================================================================
# NETWORK - MATCHES TRAINING (55 arm obs)
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
    """55 obs -> 5 actions (Stage 7: anti-gaming obs)"""
    def __init__(self, num_obs=55, num_act=5, hidden=[256, 256, 128]):
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
    def __init__(self, num_obs=55, hidden=[256, 256, 128]):
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
    def __init__(self, loco_obs=57, arm_obs=55, loco_act=12, arm_act=5):
        super().__init__()
        self.loco_actor = LocoActor(loco_obs, loco_act)
        self.loco_critic = LocoCritic(loco_obs)
        self.arm_actor = ArmActor(arm_obs, arm_act)
        self.arm_critic = ArmCritic(arm_obs)

    def get_actions(self, loco_obs, arm_obs, deterministic=True):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)
        if deterministic:
            return loco_mean, arm_mean
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        return (torch.distributions.Normal(loco_mean, loco_std).sample(),
                torch.distributions.Normal(arm_mean, arm_std).sample())


# ============================================================================
# HELPERS
# ============================================================================

def quat_to_euler_xyz(quat):
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


def get_palm_forward(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=-1)


def compute_orientation_error(palm_quat, target_dir=None):
    forward = get_palm_forward(palm_quat)
    if target_dir is None:
        target_dir = torch.zeros_like(forward)
        target_dir[:, 2] = -1.0
    dot = torch.clamp((forward * target_dir).sum(dim=-1), -1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# ENVIRONMENT
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
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=10.0,
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
                stiffness=150.0, damping=15.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=50.0, damping=5.0,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*zero.*", ".*one.*", ".*two.*", ".*three.*", ".*four.*", ".*five.*", ".*six.*"],
                stiffness=20.0, damping=2.0,
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
    action_space = 17
    observation_space = 57
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=4, env_spacing=5.0)


class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_names = self.robot.joint_names
        self.leg_idx = torch.tensor(
            [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
            device=self.device
        )
        self.arm_idx = torch.tensor(
            [joint_names.index(n) for n in ARM_JOINT_NAMES if n in joint_names],
            device=self.device
        )
        self.finger_idx = torch.tensor(
            [joint_names.index(n) for n in FINGER_JOINT_NAMES if n in joint_names],
            device=self.device
        )

        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )
        self.default_arm = self.robot.data.default_joint_pos[0, self.arm_idx].clone()

        joint_limits = self.robot.root_physx_view.get_dof_limits()
        self.finger_lower = torch.tensor(
            [joint_limits[0, i, 0].item() for i in self.finger_idx], device=self.device
        )
        self.finger_upper = torch.tensor(
            [joint_limits[0, i, 1].item() for i in self.finger_idx], device=self.device
        )

        body_names = self.robot.body_names
        self.palm_idx = None
        for i, name in enumerate(body_names):
            if "right_palm" in name.lower():
                self.palm_idx = i
                break
        if self.palm_idx is None:
            self.palm_idx = len(body_names) - 1

        self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

        # Commands
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_orient_body[:, 2] = -1.0

        self.phase = torch.zeros(self.num_envs, device=self.device)

        # Action history
        self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Anti-gaming state buffers
        self.ee_pos_at_spawn = torch.zeros(self.num_envs, 3, device=self.device)
        self.steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.initial_dist = torch.zeros(self.num_envs, device=self.device)

        # Stuck detection
        self.ee_pos_history = torch.zeros(self.num_envs, 3, device=self.device)
        self.stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stuck_resample_count = 0

        # Config - will be overridden by set_curriculum_level() from checkpoint
        self.reach_pos_threshold = args.reach_threshold
        self.min_displacement = args.min_displacement
        self.max_reach_steps = args.max_reach_steps
        self.use_orient_check = args.orient_check
        self.orient_threshold = args.orient_threshold
        self.curriculum_level = 0  # Will be set from checkpoint

        # Training-matched threshold for observation computation
        # This is the pos_threshold from the training curriculum level
        # CRITICAL: must match training so target_reached obs is consistent
        self.obs_pos_threshold = args.reach_threshold  # Default, overridden by set_curriculum_level

        # Stats
        self.total_reaches = 0
        self.validated_reaches = 0
        self.pos_only_reaches = 0
        self.timed_out_targets = 0
        self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reach_distances = []
        self.reach_displacements = []

        # Per-step reach info for logging (stores PRE-resample values)
        self.last_reach_dists = []
        self.last_reach_disps = []
        self.last_reach_count = 0

        self.mode_cfg = MODE_CONFIGS[args.mode]
        self._markers_initialized = False

        print(f"\n[PlayEnv Stage 7] Configuration:")
        print(f"  Envs: {self.num_envs}, Spacing: {args.env_spacing}m")
        print(f"  Mode: {args.mode} ({self.mode_cfg['description']})")
        print(f"  CLI Pos threshold: {self.reach_pos_threshold:.3f}m")
        print(f"  Min displacement: {self.min_displacement:.3f}m")
        print(f"  Max reach steps: {self.max_reach_steps}")
        print(f"  Orient check: {self.use_orient_check}")
        print(f"  Deterministic: {use_deterministic}")

    def set_curriculum_level(self, level):
        """Set thresholds from training curriculum level to match training obs exactly."""
        self.curriculum_level = level
        if level < len(TRAINING_CURRICULUM):
            lv = TRAINING_CURRICULUM[level]
            # Use training curriculum's pos_threshold for observation computation
            self.obs_pos_threshold = lv["pos_threshold"]
            # Also use training thresholds for reach validation
            self.reach_pos_threshold = lv["pos_threshold"]
            self.min_displacement = lv["min_displacement"]
            self.max_reach_steps = lv["max_reach_steps"]
            self.orient_threshold = lv.get("orient_threshold", 2.0)
            # Update mode workspace/distance from curriculum
            self.mode_cfg["min_target_distance"] = lv["min_target_distance"]
            self.mode_cfg["workspace_radius"] = lv["workspace_radius"]

            # Orientation check: CLI flags override curriculum
            if args.no_orient:
                self.use_orient_check = False
                print(f"\n[PlayEnv] --no_orient: Orientation check DISABLED (CLI override)")
            elif args.orient_check:
                self.use_orient_check = True
                print(f"\n[PlayEnv] --orient_check: Orientation check ENABLED (CLI override)")
            else:
                self.use_orient_check = lv.get("use_orientation", False)

            print(f"\n[PlayEnv] Loaded training curriculum Level {level}:")
            print(f"  pos_threshold: {self.reach_pos_threshold:.3f}m (used for obs AND reach check)")
            print(f"  min_displacement: {self.min_displacement:.3f}m")
            print(f"  max_reach_steps: {self.max_reach_steps}")
            print(f"  min_target_distance: {lv['min_target_distance']:.3f}m")
            print(f"  workspace_radius: {lv['workspace_radius']}")
            print(f"  use_orientation: {self.use_orient_check} (curriculum={lv.get('use_orientation', False)})")
            print(f"  orient_threshold: {self.orient_threshold:.2f} rad")

            # Showcase mode: relax thresholds for smoother demo
            if args.mode == "showcase":
                self.reach_pos_threshold = 0.08  # Relaxed from 0.04 to 0.08
                self.min_displacement = 0.04     # Relaxed from 0.12 to 0.04
                self.max_reach_steps = 70        # 1.4s timeout, very fast retry
                print(f"\n[PlayEnv] SHOWCASE overrides:")
                print(f"  pos_threshold: {self.reach_pos_threshold:.3f}m (relaxed for demo)")
                print(f"  min_displacement: {self.min_displacement:.3f}m (relaxed)")
                print(f"  max_reach_steps: {self.max_reach_steps} (faster retry)")
        else:
            print(f"\n[PlayEnv] WARNING: Level {level} not in curriculum, using CLI args")

    @property
    def robot(self):
        return self.scene["robot"]

    def _init_markers(self):
        if self._markers_initialized:
            return
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
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
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
        )
        self._markers_initialized = True

    def _compute_palm_ee(self):
        palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
        palm_forward = get_palm_forward(palm_quat)
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward
        return ee_pos, palm_quat

    def _sample_commands(self, env_ids):
        """Absolute-only sampling with min distance enforcement"""
        n = len(env_ids)
        cfg = self.mode_cfg

        self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*cfg["vx_range"])
        self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*cfg["vy_range"])
        self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*cfg["vyaw_range"])
        self.height_cmd[env_ids] = HEIGHT_DEFAULT

        # Get current EE in body frame
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        ee_world, _ = self._compute_palm_ee()
        ee_world = ee_world[env_ids]
        current_ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

        # Absolute sampling
        ws = cfg["workspace_radius"]
        azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
        elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
        radius = torch.empty(n, device=self.device).uniform_(*ws)

        x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
        y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
        z = radius * torch.sin(elevation) + self.shoulder_offset[2]

        target_body = torch.stack([
            x.clamp(0.05, 0.55),
            y.clamp(-0.55, 0.10),
            z.clamp(-0.25, 0.55)
        ], dim=-1)

        # Enforce min distance from EE
        min_dist = cfg.get("min_target_distance", 0.10)
        dist_to_ee = torch.norm(target_body - current_ee_body, dim=-1)
        too_close = dist_to_ee < min_dist
        if too_close.any():
            direction = target_body - current_ee_body
            direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            direction = direction / direction_norm
            pushed = current_ee_body + min_dist * direction
            pushed[:, 0] = pushed[:, 0].clamp(0.05, 0.55)
            pushed[:, 1] = pushed[:, 1].clamp(-0.55, 0.10)
            pushed[:, 2] = pushed[:, 2].clamp(-0.25, 0.55)
            target_body = torch.where(too_close.unsqueeze(-1).expand_as(target_body), pushed, target_body)

        self.target_pos_body[env_ids] = target_body

        # Palm down orientation
        self.target_orient_body[env_ids, 0] = 0.0
        self.target_orient_body[env_ids, 1] = 0.0
        self.target_orient_body[env_ids, 2] = -1.0

        # Record spawn state
        self.ee_pos_at_spawn[env_ids] = current_ee_body
        self.steps_since_spawn[env_ids] = 0
        self.initial_dist[env_ids] = torch.norm(target_body - current_ee_body, dim=-1).clamp(min=0.01)
        self.already_reached[env_ids] = False

    def get_loco_obs(self):
        """57 dims - Same as training"""
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
        obs = torch.cat([
            lin_vel_b, ang_vel_b, proj_gravity, joint_pos, joint_vel,
            self.height_cmd.unsqueeze(-1), self.vel_cmd, gait_phase,
            self.prev_leg_actions, self.torso_cmd, torso_euler,
        ], dim=-1)
        return obs.clamp(-10, 10).nan_to_num()

    def get_arm_obs(self):
        """55 dims = 52 (base) + 3 (anti-gaming) - MUST match training"""
        robot = self.robot
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        arm_pos = robot.data.joint_pos[:, self.arm_idx]
        arm_vel = robot.data.joint_vel[:, self.arm_idx] * 0.1
        finger_pos = robot.data.joint_pos[:, self.finger_idx]

        ee_world, palm_quat = self._compute_palm_ee()
        ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)
        ee_vel_world = (ee_world - self.prev_ee_pos) / 0.02
        ee_vel_body = quat_apply_inverse(root_quat, ee_vel_world)

        finger_normalized = (finger_pos - self.finger_lower) / (self.finger_upper - self.finger_lower + 1e-6)
        gripper_closed_ratio = finger_normalized.mean(dim=-1, keepdim=True)
        finger_vel = robot.data.joint_vel[:, self.finger_idx]
        grip_force = (finger_vel.abs().mean(dim=-1, keepdim=True) * gripper_closed_ratio).clamp(0, 1)

        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist_to_target = torch.norm(ee_world - target_world, dim=-1, keepdim=True)
        contact_detected = (dist_to_target < 0.08).float()

        target_body = self.target_pos_body
        pos_error = target_body - ee_body
        pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5
        orient_err = compute_orientation_error(palm_quat, self.target_orient_body).unsqueeze(-1) / np.pi

        # CRITICAL: Use obs_pos_threshold (from training curriculum level) not CLI arg
        # This ensures target_reached observation matches what the policy was trained with
        orient_threshold = self.orient_threshold if self.use_orient_check else 1.0
        target_reached = ((dist_to_target < self.obs_pos_threshold) &
                         (orient_err * np.pi < orient_threshold)).float()

        current_height = root_pos[:, 2:3]
        height_cmd_obs = self.height_cmd.unsqueeze(-1)
        height_err = (height_cmd_obs - current_height) / 0.4

        estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
        object_in_hand_obs = torch.zeros(self.num_envs, 1, device=self.device)
        target_orient_obs = self.target_orient_body

        lin_vel_b = quat_apply_inverse(root_quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w)
        lin_vel_xy = lin_vel_b[:, :2]
        ang_vel_z = ang_vel_b[:, 2:3]

        # 3 anti-gaming observations
        max_steps = float(self.max_reach_steps)
        steps_norm = (self.steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)
        ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1, keepdim=True)
        initial_dist_obs = self.initial_dist.unsqueeze(-1) / 0.5

        obs = torch.cat([
            arm_pos, arm_vel, finger_pos,
            ee_body, ee_vel_body, palm_quat, grip_force, gripper_closed_ratio, contact_detected,
            target_body, pos_error, pos_dist, orient_err, target_reached,
            height_cmd_obs, current_height, height_err,
            estimated_load, object_in_hand_obs, target_orient_obs,
            lin_vel_xy, ang_vel_z,
            # 3 new
            steps_norm, ee_displacement, initial_dist_obs,
        ], dim=-1)
        return obs.clamp(-10, 10).nan_to_num()

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        leg_actions = actions[:, :12]
        arm_actions = actions[:, 12:17]

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
        target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5
        target_pos[:, self.finger_idx] = self.finger_lower

        self.robot.set_joint_position_target(target_pos)
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

        # Increment step counter
        self.steps_since_spawn += 1

        # Reach check with anti-gaming validation
        ee_pos, palm_quat = self._compute_palm_ee()
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
        target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
        dist = torch.norm(ee_pos - target_world, dim=-1)

        # Condition 1: Position close
        pos_only_close = dist < self.reach_pos_threshold
        pos_close = pos_only_close.clone()

        # Condition 1b: Optional orientation
        orient_err_val = compute_orientation_error(palm_quat, self.target_orient_body)
        if self.use_orient_check:
            orient_ok = orient_err_val < self.orient_threshold
            pos_close = pos_close & orient_ok

        # Track TRUE position-only reaches (ignoring orientation) for comparison
        pos_only_new = pos_only_close & ~self.already_reached
        self.pos_only_reaches += pos_only_new.sum().item()

        # Debug: Log when position is close but orientation fails
        pos_but_not_orient = pos_only_close & ~pos_close
        if pos_but_not_orient.any() and self.use_orient_check:
            fail_ids = torch.where(pos_but_not_orient)[0]
            for idx in fail_ids:
                if self.steps_since_spawn[idx] % 50 == 0:  # Don't spam
                    print(f"  [DEBUG] Env {idx.item()}: pos={dist[idx]:.3f}m OK, "
                          f"orient_err={orient_err_val[idx]:.2f} rad > thresh={self.orient_threshold:.2f} "
                          f"(step_since_spawn={self.steps_since_spawn[idx]})")

        # Condition 2: Displacement
        ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1)
        moved_enough = ee_displacement >= self.min_displacement

        # Condition 3: Within time limit
        within_time = self.steps_since_spawn <= self.max_reach_steps

        # Validated reach = all 3 conditions
        validated_reach = pos_close & moved_enough & within_time
        new_reaches = validated_reach & ~self.already_reached

        # Clear per-step reach info
        self.last_reach_dists = []
        self.last_reach_disps = []
        self.last_reach_count = 0

        if new_reaches.any():
            reached_ids = torch.where(new_reaches)[0]
            self.total_reaches += len(reached_ids)
            self.validated_reaches += len(reached_ids)
            self.last_reach_count = len(reached_ids)
            # Record stats BEFORE resample (pre-resample values = actual reach performance)
            for idx in reached_ids:
                d = dist[idx].item()
                disp = ee_displacement[idx].item()
                self.reach_distances.append(d)
                self.reach_displacements.append(disp)
                self.last_reach_dists.append(d)
                self.last_reach_disps.append(disp)
            self.already_reached[reached_ids] = True
            # Resample AFTER recording stats
            self._sample_commands(reached_ids)

        # Handle timeouts
        timed_out = (self.steps_since_spawn > self.max_reach_steps) & ~self.already_reached
        if timed_out.any():
            timed_out_ids = torch.where(timed_out)[0]
            self.timed_out_targets += len(timed_out_ids)
            self._sample_commands(timed_out_ids)

        # Stuck detection: if EE barely moved in last 15 steps, increment counter
        ee_movement = torch.norm(ee_body - self.ee_pos_history, dim=-1)
        barely_moving = ee_movement < 0.008  # Less than 8mm in 15 steps
        self.stuck_counter = torch.where(
            barely_moving & ~self.already_reached,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter)
        )
        # Update history every 15 steps
        update_history = (self.steps_since_spawn % 15 == 0)
        if update_history.any():
            self.ee_pos_history[update_history] = ee_body[update_history].clone()

        # If stuck for 2 cycles (30 steps barely moving) AND past 40% of max steps, resample
        stuck_threshold = 2
        past_threshold = self.steps_since_spawn > (self.max_reach_steps * 2 // 5)
        stuck = (self.stuck_counter >= stuck_threshold) & past_threshold & ~self.already_reached
        if stuck.any():
            stuck_ids = torch.where(stuck)[0]
            self.stuck_resample_count += len(stuck_ids)
            self._sample_commands(stuck_ids)
            self.stuck_counter[stuck_ids] = 0

        # Markers
        self._init_markers()
        default_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(self.num_envs, -1)
        self.target_markers.visualize(translations=target_world, orientations=default_quat)
        self.ee_markers.visualize(translations=ee_pos, orientations=default_quat)

        self.prev_ee_pos = ee_pos.clone()
        self._prev_leg_actions = self.prev_leg_actions.clone()
        self._prev_arm_actions = self.prev_arm_actions.clone()
        self.prev_leg_actions = leg_actions.clone()
        self.prev_arm_actions = arm_actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {"policy": self.get_loco_obs()}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
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
        self._sample_commands(env_ids)
        self.phase[env_ids] = torch.rand(n, device=self.device)
        self.prev_leg_actions[env_ids] = 0
        self.prev_arm_actions[env_ids] = 0
        self._prev_leg_actions[env_ids] = 0
        self._prev_arm_actions[env_ids] = 0
        self.prev_ee_pos[env_ids] = 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda:0"

    print(f"\n{'=' * 70}")
    if args.record:
        print("ULC G1 STAGE 7 - VIDEO RECORDING")
    else:
        print("ULC G1 STAGE 7 - ANTI-GAMING PLAY")
    print(f"{'=' * 70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {args.mode} | {MODE_CONFIGS[args.mode]['description']}")
    print(f"  Reach threshold: {args.reach_threshold}m")
    print(f"  Min displacement: {args.min_displacement}m")
    print(f"  Max reach steps: {args.max_reach_steps}")
    print(f"  Deterministic: {use_deterministic}")
    if args.record:
        print(f"  RECORDING: {args.record_duration}s @ {args.fps}fps")
        print(f"  Camera: {args.camera_angle}")
    print(f"{'=' * 70}\n")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print(f"[Checkpoint]")
    ckpt_level = 0
    for key in ["best_reward", "iteration", "curriculum_level", "total_reaches",
                "validated_reaches", "timed_out_targets"]:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")
            if key == "curriculum_level":
                ckpt_level = checkpoint[key]

    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing
    env = PlayEnv(cfg)

    # CRITICAL: Set curriculum level from checkpoint so thresholds match training
    env.set_curriculum_level(ckpt_level)

    net = DualActorCritic(loco_obs=57, arm_obs=55, loco_act=12, arm_act=5).to(device)

    state_dict = checkpoint.get("model", checkpoint)
    model_keys = set(net.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing or unexpected:
        if missing:
            print(f"\n  MISSING: {sorted(missing)[:5]}")
        if unexpected:
            print(f"\n  UNEXPECTED: {sorted(unexpected)[:5]}")
    else:
        print(f"\n  All {len(model_keys)} keys match!")

    net.load_state_dict(state_dict, strict=True)
    net.eval()

    arm_std = net.arm_actor.log_std.clamp(-2, 1).exp().mean().item()
    print(f"[Std] Arm: {arm_std:.4f}")

    # Camera setup
    from isaacsim.core.utils.viewports import set_camera_view
    eye, target_cam = CAMERA_PRESETS[args.camera_angle]
    set_camera_view(eye=eye, target=target_cam)
    print(f"[Camera] {args.camera_angle}: eye={eye}, target={target_cam}")

    # Video recorder setup
    recorder = None
    if args.record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = os.path.join(os.getcwd(), "recordings", f"stage7_{timestamp}")
        recorder = FrameRecorder(record_dir, fps=args.fps)
        # Override steps to match recording duration
        env_dt = 0.02  # 4 decimation * 1/200 physics dt
        total_steps = int(args.record_duration / env_dt)
        args.steps = total_steps
        record_interval = max(1, int(1.0 / (env_dt * args.fps)))
        print(f"[Record] {args.record_duration}s = {total_steps} steps, "
              f"capture every {record_interval} steps")

    obs, _ = env.reset()

    # Reset camera after env reset (Isaac Lab sometimes resets viewport)
    set_camera_view(eye=eye, target=target_cam)

    prev_reaches = 0

    print(f"\n[Play] {args.steps} steps | '{args.mode}' | {'DETERM' if use_deterministic else 'STOCH'}")
    print(f"  Training Level {ckpt_level}: pos_thresh={env.reach_pos_threshold}m, "
          f"min_disp={env.min_displacement}m, max_steps={env.max_reach_steps}")
    print(f"  obs_pos_threshold={env.obs_pos_threshold}m (for target_reached observation)\n")

    with torch.no_grad():
        for step in range(args.steps):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            leg_act, arm_act = net.get_actions(loco_obs, arm_obs, deterministic=use_deterministic)
            actions = torch.cat([leg_act, arm_act], dim=-1)

            obs, reward, terminated, truncated, info = env.step(actions)

            # Capture frame for video
            if recorder and (step % record_interval == 0):
                recorder.capture_frame()

            if env.validated_reaches > prev_reaches:
                new = env.validated_reaches - prev_reaches
                avg_dist = np.mean(env.last_reach_dists) if env.last_reach_dists else 0
                avg_disp = np.mean(env.last_reach_disps) if env.last_reach_disps else 0
                print(f"[Step {step:4d}] +{new} REACH (total={env.validated_reaches}) "
                      f"dist={avg_dist:.3f}m disp={avg_disp:.3f}m")
                prev_reaches = env.validated_reaches

            if step > 0 and step % 200 == 0:
                h = env.robot.data.root_pos_w[:, 2].mean().item()
                root_quat = env.robot.data.root_quat_w
                vx_b = quat_apply_inverse(root_quat, env.robot.data.root_lin_vel_w)[:, 0].mean().item()
                cmd_vx = env.vel_cmd[:, 0].mean().item()

                ee_pos, pq = env._compute_palm_ee()
                root_pos = env.robot.data.root_pos_w
                tw = root_pos + quat_apply(root_quat, env.target_pos_body)
                ee_d = torch.norm(ee_pos - tw, dim=-1).mean().item()
                ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
                ee_disp = torch.norm(ee_body - env.ee_pos_at_spawn, dim=-1).mean().item()
                ee_spd = torch.norm((ee_pos - env.prev_ee_pos) / 0.02, dim=-1).mean().item()

                orient_err_debug = compute_orientation_error(pq, env.target_orient_body).mean().item()

                total_attempts = env.validated_reaches + env.timed_out_targets + env.stuck_resample_count
                v_rate = env.validated_reaches / max(total_attempts, 1) * 100

                print(
                    f"[Step {step:4d}] "
                    f"H={h:.3f} Vx={vx_b:+.2f}(cmd={cmd_vx:.2f}) "
                    f"EE={ee_d:.3f}m Disp={ee_disp:.3f}m Spd={ee_spd:.3f} "
                    f"Orient={orient_err_debug:.2f}rad "
                    f"VR={env.validated_reaches} TO={env.timed_out_targets} "
                    f"Stuck={env.stuck_resample_count} Rate={v_rate:.0f}%"
                )

    # Finalize video
    if recorder:
        video_path = recorder.finalize_video(f"g1_stage7_reaching_{args.mode}.mp4")
        if video_path:
            print(f"\n[VIDEO] LinkedIn-ready video: {video_path}")

    total_attempts = env.validated_reaches + env.timed_out_targets
    print(f"\n{'=' * 70}")
    print("STAGE 7 PLAY COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Validated Reaches: {env.validated_reaches}")
    print(f"  Position-Only Reaches: {env.pos_only_reaches}")
    print(f"  Timed Out: {env.timed_out_targets}")
    print(f"  Stuck Resamples: {env.stuck_resample_count}")
    print(f"  Total Attempts: {total_attempts}")
    if total_attempts > 0:
        print(f"  Validated Rate: {env.validated_reaches / total_attempts:.1%}")
    print(f"  Rate: {env.validated_reaches / args.steps * 1000:.1f} validated per 1K steps")
    if env.reach_distances:
        print(f"  Avg Reach Distance: {np.mean(env.reach_distances):.3f}m")
        print(f"  Avg Reach Displacement: {np.mean(env.reach_displacements):.3f}m")
    print(f"{'=' * 70}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
