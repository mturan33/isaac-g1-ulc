"""
Unified Stage 2: Arm Reaching — Play/Evaluation (29DoF)
========================================================
Dual Actor-Critic: Loco (FROZEN, 66->15) + Arm (39->7).
Both policies deterministic during evaluation.

MODES:
- standing: vx=0, arm reaching only (default)
- walking:  vx=0.3, walk + reach
- fast:     vx=0.5, fast walk + reach
- single:   vx=0, single target (no resample after reach)

FLAGS:
- --no_orient: Disable orientation check for reach validation

MUST match training obs/act/clamp/termination exactly.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time

# ============================================================================
# IMPORTS & CONFIG
# ============================================================================

import importlib.util
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "config", "ulc_g1_29dof_cfg.py")
_spec = importlib.util.spec_from_file_location("ulc_g1_29dof_cfg", _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

G1_29DOF_USD = _cfg_mod.G1_29DOF_USD
LOCO_JOINT_NAMES = _cfg_mod.LOCO_JOINT_NAMES
LEG_JOINT_NAMES = _cfg_mod.LEG_JOINT_NAMES
WAIST_JOINT_NAMES = _cfg_mod.WAIST_JOINT_NAMES
ARM_JOINT_NAMES = _cfg_mod.ARM_JOINT_NAMES
ARM_JOINT_NAMES_RIGHT = _cfg_mod.ARM_JOINT_NAMES_RIGHT
ARM_JOINT_NAMES_LEFT = _cfg_mod.ARM_JOINT_NAMES_LEFT
HAND_JOINT_NAMES = _cfg_mod.HAND_JOINT_NAMES
DEFAULT_LOCO_LIST = _cfg_mod.DEFAULT_LOCO_LIST
DEFAULT_ARM_LIST = _cfg_mod.DEFAULT_ARM_LIST
DEFAULT_HAND_LIST = _cfg_mod.DEFAULT_HAND_LIST
DEFAULT_ALL_POSES = _cfg_mod.DEFAULT_ALL_POSES
DEFAULT_ARM_POSES = _cfg_mod.DEFAULT_ARM_POSES
NUM_LOCO_JOINTS = _cfg_mod.NUM_LOCO_JOINTS
NUM_ARM_JOINTS = _cfg_mod.NUM_ARM_JOINTS
NUM_HAND_JOINTS = _cfg_mod.NUM_HAND_JOINTS
LEG_ACTION_SCALE = _cfg_mod.LEG_ACTION_SCALE
WAIST_ACTION_SCALE = _cfg_mod.WAIST_ACTION_SCALE
ARM_ACTION_SCALE = 2.0  # Override cfg (0.5 too small — limits reach to ~0.25m, need 0.55m)
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS
SHOULDER_OFFSET_RIGHT = _cfg_mod.SHOULDER_OFFSET_RIGHT

# Dimensions — MUST match training
LOCO_OBS_DIM = 66
LOCO_ACT_DIM = 15
ARM_OBS_DIM = 39
ARM_ACT_DIM = 7

# Curriculum from training (for checkpoint level loading) — 10 levels, must match train
CURRICULUM = [
    # Phase 1: Standing + Reaching (L0-4) — workspace grows to 0.45m
    {"pos_threshold": 0.10, "min_displacement": 0.03, "max_reach_steps": 200, "use_orientation": False, "workspace_radius": (0.10, 0.20), "orient_threshold": 99.0},
    {"pos_threshold": 0.08, "min_displacement": 0.04, "max_reach_steps": 190, "use_orientation": False, "workspace_radius": (0.12, 0.25), "orient_threshold": 99.0},
    {"pos_threshold": 0.07, "min_displacement": 0.05, "max_reach_steps": 180, "use_orientation": False, "workspace_radius": (0.15, 0.32), "orient_threshold": 99.0},
    {"pos_threshold": 0.07, "min_displacement": 0.05, "max_reach_steps": 180, "use_orientation": False, "workspace_radius": (0.15, 0.38), "orient_threshold": 99.0},
    {"pos_threshold": 0.06, "min_displacement": 0.06, "max_reach_steps": 175, "use_orientation": False, "workspace_radius": (0.18, 0.45), "orient_threshold": 99.0},
    # Phase 2: Walking + Reaching (L5-7) — workspace grows to 0.55m
    {"pos_threshold": 0.06, "min_displacement": 0.06, "max_reach_steps": 175, "use_orientation": False, "workspace_radius": (0.18, 0.45), "orient_threshold": 99.0},
    {"pos_threshold": 0.05, "min_displacement": 0.06, "max_reach_steps": 165, "use_orientation": False, "workspace_radius": (0.18, 0.50), "orient_threshold": 99.0},
    {"pos_threshold": 0.05, "min_displacement": 0.07, "max_reach_steps": 160, "use_orientation": False, "workspace_radius": (0.18, 0.55), "orient_threshold": 99.0},
    # Phase 3: Walking + Orientation (L8-9) — 0.55m maintained
    {"pos_threshold": 0.05, "min_displacement": 0.07, "max_reach_steps": 160, "use_orientation": True, "workspace_radius": (0.18, 0.55), "orient_threshold": 2.0},
    {"pos_threshold": 0.04, "min_displacement": 0.08, "max_reach_steps": 160, "use_orientation": True, "workspace_radius": (0.18, 0.55), "orient_threshold": 1.5},
]

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Arm Reaching — Play/Eval (29DoF)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="standing",
                        choices=["standing", "walking", "fast", "single"])
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--no_orient", action="store_true",
                        help="Disable orientation check for reaches")
    return parser.parse_args()

args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# ============================================================================
# QUATERNION UTILS (same as train)
# ============================================================================

def quat_to_euler_xyz(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
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
    fwd_x = 1 - 2 * (y * y + z * z)
    fwd_y = 2 * (x * y + w * z)
    fwd_z = 2 * (x * z - w * y)
    return torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)


def compute_orientation_error(palm_quat, target_dir):
    forward = get_palm_forward(palm_quat)
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# NETWORKS (same architecture as train)
# ============================================================================

class LocoActorCritic(nn.Module):
    def __init__(self, num_obs=LOCO_OBS_DIM, num_act=LOCO_ACT_DIM, hidden=[512, 256, 128]):
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

    def forward_actor(self, x):
        return self.actor(x)


class ArmActor(nn.Module):
    def __init__(self, num_obs=ARM_OBS_DIM, num_act=ARM_ACT_DIM, hidden=[256, 256, 128]):
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
    def __init__(self, num_obs=ARM_OBS_DIM, hidden=[256, 256, 128]):
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
    def __init__(self):
        super().__init__()
        self.loco_ac = LocoActorCritic()
        self.arm_actor = ArmActor()
        self.arm_critic = ArmCritic()


# ============================================================================
# ENVIRONMENT (same scene as train)
# ============================================================================

def create_env(num_envs, device):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
        contact_forces = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3, track_air_time=False,
        )
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_29DOF_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=True,
                    linear_damping=0.0, angular_damping=0.0,
                    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, HEIGHT_DEFAULT),
                joint_pos=DEFAULT_ALL_POSES,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.90,
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint",
                                      ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*"],
                    effort_limit_sim=ACTUATOR_PARAMS["legs"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["legs"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["legs"]["stiffness"],
                    damping=ACTUATOR_PARAMS["legs"]["damping"],
                    armature=ACTUATOR_PARAMS["legs"]["armature"],
                ),
                "feet": ImplicitActuatorCfg(
                    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["feet"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["feet"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["feet"]["stiffness"],
                    damping=ACTUATOR_PARAMS["feet"]["damping"],
                    armature=ACTUATOR_PARAMS["feet"]["armature"],
                ),
                "shoulders": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["shoulders"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["shoulders"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["shoulders"]["stiffness"],
                    damping=ACTUATOR_PARAMS["shoulders"]["damping"],
                    armature=ACTUATOR_PARAMS["shoulders"]["armature"],
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["arms"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["arms"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["arms"]["stiffness"],
                    damping=ACTUATOR_PARAMS["arms"]["damping"],
                    armature=ACTUATOR_PARAMS["arms"]["armature"],
                ),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=[".*_wrist_.*"],
                    effort_limit_sim=ACTUATOR_PARAMS["wrist"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["wrist"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["wrist"]["stiffness"],
                    damping=ACTUATOR_PARAMS["wrist"]["damping"],
                    armature=ACTUATOR_PARAMS["wrist"]["armature"],
                ),
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                                      ".*_hand_thumb_.*_joint"],
                    effort_limit=ACTUATOR_PARAMS["hands"]["effort_limit"],
                    velocity_limit=ACTUATOR_PARAMS["hands"]["velocity_limit"],
                    stiffness={".*": ACTUATOR_PARAMS["hands"]["stiffness"]},
                    damping={".*": ACTUATOR_PARAMS["hands"]["damping"]},
                    armature={".*": ACTUATOR_PARAMS["hands"]["armature"]},
                ),
            },
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 60.0  # Long episodes for play
        action_space = LOCO_ACT_DIM + ARM_ACT_DIM
        observation_space = LOCO_OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=5.0)

    class PlayEnv(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            jn = self.robot.joint_names

            # Joint indices (same as train)
            self.loco_idx = torch.tensor(
                [jn.index(n) for n in LOCO_JOINT_NAMES if n in jn], device=self.device)
            self.right_arm_idx = torch.tensor(
                [jn.index(n) for n in ARM_JOINT_NAMES_RIGHT if n in jn], device=self.device)
            self.left_arm_idx = torch.tensor(
                [jn.index(n) for n in ARM_JOINT_NAMES_LEFT if n in jn], device=self.device)
            self.hand_idx = torch.tensor(
                [jn.index(n) for n in HAND_JOINT_NAMES if n in jn], device=self.device)

            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")
            hip_yaw_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
            self.hip_yaw_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_yaw_names], device=self.device)

            # Palm body
            body_names = self.robot.data.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "palm" in name.lower():
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                self.palm_idx = len(body_names) - 1

            print(f"[Play] Loco: {len(self.loco_idx)}, R-Arm: {len(self.right_arm_idx)}, "
                  f"L-Arm: {len(self.left_arm_idx)}, Hands: {len(self.hand_idx)}")
            print(f"[Play] Palm body idx: {self.palm_idx}")

            # Default poses
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)
            self.default_right_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_RIGHT],
                device=self.device, dtype=torch.float32)
            self.default_left_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_LEFT],
                device=self.device, dtype=torch.float32)

            # Action scales
            leg_s = [LEG_ACTION_SCALE] * 12
            waist_s = [WAIST_ACTION_SCALE] * 3
            self.loco_action_scales = torch.tensor(leg_s + waist_s, device=self.device, dtype=torch.float32)
            self.shoulder_offset = torch.tensor(SHOULDER_OFFSET_RIGHT, device=self.device, dtype=torch.float32)

            # State
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_loco_act = torch.zeros(self.num_envs, LOCO_ACT_DIM, device=self.device)
            self.prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)

            # Target
            self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_orient = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_orient[:, 2] = -1.0
            self.prev_ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

            # Anti-gaming state
            self.ee_pos_at_spawn = torch.zeros(self.num_envs, 3, device=self.device)
            self.steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # Markers
            self._markers_init = False

            # Default curriculum (overridden in main() after checkpoint load)
            self.play_curriculum = CURRICULUM[0]

            # Set velocity based on mode
            self._setup_mode()
            self._sample_targets(torch.arange(self.num_envs, device=self.device))

        @property
        def robot(self):
            return self.scene["robot"]

        def _setup_mode(self):
            mode = args_cli.mode
            if mode == "standing":
                self.vel_cmd[:] = 0.0
            elif mode == "walking":
                self.vel_cmd[:, 0] = 0.3
            elif mode == "fast":
                self.vel_cmd[:, 0] = 0.5
            elif mode == "single":
                self.vel_cmd[:] = 0.0
            print(f"[Play] Mode: {mode}, vel_cmd: vx={self.vel_cmd[0, 0]:.1f}")

        def _compute_ee(self):
            palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
            palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
            fwd = get_palm_forward(palm_quat)
            return palm_pos + 0.02 * fwd, palm_quat

        def _sample_targets(self, env_ids):
            n = len(env_ids)
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]
            ee_w, _ = self._compute_ee()
            ee_w = ee_w[env_ids]
            current_ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            # Use workspace from curriculum level
            ws_r = self.play_curriculum.get("workspace_radius", (0.15, 0.35))
            azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
            elevation = torch.empty(n, device=self.device).uniform_(-0.3, 0.5)
            radius = torch.empty(n, device=self.device).uniform_(*ws_r)

            target_x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
            target_y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
            target_z = radius * torch.sin(elevation) + self.shoulder_offset[2]

            target_x = target_x.clamp(-0.10, 0.55)   # match train
            target_y = target_y.clamp(-0.60, -0.05)  # match train
            target_z = target_z.clamp(-0.15, 0.55)   # match train
            target_body = torch.stack([target_x, target_y, target_z], dim=-1)

            # Min distance enforcement
            min_dist = self.play_curriculum.get("min_displacement", 0.05)
            dist_to_ee = torch.norm(target_body - current_ee_body, dim=-1)
            too_close = dist_to_ee < 0.10
            if too_close.any():
                direction = target_body - current_ee_body
                direction = direction / (direction.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                pushed = current_ee_body + 0.10 * direction
                pushed[:, 0] = pushed[:, 0].clamp(0.0, 0.45)
                pushed[:, 1] = pushed[:, 1].clamp(-0.50, -0.05)
                pushed[:, 2] = pushed[:, 2].clamp(-0.10, 0.50)
                target_body = torch.where(too_close.unsqueeze(-1).expand_as(target_body),
                                          pushed, target_body)

            self.target_pos_body[env_ids] = target_body
            self.target_orient[env_ids, 0] = 0.0
            self.target_orient[env_ids, 1] = 0.0
            self.target_orient[env_ids, 2] = -1.0
            self.ee_pos_at_spawn[env_ids] = current_ee_body
            self.steps_since_spawn[env_ids] = 0

        def _init_markers(self):
            if self._markers_init:
                return
            self.target_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TargetMarkers",
                    markers={"sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)))},
                ))
            self.ee_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/EEMarkers",
                    markers={"sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
                ))
            self._markers_init = True

        def get_loco_obs(self):
            r = self.robot
            q = r.data.root_quat_w
            lv_b = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av_b = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))
            jp_leg = r.data.joint_pos[:, self.loco_idx[:12]]
            jv_leg = r.data.joint_vel[:, self.loco_idx[:12]] * 0.1
            jp_waist = r.data.joint_pos[:, self.loco_idx[12:15]]
            jv_waist = r.data.joint_vel[:, self.loco_idx[12:15]] * 0.1
            gait = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)], -1)
            torso_euler = quat_to_euler_xyz(q)
            obs = torch.cat([
                lv_b, av_b, g, jp_leg, jv_leg, jp_waist, jv_waist,
                self.height_cmd[:, None], self.vel_cmd, gait,
                self.prev_loco_act, torso_euler, self.torso_cmd,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num()

        def get_arm_obs(self):
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w
            arm_pos = r.data.joint_pos[:, self.right_arm_idx]
            arm_vel = r.data.joint_vel[:, self.right_arm_idx] * 0.1
            ee_w, palm_quat = self._compute_ee()
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)
            pos_error = self.target_pos_body - ee_body
            orient_err = compute_orientation_error(palm_quat, self.target_orient)
            orient_err_norm = orient_err.unsqueeze(-1) / np.pi
            max_steps = float(self.play_curriculum.get("max_reach_steps", 200))
            steps_norm = (self.steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)
            obs = torch.cat([
                arm_pos, arm_vel, ee_body, palm_quat,
                self.target_pos_body, self.target_orient, pos_error,
                orient_err_norm, self.prev_arm_act, steps_norm,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num()

        def _pre_physics_step(self, combined_act):
            self.actions = combined_act.clone()
            loco_act = combined_act[:, :LOCO_ACT_DIM]
            arm_act = combined_act[:, LOCO_ACT_DIM:]

            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.loco_idx] = self.default_loco + loco_act * self.loco_action_scales
            tgt[:, self.loco_idx[12]].clamp_(-0.15, 0.15)  # waist yaw
            tgt[:, self.loco_idx[13]].clamp_(-0.15, 0.15)  # waist roll
            tgt[:, self.loco_idx[14]].clamp_(-0.2, 0.2)    # waist pitch
            for hy_idx in self.hip_yaw_loco_idx:
                tgt[:, self.loco_idx[hy_idx]].clamp_(-0.3, 0.3)
            tgt[:, self.right_arm_idx] = self.default_right_arm + arm_act * ARM_ACTION_SCALE
            tgt[:, self.left_arm_idx] = self.default_left_arm
            tgt[:, self.hand_idx] = self.default_hand
            self.robot.set_joint_position_target(tgt)

            self.prev_loco_act = loco_act.clone()
            self.prev_arm_act = arm_act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
            self.steps_since_spawn += 1

            # Update markers
            ee_w, _ = self._compute_ee()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            target_w = root_pos + quat_apply(root_quat, self.target_pos_body)
            self._init_markers()
            dq = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
            self.target_markers.visualize(translations=target_w, orientations=dq)
            self.ee_markers.visualize(translations=ee_w, orientations=dq)
            self.prev_ee_pos_w = ee_w.clone()

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {"policy": self.get_loco_obs()}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_g = quat_apply_inverse(q, gvec)
            fallen = (pos[:, 2] < 0.55) | (pos[:, 2] > 1.2)
            bad_orient = proj_g[:, :2].abs().max(dim=-1)[0] > 0.7
            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_bad = (lk < -0.05) | (rk < -0.05) | (lk > 1.5) | (rk > 1.5)
            waist_bad = (jp[:, 14].abs() > 0.35) | (jp[:, 13].abs() > 0.25)
            hip_yaw_L = jp[:, self.hip_yaw_loco_idx[0]]
            hip_yaw_R = jp[:, self.hip_yaw_loco_idx[1]]
            hip_yaw_bad = (hip_yaw_L.abs() > 0.6) | (hip_yaw_R.abs() > 0.6)
            terminated = fallen | bad_orient | knee_bad | waist_bad | hip_yaw_bad
            truncated = self.episode_length_buf >= self.max_episode_length
            return terminated, truncated

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0:
                return
            n = len(env_ids)
            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            self.robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)
            root_pos = torch.tensor([[0.0, 0.0, HEIGHT_DEFAULT]], device=self.device).expand(n, -1).clone()
            root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(n, -1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
            self.prev_loco_act[env_ids] = 0
            self.prev_arm_act[env_ids] = 0
            self.phase[env_ids] = 0
            self.prev_ee_pos_w[env_ids] = 0
            self._setup_mode()
            self._sample_targets(env_ids)

    env = PlayEnv(EnvCfg())
    return env


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    # Load checkpoint
    print(f"\n[Load] {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    net = DualActorCritic().to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    ckpt_level = ckpt.get("curriculum_level", 0)
    ckpt_level = min(ckpt_level, len(CURRICULUM) - 1)
    print(f"  Iter={ckpt.get('iteration', '?')}, R={ckpt.get('best_reward', '?')}, Level={ckpt_level}")
    print(f"  VR={ckpt.get('validated_reaches', '?')}, TO={ckpt.get('timed_out_targets', '?')}")

    # Set play curriculum from checkpoint level
    env.play_curriculum = CURRICULUM[ckpt_level]
    pos_thresh = env.play_curriculum["pos_threshold"]
    use_orient = env.play_curriculum["use_orientation"] and not args_cli.no_orient
    orient_thresh = env.play_curriculum.get("orient_threshold", 99.0)
    min_disp = env.play_curriculum["min_displacement"]
    max_steps = env.play_curriculum["max_reach_steps"]

    print(f"  Level {ckpt_level}: pos_thresh={pos_thresh}, orient={use_orient}"
          f"(thresh={orient_thresh:.1f}), min_disp={min_disp}, max_steps={max_steps}")

    # Resample targets with correct curriculum
    env._sample_targets(torch.arange(env.num_envs, device=device))

    obs, _ = env.reset()

    # Metrics
    total_steps = 0
    falls = 0
    validated_reaches = 0
    timed_out = 0
    total_attempts = 0
    reach_distances = []
    reach_displacements = []
    already_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    steps_since_spawn = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    ee_pos_at_spawn = torch.zeros(env.num_envs, 3, device=device)

    # Initialize spawn EE
    ee_w, _ = env._compute_ee()
    root_pos = env.robot.data.root_pos_w
    root_quat = env.robot.data.root_quat_w
    ee_body_init = quat_apply_inverse(root_quat, ee_w - root_pos)
    ee_pos_at_spawn = ee_body_init.clone()

    mode_str = args_cli.mode.upper()
    orient_str = "NO_ORIENT" if args_cli.no_orient else "ORIENT_CHECK"
    print(f"\n{'='*80}")
    print(f"STAGE 2 PLAY: {mode_str} | Level {ckpt_level} | {orient_str}")
    print(f"  Envs: {env.num_envs}, Max steps: {args_cli.max_steps}")
    print(f"{'='*80}\n")

    t_start = time.time()

    for step in range(args_cli.max_steps):
        with torch.no_grad():
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()
            loco_action = net.loco_ac.forward_actor(loco_obs)
            arm_action = net.arm_actor(arm_obs)

        combined = torch.cat([loco_action, arm_action], dim=-1)
        obs_dict, reward, terminated, truncated, _ = env.step(combined)
        total_steps += env.num_envs
        steps_since_spawn += 1

        # Check reaches
        ee_w, palm_quat = env._compute_ee()
        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w
        ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)
        target_w = root_pos + quat_apply(root_quat, env.target_pos_body)
        dist = torch.norm(ee_w - target_w, dim=-1)

        # 3-condition validation
        pos_ok = dist < pos_thresh
        if use_orient:
            orient_err = compute_orientation_error(palm_quat, env.target_orient)
            pos_ok = pos_ok & (orient_err < orient_thresh)

        ee_disp = torch.norm(ee_body - ee_pos_at_spawn, dim=-1)
        moved_ok = ee_disp >= min_disp
        time_ok = steps_since_spawn <= max_steps

        validated = pos_ok & moved_ok & time_ok & ~already_reached

        if validated.any():
            v_ids = torch.where(validated)[0]
            for idx in v_ids:
                # Record pre-resample values
                reach_distances.append(dist[idx].item())
                reach_displacements.append(ee_disp[idx].item())
            validated_reaches += len(v_ids)
            total_attempts += len(v_ids)
            already_reached[v_ids] = True

            if args_cli.mode != "single":
                env._sample_targets(v_ids)
                ee_w2, _ = env._compute_ee()
                ee_body2 = quat_apply_inverse(root_quat[v_ids], ee_w2[v_ids] - root_pos[v_ids])
                ee_pos_at_spawn[v_ids] = ee_body2
                steps_since_spawn[v_ids] = 0
                already_reached[v_ids] = False

        # Handle timeouts
        timed_mask = (steps_since_spawn > max_steps) & ~already_reached
        if timed_mask.any():
            to_ids = torch.where(timed_mask)[0]
            timed_out += len(to_ids)
            total_attempts += len(to_ids)
            if args_cli.mode != "single":
                env._sample_targets(to_ids)
                ee_w3, _ = env._compute_ee()
                ee_body3 = quat_apply_inverse(root_quat[to_ids], ee_w3[to_ids] - root_pos[to_ids])
                ee_pos_at_spawn[to_ids] = ee_body3
                steps_since_spawn[to_ids] = 0
                already_reached[to_ids] = False

        # Count falls
        done_mask = terminated.bool()
        if done_mask.any():
            falls += done_mask.sum().item()
            done_ids = torch.where(done_mask)[0]
            steps_since_spawn[done_ids] = 0
            already_reached[done_ids] = False
            ee_w4, _ = env._compute_ee()
            ee_body4 = quat_apply_inverse(
                env.robot.data.root_quat_w[done_ids],
                ee_w4[done_ids] - env.robot.data.root_pos_w[done_ids])
            ee_pos_at_spawn[done_ids] = ee_body4

        # Periodic logging
        if (step + 1) % 200 == 0:
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            q = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=device).expand(env.num_envs, -1)
            proj_g = quat_apply_inverse(q, gvec)
            tilt = torch.rad2deg(torch.asin(torch.clamp((proj_g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))).mean().item()
            avg_dist = dist.mean().item()
            rate = validated_reaches / max(total_attempts, 1)

            print(f"  [{step+1:5d}/{args_cli.max_steps}] "
                  f"VR={validated_reaches} TO={timed_out} Rate={rate:.1%} "
                  f"EE_dist={avg_dist:.3f} H={height:.3f} Tilt={tilt:.1f} Falls={falls}")

    elapsed = time.time() - t_start

    # Final summary
    rate = validated_reaches / max(total_attempts, 1)
    avg_reach_dist = np.mean(reach_distances) if reach_distances else 0
    avg_reach_disp = np.mean(reach_displacements) if reach_displacements else 0
    height_final = env.robot.data.root_pos_w[:, 2].mean().item()

    print(f"\n{'='*80}")
    print(f"STAGE 2 PLAY RESULTS — {mode_str}")
    print(f"{'='*80}")
    print(f"  Steps: {args_cli.max_steps}, Envs: {env.num_envs}, Time: {elapsed:.1f}s")
    print(f"  Validated Reaches: {validated_reaches}")
    print(f"  Timed Out: {timed_out}")
    print(f"  Total Attempts: {total_attempts}")
    print(f"  Validated Rate: {rate:.1%}")
    print(f"  Avg Reach Distance: {avg_reach_dist:.4f}m")
    print(f"  Avg Reach Displacement: {avg_reach_disp:.4f}m")
    print(f"  Falls: {falls}")
    print(f"  Final Height: {height_final:.3f}m")
    if total_steps > 0:
        print(f"  Reach Rate: {validated_reaches / (total_steps / 1000):.1f} per 1K steps")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Level: {ckpt_level}, Orient: {use_orient}")
    print(f"{'='*80}")

    simulation_app.close()


if __name__ == "__main__":
    main()
