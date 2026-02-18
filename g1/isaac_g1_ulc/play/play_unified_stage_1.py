"""
Unified Stage 1: Standing + Omnidirectional Locomotion — Play/Evaluation
========================================================================
Decoupled Heading Control: (speed, heading_offset, body_yaw_cmd)
188-dim unified obs, 15-dim loco action (same as training).

MODES:
- standing: speed=0 (standing test)
- walk: speed=0.4, heading=0 (forward walk)
- turn: speed=0.4 + yaw_rate sweep (turning test)
- lateral: speed=0.4, heading_offset=pi/2 (sideways walk)
- omni: speed=0.5, full heading + yaw (omnidirectional demo)
- mixed: Scheduled commands (full demo sequence)
- push: walk + random push forces

2026-02-12: Initial implementation.
2026-02-12: Expanded obs 143->188 to match training.
2026-02-15: V2 sync — self-collision, quaternion fix, hip_yaw clamp (match training V2).
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
HAND_JOINT_NAMES = _cfg_mod.HAND_JOINT_NAMES
DEFAULT_LOCO_LIST = _cfg_mod.DEFAULT_LOCO_LIST
DEFAULT_ARM_LIST = _cfg_mod.DEFAULT_ARM_LIST
DEFAULT_HAND_LIST = _cfg_mod.DEFAULT_HAND_LIST
DEFAULT_ALL_POSES = _cfg_mod.DEFAULT_ALL_POSES
NUM_LOCO_JOINTS = _cfg_mod.NUM_LOCO_JOINTS
NUM_ARM_JOINTS = _cfg_mod.NUM_ARM_JOINTS
NUM_HAND_JOINTS = _cfg_mod.NUM_HAND_JOINTS
LEG_ACTION_SCALE = _cfg_mod.LEG_ACTION_SCALE
WAIST_ACTION_SCALE = _cfg_mod.WAIST_ACTION_SCALE
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS

# Must match training — NEVER change independently
LOCO_OBS_DIM = 73
ARM_OBS_DIM = 70
HAND_OBS_DIM = 45
OBS_DIM = LOCO_OBS_DIM + ARM_OBS_DIM + HAND_OBS_DIM  # 188
LOCO_ACT_DIM = NUM_LOCO_JOINTS  # 15

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Stage 1: Play/Eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="walk",
                        choices=["standing", "walk", "turn", "lateral", "omni", "mixed", "push"])
    parser.add_argument("--max_steps", type=int, default=3000)
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
from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ============================================================================
# MODE CONFIGS — (speed, heading_offset, body_yaw_cmd, yaw_mode, push_force, push_interval)
# ============================================================================

MODE_CONFIGS = {
    "standing": {"speed": 0.0, "heading": 0.0, "yaw_cmd": 0.0, "yaw_mode": 0.0,
                 "push_force": (0, 0), "push_interval": (9999, 9999)},
    "walk":     {"speed": 0.4, "heading": 0.0, "yaw_cmd": 0.0, "yaw_mode": 0.0,
                 "push_force": (0, 0), "push_interval": (9999, 9999)},
    "turn":     {"speed": 0.4, "heading": 0.0, "yaw_cmd": 0.5, "yaw_mode": 0.0,
                 "push_force": (0, 0), "push_interval": (9999, 9999)},
    "lateral":  {"speed": 0.4, "heading": 1.57, "yaw_cmd": 0.0, "yaw_mode": 0.0,
                 "push_force": (0, 0), "push_interval": (9999, 9999)},
    "omni":     {"speed": 0.5, "heading": 0.0, "yaw_cmd": 0.0, "yaw_mode": 0.0,
                 "push_force": (0, 0), "push_interval": (9999, 9999)},
    "mixed":    {"speed": 0.0, "heading": 0.0, "yaw_cmd": 0.0, "yaw_mode": 0.0,
                 "push_force": (0, 0), "push_interval": (9999, 9999)},
    "push":     {"speed": 0.4, "heading": 0.0, "yaw_cmd": 0.0, "yaw_mode": 0.0,
                 "push_force": (0, 30), "push_interval": (100, 300)},
}

mode_cfg = MODE_CONFIGS[args_cli.mode]

# Mixed mode schedule: (step, speed, heading_offset, yaw_cmd, yaw_mode)
MIXED_SCHEDULE = [
    (0,    0.0, 0.0,  0.0,  0.0),   # Stand
    (300,  0.3, 0.0,  0.0,  0.0),   # Slow forward
    (600,  0.5, 0.0,  0.0,  0.0),   # Medium forward
    (900,  0.5, 0.0,  0.5,  0.0),   # Forward + turn right (rate)
    (1200, 0.5, 0.0, -0.5,  0.0),   # Forward + turn left (rate)
    (1500, 0.4, 1.57, 0.0,  0.0),   # Lateral right (heading=+pi/2)
    (1800, 0.4,-1.57, 0.0,  0.0),   # Lateral left (heading=-pi/2)
    (2100, 0.3, 2.36, 0.0,  0.0),   # Backward-right diagonal (heading=+3pi/4)
    (2400, 0.5, 0.0,  0.0,  1.0),   # Forward + abs yaw=0 (face spawn direction)
    (2700, 0.0, 0.0,  0.0,  0.0),   # Stop
]

print("=" * 80)
print(f"UNIFIED STAGE 1: PLAY — Mode: {args_cli.mode}")
print(f"  Decoupled Heading: speed={mode_cfg['speed']}, heading={mode_cfg['heading']:.2f}, "
      f"yaw_cmd={mode_cfg['yaw_cmd']:.2f}, yaw_mode={mode_cfg['yaw_mode']}")
print(f"  Push: force={mode_cfg['push_force']}, interval={mode_cfg['push_interval']}")
print(f"  Obs: {OBS_DIM}, Act: {LOCO_ACT_DIM}")
print(f"  Checkpoint: {args_cli.checkpoint}")
print("=" * 80)

# ============================================================================
# NETWORK (must match training)
# ============================================================================

class LocoActorCritic(nn.Module):
    def __init__(self, num_obs=OBS_DIM, num_act=LOCO_ACT_DIM, hidden=[512, 256, 128]):
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

    def act(self, x, det=False):
        mean = self.actor(x)
        if det:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


# ============================================================================
# UTILS
# ============================================================================

def quat_to_euler_xyz(quat):
    """Convert wxyz quaternion (Isaac Lab convention) to roll, pitch, yaw.
    V2 fix: correctly treats col 0 as w (was treating as x).
    """
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


def wrap_to_pi(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# ENVIRONMENT (play — no curriculum, manual commands)
# ============================================================================

def create_env(num_envs, device):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_29DOF_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, retain_accelerations=True,
                    linear_damping=0.0, angular_damping=0.0,
                    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,   # V2: match training
                    solver_position_iteration_count=8,   # V2: match training
                    solver_velocity_iteration_count=2)),  # V2: match training
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.80),
                joint_pos=DEFAULT_ALL_POSES,
                joint_vel={".*": 0.0}),
            soft_joint_pos_limit_factor=0.90,
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint",
                                      ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*"],
                    effort_limit_sim=ACTUATOR_PARAMS["legs"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["legs"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["legs"]["stiffness"],
                    damping=ACTUATOR_PARAMS["legs"]["damping"],
                    armature=ACTUATOR_PARAMS["legs"]["armature"]),
                "feet": ImplicitActuatorCfg(
                    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["feet"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["feet"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["feet"]["stiffness"],
                    damping=ACTUATOR_PARAMS["feet"]["damping"],
                    armature=ACTUATOR_PARAMS["feet"]["armature"]),
                "shoulders": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["shoulders"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["shoulders"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["shoulders"]["stiffness"],
                    damping=ACTUATOR_PARAMS["shoulders"]["damping"],
                    armature=ACTUATOR_PARAMS["shoulders"]["armature"]),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                    effort_limit_sim=ACTUATOR_PARAMS["arms"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["arms"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["arms"]["stiffness"],
                    damping=ACTUATOR_PARAMS["arms"]["damping"],
                    armature=ACTUATOR_PARAMS["arms"]["armature"]),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=[".*_wrist_.*"],
                    effort_limit_sim=ACTUATOR_PARAMS["wrist"]["effort_limit"],
                    velocity_limit_sim=ACTUATOR_PARAMS["wrist"]["velocity_limit"],
                    stiffness=ACTUATOR_PARAMS["wrist"]["stiffness"],
                    damping=ACTUATOR_PARAMS["wrist"]["damping"],
                    armature=ACTUATOR_PARAMS["wrist"]["armature"]),
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
                                      ".*_hand_thumb_.*_joint"],
                    effort_limit=ACTUATOR_PARAMS["hands"]["effort_limit"],
                    velocity_limit=ACTUATOR_PARAMS["hands"]["velocity_limit"],
                    stiffness={".*": ACTUATOR_PARAMS["hands"]["stiffness"]},
                    damping={".*": ACTUATOR_PARAMS["hands"]["damping"]},
                    armature={".*": ACTUATOR_PARAMS["hands"]["armature"]}),
            })
        # Foot contact sensors — one per foot (MUST match training)
        left_foot_contact = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/left_ankle_roll_link",
            update_period=0.02,
            history_length=1,
            track_air_time=False,
            force_threshold=5.0,
        )
        right_foot_contact = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ankle_roll_link",
            update_period=0.02,
            history_length=1,
            track_air_time=False,
            force_threshold=5.0,
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 60.0
        action_space = LOCO_ACT_DIM
        observation_space = OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=5.0)

    class Env(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)

            # Lighting
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath("/World/Light").IsValid():
                light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
                light_cfg.func("/World/Light", light_cfg)
            if not stage.GetPrimAtPath("/World/DistantLight").IsValid():
                dist_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), angle=0.53)
                dist_light.func("/World/DistantLight", dist_light, translation=(0, 0, 10))

            jn = self.robot.joint_names

            # Joint indices
            self.loco_idx = torch.tensor([jn.index(n) for n in LOCO_JOINT_NAMES if n in jn], device=self.device)
            self.arm_idx = torch.tensor([jn.index(n) for n in ARM_JOINT_NAMES if n in jn], device=self.device)
            self.hand_idx = torch.tensor([jn.index(n) for n in HAND_JOINT_NAMES if n in jn], device=self.device)

            # Per-joint indices
            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")
            ankle_roll_names = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
            self.ankle_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_roll_names], device=self.device)
            hip_roll_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
            self.hip_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_roll_names], device=self.device)

            # Defaults
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)

            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # Commands — decoupled heading
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.speed_cmd = torch.full((self.num_envs,), mode_cfg["speed"], device=self.device)
            self.heading_offset = torch.full((self.num_envs,), mode_cfg["heading"], device=self.device)
            self.body_yaw_cmd = torch.full((self.num_envs,), mode_cfg["yaw_cmd"], device=self.device)
            self.yaw_mode = torch.full((self.num_envs,), mode_cfg["yaw_mode"], device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

            # State
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, LOCO_ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, LOCO_ACT_DIM, device=self.device)

            # Push
            pi_lo, pi_hi = mode_cfg["push_interval"]
            if pi_hi <= pi_lo:
                pi_hi = pi_lo + 1
            self.push_timer = torch.randint(pi_lo, pi_hi, (self.num_envs,), device=self.device)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.total_pushes = 0
            self.total_falls = 0

            # Foot contact sensor refs (one per foot — match training)
            self._left_foot_sensor = self.scene["left_foot_contact"]
            self._right_foot_sensor = self.scene["right_foot_contact"]

            # Tracking
            self.speed_history = []
            self.speed_cmd_history = []
            self.heading_history = []
            self.heading_cmd_history = []
            self.yaw_history = []
            self.yaw_cmd_history = []
            self.distance_traveled = torch.zeros(self.num_envs, device=self.device)
            self.prev_pos = None

            # Direction arrow
            arrow_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/velArrow",
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.3, 0.15, 0.15),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.9, 0.1, 0.1)),
                    ),
                },
            )
            self.vel_arrow = VisualizationMarkers(arrow_cfg)
            self._arrow_visible = False

            print(f"[Play] {self.num_envs} envs, mode={args_cli.mode}")

        @property
        def robot(self):
            return self.scene["robot"]

        def _apply_push(self):
            self.step_count += 1
            push_mask = self.step_count >= self.push_timer

            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                force = torch.zeros(n, 3, device=self.device)
                force[:, :2] = torch.randn(n, 2, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                fmin, fmax = mode_cfg["push_force"]
                mag = torch.rand(n, 1, device=self.device) * (fmax - fmin) + fmin
                force = force * mag
                forces[ids, 0] = force

                pi_lo, pi_hi = mode_cfg["push_interval"]
                if pi_hi <= pi_lo:
                    pi_hi = pi_lo + 1
                self.push_timer[ids] = torch.randint(pi_lo, pi_hi, (n,), device=self.device)
                self.step_count[ids] = 0
                self.total_pushes += n

            self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        def _pre_physics_step(self, act):
            self.actions = act.clone()
            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.loco_idx] = self.default_loco + act * self.action_scales

            # V2: Clamp hip_yaw — match training
            left_hip_yaw_idx = self.loco_idx[LOCO_JOINT_NAMES.index("left_hip_yaw_joint")]
            right_hip_yaw_idx = self.loco_idx[LOCO_JOINT_NAMES.index("right_hip_yaw_joint")]
            tgt[:, left_hip_yaw_idx].clamp_(-0.25, 0.25)
            tgt[:, right_hip_yaw_idx].clamp_(-0.25, 0.25)

            # Waist clamp
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)

            tgt[:, self.arm_idx] = self.default_arm
            tgt[:, self.hand_idx] = self.default_hand
            self.robot.set_joint_position_target(tgt)

            self._prev_act = self.prev_act.clone()
            self.prev_act = act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
            self._apply_push()

            # Track distance
            curr_pos = self.robot.data.root_pos_w[:, :2]
            if self.prev_pos is not None:
                self.distance_traveled += (curr_pos - self.prev_pos).norm(dim=-1)
            self.prev_pos = curr_pos.clone()

        def _apply_action(self):
            pass

        def _get_observations(self):
            r = self.robot
            q = r.data.root_quat_w
            lv = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            jp_leg = r.data.joint_pos[:, self.loco_idx[:12]]
            jv_leg = r.data.joint_vel[:, self.loco_idx[:12]] * 0.1
            jp_waist = r.data.joint_pos[:, self.loco_idx[12:15]]
            jv_waist = r.data.joint_vel[:, self.loco_idx[12:15]] * 0.1

            gait = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)], -1)
            torso_euler = quat_to_euler_xyz(q)

            # Foot contact from sensors (one per foot → concat)
            lf = self._left_foot_sensor.data.net_forces_w   # [N, 1, 3]
            rf = self._right_foot_sensor.data.net_forces_w  # [N, 1, 3]
            foot_contact = torch.cat([
                (lf.norm(dim=-1) > 5.0).float(),
                (rf.norm(dim=-1) > 5.0).float(),
            ], dim=-1)                                       # [N, 2]

            # Height obs — must match training
            current_height = r.data.root_pos_w[:, 2:3]
            height_err = self.height_cmd[:, None] - current_height
            root_height_vel = lv[:, 2:3]
            payload_est = torch.zeros(self.num_envs, 1, device=self.device)

            # === Loco obs (73 dim) ===
            loco_obs = torch.cat([
                lv, av, g,                              # 9
                jp_leg, jv_leg, jp_waist, jv_waist,     # 30
                self.height_cmd[:, None],               # 1
                self.speed_cmd[:, None],                # 1
                self.heading_offset[:, None],           # 1
                self.body_yaw_cmd[:, None],             # 1
                self.yaw_mode[:, None],                 # 1
                gait,                                   # 2
                self.prev_act,                          # 15
                torso_euler,                            # 3
                self.torso_cmd,                         # 3
                foot_contact,                           # 2
                current_height,                         # 1
                height_err,                             # 1
                root_height_vel,                        # 1
                payload_est,                            # 1
            ], dim=-1)  # = 73

            arm_obs = torch.zeros(self.num_envs, ARM_OBS_DIM, device=self.device)
            hand_obs = torch.zeros(self.num_envs, HAND_OBS_DIM, device=self.device)

            obs = torch.cat([loco_obs, arm_obs, hand_obs], dim=-1)  # 188
            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gravity_vec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(q, gravity_vec)
            fallen = (pos[:, 2] < 0.35) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_hyperextended = (lk < -0.05) | (rk < -0.05) | (lk > 2.0) | (rk > 2.0)  # added upper limit to prevent deep squat
            knee_collapse = (lk < 0.3) | (rk < 0.3)  # V4: prevent squat (default knee=0.42, so 0.3 is safe)
            waist_excessive = (jp[:, 14].abs() > 0.20) | (jp[:, 13].abs() > 0.10)  # V4: roll 0.15→0.10

            terminated = fallen | bad_orientation | knee_hyperextended | knee_collapse | waist_excessive
            if terminated.any():
                self.total_falls += terminated.sum().item()
            time_out = self.episode_length_buf >= self.max_episode_length
            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0:
                return
            n = len(env_ids)
            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            self.robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)
            root_pos = torch.tensor([[0.0, 0.0, HEIGHT_DEFAULT]], device=self.device).expand(n, -1).clone()
            default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(n, -1)  # V2: wxyz identity
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, default_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)
            # Re-apply commands
            self.speed_cmd[env_ids] = mode_cfg["speed"]
            self.heading_offset[env_ids] = mode_cfg["heading"]
            self.body_yaw_cmd[env_ids] = mode_cfg["yaw_cmd"]
            self.yaw_mode[env_ids] = mode_cfg["yaw_mode"]

    return Env(EnvCfg())


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    net = LocoActorCritic(OBS_DIM, LOCO_ACT_DIM).to(device)
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()

    level = ckpt.get("curriculum_level", "?")
    best_r = ckpt.get("best_reward", "?")
    iteration = ckpt.get("iteration", "?")
    print(f"\n[Load] Checkpoint: iter={iteration}, best_reward={best_r}, level={level}")

    obs, _ = env.reset()
    obs_t = obs["policy"]

    print(f"\n[Play] Mode: {args_cli.mode}, Max steps: {args_cli.max_steps}")
    if args_cli.mode == "mixed":
        print("  Mixed schedule:")
        for s, spd, hd, yc, ym in MIXED_SCHEDULE:
            mode_str = "rate" if ym == 0 else "abs"
            print(f"    Step {s:5d}: speed={spd:.1f}, heading={hd:.2f}, yaw={yc:.2f} ({mode_str})")
    print("-" * 70)

    mixed_idx = 0

    for step in range(args_cli.max_steps):
        # Update mixed mode
        if args_cli.mode == "mixed" and mixed_idx < len(MIXED_SCHEDULE) - 1:
            if step >= MIXED_SCHEDULE[mixed_idx + 1][0]:
                mixed_idx += 1
                _, spd, hd, yc, ym = MIXED_SCHEDULE[mixed_idx]
                env.speed_cmd[:] = spd
                env.heading_offset[:] = hd
                env.body_yaw_cmd[:] = yc
                env.yaw_mode[:] = ym
                mode_str = "rate" if ym == 0 else "abs"
                print(f"  >>> Step {step}: speed={spd:.1f}, heading={hd:.2f}, yaw={yc:.2f} ({mode_str})")

        with torch.no_grad():
            action = net.act(obs_t, det=True)

        obs_dict, _, _, _, _ = env.step(action)
        obs_t = obs_dict["policy"]

        # Direction arrow
        spd = env.speed_cmd
        has_vel = spd > 0.01

        if has_vel.any():
            arrow_pos = env.robot.data.root_pos_w.clone()
            arrow_pos[:, 2] += 0.7

            q_root = env.robot.data.root_quat_w
            euler = quat_to_euler_xyz(q_root)
            root_yaw = euler[:, 2]

            # Arrow heading = body heading + heading_offset (+pi for arrow_x.usd convention)
            cmd_heading = env.heading_offset + root_yaw + np.pi
            z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(env.num_envs, -1)
            arrow_quat = quat_from_angle_axis(cmd_heading, z_axis)

            arrow_scale = torch.zeros(env.num_envs, 3, device=device)
            arrow_scale[:, 0] = 0.15 + spd * 0.5
            arrow_scale[:, 1] = 0.15
            arrow_scale[:, 2] = 0.15

            env.vel_arrow.visualize(translations=arrow_pos, orientations=arrow_quat, scales=arrow_scale)
            if not env._arrow_visible:
                env.vel_arrow.set_visibility(True)
                env._arrow_visible = True
        else:
            if env._arrow_visible:
                env.vel_arrow.set_visibility(False)
                env._arrow_visible = False

        # Track
        lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
        av_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_ang_vel_w)

        actual_speed = torch.sqrt(lv_b[:, 0]**2 + lv_b[:, 1]**2).mean().item()
        env.speed_history.append(actual_speed)
        env.speed_cmd_history.append(env.speed_cmd.mean().item())

        if actual_speed > 0.05:
            actual_heading = torch.atan2(lv_b[:, 1], lv_b[:, 0]).mean().item()
            env.heading_history.append(actual_heading)
            env.heading_cmd_history.append(env.heading_offset.mean().item())

        env.yaw_history.append(av_b[:, 2].mean().item())
        env.yaw_cmd_history.append(env.body_yaw_cmd.mean().item())

        # Periodic status
        if step % 200 == 0 and step > 0:
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            q = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=device).expand(env.num_envs, -1)
            pg = quat_apply_inverse(q, gvec)
            tilt = torch.sqrt(pg[:, 0]**2 + pg[:, 1]**2).mean().item() * 180 / np.pi

            euler_log = quat_to_euler_xyz(q)
            root_yaw_deg = euler_log[:, 2].mean().item() * 180 / np.pi

            jp = env.robot.data.joint_pos[:, env.loco_idx]
            lk = jp[:, env.left_knee_idx].mean().item()
            rk = jp[:, env.right_knee_idx].mean().item()
            ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            wY = jp[:, 12].mean().item()
            wR = jp[:, 13].mean().item()
            wP = jp[:, 14].mean().item()

            wx = env.robot.data.root_pos_w[:, 0].mean().item()
            wy = env.robot.data.root_pos_w[:, 1].mean().item()
            dist = env.distance_traveled.mean().item()

            cmd_spd = env.speed_cmd.mean().item()
            cmd_hd = env.heading_offset.mean().item()
            cmd_yaw = env.body_yaw_cmd.mean().item()
            yaw_actual = av_b[:, 2].mean().item()

            print(f"  [Step {step:5d}] H={height:.3f}m Tilt={tilt:.1f}deg rootYaw={root_yaw_deg:.1f}deg")
            print(f"    Speed: actual={actual_speed:.3f} cmd={cmd_spd:.3f}")
            print(f"    Heading: cmd={cmd_hd:.3f}rad")
            print(f"    Yaw: actual={yaw_actual:.3f} cmd={cmd_yaw:.3f} mode={'abs' if env.yaw_mode.mean().item() > 0.5 else 'rate'}")
            print(f"    Knees: L={lk:.2f} R={rk:.2f} | ankR={ankR:.3f}")
            print(f"    Waist: yaw={wY:.3f} roll={wR:.3f} pitch={wP:.3f}")
            print(f"    WORLD: ({wx:.2f}, {wy:.2f}) dist={dist:.2f}m | Falls={env.total_falls} Push={env.total_pushes}")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 70)
    print("[SONUC] Tamamlandi")
    print("=" * 70)
    height = env.robot.data.root_pos_w[:, 2].mean().item()
    dist = env.distance_traveled.mean().item()

    # Speed tracking
    if env.speed_history:
        spd_arr = np.array(env.speed_history)
        spd_cmd_arr = np.array(env.speed_cmd_history)
        spd_mae = np.abs(spd_arr - spd_cmd_arr).mean()
        spd_rmse = np.sqrt(np.mean((spd_arr - spd_cmd_arr) ** 2))
    else:
        spd_mae, spd_rmse = 0, 0

    # Yaw tracking
    if env.yaw_history:
        yaw_arr = np.array(env.yaw_history)
        yaw_cmd_arr = np.array(env.yaw_cmd_history)
        yaw_mae = np.abs(yaw_arr - yaw_cmd_arr).mean()
        yaw_rmse = np.sqrt(np.mean((yaw_arr - yaw_cmd_arr) ** 2))
    else:
        yaw_mae, yaw_rmse = 0, 0

    jp = env.robot.data.joint_pos[:, env.loco_idx]
    ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
    hipR = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()

    print(f"  Mode: {args_cli.mode}")
    print(f"  Steps: {args_cli.max_steps}")
    print(f"  Final height: {height:.3f}m (target: {HEIGHT_DEFAULT}m)")
    print(f"  Distance: {dist:.2f}m")
    print(f"  Falls: {env.total_falls}, Pushes: {env.total_pushes}")
    print(f"  Speed tracking  — MAE: {spd_mae:.4f}, RMSE: {spd_rmse:.4f}")
    print(f"  Yaw tracking    — MAE: {yaw_mae:.4f}, RMSE: {yaw_rmse:.4f}")
    print(f"  Posture — AnkRoll: {ankR:.3f}, HipRoll: {hipR:.3f}")

    print("\n[Play] Simulasyon acik. Ctrl+C ile kapat.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
