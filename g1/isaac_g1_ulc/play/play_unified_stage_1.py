"""
Unified Stage 1: Omnidirectional Locomotion — Play/Evaluation (V6)
===================================================================
Cartesian velocity control: (vx, vy, vyaw).
66-dim obs, 15-dim loco action — MUST match training.

MODES:
- standing: vx=0, vy=0, vyaw=0 (standing test)
- slow: vx=0.2 (slow forward walk)
- walk: vx=0.4 (normal walking)
- fast: vx=0.8 (fast walking)
- turn: vx=0.4 + vyaw=0.5 (turning test)
- lateral: vx=0, vy=0.3 (sideways walk)
- mixed: Scheduled commands (full demo sequence)
- push: walk + random push forces

V6 (2026-02-22): Complete rewrite — Stage 2 play birebir + V6 train sync.
V6.1 (2026-02-24): Synced with V6.1 train fixes:
    self_collisions=True, hip_yaw clamp +-0.3 rad, hip_yaw termination |hip_yaw|>0.6.
    66 obs (Cartesian), wxyz quaternion.
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

# Must match training — 66 dim obs, 15 dim act
OBS_DIM = 3 + 3 + 3 + 12 + 12 + 3 + 3 + 1 + 3 + 2 + 15 + 3 + 3  # = 66
ACT_DIM = NUM_LOCO_JOINTS  # 15

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Stage 1: Play/Eval V6 (Cartesian, 66 obs)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="walk",
                        choices=["standing", "slow", "walk", "fast", "turn", "lateral", "mixed", "push"])
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensor, ContactSensorCfg

# ============================================================================
# MODE CONFIGS — Cartesian (vx, vy, vyaw)
# ============================================================================

MODE_CONFIGS = {
    "standing": {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "slow":     {"vx": 0.2, "vy": 0.0, "vyaw": 0.0, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "walk":     {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "fast":     {"vx": 0.8, "vy": 0.0, "vyaw": 0.0, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "turn":     {"vx": 0.4, "vy": 0.0, "vyaw": 0.5, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "lateral":  {"vx": 0.0, "vy": 0.3, "vyaw": 0.0, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "mixed":    {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "push_force": (0, 0), "push_interval": (9999, 9999)},
    "push":     {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "push_force": (0, 30), "push_interval": (100, 300)},
}

mode_cfg = MODE_CONFIGS[args_cli.mode]

# Mixed mode schedule (step, vx, vy, vyaw)
MIXED_SCHEDULE = [
    (0,    0.0,  0.0,  0.0),    # Stand
    (300,  0.2,  0.0,  0.0),    # Slow forward
    (600,  0.5,  0.0,  0.0),    # Medium forward
    (900,  0.5,  0.0,  0.3),    # Forward + turn right
    (1200, 0.5,  0.0, -0.3),    # Forward + turn left
    (1500, 0.3,  0.15, 0.0),    # Forward + sidestep right
    (1800, 0.3, -0.15, 0.0),    # Forward + sidestep left
    (2100, 0.8,  0.0,  0.0),    # Fast forward
    (2400,-0.2,  0.0,  0.0),    # Slow backward
    (2700, 0.0,  0.0,  0.0),    # Stop
]

print("=" * 80)
print(f"UNIFIED STAGE 1: PLAY V6 — Mode: {args_cli.mode}")
print(f"  Cartesian: vx={mode_cfg['vx']}, vy={mode_cfg['vy']}, vyaw={mode_cfg['vyaw']}")
print(f"  Push: force={mode_cfg['push_force']}, interval={mode_cfg['push_interval']}")
print(f"  Obs: {OBS_DIM}, Act: {ACT_DIM}")
print(f"  Checkpoint: {args_cli.checkpoint}")
print("=" * 80)

# ============================================================================
# NETWORK (must match training)
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, num_obs=OBS_DIM, num_act=ACT_DIM, hidden=[512, 256, 128]):
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
    V6: Correct wxyz order (matches training).
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
        # V6.2: Contact sensor for illegal contact termination
        contact_forces = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3,
            track_air_time=False,
        )
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
                    enabled_self_collisions=True,        # V6.1: match training (prevent leg crossing)
                    solver_position_iteration_count=4,    # V6: match training
                    solver_velocity_iteration_count=1)),  # V6: match training
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

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 60.0
        action_space = ACT_DIM
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
            print(f"\n[Play] Robot joints ({len(jn)}): {jn}")

            # Joint indices
            self.loco_idx = []
            for name in LOCO_JOINT_NAMES:
                if name in jn:
                    self.loco_idx.append(jn.index(name))
            self.loco_idx = torch.tensor(self.loco_idx, device=self.device)

            self.arm_idx = []
            for name in ARM_JOINT_NAMES:
                if name in jn:
                    self.arm_idx.append(jn.index(name))
            self.arm_idx = torch.tensor(self.arm_idx, device=self.device)

            self.hand_idx = []
            for name in HAND_JOINT_NAMES:
                if name in jn:
                    self.hand_idx.append(jn.index(name))
            self.hand_idx = torch.tensor(self.hand_idx, device=self.device)

            # Per-joint indices for logging
            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")
            ankle_roll_names = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
            self.ankle_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_roll_names], device=self.device)
            hip_roll_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
            self.hip_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_roll_names], device=self.device)
            # V6.1: Hip yaw indices for clamp + termination
            hip_yaw_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
            self.hip_yaw_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_yaw_names], device=self.device)

            # Symmetry pairs
            sym_pairs = [
                ("left_hip_pitch_joint", "right_hip_pitch_joint"),
                ("left_hip_roll_joint", "right_hip_roll_joint"),
                ("left_hip_yaw_joint", "right_hip_yaw_joint"),
                ("left_knee_joint", "right_knee_joint"),
                ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
                ("left_ankle_roll_joint", "right_ankle_roll_joint"),
            ]
            self.sym_left_idx = []
            self.sym_right_idx = []
            for ln, rn in sym_pairs:
                if ln in LOCO_JOINT_NAMES and rn in LOCO_JOINT_NAMES:
                    self.sym_left_idx.append(LOCO_JOINT_NAMES.index(ln))
                    self.sym_right_idx.append(LOCO_JOINT_NAMES.index(rn))
            self.sym_left_idx = torch.tensor(self.sym_left_idx, device=self.device)
            self.sym_right_idx = torch.tensor(self.sym_right_idx, device=self.device)

            # Defaults
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)

            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # Commands — Cartesian (vx, vy, vyaw)
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.vel_cmd[:, 0] = mode_cfg["vx"]
            self.vel_cmd[:, 1] = mode_cfg["vy"]
            self.vel_cmd[:, 2] = mode_cfg["vyaw"]
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

            # State
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)

            # Push
            pi_lo, pi_hi = mode_cfg["push_interval"]
            if pi_hi <= pi_lo:
                pi_hi = pi_lo + 1
            self.push_timer = torch.randint(pi_lo, pi_hi, (self.num_envs,), device=self.device)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.total_pushes = 0
            self.total_falls = 0

            # V6.2: Contact sensor for illegal contact termination
            self._contact_sensor = self.scene["contact_forces"]
            self._illegal_contact_ids, illegal_names = self._contact_sensor.find_bodies(
                "pelvis|torso_link|.*knee_link")
            print(f"[Play] Illegal contact bodies: {illegal_names} (ids: {self._illegal_contact_ids})")
            if len(self._illegal_contact_ids) == 0:
                print(f"  [WARN] No illegal contact bodies found! All: {self._contact_sensor.body_names}")

            # Tracking
            self.vx_history = []
            self.vx_cmd_history = []
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

            # Waist clamp — match training
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)

            # V6.1: Hip yaw clamp — prevent scissor gait (+-0.3 rad = +-17 deg)
            for hy_idx in self.hip_yaw_loco_idx:
                tgt[:, self.loco_idx[hy_idx]].clamp_(-0.3, 0.3)

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

            obs = torch.cat([
                lv, av, g,                              # 9
                jp_leg, jv_leg, jp_waist, jv_waist,     # 30
                self.height_cmd[:, None],               # 1
                self.vel_cmd,                           # 3: (vx, vy, vyaw) Cartesian
                gait,                                   # 2
                self.prev_act,                          # 15
                torso_euler,                            # 3
                self.torso_cmd,                         # 3
            ], dim=-1)  # = 66

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gravity_vec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(q, gravity_vec)
            # V6.2 fix: 0.55m prevents squat exploit (robot sits at 0.38-0.40m)
            fallen = (pos[:, 2] < 0.55) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            # V6.2 fix: squat exploit uses knee=2.5+ rad. Default=0.42, walk max~1.0
            knee_bad = (lk < -0.05) | (rk < -0.05) | (lk > 1.5) | (rk > 1.5)

            waist_excessive = (jp[:, 14].abs() > 0.35) | (jp[:, 13].abs() > 0.25)

            # V6.1: Hip yaw excessive rotation termination
            hip_yaw_L = jp[:, self.hip_yaw_loco_idx[0]]
            hip_yaw_R = jp[:, self.hip_yaw_loco_idx[1]]
            hip_yaw_excessive = (hip_yaw_L.abs() > 0.6) | (hip_yaw_R.abs() > 0.6)

            # V6.2: Illegal contact — pelvis/torso/knee touching ground
            if len(self._illegal_contact_ids) > 0:
                net_forces = self._contact_sensor.data.net_forces_w_history
                illegal_contact = torch.any(
                    torch.max(
                        torch.norm(net_forces[:, :, self._illegal_contact_ids], dim=-1), dim=1
                    )[0] > 1.0,
                    dim=1,
                )
            else:
                illegal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            terminated = fallen | bad_orientation | knee_bad | waist_excessive | hip_yaw_excessive | illegal_contact
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
            default_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(n, -1)  # wxyz identity
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, default_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)
            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)
            # Re-apply velocity command
            self.vel_cmd[env_ids, 0] = mode_cfg["vx"]
            self.vel_cmd[env_ids, 1] = mode_cfg["vy"]
            self.vel_cmd[env_ids, 2] = mode_cfg["vyaw"]

    return Env(EnvCfg())


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    net = ActorCritic(OBS_DIM, ACT_DIM).to(device)
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
        for s, vx, vy, vyaw in MIXED_SCHEDULE:
            print(f"    Step {s:5d}: vx={vx:.1f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
    print("-" * 70)

    mixed_idx = 0

    for step in range(args_cli.max_steps):
        # Update mixed mode commands
        if args_cli.mode == "mixed" and mixed_idx < len(MIXED_SCHEDULE) - 1:
            if step >= MIXED_SCHEDULE[mixed_idx + 1][0]:
                mixed_idx += 1
                _, vx, vy, vyaw = MIXED_SCHEDULE[mixed_idx]
                env.vel_cmd[:, 0] = vx
                env.vel_cmd[:, 1] = vy
                env.vel_cmd[:, 2] = vyaw
                print(f"  >>> Step {step}: vx={vx:.1f}, vy={vy:.2f}, vyaw={vyaw:.2f}")

        with torch.no_grad():
            action = net.act(obs_t, det=True)

        obs_dict, _, _, _, _ = env.step(action)
        obs_t = obs_dict["policy"]

        # --- Update velocity direction arrow ---
        vx_cmd = env.vel_cmd[:, 0]
        vy_cmd = env.vel_cmd[:, 1]
        vel_mag = torch.sqrt(vx_cmd**2 + vy_cmd**2)
        has_vel = vel_mag > 0.01

        if has_vel.any():
            arrow_pos = env.robot.data.root_pos_w.clone()
            arrow_pos[:, 2] += 0.7

            q_root = env.robot.data.root_quat_w
            euler = quat_to_euler_xyz(q_root)
            root_yaw = euler[:, 2]

            cmd_heading = torch.atan2(vy_cmd, vx_cmd) + root_yaw
            z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(env.num_envs, -1)
            arrow_quat = quat_from_angle_axis(cmd_heading, z_axis)

            arrow_scale = torch.zeros(env.num_envs, 3, device=device)
            arrow_scale[:, 0] = 0.15 + vel_mag * 0.5
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

        # Track velocity
        lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
        av_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_ang_vel_w)
        env.vx_history.append(lv_b[:, 0].mean().item())
        env.vx_cmd_history.append(env.vel_cmd[:, 0].mean().item())

        # Periodic status
        if step % 200 == 0 and step > 0:
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            q = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=device).expand(env.num_envs, -1)
            pg = quat_apply_inverse(q, gvec)
            tilt = torch.sqrt(pg[:, 0]**2 + pg[:, 1]**2).mean().item() * 180 / np.pi

            actual_vx = lv_b[:, 0].mean().item()
            actual_vy = lv_b[:, 1].mean().item()
            actual_vyaw = av_b[:, 2].mean().item()
            cmd_vx = env.vel_cmd[:, 0].mean().item()
            cmd_vy = env.vel_cmd[:, 1].mean().item()
            cmd_vyaw = env.vel_cmd[:, 2].mean().item()

            # Gait monitoring
            jp = env.robot.data.joint_pos[:, env.loco_idx]
            lk = jp[:, env.left_knee_idx].mean().item()
            rk = jp[:, env.right_knee_idx].mean().item()

            # Posture
            ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            hipR = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()
            dist = env.distance_traveled.mean().item()

            # World position
            wx = env.robot.data.root_pos_w[:, 0].mean().item()
            wy = env.robot.data.root_pos_w[:, 1].mean().item()

            # Joint details
            lhp = jp[:, 0].mean().item()   # left_hip_pitch
            rhp = jp[:, 1].mean().item()   # right_hip_pitch
            lhr = jp[:, 2].mean().item()   # left_hip_roll
            rhr = jp[:, 3].mean().item()   # right_hip_roll
            lhy = jp[:, 4].mean().item()   # left_hip_yaw
            rhy = jp[:, 5].mean().item()   # right_hip_yaw
            wY = jp[:, 12].mean().item()
            wR = jp[:, 13].mean().item()
            wP = jp[:, 14].mean().item()

            # Root yaw
            euler = quat_to_euler_xyz(q)
            root_yaw_deg = euler[:, 2].mean().item() * 180 / np.pi
            yaw_rate = av_b[:, 2].mean().item()

            print(f"  [Step {step:5d}] H={height:.3f}m Tilt={tilt:.1f}deg "
                  f"vx={actual_vx:.3f}(cmd:{cmd_vx:.3f}) vy={actual_vy:.3f}(cmd:{cmd_vy:.3f}) "
                  f"vyaw={actual_vyaw:.3f}(cmd:{cmd_vyaw:.3f})")
            print(f"              Knees: L={lk:.2f} R={rk:.2f} | ankR={ankR:.3f} hipR={hipR:.3f} | "
                  f"Falls={env.total_falls} Push={env.total_pushes} Dist={dist:.2f}m")
            print(f"              HipPitch: L={lhp:.3f} R={rhp:.3f} | "
                  f"HipRoll: L={lhr:.3f} R={rhr:.3f} | HipYaw: L={lhy:.3f} R={rhy:.3f}")
            print(f"              Waist: yaw={wY:.3f} roll={wR:.3f} pitch={wP:.3f} | "
                  f"rootYaw={root_yaw_deg:.1f}deg yawRate={yaw_rate:.3f}")
            print(f"              WORLD: ({wx:.2f}, {wy:.2f})")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 70)
    print("[SONUC] Tamamlandi")
    print("=" * 70)
    height = env.robot.data.root_pos_w[:, 2].mean().item()
    dist = env.distance_traveled.mean().item()

    # Velocity tracking
    if env.vx_history:
        vx_arr = np.array(env.vx_history)
        cmd_arr = np.array(env.vx_cmd_history)
        vx_err = np.abs(vx_arr - cmd_arr).mean()
        vx_rmse = np.sqrt(np.mean((vx_arr - cmd_arr) ** 2))
    else:
        vx_err, vx_rmse = 0, 0

    jp = env.robot.data.joint_pos[:, env.loco_idx]
    ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
    hipR = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()

    print(f"  Mode: {args_cli.mode}")
    print(f"  Steps: {args_cli.max_steps}")
    print(f"  Final height: {height:.3f}m (target: {HEIGHT_DEFAULT}m)")
    print(f"  Distance: {dist:.2f}m")
    print(f"  Falls: {env.total_falls}, Pushes: {env.total_pushes}")
    print(f"  Velocity tracking — MAE: {vx_err:.4f}, RMSE: {vx_rmse:.4f}")
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
