"""
Unified Stage 2 Loco: Perturbation-Robust Locomotion — Play/Evaluation
=======================================================================
Loco fine-tuned with frozen arm perturbation + payload forces.
DualActorCritic: LocoActor (66->15, fine-tuned) + ArmActor (39->7, frozen).

MODES:
- standing: vx=0, arm active + load — pure stability test
- walk: vx=0.4, arm active + load — normal walking
- mixed: varying commands + arm + load — full demo simulation
- push: walk + enhanced push — robustness test
- arm_test: standing, arm reaches rapidly, heavy load — worst case

MUST match training: obs, act, termination, clamps, arm obs, external forces.
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
ARM_ACTION_SCALE = 2.0  # Stage 2 Arm uses 2.0
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT
GAIT_FREQUENCY = _cfg_mod.GAIT_FREQUENCY
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS
SHOULDER_OFFSET_RIGHT = _cfg_mod.SHOULDER_OFFSET_RIGHT

# Must match training
OBS_DIM = 66
ACT_DIM = 15   # 12 leg + 3 waist
ARM_OBS_DIM = 39
ARM_ACT_DIM = 7

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Loco: Play/Eval — Perturbation-Robust")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="walk",
                        choices=["standing", "walk", "mixed", "push", "arm_test"])
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--load_kg", type=float, default=None,
                        help="Override payload mass (kg). Default: mode-dependent.")
    parser.add_argument("--no_arm", action="store_true",
                        help="Disable arm perturbation (baseline comparison)")
    parser.add_argument("--no_reset", action="store_true",
                        help="Disable auto-reset on termination (count but don't reset, for visual debug)")
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
# MODE CONFIGS
# ============================================================================

MODE_CONFIGS = {
    "standing": {
        "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
        "push_force": (0, 10), "push_interval": (300, 600), "push_duration": (5, 15),
        "load_kg": 1.0, "arm_change_interval": (100, 300),
    },
    "walk": {
        "vx": 0.4, "vy": 0.0, "vyaw": 0.0,
        "push_force": (0, 20), "push_interval": (200, 400), "push_duration": (5, 15),
        "load_kg": 1.0, "arm_change_interval": (100, 300),
    },
    "mixed": {
        "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
        "push_force": (0, 30), "push_interval": (150, 350), "push_duration": (5, 15),
        "load_kg": 1.5, "arm_change_interval": (100, 300),
    },
    "push": {
        "vx": 0.4, "vy": 0.0, "vyaw": 0.0,
        "push_force": (10, 60), "push_interval": (80, 200), "push_duration": (5, 15),
        "load_kg": 1.0, "arm_change_interval": (100, 300),
    },
    "arm_test": {
        "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
        "push_force": (0, 10), "push_interval": (300, 600), "push_duration": (5, 15),
        "load_kg": 2.0, "arm_change_interval": (30, 80),  # Rapid arm motion
    },
}

mode_cfg = MODE_CONFIGS[args_cli.mode]

# Override load if specified
if args_cli.load_kg is not None:
    mode_cfg["load_kg"] = args_cli.load_kg

# Mixed mode schedule (step, vx, vy, vyaw)
MIXED_SCHEDULE = [
    (0,    0.0,  0.0,  0.0),    # Stand
    (300,  0.3,  0.0,  0.0),    # Slow forward
    (600,  0.5,  0.0,  0.0),    # Medium forward
    (900,  0.5,  0.0,  0.3),    # Forward + turn right
    (1200, 0.5,  0.0, -0.3),    # Forward + turn left
    (1500, 0.3,  0.15, 0.0),    # Forward + sidestep right
    (1800, 0.3, -0.15, 0.0),    # Forward + sidestep left
    (2100, 0.0,  0.0,  0.0),    # Stop (transition test)
    (2400, 0.6,  0.0,  0.0),    # Fast forward
    (2700, 0.0,  0.0,  0.0),    # Stop
]

print("=" * 80)
print(f"STAGE 2 LOCO: PLAY — Mode: {args_cli.mode}")
print(f"  Cartesian: vx={mode_cfg['vx']}, vy={mode_cfg['vy']}, vyaw={mode_cfg['vyaw']}")
print(f"  Push: force={mode_cfg['push_force']}, interval={mode_cfg['push_interval']}")
print(f"  Load: {mode_cfg['load_kg']:.1f}kg, Arm: {'OFF' if args_cli.no_arm else 'ON'}")
print(f"  Obs: {OBS_DIM}, Loco Act: {ACT_DIM}, Arm Act: {ARM_ACT_DIM}")
print(f"  Checkpoint: {args_cli.checkpoint}")
print("=" * 80)

# ============================================================================
# QUATERNION UTILS (must match training)
# ============================================================================

def quat_to_euler_xyz(quat):
    """Convert wxyz quaternion (Isaac Lab convention) to roll, pitch, yaw."""
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
    """Get forward direction from wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fwd_x = 1 - 2 * (y * y + z * z)
    fwd_y = 2 * (x * y + w * z)
    fwd_z = 2 * (x * z - w * y)
    return torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)


def compute_orientation_error(palm_quat, target_dir):
    """Compute angle between palm forward and target direction."""
    forward = get_palm_forward(palm_quat)
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# NETWORK — DualActorCritic (must match training)
# ============================================================================

class DualActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Loco Actor: 66 -> [512,256,128](LN+ELU) -> 15
        layers = []
        prev = OBS_DIM
        for h in [512, 256, 128]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        self.loco_actor = nn.Sequential(*layers)

        # Loco Critic
        layers = []
        prev = OBS_DIM
        for h in [512, 256, 128]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.loco_critic = nn.Sequential(*layers)

        # Arm Actor: 39 -> [256,256,128](ELU, NO LayerNorm) -> 7
        layers = []
        prev = ARM_OBS_DIM
        for h in [256, 256, 128]:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ARM_ACT_DIM))
        self.arm_actor = nn.Sequential(*layers)

        self.loco_log_std = nn.Parameter(torch.zeros(ACT_DIM))

    def act_loco(self, x, det=False):
        mean = self.loco_actor(x)
        if det:
            return mean
        std = self.loco_log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def act_arm(self, arm_obs):
        """Frozen arm inference (deterministic)."""
        return self.arm_actor(arm_obs)


# ============================================================================
# ENVIRONMENT (play — arm perturbation + external forces)
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
                    disable_gravity=False, retain_accelerations=True,
                    linear_damping=0.0, angular_damping=0.0,
                    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1)),
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

            # Loco joint indices
            self.loco_idx = []
            for name in LOCO_JOINT_NAMES:
                if name in jn:
                    self.loco_idx.append(jn.index(name))
            self.loco_idx = torch.tensor(self.loco_idx, device=self.device)

            # Right arm joint indices (7)
            self.right_arm_idx = []
            for name in ARM_JOINT_NAMES_RIGHT:
                if name in jn:
                    self.right_arm_idx.append(jn.index(name))
            self.right_arm_idx = torch.tensor(self.right_arm_idx, device=self.device)

            # Left arm joint indices (7)
            self.left_arm_idx = []
            for name in ARM_JOINT_NAMES_LEFT:
                if name in jn:
                    self.left_arm_idx.append(jn.index(name))
            self.left_arm_idx = torch.tensor(self.left_arm_idx, device=self.device)

            # Hand indices
            self.hand_idx = []
            for name in HAND_JOINT_NAMES:
                if name in jn:
                    self.hand_idx.append(jn.index(name))
            self.hand_idx = torch.tensor(self.hand_idx, device=self.device)

            # Per-joint indices for logging (V6.2)
            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")
            ankle_roll_names = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
            self.ankle_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_roll_names], device=self.device)
            hip_roll_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
            self.hip_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_roll_names], device=self.device)
            hip_yaw_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
            self.hip_yaw_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_yaw_names], device=self.device)

            # Find palm body for EE + external force
            body_names = self.robot.data.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right" in name.lower() and "palm" in name.lower():
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                self.palm_idx = len(body_names) - 1
                print(f"  [WARN] right_palm not found! Using body {self.palm_idx}")
            else:
                print(f"  Palm body: '{body_names[self.palm_idx]}' (idx={self.palm_idx})")

            # Defaults
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)
            self.default_right_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_RIGHT],
                device=self.device, dtype=torch.float32)
            self.default_left_arm = torch.tensor(
                [DEFAULT_ARM_POSES[n] for n in ARM_JOINT_NAMES_LEFT],
                device=self.device, dtype=torch.float32)

            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            self.shoulder_offset = torch.tensor(SHOULDER_OFFSET_RIGHT, device=self.device, dtype=torch.float32)

            # Commands — Cartesian (vx, vy, vyaw)
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.vel_cmd[:, 0] = mode_cfg["vx"]
            self.vel_cmd[:, 1] = mode_cfg["vy"]
            self.vel_cmd[:, 2] = mode_cfg["vyaw"]
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

            # Loco state
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)

            # Arm perturbation state
            self.prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)
            self.arm_target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.arm_target_orient = torch.zeros(self.num_envs, 3, device=self.device)
            self.arm_target_orient[:, 2] = -1.0  # palm down default
            self.arm_target_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            lo, hi = mode_cfg["arm_change_interval"]
            self.arm_target_change_at = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
            self.arm_steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.arm_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.arm_freeze_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # External force state
            self.load_mass = torch.ones(self.num_envs, device=self.device) * mode_cfg["load_kg"]

            # Push timer
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            pi_lo, pi_hi = mode_cfg["push_interval"]
            self.push_timer = torch.randint(pi_lo, pi_hi + 1, (self.num_envs,), device=self.device)
            self.push_duration = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.push_force_active = torch.zeros(self.num_envs, 3, device=self.device)

            # No-reset mode (for visual debug — count but don't reset)
            self._no_reset = args_cli.no_reset

            # Contact sensor
            self._contact_sensor = self.scene["contact_forces"]
            self._illegal_contact_ids, illegal_names = self._contact_sensor.find_bodies(
                "pelvis|torso_link|.*knee_link")
            print(f"[Play] Illegal contact bodies: {illegal_names}")

            # Tracking
            self.total_falls = 0
            self.fall_reasons = {"height": 0, "orientation": 0, "knee": 0,
                                 "waist": 0, "hip_yaw": 0, "illegal_contact": 0}
            self.total_pushes = 0
            self.vx_history = []
            self.vx_cmd_history = []
            self.height_history = []
            self.distance_traveled = torch.zeros(self.num_envs, device=self.device)
            self.prev_pos = None

            # Initial arm target
            self._sample_arm_targets(torch.arange(self.num_envs, device=self.device))

            # Arm target marker (green sphere)
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/armTarget",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.9, 0.1)),
                    ),
                },
            )
            self.arm_target_marker = VisualizationMarkers(marker_cfg)

            # Velocity arrow
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

            print(f"[Play] {self.num_envs} envs, mode={args_cli.mode}, "
                  f"load={mode_cfg['load_kg']:.1f}kg, arm={'OFF' if args_cli.no_arm else 'ON'}")

        @property
        def robot(self):
            return self.scene["robot"]

        # ================================================================
        # ARM TARGET SAMPLING (Stage 2 workspace, body frame)
        # ================================================================

        def _sample_arm_targets(self, env_ids):
            n = len(env_ids)
            azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
            elevation = torch.empty(n, device=self.device).uniform_(-0.3, 0.5)
            radius = torch.empty(n, device=self.device).uniform_(0.15, 0.45)

            target_x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
            target_y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
            target_z = radius * torch.sin(elevation) + self.shoulder_offset[2]

            target_x = target_x.clamp(-0.10, 0.55)
            target_y = target_y.clamp(-0.60, -0.05)
            target_z = target_z.clamp(-0.15, 0.55)

            self.arm_target_pos_body[env_ids] = torch.stack([target_x, target_y, target_z], dim=-1)
            orient = torch.randn(n, 3, device=self.device)
            orient = orient / (orient.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            self.arm_target_orient[env_ids] = orient

            self.arm_steps_since_spawn[env_ids] = 0

            # Arm freeze mode: 30% chance
            freeze_mask = torch.rand(n, device=self.device) < 0.3
            self.arm_frozen[env_ids] = freeze_mask
            freeze_dur = torch.randint(50, 200, (n,), device=self.device)
            self.arm_freeze_timer[env_ids] = torch.where(
                freeze_mask, freeze_dur, torch.zeros(n, dtype=torch.long, device=self.device))

        # ================================================================
        # ARM OBS (Stage 2 identical — 39 dim)
        # ================================================================

        def get_arm_obs(self):
            r = self.robot
            root_pos = r.data.root_pos_w
            root_quat = r.data.root_quat_w

            arm_pos = r.data.joint_pos[:, self.right_arm_idx]
            arm_vel = r.data.joint_vel[:, self.right_arm_idx] * 0.1

            palm_pos = r.data.body_pos_w[:, self.palm_idx]
            palm_quat = r.data.body_quat_w[:, self.palm_idx]
            fwd = get_palm_forward(palm_quat)
            ee_w = palm_pos + 0.02 * fwd
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)

            pos_error = self.arm_target_pos_body - ee_body
            orient_err = compute_orientation_error(palm_quat, self.arm_target_orient)
            orient_err_norm = orient_err.unsqueeze(-1) / np.pi

            steps_norm = (self.arm_steps_since_spawn.float() / 200.0).unsqueeze(-1).clamp(0, 2)

            obs = torch.cat([
                arm_pos,                       # 7
                arm_vel,                       # 7
                ee_body,                       # 3
                palm_quat,                     # 4
                self.arm_target_pos_body,      # 3
                self.arm_target_orient,        # 3
                pos_error,                     # 3
                orient_err_norm,               # 1
                self.prev_arm_act,             # 7
                steps_norm,                    # 1
            ], dim=-1)  # = 39
            return obs.clamp(-10, 10).nan_to_num()

        # ================================================================
        # EXTERNAL FORCES (palm load + torso push — match training)
        # ================================================================

        def _apply_forces(self):
            num_bodies = self.robot.data.body_pos_w.shape[1]
            forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
            torques = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)

            # Palm load: gravity + sway
            load_mask = self.load_mass > 0.01
            if load_mask.any():
                forces[load_mask, self.palm_idx, 2] = -self.load_mass[load_mask] * 9.81
                sway = torch.randn(load_mask.sum(), 2, device=self.device)
                sway = sway * self.load_mass[load_mask, None] * 2.0
                forces[load_mask, self.palm_idx, :2] += sway

            # Torso push (3D, duration-based)
            # Continue active push
            active_push = self.push_duration > 0
            if active_push.any():
                forces[active_push, 0, :] += self.push_force_active[active_push]
                self.push_duration[active_push] -= 1

            # Start new push
            push_mask = self.step_count >= self.push_timer
            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                force = torch.randn(n, 3, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                fmin, fmax = mode_cfg["push_force"]
                mag = torch.rand(n, 1, device=self.device) * (fmax - fmin) + fmin
                self.push_force_active[ids] = force * mag
                dur_lo, dur_hi = mode_cfg["push_duration"]
                self.push_duration[ids] = torch.randint(dur_lo, dur_hi + 1, (n,), device=self.device)

                pi_lo, pi_hi = mode_cfg["push_interval"]
                self.push_timer[ids] = self.step_count[ids] + torch.randint(pi_lo, pi_hi + 1, (n,), device=self.device)
                self.total_pushes += n

            self.robot.set_external_force_and_torque(forces, torques)

        # ================================================================
        # PRE-PHYSICS STEP (match training)
        # ================================================================

        def _pre_physics_step(self, loco_act):
            self.actions = loco_act.clone()
            tgt = self.robot.data.default_joint_pos.clone()

            # Loco joints (V6.2 identical)
            tgt[:, self.loco_idx] = self.default_loco + loco_act * self.action_scales

            # Waist clamp
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)

            # Hip yaw clamp (V6.1)
            for hy_idx in self.hip_yaw_loco_idx:
                tgt[:, self.loco_idx[hy_idx]].clamp_(-0.3, 0.3)

            # RIGHT ARM: arm policy action or default
            if not args_cli.no_arm:
                arm_act = self.prev_arm_act.clone()
                tgt[:, self.right_arm_idx] = self.default_right_arm + arm_act * ARM_ACTION_SCALE
            else:
                tgt[:, self.right_arm_idx] = self.default_right_arm

            # LEFT ARM: default
            tgt[:, self.left_arm_idx] = self.default_left_arm

            # HANDS: open
            tgt[:, self.hand_idx] = self.default_hand

            self.robot.set_joint_position_target(tgt)

            # Update state
            self._prev_act = self.prev_act.clone()
            self.prev_act = loco_act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
            self.step_count += 1

            # Arm target timer
            self.arm_target_timer += 1
            self.arm_steps_since_spawn += 1

            # Arm freeze countdown
            freeze_expired = self.arm_frozen & (self.arm_freeze_timer <= 0)
            self.arm_frozen[freeze_expired] = False
            self.arm_freeze_timer[self.arm_frozen] -= 1

            # Resample arm targets
            lo, hi = mode_cfg["arm_change_interval"]
            change_mask = self.arm_target_timer >= self.arm_target_change_at
            if change_mask.any():
                change_ids = change_mask.nonzero(as_tuple=False).squeeze(-1)
                self._sample_arm_targets(change_ids)
                self.arm_target_timer[change_ids] = 0
                self.arm_target_change_at[change_ids] = torch.randint(lo, hi + 1, (len(change_ids),), device=self.device)

            # Apply external forces
            self._apply_forces()

            # Track distance
            curr_pos = self.robot.data.root_pos_w[:, :2]
            if self.prev_pos is not None:
                self.distance_traveled += (curr_pos - self.prev_pos).norm(dim=-1)
            self.prev_pos = curr_pos.clone()

        def _apply_action(self):
            pass

        # ================================================================
        # OBSERVATIONS (V6.2 identical — 66 dim)
        # ================================================================

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
                self.vel_cmd,                           # 3
                gait,                                   # 2
                self.prev_act,                          # 15
                torso_euler,                            # 3
                self.torso_cmd,                         # 3
            ], dim=-1)  # = 66

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        # ================================================================
        # TERMINATION (V6.2 identical)
        # ================================================================

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gravity_vec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(q, gravity_vec)

            fallen = (pos[:, 2] < 0.55) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_bad = (lk < -0.05) | (rk < -0.05) | (lk > 1.5) | (rk > 1.5)

            waist_excessive = (jp[:, 14].abs() > 0.35) | (jp[:, 13].abs() > 0.25)

            hip_yaw_L = jp[:, self.hip_yaw_loco_idx[0]]
            hip_yaw_R = jp[:, self.hip_yaw_loco_idx[1]]
            hip_yaw_excessive = (hip_yaw_L.abs() > 0.6) | (hip_yaw_R.abs() > 0.6)

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
                n_term = terminated.sum().item()
                self.total_falls += n_term
                # Per-reason counting (a single step can trigger multiple reasons)
                self.fall_reasons["height"] += fallen.sum().item()
                self.fall_reasons["orientation"] += bad_orientation.sum().item()
                self.fall_reasons["knee"] += knee_bad.sum().item()
                self.fall_reasons["waist"] += waist_excessive.sum().item()
                self.fall_reasons["hip_yaw"] += hip_yaw_excessive.sum().item()
                self.fall_reasons["illegal_contact"] += illegal_contact.sum().item()

            # --no_reset mode: count terminations but don't actually reset
            if self._no_reset:
                time_out = self.episode_length_buf >= self.max_episode_length
                return torch.zeros_like(terminated), time_out

            time_out = self.episode_length_buf >= self.max_episode_length
            return terminated, time_out

        # ================================================================
        # RESET
        # ================================================================

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
            self.prev_arm_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)
            self.step_count[env_ids] = 0
            pi_lo, pi_hi = mode_cfg["push_interval"]
            self.push_timer[env_ids] = torch.randint(pi_lo, pi_hi + 1, (n,), device=self.device)
            self.push_duration[env_ids] = 0
            self.push_force_active[env_ids] = 0

            # Re-apply velocity command
            self.vel_cmd[env_ids, 0] = mode_cfg["vx"]
            self.vel_cmd[env_ids, 1] = mode_cfg["vy"]
            self.vel_cmd[env_ids, 2] = mode_cfg["vyaw"]

            # Resample arm targets
            self._sample_arm_targets(env_ids)
            lo, hi = mode_cfg["arm_change_interval"]
            self.arm_target_timer[env_ids] = 0
            self.arm_target_change_at[env_ids] = torch.randint(lo, hi + 1, (n,), device=self.device)

    return Env(EnvCfg())


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    net = DualActorCritic().to(device)
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()

    level = ckpt.get("curriculum_level", "?")
    best_r = ckpt.get("best_reward", "?")
    iteration = ckpt.get("iteration", "?")
    print(f"\n[Load] Checkpoint: iter={iteration}, best_reward={best_r}, level={level}")

    # Count params
    loco_params = sum(p.numel() for n, p in net.named_parameters() if "loco_" in n)
    arm_params = sum(p.numel() for n, p in net.named_parameters() if "arm_" in n)
    print(f"  Loco params: {loco_params:,} | Arm params: {arm_params:,}")

    obs, _ = env.reset()
    obs_t = obs["policy"]

    print(f"\n[Play] Mode: {args_cli.mode}, Max steps: {args_cli.max_steps}")
    if args_cli.mode == "mixed":
        print("  Mixed schedule:")
        for s, vx, vy, vyaw in MIXED_SCHEDULE:
            print(f"    Step {s:5d}: vx={vx:.1f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
    print("-" * 70)

    mixed_idx = 0
    arm_reaches = 0  # Count how many times arm gets close to target

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

        # Arm inference (frozen, deterministic)
        if not args_cli.no_arm:
            with torch.no_grad():
                arm_obs = env.get_arm_obs()
                arm_act = net.act_arm(arm_obs).clamp(-1.5, 1.5)
                arm_act = torch.where(env.arm_frozen.unsqueeze(-1), env.prev_arm_act, arm_act)
                env.prev_arm_act = arm_act.clone()

        # Loco inference (deterministic for play)
        with torch.no_grad():
            loco_action = net.act_loco(obs_t, det=True)

        obs_dict, _, _, _, _ = env.step(loco_action)
        obs_t = obs_dict["policy"]

        # --- Arm target distance ---
        if not args_cli.no_arm:
            root_pos = env.robot.data.root_pos_w
            root_quat = env.robot.data.root_quat_w
            palm_pos = env.robot.data.body_pos_w[:, env.palm_idx]
            palm_quat = env.robot.data.body_quat_w[:, env.palm_idx]
            fwd = get_palm_forward(palm_quat)
            ee_w = palm_pos + 0.02 * fwd
            ee_body = quat_apply_inverse(root_quat, ee_w - root_pos)
            arm_dist = (env.arm_target_pos_body - ee_body).norm(dim=-1)

            # Count reaches (< 0.05m)
            if (arm_dist < 0.05).any():
                arm_reaches += (arm_dist < 0.05).sum().item()

        # --- Visualize arm target in world frame ---
        if not args_cli.no_arm:
            from isaaclab.utils.math import quat_apply
            target_w = quat_apply(root_quat, env.arm_target_pos_body) + root_pos
            env.arm_target_marker.visualize(translations=target_w)

        # --- Velocity direction arrow ---
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
        env.height_history.append(env.robot.data.root_pos_w[:, 2].mean().item())

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

            # Arm status
            arm_str = ""
            if not args_cli.no_arm:
                arm_d = arm_dist.mean().item()
                frozen_pct = env.arm_frozen.float().mean().item() * 100
                arm_str = f"armDist={arm_d:.3f}m frz={frozen_pct:.0f}% reaches={arm_reaches}"

            # Joint details
            jp = env.robot.data.joint_pos[:, env.loco_idx]
            lk = jp[:, env.left_knee_idx].mean().item()
            rk = jp[:, env.right_knee_idx].mean().item()
            ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            hipR = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()
            dist = env.distance_traveled.mean().item()

            print(f"  [Step {step:5d}] H={height:.3f}m Tilt={tilt:.1f}deg "
                  f"vx={actual_vx:.3f}(cmd:{cmd_vx:.3f}) vy={actual_vy:.3f}(cmd:{cmd_vy:.3f}) "
                  f"vyaw={actual_vyaw:.3f}(cmd:{cmd_vyaw:.3f})")
            print(f"              Knees: L={lk:.2f} R={rk:.2f} | ankR={ankR:.3f} hipR={hipR:.3f} | "
                  f"Falls={env.total_falls} Push={env.total_pushes} Dist={dist:.2f}m")
            if arm_str:
                print(f"              Arm: {arm_str} | Load={env.load_mass.mean().item():.2f}kg")

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

    # Height stability
    if env.height_history:
        h_arr = np.array(env.height_history)
        h_std = np.std(h_arr)
        h_min = np.min(h_arr)
        h_max = np.max(h_arr)
    else:
        h_std, h_min, h_max = 0, 0, 0

    jp = env.robot.data.joint_pos[:, env.loco_idx]
    ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
    hipR = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()

    print(f"  Mode: {args_cli.mode}")
    print(f"  Steps: {args_cli.max_steps}")
    print(f"  Final height: {height:.3f}m (target: {HEIGHT_DEFAULT}m)")
    print(f"  Height stability: std={h_std:.4f}, min={h_min:.3f}, max={h_max:.3f}")
    print(f"  Distance: {dist:.2f}m")
    print(f"  Falls: {env.total_falls}, Pushes: {env.total_pushes}")
    if env.total_falls > 0:
        fr = env.fall_reasons
        print(f"  Fall reasons: height={fr['height']} orient={fr['orientation']} knee={fr['knee']} "
              f"waist={fr['waist']} hip_yaw={fr['hip_yaw']} contact={fr['illegal_contact']}")
    print(f"  Velocity tracking — MAE: {vx_err:.4f}, RMSE: {vx_rmse:.4f}")
    print(f"  Posture — AnkRoll: {ankR:.3f}, HipRoll: {hipR:.3f}")
    print(f"  Load: {mode_cfg['load_kg']:.1f}kg, Arm: {'OFF' if args_cli.no_arm else 'ON'}")
    if not args_cli.no_arm:
        print(f"  Arm reaches (<0.05m): {arm_reaches}")
    avg_ep_len = args_cli.max_steps / max(env.total_falls, 1)
    survival_pct = (1.0 - env.total_falls / args_cli.max_steps) * 100 if env.total_falls < args_cli.max_steps else 0.0
    print(f"  Avg episode length: {avg_ep_len:.0f} steps")
    print(f"  Survival: {survival_pct:.1f}%")
    if args_cli.no_reset:
        print(f"  [NO_RESET mode — falls counted but robot not reset]")

    print("\n[Play] Simulasyon acik. Ctrl+C ile kapat.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
