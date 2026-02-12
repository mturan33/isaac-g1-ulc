"""
29DoF G1 Stage 2: Locomotion with Gait Control
================================================
Stage 1 checkpoint'tan fine-tune ile yurume egitimi.

ARCHITECTURE: Single Actor-Critic (Loco only) — same as Stage 1
- LocoActor: 66 obs -> 15 act (12 leg + 3 waist)
- Kollar: Default pozda sabit (policy disi)
- Eller: Acik (policy disi)

GAIT CONTROL:
- Phase-based alternating gait (sin/cos @ 1.5 Hz)
- Knee alternation: swing leg 0.6rad, stance leg default(0.42)
- Foot clearance: hip lift during swing phase
- Velocity tracking: vx, vy, vyaw

CURRICULUM (5 levels):
- L0: Slow forward walk (vx=0~0.3), no push
- L1: Medium forward walk (vx=0~0.5), light push
- L2: Forward+backward+lateral (vx=-0.2~0.7, vy, vyaw), medium push
- L3: Full velocity range (vx=-0.3~1.0, vy, vyaw), medium push
- L4: Aggressive velocity + strong push (FINAL)

REWARD DESIGN:
- Velocity tracking: vx, vy, vyaw (exp-decay)
- Gait: knee alternation + foot clearance + foot contact pattern
- Posture: ankle_roll, symmetry, hip_roll (from Stage 1 V2)
- Stability: height, orientation, angular velocity
- Penalties: action_rate, jerk, energy
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime

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

# ============================================================================
# CURRICULUM
# ============================================================================

# NOTE: 29DoF G1 body frame — vx > 0 = FORWARD, vx < 0 = BACKWARD
# Confirmed by WORLD position tracking: robot faces +Y at spawn,
# body-frame +X is the robot's forward direction.
# The old 23DoF model had -X as forward — this model is DIFFERENT!
CURRICULUM = [
    {
        "description": "L0: Slow forward walk, no push",
        "threshold": 16.0,
        "vx": (0.0, 0.3),       # Slow forward (positive = forward in 29DoF G1)
        "vy": (-0.05, 0.05),    # Nearly zero lateral
        "vyaw": (-0.1, 0.1),    # Nearly zero turning
        "push_force": (0, 0),
        "push_interval": (9999, 9999),
        "mass_scale": (0.97, 1.03),
    },
    {
        "description": "L1: Medium forward walk, light push",
        "threshold": 18.0,
        "vx": (0.0, 0.5),       # Medium forward
        "vy": (-0.1, 0.1),      # Small lateral
        "vyaw": (-0.3, 0.3),    # Small turning
        "push_force": (0, 10),
        "push_interval": (300, 600),
        "mass_scale": (0.95, 1.05),
    },
    {
        "description": "L2: Forward+backward+lateral, medium push",
        "threshold": 19.0,
        "vx": (-0.2, 0.7),      # Forward + small backward
        "vy": (-0.2, 0.2),      # Lateral
        "vyaw": (-0.5, 0.5),    # Turning
        "push_force": (0, 15),
        "push_interval": (200, 500),
        "mass_scale": (0.93, 1.07),
    },
    {
        "description": "L3: Full velocity range, medium push",
        "threshold": 20.0,
        "vx": (-0.3, 1.0),      # Full range
        "vy": (-0.3, 0.3),      # Full lateral
        "vyaw": (-0.8, 0.8),    # Full turning
        "push_force": (0, 20),
        "push_interval": (150, 400),
        "mass_scale": (0.90, 1.10),
    },
    {
        "description": "L4: Aggressive velocity + strong push (FINAL)",
        "threshold": None,
        "vx": (-0.5, 1.2),
        "vy": (-0.4, 0.4),
        "vyaw": (-1.0, 1.0),
        "push_force": (0, 30),
        "push_interval": (100, 300),
        "mass_scale": (0.85, 1.15),
    },
]

# ============================================================================
# REWARD WEIGHTS
# ============================================================================

REWARD_WEIGHTS = {
    # Velocity tracking (primary task)
    "vx": 5.0,                  # Forward velocity tracking (most important)
    "vy": 3.0,                  # Lateral velocity tracking — INCREASED (was 2.0, drift problem)
    "vyaw": 3.0,                # Yaw rate tracking — INCREASED (was 2.0, rotation problem)
    # Gait control
    "gait_knee": 3.0,           # Alternating knee bend pattern
    "gait_clearance": 2.0,      # Foot clearance during swing
    "gait_contact": 2.0,        # Alternating foot contact pattern
    # Stability
    "height": 3.0,              # Height tracking
    "orientation": 6.0,         # Upright orientation — INCREASED (was 5.0→6.0, scale 8→15, torso tilt fix)
    "ang_vel_penalty": 1.0,     # Angular velocity penalty (reduced from Stage 1)
    # Posture (from Stage 1 V2 — keep feet/legs healthy)
    "ankle_penalty": 2.0,       # Ankle roll (reduced from 4.0 — some roll needed for walking)
    "foot_flatness": 1.5,       # Foot flatness (reduced — dynamic gait needs ankle flexibility)
    "symmetry_gait": 1.5,       # Gait symmetry (NOT standing symmetry — phase-shifted)
    "hip_roll_penalty": 1.5,    # Hip roll (reduced — some roll needed for walking)
    "knee_negative_penalty": -8.0,  # HARD penalty for knee < 0.1 rad (backward bending)
    # Waist posture — keep torso upright and facing forward
    "waist_posture": 3.0,       # Penalize waist_roll, waist_pitch AND waist_yaw deviation from 0
    # Standing posture — when vel_cmd ~0, all leg joints should return to default
    "standing_posture": 3.0,    # Conditional: full weight when standing, 0 when walking
    # Yaw stability — prevent body oscillation around vertical axis
    "yaw_rate_penalty": -2.0,   # Penalize angular velocity around z-axis (prevents learned sway)
    # Physics-based penalties (from literature review)
    "vz_penalty": -2.0,        # Vertical velocity penalty — prevents bouncing/oscillation (Booster Gym: -2.0)
    "feet_slip": -0.1,         # Stance foot sliding penalty (Booster Gym: -0.1)
    # Smoothness penalties
    "action_rate": -0.02,       # Slightly less than Stage 1
    "jerk": -0.05,              # Joint acceleration penalty — INCREASED (was -0.01, literature: -0.01~-1e-7)
    "energy": -0.0003,
    "alive": 1.0,
}

# ============================================================================
# OBS DIM — Same structure as Stage 1 (66 dim)
# ============================================================================

# lin_vel_b(3) + ang_vel_b(3) + proj_gravity(3)
# + joint_pos_leg(12) + joint_vel_leg(12)
# + joint_pos_waist(3) + joint_vel_waist(3)
# + height_cmd(1) + vel_cmd(3) + gait_phase(2)
# + prev_actions(15) + torso_euler(3) + torso_cmd(3)
OBS_DIM = 3 + 3 + 3 + 12 + 12 + 3 + 3 + 1 + 3 + 2 + 15 + 3 + 3  # = 66
ACT_DIM = NUM_LOCO_JOINTS  # 15

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="29DoF G1 Stage 2: Locomotion")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Stage 1 checkpoint to fine-tune from")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 2 checkpoint")
    parser.add_argument("--experiment_name", type=str, default="g1_29dof_stage2")
    parser.add_argument("--headless", action="store_true")
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
from isaaclab.utils.math import quat_apply_inverse
from torch.utils.tensorboard import SummaryWriter

print("=" * 80)
print("29DoF G1 STAGE 2: LOCOMOTION with GAIT CONTROL")
print(f"USD: {G1_29DOF_USD}")
print(f"Obs: {OBS_DIM}, Act: {ACT_DIM}")
print(f"Joints: {NUM_LOCO_JOINTS} loco + {NUM_ARM_JOINTS} arm + {NUM_HAND_JOINTS} hand = {NUM_LOCO_JOINTS + NUM_ARM_JOINTS + NUM_HAND_JOINTS}")
print(f"Gait frequency: {GAIT_FREQUENCY} Hz")
print("=" * 80)
for i, lv in enumerate(CURRICULUM):
    print(f"  Level {i}: {lv['description']}")
    print(f"    vx={lv['vx']}, vy={lv['vy']}, vyaw={lv['vyaw']}, push={lv['push_force']}")

# ============================================================================
# NETWORK — same as Stage 1
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, num_obs=OBS_DIM, num_act=ACT_DIM, hidden=[512, 256, 128]):
        super().__init__()
        # Actor
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        # Critic
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def act(self, x, det=False):
        mean = self.actor(x)
        if det:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, x, a):
        mean = self.actor(x)
        val = self.critic(x).squeeze(-1)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return val, dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


# ============================================================================
# PPO — same as Stage 1
# ============================================================================

class PPO:
    def __init__(self, net, device, lr=3e-4, max_iter=10000):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, max_iter, 1e-5)

    def gae(self, r, v, d, nv):
        adv = torch.zeros_like(r)
        last = 0
        for t in reversed(range(len(r))):
            nxt = nv if t == len(r) - 1 else v[t + 1]
            delta = r[t] + 0.99 * nxt * (1 - d[t]) - v[t]
            adv[t] = last = delta + 0.99 * 0.95 * (1 - d[t]) * last
        return adv, adv + v

    def update(self, obs, act, old_lp, ret, adv, old_v):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tot_a, tot_c, tot_e, n = 0, 0, 0, 0
        bs = obs.shape[0]

        for _ in range(5):
            idx = torch.randperm(bs, device=self.device)
            for i in range(0, bs, 4096):
                mb = idx[i:i + 4096]
                val, lp, ent = self.net.evaluate(obs[mb], act[mb])
                ratio = (lp - old_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(0.8, 1.2) * adv[mb]
                a_loss = -torch.min(s1, s2).mean()
                v_clip = old_v[mb] + (val - old_v[mb]).clamp(-0.2, 0.2)
                c_loss = 0.5 * torch.max((val - ret[mb]) ** 2, (v_clip - ret[mb]) ** 2).mean()
                loss = a_loss + 0.5 * c_loss - 0.01 * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                n += 1

        self.sched.step()
        return {"a": tot_a / n, "c": tot_c / n, "e": tot_e / n, "lr": self.sched.get_last_lr()[0]}


# ============================================================================
# ENVIRONMENT
# ============================================================================

def quat_to_euler_xyz(quat):
    """wxyz quaternion to roll, pitch, yaw."""
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
                    disable_gravity=False,
                    retain_accelerations=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.80),
                joint_pos=DEFAULT_ALL_POSES,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.90,
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[
                        ".*_hip_yaw_joint", ".*_hip_roll_joint",
                        ".*_hip_pitch_joint", ".*_knee_joint",
                        ".*waist.*",
                    ],
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
                    joint_names_expr=[
                        ".*_hand_index_.*_joint",
                        ".*_hand_middle_.*_joint",
                        ".*_hand_thumb_.*_joint",
                    ],
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
        episode_length_s = 20.0  # Longer than Stage 1 for walking
        action_space = ACT_DIM
        observation_space = OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=4.0)  # Wider spacing for walking

    class Env(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)

            # Find joint indices for loco joints
            jn = self.robot.joint_names
            print(f"\n[Env] Robot joint names ({len(jn)} total): {jn}")

            # Loco joint indices (legs + waist)
            self.loco_idx = []
            for name in LOCO_JOINT_NAMES:
                if name in jn:
                    self.loco_idx.append(jn.index(name))
                else:
                    print(f"  [WARN] Loco joint not found: {name}")
            self.loco_idx = torch.tensor(self.loco_idx, device=self.device)
            print(f"  Loco joints: {len(self.loco_idx)} / {NUM_LOCO_JOINTS}")

            # Arm joint indices (for holding at default)
            self.arm_idx = []
            for name in ARM_JOINT_NAMES:
                if name in jn:
                    self.arm_idx.append(jn.index(name))
            self.arm_idx = torch.tensor(self.arm_idx, device=self.device)
            print(f"  Arm joints: {len(self.arm_idx)} / {NUM_ARM_JOINTS}")

            # Hand joint indices (for holding open)
            self.hand_idx = []
            for name in HAND_JOINT_NAMES:
                if name in jn:
                    self.hand_idx.append(jn.index(name))
            self.hand_idx = torch.tensor(self.hand_idx, device=self.device)
            print(f"  Hand joints: {len(self.hand_idx)} / {NUM_HAND_JOINTS}")

            # ============================================================
            # Per-joint indices within LOCO_JOINT_NAMES for reward compute
            # ============================================================

            # Knee indices (for gait alternation)
            # left_knee_joint = index 6, right_knee_joint = index 7 in LEG_JOINT_NAMES
            self.left_knee_idx = LOCO_JOINT_NAMES.index("left_knee_joint")
            self.right_knee_idx = LOCO_JOINT_NAMES.index("right_knee_joint")

            # Hip pitch indices (for foot clearance)
            self.left_hip_pitch_idx = LOCO_JOINT_NAMES.index("left_hip_pitch_joint")
            self.right_hip_pitch_idx = LOCO_JOINT_NAMES.index("right_hip_pitch_joint")

            # Ankle roll (posture from Stage 1 V2)
            ankle_roll_names = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
            self.ankle_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_roll_names if n in LOCO_JOINT_NAMES],
                device=self.device)

            # Ankle pitch (posture)
            ankle_pitch_names = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
            self.ankle_pitch_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in ankle_pitch_names if n in LOCO_JOINT_NAMES],
                device=self.device)

            # Hip roll (posture)
            hip_roll_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
            self.hip_roll_loco_idx = torch.tensor(
                [LOCO_JOINT_NAMES.index(n) for n in hip_roll_names if n in LOCO_JOINT_NAMES],
                device=self.device)

            # Symmetry pairs — used for gait symmetry (phase-shifted, not identical)
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

            print(f"  Knee idx: L={self.left_knee_idx}, R={self.right_knee_idx}")
            print(f"  Hip pitch idx: L={self.left_hip_pitch_idx}, R={self.right_hip_pitch_idx}")
            print(f"  Ankle roll idx: {self.ankle_roll_loco_idx.tolist()}")
            print(f"  Symmetry pairs: {len(self.sym_left_idx)}")

            # Default poses as tensors
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)

            # Default knee value (for gait target computation)
            self.default_knee = DEFAULT_LOCO_LIST[self.left_knee_idx]  # 0.42
            print(f"  Default knee angle: {self.default_knee:.2f} rad")

            # Action scales (per-joint)
            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # State variables
            self.curr_level = 0
            self.curr_hist = []
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # (vx, vy, vyaw)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # roll/pitch/yaw targets
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_jvel = None

            # Push timer for perturbation
            lv = CURRICULUM[self.curr_level]
            pi_lo, pi_hi = lv["push_interval"]
            if pi_hi <= pi_lo:
                pi_hi = pi_lo + 1  # Prevent randint error when push disabled
            self.push_timer = torch.randint(
                pi_lo, pi_hi,
                (self.num_envs,), device=self.device)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # Command resample timer (resample vel_cmd periodically)
            self.cmd_resample_interval = 300  # Resample every 300 steps (~6 sec)
            self.cmd_timer = torch.randint(0, self.cmd_resample_interval,
                                           (self.num_envs,), device=self.device)

            # Sample initial commands
            self._sample_commands(torch.arange(self.num_envs, device=self.device))

            print(f"\n[Env] {self.num_envs} envs, level {self.curr_level}")
            print(f"  Obs: {OBS_DIM}, Act: {ACT_DIM}")
            print(f"  Height default: {HEIGHT_DEFAULT}m, Gait freq: {GAIT_FREQUENCY}Hz")

        @property
        def robot(self):
            return self.scene["robot"]

        def _sample_commands(self, env_ids):
            """Sample velocity commands from current curriculum level."""
            lv = CURRICULUM[self.curr_level]
            n = len(env_ids)
            # vx
            vx_lo, vx_hi = lv["vx"]
            self.vel_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (vx_hi - vx_lo) + vx_lo
            # vy
            vy_lo, vy_hi = lv["vy"]
            self.vel_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (vy_hi - vy_lo) + vy_lo
            # vyaw
            vyaw_lo, vyaw_hi = lv["vyaw"]
            self.vel_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (vyaw_hi - vyaw_lo) + vyaw_lo

        def update_curriculum(self, r):
            """Multi-criteria curriculum gating (from literature: ULC paper).
            Advancement requires BOTH:
              1. Mean reward > threshold
              2. vx tracking ratio > 0.6 (actual/cmd)
            This prevents gaming where robot gets high reward without proper walking.
            """
            self.curr_hist.append(r)
            if len(self.curr_hist) >= 100:
                avg = np.mean(self.curr_hist[-100:])
                thr = CURRICULUM[self.curr_level]["threshold"]

                # Multi-criteria check
                if thr is not None and avg > thr and self.curr_level < len(CURRICULUM) - 1:
                    # Additional gate: vx tracking quality
                    lv_b = quat_apply_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)
                    vx_actual = lv_b[:, 0].mean().item()
                    vx_cmd = self.vel_cmd[:, 0].mean().item()
                    vx_ratio = vx_actual / max(abs(vx_cmd), 0.05)  # tracking ratio

                    if vx_ratio > 0.6:  # at least 60% of commanded velocity achieved
                        self.curr_level += 1
                        lv = CURRICULUM[self.curr_level]
                        print(f"\n*** LEVEL UP! Now {self.curr_level}: {lv['description']} ***")
                        print(f"    vx={lv['vx']}, vy={lv['vy']}, vyaw={lv['vyaw']}")
                        print(f"    push={lv['push_force']}, mass_scale={lv['mass_scale']}")
                        print(f"    (vx_ratio={vx_ratio:.2f}, reward={avg:.1f})")
                        self.curr_hist = []
                        # Resample all commands for new level
                        self._sample_commands(torch.arange(self.num_envs, device=self.device))

        def _apply_push(self):
            """Apply random perturbation forces to robot."""
            lv = CURRICULUM[self.curr_level]
            self.step_count += 1
            push_mask = self.step_count >= self.push_timer

            # Command resample
            self.cmd_timer += 1
            cmd_mask = self.cmd_timer >= self.cmd_resample_interval
            if cmd_mask.any():
                cmd_ids = cmd_mask.nonzero(as_tuple=False).squeeze(-1)
                self._sample_commands(cmd_ids)
                self.cmd_timer[cmd_ids] = 0

            # Push forces
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                # Random force direction (horizontal only)
                force = torch.zeros(n, 3, device=self.device)
                force[:, :2] = torch.randn(n, 2, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                fmin, fmax = lv["push_force"]
                mag = torch.rand(n, 1, device=self.device) * (fmax - fmin) + fmin
                force = force * mag
                forces[ids, 0] = force

                # Reset timer
                pi_lo, pi_hi = lv["push_interval"]
                if pi_hi <= pi_lo:
                    pi_hi = pi_lo + 1
                self.push_timer[ids] = torch.randint(pi_lo, pi_hi, (n,), device=self.device)
                self.step_count[ids] = 0

            self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        def _pre_physics_step(self, act):
            self.actions = act.clone()

            # Apply loco actions
            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.loco_idx] = self.default_loco + act * self.action_scales

            # CLAMP waist joints to prevent extreme lean
            # waist indices in loco: 12=yaw, 13=roll, 14=pitch
            # Default is 0.0 for all three. Limit deviations:
            waist_yaw_idx = self.loco_idx[12]
            waist_roll_idx = self.loco_idx[13]
            waist_pitch_idx = self.loco_idx[14]
            tgt[:, waist_yaw_idx].clamp_(-0.15, 0.15)   # ±8.6° yaw (tightened from ±0.3 — prevents sideways gaze)
            tgt[:, waist_roll_idx].clamp_(-0.15, 0.15)  # ±8.6° roll (tight — lateral lean kills posture)
            tgt[:, waist_pitch_idx].clamp_(-0.2, 0.2)   # ±11.5° pitch (prevents forward lean exploit)

            # Hold arms at default
            tgt[:, self.arm_idx] = self.default_arm

            # Hold hands open
            tgt[:, self.hand_idx] = self.default_hand

            self.robot.set_joint_position_target(tgt)

            # Update state
            self._prev_act = self.prev_act.clone()
            self.prev_act = act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            # Apply perturbation + command resample
            self._apply_push()

        def _apply_action(self):
            pass

        def _get_observations(self):
            r = self.robot
            q = r.data.root_quat_w
            lv = quat_apply_inverse(q, r.data.root_lin_vel_w)
            av = quat_apply_inverse(q, r.data.root_ang_vel_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            # Leg joint state
            jp_leg = r.data.joint_pos[:, self.loco_idx[:12]]
            jv_leg = r.data.joint_vel[:, self.loco_idx[:12]] * 0.1

            # Waist joint state
            jp_waist = r.data.joint_pos[:, self.loco_idx[12:15]]
            jv_waist = r.data.joint_vel[:, self.loco_idx[12:15]] * 0.1

            # Gait phase
            gait = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)
            ], -1)

            # Torso orientation
            torso_euler = quat_to_euler_xyz(q)

            obs = torch.cat([
                lv,                          # 3: linear velocity body
                av,                          # 3: angular velocity body
                g,                           # 3: projected gravity
                jp_leg,                      # 12: leg joint positions
                jv_leg,                      # 12: leg joint velocities
                jp_waist,                    # 3: waist joint positions
                jv_waist,                    # 3: waist joint velocities
                self.height_cmd[:, None],    # 1: height command
                self.vel_cmd,                # 3: velocity command (vx, vy, vyaw)
                gait,                        # 2: gait phase (sin, cos)
                self.prev_act,               # 15: previous actions
                torso_euler,                 # 3: torso euler angles
                self.torso_cmd,              # 3: torso command
            ], dim=-1)

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self):
            r = self.robot
            q = r.data.root_quat_w
            pos = r.data.root_pos_w

            lv_w = r.data.root_lin_vel_w
            av_w = r.data.root_ang_vel_w
            lv_b = quat_apply_inverse(q, lv_w)
            av_b = quat_apply_inverse(q, av_w)
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

            # Joint positions in loco space
            jp = r.data.joint_pos[:, self.loco_idx]
            jv = r.data.joint_vel[:, self.loco_idx]

            # ============================================
            # VELOCITY TRACKING REWARDS
            # ============================================

            # vx tracking (most important — forward walking)
            vx_err = (lv_b[:, 0] - self.vel_cmd[:, 0]) ** 2
            r_vx = torch.exp(-3.0 * vx_err)

            # vy tracking (lateral)
            vy_err = (lv_b[:, 1] - self.vel_cmd[:, 1]) ** 2
            r_vy = torch.exp(-4.0 * vy_err)

            # vyaw tracking (yaw rate)
            vyaw_err = (av_b[:, 2] - self.vel_cmd[:, 2]) ** 2
            r_vyaw = torch.exp(-3.0 * vyaw_err)

            # ============================================
            # GAIT CONTROL REWARDS
            # ============================================

            ph = self.phase
            l_swing = (ph < 0.5).float()   # Left leg swings at phase 0~0.5
            r_swing = (ph >= 0.5).float()   # Right leg swings at phase 0.5~1.0

            # --- Knee alternation ---
            # Swing leg: knee bends MORE (0.65 rad)
            # Stance leg: knee at default (0.42 rad)
            # IMPORTANT: Knee must ALWAYS be positive (>= 0.1) — negative = backward bend
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_swing_target = 0.65     # Increased bend for swing
            knee_stance_target = self.default_knee  # 0.42

            left_knee_target = l_swing * knee_swing_target + (1 - l_swing) * knee_stance_target
            right_knee_target = r_swing * knee_swing_target + (1 - r_swing) * knee_stance_target

            knee_err = (lk - left_knee_target) ** 2 + (rk - right_knee_target) ** 2
            r_gait_knee = torch.exp(-5.0 * knee_err)

            # --- Knee negative penalty ---
            # CRITICAL: Hard penalty for negative knee angles — causes backward walking + heel walking
            # Knee should always be >= 0.1 rad. Below that → escalating linear penalty
            # At knee=-0.09: violation = 0.19, penalty = 0.19 * (-8.0) = -1.52 per knee
            knee_min = 0.1
            lk_violation = torch.clamp(knee_min - lk, min=0.0)  # linear, not squared
            rk_violation = torch.clamp(knee_min - rk, min=0.0)
            r_knee_neg_penalty = lk_violation + rk_violation  # total violation magnitude

            # --- Foot clearance ---
            # During swing, hip pitch should be more negative (lift leg forward+up)
            # Default hip_pitch = -0.20, swing target = -0.35 (more forward)
            lh = jp[:, self.left_hip_pitch_idx]
            rh = jp[:, self.right_hip_pitch_idx]
            hip_swing_target = -0.35
            hip_stance_target = self.default_loco[self.left_hip_pitch_idx].item()  # -0.20

            left_hip_target = l_swing * hip_swing_target + (1 - l_swing) * hip_stance_target
            right_hip_target = r_swing * hip_swing_target + (1 - r_swing) * hip_stance_target

            hip_err = (lh - left_hip_target) ** 2 + (rh - right_hip_target) ** 2
            r_gait_clearance = torch.exp(-4.0 * hip_err)

            # --- Foot contact pattern ---
            # We want the stance foot to be on the ground (low foot velocity)
            # and swing foot to be moving (higher velocity).
            # Approximate: knee velocity should be higher during swing.
            lk_vel = jv[:, self.left_knee_idx].abs()
            rk_vel = jv[:, self.right_knee_idx].abs()

            # Reward movement during swing, stillness during stance
            swing_activity = l_swing * lk_vel + r_swing * rk_vel  # Should be high
            stance_stability = (1 - l_swing) * lk_vel + (1 - r_swing) * rk_vel  # Should be low

            # Combine: high swing_activity + low stance_stability = good
            r_gait_contact = torch.tanh(swing_activity * 0.5) * torch.exp(-2.0 * stance_stability)

            # Scale gait rewards by command magnitude — don't expect gait when standing
            vel_magnitude = self.vel_cmd[:, 0].abs() + self.vel_cmd[:, 1].abs()
            gait_scale = torch.clamp(vel_magnitude / 0.2, 0.0, 1.0)  # 0 when standing, 1 when walking

            r_gait_knee = r_gait_knee * gait_scale + (1 - gait_scale) * 1.0  # Perfect score when standing
            r_gait_clearance = r_gait_clearance * gait_scale + (1 - gait_scale) * 1.0
            r_gait_contact = r_gait_contact * gait_scale  # Zero when standing is fine

            # ============================================
            # STABILITY REWARDS
            # ============================================

            # Height
            height = pos[:, 2]
            h_err = (height - self.height_cmd).abs()
            r_height = torch.exp(-10.0 * h_err)

            # Orientation (upright) — scale 15.0 for very strong uprightness signal
            # At 3° tilt: g[:,:2]~0.052, sum=0.0027 → exp(-15*0.0027)=0.96 (was 0.98 with scale 8)
            # At 5° tilt: g[:,:2]~0.087, sum=0.0076 → exp(-15*0.0076)=0.89 (was 0.94 with scale 8)
            # At 10° tilt: g[:,:2]~0.17, sum=0.030 → exp(-15*0.030)=0.64 (was 0.79 with scale 8)
            r_orient = torch.exp(-15.0 * (g[:, :2] ** 2).sum(-1))

            # Angular velocity penalty (want low)
            r_ang = torch.exp(-1.0 * (av_b ** 2).sum(-1))

            # ============================================
            # POSTURE REWARDS (from Stage 1 V2)
            # ============================================

            # Ankle roll — prevent inverted foot (still critical during walking)
            ankle_roll_err = (jp[:, self.ankle_roll_loco_idx] - self.default_loco[self.ankle_roll_loco_idx]) ** 2
            r_ankle = torch.exp(-15.0 * ankle_roll_err.sum(-1))  # Slightly less steep than Stage 1

            # Foot flatness — ankle pitch near default
            ankle_pitch_err = (jp[:, self.ankle_pitch_loco_idx] - self.default_loco[self.ankle_pitch_loco_idx]) ** 2
            r_foot_flat = torch.exp(-8.0 * ankle_pitch_err.sum(-1))

            # Gait symmetry — during walking, legs should be phase-shifted mirrors
            # At any time: left(t) ≈ right(t + T/2)
            # Approximate: left_action ≈ right_action delayed by half phase
            # Simple version: action amplitude should be similar for both legs
            left_pos = jp[:, self.sym_left_idx]
            right_pos = jp[:, self.sym_right_idx]
            # For walking, we want similar RANGE of motion (not identical positions)
            # Compare absolute deviations from default
            left_dev = (left_pos - self.default_loco[self.sym_left_idx]).abs()
            right_dev = (right_pos - self.default_loco[self.sym_right_idx]).abs()
            sym_range_err = (left_dev - right_dev) ** 2
            r_sym_gait = torch.exp(-3.0 * sym_range_err.sum(-1))

            # Hip roll — prevent leaning
            hip_roll_err = (jp[:, self.hip_roll_loco_idx] - self.default_loco[self.hip_roll_loco_idx]) ** 2
            r_hip_roll = torch.exp(-10.0 * hip_roll_err.sum(-1))

            # Waist posture — keep ALL waist joints near 0 (ALWAYS active)
            # waist indices in loco: 12=yaw, 13=roll, 14=pitch
            waist_yaw_val = jp[:, 12]    # waist_yaw in loco space
            waist_roll_val = jp[:, 13]   # waist_roll in loco space
            waist_pitch_val = jp[:, 14]  # waist_pitch in loco space
            waist_yaw_err = (waist_yaw_val - self.default_loco[12]) ** 2
            waist_roll_err = (waist_roll_val - self.default_loco[13]) ** 2
            waist_pitch_err = (waist_pitch_val - self.default_loco[14]) ** 2
            # Penalize yaw (prevents looking sideways), roll (lateral lean), pitch (forward lean)
            r_waist_posture = (torch.exp(-25.0 * waist_yaw_err)
                             * torch.exp(-20.0 * waist_roll_err)
                             * torch.exp(-15.0 * waist_pitch_err))

            # Standing posture — when vel_cmd ~0, ALL leg joints should be at default
            # When walking, gait rewards take over and legs are free to follow gait pattern
            # standing_scale: 1.0 when standing (vel_mag=0), 0.0 when walking (vel_mag>0.2)
            standing_scale = 1.0 - gait_scale  # gait_scale computed above: 0=standing, 1=walking
            leg_pos_err = (jp[:, :12] - self.default_loco[:12]) ** 2  # all 12 leg joints
            r_standing_posture_raw = torch.exp(-3.0 * leg_pos_err.sum(-1))
            # Blend: standing_scale * default_posture + (1-standing_scale) * perfect_score
            r_standing_posture = standing_scale * r_standing_posture_raw + (1 - standing_scale) * 1.0

            # ============================================
            # PHYSICS PENALTIES (from literature: Booster Gym, Humanoid-Gym)
            # ============================================

            # Vertical velocity penalty — prevents bouncing/oscillation during walking
            vz = lv_b[:, 2]  # body-frame vertical velocity
            r_vz_penalty = vz ** 2

            # Yaw rate penalty — prevents learned body sway/oscillation around z-axis
            # Without this, policy learns to oscillate waist/hips for momentum transfer
            yaw_rate = av_b[:, 2]  # angular velocity around z-axis (rad/s)
            r_yaw_rate_penalty = yaw_rate ** 2

            # Feet slip penalty — stance foot should not slide on ground
            # Use ankle joint velocities as proxy for foot sliding
            # During stance phase, ankle velocities should be near zero
            ankle_pitch_vel = jv[:, self.ankle_pitch_loco_idx].abs()  # [N, 2]
            ankle_roll_vel = jv[:, self.ankle_roll_loco_idx].abs()    # [N, 2]
            # Stance mask: 1-swing = stance (low velocity expected)
            left_stance = 1.0 - l_swing   # 1 during stance, 0 during swing
            right_stance = 1.0 - r_swing
            # Left foot slip: stance * (left_ankle_pitch_vel + left_ankle_roll_vel)
            left_slip = left_stance * (ankle_pitch_vel[:, 0] + ankle_roll_vel[:, 0])
            right_slip = right_stance * (ankle_pitch_vel[:, 1] + ankle_roll_vel[:, 1])
            r_feet_slip = left_slip + right_slip

            # ============================================
            # SMOOTHNESS PENALTIES
            # ============================================

            # Action rate
            dact = self.prev_act - self._prev_act
            r_action_rate = (dact ** 2).sum(-1)

            # Jerk (joint acceleration)
            if self._prev_jvel is None:
                self._prev_jvel = jv.clone()
            jerk = ((jv - self._prev_jvel) ** 2).sum(-1)
            self._prev_jvel = jv.clone()

            # Energy
            r_energy = (jv.abs() * jp.abs()).sum(-1)

            # ============================================
            # TOTAL REWARD
            # ============================================

            reward = (
                # Velocity tracking
                REWARD_WEIGHTS["vx"] * r_vx
                + REWARD_WEIGHTS["vy"] * r_vy
                + REWARD_WEIGHTS["vyaw"] * r_vyaw
                # Gait control
                + REWARD_WEIGHTS["gait_knee"] * r_gait_knee
                + REWARD_WEIGHTS["gait_clearance"] * r_gait_clearance
                + REWARD_WEIGHTS["gait_contact"] * r_gait_contact
                # Stability
                + REWARD_WEIGHTS["height"] * r_height
                + REWARD_WEIGHTS["orientation"] * r_orient
                + REWARD_WEIGHTS["ang_vel_penalty"] * r_ang
                # Posture
                + REWARD_WEIGHTS["ankle_penalty"] * r_ankle
                + REWARD_WEIGHTS["foot_flatness"] * r_foot_flat
                + REWARD_WEIGHTS["symmetry_gait"] * r_sym_gait
                + REWARD_WEIGHTS["hip_roll_penalty"] * r_hip_roll
                + REWARD_WEIGHTS["waist_posture"] * r_waist_posture
                + REWARD_WEIGHTS["standing_posture"] * r_standing_posture
                + REWARD_WEIGHTS["knee_negative_penalty"] * r_knee_neg_penalty
                # Physics penalties
                + REWARD_WEIGHTS["vz_penalty"] * r_vz_penalty
                + REWARD_WEIGHTS["yaw_rate_penalty"] * r_yaw_rate_penalty
                + REWARD_WEIGHTS["feet_slip"] * r_feet_slip
                # Smoothness penalties
                + REWARD_WEIGHTS["action_rate"] * r_action_rate
                + REWARD_WEIGHTS["jerk"] * jerk
                + REWARD_WEIGHTS["energy"] * r_energy
                + REWARD_WEIGHTS["alive"]
            )

            return reward

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            gravity_vec = torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(q, gravity_vec)

            # Fall detection
            fallen = (pos[:, 2] < 0.35) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            # Knee hyperextension termination — knee < -0.05 rad is physically invalid
            jp = self.robot.data.joint_pos[:, self.loco_idx]
            lk = jp[:, self.left_knee_idx]
            rk = jp[:, self.right_knee_idx]
            knee_hyperextended = (lk < -0.05) | (rk < -0.05)

            # Waist excessive lean termination — prevents forward lean exploit
            # waist_pitch (loco idx 14): |pitch| > 0.35 rad (~20°) = too far
            # waist_roll (loco idx 13): |roll| > 0.25 rad (~14°) = too far
            waist_pitch_val = jp[:, 14]
            waist_roll_val = jp[:, 13]
            waist_excessive = (waist_pitch_val.abs() > 0.35) | (waist_roll_val.abs() > 0.25)

            terminated = fallen | bad_orientation | knee_hyperextended | waist_excessive

            # Time limit
            time_out = self.episode_length_buf >= self.max_episode_length
            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0:
                return
            n = len(env_ids)

            # Reset robot state
            default_pos = self.robot.data.default_joint_pos[env_ids].clone()
            default_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

            # Small random perturbation on initial joint positions
            noise = torch.randn_like(default_pos) * 0.02
            self.robot.write_joint_state_to_sim(default_pos + noise, default_vel, None, env_ids)

            # Reset root state
            root_pos = torch.tensor([[0.0, 0.0, HEIGHT_DEFAULT]], device=self.device).expand(n, -1).clone()
            root_pos[:, :2] += torch.randn(n, 2, device=self.device) * 0.05
            # Small random yaw for variety
            yaw = torch.randn(n, device=self.device) * 0.1
            qx = torch.zeros(n, device=self.device)
            qy = torch.zeros(n, device=self.device)
            qz = torch.sin(yaw / 2)
            qw = torch.cos(yaw / 2)
            root_quat = torch.stack([qx, qy, qz, qw], dim=-1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            # Reset internal state
            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)

            # Reset push timer
            lv = CURRICULUM[self.curr_level]
            pi_lo, pi_hi = lv["push_interval"]
            if pi_hi <= pi_lo:
                pi_hi = pi_lo + 1
            self.push_timer[env_ids] = torch.randint(pi_lo, pi_hi, (n,), device=self.device)
            self.step_count[env_ids] = 0

            # Sample new commands
            self._sample_commands(env_ids)
            self.cmd_timer[env_ids] = torch.randint(0, self.cmd_resample_interval, (n,), device=self.device)

    return Env(EnvCfg())


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    net = ActorCritic(OBS_DIM, ACT_DIM).to(device)
    ppo = PPO(net, device, lr=3e-4, max_iter=args_cli.max_iterations)

    # Load checkpoint
    start_iter = 0
    best_reward = -1e10

    if args_cli.checkpoint:
        # Resume Stage 2 training
        print(f"\n[Load] Resuming Stage 2 from {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        if "iteration" in ckpt:
            start_iter = ckpt["iteration"] + 1
        if "best_reward" in ckpt:
            best_reward = ckpt["best_reward"]
        if "curriculum_level" in ckpt:
            env.curr_level = min(ckpt["curriculum_level"], len(CURRICULUM) - 1)
        print(f"  Resumed at iter {start_iter}, best_reward={best_reward:.2f}, level={env.curr_level}")

    elif args_cli.stage1_checkpoint:
        # Fine-tune from Stage 1 checkpoint
        print(f"\n[Load] Fine-tuning from Stage 1: {args_cli.stage1_checkpoint}")
        ckpt = torch.load(args_cli.stage1_checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        s1_level = ckpt.get("curriculum_level", "?")
        s1_reward = ckpt.get("best_reward", "?")
        s1_iter = ckpt.get("iteration", "?")
        print(f"  Stage 1 checkpoint: iter={s1_iter}, reward={s1_reward}, level={s1_level}")
        print(f"  Starting Stage 2 from scratch (iteration 0)")
        # Reset log_std for fresh exploration
        net.log_std.data.fill_(np.log(0.8))
    else:
        print("\n[WARN] Sifirdan baslatiliyor — Stage 1 checkpoint olmadan Stage 2 zor olabilir!")
        print("       --stage1_checkpoint ile Stage 1 V2 checkpoint ver.")

    # Logging
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\n[Log] {log_dir}")

    # Rollout buffer
    T = 24  # steps per rollout
    obs_buf = torch.zeros(T, env.num_envs, OBS_DIM, device=device)
    act_buf = torch.zeros(T, env.num_envs, ACT_DIM, device=device)
    rew_buf = torch.zeros(T, env.num_envs, device=device)
    done_buf = torch.zeros(T, env.num_envs, device=device)
    val_buf = torch.zeros(T, env.num_envs, device=device)
    lp_buf = torch.zeros(T, env.num_envs, device=device)

    obs, _ = env.reset()
    obs_t = obs["policy"]
    ep_rewards = torch.zeros(env.num_envs, device=device)
    ep_lengths = torch.zeros(env.num_envs, device=device)
    completed_rewards = []

    print(f"\n[Train] Starting from iter {start_iter}, max {args_cli.max_iterations}")
    print(f"  Envs: {env.num_envs}, Rollout: {T} steps")

    for iteration in range(start_iter, args_cli.max_iterations):
        # Exploration decay
        progress = iteration / args_cli.max_iterations
        std = 0.8 + (0.2 - 0.8) * progress
        net.log_std.data.fill_(np.log(max(std, 0.2)))

        # Collect rollout
        for t in range(T):
            with torch.no_grad():
                action = net.act(obs_t)
                val = net.critic(obs_t).squeeze(-1)
                dist = torch.distributions.Normal(net.actor(obs_t), net.log_std.clamp(-2, 1).exp())
                lp = dist.log_prob(action).sum(-1)

            obs_buf[t] = obs_t
            act_buf[t] = action
            val_buf[t] = val
            lp_buf[t] = lp

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs_next = obs_dict["policy"]
            done = (terminated | truncated).float()

            rew_buf[t] = reward
            done_buf[t] = done

            ep_rewards += reward
            ep_lengths += 1

            # Track completed episodes
            done_mask = done.bool()
            if done_mask.any():
                for r_val in ep_rewards[done_mask].cpu().numpy():
                    completed_rewards.append(r_val)
                ep_rewards[done_mask] = 0
                ep_lengths[done_mask] = 0

            obs_t = obs_next

        # Compute returns
        with torch.no_grad():
            nv = net.critic(obs_t).squeeze(-1)
        adv, ret = ppo.gae(rew_buf, val_buf, done_buf, nv)

        # Flatten
        obs_flat = obs_buf.reshape(-1, OBS_DIM)
        act_flat = act_buf.reshape(-1, ACT_DIM)
        lp_flat = lp_buf.reshape(-1)
        ret_flat = ret.reshape(-1)
        adv_flat = adv.reshape(-1)
        val_flat = val_buf.reshape(-1)

        # PPO update
        losses = ppo.update(obs_flat, act_flat, lp_flat, ret_flat, adv_flat, val_flat)

        # Logging
        mean_reward = rew_buf.mean().item()
        env.update_curriculum(mean_reward)

        if iteration % 10 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            writer.add_scalar("reward/mean_step", mean_reward, iteration)
            writer.add_scalar("reward/mean_episode", avg_ep, iteration)
            writer.add_scalar("loss/actor", losses["a"], iteration)
            writer.add_scalar("loss/critic", losses["c"], iteration)
            writer.add_scalar("loss/entropy", losses["e"], iteration)
            writer.add_scalar("train/lr", losses["lr"], iteration)
            writer.add_scalar("train/log_std", net.log_std.data.mean().item(), iteration)
            writer.add_scalar("curriculum/level", env.curr_level, iteration)

            # Robot stats
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
            avg_vx = lv_b[:, 0].mean().item()
            cmd_vx = env.vel_cmd[:, 0].mean().item()
            writer.add_scalar("robot/height", height, iteration)
            writer.add_scalar("robot/vx_actual", avg_vx, iteration)
            writer.add_scalar("robot/vx_cmd", cmd_vx, iteration)
            writer.add_scalar("robot/vx_err", abs(avg_vx - cmd_vx), iteration)

            # Posture stats
            jp = env.robot.data.joint_pos[:, env.loco_idx]
            ankle_roll_err = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            hip_roll_err = (jp[:, env.hip_roll_loco_idx] - env.default_loco[env.hip_roll_loco_idx]).abs().mean().item()
            left_p = jp[:, env.sym_left_idx]
            right_p = jp[:, env.sym_right_idx]
            sym_err = ((left_p - env.default_loco[env.sym_left_idx]).abs() - (right_p - env.default_loco[env.sym_right_idx]).abs()).abs().mean().item()
            writer.add_scalar("posture/ankle_roll_err", ankle_roll_err, iteration)
            writer.add_scalar("posture/hip_roll_err", hip_roll_err, iteration)
            writer.add_scalar("posture/gait_sym_err", sym_err, iteration)

            # Knee angles — monitor for backward bending
            knee_l = jp[:, env.left_knee_idx].mean().item()
            knee_r = jp[:, env.right_knee_idx].mean().item()
            writer.add_scalar("posture/knee_left", knee_l, iteration)
            writer.add_scalar("posture/knee_right", knee_r, iteration)

            # Waist joints — monitor for torso tilt
            waist_yaw = jp[:, 12].mean().item()   # waist_yaw in loco space
            waist_roll = jp[:, 13].mean().item()   # waist_roll — lateral lean
            waist_pitch = jp[:, 14].mean().item()  # waist_pitch — forward lean
            writer.add_scalar("posture/waist_yaw", waist_yaw, iteration)
            writer.add_scalar("posture/waist_roll", waist_roll, iteration)
            writer.add_scalar("posture/waist_pitch", waist_pitch, iteration)

            # Yaw rate — monitor for oscillation
            av_log = quat_apply_inverse(q_root, env.robot.data.root_ang_vel_w)
            yaw_rate_log = av_log[:, 2].abs().mean().item()
            writer.add_scalar("posture/yaw_rate", yaw_rate_log, iteration)

            # Torso tilt angle (degrees) — from projected gravity
            q_root = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=env.device).expand(env.num_envs, -1)
            proj_g = quat_apply_inverse(q_root, gvec)
            tilt_rad = torch.asin(torch.clamp((proj_g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))
            tilt_deg = torch.rad2deg(tilt_rad).mean().item()
            writer.add_scalar("posture/tilt_deg", tilt_deg, iteration)

            # Lateral velocity tracking (vy)
            vy_actual = lv_b[:, 1].mean().item()
            vy_cmd = env.vel_cmd[:, 1].mean().item()
            writer.add_scalar("robot/vy_actual", vy_actual, iteration)
            writer.add_scalar("robot/vy_cmd", vy_cmd, iteration)
            writer.add_scalar("robot/vy_err", abs(vy_actual - vy_cmd), iteration)

        if iteration % 50 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            lv_b = quat_apply_inverse(env.robot.data.root_quat_w, env.robot.data.root_lin_vel_w)
            avg_vx = lv_b[:, 0].mean().item()
            cmd_vx = env.vel_cmd[:, 0].mean().item()
            # Posture
            jp = env.robot.data.joint_pos[:, env.loco_idx]
            ankR = (jp[:, env.ankle_roll_loco_idx] - env.default_loco[env.ankle_roll_loco_idx]).abs().mean().item()
            knee_l_p = jp[:, env.left_knee_idx].mean().item()
            knee_r_p = jp[:, env.right_knee_idx].mean().item()
            # Waist and tilt for terminal monitoring
            w_yaw = jp[:, 12].mean().item()
            w_roll = jp[:, 13].mean().item()
            w_pitch = jp[:, 14].mean().item()
            q_root = env.robot.data.root_quat_w
            gvec = torch.tensor([0, 0, -1.], device=env.device).expand(env.num_envs, -1)
            proj_g = quat_apply_inverse(q_root, gvec)
            tilt_d = torch.rad2deg(torch.asin(torch.clamp((proj_g[:, :2] ** 2).sum(-1).sqrt(), max=1.0))).mean().item()
            av_term = quat_apply_inverse(q_root, env.robot.data.root_ang_vel_w)
            yr_term = av_term[:, 2].abs().mean().item()
            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_reward:.2f} EpR={avg_ep:.2f} "
                  f"H={height:.3f} vx={avg_vx:.3f}(cmd:{cmd_vx:.3f}) "
                  f"Lv={env.curr_level} ankR={ankR:.3f} "
                  f"knL={knee_l_p:.2f} knR={knee_r_p:.2f} "
                  f"tilt={tilt_d:.1f}° wY={w_yaw:.3f} wR={w_roll:.3f} wP={w_pitch:.3f} "
                  f"yR={yr_term:.2f} "
                  f"LR={losses['lr']:.2e} std={np.exp(net.log_std.data.mean().item()):.3f}")

        # Save checkpoints
        if iteration % 500 == 0 and iteration > 0:
            path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
            }, path)
            print(f"  [Save] {path}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            path = os.path.join(log_dir, "model_best.pt")
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
            }, path)

    # Final save
    path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": net.state_dict(),
        "iteration": args_cli.max_iterations - 1,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
    }, path)
    print(f"\n[Done] Training complete. Best reward: {best_reward:.2f}, Level: {env.curr_level}")
    print(f"  Log dir: {log_dir}")

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
