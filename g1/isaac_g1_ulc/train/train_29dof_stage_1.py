"""
29DoF G1 Stage 1: Standing
===========================
Sifirdan ayakta durma egitimi — G1-29DoF + DEX3 Wholebody modeli.

ARCHITECTURE: Single Actor-Critic (Loco only)
- LocoActor: 66 obs -> 15 act (12 leg + 3 waist)
- Kollar: Default pozda sabit (policy disi)
- Eller: Acik (policy disi)

CURRICULUM (2 levels):
- L0: Stable standing, small perturbation
- L1: Stronger perturbation + mass randomization
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

CURRICULUM = [
    {
        "description": "L0: Stable stand, small perturbation",
        "threshold": 12.0,
        "push_force": (0, 10),
        "push_interval": (200, 500),
        "mass_scale": (0.95, 1.05),
    },
    {
        "description": "L1: Stronger perturbation + mass DR (FINAL)",
        "threshold": None,
        "push_force": (0, 30),
        "push_interval": (100, 400),
        "mass_scale": (0.85, 1.15),
    },
]

# ============================================================================
# REWARD WEIGHTS
# ============================================================================

REWARD_WEIGHTS = {
    "height": 5.0,
    "orientation": 4.0,
    "lin_vel_penalty": 3.0,
    "ang_vel_penalty": 2.0,
    "joint_posture": 2.0,
    "action_rate": -0.03,
    "jerk": -0.02,
    "energy": -0.0005,
    "alive": 1.0,
}

# ============================================================================
# OBS DIM COMPUTATION
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
    parser = argparse.ArgumentParser(description="29DoF G1 Stage 1: Standing")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="g1_29dof_stage1")
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
print("29DoF G1 STAGE 1: STANDING")
print(f"USD: {G1_29DOF_USD}")
print(f"Obs: {OBS_DIM}, Act: {ACT_DIM}")
print(f"Joints: {NUM_LOCO_JOINTS} loco + {NUM_ARM_JOINTS} arm + {NUM_HAND_JOINTS} hand = {NUM_LOCO_JOINTS + NUM_ARM_JOINTS + NUM_HAND_JOINTS}")
print("=" * 80)

# ============================================================================
# NETWORK
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
# PPO
# ============================================================================

class PPO:
    def __init__(self, net, device, lr=3e-4, max_iter=5000):
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
        episode_length_s = 15.0
        action_space = ACT_DIM
        observation_space = OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=3.0)

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

            # Default poses as tensors
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)

            # Action scales (per-joint)
            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # State variables
            self.curr_level = 0
            self.curr_hist = []
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # Always zero for standing
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)  # roll/pitch/yaw targets
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_jvel = None

            # Push timer for perturbation
            lv = CURRICULUM[self.curr_level]
            self.push_timer = torch.randint(
                lv["push_interval"][0], lv["push_interval"][1],
                (self.num_envs,), device=self.device)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            print(f"[Env] {self.num_envs} envs, level {self.curr_level}")
            print(f"  Obs: {OBS_DIM}, Act: {ACT_DIM}")
            print(f"  Height default: {HEIGHT_DEFAULT}m")

        @property
        def robot(self):
            return self.scene["robot"]

        def update_curriculum(self, r):
            self.curr_hist.append(r)
            if len(self.curr_hist) >= 100:
                avg = np.mean(self.curr_hist[-100:])
                thr = CURRICULUM[self.curr_level]["threshold"]
                if thr is not None and avg > thr and self.curr_level < len(CURRICULUM) - 1:
                    self.curr_level += 1
                    lv = CURRICULUM[self.curr_level]
                    print(f"\n*** LEVEL UP! Now {self.curr_level}: {lv['description']} ***")
                    print(f"    push_force={lv['push_force']}, mass_scale={lv['mass_scale']}")
                    self.curr_hist = []

        def _apply_push(self):
            """Apply random perturbation forces to robot."""
            lv = CURRICULUM[self.curr_level]
            self.step_count += 1
            push_mask = self.step_count >= self.push_timer

            # Always create force buffer (zeros by default = clears old forces)
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

            if push_mask.any():
                ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                n = len(ids)
                # Random force direction (horizontal only)
                force = torch.zeros(n, 3, device=self.device)
                force[:, :2] = torch.randn(n, 2, device=self.device)
                force = force / (force.norm(dim=-1, keepdim=True) + 1e-8)
                mag = torch.rand(n, 1, device=self.device) * (lv["push_force"][1] - lv["push_force"][0]) + lv["push_force"][0]
                force = force * mag
                forces[ids, 0] = force

                # Reset timer
                self.push_timer[ids] = torch.randint(
                    lv["push_interval"][0], lv["push_interval"][1], (n,), device=self.device)
                self.step_count[ids] = 0

            # Apply (zeros for non-pushed envs, clears previous forces)
            self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        def _pre_physics_step(self, act):
            self.actions = act.clone()

            # Apply loco actions
            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.loco_idx] = self.default_loco + act * self.action_scales

            # Hold arms at default
            tgt[:, self.arm_idx] = self.default_arm

            # Hold hands open (default = 0)
            tgt[:, self.hand_idx] = self.default_hand

            self.robot.set_joint_position_target(tgt)

            # Update state
            self._prev_act = self.prev_act.clone()
            self.prev_act = act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            # Apply perturbation
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
            jp_leg = r.data.joint_pos[:, self.loco_idx[:12]]  # Only leg positions
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
                self.vel_cmd,                # 3: velocity command (always 0)
                gait,                        # 2: gait phase
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

            # Height reward
            height = pos[:, 2]
            h_err = (height - self.height_cmd).abs()
            r_height = torch.exp(-10.0 * h_err)

            # Orientation reward (upright)
            r_orient = torch.exp(-5.0 * (g[:, :2] ** 2).sum(-1))

            # Linear velocity penalty (want zero)
            r_lin = torch.exp(-4.0 * (lv_b ** 2).sum(-1))

            # Angular velocity penalty (want zero)
            r_ang = torch.exp(-2.0 * (av_b ** 2).sum(-1))

            # Joint posture (close to default)
            jp = r.data.joint_pos[:, self.loco_idx]
            pose_err = (jp - self.default_loco) ** 2
            r_posture = torch.exp(-2.0 * pose_err.sum(-1))

            # Action rate
            dact = self.prev_act - self._prev_act
            r_action_rate = (dact ** 2).sum(-1)

            # Jerk
            jv = r.data.joint_vel[:, self.loco_idx]
            if self._prev_jvel is None:
                self._prev_jvel = jv.clone()
            jerk = ((jv - self._prev_jvel) ** 2).sum(-1)
            self._prev_jvel = jv.clone()

            # Energy
            r_energy = (jv.abs() * r.data.joint_pos[:, self.loco_idx].abs()).sum(-1)

            # Total reward
            reward = (
                REWARD_WEIGHTS["height"] * r_height
                + REWARD_WEIGHTS["orientation"] * r_orient
                + REWARD_WEIGHTS["lin_vel_penalty"] * r_lin
                + REWARD_WEIGHTS["ang_vel_penalty"] * r_ang
                + REWARD_WEIGHTS["joint_posture"] * r_posture
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

            # Fall detection (same as Stage 7)
            fallen = (pos[:, 2] < 0.4) | (pos[:, 2] > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7
            terminated = fallen | bad_orientation

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

            # Reset root state (pose + velocity separately, like Stage 7)
            default_height = HEIGHT_DEFAULT
            root_pos = torch.tensor([[0.0, 0.0, default_height]], device=self.device).expand(n, -1).clone()
            root_pos[:, :2] += torch.randn(n, 2, device=self.device) * 0.05  # Small xy noise
            default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, default_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            # Reset internal state
            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)

            # Reset push timer
            lv = CURRICULUM[self.curr_level]
            self.push_timer[env_ids] = torch.randint(
                lv["push_interval"][0], lv["push_interval"][1], (n,), device=self.device)
            self.step_count[env_ids] = 0

    return Env(EnvCfg())


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    net = ActorCritic(OBS_DIM, ACT_DIM).to(device)
    ppo = PPO(net, device, lr=3e-4, max_iter=args_cli.max_iterations)

    # Resume from checkpoint
    start_iter = 0
    best_reward = -1e10
    if args_cli.checkpoint:
        print(f"\n[Load] Resuming from {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        if "iteration" in ckpt:
            start_iter = ckpt["iteration"] + 1
        if "best_reward" in ckpt:
            best_reward = ckpt["best_reward"]
        if "curriculum_level" in ckpt:
            env.curr_level = ckpt["curriculum_level"]
        print(f"  Resumed at iter {start_iter}, best_reward={best_reward:.2f}, level={env.curr_level}")

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
            writer.add_scalar("robot/height", height, iteration)

        if iteration % 50 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_reward:.2f} EpR={avg_ep:.2f} "
                  f"H={height:.3f} Lv={env.curr_level} "
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
    print(f"\n[Done] Training complete. Best reward: {best_reward:.2f}")
    print(f"  Log dir: {log_dir}")

    writer.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
