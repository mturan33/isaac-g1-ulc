"""
DEX3 Grasp Training — Phase A (Fixed-Base, Finger-Only)
========================================================
Robot stands fixed (fix_root_link=True), right arm in pre-grasp pose.
Only 7 right-hand finger joints are trained via PPO.
Sphere object spawns near right palm with curriculum-based noise.

ARCHITECTURE:
  Robot root: FIXED (no loco/arm policy)
  Legs/Arms:  Held at default pose via joint position targets
  GraspActor  (45->7) [256,128,64] + ELU — finger delta position actions
  GraspCritic (45->1) [256,128,64] + ELU — value estimation

CURRICULUM (5 levels, sphere only):
  L0: radius=0.04, noise=0.00 (sphere at palm center)
  L1: radius=0.04, noise=0.01
  L2: radius=0.035, noise=0.02
  L3: radius=0.03, noise=0.03
  L4: radius=0.025, noise=0.04 (FINAL)

USAGE:
  isaaclab.bat -p .../train_grasp_phase_a.py --num_envs 2048 --headless
  isaaclab.bat -p .../train_grasp_phase_a.py --num_envs 64  # smoke test

Author: Turan (isaac-g1-ulc), 2026-04-02
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Config imports
import importlib.util
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "config", "ulc_g1_29dof_cfg.py")
_spec = importlib.util.spec_from_file_location("ulc_g1_29dof_cfg", _cfg_path)
_cfg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg_mod)

G1_29DOF_USD = _cfg_mod.G1_29DOF_USD
HAND_JOINT_NAMES_RIGHT = _cfg_mod.HAND_JOINT_NAMES_RIGHT
ARM_JOINT_NAMES_RIGHT = _cfg_mod.ARM_JOINT_NAMES_RIGHT
DEFAULT_ALL_POSES = _cfg_mod.DEFAULT_ALL_POSES
ACTUATOR_PARAMS = _cfg_mod.ACTUATOR_PARAMS
HEIGHT_DEFAULT = _cfg_mod.HEIGHT_DEFAULT

# Reward logger
_utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)
from reward_logger import RewardLogger

# ============================================================================
# DIMENSIONS
# ============================================================================

OBS_DIM = 45
ACT_DIM = 7  # 7 right-hand finger joints
FINGER_ACTION_SCALE = 0.1  # rad per action unit
FINGER_ACTION_CLAMP = 0.3  # max delta per step

# Pre-grasp right arm pose (arm extended forward, palm down)
ARM_PREGRASP_POSE = {
    "right_shoulder_pitch_joint": -0.45,
    "right_shoulder_roll_joint": 0.10,
    "right_shoulder_yaw_joint": 0.50,
    "right_elbow_joint": 0.30,
    "right_wrist_roll_joint": -0.15,
    "right_wrist_pitch_joint": 0.90,
    "right_wrist_yaw_joint": -1.40,
}

# ============================================================================
# CURRICULUM
# ============================================================================

MIN_DWELL = 1500

CURRICULUM = [
    {"description": "L0: All shapes, no noise",
     "spawn_noise": 0.00, "threshold": 5.0},
    {"description": "L1: All shapes, 1cm noise",
     "spawn_noise": 0.01, "threshold": 7.0},
    {"description": "L2: All shapes, 2cm noise",
     "spawn_noise": 0.02, "threshold": 9.0},
    {"description": "L3: All shapes, 3cm noise",
     "spawn_noise": 0.03, "threshold": 11.0},
    {"description": "L4: All shapes, 4cm noise (FINAL)",
     "spawn_noise": 0.04, "threshold": None},
]

# Shape constants — always 33/33/33 split across envs
SHAPE_NAMES = ["sphere", "cylinder", "box"]
SHAPE_IDS = {name: i for i, name in enumerate(SHAPE_NAMES)}
NUM_SHAPES = len(SHAPE_NAMES)

# ============================================================================
# REWARD WEIGHTS
# ============================================================================

REWARD_WEIGHTS = {
    "approach": 6.0,           # INCREASED from 3.0 — was 4.6% budget, need fingers to actively reach
    "contact_count": 5.0,
    "contact_force": 0.0,      # DISABLED — was 0.1% budget (dead), 1185N vs 2N target impossible
    "enclose": 1.0,
    "hold_stable": 8.0,
    "finger_closure": 4.0,     # NEW — reward closing fingers toward object, breaks passive exploit
    "action_rate": -0.05,
    "joint_vel": -0.01,
    "alive": 1.0,
}

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DEX3 Grasp Phase A Training")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--experiment_name", type=str, default="g1_grasp_phase_a")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# Print config
# ============================================================================
print("=" * 80)
print("GRASP PHASE A: Fixed-Base, Finger-Only Training")
print(f"USD: {G1_29DOF_USD}")
print(f"Grasp: {OBS_DIM} obs -> {ACT_DIM} act")
print("=" * 80)
for i, lv in enumerate(CURRICULUM):
    print(f"  Level {i}: {lv['description']}")

# ============================================================================
# NETWORK
# ============================================================================

class GraspActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor: 45 -> [256, 128, 64] ELU -> 7
        layers = []
        prev = OBS_DIM
        for h in [256, 128, 64]:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        self.actor = nn.Sequential(*layers)

        # Critic: 45 -> [256, 128, 64] ELU -> 1
        layers_c = []
        prev = OBS_DIM
        for h in [256, 128, 64]:
            layers_c += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers_c.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers_c)

        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))
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

class GraspPPO:
    def __init__(self, net, device, lr=3e-4):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)

    def gae(self, r, v, d, nv):
        adv = torch.zeros_like(r)
        last = 0
        gamma, lam = 0.99, 0.95
        for t in reversed(range(r.shape[0])):
            nxt_v = nv if t == r.shape[0] - 1 else v[t + 1]
            delta = r[t] + gamma * nxt_v * (1 - d[t]) - v[t]
            adv[t] = last = delta + gamma * lam * (1 - d[t]) * last
        ret = adv + v
        return adv, ret

    def update(self, obs, act, adv, ret, old_lp, epochs=5, mb=4096):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        tot_a, tot_c, tot_e, n = 0, 0, 0, 0
        for _ in range(epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                mb_idx = idx[start:start + mb]
                val, lp, ent = self.net.evaluate(obs[mb_idx], act[mb_idx])
                ratio = (lp - old_lp[mb_idx]).exp()
                clipped = ratio.clamp(0.8, 1.2)
                a_loss = -torch.min(ratio * adv[mb_idx], clipped * adv[mb_idx]).mean()
                v_clip = val.detach() + (val - val.detach()).clamp(-0.2, 0.2)
                c_loss = 0.5 * torch.max((val - ret[mb_idx]) ** 2, (v_clip - ret[mb_idx]) ** 2).mean()
                loss = a_loss + 0.5 * c_loss - 0.01 * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                n += 1
        return {"a": tot_a / max(n, 1), "c": tot_c / max(n, 1), "e": tot_e / max(n, 1),
                "lr": self.opt.param_groups[0]["lr"]}


# ============================================================================
# ENVIRONMENT
# ============================================================================

def create_env(num_envs, device):
    # Build init_state with pre-grasp arm pose
    init_poses = dict(DEFAULT_ALL_POSES)
    init_poses.update(ARM_PREGRASP_POSE)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0))

        # Lighting
        light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 1.0)),
        )

        # Table (kinematic box, 0.7m tall)
        table = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.8, 0.02),  # wide, deep, thin top
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.4, 0.3, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.0, 0.69),  # table top at 0.70m (0.69 + 0.01 half-height)
            ),
        )

        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_29DOF_USD,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=True,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    fix_root_link=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, HEIGHT_DEFAULT),
                joint_pos=init_poses,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.90,
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[
                        ".*_hip_yaw_joint", ".*_hip_roll_joint",
                        ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*",
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
                        ".*_hand_index_.*_joint", ".*_hand_middle_.*_joint",
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

        # 3 grasp objects — only 1 active at a time, others teleported underground
        obj_sphere = RigidObjectCfg(
            prim_path="/World/envs/env_.*/ObjSphere",
            spawn=sim_utils.SphereCfg(
                radius=0.06,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, -0.20, 0.75)),
        )
        obj_cylinder = RigidObjectCfg(
            prim_path="/World/envs/env_.*/ObjCylinder",
            spawn=sim_utils.CylinderCfg(
                radius=0.035, height=0.12,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.7, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
        )
        obj_box = RigidObjectCfg(
            prim_path="/World/envs/env_.*/ObjBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.08, 0.08, 0.08),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
        )

        finger_contact = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            filter_prim_paths_expr=[
                "/World/envs/env_.*/ObjSphere",
                "/World/envs/env_.*/ObjCylinder",
                "/World/envs/env_.*/ObjBox",
            ],
            history_length=1,
            update_period=0.0,
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 6.0
        action_space = ACT_DIM
        observation_space = OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=3.0)

    class GraspPhaseAEnv(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            jn = self.robot.joint_names
            print(f"\n[GraspA] Robot joints ({len(jn)}): {jn}")

            # Right hand finger joint indices
            self.finger_idx = []
            for name in HAND_JOINT_NAMES_RIGHT:
                if name in jn:
                    self.finger_idx.append(jn.index(name))
            self.finger_idx = torch.tensor(self.finger_idx, device=self.device)
            print(f"  Finger joints: {len(self.finger_idx)} ({[jn[i] for i in self.finger_idx]})")

            # Right arm joint indices (for holding fixed)
            self.right_arm_idx = []
            for name in ARM_JOINT_NAMES_RIGHT:
                if name in jn:
                    self.right_arm_idx.append(jn.index(name))
            self.right_arm_idx = torch.tensor(self.right_arm_idx, device=self.device)

            # Palm body index
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

            # Fingertip body indices for contact reading
            fingertip_patterns = ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]
            self.fingertip_body_idx = []
            for pat in fingertip_patterns:
                found = False
                for i, name in enumerate(body_names):
                    if pat in name:
                        self.fingertip_body_idx.append(i)
                        found = True
                        break
                if not found:
                    print(f"  [WARN] Fingertip body '{pat}' not found in {body_names}")
            self.fingertip_body_idx = torch.tensor(self.fingertip_body_idx, device=self.device)
            print(f"  Fingertip bodies: {len(self.fingertip_body_idx)} (idx={self.fingertip_body_idx.tolist()})")
            print(f"  All body names: {body_names}")

            # Contact sensor
            self._contact_sensor = self.scene["finger_contact"]
            # Find the body indices within the contact sensor for fingertips
            self._contact_fingertip_idx = []
            contact_body_names = self._contact_sensor.body_names
            for pat in fingertip_patterns:
                for i, name in enumerate(contact_body_names):
                    if pat in name:
                        self._contact_fingertip_idx.append(i)
                        break
            self._contact_fingertip_idx = torch.tensor(self._contact_fingertip_idx, device=self.device)
            print(f"  Contact sensor bodies: {len(contact_body_names)}, fingertip indices: {self._contact_fingertip_idx.tolist()}")

            # Default poses
            self.default_joint_pos = torch.zeros(len(jn), device=self.device)
            for j_name, j_val in init_poses.items():
                if j_name in jn:
                    self.default_joint_pos[jn.index(j_name)] = j_val

            # Finger joint limits
            joint_pos_limits = self.robot.data.soft_joint_pos_limits[0]
            self.finger_lower = joint_pos_limits[self.finger_idx, 0]
            self.finger_upper = joint_pos_limits[self.finger_idx, 1]
            print(f"  Finger limits: lower={self.finger_lower.cpu().numpy()}, upper={self.finger_upper.cpu().numpy()}")

            # State buffers
            self.curr_level = 0
            self.curr_hist = []
            self.env_shape_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.finger_target = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self.prev_action = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self.grasp_counter = torch.zeros(self.num_envs, device=self.device)
            self.obj_init_pos = torch.zeros(self.num_envs, 3, device=self.device)

            # Reward logger
            self.reward_logger = RewardLogger(
                reward_names=list(REWARD_WEIGHTS.keys()),
                reward_weights=REWARD_WEIGHTS,
                num_envs=self.num_envs,
                device=self.device,
            )
            self.reward_extras = {}

            # Initial object placement
            self._sample_object_pos(torch.arange(self.num_envs, device=self.device))

            print(f"\n[GraspA] {self.num_envs} envs, Level {self.curr_level}")
            print(f"  Obs: {OBS_DIM}, Act: {ACT_DIM}")
            print(f"  RewardLogger: {len(REWARD_WEIGHTS)} components")

        @property
        def robot(self):
            return self.scene["robot"]

        @property
        def obj_sphere(self):
            return self.scene["obj_sphere"]

        @property
        def obj_cylinder(self):
            return self.scene["obj_cylinder"]

        @property
        def obj_box(self):
            return self.scene["obj_box"]

        def _get_active_obj_state(self):
            """Return pos [N,3], quat [N,4], vel [N,3] from each env's assigned object."""
            # Stack all 3 objects: [3, N, dim]
            all_pos = torch.stack([
                self.obj_sphere.data.root_pos_w[:, :3],
                self.obj_cylinder.data.root_pos_w[:, :3],
                self.obj_box.data.root_pos_w[:, :3],
            ], dim=0)  # [3, N, 3]
            all_quat = torch.stack([
                self.obj_sphere.data.root_quat_w,
                self.obj_cylinder.data.root_quat_w,
                self.obj_box.data.root_quat_w,
            ], dim=0)  # [3, N, 4]
            all_vel = torch.stack([
                self.obj_sphere.data.root_lin_vel_w,
                self.obj_cylinder.data.root_lin_vel_w,
                self.obj_box.data.root_lin_vel_w,
            ], dim=0)  # [3, N, 3]

            # Select per-env using env_shape_id
            idx = self.env_shape_id  # [N]
            pos = all_pos[idx, torch.arange(self.num_envs, device=self.device)]  # [N, 3]
            quat = all_quat[idx, torch.arange(self.num_envs, device=self.device)]  # [N, 4]
            vel = all_vel[idx, torch.arange(self.num_envs, device=self.device)]  # [N, 3]
            return pos, quat, vel

        # ================================================================
        # OBJECT PLACEMENT
        # ================================================================

        def _sample_object_pos(self, env_ids):
            """Per-env random shape assignment: ~33% sphere, ~33% cylinder, ~33% box."""
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]
            noise = lv["spawn_noise"]

            # Random shape per env (0=sphere, 1=cylinder, 2=box)
            self.env_shape_id[env_ids] = torch.randint(0, NUM_SHAPES, (n,), device=self.device)

            # Get palm position in world frame
            palm_pos = self.robot.data.body_pos_w[env_ids, self.palm_idx]

            # Target position: on table, 12cm below palm — fingers must reach DOWN
            obj_pos = palm_pos.clone()
            obj_pos[:, 2] -= 0.12
            if noise > 0:
                obj_pos[:, :3] += torch.empty(n, 3, device=self.device).uniform_(-noise, noise)

            self.obj_init_pos[env_ids] = obj_pos

            # Poses
            active_pose = torch.zeros(n, 7, device=self.device)
            active_pose[:, :3] = obj_pos
            active_pose[:, 3] = 1.0

            underground = torch.zeros(n, 7, device=self.device)
            underground[:, 2] = -5.0
            underground[:, 3] = 1.0

            vel = torch.zeros(n, 6, device=self.device)

            # Per-env masks
            is_sphere = (self.env_shape_id[env_ids] == 0)
            is_cylinder = (self.env_shape_id[env_ids] == 1)
            is_box = (self.env_shape_id[env_ids] == 2)

            # Place each object: active if selected, underground otherwise
            for shape_id, obj in enumerate([self.obj_sphere, self.obj_cylinder, self.obj_box]):
                mask = (self.env_shape_id[env_ids] == shape_id)
                pose = torch.where(mask.unsqueeze(-1), active_pose, underground)
                obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)
                obj.write_root_com_velocity_to_sim(vel, env_ids=env_ids)

        # ================================================================
        # OBSERVATIONS
        # ================================================================

        def _get_observations(self) -> dict:
            r = self.robot
            obj_pos, obj_quat, obj_vel = self._get_active_obj_state()

            # Finger proprioception
            finger_pos = r.data.joint_pos[:, self.finger_idx]  # [N, 7]
            finger_vel = r.data.joint_vel[:, self.finger_idx]  # [N, 7]

            # Contact sensor
            contact_forces = self._contact_sensor.data.net_forces_w  # [N, num_bodies, 3]
            if len(self._contact_fingertip_idx) > 0:
                ft_forces = contact_forces[:, self._contact_fingertip_idx]  # [N, 3, 3]
                ft_magnitude = ft_forces.norm(dim=-1)  # [N, 3]
                ft_binary = (ft_magnitude > 0.1).float()  # [N, 3]
            else:
                ft_magnitude = torch.zeros(self.num_envs, 3, device=self.device)
                ft_binary = torch.zeros(self.num_envs, 3, device=self.device)

            # Object state relative to palm
            palm_pos = r.data.body_pos_w[:, self.palm_idx]  # [N, 3]

            obj_rel = obj_pos - palm_pos  # [N, 3]
            palm_obj_dist = obj_rel.norm(dim=-1, keepdim=True)  # [N, 1]
            obj_height = obj_pos[:, 2:3]  # [N, 1]

            # Grasp phase
            num_contacts = ft_binary.sum(dim=-1, keepdim=True)
            grasp_phase = torch.zeros(self.num_envs, 1, device=self.device)
            grasp_phase[num_contacts.squeeze(-1) >= 1] = 1.0
            grasp_phase[self.grasp_counter >= 10] = 2.0

            shape_id = self.env_shape_id.float().unsqueeze(-1)  # [N, 1] per-env

            obs = torch.cat([
                finger_pos,          # 7
                finger_vel * 0.1,    # 7 (scaled)
                ft_magnitude * 0.1,  # 3 (scaled)
                ft_binary,           # 3
                obj_rel,             # 3
                obj_quat,            # 4
                obj_vel * 0.5,       # 3 (scaled)
                palm_obj_dist,       # 1
                obj_height,          # 1
                grasp_phase,         # 1
                shape_id,            # 1
                self.prev_action,    # 7
                obj_rel,             # 3 (EE-obj same as palm-obj in Phase A)
                palm_pos[:, 2:3],    # 1 (EE height)
            ], dim=-1)

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        # ================================================================
        # REWARDS
        # ================================================================

        def _get_rewards(self):
            r = self.robot
            obj_pos, _, obj_vel = self._get_active_obj_state()

            # Palm and object positions
            palm_pos = r.data.body_pos_w[:, self.palm_idx]

            # Fingertip positions (world frame)
            fingertip_pos = r.data.body_pos_w[:, self.fingertip_body_idx]  # [N, 3_tips, 3_xyz]
            finger_obj_dist = (fingertip_pos - obj_pos.unsqueeze(1)).norm(dim=-1)  # [N, 3]
            avg_finger_dist = finger_obj_dist.mean(dim=-1)  # [N]

            # Contact
            contact_forces = self._contact_sensor.data.net_forces_w
            if len(self._contact_fingertip_idx) > 0:
                ft_forces = contact_forces[:, self._contact_fingertip_idx]
                ft_magnitude = ft_forces.norm(dim=-1)  # [N, 3]
                ft_binary = (ft_magnitude > 0.1).float()
            else:
                ft_magnitude = torch.zeros(self.num_envs, 3, device=self.device)
                ft_binary = torch.zeros(self.num_envs, 3, device=self.device)

            num_contacts = ft_binary.sum(dim=-1)  # [N], 0-3
            total_force = ft_magnitude.sum(dim=-1)  # [N]

            # Update grasp counter
            grasping = (num_contacts >= 2).float()
            self.grasp_counter = self.grasp_counter * grasping + grasping  # reset if not grasping
            is_grasped = (self.grasp_counter >= 10).float()

            # Object speed
            obj_speed = obj_vel.norm(dim=-1)

            # === REWARD COMPONENTS (raw, unweighted) ===
            r_approach = torch.exp(-10.0 * avg_finger_dist)
            r_contact_count = (num_contacts / 3.0).clamp(0, 1)
            r_contact_force = torch.exp(-torch.abs(total_force - 2.0))  # kept for logging, weight=0
            r_enclose = (num_contacts >= 2).float() * 5.0
            r_hold_stable = is_grasped * torch.exp(-5.0 * obj_speed)

            # Finger closure: reward fingers moving toward closed position
            # finger_pos range: lower_limit (open) to upper_limit (closed for index/middle)
            # Normalize to [0=open, 1=closed]
            finger_pos = r.data.joint_pos[:, self.finger_idx]
            finger_range = self.finger_upper - self.finger_lower  # [7]
            finger_range = finger_range.clamp(min=0.1)  # avoid div by zero
            finger_normalized = (finger_pos - self.finger_lower) / finger_range  # [N, 7], 0=open, 1=closed
            # Average closure across all fingers (higher = more closed)
            r_finger_closure = finger_normalized.mean(dim=-1).clamp(0, 1)  # [N]

            # Penalties
            action_diff = self.actions - self.prev_action
            r_action_rate = (action_diff ** 2).sum(-1)
            finger_vel = r.data.joint_vel[:, self.finger_idx].abs().sum(-1)
            r_joint_vel = finger_vel
            r_alive = torch.ones(self.num_envs, device=self.device)

            # Record to logger
            rl = self.reward_logger
            rl.record("approach", r_approach)
            rl.record("contact_count", r_contact_count)
            rl.record("contact_force", r_contact_force)
            rl.record("enclose", r_enclose)
            rl.record("hold_stable", r_hold_stable)
            rl.record("finger_closure", r_finger_closure)
            rl.record("action_rate", r_action_rate)
            rl.record("joint_vel", r_joint_vel)
            rl.record("alive", r_alive)
            self.reward_extras = rl.get_extras()

            # Weighted sum
            reward = (
                REWARD_WEIGHTS["approach"] * r_approach
                + REWARD_WEIGHTS["contact_count"] * r_contact_count
                + REWARD_WEIGHTS["contact_force"] * r_contact_force
                + REWARD_WEIGHTS["enclose"] * r_enclose
                + REWARD_WEIGHTS["hold_stable"] * r_hold_stable
                + REWARD_WEIGHTS["finger_closure"] * r_finger_closure
                + REWARD_WEIGHTS["action_rate"] * r_action_rate
                + REWARD_WEIGHTS["joint_vel"] * r_joint_vel
                + REWARD_WEIGHTS["alive"] * r_alive
            )
            return reward

        # ================================================================
        # TERMINATION
        # ================================================================

        def _get_dones(self):
            obj_pos, _, _ = self._get_active_obj_state()
            palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]

            # Object fell far below initial position
            obj_fallen = obj_pos[:, 2] < 0.3

            # Object too far from palm
            palm_obj_dist = (obj_pos - palm_pos).norm(dim=-1)
            obj_far = palm_obj_dist > 0.5

            terminated = obj_fallen | obj_far
            time_out = self.episode_length_buf >= self.max_episode_length
            return terminated, time_out

        # ================================================================
        # RESET
        # ================================================================

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            # Reset finger targets to open
            self.finger_target[env_ids] = 0.0
            self.prev_action[env_ids] = 0.0
            self.grasp_counter[env_ids] = 0.0

            # Resample object position
            self._sample_object_pos(env_ids)

        # ================================================================
        # ACTION
        # ================================================================

        def _pre_physics_step(self, actions):
            self.actions = actions.clamp(-FINGER_ACTION_CLAMP, FINGER_ACTION_CLAMP)

        def _apply_action(self):
            # Delta finger control
            self.finger_target += self.actions * FINGER_ACTION_SCALE
            self.finger_target = self.finger_target.clamp(
                self.finger_lower.unsqueeze(0),
                self.finger_upper.unsqueeze(0))

            # Build full joint target: default for all, finger targets for right hand
            target = self.default_joint_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
            target[:, self.finger_idx] = self.finger_target

            self.robot.set_joint_position_target(target)
            self.prev_action = self.actions.clone()

        # ================================================================
        # CURRICULUM
        # ================================================================

        def update_curriculum(self, r):
            self.curr_hist.append(r)
            if len(self.curr_hist) < max(100, MIN_DWELL):
                return
            avg = np.mean(self.curr_hist[-100:])
            thr = CURRICULUM[self.curr_level]["threshold"]
            if thr is not None and avg > thr and self.curr_level < len(CURRICULUM) - 1:
                self.curr_level += 1
                new_lv = CURRICULUM[self.curr_level]
                print(f"\n*** LEVEL UP! Now {self.curr_level}: {new_lv['description']} ***")
                self.curr_hist = []
                self._sample_object_pos(torch.arange(self.num_envs, device=self.device))

    return GraspPhaseAEnv(EnvCfg())


# ============================================================================
# TRAINING LOOP
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)
    net = GraspActorCritic().to(device)

    start_iter = 0
    best_reward = float('-inf')

    if args_cli.checkpoint:
        print(f"\n[Load] Resuming from {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0) + 1
        best_reward = ckpt.get("best_reward", float('-inf'))
        env.curr_level = min(ckpt.get("curriculum_level", 0), len(CURRICULUM) - 1)
        ppo = GraspPPO(net, device)
        if "optimizer" in ckpt:
            ppo.opt.load_state_dict(ckpt["optimizer"])
        print(f"  Resumed: iter={start_iter}, R={best_reward:.2f}, Lv={env.curr_level}")
    else:
        ppo = GraspPPO(net, device)

    # Logging
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\n[Log] {log_dir}")

    # Rollout buffers
    T = 24
    obs_buf = torch.zeros(T, env.num_envs, OBS_DIM, device=device)
    act_buf = torch.zeros(T, env.num_envs, ACT_DIM, device=device)
    rew_buf = torch.zeros(T, env.num_envs, device=device)
    done_buf = torch.zeros(T, env.num_envs, device=device)
    val_buf = torch.zeros(T, env.num_envs, device=device)
    lp_buf = torch.zeros(T, env.num_envs, device=device)

    obs, _ = env.reset()
    obs_t = obs["policy"]
    ep_rewards = torch.zeros(env.num_envs, device=device)
    completed_rewards = []

    print(f"\n{'='*80}")
    print("STARTING GRASP PHASE A TRAINING")
    print(f"  Grasp: {OBS_DIM} obs -> {ACT_DIM} act")
    print(f"  Network: [256,128,64] + ELU, lr=3e-4")
    print(f"  Rewards: {', '.join(f'{k}={v}' for k, v in REWARD_WEIGHTS.items())}")
    print(f"{'='*80}\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        # Std decay: 0.5 -> 0.15
        progress = iteration / args_cli.max_iterations
        std = max(0.5 - 0.35 * progress, 0.15)
        net.log_std.data.fill_(np.log(std))

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
            done_mask = done.bool()
            if done_mask.any():
                for r_val in ep_rewards[done_mask].cpu().numpy():
                    completed_rewards.append(r_val)
                ep_rewards[done_mask] = 0

            obs_t = obs_next

        # Compute returns
        with torch.no_grad():
            nv = net.critic(obs_t).squeeze(-1)
        adv, ret = ppo.gae(rew_buf, val_buf, done_buf, nv)

        # PPO update
        losses = ppo.update(
            obs_buf.reshape(-1, OBS_DIM),
            act_buf.reshape(-1, ACT_DIM),
            adv.reshape(-1),
            ret.reshape(-1),
            lp_buf.reshape(-1),
        )

        mean_reward = rew_buf.mean().item()
        env.update_curriculum(mean_reward)

        # TensorBoard
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

            # Contact metrics
            contact_forces = env._contact_sensor.data.net_forces_w
            if len(env._contact_fingertip_idx) > 0:
                ft_mag = contact_forces[:, env._contact_fingertip_idx].norm(dim=-1)
                avg_contacts = (ft_mag > 0.1).float().sum(dim=-1).mean().item()
                avg_force = ft_mag.sum(dim=-1).mean().item()
            else:
                avg_contacts = 0
                avg_force = 0
            writer.add_scalar("grasp/avg_contacts", avg_contacts, iteration)
            writer.add_scalar("grasp/avg_force", avg_force, iteration)
            writer.add_scalar("grasp/grasp_counter_avg", env.grasp_counter.mean().item(), iteration)

            # Object state
            obj_pos = env._get_active_obj_state()[0]
            palm_pos = env.robot.data.body_pos_w[:, env.palm_idx]
            palm_obj_dist = (obj_pos - palm_pos).norm(dim=-1).mean().item()
            writer.add_scalar("grasp/palm_obj_dist", palm_obj_dist, iteration)
            writer.add_scalar("grasp/obj_height", obj_pos[:, 2].mean().item(), iteration)

            # Reward breakdown
            for key, val in env.reward_extras.items():
                writer.add_scalar(key, val, iteration)

        # Console log
        if iteration % 50 == 0:
            avg_ep = np.mean(completed_rewards[-100:]) if completed_rewards else 0

            contact_forces = env._contact_sensor.data.net_forces_w
            if len(env._contact_fingertip_idx) > 0:
                ft_mag = contact_forces[:, env._contact_fingertip_idx].norm(dim=-1)
                avg_contacts = (ft_mag > 0.1).float().sum(dim=-1).mean().item()
            else:
                avg_contacts = 0

            obj_pos = env._get_active_obj_state()[0]
            palm_pos = env.robot.data.body_pos_w[:, env.palm_idx]
            palm_obj_dist = (obj_pos - palm_pos).norm(dim=-1).mean().item()

            print(f"[{iteration:5d}/{args_cli.max_iterations}] "
                  f"R={mean_reward:.2f} EpR={avg_ep:.2f} "
                  f"contacts={avg_contacts:.1f} dist={palm_obj_dist:.3f} "
                  f"grasp={env.grasp_counter.mean().item():.1f} "
                  f"Lv={env.curr_level} shapes=S{(env.env_shape_id==0).sum().item()}/C{(env.env_shape_id==1).sum().item()}/B{(env.env_shape_id==2).sum().item()} std={np.exp(net.log_std.data.mean().item()):.3f}")

        # Save checkpoints
        if iteration % 500 == 0 and iteration > 0:
            path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                "model": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
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
                "optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
            }, path)

    # Final save
    path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model": net.state_dict(),
        "optimizer": ppo.opt.state_dict(),
        "iteration": args_cli.max_iterations - 1,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
    }, path)
    print(f"\n[Done] Final: {path}, R={best_reward:.2f}, Lv={env.curr_level}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
