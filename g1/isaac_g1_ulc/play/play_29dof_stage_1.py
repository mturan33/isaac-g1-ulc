"""
29DoF G1 Stage 1: Standing — Play/Evaluation Script
====================================================
Gorsel test icin. Checkpoint yukleyip robotu izle.

MODES:
- calm: Itme yok, sadece ayakta durma
- push: Periyodik itme kuvvetleri
- stress: Agresif itme + kucuk initial perturbation
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time

# ============================================================================
# IMPORTS & CONFIG (same importlib trick as train script)
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

# Obs/Act dims (must match training)
OBS_DIM = 3 + 3 + 3 + 12 + 12 + 3 + 3 + 1 + 3 + 2 + 15 + 3 + 3  # = 66
ACT_DIM = NUM_LOCO_JOINTS  # 15

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="29DoF G1 Stage 1: Play")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--mode", type=str, default="push", choices=["calm", "push", "stress"],
                        help="calm: no push, push: periodic push, stress: aggressive push")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max simulation steps")
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

# ============================================================================
# MODE CONFIGS
# ============================================================================

MODE_CONFIGS = {
    "calm": {"push_force": (0, 0), "push_interval": (9999, 9999)},
    "push": {"push_force": (0, 30), "push_interval": (100, 300)},
    "stress": {"push_force": (10, 50), "push_interval": (50, 150)},
}

mode_cfg = MODE_CONFIGS[args_cli.mode]

print("=" * 80)
print(f"29DoF G1 STAGE 1: PLAY — Mode: {args_cli.mode}")
print(f"Checkpoint: {args_cli.checkpoint}")
print(f"Push force: {mode_cfg['push_force']}, interval: {mode_cfg['push_interval']}")
print(f"Obs: {OBS_DIM}, Act: {ACT_DIM}")
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
# QUATERNION UTILS
# ============================================================================

def quat_to_euler_xyz(quat):
    """xyzw quaternion to roll, pitch, yaw."""
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


# ============================================================================
# ENVIRONMENT (simplified from training, no curriculum)
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
                    enabled_self_collisions=False,
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
        episode_length_s = 30.0  # Longer for play
        action_space = ACT_DIM
        observation_space = OBS_DIM
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=3.0)

    class Env(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
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

            # Default poses
            self.default_loco = torch.tensor(DEFAULT_LOCO_LIST, device=self.device, dtype=torch.float32)
            self.default_arm = torch.tensor(DEFAULT_ARM_LIST, device=self.device, dtype=torch.float32)
            self.default_hand = torch.tensor(DEFAULT_HAND_LIST, device=self.device, dtype=torch.float32)

            # Action scales
            leg_scales = [LEG_ACTION_SCALE] * 12
            waist_scales = [WAIST_ACTION_SCALE] * 3
            self.action_scales = torch.tensor(leg_scales + waist_scales, device=self.device, dtype=torch.float32)

            # State
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
            self._prev_jvel = None

            # Push system
            self.push_timer = torch.randint(
                mode_cfg["push_interval"][0], mode_cfg["push_interval"][1],
                (self.num_envs,), device=self.device)
            self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.total_pushes = 0
            self.total_falls = 0

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
                mag = torch.rand(n, 1, device=self.device) * (
                    mode_cfg["push_force"][1] - mode_cfg["push_force"][0]) + mode_cfg["push_force"][0]
                force = force * mag
                forces[ids, 0] = force

                self.push_timer[ids] = torch.randint(
                    mode_cfg["push_interval"][0], mode_cfg["push_interval"][1], (n,), device=self.device)
                self.step_count[ids] = 0
                self.total_pushes += n

            self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        def _pre_physics_step(self, act):
            self.actions = act.clone()
            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.loco_idx] = self.default_loco + act * self.action_scales
            tgt[:, self.arm_idx] = self.default_arm
            tgt[:, self.hand_idx] = self.default_hand
            self.robot.set_joint_position_target(tgt)

            self._prev_act = self.prev_act.clone()
            self.prev_act = act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
            self._apply_push()

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
                lv, av, g,
                jp_leg, jv_leg, jp_waist, jv_waist,
                self.height_cmd[:, None], self.vel_cmd, gait,
                self.prev_act, torso_euler, self.torso_cmd,
            ], dim=-1)
            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            pos = self.robot.data.root_pos_w
            q = self.robot.data.root_quat_w
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))
            fallen = pos[:, 2] < 0.4
            tilted = g[:, 2] < 0.5
            terminated = fallen | tilted
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
            default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)
            self.robot.write_root_pose_to_sim(torch.cat([root_pos, default_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            self.prev_act[env_ids] = 0
            self._prev_act[env_ids] = 0
            self.phase[env_ids] = torch.rand(n, device=self.device)

    return Env(EnvCfg())


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda"
    env = create_env(args_cli.num_envs, device)

    # Load model
    net = ActorCritic(OBS_DIM, ACT_DIM).to(device)
    ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()

    level = ckpt.get("curriculum_level", "?")
    best_r = ckpt.get("best_reward", "?")
    iteration = ckpt.get("iteration", "?")
    print(f"\n[Load] Checkpoint loaded:")
    print(f"  Iteration: {iteration}, Best reward: {best_r}, Level: {level}")

    # Run
    obs, _ = env.reset()
    obs_t = obs["policy"]

    print(f"\n[Play] Baslatiliyor... Mode: {args_cli.mode}, Max steps: {args_cli.max_steps}")
    print("-" * 60)

    for step in range(args_cli.max_steps):
        with torch.no_grad():
            action = net.act(obs_t, det=True)  # Deterministic!

        obs_dict, _, _, _, _ = env.step(action)
        obs_t = obs_dict["policy"]

        # Periodic status
        if step % 200 == 0 and step > 0:
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            q = env.robot.data.root_quat_w
            g = quat_apply_inverse(q, torch.tensor([0, 0, -1.], device=device).expand(env.num_envs, -1))
            tilt = torch.acos(g[:, 2].clamp(-1, 1)).mean().item() * 180 / np.pi
            print(f"  [Step {step:5d}] H={height:.3f}m, Tilt={tilt:.1f}deg, "
                  f"Falls={env.total_falls}, Pushes={env.total_pushes}")

    # Final stats
    print("-" * 60)
    height = env.robot.data.root_pos_w[:, 2].mean().item()
    print(f"\n[Sonuc] {args_cli.max_steps} step tamamlandi")
    print(f"  Final height: {height:.3f}m (target: {HEIGHT_DEFAULT}m)")
    print(f"  Total falls: {env.total_falls}")
    print(f"  Total pushes: {env.total_pushes}")
    print(f"  Mode: {args_cli.mode}")

    # Keep window open
    print("\n[Play] Simulasyon acik. Ctrl+C ile kapat.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
