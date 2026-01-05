"""
ULC G1 Stage 1: Play/Test Script (v6)
=====================================

Test the trained standing policy with visualization.

Usage:
    ./isaaclab.bat -p play_ulc.py --checkpoint logs/ulc/ulc_g1_stage1_xxx/model_best.pt --num_envs 4
"""

import torch
import torch.nn as nn
import numpy as np
import argparse

HEIGHT_MIN = 0.65
HEIGHT_MAX = 0.85
HEIGHT_DEFAULT = 0.75

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 1 Play")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    return parser.parse_args()

args_cli = parse_args()

# Isaac Lab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

G1_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1/g1_minimal.usd"

print("=" * 60)
print("ULC G1 PLAY v6 - Testing trained model")
print("=" * 60)


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
        super().__init__()

        actor_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            actor_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            critic_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(num_actions))
        self.log_std_min = -3.0
        self.log_std_max = 0.5

    def get_std(self):
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max).exp()

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

    def act(self, obs, deterministic=True):
        action_mean = self.actor(obs)
        if deterministic:
            return action_mean
        std = self.get_std()
        dist = torch.distributions.Normal(action_mean, std)
        return dist.sample()


def create_play_env(num_envs: int, device: str):

    @configclass
    class PlaySceneCfg(InteractiveSceneCfg):
        terrain = sim_utils.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_USD_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                    enable_gyroscopic_forces=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8),
                joint_pos={
                    "left_hip_pitch_joint": -0.1,
                    "right_hip_pitch_joint": -0.1,
                    "left_hip_roll_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.25,
                    "right_knee_joint": 0.25,
                    "left_ankle_pitch_joint": -0.15,
                    "right_ankle_pitch_joint": -0.15,
                    "left_ankle_roll_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "left_elbow_pitch_joint": 0.0,
                    "right_elbow_pitch_joint": 0.0,
                    "left_elbow_roll_joint": 0.0,
                    "right_elbow_roll_joint": 0.0,
                    "torso_joint": 0.0,
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                    stiffness=150.0,
                    damping=10.0,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                    stiffness=50.0,
                    damping=5.0,
                ),
                "torso": ImplicitActuatorCfg(
                    joint_names_expr=["torso_joint"],
                    stiffness=100.0,
                    damping=10.0,
                ),
            },
        )

    @configclass
    class PlayEnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 30.0

        # New API: use spaces instead of num_*
        action_space = 12
        observation_space = 46
        state_space = 0

        sim = sim_utils.SimulationCfg(
            dt=1/200,
            render_interval=decimation,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        scene = PlaySceneCfg(num_envs=num_envs, env_spacing=2.5)

    class PlayEnv(DirectRLEnv):
        cfg: PlayEnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            joint_names = self.robot.joint_names
            leg_joint_names = [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
            ]

            self.leg_indices = []
            for name in leg_joint_names:
                if name in joint_names:
                    self.leg_indices.append(joint_names.index(name))
            self.leg_indices = torch.tensor(self.leg_indices, device=self.device)

            self.default_leg_positions = torch.tensor([
                -0.1, -0.1, 0.0, 0.0, 0.0, 0.0,
                0.25, 0.25, -0.15, -0.15, 0.0, 0.0,
            ], device=self.device)

            self.target_heights = torch.rand(self.num_envs, device=self.device) * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN
            self.spawn_positions = torch.zeros(self.num_envs, 2, device=self.device)
            self.previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self.action_scale = 0.5

            print(f"[ULC_G1_Play] Leg joints: {len(self.leg_indices)}")

        @property
        def robot(self):
            return self.scene["robot"]

        def _setup_scene(self):
            self.cfg.scene.robot.spawn.func(
                self.cfg.scene.robot.spawn,
                self.cfg.scene.robot.prim_path.replace(".*", "0"),
                self.cfg.scene.robot,
            )
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

        def _pre_physics_step(self, actions):
            robot = self.robot
            targets = robot.data.default_joint_pos.clone()
            leg_targets = self.default_leg_positions.unsqueeze(0) + actions * self.action_scale
            targets[:, self.leg_indices] = leg_targets
            robot.set_joint_position_target(targets)
            self.previous_actions = actions.clone()

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            robot = self.robot
            base_quat = robot.data.root_quat_w
            base_lin_vel = robot.data.root_lin_vel_w
            base_ang_vel = robot.data.root_ang_vel_w

            from isaaclab.utils.math import quat_apply_inverse

            base_lin_vel_b = quat_apply_inverse(base_quat, base_lin_vel)
            base_ang_vel_b = quat_apply_inverse(base_quat, base_ang_vel)

            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)

            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            leg_pos = joint_pos[:, self.leg_indices]
            leg_vel = joint_vel[:, self.leg_indices]

            height_cmd = self.target_heights.unsqueeze(-1)

            obs = torch.cat([
                base_lin_vel_b, base_ang_vel_b, proj_gravity,
                leg_pos, leg_vel, height_cmd, self.previous_actions,
            ], dim=-1)

            obs = torch.clamp(obs, -100.0, 100.0)
            obs = torch.nan_to_num(obs, nan=0.0)

            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            robot = self.robot
            base_pos = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w

            height = base_pos[:, 2]
            too_low = height < 0.3
            too_high = height > 1.2

            from isaaclab.utils.math import quat_apply_inverse
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            too_tilted = (torch.abs(proj_gravity[:, 0]) > 0.7) | (torch.abs(proj_gravity[:, 1]) > 0.7)

            xy_drift = torch.norm(base_pos[:, :2] - self.spawn_positions, dim=-1)
            too_far = xy_drift > 1.0

            terminated = too_low | too_high | too_tilted | too_far
            time_out = self.episode_length_buf >= self.max_episode_length

            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if len(env_ids) == 0:
                return

            robot = self.robot

            pos = torch.tensor([0.0, 0.0, 0.8], device=self.device).expand(len(env_ids), -1).clone()
            pos[:, 2] = 0.8

            quat = torch.zeros(len(env_ids), 4, device=self.device)
            quat[:, 3] = 1.0

            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

            default_pos = robot.data.default_joint_pos[env_ids]
            robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)

            self.spawn_positions[env_ids] = pos[:, :2].clone()
            self.target_heights[env_ids] = torch.rand(len(env_ids), device=self.device) * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN
            self.previous_actions[env_ids] = 0.0

    cfg = PlayEnvCfg()
    cfg.scene.num_envs = num_envs
    env = PlayEnv(cfg)

    return env


def play():
    device = "cuda:0"

    env = create_play_env(args_cli.num_envs, device)

    actor_critic = ActorCriticNetwork(46, 12).to(device)

    print(f"\n[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

    print(f"[INFO] Model loaded from iteration {checkpoint.get('iteration', 'unknown')}")
    print(f"[INFO] Best/Mean reward: {checkpoint.get('best_reward', 'unknown')}")
    print(f"[INFO] Running {args_cli.steps} steps...")
    print("=" * 60)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_terminated = 0

    for step in range(args_cli.steps):
        with torch.no_grad():
            actions = actor_critic.act(obs, deterministic=True)

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        total_terminated += terminated.sum().item()

        if step % 100 == 0:
            robot = env.robot
            base_pos = robot.data.root_pos_w
            height = base_pos[:, 2].mean().item()
            target = env.target_heights.mean().item()

            xy_drift = torch.norm(base_pos[:, :2] - env.spawn_positions, dim=-1).mean().item()

            print(f"Step {step:4d} | Height: {height:.3f}m | Target: {target:.3f}m | Drift: {xy_drift:.3f}m | Terminated: {int(terminated.sum())}")

    print("=" * 60)
    print(f"Completed {args_cli.steps} steps")
    print(f"Total episodes (resets): {total_terminated}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    play()
    simulation_app.close()