"""
ULC G1 Play Script - Test trained model
=======================================

Usage:
    cd IsaacLab
    ./isaaclab.bat -p <path>/play_ulc.py --checkpoint <path>/model_best.pt --num_envs 4
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play ULC G1")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Constants
NUM_ACTIONS = 12
NUM_OBSERVATIONS = 46
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"


class ActorCriticNetwork(nn.Module):
    """Same architecture as training."""

    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
        super().__init__()

        actor_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.ones(num_actions))

    def act_inference(self, obs):
        return self.actor(obs)


class EmpiricalNormalization(nn.Module):
    def __init__(self, input_shape, epsilon=1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.epsilon = epsilon

    def normalize(self, x):
        return torch.clamp(
            (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon),
            min=-5.0, max=5.0
        )


@configclass
class ULC_G1_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=100.0,
                damping=5.0,
            ),
        },
    )


@configclass
class ULC_G1_PlayEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    num_actions = NUM_ACTIONS
    num_observations = NUM_OBSERVATIONS
    num_states = 0

    observation_space = gym.spaces.Dict({
        "policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(NUM_OBSERVATIONS,))
    })
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_ACTIONS,))
    state_space = None

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: ULC_G1_SceneCfg = ULC_G1_SceneCfg(num_envs=4, env_spacing=2.5)


class ULC_G1_PlayEnv(DirectRLEnv):
    cfg: ULC_G1_PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.target_height = 0.75
        self.previous_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
        self._setup_joint_indices()

    def _setup_joint_indices(self):
        robot = self.scene["robot"]
        joint_names = robot.data.joint_names

        self.leg_indices = []
        for i, name in enumerate(joint_names):
            if any(x in name.lower() for x in ["hip", "knee", "ankle"]):
                self.leg_indices.append(i)

        if len(self.leg_indices) >= 12:
            self.leg_indices = self.leg_indices[:12]
        else:
            while len(self.leg_indices) < 12:
                self.leg_indices.append(self.leg_indices[-1] if self.leg_indices else 0)

        self.leg_indices = torch.tensor(self.leg_indices, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        from isaaclab.assets import Articulation
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        action_scale = 0.5
        targets = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        num_leg_joints = min(len(self.leg_indices), actions.shape[1])
        targets[:, self.leg_indices[:num_leg_joints]] = actions[:, :num_leg_joints] * action_scale
        self.robot.set_joint_position_target(targets)
        self.previous_actions = actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self):
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

        height_cmd = torch.ones(self.num_envs, 1, device=self.device) * self.target_height

        obs = torch.cat([
            base_lin_vel_b, base_ang_vel_b, proj_gravity,
            leg_pos, leg_vel, height_cmd, self.previous_actions,
        ], dim=-1)

        return {"policy": torch.clamp(obs, -100.0, 100.0)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        robot = self.robot
        height = robot.data.root_pos_w[:, 2]
        too_low = height < 0.3
        too_high = height > 1.2

        from isaaclab.utils.math import quat_apply_inverse
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
        too_tilted = (torch.abs(proj_gravity[:, 0]) > 0.7) | (torch.abs(proj_gravity[:, 1]) > 0.7)

        terminated = too_low | too_high | too_tilted
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        robot = self.robot
        pos = torch.tensor([0.0, 0.0, 0.8], device=self.device).expand(len(env_ids), -1).clone()
        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 3] = 1.0

        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

        default_pos = robot.data.default_joint_pos[env_ids]
        robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)
        self.previous_actions[env_ids] = 0.0


def play():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ULC G1 PLAY - Testing trained model")
    print("=" * 60)

    # Create environment
    cfg = ULC_G1_PlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = ULC_G1_PlayEnv(cfg)

    # Load model
    print(f"\n[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)

    actor_critic = ActorCriticNetwork(NUM_OBSERVATIONS, NUM_ACTIONS).to(device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()

    obs_normalizer = EmpiricalNormalization((NUM_OBSERVATIONS,)).to(device)
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    print(f"[INFO] Model loaded from iteration {checkpoint.get('iteration', 'unknown')}")
    print(f"[INFO] Running {args_cli.num_steps} steps...")
    print("=" * 60)

    # Run
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_reward = 0
    episode_count = 0

    for step in range(args_cli.num_steps):
        with torch.no_grad():
            obs_norm = obs_normalizer.normalize(obs)
            actions = actor_critic.act_inference(obs_norm)

        next_obs_dict, rewards, terminated, truncated, _ = env.step(actions)
        obs = next_obs_dict["policy"]

        # Print status every 100 steps
        if step % 100 == 0:
            height = env.robot.data.root_pos_w[:, 2].mean().item()
            print(f"Step {step:4d} | Height: {height:.3f}m | Terminated: {terminated.sum().item()}")

        # Count resets
        if terminated.any() or truncated.any():
            episode_count += (terminated | truncated).sum().item()

    print("=" * 60)
    print(f"Completed {args_cli.num_steps} steps")
    print(f"Episodes (resets): {episode_count}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    play()
    simulation_app.close()