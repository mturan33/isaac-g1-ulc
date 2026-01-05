"""
ULC G1 Training Script - Stage 1: Standing (v5 - FIXED)
========================================================

Fixes in v5:
- Best model tracking based on mean_reward (not episode_reward)
- std decay to reduce exploration over time
- Randomized height commands (0.65-0.85m)
- Better default joint positions for G1
- Higher action scale (1.0)
- Proper episode reward tracking

Usage:
    cd IsaacLab
    ./isaaclab.bat -p <path>/train_ulc_v5.py --num_envs 4096 --headless

Author: Mehmet Turan Yardımcı
Date: January 2026
Project: ULC-VLM for G1 Humanoid
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
from datetime import datetime
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ULC G1 Training - Stage 1 Standing v5")
parser.add_argument("--task", type=str, default="ULC-G1-Standing-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage1", help="Experiment name")
parser.add_argument("--save_interval", type=int, default=200, help="Checkpoint save interval")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_obs_tensor(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


# =============================================================================
# OBSERVATION NORMALIZATION
# =============================================================================

class EmpiricalNormalization(nn.Module):
    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        self.running_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.running_var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon),
            min=-5.0, max=5.0
        )


# =============================================================================
# ACTOR-CRITIC WITH STD DECAY
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic with learnable but bounded std."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list = [512, 256, 128]):
        super().__init__()

        # Actor
        actor_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable log_std - start at 0.0 (std=1.0), will decay
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # Std bounds
        self.log_std_min = -2.0  # std_min = 0.135
        self.log_std_max = 0.5   # std_max = 1.65

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def get_std(self):
        """Get clamped std."""
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max).exp()

    def forward(self, obs):
        mean = self.actor(obs)
        std = self.get_std()
        value = self.critic(obs)
        return mean, std, value.squeeze(-1)

    def act(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    def act_inference(self, obs):
        """Deterministic action for inference."""
        mean = self.actor(obs)
        return mean


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, num_obs, num_actions, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.step = 0

        self.obs = torch.zeros(num_steps, num_envs, num_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

    def add(self, obs, actions, rewards, dones, values, log_probs):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.step = (self.step + 1) % self.num_steps

    def compute_gae(self, last_values, gamma=0.99, gae_lambda=0.95):
        last_gae = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        indices = torch.randperm(self.num_steps * self.num_envs, device=self.device)

        obs_flat = self.obs.view(-1, self.obs.shape[-1])
        actions_flat = self.actions.view(-1, self.actions.shape[-1])
        log_probs_flat = self.log_probs.view(-1)
        advantages_flat = self.advantages.view(-1)
        returns_flat = self.returns.view(-1)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                "obs": obs_flat[batch_indices],
                "actions": actions_flat[batch_indices],
                "old_log_probs": log_probs_flat[batch_indices],
                "advantages": advantages_flat[batch_indices],
                "returns": returns_flat[batch_indices],
            }

    def reset(self):
        self.step = 0


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_ulc_g1_env(num_envs: int, device: str = "cuda"):
    """Create ULC G1 environment with proper config."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab.actuators import ImplicitActuatorCfg

    G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

    NUM_ACTIONS = 12
    NUM_OBSERVATIONS = 46

    # Target height range for randomization
    HEIGHT_MIN = 0.65
    HEIGHT_MAX = 0.85
    HEIGHT_DEFAULT = 0.75

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
                joint_pos={
                    # Default standing pose - slightly bent knees
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
                    # Arms and torso
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
                    stiffness=150.0,  # Increased for better standing
                    damping=10.0,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*shoulder.*", ".*elbow.*", "torso_joint"],
                    stiffness=50.0,
                    damping=5.0,
                ),
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=[".*five.*", ".*three.*", ".*zero.*", ".*six.*", ".*four.*", ".*one.*", ".*two.*"],
                    stiffness=10.0,
                    damping=1.0,
                ),
            },
        )

    @configclass
    class ULC_G1_Stage1_EnvCfg(DirectRLEnvCfg):
        episode_length_s = 10.0
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

        scene: ULC_G1_SceneCfg = ULC_G1_SceneCfg(num_envs=4096, env_spacing=2.5)

    class ULC_G1_Stage1_Env(DirectRLEnv):
        cfg: ULC_G1_Stage1_EnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            # Per-environment target height (randomized on reset)
            self.target_heights = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT

            # Action buffers
            self.previous_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
            self._prev_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
            self._prev_joint_vel = None

            # Episode tracking
            self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
            self.episode_lengths = torch.zeros(self.num_envs, device=self.device)

            self._setup_joint_indices()

            print(f"[ULC_G1_Stage1] Initialized with {self.num_envs} envs")
            print(f"[ULC_G1_Stage1] Height range: {HEIGHT_MIN}-{HEIGHT_MAX}m")
            print(f"[ULC_G1_Stage1] Observations: {NUM_OBSERVATIONS}, Actions: {NUM_ACTIONS}")

        def _setup_joint_indices(self):
            """Find leg joint indices - proper ordering for G1."""
            robot = self.scene["robot"]
            joint_names = robot.data.joint_names

            # G1 leg joints in proper order
            leg_joint_patterns = [
                "left_hip_pitch", "right_hip_pitch",
                "left_hip_roll", "right_hip_roll",
                "left_hip_yaw", "right_hip_yaw",
                "left_knee", "right_knee",
                "left_ankle_pitch", "right_ankle_pitch",
                "left_ankle_roll", "right_ankle_roll",
            ]

            self.leg_indices = []
            for pattern in leg_joint_patterns:
                for i, name in enumerate(joint_names):
                    if pattern in name.lower():
                        self.leg_indices.append(i)
                        break

            if len(self.leg_indices) < 12:
                print(f"[WARNING] Only found {len(self.leg_indices)} leg joints, padding...")
                while len(self.leg_indices) < 12:
                    self.leg_indices.append(self.leg_indices[-1] if self.leg_indices else 0)

            self.leg_indices = torch.tensor(self.leg_indices[:12], device=self.device, dtype=torch.long)

            # Store default leg positions for reference
            self.default_leg_positions = torch.tensor([
                -0.1, -0.1,  # hip pitch
                0.0, 0.0,    # hip roll
                0.0, 0.0,    # hip yaw
                0.25, 0.25,  # knee
                -0.15, -0.15,  # ankle pitch
                0.0, 0.0,    # ankle roll
            ], device=self.device)

            print(f"[ULC_G1_Stage1] Leg joints: {len(self.leg_indices)}")
            print(f"[ULC_G1_Stage1] Joint names: {[joint_names[i] for i in self.leg_indices.tolist()]}")

        def _setup_scene(self):
            from isaaclab.assets import Articulation
            self.robot = Articulation(self.cfg.scene.robot)
            self.scene.articulations["robot"] = self.robot
            self.scene.clone_environments(copy_from_source=False)

        def _pre_physics_step(self, actions: torch.Tensor):
            self.actions = torch.clamp(actions, -1.0, 1.0)

            # Higher action scale for more responsive control
            action_scale = 1.0

            # Start from default positions and add actions
            targets = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)

            # Apply actions relative to default standing pose
            leg_targets = self.default_leg_positions.unsqueeze(0) + actions * action_scale

            num_leg_joints = min(len(self.leg_indices), actions.shape[1])
            targets[:, self.leg_indices[:num_leg_joints]] = leg_targets[:, :num_leg_joints]

            self.robot.set_joint_position_target(targets)
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

            # Per-environment height command
            height_cmd = self.target_heights.unsqueeze(-1)

            obs = torch.cat([
                base_lin_vel_b,          # 3
                base_ang_vel_b,          # 3
                proj_gravity,            # 3
                leg_pos,                 # 12
                leg_vel,                 # 12
                height_cmd,              # 1
                self.previous_actions,   # 12
            ], dim=-1)

            obs = torch.clamp(obs, -100.0, 100.0)
            obs = torch.nan_to_num(obs, nan=0.0)

            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            robot = self.robot

            base_pos = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w
            base_lin_vel = robot.data.root_lin_vel_w
            base_ang_vel = robot.data.root_ang_vel_w
            joint_vel = robot.data.joint_vel

            # Height reward (per-env target)
            height = base_pos[:, 2]
            height_error = torch.abs(height - self.target_heights)
            r_height = torch.exp(-10.0 * height_error ** 2)

            # Orientation reward (stay upright)
            from isaaclab.utils.math import quat_apply_inverse
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            orientation_error = torch.sum(proj_gravity[:, :2] ** 2, dim=-1)
            r_orientation = torch.exp(-5.0 * orientation_error)

            # XY velocity penalty (stay still)
            xy_vel = torch.norm(base_lin_vel[:, :2], dim=-1)
            r_xy_velocity = torch.exp(-2.0 * xy_vel ** 2)

            # Angular velocity penalty (don't rotate)
            ang_vel_norm = torch.norm(base_ang_vel, dim=-1)
            r_ang_velocity = torch.exp(-0.5 * ang_vel_norm ** 2)

            # Joint acceleration penalty
            if self._prev_joint_vel is not None:
                joint_acc = joint_vel - self._prev_joint_vel
                r_joint_acc = -0.0005 * torch.sum(joint_acc ** 2, dim=-1)
            else:
                r_joint_acc = torch.zeros(self.num_envs, device=self.device)
            self._prev_joint_vel = joint_vel.clone()

            # Action rate penalty
            action_diff = self.actions - self._prev_actions
            r_action_rate = -0.005 * torch.sum(action_diff ** 2, dim=-1)
            self._prev_actions = self.actions.clone()

            # Total reward
            reward = (
                5.0 * r_height +
                3.0 * r_orientation +
                3.0 * r_xy_velocity +
                1.0 * r_ang_velocity +
                r_joint_acc +
                r_action_rate
            )

            # Track episode rewards
            self.episode_rewards += reward
            self.episode_lengths += 1

            # Log components
            self.extras["Episode_Reward/height_tracking"] = r_height
            self.extras["Episode_Reward/orientation"] = r_orientation
            self.extras["Episode_Reward/xy_velocity"] = r_xy_velocity

            return reward

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

            terminated = too_low | too_high | too_tilted
            time_out = self.episode_length_buf >= self.max_episode_length

            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if len(env_ids) == 0:
                return

            robot = self.robot

            # Reset position with small random variation
            pos = torch.tensor([0.0, 0.0, 0.8], device=self.device).expand(len(env_ids), -1).clone()
            pos = pos + torch.randn_like(pos) * 0.02
            pos[:, 2] = 0.8

            quat = torch.zeros(len(env_ids), 4, device=self.device)
            quat[:, 3] = 1.0

            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

            # Reset joints to default standing pose
            default_pos = robot.data.default_joint_pos[env_ids]
            robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)

            # Randomize target height for each reset environment
            self.target_heights[env_ids] = torch.rand(len(env_ids), device=self.device) * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN

            # Reset buffers
            self.previous_actions[env_ids] = 0.0
            self._prev_actions[env_ids] = 0.0
            self.episode_rewards[env_ids] = 0.0
            self.episode_lengths[env_ids] = 0.0

        def get_episode_stats(self, terminated, truncated):
            """Get episode statistics for completed episodes."""
            done = terminated | truncated
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)

            if len(done_indices) > 0:
                ep_rewards = self.episode_rewards[done_indices].clone()
                ep_lengths = self.episode_lengths[done_indices].clone()
                return ep_rewards.cpu().numpy(), ep_lengths.cpu().numpy()
            return np.array([]), np.array([])

    cfg = ULC_G1_Stage1_EnvCfg()
    env = ULC_G1_Stage1_Env(cfg)
    return env, NUM_OBSERVATIONS, NUM_ACTIONS


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train():
    print("=" * 80)
    print("ULC G1 TRAINING - STAGE 1: STANDING (v5 - FIXED)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[INFO] Creating environment with {args_cli.num_envs} envs...")
    env, num_obs, num_actions = create_ulc_g1_env(args_cli.num_envs, device)

    num_envs = env.num_envs

    print(f"[INFO] Num Envs: {num_envs}")
    print(f"[INFO] Observations: {num_obs}, Actions: {num_actions}")

    # Hyperparameters
    num_steps_per_rollout = 24
    num_learning_epochs = 5
    num_mini_batches = 4
    clip_param = 0.2
    value_loss_coef = 1.0
    entropy_coef = 0.005  # Reduced entropy for less exploration
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    max_grad_norm = 1.0

    # std decay parameters
    std_decay_rate = 0.999  # Decay per iteration

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", f"{args_cli.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"[INFO] Logging to: {log_dir}")

    # Networks
    actor_critic = ActorCriticNetwork(num_obs, num_actions).to(device)
    obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args_cli.max_iterations
    )

    # Buffer
    buffer = RolloutBuffer(num_steps_per_rollout, num_envs, num_obs, num_actions, device)

    # Training state
    best_reward = float("-inf")
    total_timesteps = 0
    episode_rewards_history = deque(maxlen=100)
    episode_lengths_history = deque(maxlen=100)

    # Resume from checkpoint
    start_iteration = 0
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location=device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        start_iteration = checkpoint.get("iteration", 0)
        best_reward = checkpoint.get("best_reward", float("-inf"))

    print(f"[INFO] Starting training from iteration {start_iteration}")
    print("=" * 80)

    # Reset environment
    obs_dict, _ = env.reset()
    obs = get_obs_tensor(obs_dict)

    training_start = time.time()

    for iteration in range(start_iteration, args_cli.max_iterations):
        iter_start = time.time()

        # Apply std decay
        with torch.no_grad():
            actor_critic.log_std.data *= std_decay_rate
            actor_critic.log_std.data.clamp_(actor_critic.log_std_min, actor_critic.log_std_max)

        buffer.reset()
        rollout_rewards = []

        # Collect rollout
        with torch.no_grad():
            for step in range(num_steps_per_rollout):
                obs_norm = obs_normalizer.normalize(obs)
                actions, log_probs, values = actor_critic.act(obs_norm)

                next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
                next_obs = get_obs_tensor(next_obs_dict)
                dones = terminated | truncated

                buffer.add(obs, actions, rewards, dones.float(), values, log_probs)
                obs_normalizer.update(obs)

                # Track episode completions
                ep_rewards, ep_lengths = env.get_episode_stats(terminated, truncated)
                for r in ep_rewards:
                    episode_rewards_history.append(r)
                for l in ep_lengths:
                    episode_lengths_history.append(l)

                rollout_rewards.append(rewards.mean().item())
                total_timesteps += num_envs
                obs = next_obs

        # Compute GAE
        with torch.no_grad():
            obs_norm = obs_normalizer.normalize(obs)
            _, _, last_values = actor_critic(obs_norm)
        buffer.compute_gae(last_values, gamma, gae_lambda)

        # PPO Update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = (num_steps_per_rollout * num_envs) // num_mini_batches

        for epoch in range(num_learning_epochs):
            for batch in buffer.get_batches(batch_size):
                obs_norm = obs_normalizer.normalize(batch["obs"])
                new_log_probs, values, entropy = actor_critic.evaluate(obs_norm, batch["actions"])

                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * F.mse_loss(values, batch["returns"])

                loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
                optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        scheduler.step()

        # Logging
        iter_time = time.time() - iter_start
        mean_reward = np.mean(rollout_rewards)
        mean_ep_reward = np.mean(episode_rewards_history) if episode_rewards_history else 0
        mean_ep_length = np.mean(episode_lengths_history) if episode_lengths_history else 0
        steps_per_sec = (num_steps_per_rollout * num_envs) / iter_time
        mean_std = actor_critic.get_std().mean().item()

        writer.add_scalar("Loss/surrogate", total_actor_loss / num_updates, iteration)
        writer.add_scalar("Loss/value_function", total_critic_loss / num_updates, iteration)
        writer.add_scalar("Loss/entropy", total_entropy / num_updates, iteration)
        writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        writer.add_scalar("Train/mean_episode_reward", mean_ep_reward, iteration)
        writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)
        writer.add_scalar("Perf/total_fps", steps_per_sec, iteration)
        writer.flush()  # Force write to disk

        # Console output
        if iteration % 10 == 0:
            elapsed = time.time() - training_start
            remaining = (args_cli.max_iterations - iteration) * iter_time

            print("#" * 80)
            print(f"{'Learning iteration ' + str(iteration) + '/' + str(args_cli.max_iterations):^80}")
            print(f"{'Computation: ' + f'{int(steps_per_sec)} steps/s':^80}")
            print(f"{'Mean reward:':>35} {mean_reward:.2f}")
            print(f"{'Mean episode reward:':>35} {mean_ep_reward:.2f}")
            print(f"{'Mean episode length:':>35} {mean_ep_length:.1f}")
            print(f"{'Mean std:':>35} {mean_std:.3f}")
            print(f"{'Actor loss:':>35} {total_actor_loss / num_updates:.4f}")
            print(f"{'Critic loss:':>35} {total_critic_loss / num_updates:.4f}")
            print("-" * 80)
            print(f"{'Total timesteps:':>35} {total_timesteps}")
            print(f"{'Elapsed:':>35} {format_time(elapsed)}")
            print(f"{'ETA:':>35} {format_time(remaining)}")
            print("#" * 80)
            print()

        # Save checkpoints - use MEAN_REWARD for best model (not episode reward)
        checkpoint_data = {
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_normalizer": obs_normalizer.state_dict(),
            "iteration": iteration,
            "best_reward": best_reward,
            "mean_reward": mean_reward,
        }

        # Best model based on mean_reward (not episode_reward which may be 0)
        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint_data["best_reward"] = best_reward
            torch.save(checkpoint_data, os.path.join(log_dir, "model_best.pt"))
            print(f"[BEST] New best model saved! Reward: {best_reward:.2f}")

        if iteration % args_cli.save_interval == 0 and iteration > 0:
            torch.save(checkpoint_data, os.path.join(log_dir, f"model_{iteration}.pt"))
            print(f"[CHECKPOINT] Saved model_{iteration}.pt")

    # Final save
    torch.save(checkpoint_data, os.path.join(log_dir, "model_final.pt"))

    print("\n" + "=" * 80)
    print(f"{'TRAINING COMPLETE':^80}")
    print(f"{'Best reward: ' + f'{best_reward:.2f}':^80}")
    print(f"{'Log dir: ' + log_dir:^80}")
    print("=" * 80)

    writer.close()
    env.close()


if __name__ == "__main__":
    train()
    simulation_app.close()