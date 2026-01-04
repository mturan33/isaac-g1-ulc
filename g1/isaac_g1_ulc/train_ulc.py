"""
ULC G1 Training Script - Stage 1: Standing
==========================================

Unified Loco-Manipulation Controller için eğitim script'i.
İlk olarak ayakta durma (standing) öğretiyoruz.

Usage:
    cd IsaacLab
    ./isaaclab.bat -p <path>/train_ulc.py --num_envs 4096 --headless

    # Visual mode (debug):
    ./isaaclab.bat -p <path>/train_ulc.py --num_envs 64

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

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments BEFORE importing torch (Isaac Lab requirement)
parser = argparse.ArgumentParser(description="ULC G1 Training - Stage 1 Standing")
parser.add_argument("--task", type=str, default="ULC-G1-Standing-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=2000, help="Max training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage1", help="Experiment name")
parser.add_argument("--save_interval", type=int, default=200, help="Checkpoint save interval")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

# Enable TF32 for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_time(seconds):
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_obs_tensor(obs):
    """Extract observation tensor from dict."""
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


# =============================================================================
# OBSERVATION NORMALIZATION
# =============================================================================

class EmpiricalNormalization(nn.Module):
    """Welford's online normalization."""

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
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.running_var = M2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon),
            min=-5.0, max=5.0
        )


# =============================================================================
# ACTOR-CRITIC NETWORK
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic with ELU activation."""

    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128], init_noise_std=1.0):
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

        # Learnable std
        self.log_std = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std))

        self._init_weights()

    def _init_weights(self):
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(obs).squeeze(-1)
        return actions, log_probs, values

    def evaluate(self, obs, actions):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs).squeeze(-1)
        return log_probs, values, entropy

    def act_inference(self, obs):
        return self.actor(obs)


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """On-policy rollout storage."""

    def __init__(self, num_envs, num_steps, num_obs, num_actions, device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0

        # Allocate buffers
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

        # Normalize advantages
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

    # IMPORTANT: Actuator configs are in isaaclab.actuators module in Isaac Lab 2.3+
    from isaaclab.actuators import ImplicitActuatorCfg

    # G1 USD path
    G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

    # Constants for observation/action space
    NUM_ACTIONS = 12  # Legs only for Stage 1
    NUM_OBSERVATIONS = 46  # 3+3+3+12+12+1+12

    @configclass
    class ULC_G1_SceneCfg(InteractiveSceneCfg):
        """Scene config."""

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
    class ULC_G1_Stage1_EnvCfg(DirectRLEnvCfg):
        """Stage 1 config - Standing."""

        episode_length_s = 10.0
        decimation = 4
        num_actions = NUM_ACTIONS
        num_observations = NUM_OBSERVATIONS
        num_states = 0

        # IMPORTANT: Isaac Lab 2.3+ requires observation_space and action_space
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

        scene: ULC_G1_SceneCfg = ULC_G1_SceneCfg(num_envs=num_envs, env_spacing=2.5)

    # Custom environment class
    class ULC_G1_Stage1_Env(DirectRLEnv):
        """Stage 1: Standing environment."""

        cfg: ULC_G1_Stage1_EnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            self.target_height = 0.75
            self.previous_actions = torch.zeros(self.num_envs, NUM_ACTIONS, device=self.device)
            self._prev_joint_vel = None

            # Find leg joint indices
            self._setup_joint_indices()

            print(f"[ULC_G1_Stage1] Initialized with {self.num_envs} envs")
            print(f"[ULC_G1_Stage1] Observations: {NUM_OBSERVATIONS}, Actions: {NUM_ACTIONS}")

        def _setup_joint_indices(self):
            """Find leg joint indices."""
            robot = self.scene["robot"]
            joint_names = robot.data.joint_names

            self.leg_indices = []
            for i, name in enumerate(joint_names):
                if any(x in name.lower() for x in ["hip", "knee", "ankle"]):
                    self.leg_indices.append(i)

            # Ensure we have exactly 12 leg joints or pad/trim
            if len(self.leg_indices) >= 12:
                self.leg_indices = self.leg_indices[:12]
            else:
                # Pad with first available joints if needed
                while len(self.leg_indices) < 12:
                    self.leg_indices.append(self.leg_indices[-1] if self.leg_indices else 0)

            self.leg_indices = torch.tensor(self.leg_indices, device=self.device, dtype=torch.long)
            print(f"[ULC_G1_Stage1] Leg joints: {len(self.leg_indices)}")
            print(f"[ULC_G1_Stage1] Joint names: {joint_names}")

        def _setup_scene(self):
            """Setup the scene - robot is spawned via scene config."""
            from isaaclab.assets import Articulation

            # Robot is already spawned by scene config, just get reference
            self.robot = Articulation(self.cfg.scene.robot)
            self.scene.articulations["robot"] = self.robot

            # Clone environments
            self.scene.clone_environments(copy_from_source=False)

            # NOTE: Light is already created by scene config (dome_light)
            # Do NOT create it again here!

        def _pre_physics_step(self, actions: torch.Tensor):
            self.actions = torch.clamp(actions, -1.0, 1.0)

            action_scale = 0.5
            targets = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)

            # Apply actions to leg joints only
            num_leg_joints = min(len(self.leg_indices), actions.shape[1])
            targets[:, self.leg_indices[:num_leg_joints]] = actions[:, :num_leg_joints] * action_scale

            self.robot.set_joint_position_target(targets)
            self.previous_actions = actions.clone()

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            robot = self.robot

            # Base state
            base_quat = robot.data.root_quat_w
            base_lin_vel = robot.data.root_lin_vel_w
            base_ang_vel = robot.data.root_ang_vel_w

            # Transform to base frame
            from isaaclab.utils.math import quat_apply_inverse
            base_lin_vel_b = quat_apply_inverse(base_quat, base_lin_vel)
            base_ang_vel_b = quat_apply_inverse(base_quat, base_ang_vel)

            # Projected gravity
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)

            # Joint state (legs only)
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel

            leg_pos = joint_pos[:, self.leg_indices]
            leg_vel = joint_vel[:, self.leg_indices]

            # Height command
            height_cmd = torch.ones(self.num_envs, 1, device=self.device) * self.target_height

            # Build observation
            obs = torch.cat([
                base_lin_vel_b,                    # 3
                base_ang_vel_b,                    # 3
                proj_gravity,                       # 3
                leg_pos,                            # 12
                leg_vel,                            # 12
                height_cmd,                         # 1
                self.previous_actions,              # 12
            ], dim=-1)

            obs = torch.clamp(obs, -100.0, 100.0)
            obs = torch.nan_to_num(obs, nan=0.0)

            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            robot = self.robot

            base_pos = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w
            base_lin_vel = robot.data.root_lin_vel_w
            joint_vel = robot.data.joint_vel

            # Height reward
            height = base_pos[:, 2]
            height_error = torch.abs(height - self.target_height)
            r_height = torch.exp(-10.0 * height_error ** 2)

            # Orientation reward
            from isaaclab.utils.math import quat_apply_inverse
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            orientation_error = torch.sum(proj_gravity[:, :2] ** 2, dim=-1)
            r_orientation = torch.exp(-5.0 * orientation_error)

            # Velocity penalty (should be standing still)
            xy_vel = torch.norm(base_lin_vel[:, :2], dim=-1)
            r_velocity = torch.exp(-2.0 * xy_vel ** 2)

            # Joint acceleration penalty
            if self._prev_joint_vel is not None:
                joint_acc = joint_vel - self._prev_joint_vel
                r_joint_acc = -0.0005 * torch.sum(joint_acc ** 2, dim=-1)
            else:
                r_joint_acc = torch.zeros(self.num_envs, device=self.device)
            self._prev_joint_vel = joint_vel.clone()

            # Action rate penalty
            if hasattr(self, '_prev_actions'):
                action_diff = self.actions - self._prev_actions
                r_action_rate = -0.01 * torch.sum(action_diff ** 2, dim=-1)
            else:
                r_action_rate = torch.zeros(self.num_envs, device=self.device)
            self._prev_actions = self.actions.clone()

            # Total reward
            reward = (
                5.0 * r_height +
                3.0 * r_orientation +
                4.0 * r_velocity +
                r_joint_acc +
                r_action_rate
            )

            # Log components
            self.extras["Episode_Reward/height_tracking"] = r_height
            self.extras["Episode_Reward/orientation"] = r_orientation
            self.extras["Episode_Reward/velocity"] = r_velocity

            return reward

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            robot = self.robot

            base_pos = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w

            # Height check
            height = base_pos[:, 2]
            too_low = height < 0.3
            too_high = height > 1.2

            # Orientation check
            from isaaclab.utils.math import quat_apply_inverse
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            too_tilted = (torch.abs(proj_gravity[:, 0]) > 0.7) | (torch.abs(proj_gravity[:, 1]) > 0.7)

            terminated = too_low | too_high | too_tilted
            time_out = self.episode_length_buf >= self.max_episode_length

            # Termination penalty
            self.extras["Episode_Termination/fell"] = terminated

            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if len(env_ids) == 0:
                return

            robot = self.robot

            # Reset position
            pos = torch.tensor([0.0, 0.0, 0.8], device=self.device).expand(len(env_ids), -1).clone()
            pos = pos + torch.randn_like(pos) * 0.02
            pos[:, 2] = 0.8

            # Reset orientation (identity)
            quat = torch.zeros(len(env_ids), 4, device=self.device)
            quat[:, 3] = 1.0

            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

            # Reset joints
            default_pos = robot.data.default_joint_pos[env_ids]
            robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)

            # Reset buffers
            self.previous_actions[env_ids] = 0.0
            if hasattr(self, '_prev_actions'):
                self._prev_actions[env_ids] = 0.0

    # Create and return environment
    cfg = ULC_G1_Stage1_EnvCfg()
    env = ULC_G1_Stage1_Env(cfg)
    return env, NUM_OBSERVATIONS, NUM_ACTIONS


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train():
    """Main training function."""

    print("=" * 80)
    print("ULC G1 TRAINING - STAGE 1: STANDING")
    print("=" * 80)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
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
    entropy_coef = 0.01
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 3e-4
    max_grad_norm = 1.0

    # Create networks
    actor_critic = ActorCriticNetwork(num_obs, num_actions).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args_cli.max_iterations, eta_min=1e-5)

    # Observation normalizer
    obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)

    # Rollout buffer
    buffer = RolloutBuffer(num_envs, num_steps_per_rollout, num_obs, num_actions, device)

    # Load checkpoint if provided
    start_iteration = 0
    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location=device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        start_iteration = checkpoint.get("iteration", 0)

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", f"{args_cli.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] Starting training from iteration {start_iteration}")
    print("=" * 80)

    # Reset environment
    obs_dict, _ = env.reset()
    obs = get_obs_tensor(obs_dict)

    # Verify observation shape
    print(f"[DEBUG] Observation shape: {obs.shape}")
    assert obs.shape[-1] == num_obs, f"Observation mismatch: {obs.shape[-1]} != {num_obs}"

    # Training stats
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_reward = float('-inf')
    training_start = time.time()
    total_timesteps = 0

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    for iteration in range(start_iteration, args_cli.max_iterations):
        iter_start = time.time()

        # Collect rollouts
        buffer.reset()
        rollout_rewards = []

        with torch.no_grad():
            for step in range(num_steps_per_rollout):
                # Normalize observations
                obs_norm = obs_normalizer.normalize(obs)

                # Get actions
                actions, log_probs, values = actor_critic(obs_norm)

                # Step environment
                next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
                next_obs = get_obs_tensor(next_obs_dict)
                dones = terminated | truncated

                # Store transition
                buffer.add(obs, actions, rewards, dones.float(), values, log_probs)

                # Update normalizer
                obs_normalizer.update(obs)

                # Track episodes
                if "episode" in infos:
                    for ep_info in infos["episode"]:
                        episode_rewards.append(ep_info["r"])
                        episode_lengths.append(ep_info["l"])

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

                # Actor loss
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * batch["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = 0.5 * F.mse_loss(values, batch["returns"])

                # Total loss
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
        mean_ep_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_ep_length = np.mean(episode_lengths) if episode_lengths else 0
        steps_per_sec = (num_steps_per_rollout * num_envs) / iter_time
        mean_std = actor_critic.log_std.exp().mean().item()

        writer.add_scalar("Loss/surrogate", total_actor_loss / num_updates, iteration)
        writer.add_scalar("Loss/value_function", total_critic_loss / num_updates, iteration)
        writer.add_scalar("Loss/entropy", total_entropy / num_updates, iteration)
        writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        writer.add_scalar("Train/mean_episode_reward", mean_ep_reward, iteration)
        writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)
        writer.add_scalar("Perf/total_fps", steps_per_sec, iteration)

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

        # Save checkpoints
        checkpoint_data = {
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_normalizer": obs_normalizer.state_dict(),
            "iteration": iteration,
        }

        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            torch.save(checkpoint_data, os.path.join(log_dir, "model_best.pt"))

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