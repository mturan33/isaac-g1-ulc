#!/usr/bin/env python3
"""
ULC G1 Stage 5 Training - FULL MECHANICAL WORKSPACE
====================================================

Stage 4 checkpoint'inden başlayıp %100 mekanik limite genişletir.

KULLANIM:
./isaaclab.bat -p train_ulc_stage_5.py ^
    --stage4_checkpoint logs/ulc/ulc_g1_stage4_.../model_best.pt ^
    --num_envs 4096 --headless
"""

import argparse
import os
from datetime import datetime

# ============================================================
# ARGUMENT PARSING (Isaac Lab uyumlu)
# ============================================================

parser = argparse.ArgumentParser(description="ULC G1 Stage 5 - Full Workspace Training")
parser.add_argument("--stage4_checkpoint", type=str, default=None,
                    help="Path to Stage 4 checkpoint (optional, starts from scratch if not provided)")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=6000,
                    help="Maximum training iterations")

# Isaac Lab launcher - headless dahil tüm argümanları ekler
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================
# IMPORTS (simulation app başladıktan sonra)
# ============================================================

import torch
import torch.nn as nn
from collections import deque

# Isaac Lab imports
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg

# ============================================================
# G1 FULL MECHANICAL LIMITS (Unitree Official)
# ============================================================

G1_FULL_LIMITS = {
    "shoulder_pitch": 2.6,  # ±149°
    "shoulder_roll": 1.6,  # ±92°
    "shoulder_yaw": 2.6,  # ±149°
    "elbow": 1.6,  # ±92°
    "hip_pitch": 1.57,  # ±90° - Squat!
    "knee": 2.0,  # 115° - Squat!
    "waist_pitch": 0.52,  # ±30° - Eğilme!
}

# ============================================================
# STAGE 5 CURRICULUM
# ============================================================

STAGE5_CURRICULUM = {
    0: {
        "name": "Stage4 Baseline",
        "vx_range": (-0.3, 0.8),
        "height_range": (0.60, 0.80),
        "shoulder_pitch_range": 0.8,
        "elbow_range": 0.8,
        "torso_pitch_range": 0.3,
        "threshold": 26.0,
        "min_iterations": 300,
    },
    1: {
        "name": "Extended Arms (60%)",
        "vx_range": (-0.3, 0.8),
        "height_range": (0.55, 0.80),
        "shoulder_pitch_range": 1.5,
        "elbow_range": 1.0,
        "torso_pitch_range": 0.35,
        "threshold": 25.0,
        "min_iterations": 400,
    },
    2: {
        "name": "Light Squat + Arms 80%",
        "vx_range": (-0.4, 0.9),
        "height_range": (0.45, 0.82),
        "shoulder_pitch_range": 2.0,
        "elbow_range": 1.3,
        "torso_pitch_range": 0.40,
        "threshold": 24.0,
        "min_iterations": 500,
    },
    3: {
        "name": "Deep Squat + Full Arms",
        "vx_range": (-0.5, 1.0),
        "height_range": (0.38, 0.85),
        "shoulder_pitch_range": 2.6,
        "elbow_range": 1.6,
        "torso_pitch_range": 0.45,
        "threshold": 23.0,
        "min_iterations": 600,
    },
    4: {
        "name": "FULL WORKSPACE",
        "vx_range": (-0.5, 1.0),
        "height_range": (0.35, 0.85),
        "shoulder_pitch_range": 2.6,
        "elbow_range": 1.6,
        "torso_pitch_range": 0.52,
        "threshold": None,
        "min_iterations": None,
    },
}


# ============================================================
# ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):
    """Actor-Critic with LayerNorm."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Actor
        actor_layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ELU(),
            ])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ELU(),
            ])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        return self.actor(obs)

    def get_value(self, obs):
        return self.critic(obs)

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return value, log_prob, entropy

    def act(self, obs, deterministic=False):
        mean = self.actor(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()


# ============================================================
# CURRICULUM MANAGER WITH COMMAND SAMPLING
# ============================================================

class Stage5CurriculumManager:
    """Curriculum manager that handles command sampling with expanded ranges."""

    def __init__(self, env, curriculum_config):
        self.env = env
        self.curriculum = curriculum_config
        self.current_level = 0
        self.level_iterations = 0
        self.reward_history = deque(maxlen=100)
        self.device = env.device

        # Apply initial level
        self._apply_level(0)

    def _apply_level(self, level):
        """Apply curriculum level settings."""
        cfg = self.curriculum[level]
        self.current_config = cfg

        self.vx_range = cfg.get("vx_range", (-0.5, 1.0))
        self.height_range = cfg.get("height_range", (0.35, 0.85))
        self.shoulder_pitch_range = cfg.get("shoulder_pitch_range", 2.6)
        self.elbow_range = cfg.get("elbow_range", 1.6)
        self.torso_pitch_range = cfg.get("torso_pitch_range", 0.52)

        print(f"[Stage5] Level {level}: {cfg['name']}")
        print(f"  Arms: shoulder={self.shoulder_pitch_range:.1f}, elbow={self.elbow_range:.1f}")
        print(f"  Height: {self.height_range}")

    def sample_commands(self):
        """Sample commands with current curriculum ranges."""
        num_envs = self.env.num_envs
        device = self.device

        # Velocity
        self.env.velocity_commands[:, 0] = torch.empty(num_envs, device=device).uniform_(*self.vx_range)
        self.env.velocity_commands[:, 1] = torch.empty(num_envs, device=device).uniform_(-0.3, 0.3)
        self.env.velocity_commands[:, 2] = torch.empty(num_envs, device=device).uniform_(-0.5, 0.5)

        # Height (squat!)
        self.env.height_command[:, 0] = torch.empty(num_envs, device=device).uniform_(*self.height_range)

        # Torso
        self.env.torso_commands[:, 0] = torch.empty(num_envs, device=device).uniform_(
            -0.3, 0.3)  # roll
        self.env.torso_commands[:, 1] = torch.empty(num_envs, device=device).uniform_(
            -self.torso_pitch_range, self.torso_pitch_range)  # pitch
        self.env.torso_commands[:, 2] = torch.empty(num_envs, device=device).uniform_(
            -0.3, 0.3)  # yaw

        # Arms - FULL RANGE!
        sp = self.shoulder_pitch_range
        el = self.elbow_range

        # Left arm (0-4)
        self.env.arm_commands[:, 0] = torch.empty(num_envs, device=device).uniform_(-sp, sp)
        self.env.arm_commands[:, 1] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)
        self.env.arm_commands[:, 2] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)
        self.env.arm_commands[:, 3] = torch.empty(num_envs, device=device).uniform_(-el, el)
        self.env.arm_commands[:, 4] = torch.empty(num_envs, device=device).uniform_(-0.5, 0.5)

        # Right arm (5-9)
        self.env.arm_commands[:, 5] = torch.empty(num_envs, device=device).uniform_(-sp, sp)
        self.env.arm_commands[:, 6] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)
        self.env.arm_commands[:, 7] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)
        self.env.arm_commands[:, 8] = torch.empty(num_envs, device=device).uniform_(-el, el)
        self.env.arm_commands[:, 9] = torch.empty(num_envs, device=device).uniform_(-0.5, 0.5)

    def resample_for_reset(self, env_ids):
        """Resample commands for reset environments."""
        if len(env_ids) == 0:
            return

        num = len(env_ids)
        device = self.device
        sp = self.shoulder_pitch_range
        el = self.elbow_range

        self.env.velocity_commands[env_ids, 0] = torch.empty(num, device=device).uniform_(*self.vx_range)
        self.env.height_command[env_ids, 0] = torch.empty(num, device=device).uniform_(*self.height_range)
        self.env.torso_commands[env_ids, 1] = torch.empty(num, device=device).uniform_(
            -self.torso_pitch_range, self.torso_pitch_range)
        self.env.arm_commands[env_ids, 0] = torch.empty(num, device=device).uniform_(-sp, sp)
        self.env.arm_commands[env_ids, 3] = torch.empty(num, device=device).uniform_(-el, el)
        self.env.arm_commands[env_ids, 5] = torch.empty(num, device=device).uniform_(-sp, sp)
        self.env.arm_commands[env_ids, 8] = torch.empty(num, device=device).uniform_(-el, el)

    def update(self, iteration, mean_reward):
        """Check for level promotion."""
        self.level_iterations += 1
        self.reward_history.append(mean_reward)

        if len(self.reward_history) < 50:
            return False

        cfg = self.curriculum[self.current_level]
        threshold = cfg.get("threshold")
        min_iters = cfg.get("min_iterations", 0)

        if threshold is None:
            return False

        avg_reward = sum(self.reward_history) / len(self.reward_history)

        if self.level_iterations >= min_iters and avg_reward >= threshold:
            next_level = self.current_level + 1
            if next_level in self.curriculum:
                self.current_level = next_level
                self._apply_level(next_level)
                self.level_iterations = 0
                self.reward_history.clear()
                return True
        return False

    def get_workspace_pct(self):
        return (self.shoulder_pitch_range / 2.6) * 100


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:
    """Simple PPO trainer."""

    def __init__(self, env, policy, curriculum_mgr, device="cuda:0"):
        self.env = env
        self.policy = policy
        self.curriculum = curriculum_mgr
        self.device = device

        # Hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 1.0
        self.max_grad_norm = 1.0
        self.num_epochs = 5
        self.mini_batch_size = 4096
        self.num_steps = 24

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

    def collect_rollout(self):
        """Collect experience."""
        obs_list, act_list, rew_list, done_list, val_list, logp_list = [], [], [], [], [], []

        obs_dict = self.env.reset()
        obs = obs_dict["policy"]

        for _ in range(self.num_steps):
            with torch.no_grad():
                action = self.policy.act(obs)
                value = self.policy.get_value(obs).squeeze(-1)
                mean = self.policy.actor(obs)
                std = torch.exp(self.policy.log_std)
                log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)

            obs_list.append(obs)
            act_list.append(action)
            val_list.append(value)
            logp_list.append(log_prob)

            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            obs = obs_dict["policy"]
            done = terminated | truncated

            rew_list.append(reward)
            done_list.append(done)

            # Resample commands for reset environments
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.curriculum.resample_for_reset(reset_ids)

        return {
            "obs": torch.stack(obs_list),
            "actions": torch.stack(act_list),
            "rewards": torch.stack(rew_list),
            "dones": torch.stack(done_list),
            "values": torch.stack(val_list),
            "log_probs": torch.stack(logp_list),
            "last_obs": obs,
        }

    def compute_returns(self, rollout):
        """Compute GAE."""
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]

        with torch.no_grad():
            last_value = self.policy.get_value(rollout["last_obs"]).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(rewards.shape[0])):
            next_val = last_value if t == rewards.shape[0] - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (~dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (~dones[t]) * last_gae

        returns = advantages + values
        return returns, advantages

    def update(self, rollout, returns, advantages):
        """PPO update."""
        obs = rollout["obs"].view(-1, rollout["obs"].shape[-1])
        actions = rollout["actions"].view(-1, rollout["actions"].shape[-1])
        old_log_probs = rollout["log_probs"].view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = (advantages.view(-1) - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        n_updates = 0

        for _ in range(self.num_epochs):
            perm = torch.randperm(obs.shape[0])
            for start in range(0, obs.shape[0], self.mini_batch_size):
                idx = perm[start:start + self.mini_batch_size]

                values, log_probs, entropy = self.policy.evaluate(obs[idx], actions[idx])

                ratio = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratio * advantages_flat[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_flat[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - returns_flat[idx]) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / n_updates

    def train_one_iteration(self):
        """Single training iteration."""
        rollout = self.collect_rollout()
        returns, advantages = self.compute_returns(rollout)
        loss = self.update(rollout, returns, advantages)

        return {
            "loss": loss,
            "mean_reward": rollout["rewards"].mean().item(),
        }


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("🔥 ULC G1 STAGE 5 - FULL MECHANICAL WORKSPACE 🔥")
    print("=" * 70)
    print(f"Stage 4 Checkpoint: {args.stage4_checkpoint}")
    print(f"Environments: {args.num_envs}")
    print(f"Max Iterations: {args.max_iterations}")
    print()

    print("G1 FULL LIMITS (Unitree Official):")
    print("  shoulder_pitch: ±2.6 rad (±149°)")
    print("  elbow: ±1.6 rad (±92°)")
    print("  hip_pitch: ±1.57 rad (±90°) - Squat!")
    print("  knee: 2.0 rad (115°) - Squat!")
    print()

    # Load Stage 4 checkpoint if provided
    stage4_ckpt = None
    if args.stage4_checkpoint:
        print("[INFO] Loading Stage 4 checkpoint...")
        if not os.path.exists(args.stage4_checkpoint):
            print(f"[WARNING] Checkpoint not found: {args.stage4_checkpoint}")
            print("[INFO] Starting with random initialization")
        else:
            stage4_ckpt = torch.load(args.stage4_checkpoint, map_location="cuda:0", weights_only=True)
            print("[INFO] Stage 4 checkpoint loaded ✓")
            print(f"[INFO] Checkpoint keys: {list(stage4_ckpt.keys())}")
    else:
        print("[INFO] No checkpoint provided, starting from scratch")

    # Create environment
    print(f"[INFO] Creating environment with {args.num_envs} envs...")

    env_cfg = ULC_G1_Stage4_EnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # IMPORTANT: Update termination for squat
    env_cfg.termination["base_height_min"] = 0.25

    # Add required fields for newer Isaac Lab
    import gymnasium as gym
    obs_dim = env_cfg.num_observations
    act_dim = env_cfg.num_actions
    env_cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,))
    env_cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,))

    env = ULC_G1_Env(cfg=env_cfg)
    env.current_stage = 4  # Set to Stage 4 mode

    obs_dim = env_cfg.num_observations
    action_dim = env_cfg.num_actions
    print(f"[INFO] Obs: {obs_dim}, Actions: {action_dim}")

    # Create policy
    policy = ActorCritic(obs_dim, action_dim).to("cuda:0")

    # Load Stage 4 weights if checkpoint was provided
    if stage4_ckpt is not None:
        print("[INFO] Loading Stage 4 weights...")

        # Check what keys are in the checkpoint
        if "model_state_dict" in stage4_ckpt:
            state_dict = stage4_ckpt["model_state_dict"]
        elif "model" in stage4_ckpt:
            state_dict = stage4_ckpt["model"]
        elif "actor" in stage4_ckpt:
            # RSL-RL format - need to reconstruct
            print("[INFO] RSL-RL checkpoint format detected")
            state_dict = None  # Will handle separately
        else:
            # Maybe the checkpoint IS the state dict
            state_dict = stage4_ckpt

        if state_dict is not None:
            try:
                policy.load_state_dict(state_dict)
                print("[INFO] Weights loaded successfully ✓")
            except Exception as e:
                print(f"[WARNING] Could not load weights directly: {e}")
                print("[INFO] Starting with random initialization")
        else:
            print("[INFO] Could not extract weights, starting fresh")
    else:
        print("[INFO] Starting with random initialization")

    # Create curriculum manager
    curriculum = Stage5CurriculumManager(env, STAGE5_CURRICULUM)
    curriculum.sample_commands()

    # Create trainer
    trainer = PPOTrainer(env, policy, curriculum, device="cuda:0")

    # Log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage5_full_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log dir: {log_dir}")

    # Training loop
    print("\n[INFO] Starting training...")
    best_reward = -float('inf')

    for iteration in range(args.max_iterations):
        stats = trainer.train_one_iteration()
        mean_reward = stats["mean_reward"]

        # Curriculum update
        promoted = curriculum.update(iteration, mean_reward)
        if promoted:
            pct = curriculum.get_workspace_pct()
            print(f"\n{'=' * 60}")
            print(f"🎯 LEVEL UP! Level {curriculum.current_level}: {curriculum.current_config['name']}")
            print(f"   Workspace: {pct:.0f}%")
            print(f"{'=' * 60}\n")

        # Track best
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": curriculum.current_level,
            }, os.path.join(log_dir, "model_best.pt"))

        # Logging
        if iteration % 50 == 0:
            pct = curriculum.get_workspace_pct()
            print(f"Iter {iteration:5d} | Reward: {mean_reward:7.2f} | Best: {best_reward:7.2f} | "
                  f"Level: {curriculum.current_level} ({pct:.0f}%)")

        # Periodic save
        if iteration % 500 == 0 and iteration > 0:
            torch.save({
                "model_state_dict": policy.state_dict(),
                "iteration": iteration,
                "curriculum_level": curriculum.current_level,
            }, os.path.join(log_dir, f"model_{iteration}.pt"))

    # Final save
    torch.save({
        "model_state_dict": policy.state_dict(),
        "iteration": args.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": curriculum.current_level,
    }, os.path.join(log_dir, "model_final.pt"))

    print("\n" + "=" * 70)
    print("🎉 STAGE 5 COMPLETE!")
    print(f"Final Level: {curriculum.current_level}")
    print(f"Workspace: {curriculum.get_workspace_pct():.0f}%")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Saved: {log_dir}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()