"""
G1 Stage 6: Reactive Balance Training
=====================================

ARCHITECTURE:
- Arm Policy: FROZEN (Stage 5 - reaching expert)
- Balance Loco Policy: TRAINABLE (learns to balance while arm moves)

The balance policy learns:
1. Keep CoM over support polygon
2. Squat when arm reaches down
3. Lean forward when arm reaches forward
4. Smooth weight shifts during arm movement

USAGE:
./isaaclab.bat -p .../train_reactive_balance.py \
    --num_envs 2048 \
    --max_iterations 15000 \
    --arm_checkpoint logs/ulc/g1_arm_reach_.../model_19998.pt \
    --headless

Optional: Start from Stage 3 weights for leg initialization
./isaaclab.bat -p .../train_reactive_balance.py \
    --num_envs 2048 \
    --arm_checkpoint logs/ulc/g1_arm_reach_.../model_19998.pt \
    --loco_init logs/ulc/ulc_g1_stage3_.../model_best.pt \
    --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# ============================================================================
# ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="G1 Reactive Balance Training")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=15000, help="Max training iterations")
parser.add_argument("--arm_checkpoint", type=str, required=True,
                    help="Path to Stage 5 (arm) checkpoint - REQUIRED, will be FROZEN")
parser.add_argument("--loco_init", type=str, default=None,
                    help="Optional: Initialize legs from Stage 3 checkpoint")
parser.add_argument("--resume", type=str, default=None, help="Resume training")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_reactive_balance_env import G1ReactiveBalanceEnv, G1ReactiveBalanceEnvCfg

from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# NETWORKS
# ============================================================================

class FrozenArmPolicy(nn.Module):
    """
    Arm policy from Stage 5 (FROZEN).
    Observation: 29 dims (arm-specific)
    Output: 5 dims (right arm actions)
    """
    def __init__(self, num_obs=29, num_act=5, hidden=[256, 128, 64]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class BalanceLocoPolicy(nn.Module):
    """
    Balance Locomotion Policy (TRAINABLE).

    Learns to keep balance while the arm moves.
    Sees: base state, leg joints, commands, CoM, arm state
    Outputs: 12 leg actions

    Key features:
    - CoM-aware (sees center of mass position and velocity)
    - Arm-aware (sees arm joint positions and target)
    - Learns adaptive posture (squat, lean)
    """
    def __init__(self, num_obs=72, num_act=12, hidden=[512, 256, 128]):
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
        critic_layers = []
        prev = num_obs
        for h in hidden:
            critic_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Log std for exploration
        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, obs: torch.Tensor):
        return self.actor(obs), self.critic(obs)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, _ = self.forward(obs)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, value = self.forward(obs)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return value.squeeze(-1), dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_frozen_arm_policy(checkpoint_path: str, device: str) -> FrozenArmPolicy:
    """Load Stage 5 arm policy and freeze it."""
    print(f"\n[Load] Loading FROZEN arm policy: {checkpoint_path}")

    policy = FrozenArmPolicy().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Extract actor weights
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key] = value

    if actor_state:
        policy.load_state_dict(actor_state, strict=False)
        print(f"  ✓ Loaded {len(actor_state)} weights")
        print(f"  ✓ All parameters FROZEN")
    else:
        raise ValueError("No actor weights found in arm checkpoint!")

    return policy


def load_loco_init(checkpoint_path: str, policy: BalanceLocoPolicy, device: str):
    """Optionally initialize balance policy from Stage 3."""
    print(f"\n[Init] Initializing from Stage 3: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "actor_critic" in ckpt:
        state_dict = ckpt["actor_critic"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Stage 3 has different observation size, so we do partial transfer
    # Only transfer first layer weights that correspond to shared observations

    loaded = 0
    for name, param in policy.named_parameters():
        if name in state_dict:
            try:
                # Check shape compatibility
                if param.shape == state_dict[name].shape:
                    param.data.copy_(state_dict[name])
                    loaded += 1
            except:
                pass

    print(f"  ✓ Transferred {loaded} parameters")
    return policy


# ============================================================================
# ARM OBSERVATION EXTRACTOR
# ============================================================================

def extract_arm_obs(full_obs: torch.Tensor) -> torch.Tensor:
    """
    Extract arm-relevant observations for frozen arm policy.

    From full_obs (72 dims):
    - Base state for context: 0-9 (9)
    - Arm joints are at: 53-63 (arm_pos=5, arm_vel=5)
    - Target is at: 63-64 (just z for now, need full target)

    For Stage 5 arm policy (29 dims expected):
    - base_state: 9
    - arm_joint_pos: 5
    - arm_joint_vel: 5
    - target_pos: 3
    - ee_pos: 3
    - pos_error: 3
    - pos_dist: 1

    We need to reconstruct this from what we have.
    """
    batch_size = full_obs.shape[0]
    device = full_obs.device

    # Extract what we have
    base_state = full_obs[:, 0:9]      # 9
    arm_joint_pos = full_obs[:, 53:58]  # 5
    arm_joint_vel = full_obs[:, 58:63] * 10  # 5 (undo 0.1 scaling)
    target_z = full_obs[:, 63:64]       # 1

    # Reconstruct target_pos (we only have z, approximate x,y)
    # Assume target is in front of robot
    target_pos = torch.cat([
        torch.zeros(batch_size, 2, device=device),  # x, y
        target_z  # z
    ], dim=-1)  # 3

    # Approximate ee_pos from arm joints (rough forward kinematics)
    # This is a simplification - in practice, env should provide this
    ee_pos_approx = torch.zeros(batch_size, 3, device=device)

    # pos_error and dist (approximate)
    pos_error = target_pos - ee_pos_approx
    pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5

    arm_obs = torch.cat([
        base_state,      # 9
        arm_joint_pos,   # 5
        arm_joint_vel,   # 5
        target_pos,      # 3
        ee_pos_approx,   # 3
        pos_error,       # 3
        pos_dist,        # 1
    ], dim=-1)  # 29

    return arm_obs


# ============================================================================
# PPO TRAINER
# ============================================================================

class PPO:
    def __init__(self, policy: BalanceLocoPolicy, device: str, lr: float = 3e-4):
        self.policy = policy
        self.device = device

        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, args.max_iterations, eta_min=1e-5
        )

        print(f"\n[PPO] Balance policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    def gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, old_log_probs, returns, advantages, old_values):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                values, log_probs, entropy = self.policy.evaluate(
                    obs[mb_idx], actions[mb_idx]
                )

                ratio = (log_probs - old_log_probs[mb_idx]).exp()
                surr1 = ratio * advantages[mb_idx]
                surr2 = ratio.clamp(0.8, 1.2) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                value_clipped = old_values[mb_idx] + (values - old_values[mb_idx]).clamp(-0.2, 0.2)
                critic_loss = 0.5 * torch.max(
                    (values - returns[mb_idx]) ** 2,
                    (value_clipped - returns[mb_idx]) ** 2
                ).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.sched.step()

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "lr": self.sched.get_last_lr()[0],
        }


# ============================================================================
# MAIN
# ============================================================================

def train():
    device = "cuda:0"

    print("\n" + "=" * 70)
    print("G1 REACTIVE BALANCE TRAINING")
    print("=" * 70)
    print(f"  Environments: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Arm checkpoint (FROZEN): {args.arm_checkpoint}")
    print(f"  Loco init (optional): {args.loco_init}")
    print("=" * 70)
    print("  ARCHITECTURE:")
    print("    Arm Policy: FROZEN (reaching expert)")
    print("    Balance Policy: TRAINABLE (learns balance)")
    print("=" * 70 + "\n")

    # Create environment
    env_cfg = G1ReactiveBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = G1ReactiveBalanceEnv(cfg=env_cfg)

    # Load frozen arm policy
    arm_policy = load_frozen_arm_policy(args.arm_checkpoint, device)

    # Create balance policy
    balance_policy = BalanceLocoPolicy().to(device)

    # Optional: Initialize from Stage 3
    if args.loco_init:
        balance_policy = load_loco_init(args.loco_init, balance_policy, device)

    # Resume if specified
    start_iter = 0
    best_reward = float('-inf')
    if args.resume:
        print(f"\n[Resume] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        balance_policy.load_state_dict(ckpt["model_state_dict"])
        start_iter = ckpt.get("iteration", 0)
        best_reward = ckpt.get("best_reward", float('-inf'))
        env.curriculum_level = ckpt.get("curriculum_level", 0)
        print(f"  ✓ Resumed from iteration {start_iter}")

    # PPO trainer
    ppo = PPO(balance_policy, device)

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/g1_reactive_balance_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    print(f"\n[INFO] Logging to: {log_dir}")

    # Exploration
    balance_policy.log_std.data.fill_(np.log(0.4))

    # Initial reset
    obs, _ = env.reset()
    obs = obs["policy"]

    start_time = datetime.now()
    rollout_steps = 24

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("  Balance policy learns to compensate for arm movement")
    print("  Arm policy is FROZEN (uses Stage 5 reaching)")
    print("=" * 70 + "\n")

    for iteration in range(start_iter, args.max_iterations):
        iter_start = datetime.now()

        # Collect rollouts
        obs_buffer = []
        act_buffer = []
        rew_buffer = []
        done_buffer = []
        val_buffer = []
        logp_buffer = []

        for _ in range(rollout_steps):
            # Get arm actions from frozen policy
            with torch.no_grad():
                arm_obs = extract_arm_obs(obs)
                arm_actions = arm_policy(arm_obs)

            # Set arm actions in environment
            env.set_frozen_arm_actions(arm_actions)

            # Get leg actions from balance policy
            with torch.no_grad():
                mean, value = balance_policy.forward(obs)
                std = balance_policy.log_std.clamp(-2, 1).exp()
                dist = torch.distributions.Normal(mean, std)
                leg_action = dist.sample()
                log_prob = dist.log_prob(leg_action).sum(-1)

            obs_buffer.append(obs)
            act_buffer.append(leg_action)
            val_buffer.append(value.squeeze(-1))
            logp_buffer.append(log_prob)

            # Step environment (only leg actions)
            obs_dict, reward, terminated, truncated, _ = env.step(leg_action)
            obs = obs_dict["policy"]

            rew_buffer.append(reward)
            done_buffer.append((terminated | truncated).float())

        # Stack
        obs_buffer = torch.stack(obs_buffer)
        act_buffer = torch.stack(act_buffer)
        rew_buffer = torch.stack(rew_buffer)
        done_buffer = torch.stack(done_buffer)
        val_buffer = torch.stack(val_buffer)
        logp_buffer = torch.stack(logp_buffer)

        # GAE
        with torch.no_grad():
            _, next_value = balance_policy.forward(obs)
            next_value = next_value.squeeze(-1)

        advantages, returns = ppo.gae(rew_buffer, val_buffer, done_buffer, next_value)

        # Update
        num_obs = env_cfg.num_observations
        num_act = env_cfg.num_actions
        update_info = ppo.update(
            obs_buffer.view(-1, num_obs),
            act_buffer.view(-1, num_act),
            logp_buffer.view(-1),
            returns.view(-1),
            advantages.view(-1),
            val_buffer.view(-1),
        )

        # Curriculum
        if iteration % 24 == 0:
            env.update_curriculum(iteration)

        # Exploration decay
        if iteration > 0 and iteration % 500 == 0:
            current_std = balance_policy.log_std.data.exp().mean().item()
            new_std = max(0.15, current_std * 0.95)
            balance_policy.log_std.data.fill_(np.log(new_std))

        # Stats
        mean_reward = rew_buffer.mean().item()
        iter_time = (datetime.now() - iter_start).total_seconds()
        fps = rollout_steps * args.num_envs / iter_time

        # Save best
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model_state_dict": balance_policy.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curriculum_level,
            }, f"{log_dir}/model_best.pt")
            print(f"[BEST] New best reward: {best_reward:.2f}")

        # Logging
        writer.add_scalar("Train/reward", mean_reward, iteration)
        writer.add_scalar("Train/best_reward", best_reward, iteration)
        writer.add_scalar("Loss/actor", update_info["actor_loss"], iteration)
        writer.add_scalar("Loss/critic", update_info["critic_loss"], iteration)
        writer.add_scalar("Curriculum/level", env.curriculum_level + 1, iteration)
        writer.add_scalar("Success/total_reaches", env.total_reaches, iteration)

        for key, value in env.extras.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"Env/{key}", value, iteration)

        writer.flush()

        # Console
        if iteration % 10 == 0:
            elapsed = datetime.now() - start_time
            eta = elapsed / (iteration - start_iter + 1) * (args.max_iterations - iteration)

            com_x = env.extras.get("M/com_x", 0)
            com_y = env.extras.get("M/com_y", 0)

            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Lv={env.curriculum_level + 1} | "
                f"Reach={env.total_reaches} | "
                f"CoM=({com_x:.2f},{com_y:.2f}) | "
                f"FPS={fps:.0f} | "
                f"{str(elapsed).split('.')[0]}"
            )

        # Checkpoints
        if (iteration + 1) % 1000 == 0:
            torch.save({
                "model_state_dict": balance_policy.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curriculum_level,
            }, f"{log_dir}/model_{iteration + 1}.pt")

    # Final save
    torch.save({
        "model_state_dict": balance_policy.state_dict(),
        "iteration": args.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": env.curriculum_level,
    }, f"{log_dir}/model_final.pt")

    writer.close()
    env.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curriculum_level + 1}/6")
    print(f"  Total Reaches: {env.total_reaches}")
    print(f"  Log Dir: {log_dir}")
    print("=" * 70)


if __name__ == "__main__":
    train()
    simulation_app.close()