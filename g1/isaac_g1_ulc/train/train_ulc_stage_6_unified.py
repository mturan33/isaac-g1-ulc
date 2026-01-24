"""
G1 Stage 6: Dual Trainable Loco-Manipulation
=============================================

HER İKİ POLİCY EĞİTİLİYOR:
- Arm Policy: LOW LR (1e-5) → Yeni pozisyonlara adapte olur
- Balance Policy: HIGH LR (3e-4) → Squat/lean öğrenir

NEDEN?
- Stage 5 arm policy sadece STANDING pozisyonunda eğitildi
- Squat pozisyonunda reaching yapamıyor
- Arm policy de yeni workspace'e adapte olmalı

SONUÇ:
- Robot squat yapınca arm policy de adapte olur
- Her ikisi birlikte koordineli çalışır
- Yerden nesne almak mümkün olur

USAGE:
./isaaclab.bat -p .../train_dual_trainable.py \
    --num_envs 2048 \
    --max_iterations 15000 \
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

parser = argparse.ArgumentParser(description="G1 Dual Trainable Training")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=15000)
parser.add_argument("--arm_checkpoint", type=str, required=True,
                    help="Stage 5 arm checkpoint (will be FINE-TUNED)")
parser.add_argument("--loco_init", type=str, default=None,
                    help="Optional: Stage 3 checkpoint for balance init")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--arm_lr", type=float, default=1e-5,
                    help="Learning rate for arm policy (low for fine-tuning)")
parser.add_argument("--balance_lr", type=float, default=3e-4,
                    help="Learning rate for balance policy")

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

class ArmPolicy(nn.Module):
    """
    Arm policy - TRAINABLE with low learning rate.
    Will adapt to new positions (squat, lean).
    """
    def __init__(self, num_obs=29, num_act=5, hidden=[256, 128, 64]):
        super().__init__()

        # NO LayerNorm - matches Stage 5
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.ones(num_act) * -1.0)  # Low exploration

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = self.log_std.clamp(-3, 0).exp()  # Very low std for fine-tuning
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.forward(obs)
        std = self.log_std.clamp(-3, 0).exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


class BalancePolicy(nn.Module):
    """
    Balance policy - TRAINABLE with high learning rate.
    Learns squat/lean to enable arm reaching.
    """
    def __init__(self, num_obs=72, num_act=12, hidden=[512, 256, 128]):
        super().__init__()

        # With LayerNorm for stability
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        # Critic (shared for both policies)
        critic_layers = []
        prev = num_obs + 29  # Balance obs + Arm obs
        for h in hidden:
            critic_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def get_value(self, balance_obs: torch.Tensor, arm_obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([balance_obs, arm_obs], dim=-1)
        return self.critic(combined)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.forward(obs)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# ============================================================================
# DUAL CONTROLLER
# ============================================================================

class DualController(nn.Module):
    """
    Dual controller managing both policies.
    """
    def __init__(self, arm_policy: ArmPolicy, balance_policy: BalancePolicy):
        super().__init__()
        self.arm = arm_policy
        self.balance = balance_policy

    def forward(self, balance_obs: torch.Tensor, arm_obs: torch.Tensor):
        leg_actions = self.balance(balance_obs)
        arm_actions = self.arm(arm_obs)
        return leg_actions, arm_actions

    def get_value(self, balance_obs: torch.Tensor, arm_obs: torch.Tensor):
        return self.balance.get_value(balance_obs, arm_obs)


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_arm_checkpoint(checkpoint_path: str, device: str) -> ArmPolicy:
    """Load Stage 5 arm policy for fine-tuning."""
    print(f"\n[Load] Loading arm policy for FINE-TUNING: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Analyze architecture
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key] = value

    hidden = []
    if "actor.0.weight" in actor_state:
        hidden.append(actor_state["actor.0.weight"].shape[0])
    if "actor.2.weight" in actor_state:
        hidden.append(actor_state["actor.2.weight"].shape[0])
    if "actor.4.weight" in actor_state:
        hidden.append(actor_state["actor.4.weight"].shape[0])

    num_obs = actor_state["actor.0.weight"].shape[1]
    num_act = actor_state["actor.6.weight"].shape[0]

    print(f"  Architecture: {num_obs} -> {hidden} -> {num_act}")

    policy = ArmPolicy(num_obs=num_obs, num_act=num_act, hidden=hidden).to(device)
    policy.load_state_dict(actor_state, strict=False)

    print(f"  ✓ Loaded weights, policy is TRAINABLE")

    return policy


def load_balance_init(checkpoint_path: str, policy: BalancePolicy, device: str):
    """Initialize balance policy from Stage 3."""
    print(f"\n[Init] Balance policy from Stage 3: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "actor_critic" in ckpt:
        state_dict = ckpt["actor_critic"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    loaded = 0
    partial = 0

    for name, param in policy.named_parameters():
        if name in state_dict:
            src = state_dict[name]

            if param.shape == src.shape:
                param.data.copy_(src)
                loaded += 1
            elif name == "actor.0.weight":
                min_in = min(param.shape[1], src.shape[1])
                param.data[:, :min_in] = src[:, :min_in]
                param.data[:, min_in:] = torch.randn_like(param.data[:, min_in:]) * 0.01
                partial += 1

    print(f"  ✓ {loaded} full + {partial} partial transfers")
    return policy


# ============================================================================
# ARM OBSERVATION EXTRACTOR
# ============================================================================

class ArmObsExtractor:
    def __init__(self, device: str):
        self.device = device

    def extract(self, balance_obs: torch.Tensor, env) -> torch.Tensor:
        """Extract arm observations from environment."""
        batch_size = balance_obs.shape[0]

        # From balance_obs
        lin_vel_b = balance_obs[:, 0:3]
        ang_vel_b = balance_obs[:, 3:6]
        arm_joint_pos = balance_obs[:, 49:54]
        arm_joint_vel = balance_obs[:, 54:59] * 10

        # From environment (actual values)
        target_pos = env.target_pos_body.clone()
        ee_pos = env._compute_ee_pos_body()
        pos_error = target_pos - ee_pos
        pos_dist = pos_error.norm(dim=-1, keepdim=True)
        prev_arm_actions = env.smoothed_arm_actions.clone()

        # Build 29-dim arm observation
        arm_obs = torch.cat([
            arm_joint_pos,          # 5
            arm_joint_vel,          # 5
            target_pos,             # 3
            ee_pos,                 # 3
            pos_error,              # 3
            pos_dist / 0.5,         # 1
            prev_arm_actions,       # 5
            lin_vel_b,              # 3
            ang_vel_b[:, 2:3],      # 1
        ], dim=-1)

        return arm_obs.clamp(-10, 10)


# ============================================================================
# DUAL PPO TRAINER
# ============================================================================

class DualPPO:
    """
    PPO trainer with separate learning rates for arm and balance.
    """
    def __init__(self, controller: DualController, device: str,
                 arm_lr: float = 1e-5, balance_lr: float = 3e-4):
        self.controller = controller
        self.device = device

        # Separate optimizers with different learning rates
        self.arm_opt = torch.optim.AdamW(
            controller.arm.parameters(), lr=arm_lr, weight_decay=1e-6
        )
        self.balance_opt = torch.optim.AdamW(
            controller.balance.parameters(), lr=balance_lr, weight_decay=1e-5
        )

        # Schedulers
        self.arm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.arm_opt, args.max_iterations, eta_min=arm_lr * 0.1
        )
        self.balance_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.balance_opt, args.max_iterations, eta_min=1e-5
        )

        arm_params = sum(p.numel() for p in controller.arm.parameters())
        balance_params = sum(p.numel() for p in controller.balance.parameters())
        print(f"\n[DualPPO] Arm params: {arm_params:,} (lr={arm_lr})")
        print(f"[DualPPO] Balance params: {balance_params:,} (lr={balance_lr})")

    def gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, balance_obs, arm_obs, leg_actions, arm_actions,
               old_leg_logp, old_arm_logp, returns, advantages, old_values):

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"arm_loss": 0, "balance_loss": 0, "arm_entropy": 0, "balance_entropy": 0}
        num_updates = 0

        batch_size = balance_obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                # ═══════════════════════════════════════════════════
                # BALANCE POLICY UPDATE (HIGH LR)
                # ═══════════════════════════════════════════════════
                balance_logp, balance_entropy = self.controller.balance.evaluate(
                    balance_obs[mb_idx], leg_actions[mb_idx]
                )

                value = self.controller.get_value(
                    balance_obs[mb_idx], arm_obs[mb_idx]
                ).squeeze(-1)

                ratio = (balance_logp - old_leg_logp[mb_idx]).exp()
                surr1 = ratio * advantages[mb_idx]
                surr2 = ratio.clamp(0.8, 1.2) * advantages[mb_idx]
                balance_actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((value - returns[mb_idx]) ** 2).mean()

                balance_loss = balance_actor_loss + 0.5 * value_loss - 0.01 * balance_entropy.mean()

                self.balance_opt.zero_grad()
                balance_loss.backward()
                nn.utils.clip_grad_norm_(self.controller.balance.parameters(), 0.5)
                self.balance_opt.step()

                # ═══════════════════════════════════════════════════
                # ARM POLICY UPDATE (LOW LR)
                # ═══════════════════════════════════════════════════
                arm_logp, arm_entropy = self.controller.arm.evaluate(
                    arm_obs[mb_idx], arm_actions[mb_idx]
                )

                ratio = (arm_logp - old_arm_logp[mb_idx]).exp()
                surr1 = ratio * advantages[mb_idx]
                surr2 = ratio.clamp(0.9, 1.1) * advantages[mb_idx]  # Tighter clip for arm
                arm_loss = -torch.min(surr1, surr2).mean() - 0.005 * arm_entropy.mean()

                self.arm_opt.zero_grad()
                arm_loss.backward()
                nn.utils.clip_grad_norm_(self.controller.arm.parameters(), 0.3)  # Smaller grad clip
                self.arm_opt.step()

                metrics["arm_loss"] += arm_loss.item()
                metrics["balance_loss"] += balance_loss.item()
                metrics["arm_entropy"] += arm_entropy.mean().item()
                metrics["balance_entropy"] += balance_entropy.mean().item()
                num_updates += 1

        self.arm_sched.step()
        self.balance_sched.step()

        for k in metrics:
            metrics[k] /= num_updates

        metrics["arm_lr"] = self.arm_sched.get_last_lr()[0]
        metrics["balance_lr"] = self.balance_sched.get_last_lr()[0]

        return metrics


# ============================================================================
# MAIN
# ============================================================================

def train():
    device = "cuda:0"

    print("\n" + "=" * 70)
    print("G1 DUAL TRAINABLE LOCO-MANIPULATION")
    print("=" * 70)
    print(f"  Environments: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Arm checkpoint: {args.arm_checkpoint}")
    print(f"  Balance init: {args.loco_init}")
    print("-" * 70)
    print("  ARCHITECTURE:")
    print(f"    Arm Policy: TRAINABLE (lr={args.arm_lr}) - adapts to squat/lean")
    print(f"    Balance Policy: TRAINABLE (lr={args.balance_lr}) - learns squat/lean")
    print("=" * 70 + "\n")

    # Environment
    env_cfg = G1ReactiveBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = G1ReactiveBalanceEnv(cfg=env_cfg)

    # Load policies
    arm_policy = load_arm_checkpoint(args.arm_checkpoint, device)
    balance_policy = BalancePolicy().to(device)

    if args.loco_init:
        balance_policy = load_balance_init(args.loco_init, balance_policy, device)

    # Dual controller
    controller = DualController(arm_policy, balance_policy)

    # Resume
    start_iter = 0
    best_reward = float('-inf')
    if args.resume:
        print(f"\n[Resume] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["controller"])
        start_iter = ckpt.get("iteration", 0)
        best_reward = ckpt.get("best_reward", float('-inf'))
        env.curriculum_level = ckpt.get("curriculum_level", 0)

    # PPO
    ppo = DualPPO(controller, device, arm_lr=args.arm_lr, balance_lr=args.balance_lr)

    # Arm obs extractor
    arm_extractor = ArmObsExtractor(device)

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/g1_dual_trainable_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    print(f"\n[INFO] Logging to: {log_dir}")

    # Exploration
    balance_policy.log_std.data.fill_(np.log(0.4))
    arm_policy.log_std.data.fill_(np.log(0.15))  # Low exploration for arm

    obs, _ = env.reset()
    balance_obs = obs["policy"]

    start_time = datetime.now()
    rollout_steps = 24

    print("\n" + "=" * 70)
    print("STARTING DUAL TRAINING")
    print("  Both policies learn together!")
    print("  Arm adapts to squat positions, Balance learns to squat")
    print("=" * 70 + "\n")

    for iteration in range(start_iter, args.max_iterations):
        iter_start = datetime.now()

        # Buffers
        balance_obs_buf = []
        arm_obs_buf = []
        leg_act_buf = []
        arm_act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        leg_logp_buf = []
        arm_logp_buf = []

        for _ in range(rollout_steps):
            arm_obs = arm_extractor.extract(balance_obs, env)

            with torch.no_grad():
                # Balance policy
                leg_mean = balance_policy(balance_obs)
                leg_std = balance_policy.log_std.clamp(-2, 1).exp()
                leg_dist = torch.distributions.Normal(leg_mean, leg_std)
                leg_action = leg_dist.sample()
                leg_logp = leg_dist.log_prob(leg_action).sum(-1)

                # Arm policy
                arm_mean = arm_policy(arm_obs)
                arm_std = arm_policy.log_std.clamp(-3, 0).exp()
                arm_dist = torch.distributions.Normal(arm_mean, arm_std)
                arm_action = arm_dist.sample()
                arm_logp = arm_dist.log_prob(arm_action).sum(-1)

                # Value
                value = balance_policy.get_value(balance_obs, arm_obs).squeeze(-1)

            # Store
            balance_obs_buf.append(balance_obs)
            arm_obs_buf.append(arm_obs)
            leg_act_buf.append(leg_action)
            arm_act_buf.append(arm_action)
            val_buf.append(value)
            leg_logp_buf.append(leg_logp)
            arm_logp_buf.append(arm_logp)

            # Set arm actions and step
            env.set_frozen_arm_actions(arm_action)
            obs_dict, reward, terminated, truncated, _ = env.step(leg_action)
            balance_obs = obs_dict["policy"]

            rew_buf.append(reward)
            done_buf.append((terminated | truncated).float())

        # Stack
        balance_obs_buf = torch.stack(balance_obs_buf)
        arm_obs_buf = torch.stack(arm_obs_buf)
        leg_act_buf = torch.stack(leg_act_buf)
        arm_act_buf = torch.stack(arm_act_buf)
        rew_buf = torch.stack(rew_buf)
        done_buf = torch.stack(done_buf)
        val_buf = torch.stack(val_buf)
        leg_logp_buf = torch.stack(leg_logp_buf)
        arm_logp_buf = torch.stack(arm_logp_buf)

        # GAE
        with torch.no_grad():
            arm_obs_final = arm_extractor.extract(balance_obs, env)
            next_value = balance_policy.get_value(balance_obs, arm_obs_final).squeeze(-1)

        advantages, returns = ppo.gae(rew_buf, val_buf, done_buf, next_value)

        # Update
        T, N = balance_obs_buf.shape[:2]
        metrics = ppo.update(
            balance_obs_buf.view(T*N, -1),
            arm_obs_buf.view(T*N, -1),
            leg_act_buf.view(T*N, -1),
            arm_act_buf.view(T*N, -1),
            leg_logp_buf.view(T*N),
            arm_logp_buf.view(T*N),
            returns.view(T*N),
            advantages.view(T*N),
            val_buf.view(T*N),
        )

        # Curriculum
        if iteration % 24 == 0:
            env.update_curriculum(iteration)

        # Stats
        mean_reward = rew_buf.mean().item()
        iter_time = (datetime.now() - iter_start).total_seconds()
        fps = rollout_steps * args.num_envs / iter_time

        # Save best
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "controller": controller.state_dict(),
                "arm": arm_policy.state_dict(),
                "balance": balance_policy.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curriculum_level,
            }, f"{log_dir}/model_best.pt")
            print(f"[BEST] New best reward: {best_reward:.2f}")

        # Logging
        writer.add_scalar("Train/reward", mean_reward, iteration)
        writer.add_scalar("Train/best_reward", best_reward, iteration)
        writer.add_scalar("Loss/arm", metrics["arm_loss"], iteration)
        writer.add_scalar("Loss/balance", metrics["balance_loss"], iteration)
        writer.add_scalar("LR/arm", metrics["arm_lr"], iteration)
        writer.add_scalar("LR/balance", metrics["balance_lr"], iteration)
        writer.add_scalar("Curriculum/level", env.curriculum_level + 1, iteration)
        writer.add_scalar("Success/total_reaches", env.total_reaches, iteration)

        for key, value in env.extras.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"Env/{key}", value, iteration)

        writer.flush()

        # Console
        if iteration % 10 == 0:
            elapsed = datetime.now() - start_time
            arm_dist = env.extras.get("M/arm_dist", 0)
            height = env.extras.get("M/height", 0)

            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Lv={env.curriculum_level + 1} | "
                f"Reach={env.total_reaches} | "
                f"H={height:.2f} | "
                f"Arm={arm_dist:.3f}m | "
                f"FPS={fps:.0f}"
            )

        # Checkpoints
        if (iteration + 1) % 1000 == 0:
            torch.save({
                "controller": controller.state_dict(),
                "arm": arm_policy.state_dict(),
                "balance": balance_policy.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curriculum_level,
            }, f"{log_dir}/model_{iteration + 1}.pt")

    # Final
    torch.save({
        "controller": controller.state_dict(),
        "arm": arm_policy.state_dict(),
        "balance": balance_policy.state_dict(),
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