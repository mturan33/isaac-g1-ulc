#!/usr/bin/env python3
"""
ULC G1 Stage 5 Training - FULL MECHANICAL WORKSPACE
====================================================

Stage 4 checkpoint'inden devam eder.
Kademeli workspace genişletme ile full range'e ulaşır.

AMAÇ:
- Kolları maksimum açılarda kontrol et (±2.6 rad)
- Deep squat yapabilme (0.35m'ye kadar)
- Yerden nesne almaya hazır

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_5.py ^
    --stage4_checkpoint logs/ulc/ulc_g1_stage4_2026-01-09_20-52-36/model_best.pt ^
    --num_envs 4096 --headless --max_iterations 8000
"""

import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="ULC G1 Stage 5 - Full Workspace")
parser.add_argument("--stage4_checkpoint", type=str, required=True,
                    help="Path to Stage 4 model_best.pt")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=8000)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from Stage 5 checkpoint")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import gymnasium as gym

from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[WARNING] TensorBoard not available")

# ============================================================
# STAGE 5 CURRICULUM - Kademeli Workspace Genişletme
# ============================================================

CURRICULUM = {
    0: {
        "name": "Stage4 Baseline (Warmup)",
        "arm_range": 0.8,
        "height_range": (0.60, 0.80),
        "torso_range": 0.3,
        "threshold": 26.0,
        "min_iters": 200,
    },
    1: {
        "name": "Arms 60%",
        "arm_range": 1.5,
        "height_range": (0.55, 0.80),
        "torso_range": 0.4,
        "threshold": 25.0,
        "min_iters": 400,
    },
    2: {
        "name": "Arms 80% + Light Squat",
        "arm_range": 2.0,
        "height_range": (0.45, 0.82),
        "torso_range": 0.45,
        "threshold": 24.0,
        "min_iters": 500,
    },
    3: {
        "name": "Full Arms + Deep Squat",
        "arm_range": 2.6,
        "height_range": (0.38, 0.85),
        "torso_range": 0.5,
        "threshold": 22.0,
        "min_iters": 600,
    },
    4: {
        "name": "FULL MECHANICAL WORKSPACE",
        "arm_range": 2.6,
        "height_range": (0.35, 0.85),
        "torso_range": 0.52,
        "threshold": None,  # Final level
        "min_iters": None,
    },
}


# ============================================================
# ACTOR-CRITIC (Stage 4 ile AYNI mimari)
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Actor
        actor_layers = []
        prev = obs_dim
        for h in hidden_dims:
            actor_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()])
            prev = h
        actor_layers.append(nn.Linear(prev, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        prev = obs_dim
        for h in hidden_dims:
            critic_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def act(self, obs, deterministic=False):
        mean = self.actor(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std.clamp(-2, 1))
        return torch.distributions.Normal(mean, std).sample()

    def get_value(self, obs):
        return self.critic(obs)

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std.clamp(-2, 1))
        dist = torch.distributions.Normal(mean, std)
        return self.critic(obs).squeeze(-1), dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# ============================================================
# CURRICULUM MANAGER
# ============================================================

class CurriculumManager:
    """Kademeli workspace genişletme."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.level = 0
        self.level_iters = 0
        self.rewards = deque(maxlen=100)
        self.device = env.device
        self._apply(0)

    def _apply(self, level):
        cfg = self.config[level]
        self.arm_range = cfg["arm_range"]
        self.height_range = cfg["height_range"]
        self.torso_range = cfg["torso_range"]
        print(f"\n{'=' * 60}")
        print(f"[Curriculum] Level {level}: {cfg['name']}")
        print(f"  Arms: ±{self.arm_range:.1f} rad ({self.arm_range / 2.6 * 100:.0f}%)")
        print(f"  Height: {self.height_range[0]:.2f} - {self.height_range[1]:.2f}m")
        print(f"  Torso: ±{self.torso_range:.2f} rad")
        print(f"{'=' * 60}\n")

    def sample_commands(self):
        """Tüm env'ler için yeni komutlar sample et."""
        n = self.env.num_envs
        d = self.device
        ar = self.arm_range
        hr = self.height_range
        tr = self.torso_range

        # Height command
        self.env.height_command[:] = torch.empty(n, device=d).uniform_(*hr)

        # Velocity commands
        self.env.velocity_commands[:, 0] = torch.empty(n, device=d).uniform_(-0.8, 1.2)  # vx
        self.env.velocity_commands[:, 1] = torch.empty(n, device=d).uniform_(-0.4, 0.4)  # vy
        self.env.velocity_commands[:, 2] = torch.empty(n, device=d).uniform_(-0.8, 0.8)  # yaw

        # Torso commands
        self.env.torso_commands[:, 0] = torch.empty(n, device=d).uniform_(-tr, tr)  # roll
        self.env.torso_commands[:, 1] = torch.empty(n, device=d).uniform_(-tr, tr)  # pitch
        self.env.torso_commands[:, 2] = torch.empty(n, device=d).uniform_(-0.5, 0.5)  # yaw

        # Arm commands - 5 joints per arm
        # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll
        self.env.left_arm_cmd[:, 0] = torch.empty(n, device=d).uniform_(-ar, ar)        # shoulder_pitch
        self.env.left_arm_cmd[:, 1] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)  # shoulder_roll
        self.env.left_arm_cmd[:, 2] = torch.empty(n, device=d).uniform_(-ar, ar)        # shoulder_yaw
        self.env.left_arm_cmd[:, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)  # elbow_pitch
        self.env.left_arm_cmd[:, 4] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)  # elbow_roll

        # Right arm
        self.env.right_arm_cmd[:, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.right_arm_cmd[:, 1] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.right_arm_cmd[:, 2] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.right_arm_cmd[:, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.right_arm_cmd[:, 4] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)

        # Combined arm_commands
        self.env.arm_commands = torch.cat([self.env.left_arm_cmd, self.env.right_arm_cmd], dim=-1)

    def resample(self, ids):
        """Reset olan env'ler için yeni komutlar."""
        if len(ids) == 0:
            return
        n = len(ids)
        d = self.device
        ar = self.arm_range
        hr = self.height_range
        tr = self.torso_range

        self.env.height_command[ids] = torch.empty(n, device=d).uniform_(*hr)

        self.env.velocity_commands[ids, 0] = torch.empty(n, device=d).uniform_(-0.8, 1.2)
        self.env.velocity_commands[ids, 1] = torch.empty(n, device=d).uniform_(-0.4, 0.4)
        self.env.velocity_commands[ids, 2] = torch.empty(n, device=d).uniform_(-0.8, 0.8)

        self.env.torso_commands[ids, 0] = torch.empty(n, device=d).uniform_(-tr, tr)
        self.env.torso_commands[ids, 1] = torch.empty(n, device=d).uniform_(-tr, tr)
        self.env.torso_commands[ids, 2] = torch.empty(n, device=d).uniform_(-0.5, 0.5)

        # Left arm
        self.env.left_arm_cmd[ids, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.left_arm_cmd[ids, 1] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.left_arm_cmd[ids, 2] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.left_arm_cmd[ids, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.left_arm_cmd[ids, 4] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)

        # Right arm
        self.env.right_arm_cmd[ids, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.right_arm_cmd[ids, 1] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.right_arm_cmd[ids, 2] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.right_arm_cmd[ids, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.right_arm_cmd[ids, 4] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)

        # Update combined
        self.env.arm_commands[ids] = torch.cat([self.env.left_arm_cmd[ids], self.env.right_arm_cmd[ids]], dim=-1)

    def update(self, reward):
        """Curriculum level güncelle."""
        self.level_iters += 1
        self.rewards.append(reward)

        if len(self.rewards) < 50:
            return False

        cfg = self.config[self.level]
        if cfg["threshold"] is None:
            return False

        avg = sum(self.rewards) / len(self.rewards)
        if self.level_iters >= cfg["min_iters"] and avg >= cfg["threshold"]:
            if self.level + 1 in self.config:
                self.level += 1
                self._apply(self.level)
                self.level_iters = 0
                self.rewards.clear()
                return True
        return False

    def workspace_pct(self):
        return (self.arm_range / 2.6) * 100


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:
    def __init__(self, env, policy, curriculum, device="cuda:0"):
        self.env = env
        self.policy = policy
        self.curriculum = curriculum
        self.device = device
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.max_iterations, eta_min=1e-5
        )

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.epochs = 5
        self.batch_size = 4096
        self.steps = 24

        obs_dict, _ = env.reset()
        self.obs = obs_dict["policy"]

    def collect(self):
        obs_l, act_l, rew_l, done_l, val_l, logp_l = [], [], [], [], [], []
        obs = self.obs

        for _ in range(self.steps):
            with torch.no_grad():
                action = self.policy.act(obs)
                value = self.policy.get_value(obs).squeeze(-1)
                mean = self.policy.actor(obs)
                std = torch.exp(self.policy.log_std.clamp(-2, 1))
                logp = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)

            obs_l.append(obs)
            act_l.append(action)
            val_l.append(value)
            logp_l.append(logp)

            obs_dict, reward, term, trunc, _ = self.env.step(action)
            obs = obs_dict["policy"]
            done = term | trunc

            rew_l.append(reward)
            done_l.append(done)

            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.curriculum.resample(reset_ids)

        self.obs = obs

        return {
            "obs": torch.stack(obs_l),
            "act": torch.stack(act_l),
            "rew": torch.stack(rew_l),
            "done": torch.stack(done_l),
            "val": torch.stack(val_l),
            "logp": torch.stack(logp_l),
            "last_obs": obs,
        }

    def compute_gae(self, r):
        with torch.no_grad():
            last_val = self.policy.get_value(r["last_obs"]).squeeze(-1)

        adv = torch.zeros_like(r["rew"])
        gae = 0
        for t in reversed(range(r["rew"].shape[0])):
            nv = last_val if t == r["rew"].shape[0] - 1 else r["val"][t + 1]
            mask = (~r["done"][t]).float()
            delta = r["rew"][t] + self.gamma * nv * mask - r["val"][t]
            adv[t] = gae = delta + self.gamma * self.lam * mask * gae

        return adv + r["val"], adv

    def update(self, r, ret, adv):
        obs = r["obs"].view(-1, r["obs"].shape[-1])
        act = r["act"].view(-1, r["act"].shape[-1])
        old_logp = r["logp"].view(-1)
        old_val = r["val"].view(-1)
        ret_flat = ret.view(-1)
        adv_flat = (adv.view(-1) - adv.mean()) / (adv.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.epochs):
            perm = torch.randperm(obs.shape[0])
            for i in range(0, obs.shape[0], self.batch_size):
                idx = perm[i:i + self.batch_size]
                val, logp, ent = self.policy.evaluate(obs[idx], act[idx])

                ratio = torch.exp(logp - old_logp[idx])
                s1 = ratio * adv_flat[idx]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_flat[idx]
                actor_loss = -torch.min(s1, s2).mean()

                # Value clipping
                val_clipped = old_val[idx] + (val - old_val[idx]).clamp(-0.2, 0.2)
                critic_loss = 0.5 * torch.max(
                    (val - ret_flat[idx]) ** 2,
                    (val_clipped - ret_flat[idx]) ** 2
                ).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += ent.mean().item()
                num_updates += 1

        self.scheduler.step()

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train_iter(self):
        r = self.collect()
        ret, adv = self.compute_gae(r)
        update_info = self.update(r, ret, adv)
        return r["rew"].mean().item(), update_info


# ============================================================
# MAIN
# ============================================================

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    print("\n" + "=" * 70)
    print("🔥 ULC G1 STAGE 5 - FULL MECHANICAL WORKSPACE 🔥")
    print("=" * 70)
    print(f"Stage 4 checkpoint: {args.stage4_checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print()
    print("HEDEF:")
    print("  ✓ Kollar: ±2.6 rad (maksimum açı)")
    print("  ✓ Squat: 0.35m'ye kadar (yerden nesne alma)")
    print("  ✓ Torso: ±0.52 rad (geniş açı)")
    print("=" * 70)

    # Load Stage 4 checkpoint
    print(f"\n[INFO] Loading checkpoint...")
    if not os.path.exists(args.stage4_checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.stage4_checkpoint}")
        return

    ckpt = torch.load(args.stage4_checkpoint, map_location="cuda:0", weights_only=False)
    print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")
    if "best_reward" in ckpt:
        print(f"[INFO] Stage 4 best reward: {ckpt['best_reward']:.2f}")

    # Get checkpoint dimensions
    state_dict = ckpt.get("actor_critic", ckpt)
    ckpt_obs = state_dict["actor.0.weight"].shape[1]
    ckpt_act = state_dict["log_std"].shape[0]
    print(f"[INFO] Checkpoint dims: obs={ckpt_obs}, act={ckpt_act}")

    # Create environment
    print(f"\n[INFO] Creating environment with {args.num_envs} envs...")
    cfg = ULC_G1_Stage4_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.termination["base_height_min"] = 0.25  # Allow deep squat

    # Add gym spaces
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(cfg.num_observations,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_actions,))

    env = ULC_G1_Env(cfg=cfg)
    env.current_stage = 4  # Stage 4 modunda çalış

    # Get actual dimensions from env
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[1]
    action_dim = cfg.num_actions
    print(f"[INFO] Environment dims: obs={obs_dim}, act={action_dim}")

    # Verify dimensions match
    if ckpt_obs != obs_dim or ckpt_act != action_dim:
        print(f"\n[ERROR] Dimension mismatch!")
        print(f"  Checkpoint: obs={ckpt_obs}, act={ckpt_act}")
        print(f"  Environment: obs={obs_dim}, act={action_dim}")
        env.close()
        simulation_app.close()
        return

    # Create policy
    policy = ActorCritic(ckpt_obs, ckpt_act).to("cuda:0")
    policy.load_state_dict(state_dict)
    print("[INFO] ✓ Checkpoint weights loaded successfully!")

    # Resume from Stage 5 checkpoint if provided
    start_iter = 0
    if args.checkpoint:
        print(f"\n[INFO] Resuming from: {args.checkpoint}")
        s5_ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
        policy.load_state_dict(s5_ckpt["actor_critic"])
        start_iter = s5_ckpt.get("iteration", 0) + 1
        print(f"[INFO] Resuming from iteration {start_iter}")

    # Setup curriculum
    curriculum = CurriculumManager(env, CURRICULUM)
    curriculum.sample_commands()

    # Setup trainer
    trainer = PPOTrainer(env, policy, curriculum)

    # Log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage5_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n[INFO] Log directory: {log_dir}")

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir)

    # Training state
    best_reward = float('-inf')
    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("STARTING STAGE 5 TRAINING")
    print("=" * 70 + "\n")

    for iteration in range(start_iter, args.max_iterations):
        iter_start = datetime.now()

        # Training iteration
        reward, update_info = trainer.train_iter()

        # Timing
        iter_time = (datetime.now() - iter_start).total_seconds()
        fps = trainer.steps * args.num_envs / iter_time

        # Check curriculum progress
        level_up = curriculum.update(reward)
        if level_up:
            print(f"\n🎯 LEVEL UP! → Level {curriculum.level} ({curriculum.workspace_pct():.0f}% workspace)\n")

        # Save best model
        if reward > best_reward:
            best_reward = reward
            torch.save({
                "actor_critic": policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": curriculum.level,
                "workspace_pct": curriculum.workspace_pct(),
            }, f"{log_dir}/model_best.pt")

        # TensorBoard logging
        if writer:
            writer.add_scalar("Train/reward", reward, iteration)
            writer.add_scalar("Train/best_reward", best_reward, iteration)
            writer.add_scalar("Curriculum/level", curriculum.level, iteration)
            writer.add_scalar("Curriculum/workspace_pct", curriculum.workspace_pct(), iteration)
            writer.add_scalar("Loss/actor", update_info["actor_loss"], iteration)
            writer.add_scalar("Loss/critic", update_info["critic_loss"], iteration)
            writer.add_scalar("Train/entropy", update_info["entropy"], iteration)
            writer.add_scalar("Train/lr", update_info["lr"], iteration)
            writer.add_scalar("Train/fps", fps, iteration)

            # Environment extras
            for key, value in env.extras.items():
                writer.add_scalar(f"Env/{key}", value, iteration)

        # Console logging with elapsed/ETA
        if iteration % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            iters_done = iteration - start_iter + 1
            iters_remaining = args.max_iterations - iteration - 1
            eta = elapsed / iters_done * iters_remaining if iters_done > 0 else 0

            # Get env metrics
            height = env.extras.get("M/height", 0)
            vx = env.extras.get("M/vx", 0)
            left_arm_err = env.extras.get("M/left_arm_err", 0)
            right_arm_err = env.extras.get("M/right_arm_err", 0)

            print(
                f"#{iteration:5d} | "
                f"R={reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Lv={curriculum.level} ({curriculum.workspace_pct():3.0f}%) | "
                f"H={height:.2f}m | "
                f"L_arm={left_arm_err:.3f} | "
                f"R_arm={right_arm_err:.3f} | "
                f"FPS={fps:5.0f} | "
                f"{format_time(elapsed)} / {format_time(eta)}"
            )

        # Periodic checkpoint
        if (iteration + 1) % 500 == 0:
            torch.save({
                "actor_critic": policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": curriculum.level,
            }, f"{log_dir}/model_{iteration + 1}.pt")
            print(f"[SAVE] Checkpoint saved: model_{iteration + 1}.pt")

        if writer:
            writer.flush()

    # Final save
    torch.save({
        "actor_critic": policy.state_dict(),
        "iteration": args.max_iterations,
        "best_reward": best_reward,
        "final_curriculum_level": curriculum.level,
    }, f"{log_dir}/model_final.pt")

    if writer:
        writer.close()

    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print(f"🎉 TRAINING COMPLETE!")
    print(f"   Total Time: {format_time(total_time)}")
    print(f"   Final Level: {curriculum.level} ({curriculum.workspace_pct():.0f}% workspace)")
    print(f"   Best Reward: {best_reward:.2f}")
    print(f"   Saved to: {log_dir}")
    print()
    print("KAZANILAN YETENEKLER:")
    print(f"   ✓ Kol açısı: ±{curriculum.arm_range:.1f} rad")
    print(f"   ✓ Squat derinliği: {curriculum.height_range[0]:.2f}m")
    print(f"   ✓ Torso açısı: ±{curriculum.torso_range:.2f} rad")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()