#!/usr/bin/env python3
"""
ULC G1 Stage 5 Training - FULL MECHANICAL WORKSPACE
====================================================

Stage 4 checkpoint'inden devam eder.
Orijinal config'i DEĞİŞTİRMEZ - sadece command range'leri override eder.

KULLANIM:
./isaaclab.bat -p train_ulc_stage_5.py ^
    --stage4_checkpoint logs/ulc/ulc_g1_stage4_.../model_best.pt ^
    --num_envs 4096 --headless --max_iterations 8000
"""

import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="ULC G1 Stage 5")
parser.add_argument("--stage4_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=8000)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from collections import deque
import gymnasium as gym

import isaaclab_tasks
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg

# ============================================================
# STAGE 5 CURRICULUM - Kademeli Genişletme
# ============================================================

CURRICULUM = {
    0: {
        "name": "Stage4 Baseline",
        "arm_range": 0.8,
        "height_range": (0.60, 0.80),
        "threshold": 26.0,
        "min_iters": 300,
    },
    1: {
        "name": "Arms 60%",
        "arm_range": 1.5,
        "height_range": (0.55, 0.80),
        "threshold": 25.0,
        "min_iters": 400,
    },
    2: {
        "name": "Arms 80% + Light Squat",
        "arm_range": 2.0,
        "height_range": (0.45, 0.82),
        "threshold": 24.0,
        "min_iters": 500,
    },
    3: {
        "name": "Full Arms + Deep Squat",
        "arm_range": 2.6,
        "height_range": (0.38, 0.85),
        "threshold": 23.0,
        "min_iters": 600,
    },
    4: {
        "name": "FULL WORKSPACE",
        "arm_range": 2.6,
        "height_range": (0.35, 0.85),
        "threshold": None,
        "min_iters": None,
    },
}


# ============================================================
# ACTOR-CRITIC (Stage 4 ile AYNI mimari)
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        actor_layers = []
        prev = obs_dim
        for h in hidden_dims:
            actor_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()])
            prev = h
        actor_layers.append(nn.Linear(prev, action_dim))
        self.actor = nn.Sequential(*actor_layers)

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
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std).sample()

    def get_value(self, obs):
        return self.critic(obs)

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return self.critic(obs).squeeze(-1), dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# ============================================================
# CURRICULUM MANAGER
# ============================================================

class CurriculumManager:
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
        print(f"[Curriculum] Level {level}: {cfg['name']}")
        print(f"  Arms: ±{self.arm_range:.1f} rad, Height: {self.height_range}")

    def sample_commands(self):
        n = self.env.num_envs
        d = self.device
        ar = self.arm_range
        hr = self.height_range

        # Height
        self.env.height_command[:, 0] = torch.empty(n, device=d).uniform_(*hr)

        # Arms - genişletilmiş range
        self.env.arm_commands[:, 0] = torch.empty(n, device=d).uniform_(-ar, ar)  # L shoulder_pitch
        self.env.arm_commands[:, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)  # L elbow
        self.env.arm_commands[:, 5] = torch.empty(n, device=d).uniform_(-ar, ar)  # R shoulder_pitch
        self.env.arm_commands[:, 8] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)  # R elbow

    def resample(self, ids):
        if len(ids) == 0:
            return
        n = len(ids)
        d = self.device
        ar = self.arm_range
        hr = self.height_range

        self.env.height_command[ids, 0] = torch.empty(n, device=d).uniform_(*hr)
        self.env.arm_commands[ids, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.arm_commands[ids, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.arm_commands[ids, 5] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.arm_commands[ids, 8] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)

    def update(self, reward):
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
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.epochs = 5
        self.batch_size = 4096
        self.steps = 24

        # Get initial obs
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
                std = torch.exp(self.policy.log_std)
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
            delta = r["rew"][t] + self.gamma * nv * (~r["done"][t]) - r["val"][t]
            adv[t] = gae = delta + self.gamma * self.lam * (~r["done"][t]) * gae

        return adv + r["val"], adv

    def update(self, r, ret, adv):
        obs = r["obs"].view(-1, r["obs"].shape[-1])
        act = r["act"].view(-1, r["act"].shape[-1])
        old_logp = r["logp"].view(-1)
        ret_flat = ret.view(-1)
        adv_flat = (adv.view(-1) - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.epochs):
            perm = torch.randperm(obs.shape[0])
            for i in range(0, obs.shape[0], self.batch_size):
                idx = perm[i:i + self.batch_size]
                val, logp, ent = self.policy.evaluate(obs[idx], act[idx])

                ratio = torch.exp(logp - old_logp[idx])
                s1 = ratio * adv_flat[idx]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_flat[idx]

                loss = -torch.min(s1, s2).mean() + 0.5 * ((val - ret_flat[idx]) ** 2).mean() - 0.01 * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

    def train_iter(self):
        r = self.collect()
        ret, adv = self.compute_gae(r)
        self.update(r, ret, adv)
        return r["rew"].mean().item()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🔥 ULC G1 STAGE 5 - FULL WORKSPACE 🔥")
    print("=" * 60)

    # Load checkpoint
    print(f"[INFO] Loading: {args.stage4_checkpoint}")
    ckpt = torch.load(args.stage4_checkpoint, map_location="cuda:0", weights_only=True)
    print(f"[INFO] Keys: {list(ckpt.keys())}")

    # Create env
    print(f"[INFO] Creating env with {args.num_envs} envs...")
    cfg = ULC_G1_Stage4_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.termination["base_height_min"] = 0.25  # Allow squat

    # Add spaces for newer Isaac Lab
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(cfg.num_observations,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_actions,))

    env = ULC_G1_Env(cfg=cfg)
    env.current_stage = 4

    # Get actual dims
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[1]
    action_dim = cfg.num_actions
    print(f"[INFO] Obs: {obs_dim}, Actions: {action_dim}")

    # Create policy
    policy = ActorCritic(obs_dim, action_dim).to("cuda:0")

    # Load weights
    state_dict = ckpt.get("actor_critic", ckpt)

    # Check dimensions
    ckpt_obs = state_dict["actor.0.weight"].shape[1]
    ckpt_act = state_dict["log_std"].shape[0]
    print(f"[INFO] Checkpoint dims: obs={ckpt_obs}, act={ckpt_act}")

    if ckpt_obs == obs_dim and ckpt_act == action_dim:
        policy.load_state_dict(state_dict)
        print("[INFO] ✓ Weights loaded!")
    else:
        print(f"[WARNING] Dim mismatch! Starting fresh.")

    # Curriculum
    curriculum = CurriculumManager(env, CURRICULUM)
    curriculum.sample_commands()

    # Trainer
    trainer = PPOTrainer(env, policy, curriculum)

    # Log dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage5_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logs: {log_dir}")

    # Train
    print("\n[INFO] Training...")
    best = -float('inf')

    for i in range(args.max_iterations):
        reward = trainer.train_iter()

        if curriculum.update(reward):
            print(f"\n🎯 LEVEL UP! → {curriculum.level} ({curriculum.workspace_pct():.0f}%)\n")

        if reward > best:
            best = reward
            torch.save({
                "actor_critic": policy.state_dict(),
                "iteration": i,
                "best_reward": best,
                "curriculum_level": curriculum.level,
            }, f"{log_dir}/model_best.pt")

        if i % 50 == 0:
            print(
                f"Iter {i:5d} | Rew: {reward:7.2f} | Best: {best:7.2f} | Lvl: {curriculum.level} ({curriculum.workspace_pct():.0f}%)")

        if i % 1000 == 0 and i > 0:
            torch.save({"actor_critic": policy.state_dict(), "iteration": i}, f"{log_dir}/model_{i}.pt")

    torch.save({"actor_critic": policy.state_dict(), "best_reward": best}, f"{log_dir}/model_final.pt")

    print("\n" + "=" * 60)
    print(f"🎉 DONE! Level: {curriculum.level}, Best: {best:.2f}")
    print(f"Saved: {log_dir}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()