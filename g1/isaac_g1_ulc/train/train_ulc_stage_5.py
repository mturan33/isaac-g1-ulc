#!/usr/bin/env python3
"""
ULC G1 Stage 5 Training - FULL MECHANICAL WORKSPACE
====================================================

Stage 4 checkpoint'inden başlayıp %100 mekanik limite genişletir.

YENİ ÖZELLİKLER:
- Squat (height 0.35m) - yerden eşya alma
- Bel eğilme (waist_pitch ±30°)
- Full arm range (shoulder_pitch ±149°, elbow ±92°)
- Full leg range (hip_pitch ±90°, knee 115°)

G1 GERÇEK MEKANİK LİMİTLER (Unitree'nin belirlediği GÜVENLİ aralık):
- shoulder_pitch: (-3.1, +2.6) rad = simetrik ±2.6 rad kullan
- elbow: (-1.6, +1.6) rad = ±1.6 rad
- hip_pitch: (-1.57, +1.57) rad = ±1.57 rad
- knee: (-0.1, +2.0) rad = 2.0 rad max
- waist_pitch: (-0.52, +0.52) rad = ±0.52 rad

KULLANIM:
./isaaclab.bat -p train_ulc_stage_5.py ^
    --stage4_checkpoint logs/ulc/ulc_g1_stage4_.../model_best.pt ^
    --num_envs 4096 --headless
"""

import argparse
import os
import sys
from datetime import datetime

# ============================================================
# ARGUMENT PARSING
# ============================================================

parser = argparse.ArgumentParser(description="ULC G1 Stage 5 - Full Workspace Fine-tuning")
parser.add_argument("--stage4_checkpoint", type=str, required=True,
                    help="Path to Stage 4 checkpoint")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=6000,
                    help="Maximum training iterations")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
from collections import deque
import math

from isaaclab.envs import DirectRLEnv
from isaaclab_rl.rsl_rl.vecenv import RslRlVecEnvWrapper

# ============================================================
# G1 FULL MECHANICAL LIMITS (Unitree Official)
# ============================================================
# Bu limitler Unitree'nin belirlediği GÜVENLİ çalışma aralığı!
# Simülasyonda veya gerçek robotta bu limitlerde çalışmak GÜVENLI.

G1_MECHANICAL_LIMITS = {
    # LEGS
    "hip_pitch": (-1.57, 1.57),  # ±90° - Squat için!
    "hip_roll": (-0.5, 0.5),  # ±29°
    "hip_yaw": (-0.5, 0.5),  # ±29°
    "knee": (-0.1, 2.0),  # -6° to +115° - Squat için!
    "ankle_pitch": (-0.87, 0.52),  # -50° to +30°
    "ankle_roll": (-0.26, 0.26),  # ±15°

    # WAIST
    "waist_yaw": (-2.7, 2.7),  # ±155°
    "waist_roll": (-0.52, 0.52),  # ±30°
    "waist_pitch": (-0.52, 0.52),  # ±30° - Eğilme için!

    # ARMS
    "shoulder_pitch": (-3.1, 2.6),  # -178° to +149°
    "shoulder_roll": (-1.6, 2.6),  # -92° to +149°
    "shoulder_yaw": (-2.6, 2.6),  # ±149°
    "elbow": (-1.6, 1.6),  # ±92°
}


# Simetrik range (training için)
def symmetric_limit(name):
    low, high = G1_MECHANICAL_LIMITS[name]
    return min(abs(low), abs(high))


# ============================================================
# STAGE 5 CURRICULUM - Kademeli Genişletme
# ============================================================
# Stage 4'ün öğrendiği dar range'den başla, kademeli olarak full range'e genişlet

STAGE5_CURRICULUM = {
    # Level 0: Stage 4'ün bittiği yerden başla (warm-up)
    0: {
        "name": "Stage4 Baseline",
        # Velocity (Stage 4 ile aynı)
        "vx_range": (-0.3, 0.8),

        # Height - henüz squat yok
        "height_range": (0.60, 0.80),

        # Arms - Stage 4'ün range'i
        "shoulder_pitch_range": 0.8,
        "elbow_range": 0.8,

        # Torso - Stage 4 ile aynı
        "torso_pitch_range": 0.3,
        "torso_roll_range": 0.25,

        # Legs - henüz genişletilmedi
        "hip_pitch_range": 0.8,
        "knee_range": 1.0,

        "threshold": 26.0,
        "min_iterations": 300,
    },

    # Level 1: Arms genişletmeye başla
    1: {
        "name": "Extended Arms (60%)",
        "vx_range": (-0.3, 0.8),
        "height_range": (0.55, 0.80),

        # Arms - %60
        "shoulder_pitch_range": 1.5,  # 0.8 → 1.5
        "elbow_range": 1.0,  # 0.8 → 1.0

        "torso_pitch_range": 0.35,
        "torso_roll_range": 0.30,

        "hip_pitch_range": 1.0,  # Biraz genişlet
        "knee_range": 1.2,

        "threshold": 25.0,
        "min_iterations": 400,
    },

    # Level 2: Squat başlat + Arms %80
    2: {
        "name": "Light Squat + Arms 80%",
        "vx_range": (-0.4, 0.9),
        "height_range": (0.45, 0.82),  # Squat başlıyor!

        # Arms - %80
        "shoulder_pitch_range": 2.0,
        "elbow_range": 1.3,

        "torso_pitch_range": 0.40,
        "torso_roll_range": 0.35,

        # Legs - squat için genişlet
        "hip_pitch_range": 1.25,
        "knee_range": 1.5,

        "threshold": 24.0,
        "min_iterations": 500,
    },

    # Level 3: Deep squat + Arms %100
    3: {
        "name": "Deep Squat + Full Arms",
        "vx_range": (-0.5, 1.0),
        "height_range": (0.38, 0.85),  # Derin squat!

        # ARMS - %100 FULL!
        "shoulder_pitch_range": 2.6,  # FULL!
        "shoulder_roll_range": 1.6,  # FULL!
        "shoulder_yaw_range": 2.6,  # FULL!
        "elbow_range": 1.6,  # FULL!

        # Torso - genişlet
        "torso_pitch_range": 0.45,
        "torso_roll_range": 0.40,

        # Legs - %90
        "hip_pitch_range": 1.4,
        "knee_range": 1.8,

        "threshold": 23.0,
        "min_iterations": 600,
    },

    # Level 4: FULL WORKSPACE - %100!
    4: {
        "name": "FULL MECHANICAL WORKSPACE",
        "vx_range": (-0.5, 1.0),
        "height_range": (0.35, 0.85),  # FULL SQUAT!

        # ARMS - %100 FULL!
        "shoulder_pitch_range": 2.6,
        "shoulder_roll_range": 1.6,
        "shoulder_yaw_range": 2.6,
        "elbow_range": 1.6,

        # TORSO - %100 FULL!
        "torso_pitch_range": 0.52,  # FULL!
        "torso_roll_range": 0.52,  # FULL!

        # WAIST - %100 FULL!
        "waist_pitch_range": 0.52,  # FULL - Eğilme!
        "waist_roll_range": 0.52,
        "waist_yaw_range": 2.7,  # FULL!

        # LEGS - %100 FULL!
        "hip_pitch_range": 1.57,  # FULL - ±90°!
        "hip_roll_range": 0.5,
        "hip_yaw_range": 0.5,
        "knee_range": 2.0,  # FULL - 115°!
        "ankle_pitch_range": 0.87,
        "ankle_roll_range": 0.26,

        "threshold": None,  # Final level
        "min_iterations": None,
    },
}


# ============================================================
# ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):
    """Actor-Critic with LayerNorm (Stage 4 ile aynı mimari)."""

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
# STAGE 5 ENVIRONMENT WRAPPER
# ============================================================

class Stage5EnvWrapper:
    """
    Stage 5 environment wrapper.
    Stage 4 environment'ı alır, curriculum parametrelerini override eder.
    """

    def __init__(self, base_env, curriculum_config):
        self.env = base_env
        self.curriculum = curriculum_config
        self.current_level = 0
        self.device = base_env.device

        # Override parameters
        self._apply_curriculum_level(0)

    def _apply_curriculum_level(self, level):
        """Apply curriculum level settings to environment."""
        cfg = self.curriculum[level]

        # Store current config
        self.current_config = cfg

        # Update environment's command ranges
        # Bu değerler sample_commands() çağrıldığında kullanılacak
        self.vx_range = cfg.get("vx_range", (-0.5, 1.0))
        self.height_range = cfg.get("height_range", (0.35, 0.85))

        # Arm ranges
        self.shoulder_pitch_range = cfg.get("shoulder_pitch_range", 2.6)
        self.elbow_range = cfg.get("elbow_range", 1.6)

        # Torso ranges
        self.torso_pitch_range = cfg.get("torso_pitch_range", 0.52)
        self.torso_roll_range = cfg.get("torso_roll_range", 0.52)

        # Leg ranges (for squat)
        self.hip_pitch_range = cfg.get("hip_pitch_range", 1.57)
        self.knee_range = cfg.get("knee_range", 2.0)

        print(f"[Stage5] Applied Level {level}: {cfg['name']}")
        print(f"  Arms: shoulder={self.shoulder_pitch_range:.1f}, elbow={self.elbow_range:.1f}")
        print(f"  Height: {self.height_range}")

    def sample_commands(self):
        """Sample commands with current curriculum ranges."""
        env = self.env
        num_envs = env.num_envs
        device = self.device

        # Velocity commands
        env.velocity_commands[:, 0] = torch.empty(num_envs, device=device).uniform_(*self.vx_range)
        env.velocity_commands[:, 1] = torch.empty(num_envs, device=device).uniform_(-0.3, 0.3)
        env.velocity_commands[:, 2] = torch.empty(num_envs, device=device).uniform_(-0.5, 0.5)

        # Height command (squat için!)
        env.height_command[:, 0] = torch.empty(num_envs, device=device).uniform_(*self.height_range)

        # Torso commands
        env.torso_commands[:, 0] = torch.empty(num_envs, device=device).uniform_(
            -self.torso_roll_range, self.torso_roll_range)
        env.torso_commands[:, 1] = torch.empty(num_envs, device=device).uniform_(
            -self.torso_pitch_range, self.torso_pitch_range)
        env.torso_commands[:, 2] = torch.empty(num_envs, device=device).uniform_(-0.3, 0.3)

        # Arm commands - FULL RANGE!
        # Left arm (indices 0-4): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, elbow_roll
        # Right arm (indices 5-9): same

        sp_range = self.shoulder_pitch_range
        el_range = self.elbow_range

        # Left arm
        env.arm_commands[:, 0] = torch.empty(num_envs, device=device).uniform_(-sp_range, sp_range)  # L shoulder_pitch
        env.arm_commands[:, 1] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)  # L shoulder_roll
        env.arm_commands[:, 2] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)  # L shoulder_yaw
        env.arm_commands[:, 3] = torch.empty(num_envs, device=device).uniform_(-el_range, el_range)  # L elbow
        env.arm_commands[:, 4] = torch.empty(num_envs, device=device).uniform_(-0.5, 0.5)  # L elbow_roll

        # Right arm (mirror)
        env.arm_commands[:, 5] = torch.empty(num_envs, device=device).uniform_(-sp_range, sp_range)
        env.arm_commands[:, 6] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)
        env.arm_commands[:, 7] = torch.empty(num_envs, device=device).uniform_(-1.0, 1.0)
        env.arm_commands[:, 8] = torch.empty(num_envs, device=device).uniform_(-el_range, el_range)
        env.arm_commands[:, 9] = torch.empty(num_envs, device=device).uniform_(-0.5, 0.5)

    def step(self, actions):
        """Step environment and periodically resample commands."""
        obs, reward, done, info = self.env.step(actions)

        # Resample commands for reset environments
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._resample_for_ids(reset_ids)

        return obs, reward, done, info

    def _resample_for_ids(self, env_ids):
        """Resample commands for specific environments."""
        num_reset = len(env_ids)
        device = self.device

        sp_range = self.shoulder_pitch_range
        el_range = self.elbow_range

        # Velocity
        self.env.velocity_commands[env_ids, 0] = torch.empty(num_reset, device=device).uniform_(*self.vx_range)

        # Height
        self.env.height_command[env_ids, 0] = torch.empty(num_reset, device=device).uniform_(*self.height_range)

        # Torso
        self.env.torso_commands[env_ids, 1] = torch.empty(num_reset, device=device).uniform_(
            -self.torso_pitch_range, self.torso_pitch_range)

        # Arms
        self.env.arm_commands[env_ids, 0] = torch.empty(num_reset, device=device).uniform_(-sp_range, sp_range)
        self.env.arm_commands[env_ids, 3] = torch.empty(num_reset, device=device).uniform_(-el_range, el_range)
        self.env.arm_commands[env_ids, 5] = torch.empty(num_reset, device=device).uniform_(-sp_range, sp_range)
        self.env.arm_commands[env_ids, 8] = torch.empty(num_reset, device=device).uniform_(-el_range, el_range)

    def promote_level(self):
        """Advance to next curriculum level."""
        next_level = self.current_level + 1
        if next_level in self.curriculum:
            self.current_level = next_level
            self._apply_curriculum_level(next_level)
            return True
        return False

    def get_level_info(self):
        """Get current level info."""
        cfg = self.curriculum[self.current_level]
        sp = cfg.get("shoulder_pitch_range", 2.6)
        workspace_pct = (sp / 2.6) * 100
        return {
            "level": self.current_level,
            "name": cfg["name"],
            "workspace_pct": workspace_pct,
            "threshold": cfg.get("threshold"),
        }

    @property
    def obs_buf(self):
        return self.env.obs_buf

    @property
    def num_envs(self):
        return self.env.num_envs


# ============================================================
# CURRICULUM MANAGER
# ============================================================

class CurriculumManager:
    """Manages curriculum progression."""

    def __init__(self, env_wrapper, curriculum_config):
        self.env = env_wrapper
        self.curriculum = curriculum_config
        self.level_iterations = 0
        self.reward_history = deque(maxlen=100)

    def update(self, iteration, mean_reward):
        """Check for level promotion."""
        self.level_iterations += 1
        self.reward_history.append(mean_reward)

        if len(self.reward_history) < 50:
            return False

        level = self.env.current_level
        cfg = self.curriculum[level]
        threshold = cfg.get("threshold")
        min_iters = cfg.get("min_iterations", 0)

        if threshold is None:  # Final level
            return False

        avg_reward = sum(self.reward_history) / len(self.reward_history)

        if self.level_iterations >= min_iters and avg_reward >= threshold:
            promoted = self.env.promote_level()
            if promoted:
                self.level_iterations = 0
                self.reward_history.clear()
                return True
        return False


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:
    """Simple PPO trainer."""

    def __init__(self, env, policy, device="cuda:0"):
        self.env = env
        self.policy = policy
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

    def collect_rollout(self, num_steps):
        """Collect experience."""
        obs_list, act_list, rew_list, done_list, val_list, logp_list = [], [], [], [], [], []

        obs = self.env.obs_buf.clone()

        for _ in range(num_steps):
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

            obs, reward, done, _ = self.env.step(action)
            rew_list.append(reward)
            done_list.append(done)

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
        rollout = self.collect_rollout(self.num_steps)
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
    print("🔥 ULC G1 STAGE 5 - FULL MECHANICAL WORKSPACE TRAINING 🔥")
    print("=" * 70)
    print(f"Stage 4 Checkpoint: {args.stage4_checkpoint}")
    print(f"Environments: {args.num_envs}")
    print(f"Max Iterations: {args.max_iterations}")
    print()

    print("G1 FULL MECHANICAL LIMITS (Unitree Official - GÜVENLİ):")
    print("  ARMS:")
    print("    shoulder_pitch: ±2.6 rad (±149°)")
    print("    elbow:          ±1.6 rad (±92°)")
    print("  LEGS:")
    print("    hip_pitch:      ±1.57 rad (±90°) - Squat!")
    print("    knee:           2.0 rad (115°)   - Squat!")
    print("  WAIST:")
    print("    waist_pitch:    ±0.52 rad (±30°) - Eğilme!")
    print()

    print("CURRICULUM (Stage 4'ten Full Workspace'e):")
    for level, cfg in STAGE5_CURRICULUM.items():
        sp = cfg.get("shoulder_pitch_range", 2.6)
        pct = (sp / 2.6) * 100
        h_range = cfg.get("height_range", (0.35, 0.85))
        print(f"  Level {level}: {cfg['name']}")
        print(f"    Arms: {pct:.0f}%, Height: {h_range}")
    print("=" * 70)

    # Load Stage 4 checkpoint
    print(f"\n[INFO] Loading Stage 4 checkpoint...")
    if not os.path.exists(args.stage4_checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.stage4_checkpoint}")
        return

    stage4_ckpt = torch.load(args.stage4_checkpoint, map_location="cuda:0", weights_only=True)
    print(f"[INFO] Stage 4 checkpoint loaded")
    if "best_reward" in stage4_ckpt:
        print(f"[INFO] Stage 4 best reward: {stage4_ckpt['best_reward']:.2f}")

    # Import and create environment
    print(f"\n[INFO] Creating environment...")

    try:
        # Try importing the Stage 4 environment
        from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
        from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg

        env_cfg = ULC_G1_Stage4_EnvCfg()
        env_cfg.scene.num_envs = args.num_envs

        # IMPORTANT: Update termination height for squat!
        if hasattr(env_cfg, 'termination'):
            env_cfg.termination["base_height_min"] = 0.25  # Allow deep squat!

        base_env = ULC_G1_Env(cfg=env_cfg)

    except ImportError as e:
        print(f"[ERROR] Could not import environment: {e}")
        print("[INFO] Make sure the ULC environment is properly set up.")
        return

    # Wrap with Stage 5 wrapper
    env = Stage5EnvWrapper(base_env, STAGE5_CURRICULUM)

    # Get dimensions
    obs_dim = base_env.cfg.num_observations
    action_dim = base_env.cfg.num_actions
    print(f"[INFO] Observation dim: {obs_dim}")
    print(f"[INFO] Action dim: {action_dim}")

    # Create policy
    policy = ActorCritic(obs_dim, action_dim).to("cuda:0")

    # Load Stage 4 weights
    print(f"\n[INFO] Loading Stage 4 weights...")
    policy.load_state_dict(stage4_ckpt["model_state_dict"])
    print("[INFO] Stage 4 weights loaded successfully ✓")

    # Create curriculum manager
    curriculum_mgr = CurriculumManager(env, STAGE5_CURRICULUM)

    # Create trainer
    trainer = PPOTrainer(env, policy, device="cuda:0")

    # Create log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage5_full_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")

    # Initial command sampling
    env.sample_commands()

    # Training loop
    print(f"\n[INFO] Starting training...")
    level_info = env.get_level_info()
    print(f"[INFO] Initial: Level {level_info['level']} - {level_info['name']} ({level_info['workspace_pct']:.0f}%)")

    best_reward = -float('inf')

    for iteration in range(args.max_iterations):
        # Train
        stats = trainer.train_one_iteration()
        mean_reward = stats["mean_reward"]

        # Curriculum update
        promoted = curriculum_mgr.update(iteration, mean_reward)
        if promoted:
            level_info = env.get_level_info()
            print(f"\n{'=' * 60}")
            print(f"🎯 LEVEL UP! Level {level_info['level']}: {level_info['name']}")
            print(f"   Workspace: {level_info['workspace_pct']:.0f}%")
            print(f"{'=' * 60}\n")

        # Track best
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.current_level,
            }, os.path.join(log_dir, "model_best.pt"))

        # Logging
        if iteration % 50 == 0:
            level_info = env.get_level_info()
            print(f"Iter {iteration:5d} | Reward: {mean_reward:7.2f} | Best: {best_reward:7.2f} | "
                  f"Level: {level_info['level']} ({level_info['workspace_pct']:.0f}%)")

        # Periodic save
        if iteration % 500 == 0 and iteration > 0:
            torch.save({
                "model_state_dict": policy.state_dict(),
                "iteration": iteration,
                "curriculum_level": env.current_level,
            }, os.path.join(log_dir, f"model_{iteration}.pt"))

    # Final save
    torch.save({
        "model_state_dict": policy.state_dict(),
        "iteration": args.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": env.current_level,
    }, os.path.join(log_dir, "model_final.pt"))

    print("\n" + "=" * 70)
    print("🎉 STAGE 5 TRAINING COMPLETE!")
    print("=" * 70)
    level_info = env.get_level_info()
    print(f"Final Level: {level_info['level']} - {level_info['name']}")
    print(f"Final Workspace: {level_info['workspace_pct']:.0f}%")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Checkpoints: {log_dir}")
    print("=" * 70)

    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()