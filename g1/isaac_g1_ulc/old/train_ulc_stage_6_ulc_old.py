#!/usr/bin/env python3
"""
ULC G1 Stage 6 Training - ARM TRACKING FIX + STABILITY
=======================================================

Stage 5 checkpoint'inden devam eder.
Kolların komutları takip etmesini sağlar + zıplamayı azaltır.

SORUNLAR (Stage 5):
1. Kollar komutları takip etmiyor (residual scale çok küçük)
2. Robot zıplıyor/titriyor (vertical velocity penalty yok)
3. Policy stabilitey tercih ediyor (arm tracking reward düşük)

ÇÖZÜMLER:
1. Residual scale artırıldı (0.5 → 1.5)
2. Vertical velocity penalty eklendi
3. Arm tracking reward artırıldı (4.0 → 10.0)
4. Action rate penalty artırıldı
5. Contact smoothness penalty eklendi

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_6.py ^
    --stage5_checkpoint logs/ulc/ulc_g1_stage5_2026-01-11_00-23-34/model_30000.pt ^
    --num_envs 4096 --headless --max_iterations 10000
"""

import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="ULC G1 Stage 6 - Arm Fix + Stability")
parser.add_argument("--stage5_checkpoint", type=str, required=True,
                    help="Path to Stage 5 model checkpoint")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from Stage 6 checkpoint")

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

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env, quat_to_euler_xyz
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[WARNING] TensorBoard not available")


# ============================================================
# STAGE 6 ENVIRONMENT - FIXED ARM TRACKING + STABILITY
# ============================================================

class ULC_G1_Stage6_Env(ULC_G1_Env):
    """
    Stage 6 Environment with fixed arm tracking and stability.

    Changes from Stage 5:
    1. Larger residual scales for arms (1.5 vs 0.5)
    2. Higher arm tracking reward (10.0 vs 4.0)
    3. Vertical velocity penalty (anti-bouncing)
    4. Smoother action rate penalty
    5. Foot contact force penalty
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # FIXED: Larger residual scales for better arm tracking
        self.residual_scales = torch.tensor(
            [1.5, 1.0, 1.0, 1.2, 0.8,  # Left arm: shoulder_p, shoulder_r, shoulder_y, elbow_p, elbow_r
             1.5, 1.0, 1.0, 1.2, 0.8],  # Right arm
            device=self.device
        )

        # Action history for smoothness
        self.action_history = torch.zeros(self.num_envs, 3, self.cfg.num_actions, device=self.device)

        print(f"[Stage 6] Residual scales: {self.residual_scales[:5].tolist()}")
        print(f"[Stage 6] Arm tracking enabled with high reward weight")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions with LARGER residual scales for arms."""
        self.actions = torch.clamp(actions, -1.0, 1.0)

        # Split actions
        leg_actions = self.actions[:, :12]
        arm_actions = self.actions[:, 12:]

        # Update action history (for smoothness calculation)
        self.action_history = torch.roll(self.action_history, 1, dims=1)
        self.action_history[:, 0] = self.actions

        # Compute targets
        target_pos = self.robot.data.default_joint_pos.clone()

        # Legs: direct action
        target_pos[:, self.leg_joint_indices] = self.default_leg + leg_actions * 0.4

        # Arms: FIXED - Larger residual range for better tracking
        arm_cmd = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)

        # CHANGED: Direct scaling without double tanh
        # Policy output [-1, 1] → residual [-residual_scale, +residual_scale]
        arm_residual = arm_actions * self.residual_scales
        arm_target = arm_cmd + arm_residual

        # Clamp to joint limits (approximate)
        arm_target = torch.clamp(arm_target, -2.6, 2.6)

        target_pos[:, self.arm_joint_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)

        # Update gait phase
        self.gait_phase = (self.gait_phase + self.gait_frequency * self.step_dt) % 1.0

        # Store action history
        self._prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with ENHANCED arm tracking and stability penalties."""
        robot = self.robot
        quat = robot.data.root_quat_w
        pos = robot.data.root_pos_w

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # Joint states
        leg_pos = robot.data.joint_pos[:, self.leg_joint_indices]
        leg_vel = robot.data.joint_vel[:, self.leg_joint_indices]
        left_arm_pos = robot.data.joint_pos[:, self.left_arm_indices]
        right_arm_pos = robot.data.joint_pos[:, self.right_arm_indices]

        # Torso orientation
        torso_euler = self.get_torso_euler()

        rewards = {}

        # ==================== VELOCITY TRACKING ====================
        rewards["vx"] = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.velocity_commands[:, 0]) ** 2)
        rewards["vy"] = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.velocity_commands[:, 1]) ** 2)
        rewards["vyaw"] = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.velocity_commands[:, 2]) ** 2)

        # ==================== HEIGHT TRACKING ====================
        rewards["height_tracking"] = torch.exp(-10.0 * (pos[:, 2] - self.height_command) ** 2)

        # ==================== BASE ORIENTATION ====================
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        base_tilt_error = proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2
        rewards["orientation"] = torch.exp(-3.0 * base_tilt_error)

        # ==================== TORSO TRACKING ====================
        roll_err = (torso_euler[:, 0] - self.torso_commands[:, 0]) ** 2
        pitch_err = (torso_euler[:, 1] - self.torso_commands[:, 1]) ** 2
        yaw_err = (torso_euler[:, 2] - self.torso_commands[:, 2]) ** 2

        rewards["torso_roll"] = torch.exp(-5.0 * roll_err)
        rewards["torso_pitch"] = torch.exp(-5.0 * pitch_err)
        rewards["torso_yaw"] = torch.exp(-3.0 * yaw_err)

        # ==================== ARM TRACKING (ENHANCED) ====================
        left_arm_err = (left_arm_pos - self.left_arm_cmd).pow(2).mean(-1)  # Mean instead of sum
        right_arm_err = (right_arm_pos - self.right_arm_cmd).pow(2).mean(-1)

        # CHANGED: Use softer exponential for better gradient when far from target
        rewards["left_arm"] = torch.exp(-2.0 * left_arm_err)  # Was -3.0
        rewards["right_arm"] = torch.exp(-2.0 * right_arm_err)

        # Additional: Linear arm tracking reward (provides gradient even when far)
        rewards["left_arm_linear"] = 1.0 - torch.clamp(left_arm_err, 0, 1)
        rewards["right_arm_linear"] = 1.0 - torch.clamp(right_arm_err, 0, 1)

        # ==================== GAIT QUALITY ====================
        left_knee, right_knee = leg_pos[:, 6], leg_pos[:, 7]
        phase = self.gait_phase
        left_swing = (phase < 0.5).float()
        right_swing = (phase >= 0.5).float()

        knee_target_swing = 0.6
        knee_target_stance = 0.3
        knee_err = (
                (left_knee - (left_swing * knee_target_swing + (1 - left_swing) * knee_target_stance)) ** 2 +
                (right_knee - (right_swing * knee_target_swing + (1 - right_swing) * knee_target_stance)) ** 2
        )
        rewards["gait"] = torch.exp(-3.0 * knee_err)

        # ==================== STABILITY REWARDS ====================
        # CoM stability (reduced weight - don't over-penalize arm movement)
        xy_velocity = torch.norm(lin_vel_b[:, :2], dim=-1)
        rewards["com_stability"] = torch.exp(-2.0 * xy_velocity ** 2)

        # ==================== NEW: ANTI-BOUNCING PENALTIES ====================
        # Vertical velocity penalty (CRITICAL for reducing bouncing)
        vertical_vel = lin_vel_b[:, 2].abs()
        rewards["vertical_vel_penalty"] = vertical_vel.pow(2)

        # Vertical acceleration penalty
        if hasattr(self, '_prev_vertical_vel'):
            vertical_accel = (lin_vel_b[:, 2] - self._prev_vertical_vel).abs()
            rewards["vertical_accel_penalty"] = vertical_accel.pow(2)
        else:
            rewards["vertical_accel_penalty"] = torch.zeros(self.num_envs, device=self.device)
        self._prev_vertical_vel = lin_vel_b[:, 2].clone()

        # ==================== ACTION SMOOTHNESS PENALTIES ====================
        # Leg action rate
        leg_action_diff = self.actions[:, :12] - self._prev_actions[:, :12]
        rewards["smooth_legs"] = leg_action_diff.pow(2).sum(-1)

        # Arm action rate (gentler - we want arms to move)
        arm_action_diff = self.actions[:, 12:] - self._prev_actions[:, 12:]
        rewards["smooth_arms"] = arm_action_diff.pow(2).sum(-1)

        # Second derivative smoothness (jerk penalty)
        if self.action_history.shape[1] >= 3:
            action_jerk = self.action_history[:, 0] - 2 * self.action_history[:, 1] + self.action_history[:, 2]
            rewards["action_jerk"] = action_jerk.pow(2).sum(-1)
        else:
            rewards["action_jerk"] = torch.zeros(self.num_envs, device=self.device)

        # Torque penalty
        rewards["torque"] = (leg_vel.abs() * self.actions[:, :12].abs()).sum(-1)

        # ==================== TOTAL REWARD ====================
        reward_weights = {
            # Tracking rewards
            "vx": 2.0,
            "vy": 1.0,
            "vyaw": 1.0,
            "gait": 1.5,
            "height_tracking": 2.0,
            "orientation": 1.5,
            "torso_pitch": 2.0,
            "torso_roll": 1.5,
            "torso_yaw": 1.0,

            # ARM TRACKING - SIGNIFICANTLY INCREASED
            "left_arm": 8.0,  # Was 4.0
            "right_arm": 8.0,  # Was 4.0
            "left_arm_linear": 3.0,  # NEW: Linear reward
            "right_arm_linear": 3.0,  # NEW: Linear reward

            # Stability (reduced to allow arm movement)
            "com_stability": 2.0,  # Was 4.0

            # Anti-bouncing penalties (NEW)
            "vertical_vel_penalty": -5.0,  # NEW
            "vertical_accel_penalty": -2.0,  # NEW

            # Smoothness penalties
            "smooth_legs": -0.02,  # Increased from -0.01
            "smooth_arms": -0.005,  # Keep gentle
            "action_jerk": -0.01,  # NEW
            "torque": -0.0003,
        }

        total_reward = torch.zeros(self.num_envs, device=self.device)
        for key, weight in reward_weights.items():
            if key in rewards:
                total_reward += weight * rewards[key]

        # Alive bonus
        total_reward += 0.5

        # Store extras for logging
        self.extras = {
            "R/vx": rewards["vx"].mean().item(),
            "R/height": rewards["height_tracking"].mean().item(),
            "R/left_arm": rewards["left_arm"].mean().item(),
            "R/right_arm": rewards["right_arm"].mean().item(),
            "R/com_stability": rewards["com_stability"].mean().item(),
            "R/vertical_vel": rewards["vertical_vel_penalty"].mean().item(),
            "M/height": pos[:, 2].mean().item(),
            "M/vx": lin_vel_b[:, 0].mean().item(),
            "M/vz": lin_vel_b[:, 2].mean().item(),  # NEW: Track vertical velocity
            "M/pitch": torso_euler[:, 1].mean().item(),
            "M/left_arm_err": left_arm_err.mean().item(),
            "M/right_arm_err": right_arm_err.mean().item(),
            "M/arm_err": (left_arm_err + right_arm_err).mean().item() / 2,
        }

        return total_reward.clamp(-10, 40)


# ============================================================
# ACTOR-CRITIC (Same as Stage 5)
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
# CURRICULUM MANAGER (Simplified for Stage 6)
# ============================================================

class Stage6CurriculumManager:
    """Simple curriculum for arm tracking fine-tuning."""

    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.level = 0
        self.rewards = deque(maxlen=100)

        # Arm range progression
        self.arm_ranges = [0.8, 1.2, 1.8, 2.4, 2.6]
        self.thresholds = [22.0, 21.0, 20.0, 19.0, None]
        self.min_iters = [200, 300, 400, 500, None]
        self.level_iters = 0

        self._apply_level(0)

    def _apply_level(self, level):
        self.level = level
        arm_range = self.arm_ranges[level]
        print(f"\n{'=' * 60}")
        print(f"[Stage 6 Curriculum] Level {level}: Arm range ±{arm_range:.1f} rad")
        print(f"{'=' * 60}\n")

    def sample_commands(self):
        """Sample commands with current arm range."""
        n = self.env.num_envs
        d = self.device
        ar = self.arm_ranges[self.level]

        # Height (moderate range for stability)
        self.env.height_command[:] = torch.empty(n, device=d).uniform_(0.60, 0.80)

        # Velocity (reduced for arm tracking focus)
        self.env.velocity_commands[:, 0] = torch.empty(n, device=d).uniform_(-0.3, 0.5)
        self.env.velocity_commands[:, 1] = torch.empty(n, device=d).uniform_(-0.2, 0.2)
        self.env.velocity_commands[:, 2] = torch.empty(n, device=d).uniform_(-0.3, 0.3)

        # Torso (minimal)
        self.env.torso_commands[:, 0] = torch.empty(n, device=d).uniform_(-0.2, 0.2)
        self.env.torso_commands[:, 1] = torch.empty(n, device=d).uniform_(-0.2, 0.2)
        self.env.torso_commands[:, 2] = torch.empty(n, device=d).uniform_(-0.3, 0.3)

        # Arms - MAIN FOCUS
        # Shoulder pitch (main movement)
        self.env.left_arm_cmd[:, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.right_arm_cmd[:, 0] = torch.empty(n, device=d).uniform_(-ar, ar)

        # Shoulder roll/yaw (limited)
        self.env.left_arm_cmd[:, 1] = torch.empty(n, device=d).uniform_(-ar * 0.5, ar * 0.5)
        self.env.left_arm_cmd[:, 2] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)
        self.env.right_arm_cmd[:, 1] = torch.empty(n, device=d).uniform_(-ar * 0.5, ar * 0.5)
        self.env.right_arm_cmd[:, 2] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)

        # Elbow pitch/roll
        self.env.left_arm_cmd[:, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.left_arm_cmd[:, 4] = torch.empty(n, device=d).uniform_(-ar * 0.3, ar * 0.3)
        self.env.right_arm_cmd[:, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.right_arm_cmd[:, 4] = torch.empty(n, device=d).uniform_(-ar * 0.3, ar * 0.3)

        # Update combined
        self.env.arm_commands = torch.cat([self.env.left_arm_cmd, self.env.right_arm_cmd], dim=-1)

    def resample(self, ids):
        """Resample for reset environments."""
        if len(ids) == 0:
            return

        n = len(ids)
        d = self.device
        ar = self.arm_ranges[self.level]

        self.env.height_command[ids] = torch.empty(n, device=d).uniform_(0.60, 0.80)

        self.env.velocity_commands[ids, 0] = torch.empty(n, device=d).uniform_(-0.3, 0.5)
        self.env.velocity_commands[ids, 1] = torch.empty(n, device=d).uniform_(-0.2, 0.2)
        self.env.velocity_commands[ids, 2] = torch.empty(n, device=d).uniform_(-0.3, 0.3)

        self.env.torso_commands[ids, 0] = torch.empty(n, device=d).uniform_(-0.2, 0.2)
        self.env.torso_commands[ids, 1] = torch.empty(n, device=d).uniform_(-0.2, 0.2)
        self.env.torso_commands[ids, 2] = torch.empty(n, device=d).uniform_(-0.3, 0.3)

        self.env.left_arm_cmd[ids, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.left_arm_cmd[ids, 1] = torch.empty(n, device=d).uniform_(-ar * 0.5, ar * 0.5)
        self.env.left_arm_cmd[ids, 2] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)
        self.env.left_arm_cmd[ids, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.left_arm_cmd[ids, 4] = torch.empty(n, device=d).uniform_(-ar * 0.3, ar * 0.3)

        self.env.right_arm_cmd[ids, 0] = torch.empty(n, device=d).uniform_(-ar, ar)
        self.env.right_arm_cmd[ids, 1] = torch.empty(n, device=d).uniform_(-ar * 0.5, ar * 0.5)
        self.env.right_arm_cmd[ids, 2] = torch.empty(n, device=d).uniform_(-ar * 0.4, ar * 0.4)
        self.env.right_arm_cmd[ids, 3] = torch.empty(n, device=d).uniform_(-ar * 0.6, ar * 0.6)
        self.env.right_arm_cmd[ids, 4] = torch.empty(n, device=d).uniform_(-ar * 0.3, ar * 0.3)

        self.env.arm_commands[ids] = torch.cat([self.env.left_arm_cmd[ids], self.env.right_arm_cmd[ids]], dim=-1)

    def update(self, reward):
        """Update curriculum level."""
        self.level_iters += 1
        self.rewards.append(reward)

        if len(self.rewards) < 50:
            return False

        if self.level >= len(self.arm_ranges) - 1:
            return False

        avg = sum(self.rewards) / len(self.rewards)
        threshold = self.thresholds[self.level]
        min_iter = self.min_iters[self.level]

        if threshold is not None and self.level_iters >= min_iter and avg >= threshold:
            self.level += 1
            self._apply_level(self.level)
            self.level_iters = 0
            self.rewards.clear()
            return True

        return False


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:
    def __init__(self, env, policy, curriculum, device="cuda:0"):
        self.env = env
        self.policy = policy
        self.curriculum = curriculum
        self.device = device

        # Lower learning rate for fine-tuning
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.max_iterations, eta_min=1e-6
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

                val_clipped = old_val[idx] + (val - old_val[idx]).clamp(-0.2, 0.2)
                critic_loss = 0.5 * torch.max(
                    (val - ret_flat[idx]) ** 2,
                    (val_clipped - ret_flat[idx]) ** 2
                ).mean()

                # Slightly higher entropy bonus for exploration during fine-tuning
                loss = actor_loss + 0.5 * critic_loss - 0.015 * ent.mean()

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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    print("\n" + "=" * 70)
    print("🔧 ULC G1 STAGE 6 - ARM TRACKING FIX + STABILITY 🔧")
    print("=" * 70)
    print(f"Stage 5 checkpoint: {args.stage5_checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print()
    print("HEDEFLER:")
    print("  ✓ Kolların komutları takip etmesi")
    print("  ✓ Zıplama/titreme azaltma")
    print("  ✓ Smooth hareketler")
    print("=" * 70)

    # Load checkpoint
    print(f"\n[INFO] Loading Stage 5 checkpoint...")
    if not os.path.exists(args.stage5_checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.stage5_checkpoint}")
        return

    ckpt = torch.load(args.stage5_checkpoint, map_location="cuda:0", weights_only=False)
    state_dict = ckpt.get("actor_critic", ckpt)
    ckpt_obs = state_dict["actor.0.weight"].shape[1]
    ckpt_act = state_dict["log_std"].shape[0]
    print(f"[INFO] Checkpoint dims: obs={ckpt_obs}, act={ckpt_act}")

    # Create environment with Stage 6 modifications
    print(f"\n[INFO] Creating Stage 6 environment...")
    cfg = ULC_G1_Stage4_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.termination["base_height_min"] = 0.3
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(cfg.num_observations,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_actions,))

    env = ULC_G1_Stage6_Env(cfg=cfg)
    env.current_stage = 4

    # Verify dimensions
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[1]
    action_dim = cfg.num_actions

    if ckpt_obs != obs_dim or ckpt_act != action_dim:
        print(f"\n[ERROR] Dimension mismatch!")
        env.close()
        simulation_app.close()
        return

    # Create and load policy
    policy = ActorCritic(ckpt_obs, ckpt_act).to("cuda:0")
    policy.load_state_dict(state_dict)
    print("[INFO] ✓ Stage 5 weights loaded!")

    # Resume from Stage 6 checkpoint if provided
    start_iter = 0
    if args.checkpoint:
        print(f"\n[INFO] Resuming from: {args.checkpoint}")
        s6_ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
        policy.load_state_dict(s6_ckpt["actor_critic"])
        start_iter = s6_ckpt.get("iteration", 0) + 1
        print(f"[INFO] Resuming from iteration {start_iter}")

    # Setup curriculum
    curriculum = Stage6CurriculumManager(env)
    curriculum.sample_commands()

    # Setup trainer
    trainer = PPOTrainer(env, policy, curriculum)

    # Log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage6_{timestamp}"
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
    print("STARTING STAGE 6 TRAINING")
    print("=" * 70 + "\n")

    for iteration in range(start_iter, args.max_iterations):
        iter_start = datetime.now()

        reward, update_info = trainer.train_iter()

        iter_time = (datetime.now() - iter_start).total_seconds()
        fps = trainer.steps * args.num_envs / iter_time

        level_up = curriculum.update(reward)
        if level_up:
            print(f"\n🎯 LEVEL UP! → Arm range ±{curriculum.arm_ranges[curriculum.level]:.1f} rad\n")

        if reward > best_reward:
            best_reward = reward
            torch.save({
                "actor_critic": policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": curriculum.level,
            }, f"{log_dir}/model_best.pt")

        if writer:
            writer.add_scalar("Train/reward", reward, iteration)
            writer.add_scalar("Train/best_reward", best_reward, iteration)
            writer.add_scalar("Curriculum/level", curriculum.level, iteration)
            writer.add_scalar("Loss/actor", update_info["actor_loss"], iteration)
            writer.add_scalar("Loss/critic", update_info["critic_loss"], iteration)
            writer.add_scalar("Train/entropy", update_info["entropy"], iteration)
            writer.add_scalar("Train/lr", update_info["lr"], iteration)
            writer.add_scalar("Train/fps", fps, iteration)

            for key, value in env.extras.items():
                writer.add_scalar(f"Env/{key}", value, iteration)

        if iteration % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            iters_done = iteration - start_iter + 1
            iters_remaining = args.max_iterations - iteration - 1
            eta = elapsed / iters_done * iters_remaining if iters_done > 0 else 0

            height = env.extras.get("M/height", 0)
            vz = env.extras.get("M/vz", 0)
            arm_err = env.extras.get("M/arm_err", 0)

            print(
                f"#{iteration:5d} | "
                f"R={reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Lv={curriculum.level} | "
                f"H={height:.2f} | "
                f"Vz={vz:.3f} | "
                f"ArmErr={arm_err:.3f} | "
                f"FPS={fps:5.0f} | "
                f"{format_time(elapsed)} / {format_time(eta)}"
            )

        if (iteration + 1) % 500 == 0:
            torch.save({
                "actor_critic": policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": curriculum.level,
            }, f"{log_dir}/model_{iteration + 1}.pt")
            print(f"[SAVE] Checkpoint: model_{iteration + 1}.pt")

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
    print(f"🎉 STAGE 6 TRAINING COMPLETE!")
    print(f"   Total Time: {format_time(total_time)}")
    print(f"   Best Reward: {best_reward:.2f}")
    print(f"   Final Arm Range: ±{curriculum.arm_ranges[curriculum.level]:.1f} rad")
    print(f"   Saved to: {log_dir}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()