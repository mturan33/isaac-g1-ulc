#!/usr/bin/env python3
"""
ULC G1 Stage 4 - Hierarchical Training
=======================================

STAGE 3 → STAGE 4 (Hierarchical DiffIK)

MİMARİ:
- Kollar: DiffIK ile kontrol (deterministic)
- Bacaklar: RL policy ile kontrol (Stage 3'ten transfer)
- Policy sadece dengeyi öğreniyor

YENİ ÖZELLİK: Arm Target Height in Observation!
- Robot kol hedefinin yüksekliğini görüyor
- Hedef aşağıda → robot eğilmeyi öğreniyor
- Bu sayede manipulation için daha geniş workspace

OBSERVATION (65 dims):
- [0:51]  Stage 3 base observations
- [51:54] left_ee_offset (from base)
- [54:57] right_ee_offset (from base)
- [57:60] left_target_offset
- [60:63] right_target_offset
- [63]    left_target_relative_height (target_z - base_z)
- [64]    right_target_relative_height

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../train/train_ulc_stage_4_hierarchical.py ^
    --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt ^
    --num_envs 4096 --headless --max_iterations 10000
"""

import argparse
import os
import sys
import time
import math

parser = argparse.ArgumentParser(description="ULC G1 Stage 4 Hierarchical")
parser.add_argument("--stage3_checkpoint", type=str, required=True,
                    help="Stage 3 checkpoint path")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from datetime import datetime
import gymnasium as gym

# Import base environment
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_EnvCfg


# ============================================================
# HIERARCHICAL ENVIRONMENT
# ============================================================

class HierarchicalG1Env(ULC_G1_Env):
    """
    Hierarchical G1 Environment for Stage 4

    Key Features:
    - DiffIK controls arms (deterministic, smooth)
    - RL policy controls legs only (12 joints)
    - Arm target HEIGHT included in observation
    - Robot learns to crouch when arm targets are low

    Observation (65 dims):
    - Base state (51 dims from Stage 3)
    - Arm EE offsets (6 dims)
    - Arm target offsets (6 dims)
    - Arm target relative heights (2 dims) ← NEW!
    """

    def __init__(self, cfg, **kwargs):
        # Override for hierarchical
        cfg.num_actions = 12  # Only leg joints
        cfg.num_observations = 65  # Stage 3 (51) + arm info (14)

        super().__init__(cfg, **kwargs)

        # ============================================================
        # DiffIK Parameters
        # ============================================================
        self.diffik_damping = 0.05
        self.diffik_max_delta = 0.02  # Conservative for stability

        # Find indices
        self._find_ee_indices()

        # ============================================================
        # Arm State
        # ============================================================
        self.left_arm_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_arm_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_arm_joints = torch.zeros(self.num_envs, 5, device=self.device)
        self.right_arm_joints = torch.zeros(self.num_envs, 5, device=self.device)

        # Initial offsets
        self.init_left_offset = None
        self.init_right_offset = None

        # ============================================================
        # Target Timing (SLOW changes!)
        # ============================================================
        self.target_hold_time = 3.0  # seconds
        self.target_timer = torch.zeros(self.num_envs, device=self.device)
        self.target_transition_time = 1.5  # smooth transition
        self.transition_progress = torch.ones(self.num_envs, device=self.device)

        self.prev_left_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_right_target = torch.zeros(self.num_envs, 3, device=self.device)

        # ============================================================
        # Workspace (includes vertical range for crouching!)
        # ============================================================
        # Relative to initial EE position
        self.arm_workspace_min = torch.tensor([-0.15, -0.3, -0.25], device=self.device)  # Can go DOWN
        self.arm_workspace_max = torch.tensor([0.3, 0.3, 0.2], device=self.device)

        # ============================================================
        # Curriculum
        # ============================================================
        self.curriculum_level = 0
        self.arm_amplitude = 0.05  # Start small
        self.height_range = 0.05  # Start with small height changes

        # Previous actions
        self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)

        print(f"\n{'=' * 60}")
        print("  HIERARCHICAL G1 ENVIRONMENT - STAGE 4")
        print(f"{'=' * 60}")
        print(f"  Actions: 12 (legs only)")
        print(f"  Observations: 65 (base + legs + arm EE + arm HEIGHT)")
        print(f"  DiffIK damping: {self.diffik_damping}")
        print(f"  Target hold time: {self.target_hold_time}s")
        print(f"  Workspace Z range: [{self.arm_workspace_min[2]:.2f}, {self.arm_workspace_max[2]:.2f}]m")
        print(f"{'=' * 60}\n")

    def _find_ee_indices(self):
        """Find end-effector body indices."""
        body_names = self.robot.data.body_names

        self.left_ee_idx = None
        self.right_ee_idx = None

        for i, name in enumerate(body_names):
            if "left_wrist_yaw_link" in name or "left_palm" in name:
                self.left_ee_idx = i
            elif "right_wrist_yaw_link" in name or "right_palm" in name:
                self.right_ee_idx = i

        # Fallback
        if self.left_ee_idx is None:
            self.left_ee_idx = 28
        if self.right_ee_idx is None:
            self.right_ee_idx = 29

        print(f"[Hierarchical] Left EE index: {self.left_ee_idx}")
        print(f"[Hierarchical] Right EE index: {self.right_ee_idx}")

    def _reset_idx(self, env_ids):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) > 0:
            base_pos = self.robot.data.root_pos_w[env_ids, :3]
            left_ee = self.robot.data.body_pos_w[env_ids, self.left_ee_idx]
            right_ee = self.robot.data.body_pos_w[env_ids, self.right_ee_idx]

            # Store initial offsets
            if self.init_left_offset is None:
                self.init_left_offset = (left_ee - base_pos).clone()
                self.init_right_offset = (right_ee - base_pos).clone()
                self.init_left_offset = self.init_left_offset.expand(self.num_envs, -1).clone()
                self.init_right_offset = self.init_right_offset.expand(self.num_envs, -1).clone()

            # Reset targets
            self.left_arm_target[env_ids] = self.init_left_offset[env_ids].clone()
            self.right_arm_target[env_ids] = self.init_right_offset[env_ids].clone()
            self.prev_left_target[env_ids] = self.init_left_offset[env_ids].clone()
            self.prev_right_target[env_ids] = self.init_right_offset[env_ids].clone()

            # Reset arm joints
            self.left_arm_joints[env_ids] = self.robot.data.joint_pos[env_ids][:, self.left_arm_indices]
            self.right_arm_joints[env_ids] = self.robot.data.joint_pos[env_ids][:, self.right_arm_indices]

            # Reset timers
            self.target_timer[env_ids] = 0.0
            self.transition_progress[env_ids] = 1.0

            self._prev_leg_actions[env_ids] = 0.0

    def _get_observations(self) -> dict:
        """
        Get observations with arm target HEIGHT.

        Observation (65 dims):
        - [0:3]   base_lin_vel_b
        - [3:6]   base_ang_vel_b
        - [6:9]   projected_gravity
        - [9:21]  leg_joint_pos (normalized)
        - [21:33] leg_joint_vel (scaled)
        - [33]    height_command
        - [34:37] velocity_commands
        - [37:39] gait_phase
        - [39:51] prev_leg_actions
        - [51:54] left_ee_offset
        - [54:57] right_ee_offset
        - [57:60] left_target_offset
        - [60:63] right_target_offset
        - [63]    left_target_z (relative to base) ← CRITICAL FOR CROUCHING!
        - [64]    right_target_z (relative to base)
        """
        # Base state
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b

        # Leg joints
        leg_pos = self.robot.data.joint_pos[:, self.leg_joint_indices]
        leg_pos_norm = (leg_pos - self.default_leg) / 0.5
        leg_vel = self.robot.data.joint_vel[:, self.leg_joint_indices] * 0.1

        # Commands
        height_cmd = self.height_command.unsqueeze(-1)
        vel_cmd = self.velocity_commands

        # Gait phase
        gait_sin = torch.sin(2 * math.pi * self.gait_phase).unsqueeze(-1)
        gait_cos = torch.cos(2 * math.pi * self.gait_phase).unsqueeze(-1)

        # Arm EE positions
        base_pos = self.robot.data.root_pos_w[:, :3]
        left_ee_offset = self.robot.data.body_pos_w[:, self.left_ee_idx] - base_pos
        right_ee_offset = self.robot.data.body_pos_w[:, self.right_ee_idx] - base_pos

        # Interpolated targets
        t = self.transition_progress.unsqueeze(-1).clamp(0, 1)
        current_left = self.prev_left_target * (1 - t) + self.left_arm_target * t
        current_right = self.prev_right_target * (1 - t) + self.right_arm_target * t

        # Target Z relative to base (CRITICAL!)
        # Negative = target is below base → robot should crouch
        left_target_z = current_left[:, 2:3]  # (num_envs, 1)
        right_target_z = current_right[:, 2:3]

        obs = torch.cat([
            base_lin_vel,  # 3
            base_ang_vel,  # 3
            proj_gravity,  # 3
            leg_pos_norm,  # 12
            leg_vel,  # 12
            height_cmd,  # 1
            vel_cmd,  # 3
            gait_sin,  # 1
            gait_cos,  # 1
            self._prev_leg_actions,  # 12
            left_ee_offset,  # 3
            right_ee_offset,  # 3
            current_left,  # 3
            current_right,  # 3
            left_target_z,  # 1 ← NEW
            right_target_z,  # 1 ← NEW
        ], dim=-1)

        return {"policy": obs}

    def _update_arm_targets(self):
        """Update arm targets with slow transitions."""
        self.target_timer += self.step_dt
        self.transition_progress += self.step_dt / self.target_transition_time
        self.transition_progress = self.transition_progress.clamp(0, 1)

        need_new_target = self.target_timer >= self.target_hold_time
        new_ids = need_new_target.nonzero(as_tuple=False).squeeze(-1)

        if len(new_ids) > 0:
            n = len(new_ids)

            # Store current as previous
            self.prev_left_target[new_ids] = self.left_arm_target[new_ids].clone()
            self.prev_right_target[new_ids] = self.right_arm_target[new_ids].clone()

            # Generate new offsets
            # XY: small amplitude
            # Z: can include height changes (crouching)
            xy_offset = (torch.rand(n, 2, device=self.device) - 0.5) * 2 * self.arm_amplitude
            z_offset = (torch.rand(n, 1, device=self.device) - 0.5) * 2 * self.height_range

            random_offset = torch.cat([xy_offset, z_offset], dim=-1)

            if self.init_left_offset is not None:
                self.left_arm_target[new_ids] = self.init_left_offset[new_ids] + random_offset
                self.right_arm_target[new_ids] = self.init_right_offset[new_ids] + random_offset

                # Clamp
                self.left_arm_target[new_ids] = torch.clamp(
                    self.left_arm_target[new_ids],
                    self.init_left_offset[new_ids] + self.arm_workspace_min,
                    self.init_left_offset[new_ids] + self.arm_workspace_max
                )
                self.right_arm_target[new_ids] = torch.clamp(
                    self.right_arm_target[new_ids],
                    self.init_right_offset[new_ids] + self.arm_workspace_min,
                    self.init_right_offset[new_ids] + self.arm_workspace_max
                )

            # Reset timers
            self.target_timer[new_ids] = 0.0
            self.transition_progress[new_ids] = 0.0

    def _apply_diffik(self):
        """Apply DiffIK using Damped Least Squares."""
        base_pos = self.robot.data.root_pos_w[:, :3]

        # Interpolated targets
        t = self.transition_progress.unsqueeze(-1).clamp(0, 1)
        current_left = self.prev_left_target * (1 - t) + self.left_arm_target * t
        current_right = self.prev_right_target * (1 - t) + self.right_arm_target * t

        # Errors
        left_ee = self.robot.data.body_pos_w[:, self.left_ee_idx]
        left_error = (base_pos + current_left) - left_ee

        right_ee = self.robot.data.body_pos_w[:, self.right_ee_idx]
        right_error = (base_pos + current_right) - right_ee

        try:
            jacobians = self.robot.root_physx_view.get_jacobians()

            # Left IK
            left_J = jacobians[:, self.left_ee_idx, :3, :][:, :, self.left_arm_indices]
            left_delta = self._damped_ls(left_J, left_error)

            # Right IK
            right_J = jacobians[:, self.right_ee_idx, :3, :][:, :, self.right_arm_indices]
            right_delta = self._damped_ls(right_J, right_error)

            # Update
            self.left_arm_joints = torch.clamp(self.left_arm_joints + left_delta, -2.6, 2.6)
            self.right_arm_joints = torch.clamp(self.right_arm_joints + right_delta, -2.6, 2.6)

        except Exception as e:
            pass  # Keep current positions

    def _damped_ls(self, J, error):
        """Damped Least Squares IK."""
        batch = J.shape[0]
        JJT = torch.bmm(J, J.transpose(1, 2))
        damping = (self.diffik_damping ** 2) * torch.eye(3, device=self.device).unsqueeze(0).expand(batch, -1, -1)

        try:
            x = torch.linalg.solve(JJT + damping, error.unsqueeze(-1))
            delta = torch.bmm(J.transpose(1, 2), x).squeeze(-1)
            return torch.clamp(delta, -self.diffik_max_delta, self.diffik_max_delta)
        except:
            return torch.zeros(batch, J.shape[2], device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply leg actions + DiffIK arms."""
        self.actions = torch.clamp(actions, -1.0, 1.0)

        self._update_arm_targets()
        self._apply_diffik()

        # Leg targets
        leg_targets = self.default_leg + self.actions * 0.4

        # Full joint target
        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_joint_indices] = leg_targets
        target_pos[:, self.left_arm_indices] = self.left_arm_joints
        target_pos[:, self.right_arm_indices] = self.right_arm_joints

        self.robot.set_joint_position_target(target_pos)

        self.gait_phase = (self.gait_phase + self.gait_frequency * self.step_dt) % 1.0
        self._prev_leg_actions = self.actions.clone()

    def _compute_rewards(self):
        """Rewards focused on balance during arm movement."""
        rewards = {}

        # 1. Height tracking - but with adaptive target!
        height = self.robot.data.root_pos_w[:, 2]

        # Interpolated arm target Z
        t = self.transition_progress.unsqueeze(-1).clamp(0, 1)
        current_left_z = (self.prev_left_target[:, 2] * (1 - t.squeeze()) +
                          self.left_arm_target[:, 2] * t.squeeze())
        current_right_z = (self.prev_right_target[:, 2] * (1 - t.squeeze()) +
                           self.right_arm_target[:, 2] * t.squeeze())

        # Average arm target height
        avg_arm_z = (current_left_z + current_right_z) / 2

        # Adjusted height command: if arms go down, robot should be lower
        # But not too much - gradual adjustment
        height_adjustment = torch.clamp(avg_arm_z * 0.3, -0.1, 0.0)  # Max 10cm crouch
        adjusted_height_cmd = self.height_command + height_adjustment

        height_error = torch.abs(height - adjusted_height_cmd)
        rewards["height"] = torch.exp(-height_error * 5.0)

        # 2. Velocity tracking
        lin_vel = self.robot.data.root_lin_vel_b
        vel_error_x = torch.abs(lin_vel[:, 0] - self.velocity_commands[:, 0])
        vel_error_y = torch.abs(lin_vel[:, 1] - self.velocity_commands[:, 1])
        rewards["velocity_x"] = torch.exp(-vel_error_x * 3.0)
        rewards["velocity_y"] = torch.exp(-vel_error_y * 3.0)

        # 3. Orientation (CRITICAL!)
        proj_gravity = self.robot.data.projected_gravity_b
        orientation_error = torch.norm(proj_gravity[:, :2], dim=-1)
        rewards["orientation"] = torch.exp(-orientation_error * 5.0)

        # 4. Angular velocity
        ang_vel = torch.norm(self.robot.data.root_ang_vel_b, dim=-1)
        rewards["angular_vel"] = torch.exp(-ang_vel * 0.5)

        # 5. Vertical velocity
        vz = torch.abs(self.robot.data.root_lin_vel_w[:, 2])
        rewards["vertical_vel"] = torch.exp(-vz * 3.0)

        # 6. Action smoothness
        action_rate = torch.norm(self.actions - self._prev_leg_actions, dim=-1)
        rewards["action_smooth"] = torch.exp(-action_rate * 2.0)

        # 7. Arm tracking
        base_pos = self.robot.data.root_pos_w[:, :3]
        current_left = self.prev_left_target * (1 - t) + self.left_arm_target * t
        current_right = self.prev_right_target * (1 - t) + self.right_arm_target * t

        left_ee = self.robot.data.body_pos_w[:, self.left_ee_idx]
        right_ee = self.robot.data.body_pos_w[:, self.right_ee_idx]

        left_err = torch.norm(left_ee - (base_pos + current_left), dim=-1)
        right_err = torch.norm(right_ee - (base_pos + current_right), dim=-1)

        rewards["arm_tracking"] = torch.exp(-(left_err + right_err) * 5.0)

        # Weights
        weights = {
            "height": 2.0,
            "velocity_x": 1.0,
            "velocity_y": 1.0,
            "orientation": 4.0,
            "angular_vel": 1.0,
            "vertical_vel": 2.0,
            "action_smooth": 1.0,
            "arm_tracking": 1.5,
        }

        total = sum(weights[k] * rewards[k] for k in rewards)
        return total, rewards

    def update_curriculum(self, mean_reward):
        """Update curriculum - increase arm range and height range."""
        thresholds = [10.0, 12.0, 14.0, 16.0, 18.0]
        amplitudes = [0.05, 0.08, 0.12, 0.15, 0.18, 0.22]
        heights = [0.05, 0.08, 0.12, 0.15, 0.18, 0.22]

        if self.curriculum_level < len(thresholds):
            if mean_reward > thresholds[self.curriculum_level]:
                self.curriculum_level += 1
                self.arm_amplitude = amplitudes[self.curriculum_level]
                self.height_range = heights[self.curriculum_level]
                print(f"\n[CURRICULUM] Level {self.curriculum_level}:")
                print(f"  arm_amplitude = {self.arm_amplitude}m")
                print(f"  height_range = {self.height_range}m")


# ============================================================
# ACTOR-CRITIC NETWORK
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def act(self, obs, deterministic=False):
        mean = self.actor(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)

        return log_prob, entropy, value


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:
    def __init__(self, env, policy, device="cuda:0"):
        self.env = env
        self.policy = policy
        self.device = device

        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 1.0
        self.num_epochs = 5
        self.mini_batch_size = 4096
        self.horizon = 24

        self.optimizer = optim.Adam(policy.parameters(), lr=self.lr)
        self.best_reward = float('-inf')
        self.reward_history = deque(maxlen=100)

    def collect_rollouts(self):
        obs_list, actions_list, rewards_list = [], [], []
        values_list, log_probs_list, dones_list = [], [], []

        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]

        for _ in range(self.horizon):
            with torch.no_grad():
                action = self.policy.act(obs)
                value = self.policy.critic(obs).squeeze(-1)
                mean = self.policy.actor(obs)
                std = torch.exp(self.policy.log_std)
                log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(dim=-1)

            obs_list.append(obs)
            actions_list.append(action)
            values_list.append(value)
            log_probs_list.append(log_prob)

            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            obs = obs_dict["policy"]
            done = terminated | truncated

            rewards_list.append(reward)
            dones_list.append(done.float())

        obs_batch = torch.stack(obs_list)
        actions_batch = torch.stack(actions_list)
        rewards_batch = torch.stack(rewards_list)
        values_batch = torch.stack(values_list)
        log_probs_batch = torch.stack(log_probs_list)
        dones_batch = torch.stack(dones_list)

        with torch.no_grad():
            next_value = self.policy.critic(obs).squeeze(-1)

        returns, advantages = self._compute_gae(rewards_batch, values_batch, dones_batch, next_value)

        return {
            "obs": obs_batch.view(-1, obs_batch.shape[-1]),
            "actions": actions_batch.view(-1, actions_batch.shape[-1]),
            "returns": returns.view(-1),
            "advantages": advantages.view(-1),
            "old_log_probs": log_probs_batch.view(-1),
        }

    def _compute_gae(self, rewards, values, dones, next_value):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, rollout_data):
        total_samples = rollout_data["obs"].shape[0]
        indices = torch.randperm(total_samples, device=self.device)

        total_loss = 0
        num_updates = 0

        for epoch in range(self.num_epochs):
            for start in range(0, total_samples, self.mini_batch_size):
                batch_idx = indices[start:start + self.mini_batch_size]

                obs = rollout_data["obs"][batch_idx]
                actions = rollout_data["actions"][batch_idx]
                returns = rollout_data["returns"][batch_idx]
                advantages = rollout_data["advantages"][batch_idx]
                old_log_probs = rollout_data["old_log_probs"][batch_idx]

                log_probs, entropy, values = self.policy.evaluate(obs, actions)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)

    def train(self, max_iterations, log_dir, stage3_checkpoint):
        os.makedirs(log_dir, exist_ok=True)

        # Try to transfer weights from Stage 3
        print(f"\n[INFO] Loading Stage 3 checkpoint: {stage3_checkpoint}")
        if os.path.exists(stage3_checkpoint):
            ckpt = torch.load(stage3_checkpoint, map_location=self.device, weights_only=True)
            stage3_state = ckpt.get("actor_critic", ckpt)

            # Stage 3: obs=51, act=12
            # Stage 4: obs=65, act=12
            # We can transfer some weights!

            stage3_obs_dim = stage3_state["actor.0.weight"].shape[1]
            print(f"[INFO] Stage 3 obs dim: {stage3_obs_dim}")
            print(f"[INFO] Stage 4 obs dim: 65")

            # Can't directly transfer due to different obs dims
            # But action dim is same (12), so let's initialize randomly
            print("[INFO] Using fresh initialization (obs dim changed)")
        else:
            print(f"[WARN] Stage 3 checkpoint not found!")

        print(f"\n{'=' * 70}")
        print("  STAGE 4 HIERARCHICAL TRAINING")
        print(f"{'=' * 70}")
        print(f"  From: Stage 3 (locomotion + torso)")
        print(f"  New: DiffIK arms + arm height in obs")
        print(f"  Observations: 65")
        print(f"  Actions: 12 (legs only)")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Log dir: {log_dir}")
        print(f"{'=' * 70}\n")

        start_time = time.time()

        for iteration in range(max_iterations):
            rollout_data = self.collect_rollouts()
            loss = self.update(rollout_data)

            mean_reward = rollout_data["returns"].mean().item()
            self.reward_history.append(mean_reward)
            avg_reward = sum(self.reward_history) / len(self.reward_history)

            self.env.update_curriculum(avg_reward)

            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save(os.path.join(log_dir, "model_best.pt"))

            if (iteration + 1) % 500 == 0:
                self.save(os.path.join(log_dir, f"model_{iteration + 1}.pt"))

            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                fps = rollout_data["obs"].shape[0] / (elapsed / (iteration + 1)) if iteration > 0 else 0

                amp = self.env.arm_amplitude
                h_range = self.env.height_range
                level = self.env.curriculum_level

                print(f"[{iteration:5d}] R={mean_reward:.2f} (avg={avg_reward:.2f}) | "
                      f"Loss={loss:.4f} | Amp={amp:.2f} | H={h_range:.2f} | "
                      f"Lvl={level} | FPS={fps:.0f}")

        self.save(os.path.join(log_dir, "model_final.pt"))
        print(f"\n[DONE] Best: {self.best_reward:.2f}")

    def save(self, path):
        torch.save({
            "actor_critic": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_reward": self.best_reward,
            "curriculum_level": self.env.curriculum_level,
        }, path)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  ULC G1 STAGE 4 - HIERARCHICAL (DiffIK Arms)")
    print("=" * 70 + "\n")

    cfg = ULC_G1_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 20.0
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(65,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))

    env = HierarchicalG1Env(cfg=cfg)

    policy = ActorCritic(obs_dim=65, action_dim=12).to("cuda:0")
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"[INFO] Policy parameters: {num_params:,}")

    trainer = PPOTrainer(env, policy, device="cuda:0")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage4_hier_{timestamp}"

    trainer.train(
        max_iterations=args.max_iterations,
        log_dir=log_dir,
        stage3_checkpoint=args.stage3_checkpoint
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()