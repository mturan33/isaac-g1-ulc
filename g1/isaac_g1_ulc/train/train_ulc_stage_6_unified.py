"""
G1 Stage 6: Unified Loco-Manipulation Training
==============================================

Bu script Stage 3 (walking) ve Stage 5 (arm reaching) policy'lerini
birleştirerek tam loco-manipulation eğitimi yapar.

WEIGHT TRANSFER STRATEGY:
1. Stage 3 checkpoint → leg controller weights
2. Stage 5 checkpoint → arm controller weights
3. Yeni katmanlar: Koordinasyon için

KULLANIM (yeni eğitim - weight transfer ile):
./isaaclab.bat -p .../train/train_ulc_stage_6_unified.py \
    --num_envs 2048 \
    --max_iterations 15000 \
    --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt \
    --stage5_checkpoint logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt \
    --headless

KULLANIM (checkpoint'ten devam):
./isaaclab.bat -p .../train/train_ulc_stage_6_unified.py \
    --num_envs 2048 \
    --max_iterations 10000 \
    --resume logs/ulc/g1_unified_.../model_XXXX.pt \
    --headless

CURRICULUM STAGES:
1. Standing + basic arm reach
2. Slow walking + arm reach
3. Walking + arm reach
4. Walking with squat + arm reach
5. Deep squat + far reach (pick-up preparation)
6. Full loco-manipulation

TARGET: Yerden nesne alıp masaya bırakabilecek beceri
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
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="G1 Stage 6: Unified Loco-Manipulation Training")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=15000, help="Max training iterations")
parser.add_argument("--stage3_checkpoint", type=str, default=None, help="Path to Stage 3 (walking) checkpoint")
parser.add_argument("--stage5_checkpoint", type=str, default=None, help="Path to Stage 5 (arm) checkpoint")
parser.add_argument("--resume", type=str, default=None, help="Resume from Stage 6 checkpoint")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Add env path
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_unified_env import G1UnifiedEnv, G1UnifiedEnvCfg

from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class UnifiedActorCritic(nn.Module):
    """
    Unified Actor-Critic network for loco-manipulation.

    Architecture designed for weight transfer:
    - Leg encoder: Takes leg observations, maps to latent
    - Arm encoder: Takes arm observations, maps to latent
    - Fusion layer: Combines leg + arm latents
    - Actor head: Outputs combined actions
    - Critic head: Outputs value estimate

    This allows loading Stage 3 weights into leg encoder
    and Stage 5 weights into arm encoder.
    """

    def __init__(self, num_obs=80, num_act=17, hidden=[512, 256, 128]):
        super().__init__()

        # Observation split sizes
        # lin_vel: 3, ang_vel: 3, gravity: 3 = 9 (shared)
        # leg_pos: 12, leg_vel: 12 = 24 (leg specific)
        # arm_pos: 5, arm_vel: 5 = 10 (arm specific)
        # commands: 10 (shared)
        # gait: 2 (leg specific)
        # prev_leg: 12 (leg specific)
        # prev_arm: 5 (arm specific)
        # arm_target: 3, ee_pos: 3, pos_error: 3, dist: 1 = 10 (arm specific)
        # torso_euler: 3 (shared)

        self.shared_dim = 9 + 10 + 3  # 22
        self.leg_specific_dim = 24 + 2 + 12  # 38
        self.arm_specific_dim = 10 + 5 + 10  # 25

        # Leg encoder (for Stage 3 weight transfer)
        self.leg_encoder = nn.Sequential(
            nn.Linear(self.shared_dim + self.leg_specific_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )

        # Arm encoder (for Stage 5 weight transfer)
        self.arm_encoder = nn.Sequential(
            nn.Linear(self.shared_dim + self.arm_specific_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
        )

        # Fusion layer
        fusion_input_dim = 128 + 64  # leg_latent + arm_latent
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, num_act),
        )

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(num_act))

        # Alternative: Simple unified network (fallback)
        self.use_simple_network = True

        if self.use_simple_network:
            # Simple unified network (proven to work)
            layers = []
            prev = num_obs
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
                prev = h
            layers.append(nn.Linear(prev, num_act))
            self.actor = nn.Sequential(*layers)

            layers = []
            prev = num_obs
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.critic = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # Small init for policy head
        if self.use_simple_network:
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)

    def _split_obs(self, obs: torch.Tensor):
        """Split observation into shared, leg-specific, and arm-specific parts."""
        # Indices based on observation structure in g1_unified_env.py
        # lin_vel_b: 0-3, ang_vel_b: 3-6, proj_gravity: 6-9
        # leg_joint_pos: 9-21, leg_joint_vel: 21-33
        # arm_joint_pos: 33-38, arm_joint_vel: 38-43
        # height_cmd: 43-44, vel_cmd: 44-47, torso_cmd: 47-50
        # gait_phase: 50-52
        # prev_leg_actions: 52-64
        # prev_arm_actions: 64-69
        # target_pos: 69-72, ee_pos: 72-75, pos_error: 75-78, pos_dist: 78-79
        # torso_euler: 79-82

        shared = torch.cat([
            obs[:, 0:9],    # velocities + gravity
            obs[:, 43:50],  # height + vel + torso commands
            obs[:, 79:80],  # Note: we only have 80 dims, so this is truncated
        ], dim=-1)

        leg_specific = torch.cat([
            obs[:, 9:33],   # leg joint pos/vel
            obs[:, 50:52],  # gait phase
            obs[:, 52:64],  # prev leg actions
        ], dim=-1)

        arm_specific = torch.cat([
            obs[:, 33:43],  # arm joint pos/vel
            obs[:, 64:69],  # prev arm actions
            obs[:, 69:79],  # arm target info
        ], dim=-1)

        return shared, leg_specific, arm_specific

    def forward(self, obs: torch.Tensor):
        """Forward pass."""
        if self.use_simple_network:
            return self.actor(obs), self.critic(obs)

        shared, leg_specific, arm_specific = self._split_obs(obs)

        leg_input = torch.cat([shared, leg_specific], dim=-1)
        arm_input = torch.cat([shared, arm_specific], dim=-1)

        leg_latent = self.leg_encoder(leg_input)
        arm_latent = self.arm_encoder(arm_input)

        fusion = self.fusion(torch.cat([leg_latent, arm_latent], dim=-1))

        action_mean = self.actor_head(fusion)
        value = self.critic_head(fusion)

        return action_mean, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action."""
        mean, _ = self.forward(obs)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        mean, value = self.forward(obs)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return value.squeeze(-1), dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# ============================================================================
# WEIGHT TRANSFER
# ============================================================================

def transfer_weights_from_checkpoints(
    net: UnifiedActorCritic,
    stage3_path: str | None,
    stage5_path: str | None,
    device: str = "cuda:0"
):
    """
    Transfer weights from Stage 3 and Stage 5 checkpoints.

    Stage 3 (walking): 51 obs → 12 actions
    Stage 5 (arm):     29 obs → 5 actions
    Unified:           80 obs → 17 actions
    """

    transferred_count = 0

    if stage3_path and os.path.exists(stage3_path):
        print(f"\n[Weight Transfer] Loading Stage 3 checkpoint: {stage3_path}")
        stage3_ckpt = torch.load(stage3_path, map_location=device, weights_only=False)

        if "actor_critic" in stage3_ckpt:
            stage3_state = stage3_ckpt["actor_critic"]
        elif "model_state_dict" in stage3_ckpt:
            stage3_state = stage3_ckpt["model_state_dict"]
        else:
            stage3_state = stage3_ckpt

        # Transfer to leg encoder if using hierarchical network
        if not net.use_simple_network:
            # Try to match weights
            for key, value in stage3_state.items():
                if "actor.0" in key:
                    # First layer - need to adapt input size
                    print(f"  [Stage 3] Found first layer: {key} shape {value.shape}")

        # For simple network: transfer first layer weights (partial)
        if net.use_simple_network:
            net_state = net.state_dict()

            # Stage 3 had 51 obs, we have 80
            # We can transfer the first layer weights for the overlapping part
            for key in stage3_state:
                if "actor.0.weight" in key:
                    stage3_weight = stage3_state[key]  # [hidden, 51]
                    current_weight = net_state["actor.0.weight"]  # [hidden, 80]

                    # Copy weights for overlapping observations
                    # First 51 observations roughly align
                    min_in = min(stage3_weight.shape[1], current_weight.shape[1])
                    min_out = min(stage3_weight.shape[0], current_weight.shape[0])

                    net_state["actor.0.weight"][:min_out, :min_in] = stage3_weight[:min_out, :min_in]
                    print(f"  [Stage 3] Transferred actor.0.weight: [{min_out}, {min_in}] of {current_weight.shape}")
                    transferred_count += 1

                if "actor.0.bias" in key:
                    stage3_bias = stage3_state[key]
                    current_bias = net_state["actor.0.bias"]
                    min_len = min(len(stage3_bias), len(current_bias))
                    net_state["actor.0.bias"][:min_len] = stage3_bias[:min_len]
                    print(f"  [Stage 3] Transferred actor.0.bias: {min_len}")
                    transferred_count += 1

                # Transfer middle layers if they match
                for layer_idx in [2, 3, 4, 5, 6, 7]:
                    layer_key_w = f"actor.{layer_idx}.weight"
                    layer_key_b = f"actor.{layer_idx}.bias"
                    if layer_key_w in key:
                        if layer_key_w in net_state and stage3_state[key].shape == net_state[layer_key_w].shape:
                            net_state[layer_key_w] = stage3_state[key]
                            transferred_count += 1
                    if layer_key_b in key:
                        if layer_key_b in net_state and stage3_state[key].shape == net_state[layer_key_b].shape:
                            net_state[layer_key_b] = stage3_state[key]
                            transferred_count += 1

            net.load_state_dict(net_state, strict=False)

        if "best_reward" in stage3_ckpt:
            print(f"  [Stage 3] Best reward was: {stage3_ckpt['best_reward']:.2f}")

    if stage5_path and os.path.exists(stage5_path):
        print(f"\n[Weight Transfer] Loading Stage 5 checkpoint: {stage5_path}")
        stage5_ckpt = torch.load(stage5_path, map_location=device, weights_only=False)

        if "model_state_dict" in stage5_ckpt:
            stage5_state = stage5_ckpt["model_state_dict"]
        else:
            stage5_state = stage5_ckpt

        # Stage 5 arm policy had 29 obs → 5 actions
        # We can use this to initialize the arm-related parts

        for key, value in stage5_state.items():
            if "actor" in key:
                print(f"  [Stage 5] Found: {key} shape {value.shape}")

        # For arm actions (last 5 actions in our 17-action space)
        # The arm policy knowledge is less directly transferable
        # but we can use it to inform initialization

        print(f"  [Stage 5] Arm checkpoint loaded for reference")

    print(f"\n[Weight Transfer] Total transferred: {transferred_count} tensors")
    return transferred_count


# ============================================================================
# PPO TRAINER
# ============================================================================

class PPO:
    """PPO trainer with GAE."""

    def __init__(self, net: UnifiedActorCritic, device: str, lr: float = 3e-4):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, args.max_iterations, eta_min=1e-5
        )

    def gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, old_log_probs, returns, advantages, old_values):
        """PPO update with clipping."""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):  # PPO epochs
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                values, log_probs, entropy = self.net.evaluate(
                    obs[mb_idx], actions[mb_idx]
                )

                # Policy loss with clipping
                ratio = (log_probs - old_log_probs[mb_idx]).exp()
                surr1 = ratio * advantages[mb_idx]
                surr2 = ratio.clamp(0.8, 1.2) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                value_clipped = old_values[mb_idx] + (values - old_values[mb_idx]).clamp(-0.2, 0.2)
                critic_loss = 0.5 * torch.max(
                    (values - returns[mb_idx]) ** 2,
                    (value_clipped - returns[mb_idx]) ** 2
                ).mean()

                # Combined loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
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
# TRAINING WRAPPER
# ============================================================================

class CurriculumEnvWrapper:
    """Wrapper for curriculum logging."""

    def __init__(self, env: G1UnifiedEnv):
        self.env = env
        self._iteration = 0
        self._step_count = 0
        self._writer = None

    def set_writer(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def step(self, actions):
        self._step_count += 1
        result = self.env.step(actions)

        if self._step_count % 24 == 0:
            self._iteration += 1

            success_rate = self.env.update_curriculum(self._iteration)

            if self._writer is not None:
                self._writer.add_scalar('Curriculum/level', self.env.curriculum_level + 1, self._iteration)
                self._writer.add_scalar('Success/rate', success_rate, self._iteration)
                self._writer.add_scalar('Success/total_reaches', self.env.total_reaches, self._iteration)
                self._writer.add_scalar('Success/stage_reaches', self.env.stage_reaches, self._iteration)

                for key, value in self.env.extras.items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(f'Env/{key}', value, self._iteration)

            if self._iteration % 50 == 0:
                level = self.env.curriculum_level + 1
                total_r = self.env.total_reaches
                stage_r = self.env.stage_reaches
                stage_a = self.env.stage_attempts
                stage_sr = stage_r / max(stage_a, 1) * 100

                print(f"[Stage 6] Iter {self._iteration:5d} | "
                      f"Level {level}/6 | "
                      f"Stage SR: {stage_sr:.1f}% | "
                      f"Reaches: {total_r} | "
                      f"Height: {self.env.extras.get('M/height', 0):.3f}m | "
                      f"ArmDist: {self.env.extras.get('M/arm_dist', 0):.3f}m")

        return result

    def reset(self):
        return self.env.reset()

    def close(self):
        if self._writer:
            self._writer.close()
        self.env.close()


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train():
    device = "cuda:0"

    print("\n" + "=" * 70)
    print("G1 STAGE 6: UNIFIED LOCO-MANIPULATION TRAINING")
    print("=" * 70)
    print(f"  Environments: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Stage 3 checkpoint: {args.stage3_checkpoint}")
    print(f"  Stage 5 checkpoint: {args.stage5_checkpoint}")
    print(f"  Resume: {args.resume}")
    print("=" * 70 + "\n")

    # Create environment
    env_cfg = G1UnifiedEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = G1UnifiedEnv(cfg=env_cfg)
    env_wrapper = CurriculumEnvWrapper(env)

    num_obs = env_cfg.num_observations
    num_act = env_cfg.num_actions

    # Create network
    net = UnifiedActorCritic(num_obs=num_obs, num_act=num_act).to(device)

    # Weight transfer or resume
    start_iter = 0

    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["actor_critic"])
        start_iter = ckpt.get("iteration", 0)
        env.curriculum_level = ckpt.get("curriculum_level", 0)
        print(f"[INFO] Resumed from iteration {start_iter}, level {env.curriculum_level}")

    elif args.stage3_checkpoint or args.stage5_checkpoint:
        transfer_weights_from_checkpoints(
            net, args.stage3_checkpoint, args.stage5_checkpoint, device
        )

    # Create trainer
    ppo = PPO(net, device)

    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/g1_unified_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    env_wrapper.set_writer(log_dir)

    print(f"\n[INFO] Logging to: {log_dir}")

    # Training state
    best_reward = float('-inf')
    net.log_std.data.fill_(np.log(0.5))  # Start with moderate exploration

    # Initial reset
    obs, _ = env.reset()
    obs = obs["policy"]

    start_time = datetime.now()
    rollout_steps = 24

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
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
            with torch.no_grad():
                mean, value = net(obs)
                std = net.log_std.clamp(-2, 1).exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

            obs_buffer.append(obs)
            act_buffer.append(action)
            val_buffer.append(value.squeeze(-1))
            logp_buffer.append(log_prob)

            obs_dict, reward, terminated, truncated, _ = env_wrapper.step(action)
            obs = obs_dict["policy"]

            rew_buffer.append(reward)
            done_buffer.append((terminated | truncated).float())

        # Stack buffers
        obs_buffer = torch.stack(obs_buffer)
        act_buffer = torch.stack(act_buffer)
        rew_buffer = torch.stack(rew_buffer)
        done_buffer = torch.stack(done_buffer)
        val_buffer = torch.stack(val_buffer)
        logp_buffer = torch.stack(logp_buffer)

        # Compute returns and advantages
        with torch.no_grad():
            _, next_value = net(obs)
            next_value = next_value.squeeze(-1)

        advantages, returns = ppo.gae(rew_buffer, val_buffer, done_buffer, next_value)

        # PPO update
        update_info = ppo.update(
            obs_buffer.view(-1, num_obs),
            act_buffer.view(-1, num_act),
            logp_buffer.view(-1),
            returns.view(-1),
            advantages.view(-1),
            val_buffer.view(-1),
        )

        # Anneal exploration
        progress = iteration / args.max_iterations
        std = 0.5 + (0.15 - 0.5) * progress  # 0.5 → 0.15
        net.log_std.data.fill_(np.log(std))

        # Calculate stats
        mean_reward = rew_buffer.mean().item()

        iter_time = (datetime.now() - iter_start).total_seconds()
        fps = rollout_steps * args.num_envs / iter_time

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "actor_critic": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curriculum_level,
            }, f"{log_dir}/model_best.pt")
            print(f"[BEST] New best reward: {best_reward:.2f}")

        # TensorBoard logging
        if env_wrapper._writer:
            env_wrapper._writer.add_scalar("Train/reward", mean_reward, iteration)
            env_wrapper._writer.add_scalar("Train/std", std, iteration)
            env_wrapper._writer.add_scalar("Train/best_reward", best_reward, iteration)
            env_wrapper._writer.add_scalar("Loss/actor", update_info["actor_loss"], iteration)
            env_wrapper._writer.add_scalar("Loss/critic", update_info["critic_loss"], iteration)
            env_wrapper._writer.add_scalar("Loss/entropy", update_info["entropy"], iteration)
            env_wrapper._writer.add_scalar("Train/fps", fps, iteration)
            env_wrapper._writer.flush()

        # Console logging
        if iteration % 10 == 0:
            elapsed = datetime.now() - start_time
            eta = elapsed / (iteration - start_iter + 1) * (args.max_iterations - iteration)

            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Std={std:.3f} | "
                f"Lv={env.curriculum_level + 1} | "
                f"Reach={env.total_reaches} | "
                f"FPS={fps:.0f} | "
                f"{str(elapsed).split('.')[0]} / {str(eta).split('.')[0]}"
            )

        # Periodic checkpoints
        if (iteration + 1) % 1000 == 0:
            torch.save({
                "actor_critic": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curriculum_level,
            }, f"{log_dir}/model_{iteration + 1}.pt")

    # Final save
    torch.save({
        "actor_critic": net.state_dict(),
        "iteration": args.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": env.curriculum_level,
    }, f"{log_dir}/model_final.pt")

    env_wrapper.close()

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