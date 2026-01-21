"""
G1 Arm Reach - V3 Play Script (ARM POSITION PERSISTENCE)
=========================================================

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_5_arm.py

Belirli checkpoint ile:
./isaaclab.bat -p .../play/play_ulc_stage_5_arm.py --checkpoint logs/ulc/ulc_g1_arm_reach_v3_XXXX/model_2000.pt
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Reach Play - V3")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of steps to run")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import glob

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

# V3 environment kullan
from g1_arm_dual_orient_env import G1ArmReachEnv, G1ArmReachEnvCfg


def find_latest_checkpoint():
    """Find the latest Stage 5 checkpoint."""
    # V3 önce
    patterns = [
        "logs/ulc/ulc_g1_arm_reach_v3_*",
        "logs/ulc/ulc_g1_stage5_arm_*"
    ]

    for pattern in patterns:
        log_dirs = glob.glob(pattern)
        if log_dirs:
            latest_dir = max(log_dirs, key=os.path.getmtime)

            # Find best or latest model
            best_model = os.path.join(latest_dir, "model_best.pt")
            if os.path.exists(best_model):
                return best_model

            models = glob.glob(os.path.join(latest_dir, "model_*.pt"))
            if models:
                return max(models, key=lambda x: int(x.split("_")[-1].replace(".pt", "")))

    return None


class ActorCriticPolicy(nn.Module):
    """RSL-RL compatible actor-critic policy."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        # Actor network (same architecture as training)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, act_dim),
        )

        # Action std (learnable)
        self.std = nn.Parameter(torch.ones(act_dim) * 0.3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean)."""
        return self.actor(obs)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get action, optionally with noise."""
        mean = self.actor(obs)
        if deterministic:
            return mean
        else:
            std = self.std.expand_as(mean)
            return mean + std * torch.randn_like(mean)


def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: str) -> ActorCriticPolicy:
    """Load policy from RSL-RL checkpoint."""

    policy = ActorCriticPolicy(obs_dim, act_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # RSL-RL weight mapping
    # RSL-RL format: actor.0.weight, actor.0.bias, actor.2.weight, ...
    # Our format: actor.0.weight, actor.0.bias, actor.2.weight, ...

    new_state_dict = {}
    loaded_keys = []

    for key, value in state_dict.items():
        # Actor weights
        if key.startswith('actor.'):
            new_key = key  # Same format
            new_state_dict[new_key] = value
            loaded_keys.append(key)

        # Action std
        elif 'std' in key.lower():
            new_state_dict['std'] = value
            loaded_keys.append(key)

    if new_state_dict:
        try:
            policy.load_state_dict(new_state_dict, strict=False)
            print(f"[INFO] Loaded {len(loaded_keys)} weights from checkpoint")
            for key in loaded_keys[:5]:
                print(f"  ✓ {key}")
            if len(loaded_keys) > 5:
                print(f"  ... and {len(loaded_keys) - 5} more")
        except Exception as e:
            print(f"[WARN] Partial load: {e}")
            # Try loading what we can
            current_state = policy.state_dict()
            for key, value in new_state_dict.items():
                if key in current_state and current_state[key].shape == value.shape:
                    current_state[key] = value
            policy.load_state_dict(current_state)
    else:
        print("[WARN] No compatible weights found in checkpoint!")
        print("Available keys:", list(state_dict.keys())[:10])

    return policy


def main():
    # Setup environment
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1

    env = G1ArmReachEnv(cfg=env_cfg)

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("\n" + "=" * 60)
        print("[ERROR] No checkpoint found!")
        print("=" * 60)
        print("\nRun training first or specify --checkpoint path")
        print("\nExample:")
        print("  --checkpoint logs/ulc/ulc_g1_stage5_arm_2026-01-20_21-47-13/model_2000.pt")
        print("=" * 60)
        env.close()
        simulation_app.close()
        return

    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")

    # Load policy
    obs_dim = env_cfg.num_observations  # 28
    act_dim = env_cfg.num_actions       # 5
    device = "cuda:0"

    policy = load_policy(checkpoint_path, obs_dim, act_dim, device)
    policy.eval()

    print("\n" + "=" * 60)
    print("G1 ARM REACH PLAY - V3 (ARM POSITION PERSISTENCE)")
    print("=" * 60)
    print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  From: {os.path.dirname(checkpoint_path)}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Steps: {args.num_steps}")
    print("-" * 60)
    print("  Controls:")
    print("    - Watch the robot reach for green targets")
    print("    - Orange sphere = end effector")
    print("    - White sphere = shoulder center")
    print("=" * 60 + "\n")

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_reaches = 0
    episode_reaches = 0
    episode_rewards = 0.0

    # Stats tracking
    min_pos_dist = float('inf')
    pos_dist_history = []

    # Run loop
    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"]
        episode_rewards += rewards.item()

        # Get current stats
        ee_pos = env._compute_ee_pos() - env.robot.data.root_pos_w
        target_pos = env.target_pos
        pos_dist = (ee_pos - target_pos).norm().item()

        pos_dist_history.append(pos_dist)
        min_pos_dist = min(min_pos_dist, pos_dist)

        # Check reaches
        current_reaches = int(env.reach_count[0].item())
        if current_reaches > episode_reaches:
            episode_reaches = current_reaches
            total_reaches += 1
            print(f"[Step {step:4d}] 🎯 REACH #{total_reaches}! Dist was: {pos_dist:.3f}m")

        # Print stats every 200 steps
        if step % 200 == 0 and step > 0:
            ee_quat = env._compute_ee_quat()
            target_quat = env.target_quat
            ori_dot = torch.sum(ee_quat * target_quat, dim=-1).abs().item()
            ori_err_deg = 2 * torch.acos(torch.tensor(min(ori_dot, 1.0))).item() * 57.3

            avg_dist = sum(pos_dist_history[-200:]) / min(200, len(pos_dist_history))

            print(f"[Step {step:4d}] Pos: {pos_dist:.3f}m (avg: {avg_dist:.3f}m, min: {min_pos_dist:.3f}m) | "
                  f"Ori err: {ori_err_deg:.1f}° | Reaches: {total_reaches} | "
                  f"Reward: {episode_rewards:.1f}")

    # Final stats
    avg_pos_dist = sum(pos_dist_history) / len(pos_dist_history)

    print("\n" + "=" * 60)
    print("PLAY COMPLETE")
    print("=" * 60)
    print(f"  Total reaches: {total_reaches}")
    print(f"  Total reward: {episode_rewards:.1f}")
    print(f"  Avg position distance: {avg_pos_dist:.3f}m")
    print(f"  Min position distance: {min_pos_dist:.3f}m")
    print(f"  Pos threshold: {env_cfg.pos_threshold}m")
    print("-" * 60)
    if total_reaches > 0:
        print(f"  ✅ Robot reached {total_reaches} targets!")
    else:
        print(f"  ❌ Robot couldn't reach any targets")
        print(f"     Min distance ({min_pos_dist:.3f}m) vs threshold ({env_cfg.pos_threshold}m)")
    print("=" * 60 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()