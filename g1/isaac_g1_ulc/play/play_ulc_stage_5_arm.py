"""
G1 Arm Reach with Orientation - Stage 5 Play Script
====================================================

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_5_arm.py
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Orient Play - Stage 5")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import glob

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_arm_dual_orient_env import G1ArmOrientEnv, G1ArmOrientEnvCfg


def find_latest_checkpoint():
    """Find the latest Stage 5 checkpoint."""
    log_dirs = glob.glob("logs/ulc/ulc_g1_stage5_arm_*")
    if not log_dirs:
        return None
    latest_dir = max(log_dirs, key=os.path.getmtime)

    # Find best or latest model
    models = glob.glob(os.path.join(latest_dir, "model_*.pt"))
    if not models:
        return None

    # Prefer model_best.pt if exists
    best_model = os.path.join(latest_dir, "model_best.pt")
    if os.path.exists(best_model):
        return best_model

    # Otherwise latest
    return max(models, key=lambda x: int(x.split("_")[-1].replace(".pt", "")))


def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: str):
    """Load policy from checkpoint."""
    import torch.nn as nn

    class SimplePolicy(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, act_dim),
            )

        def forward(self, x):
            return self.net(x)

    policy = SimplePolicy(obs_dim, act_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Extract actor weights
    actor_weights = {}
    for key, value in state_dict.items():
        if 'actor' in key:
            # Map RSL-RL actor weights to our simple policy
            new_key = key.replace('actor.', '').replace('actor_', '')
            if '0.' in new_key:
                new_key = new_key.replace('0.', 'net.0.')
            elif '2.' in new_key:
                new_key = new_key.replace('2.', 'net.2.')
            elif '4.' in new_key:
                new_key = new_key.replace('4.', 'net.4.')
            elif '6.' in new_key:
                new_key = new_key.replace('6.', 'net.6.')
            actor_weights[new_key] = value

    if actor_weights:
        try:
            policy.load_state_dict(actor_weights, strict=False)
            print(f"[INFO] Loaded {len(actor_weights)} actor weights")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}")

    return policy


def main():
    # Setup environment
    env_cfg = G1ArmOrientEnvCfg()
    env_cfg.scene.num_envs = 1

    env = G1ArmOrientEnv(cfg=env_cfg)

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("\n[ERROR] No checkpoint found!")
        print("Run training first or specify --checkpoint path")
        env.close()
        simulation_app.close()
        return

    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")

    # Load policy
    obs_dim = env_cfg.num_observations  # 28
    act_dim = env_cfg.num_actions  # 5
    device = "cuda:0"

    policy = load_policy(checkpoint_path, obs_dim, act_dim, device)
    policy.eval()

    print("\n" + "=" * 60)
    print("G1 ARM ORIENT PLAY - STAGE 5")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Steps: {args.num_steps}")
    print("=" * 60 + "\n")

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    reach_count = 0

    # Run loop
    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)

        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"]

        # Get current stats
        current_reaches = int(env.reach_count[0].item())
        if current_reaches > reach_count:
            reach_count = current_reaches
            print(f"[Step {step}] 🎯 REACH! Total: {reach_count}")

        # Print stats every 100 steps
        if step % 100 == 0:
            ee_pos = env._compute_ee_pos() - env.robot.data.root_pos_w
            target_pos = env.target_pos
            pos_dist = (ee_pos - target_pos).norm().item()

            ee_quat = env._compute_ee_quat()
            target_quat = env.target_quat
            # Simple dot product for orientation error
            ori_dot = torch.sum(ee_quat * target_quat, dim=-1).abs().item()
            ori_err_deg = 2 * torch.acos(torch.tensor(min(ori_dot, 1.0))).item() * 57.3

            print(f"[Step {step:4d}] Pos dist: {pos_dist:.3f}m | Ori err: {ori_err_deg:.1f}° | Reaches: {reach_count}")

    print("\n" + "=" * 60)
    print(f"PLAY COMPLETE - Total reaches: {reach_count}")
    print("=" * 60 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()