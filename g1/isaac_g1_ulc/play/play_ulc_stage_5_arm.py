"""
G1 Arm Reach - Play Script V3.3
================================

KULLANIM:
./isaaclab.bat -p .../play/play_ulc_stage_5_arm.py --checkpoint logs/ulc/g1_arm_reach_.../model_XXXX.pt --num_envs 4
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Reach Play")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--steps", type=int, default=2000, help="Number of steps to run")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Headless olmadan çalıştır (görselleştirme için)
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_arm_dual_orient_env import G1ArmReachEnv, G1ArmReachEnvCfg


def main():
    # Environment setup
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = G1ArmReachEnv(cfg=env_cfg)

    # Load checkpoint
    checkpoint_path = args.checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)

    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

    # Extract actor weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Build simple actor network
    obs_dim = env_cfg.num_observations
    act_dim = env_cfg.num_actions

    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, act_dim),
    ).to("cuda:0")

    # Load weights
    actor_state = {}
    for key, value in state_dict.items():
        if "actor" in key:
            new_key = key.replace("actor.", "")
            actor_state[new_key] = value

    if actor_state:
        actor.load_state_dict(actor_state)
        print(f"[INFO] Loaded {len(actor_state)} actor weights")
    else:
        print("[WARNING] No actor weights found in checkpoint!")

    actor.eval()

    # Get current threshold
    current_threshold = getattr(env, 'current_pos_threshold', env_cfg.stage_thresholds[0])

    print(f"\n{'='*60}")
    print(f"G1 ARM REACH PLAY - V3.3")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  From: {checkpoint_dir}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Steps: {args.steps}")
    print(f"  Current threshold: {current_threshold*100:.0f}cm")
    print(f"{'-'*60}")
    print(f"  Controls:")
    print(f"    - Watch the robot reach for green targets")
    print(f"    - Orange sphere = end effector")
    print(f"    - White sphere = shoulder center")
    print(f"{'='*60}\n")

    # Run
    obs, _ = env.reset()
    total_reward = 0.0
    total_reaches = 0
    min_distance = float('inf')
    distance_sum = 0.0
    distance_count = 0

    with torch.no_grad():
        for step in range(args.steps):
            # Get action from actor
            obs_tensor = obs["policy"]
            action = actor(obs_tensor)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward.mean().item()

            # Track distance
            if hasattr(env, 'prev_distance'):
                dist = env.prev_distance.mean().item()
                distance_sum += dist
                distance_count += 1
                if dist < min_distance:
                    min_distance = dist

            # Check for reaches
            if hasattr(env, 'total_reaches'):
                new_reaches = env.total_reaches - total_reaches
                if new_reaches > 0:
                    total_reaches = env.total_reaches
                    print(f"[Step {step:4d}] 🎯 REACH #{total_reaches}! Dist was: {dist:.3f}m")

            # Progress update
            if step > 0 and step % 200 == 0:
                avg_dist = distance_sum / max(distance_count, 1)
                ori_err = 0.0
                if hasattr(env, 'target_quat'):
                    from g1_arm_dual_orient_env import quat_diff_rad
                    ee_quat = env._compute_ee_quat()
                    ori_err = quat_diff_rad(ee_quat, env.target_quat).mean().item()
                    ori_err_deg = ori_err * 180 / 3.14159

                print(f"[Step {step:4d}] Pos: {dist:.3f}m (avg: {avg_dist:.3f}m, min: {min_distance:.3f}m) | "
                      f"Ori err: {ori_err_deg:.1f}° | Reaches: {total_reaches} | Reward: {total_reward:.1f}")

    # Summary
    avg_distance = distance_sum / max(distance_count, 1)

    print(f"\n{'='*60}")
    print(f"PLAY COMPLETE")
    print(f"{'='*60}")
    print(f"  Total reaches: {total_reaches}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Avg position distance: {avg_distance:.3f}m")
    print(f"  Min position distance: {min_distance:.3f}m")
    print(f"  Current threshold: {current_threshold*100:.0f}cm")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()