# Copyright (c) 2025, VLM-RL G1 Project
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for G1 Locomanipulation with Differential IK.

Robot ayakta durur - current EE pose'u korur, lower body sıfır velocity.
"""

from __future__ import annotations

import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test G1 Locomanipulation DiffIK - Standing")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

# Trigger gym registration with wildcard import
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import *


def main():
    print("\n" + "=" * 70)
    print("  Testing G1 Locomanipulation with Differential IK")
    print("  Robot will STAND STILL (holding current pose)")
    print("=" * 70)

    # Create environment
    env = gym.make(
        "Isaac-PickPlace-Locomanipulation-G1-DiffIK-v0",
        num_envs=args_cli.num_envs,
    )

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    print(f"\n[INFO] Environment created!")
    print(f"  - Device: {device}")
    print(f"  - Num envs: {num_envs}")
    print(f"  - Action dim: {env.action_space.shape[-1]}")

    # Reset
    obs_dict, info = env.reset()
    print(f"\n[INFO] Environment reset, observation keys: {list(obs_dict.keys())}")

    # Get robot articulation
    robot = env.unwrapped.scene["robot"]

    # EE body indices (from DiffIK init log)
    left_ee_idx = 28  # left_wrist_yaw_link
    right_ee_idx = 29  # right_wrist_yaw_link

    # Get initial EE poses
    init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
    init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
    init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
    init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

    print(f"\n[INFO] Initial EE poses:")
    print(f"  - Left EE pos:  {init_left_pos[0].cpu().numpy()}")
    print(f"  - Right EE pos: {init_right_pos[0].cpu().numpy()}")

    # Action buffer: [left_ee(7), right_ee(7), hands(14), loco(4)] = 32
    action = torch.zeros(num_envs, 32, device=device)

    print(f"\n[INFO] Running simulation for 1000 steps...")
    print("  Robot should stand still, holding arms in place.")
    print("  Press Ctrl+C to stop.\n")

    step_count = 0

    try:
        while simulation_app.is_running() and step_count < 1000:
            # Get current EE poses from robot
            current_left_pos = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat = robot.data.body_quat_w[:, left_ee_idx]
            current_right_pos = robot.data.body_pos_w[:, right_ee_idx]
            current_right_quat = robot.data.body_quat_w[:, right_ee_idx]

            # Build action - HOLD CURRENT POSE
            action[:, 0:3] = current_left_pos  # Left EE position
            action[:, 3:7] = current_left_quat  # Left EE quaternion
            action[:, 7:10] = current_right_pos  # Right EE position
            action[:, 10:14] = current_right_quat  # Right EE quaternion
            action[:, 14:28] = 0.0  # Hands - neutral
            action[:, 28:32] = 0.0  # Lower body - ZERO velocity

            # Step
            obs_dict, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Log every 100 steps
            if step_count % 100 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                print(f"[Step {step_count:4d}] Height: {root_height:.3f}m | Reward: {reward.mean().item():.4f}")

            # Reset if terminated
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}, resetting...")
                obs_dict, info = env.reset()

    except KeyboardInterrupt:
        print("\n[!] Stopped by user")

    print("\n" + "=" * 70)
    print("  ✓ Test completed!")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()