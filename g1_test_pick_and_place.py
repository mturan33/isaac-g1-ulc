# Copyright (c) 2025, VLM-RL G1 Project
# G1 DiffIK Test - V22
# Uses OFFICIAL Isaac Lab FixedBase Environment!

"""
G1 DiffIK Test V22 - Using Official FixedBase Environment

Uses: Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
This is the official NVIDIA environment with:
- Robot base fixed in place
- Upper body IK control
- No locomotion needed
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 DiffIK V22 - FixedBase Environment")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Import Isaac Lab gym registration
import isaaclab_tasks  # noqa: F401

print("\n" + "=" * 70)
print("  G1 DiffIK Test - V22")
print("  Using Official Isaac Lab FixedBase Environment")
print("=" * 70 + "\n")


def main():
    # ============================================================
    # Create the OFFICIAL FixedBase Environment
    # ============================================================

    env_id = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
    print(f"[INFO] Creating environment: {env_id}")

    env = gym.make(env_id, num_envs=1)

    print(f"[INFO] Environment created successfully!")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")

    # Reset environment
    obs, info = env.reset()
    print(f"[INFO] Observation shape: {obs['policy'].shape}")

    # Get initial state
    scene = env.unwrapped.scene
    robot = scene["robot"]

    init_root_pos = robot.data.root_pos_w.clone()
    print(f"[INFO] Initial root position: {init_root_pos[0].tolist()}")

    # ============================================================
    # Run simple test - verify fixed base
    # ============================================================

    print("\n" + "=" * 60)
    print("  Running Fixed Base Verification Test")
    print("=" * 60 + "\n")

    for step in range(300):
        # Random action or zero action
        if step < 100:
            # Zero action first to see natural behavior
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        else:
            # Then random actions
            action = torch.randn(env.action_space.shape, device=env.unwrapped.device) * 0.1

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Check root drift
        current_root_pos = robot.data.root_pos_w
        root_drift = torch.norm(current_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
        height = current_root_pos[0, 2].item()

        # Log every 50 steps
        if step % 50 == 0:
            status = "✅ FIXED" if root_drift < 0.01 else f"❌ DRIFT {root_drift:.4f}m"
            action_mode = "ZERO" if step < 100 else "RANDOM"
            print(f"[{step:3d}] Root drift: {root_drift:.6f}m | Height: {height:.3f}m | "
                  f"Action: {action_mode} | {status}")

    # Final check
    final_root_pos = robot.data.root_pos_w
    final_drift = torch.norm(final_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()

    print("\n" + "=" * 60)
    print("  SUMMARY - FIXEDBASE ENVIRONMENT TEST")
    print("=" * 60)
    print(f"\n  Final root drift: {final_drift:.6f}m")

    if final_drift < 0.01:
        print(f"\n  🎉 SUCCESS! FixedBase environment works!")
        print(f"  Robot stays in place - ready for DiffIK testing!")
    else:
        print(f"\n  ❌ Unexpected drift - investigate further")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()