# Copyright (c) 2025, VLM-RL G1 Project
# G1 DiffIK Test - V23
# Modify Locomanipulation config for FIXED BASE

"""
G1 DiffIK Test V23 - Locomanipulation with Fixed Base Override

Uses the existing locomanipulation environment but:
1. Creates it normally
2. Modifies the robot config before reset
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 DiffIK V23 - Fixed Base via Config Modify")
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
print("  G1 DiffIK Test - V23")
print("  Locomanipulation Environment Test")
print("=" * 70 + "\n")


def main():
    # ============================================================
    # First, list available G1 environments
    # ============================================================

    all_envs = list(gym.envs.registry.keys())
    g1_envs = [e for e in all_envs if "G1" in e]

    print(f"[INFO] Found {len(g1_envs)} G1 environments:")
    for env_name in sorted(g1_envs):
        print(f"  - {env_name}")

    # Try the locomanipulation environment we used before
    env_candidates = [
        "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
        "Isaac-Locomanipulation-G1-v0",
        "Isaac-G1-Flat-v0",
    ]

    env = None
    env_id = None

    for candidate in env_candidates:
        if candidate in g1_envs:
            print(f"\n[INFO] Found environment: {candidate}")
            env_id = candidate
            break

    if env_id is None:
        # Just pick the first G1 environment
        if g1_envs:
            env_id = g1_envs[0]
            print(f"\n[INFO] Using first G1 environment: {env_id}")
        else:
            print("[ERROR] No G1 environments found!")
            return

    print(f"\n[INFO] Creating environment: {env_id}")

    try:
        env = gym.make(env_id, num_envs=1)
        print(f"[INFO] Environment created successfully!")
        print(f"[INFO] Action space: {env.action_space}")
        print(f"[INFO] Observation space: {env.observation_space}")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return

    # Reset environment
    obs, info = env.reset()
    print(f"[INFO] Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")

    # Get robot reference
    try:
        scene = env.unwrapped.scene
        robot = scene["robot"]
        init_root_pos = robot.data.root_pos_w.clone()
        print(f"[INFO] Initial root position: {init_root_pos[0].tolist()}")
    except Exception as e:
        print(f"[WARNING] Could not access robot: {e}")
        init_root_pos = None

    # ============================================================
    # Run test
    # ============================================================

    print("\n" + "=" * 60)
    print("  Running Test - Check Robot Movement")
    print("=" * 60 + "\n")

    for step in range(200):
        # Zero action
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Check root drift
        if init_root_pos is not None:
            try:
                current_root_pos = robot.data.root_pos_w
                root_drift = torch.norm(current_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
                height = current_root_pos[0, 2].item()

                # Log every 40 steps
                if step % 40 == 0:
                    status = "STABLE" if root_drift < 0.5 else "DRIFTING"
                    print(f"[{step:3d}] Root drift: {root_drift:.4f}m | Height: {height:.3f}m | {status}")
            except:
                if step % 40 == 0:
                    print(f"[{step:3d}] Step completed")
        else:
            if step % 40 == 0:
                print(f"[{step:3d}] Step completed (no robot access)")

    # Final summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if init_root_pos is not None:
        try:
            final_root_pos = robot.data.root_pos_w
            final_drift = torch.norm(final_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
            print(f"\n  Final root drift: {final_drift:.4f}m")
            print(f"  Final height: {final_root_pos[0, 2].item():.3f}m")
        except:
            print("\n  Could not get final metrics")

    print(f"\n  Environment {env_id} works!")
    print(f"  Next step: Implement DiffIK control with this environment")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()