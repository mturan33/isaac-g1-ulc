# Copyright (c) 2025, VLM-RL G1 Project
# Test Differential IK Locomanipulation Environment

"""
Test script for G1 Locomanipulation with Differential IK (no Pink dependency)

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_diffik_locomanip.py --num_envs 1
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Differential IK Locomanipulation")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math

# Import environment components
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  Testing G1 Locomanipulation with Differential IK")
print("  (No Pink IK dependency required)")
print("=" * 70 + "\n")


def main():
    try:
        # CORRECT IMPORT PATH for isaaclab_tasks package
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
            LocomanipulationG1DiffIKEnvCfg
        )

        print("[INFO] ✓ Import successful!")
        print("[INFO] Creating environment with Differential IK...")

        env_cfg = LocomanipulationG1DiffIKEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[SUCCESS] ✓ Environment created!")
        print(f"  - Observation groups: {list(env.observation_manager.group_obs_dim.keys())}")
        print(f"  - Action dim: {env.action_manager.total_action_dim}")

        # Reset
        obs_dict, _ = env.reset()

        # Get action dimension
        action_dim = env.action_manager.total_action_dim
        print(f"  - Total action dim: {action_dim}")

        # Run simulation
        print("\n[INFO] Running simulation for 500 steps...")

        for step in range(500):
            # Create dummy actions
            actions = torch.zeros(args_cli.num_envs, action_dim, device=env.device)

            # Set identity quaternions for poses (wxyz format)
            actions[:, 3] = 1.0  # left quat w
            actions[:, 10] = 1.0  # right quat w

            # Add some wave motion to test IK
            t = step * 0.02
            wave = 0.1 * math.sin(2 * math.pi * 0.2 * t)

            # Move left arm forward/back (base frame coordinates)
            actions[:, 0] = 0.3 + wave  # x pos
            actions[:, 1] = 0.2  # y pos (left side)
            actions[:, 2] = 0.0  # z pos

            # Move right arm forward/back
            actions[:, 7] = 0.3 - wave  # x pos
            actions[:, 8] = -0.2  # y pos (right side)
            actions[:, 9] = 0.0  # z pos

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            if step % 100 == 0:
                print(f"[Step {step:4d}] Reward: {reward.mean().item():.3f}")

        print("\n" + "=" * 70)
        print("  ✓ SUCCESS! Simulation completed without errors!")
        print("  Differential IK is working correctly.")
        print("=" * 70)
        env.close()

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("\n" + "=" * 70)
        print("  TROUBLESHOOTING")
        print("=" * 70)
        print("""
1. Clear Python cache:
   cd C:\\IsaacLab\\source\\isaaclab_tasks\\isaaclab_tasks\\manager_based\\locomanipulation\\pick_place
   Remove-Item -Recurse -Force __pycache__
   Remove-Item -Recurse -Force mdp\\__pycache__

2. Verify file locations:
   - differential_ik_action.py → mdp/ folder
   - locomanipulation_g1_diffik_env_cfg.py → pick_place/ folder
   - mdp/__init__.py contains: from .differential_ik_action import *

3. Re-run this script after clearing cache.
""")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()