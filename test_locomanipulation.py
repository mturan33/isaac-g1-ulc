# Copyright (c) 2025, VLM-RL G1 Project
# Test Differential IK Locomanipulation Environment

"""
Test script for G1 Locomanipulation with Differential IK (no Pink dependency)

Installation Steps:
1. Copy differential_ik_action.py to:
   C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomanipulation\pick_place\mdp\

2. Add to mdp/__init__.py:
   from .differential_ik_action import *

3. Run this script:
   cd C:\IsaacLab
   .\isaaclab.bat -p <path>\test_diffik_locomanip.py

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
        # Import our modified config
        from locomanipulation_g1_diffik_env_cfg import LocomanipulationG1DiffIKEnvCfg

        print("[INFO] Creating environment with Differential IK...")

        env_cfg = LocomanipulationG1DiffIKEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[SUCCESS] Environment created!")
        print(f"  - Observation dim: {env.observation_manager.group_obs_dim}")
        print(f"  - Action dim: {env.action_manager.total_action_dim}")

        # Reset
        obs_dict, _ = env.reset()

        # Get action dimension
        action_dim = env.action_manager.total_action_dim
        print(f"  - Total action dim: {action_dim}")

        # Run simulation
        print("\n[INFO] Running simulation...")

        for step in range(500):
            # Create dummy actions
            # Action format: [left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), hand_joints(14), lower_body(4)]
            actions = torch.zeros(args_cli.num_envs, action_dim, device=env.device)

            # Set identity quaternions for poses (wxyz format)
            actions[:, 3] = 1.0  # left quat w
            actions[:, 10] = 1.0  # right quat w

            # Add some wave motion to test IK
            t = step * 0.02
            wave = 0.1 * math.sin(2 * math.pi * 0.2 * t)

            # Move left arm forward/back
            actions[:, 0] = 0.3 + wave  # x pos
            actions[:, 1] = 0.2  # y pos
            actions[:, 2] = 0.0  # z pos

            # Move right arm forward/back
            actions[:, 7] = 0.3 - wave  # x pos
            actions[:, 8] = -0.2  # y pos
            actions[:, 9] = 0.0  # z pos

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            if step % 100 == 0:
                print(f"[Step {step:4d}] Reward: {reward.mean().item():.3f}")

        print("\n[SUCCESS] Simulation completed without errors!")
        env.close()

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("\n[INFO] Please install the Differential IK action module:")
        print("  1. Copy differential_ik_action.py to:")
        print(
            "     C:\\IsaacLab\\source\\isaaclab_tasks\\isaaclab_tasks\\manager_based\\locomanipulation\\pick_place\\mdp\\")
        print("  2. Add to mdp/__init__.py:")
        print("     from .differential_ik_action import *")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()