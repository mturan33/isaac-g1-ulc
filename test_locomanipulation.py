# Copyright (c) 2025, VLM-RL G1 Project
# Test Locomanipulation Environment

"""
Test Isaac-PickPlace-Locomanipulation-G1-Abs-v0

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\test_locomanipulation.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Locomanipulation Environment")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Import isaaclab_tasks to register environments
import isaaclab_tasks  # noqa: F401


def main():
    print("\n" + "=" * 70)
    print("  Testing Locomanipulation Environment")
    print("=" * 70 + "\n")

    # List available environments
    print("[INFO] Available Locomanipulation environments:")
    for env_name in gym.envs.registry.keys():
        if 'Locomanipulation' in env_name or 'PickPlace' in env_name:
            print(f"  - {env_name}")

    print("\n" + "-" * 70)

    # Try to create the environment
    task_name = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"

    try:
        print(f"[INFO] Creating environment: {task_name}")

        # Get environment config
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
            LocomanipulationG1EnvCfg
        )

        env_cfg = LocomanipulationG1EnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        print(f"[INFO] Environment config loaded!")
        print(f"[INFO] Scene config: {type(env_cfg.scene)}")

        # Print some config details
        if hasattr(env_cfg, 'actions'):
            print(f"[INFO] Actions config available")
        if hasattr(env_cfg, 'observations'):
            print(f"[INFO] Observations config available")

        print("\n[SUCCESS] Locomanipulation environment config is accessible!")
        print("[NOTE] Full environment creation requires additional assets.")

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("[INFO] Locomanipulation environment may not be fully installed.")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("  Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()