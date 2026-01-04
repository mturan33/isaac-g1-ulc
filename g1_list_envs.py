# Copyright (c) 2025, VLM-RL G1 Project
# G1 Environment Lister - Find available G1 environments

"""
List all available G1 environments in Isaac Lab
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List G1 Environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

# Import Isaac Lab gym registration
import isaaclab_tasks  # noqa: F401

print("\n" + "=" * 70)
print("  Available G1 Environments in Isaac Lab")
print("=" * 70 + "\n")

# Get all registered environments
all_envs = list(gym.envs.registry.keys())

# Filter for G1 environments
g1_envs = [e for e in all_envs if "G1" in e or "g1" in e.lower()]

print(f"Found {len(g1_envs)} G1-related environments:\n")

for env_name in sorted(g1_envs):
    print(f"  - {env_name}")

# Also search for FixedBase environments
print("\n" + "-" * 50)
print("FixedBase environments:\n")
fixed_envs = [e for e in all_envs if "FixedBase" in e or "fixed" in e.lower()]
for env_name in sorted(fixed_envs):
    print(f"  - {env_name}")

# Also search for PickPlace environments
print("\n" + "-" * 50)
print("PickPlace environments:\n")
pick_envs = [e for e in all_envs if "PickPlace" in e or "Pick" in e]
for env_name in sorted(pick_envs):
    print(f"  - {env_name}")

# Search for Locomanipulation
print("\n" + "-" * 50)
print("Locomanipulation environments:\n")
loco_envs = [e for e in all_envs if "Locomanip" in e or "locomanip" in e.lower()]
for env_name in sorted(loco_envs):
    print(f"  - {env_name}")

print("\n" + "=" * 70)

simulation_app.close()