# List all G1 and Locomanipulation environments
# Usage: .\isaaclab.bat -p list_envs.py

import gymnasium as gym

# Import isaaclab_tasks to register all environments
import isaaclab_tasks

# Get all registered environments
all_envs = list(gym.envs.registry.keys())

# Filter for G1 and Locomanipulation
print("\n" + "=" * 60)
print("  Available G1 / Locomanipulation Environments")
print("=" * 60 + "\n")

g1_envs = [e for e in all_envs if 'G1' in e or 'g1' in e]
locomanip_envs = [e for e in all_envs if 'Locomanip' in e.lower() or 'loco' in e.lower()]
humanoid_envs = [e for e in all_envs if 'Humanoid' in e]

print("G1 Environments:")
for env in sorted(set(g1_envs)):
    print(f"  - {env}")

print(f"\nLocomanipulation Environments:")
for env in sorted(set(locomanip_envs)):
    print(f"  - {env}")

print(f"\nHumanoid Environments:")
for env in sorted(set(humanoid_envs)):
    print(f"  - {env}")

print(f"\nTotal environments: {len(all_envs)}")
print("=" * 60)