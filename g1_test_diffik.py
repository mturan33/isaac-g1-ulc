# Copyright (c) 2025, VLM-RL G1 Project
# G1 DiffIK Test - V25
# Directly use Isaac-PickPlace-Locomanipulation-G1-Abs-v0

"""
G1 DiffIK Test V25 - Direct Locomanipulation Environment

We KNOW this environment exists from documentation and previous usage.
Just try it directly without listing.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 DiffIK V25 - Locomanipulation Direct")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Import Isaac Lab gym registration - ALL of them
import isaaclab_tasks  # noqa: F401

# Also try importing the specific module
try:
    import isaaclab_tasks.manager_based.manipulation  # noqa: F401

    print("[INFO] Loaded manipulation tasks")
except ImportError as e:
    print(f"[WARN] Could not load manipulation tasks: {e}")

print("\n" + "=" * 70)
print("  G1 DiffIK Test - V25")
print("  Direct Locomanipulation Environment")
print("=" * 70 + "\n")


def main():
    # The environment we know exists from documentation
    env_id = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"

    print(f"[INFO] Attempting to create: {env_id}")

    # First check if it's registered
    all_envs = list(gym.envs.registry.keys())

    # Search for similar names
    g1_envs = [e for e in all_envs if "G1" in e]
    pick_envs = [e for e in all_envs if "Pick" in e]
    loco_envs = [e for e in all_envs if "Loco" in e or "loco" in e]
    manip_envs = [e for e in all_envs if "Manip" in e or "manip" in e]

    print(f"\n[DEBUG] G1 environments ({len(g1_envs)}):")
    for e in sorted(g1_envs)[:10]:
        print(f"  - {e}")

    print(f"\n[DEBUG] Pick environments ({len(pick_envs)}):")
    for e in sorted(pick_envs)[:10]:
        print(f"  - {e}")

    print(f"\n[DEBUG] Loco environments ({len(loco_envs)}):")
    for e in sorted(loco_envs)[:10]:
        print(f"  - {e}")

    print(f"\n[DEBUG] Manip environments ({len(manip_envs)}):")
    for e in sorted(manip_envs)[:10]:
        print(f"  - {e}")

    # Check if our target is registered
    if env_id in all_envs:
        print(f"\n✅ Found {env_id} in registry!")
    else:
        print(f"\n❌ {env_id} NOT in registry")
        print(f"\n[INFO] Searching for similar...")

        # Search for anything with G1 and Pick or Loco
        similar = [e for e in all_envs if "G1" in e and ("Pick" in e or "Loco" in e or "Manip" in e)]
        if similar:
            print(f"[INFO] Similar environments found:")
            for e in similar:
                print(f"  - {e}")
            env_id = similar[0]
            print(f"\n[INFO] Trying: {env_id}")
        else:
            print("[ERROR] No similar G1 manipulation environments found")

            # Let's also check isaaclab specific registrations
            print("\n[DEBUG] All environments containing 'Isaac':")
            isaac_envs = [e for e in all_envs if "Isaac" in e]
            for e in sorted(isaac_envs)[:30]:
                print(f"  - {e}")
            return

    # Try to create the environment
    try:
        print(f"\n[INFO] Creating environment: {env_id}")
        env = gym.make(env_id, num_envs=1)
        print(f"[INFO] ✅ Environment created successfully!")
        print(f"[INFO] Action space: {env.action_space}")
        print(f"[INFO] Observation space: {env.observation_space}")

        # Reset and test
        obs, info = env.reset()
        print(f"[INFO] Reset successful")

        # Get robot reference
        scene = env.unwrapped.scene
        robot = scene["robot"]
        init_root_pos = robot.data.root_pos_w.clone()
        print(f"[INFO] Initial root position: {init_root_pos[0].tolist()}")

        # Quick test loop
        print("\n[INFO] Running quick test (100 steps)...")
        for step in range(100):
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 25 == 0:
                current_root = robot.data.root_pos_w
                drift = torch.norm(current_root[:, :2] - init_root_pos[:, :2], dim=-1).item()
                print(f"[{step:3d}] Root drift: {drift:.4f}m")

        print("\n✅ Environment works! Ready for DiffIK integration.")
        env.close()

    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()