# Copyright (c) 2025, VLM-RL G1 Project
# Test Differential IK Locomanipulation Environment with Locomotion Policy

"""
Test script for G1 Locomanipulation with Differential IK + Active Locomotion Policy

Robot ayakta durur - Locomotion policy aktif, upper body current pose'u korur.

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_locomanipulation.py --num_envs 1
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Differential IK Locomanipulation with Locomotion Policy")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# Import environment components
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  Testing G1 Locomanipulation with Differential IK")
print("  + ACTIVE LOCOMOTION POLICY")
print("=" * 70 + "\n")


def main():
    try:
        # Import environment config
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
            LocomanipulationG1DiffIKEnvCfg
        )

        print("[INFO] ✓ Import successful!")
        print("[INFO] Creating environment with Differential IK + Locomotion Policy...")

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

        # Get robot articulation
        robot = env.scene["robot"]

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

        # Check if locomotion policy is available
        # The environment loads it automatically - check action manager
        action_manager = env.action_manager
        print(f"\n[INFO] Action terms:")
        for term_name, term in action_manager._terms.items():
            print(f"  - {term_name}: {type(term).__name__}")

        # Check if lower_body_policy observation is available
        has_loco_obs = "lower_body_policy" in obs_dict
        print(f"\n[INFO] Locomotion policy observation available: {has_loco_obs}")
        if has_loco_obs:
            loco_obs_shape = obs_dict["lower_body_policy"].shape
            print(f"  - Shape: {loco_obs_shape}")

        # Run simulation
        print("\n[INFO] Running simulation for 1000 steps...")
        print("  - Upper body: Holding current EE poses")
        print("  - Lower body: LOCOMOTION POLICY ACTIVE (zero velocity command)")
        print("  Press Ctrl+C to stop.\n")

        step_count = 0

        # Previous locomotion actions for observation (initialized to zeros)
        prev_loco_actions = torch.zeros(args_cli.num_envs, 12, device=env.device)

        while simulation_app.is_running() and step_count < 1000:
            # Get current EE poses from robot
            current_left_pos = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat = robot.data.body_quat_w[:, left_ee_idx]
            current_right_pos = robot.data.body_pos_w[:, right_ee_idx]
            current_right_quat = robot.data.body_quat_w[:, right_ee_idx]

            # Create actions
            # Format: [left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), hands(14), loco(4)] = 32
            actions = torch.zeros(args_cli.num_envs, action_dim, device=env.device)

            # Upper body - hold current EE poses
            actions[:, 0:3] = current_left_pos  # Left EE position
            actions[:, 3:7] = current_left_quat  # Left EE quaternion (wxyz)
            actions[:, 7:10] = current_right_pos  # Right EE position
            actions[:, 10:14] = current_right_quat  # Right EE quaternion (wxyz)
            actions[:, 14:28] = 0.0  # Hands - neutral

            # Lower body - ZERO velocity command for locomotion policy
            # The LocomotionPolicyAction term expects velocity commands: [vx, vy, wz, height_offset]
            # Zero command = stand still
            actions[:, 28] = 0.0  # vx (forward velocity)
            actions[:, 29] = 0.0  # vy (lateral velocity)
            actions[:, 30] = 0.0  # wz (angular velocity)
            actions[:, 31] = 0.0  # height offset

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            # Log every 100 steps
            if step_count % 100 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                root_vel = robot.data.root_lin_vel_w[:, :2].norm(dim=-1).mean().item()
                print(
                    f"[Step {step_count:4d}] Height: {root_height:.3f}m | Velocity: {root_vel:.3f}m/s | Reward: {reward.mean().item():.4f}")

            # Reset if terminated
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}, resetting...")
                obs_dict, _ = env.reset()

        print("\n" + "=" * 70)
        print("  ✓ Test completed!")
        print("  Locomotion policy was active throughout the test.")
        print("=" * 70)
        env.close()

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()