# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V16
# Uses FIXED differential_ik_action_v4.py

"""
G1 Pick-and-Place Demo V16

This test uses the FIXED DiffIK action term (V4) that properly converts
positions from WORLD frame to BASE frame before computing IK.

SETUP REQUIRED:
1. Copy differential_ik_action_v4.py to replace differential_ik_action.py
2. Then run this test
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V16 - Fixed DiffIK")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V16")
print("  Using FIXED DiffIK V4 (BASE frame)")
print("=" * 70 + "\n")


def main():
    try:
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
            LocomanipulationG1DiffIKEnvCfg
        )

        print("[INFO] Creating environment...")

        env_cfg = LocomanipulationG1DiffIKEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)

        print("[SUCCESS] Environment created!")

        obs_dict, _ = env.reset()

        action_dim = env.action_manager.total_action_dim
        num_envs = args_cli.num_envs
        device = env.device

        robot = env.scene["robot"]

        # Get body indices
        left_ee_idx = robot.body_names.index("left_wrist_yaw_link")
        right_ee_idx = robot.body_names.index("right_wrist_yaw_link")

        # Get initial EE positions (WORLD frame)
        init_left_pos_w = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat_w = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos_w = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat_w = robot.data.body_quat_w[:, right_ee_idx].clone()

        print(f"\n[INFO] Initial EE positions (WORLD frame):")
        print(f"  - Left EE:  {init_left_pos_w[0].tolist()}")
        print(f"  - Right EE: {init_right_pos_w[0].tolist()}")

        # Define target: Move RIGHT arm FORWARD (Y+ in world) by 0.15m
        target_right_pos_w = init_right_pos_w.clone()
        target_right_pos_w[:, 1] += 0.15  # Y+ = forward in world frame
        target_right_quat_w = init_right_quat_w.clone()

        print(f"\n[INFO] Target RIGHT EE (WORLD frame): {target_right_pos_w[0].tolist()}")
        print(f"[INFO] Movement: +0.15m in Y direction (forward)")

        print("\n" + "=" * 50)
        print("  Starting Control Loop")
        print("=" * 50 + "\n")

        dt = env_cfg.sim.dt * env_cfg.decimation

        for step in range(300):
            # Get current positions for left arm (keep still)
            current_left_pos_w = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat_w = robot.data.body_quat_w[:, left_ee_idx]

            # Create actions (in WORLD frame - V4 DiffIK will convert to BASE)
            actions = torch.zeros(num_envs, action_dim, device=device)

            # Left arm - maintain current position
            actions[:, 0:3] = current_left_pos_w
            actions[:, 3:7] = current_left_quat_w

            # Right arm - move to target
            actions[:, 7:10] = target_right_pos_w
            actions[:, 10:14] = target_right_quat_w

            # Hands - neutral
            actions[:, 14:28] = 0.0

            # Lower body - stand still
            actions[:, 28:32] = 0.0

            # Step environment - DiffIK action term will process
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Log every 20 steps
            if step % 20 == 0:
                current_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
                ee_error = torch.norm(current_right_pos_w - target_right_pos_w, dim=-1).item()
                root_height = robot.data.root_pos_w[:, 2].mean().item()

                # Calculate movement from initial
                movement = current_right_pos_w - init_right_pos_w

                status = "STABLE" if root_height > 0.5 else "FALLEN"

                print(f"[{step:3d}] EE Error: {ee_error:.4f}m | Base Z: {root_height:.3f}m {status}")
                print(
                    f"      Current: [{current_right_pos_w[0, 0]:.3f}, {current_right_pos_w[0, 1]:.3f}, {current_right_pos_w[0, 2]:.3f}]")
                print(
                    f"      Target:  [{target_right_pos_w[0, 0]:.3f}, {target_right_pos_w[0, 1]:.3f}, {target_right_pos_w[0, 2]:.3f}]")
                print(f"      Movement: X={movement[0, 0]:.4f}, Y={movement[0, 1]:.4f}, Z={movement[0, 2]:.4f}")

            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step}")
                break

        # Summary
        final_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
        final_movement = final_right_pos_w - init_right_pos_w
        final_error = torch.norm(final_right_pos_w - target_right_pos_w, dim=-1).item()

        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        print(f"  Initial EE:  {init_right_pos_w[0].tolist()}")
        print(f"  Final EE:    {final_right_pos_w[0].tolist()}")
        print(f"  Target:      {target_right_pos_w[0].tolist()}")
        print(
            f"  Movement:    X={final_movement[0, 0]:.4f}, Y={final_movement[0, 1]:.4f}, Z={final_movement[0, 2]:.4f}")
        print(f"  Final error: {final_error:.4f}m")

        y_movement = final_movement[0, 1].item()
        if y_movement > 0.10:
            print(f"\n  SUCCESS! Arm moved forward {y_movement:.3f}m")
        elif y_movement > 0.05:
            print(f"\n  PARTIAL: Arm moved {y_movement:.3f}m (target: 0.15m)")
        elif y_movement > 0.01:
            print(f"\n  MINIMAL: Arm moved only {y_movement:.3f}m")
        else:
            print(f"\n  FAILED: Arm did not move forward (Y={y_movement:.4f})")

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()