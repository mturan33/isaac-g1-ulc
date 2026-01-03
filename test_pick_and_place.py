# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V19
# FIXED LEGS - Bypass Locomotion Policy Completely

"""
G1 Pick-and-Place Demo V19

Key feature: FIXED LEGS
- Locomotion policy is given zero velocity command
- Leg joints are DIRECTLY WRITTEN to default positions each step
- Upper body uses DiffIK normally
- Result: Robot stands still while arms move freely
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V19 - Fixed Legs")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V19")
print("  FIXED LEGS - Locomotion Bypassed")
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

        # ============================================================
        # FIXED LEGS: Find leg joint indices
        # ============================================================

        all_joint_names = robot.data.joint_names
        print(f"\n[INFO] Total joints: {len(all_joint_names)}")

        # Find leg joint indices (hip, knee, ankle for both legs)
        leg_joint_keywords = [
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

        leg_joint_indices = []
        for i, name in enumerate(all_joint_names):
            for keyword in leg_joint_keywords:
                if keyword in name:
                    leg_joint_indices.append(i)
                    print(f"  Leg joint [{i}]: {name}")
                    break

        print(f"\n[INFO] Found {len(leg_joint_indices)} leg joints")

        # Get default leg joint positions
        default_joint_pos = robot.data.default_joint_pos.clone()
        default_leg_pos = default_joint_pos[:, leg_joint_indices].clone()

        print(f"[INFO] Default leg positions: {default_leg_pos[0].tolist()}")

        # Get initial EE positions
        init_left_pos_w = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat_w = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos_w = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat_w = robot.data.body_quat_w[:, right_ee_idx].clone()

        print(f"\n[INFO] Initial Right EE (WORLD): {init_right_pos_w[0].tolist()}")

        # Define target: Move RIGHT arm FORWARD (Y+) by 0.15m
        target_right_pos_w = init_right_pos_w.clone()
        target_right_pos_w[:, 1] += 0.15
        target_right_quat_w = init_right_quat_w.clone()

        print(f"[INFO] Target Right EE (WORLD):  {target_right_pos_w[0].tolist()}")

        # Record initial root position
        init_root_pos = robot.data.root_pos_w.clone()

        # Success tracking
        SUCCESS_THRESHOLD = 0.02
        best_error = float('inf')
        best_step = 0
        success_reached = False
        success_step = None

        print("\n" + "=" * 60)
        print("  Control Loop with FIXED LEGS")
        print("=" * 60 + "\n")

        for step in range(300):
            # Get current left arm pose (maintain)
            current_left_pos_w = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat_w = robot.data.body_quat_w[:, left_ee_idx]

            # ============================================================
            # Create actions for env.step()
            # ============================================================

            actions = torch.zeros(num_envs, action_dim, device=device)

            # Left arm - maintain current position
            actions[:, 0:3] = current_left_pos_w
            actions[:, 3:7] = current_left_quat_w

            # Right arm - move to target
            actions[:, 7:10] = target_right_pos_w
            actions[:, 10:14] = target_right_quat_w

            # Hands neutral
            actions[:, 14:28] = 0.0

            # Lower body - ZERO velocity (locomotion will be overridden anyway)
            actions[:, 28:32] = 0.0

            # Step the environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # ============================================================
            # FIXED LEGS: Override leg joints AFTER env.step()
            # ============================================================

            # Get current joint positions and velocities
            current_joint_pos = robot.data.joint_pos.clone()
            current_joint_vel = robot.data.joint_vel.clone()

            # Override leg joints with default positions
            current_joint_pos[:, leg_joint_indices] = default_leg_pos

            # Set leg velocities to zero
            current_joint_vel[:, leg_joint_indices] = 0.0

            # Write directly to simulation
            robot.write_joint_state_to_sim(
                position=current_joint_pos,
                velocity=current_joint_vel
            )

            # ============================================================
            # Check progress
            # ============================================================

            current_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
            ee_error = torch.norm(current_right_pos_w - target_right_pos_w, dim=-1).item()
            movement = current_right_pos_w - init_right_pos_w

            # Check root drift
            current_root_pos = robot.data.root_pos_w
            root_drift = torch.norm(current_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
            root_height = current_root_pos[:, 2].mean().item()

            # Track best
            if ee_error < best_error:
                best_error = ee_error
                best_step = step

            # Check success
            if ee_error < SUCCESS_THRESHOLD and not success_reached:
                success_reached = True
                success_step = step
                print(f"\n*** SUCCESS at step {step}! EE Error: {ee_error:.4f}m ***\n")

            # Log every 20 steps
            if step % 20 == 0:
                status = "STABLE" if root_height > 0.5 else "FALLEN"
                success_mark = " <-- SUCCESS!" if ee_error < SUCCESS_THRESHOLD else ""

                print(f"[{step:3d}] EE Err: {ee_error:.4f}m | Y move: {movement[0, 1]:.3f}m | "
                      f"Root drift: {root_drift:.4f}m | Height: {root_height:.3f}m | {status}{success_mark}")

            # Stop if success held for 50 steps
            if success_reached and step > success_step + 50:
                print(f"\n[INFO] Success maintained, stopping at step {step}")
                break

            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step}")
                break

        # Final summary
        final_root_pos = robot.data.root_pos_w
        final_root_drift = torch.norm(init_root_pos[:, :2] - final_root_pos[:, :2], dim=-1).item()

        final_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
        final_movement = final_right_pos_w - init_right_pos_w
        final_ee_error = torch.norm(final_right_pos_w - target_right_pos_w, dim=-1).item()

        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"\n  ARM CONTROL:")
        print(f"    Best EE error: {best_error:.4f}m at step {best_step}")
        print(f"    Final EE error: {final_ee_error:.4f}m")
        print(f"    Final Y movement: {final_movement[0, 1]:.4f}m (target: 0.15m)")

        print(f"\n  FIXED LEGS:")
        print(f"    Root drift: {final_root_drift:.4f}m")
        print(f"    Final height: {final_root_pos[0, 2]:.3f}m")

        if success_reached:
            print(f"\n  *** ARM SUCCESS! Target reached at step {success_step} ***")

        if final_root_drift < 0.05:
            print(f"  *** FIXED LEGS SUCCESS! Robot stayed in place ***")
        else:
            print(f"  *** WARNING: Root drifted {final_root_drift:.3f}m ***")

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()