# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V17
# FIXED: Stop when target reached, track success properly

"""
G1 Pick-and-Place Demo V17

Key improvements:
1. Detects SUCCESS when EE error < threshold
2. Stops early when target is reached
3. Reports actual success, not just final state
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V17")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V17")
print("  DiffIK V4 with Early Success Detection")
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

        print(f"\n[INFO] Initial Right EE (WORLD): {init_right_pos_w[0].tolist()}")

        # Define target: Move RIGHT arm FORWARD (Y+) by 0.15m
        target_right_pos_w = init_right_pos_w.clone()
        target_right_pos_w[:, 1] += 0.15  # Y+ = forward
        target_right_quat_w = init_right_quat_w.clone()

        print(f"[INFO] Target Right EE (WORLD):  {target_right_pos_w[0].tolist()}")
        print(f"[INFO] Movement: +0.15m in Y direction")

        # Success tracking
        SUCCESS_THRESHOLD = 0.02  # 2cm
        best_error = float('inf')
        best_step = 0
        success_reached = False
        success_step = None

        print("\n" + "=" * 50)
        print("  Control Loop (stops when target reached)")
        print("=" * 50 + "\n")

        for step in range(300):
            # Get current left arm pose (maintain)
            current_left_pos_w = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat_w = robot.data.body_quat_w[:, left_ee_idx]

            # Create actions
            actions = torch.zeros(num_envs, action_dim, device=device)

            # Left arm - maintain position
            actions[:, 0:3] = current_left_pos_w
            actions[:, 3:7] = current_left_quat_w

            # Right arm - move to target
            actions[:, 7:10] = target_right_pos_w
            actions[:, 10:14] = target_right_quat_w

            # Hands neutral
            actions[:, 14:28] = 0.0

            # Lower body - stand still (zero velocity command)
            actions[:, 28:32] = 0.0

            # Step
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Check error
            current_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
            ee_error = torch.norm(current_right_pos_w - target_right_pos_w, dim=-1).item()
            movement = current_right_pos_w - init_right_pos_w
            root_height = robot.data.root_pos_w[:, 2].mean().item()

            # Track best
            if ee_error < best_error:
                best_error = ee_error
                best_step = step

            # Check success
            if ee_error < SUCCESS_THRESHOLD and not success_reached:
                success_reached = True
                success_step = step
                print(f"\n*** SUCCESS at step {step}! Error: {ee_error:.4f}m ***\n")

            # Log every 20 steps
            if step % 20 == 0:
                status = "STABLE" if root_height > 0.5 else "FALLEN"
                success_mark = " <-- SUCCESS!" if ee_error < SUCCESS_THRESHOLD else ""
                print(
                    f"[{step:3d}] Error: {ee_error:.4f}m | Y move: {movement[0, 1]:.4f}m | Base Z: {root_height:.3f}m {status}{success_mark}")

            # Stop if success held for a while
            if success_reached and step > success_step + 30:
                print(f"\n[INFO] Success maintained, stopping at step {step}")
                break

            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step}")
                break

        # Final summary
        final_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
        final_movement = final_right_pos_w - init_right_pos_w
        final_error = torch.norm(final_right_pos_w - target_right_pos_w, dim=-1).item()

        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        print(f"  Best error achieved: {best_error:.4f}m at step {best_step}")
        print(f"  Final error: {final_error:.4f}m")
        print(f"  Final Y movement: {final_movement[0, 1]:.4f}m")

        if success_reached:
            print(f"\n  *** SUCCESS! Target reached at step {success_step} ***")
            print(f"  (Best error: {best_error * 100:.1f}cm < threshold {SUCCESS_THRESHOLD * 100:.0f}cm)")
        elif best_error < 0.05:
            print(f"\n  PARTIAL SUCCESS: Got within {best_error * 100:.1f}cm of target")
        else:
            print(f"\n  FAILED: Best error was {best_error * 100:.1f}cm")

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()