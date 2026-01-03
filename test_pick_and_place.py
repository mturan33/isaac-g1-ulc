# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V18
# Position Hold Controller for Lower Body

"""
G1 Pick-and-Place Demo V18

Key feature: ROOT POSITION HOLD
- Records initial root position and yaw
- Computes position/yaw error each step
- Sends velocity commands to locomotion to correct drift
- Acts like a simple path planner that keeps robot at target pose
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V18 - Position Hold")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_to_euler

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V18")
print("  DiffIK + Root Position Hold Controller")
print("=" * 70 + "\n")


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion [w, x, y, z]."""
    # quat shape: [num_envs, 4] with [w, x, y, z] format
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return yaw


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


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
        # POSITION HOLD: Record initial root pose
        # ============================================================

        init_root_pos = robot.data.root_pos_w.clone()  # [num_envs, 3]
        init_root_quat = robot.data.root_quat_w.clone()  # [num_envs, 4]
        init_root_yaw = get_yaw_from_quat(init_root_quat)  # [num_envs]

        print(f"\n[POSITION HOLD] Initial root pose:")
        print(f"  Position: [{init_root_pos[0, 0]:.3f}, {init_root_pos[0, 1]:.3f}, {init_root_pos[0, 2]:.3f}]")
        print(f"  Yaw: {math.degrees(init_root_yaw[0].item()):.1f} degrees")

        # Position hold gains (P controller)
        Kp_xy = 2.0  # Position gain (m/s per m error)
        Kp_yaw = 2.0  # Yaw gain (rad/s per rad error)
        max_vel_xy = 0.5  # Max velocity command (m/s)
        max_vel_yaw = 1.0  # Max yaw rate command (rad/s)

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

        # Success tracking
        SUCCESS_THRESHOLD = 0.02
        best_error = float('inf')
        best_step = 0
        success_reached = False
        success_step = None

        print("\n" + "=" * 60)
        print("  Control Loop with Position Hold")
        print("=" * 60 + "\n")

        for step in range(300):
            # ============================================================
            # POSITION HOLD: Compute correction velocities
            # ============================================================

            # Current root pose
            current_root_pos = robot.data.root_pos_w
            current_root_quat = robot.data.root_quat_w
            current_root_yaw = get_yaw_from_quat(current_root_quat)

            # Position error (in world frame)
            pos_error = init_root_pos - current_root_pos  # [num_envs, 3]

            # Yaw error
            yaw_error = normalize_angle(init_root_yaw - current_root_yaw)  # [num_envs]

            # Transform position error to robot's local frame
            # (rotate by negative current yaw)
            cos_yaw = torch.cos(-current_root_yaw)
            sin_yaw = torch.sin(-current_root_yaw)

            # Local frame: X = forward, Y = left
            # World frame: X = right, Y = forward
            # Robot forward (local X) = World Y
            # Robot left (local Y) = World -X
            local_error_x = pos_error[:, 1] * cos_yaw - pos_error[:, 0] * sin_yaw  # Forward error
            local_error_y = pos_error[:, 1] * sin_yaw + pos_error[:, 0] * cos_yaw  # Left error

            # P controller for velocities
            vel_x = torch.clamp(Kp_xy * local_error_x, -max_vel_xy, max_vel_xy)  # Forward vel
            vel_y = torch.clamp(Kp_xy * local_error_y, -max_vel_xy, max_vel_xy)  # Left vel
            vel_yaw = torch.clamp(Kp_yaw * yaw_error, -max_vel_yaw, max_vel_yaw)  # Yaw rate

            # ============================================================
            # Create actions
            # ============================================================

            # Get current left arm pose (maintain)
            current_left_pos_w = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat_w = robot.data.body_quat_w[:, left_ee_idx]

            actions = torch.zeros(num_envs, action_dim, device=device)

            # Left arm - maintain position
            actions[:, 0:3] = current_left_pos_w
            actions[:, 3:7] = current_left_quat_w

            # Right arm - move to target
            actions[:, 7:10] = target_right_pos_w
            actions[:, 10:14] = target_right_quat_w

            # Hands neutral
            actions[:, 14:28] = 0.0

            # ============================================================
            # Lower body - POSITION HOLD velocity commands
            # ============================================================
            # Assuming action format: [vx, vy, vyaw, ?]
            # These are velocity commands in robot's local frame

            actions[:, 28] = vel_x.unsqueeze(-1) if vel_x.dim() == 1 else vel_x  # Forward velocity
            actions[:, 29] = vel_y.unsqueeze(-1) if vel_y.dim() == 1 else vel_y  # Lateral velocity
            actions[:, 30] = vel_yaw.unsqueeze(-1) if vel_yaw.dim() == 1 else vel_yaw  # Yaw rate
            actions[:, 31] = 0.0  # Height or other

            # Step
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Check EE error
            current_right_pos_w = robot.data.body_pos_w[:, right_ee_idx]
            ee_error = torch.norm(current_right_pos_w - target_right_pos_w, dim=-1).item()
            movement = current_right_pos_w - init_right_pos_w
            root_height = robot.data.root_pos_w[:, 2].mean().item()

            # Root drift from initial
            root_drift_xy = torch.norm(pos_error[:, :2], dim=-1).item()
            yaw_drift_deg = math.degrees(abs(yaw_error[0].item()))

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
                      f"Root drift: {root_drift_xy:.3f}m, {yaw_drift_deg:.1f}deg | {status}{success_mark}")

                # Show velocity commands being sent
                if step % 60 == 0:
                    print(f"      Vel cmds: vx={vel_x[0]:.3f}, vy={vel_y[0]:.3f}, vyaw={vel_yaw[0]:.3f}")

            # Stop if success held
            if success_reached and step > success_step + 50:
                print(f"\n[INFO] Success maintained, stopping at step {step}")
                break

            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step}")
                break

        # Final summary
        final_root_pos = robot.data.root_pos_w
        final_root_quat = robot.data.root_quat_w
        final_root_yaw = get_yaw_from_quat(final_root_quat)

        final_pos_drift = torch.norm(init_root_pos[:, :2] - final_root_pos[:, :2], dim=-1).item()
        final_yaw_drift = math.degrees(abs(normalize_angle(init_root_yaw - final_root_yaw)[0].item()))

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

        print(f"\n  POSITION HOLD:")
        print(f"    Initial root: [{init_root_pos[0, 0]:.3f}, {init_root_pos[0, 1]:.3f}]")
        print(f"    Final root:   [{final_root_pos[0, 0]:.3f}, {final_root_pos[0, 1]:.3f}]")
        print(f"    Position drift: {final_pos_drift:.4f}m")
        print(f"    Yaw drift: {final_yaw_drift:.1f} degrees")

        if success_reached:
            print(f"\n  *** ARM SUCCESS! Target reached at step {success_step} ***")

        if final_pos_drift < 0.1 and final_yaw_drift < 15:
            print(f"  *** POSITION HOLD SUCCESS! Robot stayed in place ***")
        elif final_pos_drift < 0.2:
            print(f"  *** POSITION HOLD PARTIAL: Some drift but manageable ***")
        else:
            print(f"  *** POSITION HOLD NEEDS TUNING: Significant drift ***")

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()