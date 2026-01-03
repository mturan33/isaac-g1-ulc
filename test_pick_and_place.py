# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V15
# Using OFFICIAL Isaac Lab DifferentialIKController with proper frame transforms

"""
G1 Pick-and-Place Demo V15

Key fixes based on Isaac Lab tutorial:
1. Uses official DifferentialIKController from isaaclab.controllers
2. Proper frame transforms: subtract_frame_transforms for world->base conversion
3. Correct Jacobian body index for floating base robot (NO -1!)
4. set_command() and compute() pattern from official tutorial

Reference: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_diff_ik.html

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v15.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V15 - Official DiffIK")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V15")
print("  Official Isaac Lab DifferentialIKController")
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

        print(f"[SUCCESS] ✓ Environment created!")

        obs_dict, _ = env.reset()

        action_dim = env.action_manager.total_action_dim
        num_envs = args_cli.num_envs
        device = env.device

        robot = env.scene["robot"]

        # ============================================================
        # Setup Official DifferentialIKController
        # ============================================================

        # Controller config - using damped least squares (dls) like tutorial
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",  # Full pose control (position + orientation)
            use_relative_mode=False,  # Absolute pose commands
            ik_method="dls",  # Damped least squares
            ik_params={"lambda_val": 0.05}  # Damping coefficient
        )

        # Create controller for right arm
        right_arm_controller = DifferentialIKController(
            diff_ik_cfg,
            num_envs=num_envs,
            device=device
        )

        print(f"[INFO] DifferentialIKController created:")
        print(f"  - command_type: {diff_ik_cfg.command_type}")
        print(f"  - ik_method: {diff_ik_cfg.ik_method}")
        print(f"  - action_dim: {right_arm_controller.action_dim}")

        # ============================================================
        # Find joint and body indices (like tutorial)
        # ============================================================

        # Right arm joints for IK control
        right_arm_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # Find joint indices
        all_joint_names = robot.data.joint_names
        right_arm_joint_ids = []
        for name in right_arm_joint_names:
            for i, jname in enumerate(all_joint_names):
                if name == jname:
                    right_arm_joint_ids.append(i)
                    break

        right_arm_joint_ids = torch.tensor(right_arm_joint_ids, device=device, dtype=torch.long)
        print(f"\n[INFO] Right arm joints: {right_arm_joint_names}")
        print(f"[INFO] Right arm joint IDs: {right_arm_joint_ids.tolist()}")

        # Find EE body index
        right_ee_body_name = "right_wrist_yaw_link"
        right_ee_body_idx = robot.body_names.index(right_ee_body_name)

        # CRITICAL: Jacobian index for floating base robot
        # Tutorial says: "For a fixed base robot, the frame index is one less than the body index"
        # G1 is FLOATING BASE, so we use body_idx directly!
        if robot.is_fixed_base:
            right_ee_jacobi_idx = right_ee_body_idx - 1
            print(f"[INFO] Robot is FIXED BASE - using jacobi_idx = body_idx - 1")
        else:
            right_ee_jacobi_idx = right_ee_body_idx
            print(f"[INFO] Robot is FLOATING BASE - using jacobi_idx = body_idx")

        print(
            f"[INFO] Right EE body: {right_ee_body_name} (body_idx={right_ee_body_idx}, jacobi_idx={right_ee_jacobi_idx})")

        # ============================================================
        # Get initial positions (in BASE frame!)
        # ============================================================

        # Get current EE pose in world frame
        ee_pose_w = robot.data.body_pose_w[:, right_ee_body_idx]  # [pos(3), quat(4)]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        # Get root pose in world frame
        root_pose_w = robot.data.root_pose_w  # [pos(3), quat(4)]
        root_pos_w = root_pose_w[:, 0:3]
        root_quat_w = root_pose_w[:, 3:7]

        # CRITICAL: Convert EE pose from world frame to BASE frame!
        # This is what the tutorial does with subtract_frame_transforms
        init_ee_pos_b, init_ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        print(f"\n[INFO] Initial positions:")
        print(f"  - Root (world): {root_pos_w[0].tolist()}")
        print(f"  - EE (world): {ee_pos_w[0].tolist()}")
        print(f"  - EE (BASE frame): {init_ee_pos_b[0].tolist()}")

        # ============================================================
        # Define target pose (in BASE frame!)
        # ============================================================

        # Target: move EE forward (Y+ in base frame) by 0.15m
        target_ee_pos_b = init_ee_pos_b.clone()
        target_ee_pos_b[:, 1] += 0.15  # Forward in base frame
        target_ee_quat_b = init_ee_quat_b.clone()

        # Create command tensor [pos(3), quat(4)]
        ik_command = torch.cat([target_ee_pos_b, target_ee_quat_b], dim=-1)

        print(f"\n[INFO] Target EE (BASE frame): {target_ee_pos_b[0].tolist()}")
        print(f"[INFO] Movement: +0.15m in Y direction (forward)")

        # Set the command
        right_arm_controller.reset()
        right_arm_controller.set_command(ik_command)

        # ============================================================
        # Simulation loop
        # ============================================================

        print("\n" + "=" * 50)
        print("  Starting IK Control Loop")
        print("=" * 50 + "\n")

        dt = env_cfg.sim.dt * env_cfg.decimation

        for step in range(300):
            # Get current states
            root_pose_w = robot.data.root_pose_w
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]

            ee_pose_w = robot.data.body_pose_w[:, right_ee_body_idx]
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            # CRITICAL: Convert to BASE frame!
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            # Get Jacobian - proper indexing for floating base
            jacobians = robot.root_physx_view.get_jacobians()
            # Shape: [num_envs, num_bodies-1, 6, num_dofs+6]
            # For floating base: [num_envs, num_bodies-1, 6, num_joints+6]
            # We need: [num_envs, 6, num_arm_joints]

            # Extract Jacobian for right arm
            # jacobians[:, jacobi_idx, :, joint_ids+6] for floating base
            right_jacobian = jacobians[:, right_ee_jacobi_idx, :, :]
            # Select only the columns for right arm joints (add 6 for floating base DOFs)
            joint_cols = right_arm_joint_ids + 6  # Add 6 for floating base
            right_jacobian = right_jacobian[:, :, joint_cols]

            # Get current joint positions for right arm
            joint_pos = robot.data.joint_pos[:, right_arm_joint_ids]

            # Compute IK
            joint_pos_des = right_arm_controller.compute(
                ee_pos_b, ee_quat_b, right_jacobian, joint_pos
            )

            # Apply joint position targets
            robot.set_joint_position_target(
                joint_pos_des,
                joint_ids=right_arm_joint_ids.tolist()
            )

            # Create actions for env.step (keeping other components)
            actions = torch.zeros(num_envs, action_dim, device=device)

            # Upper body IK action (simplified - just pass through)
            # Left arm - keep still
            left_ee_idx = robot.body_names.index("left_wrist_yaw_link")
            left_ee_pose_w = robot.data.body_pose_w[:, left_ee_idx]
            actions[:, 0:3] = left_ee_pose_w[:, 0:3]
            actions[:, 3:7] = left_ee_pose_w[:, 3:7]

            # Right arm - pass current target (will be overridden by our direct control)
            actions[:, 7:10] = ee_pos_w  # Just placeholder
            actions[:, 10:14] = ee_quat_w

            # Hands - neutral
            actions[:, 14:28] = 0.0

            # Lower body - stand still
            actions[:, 28:32] = 0.0

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Log every 20 steps
            if step % 20 == 0:
                # Error in base frame
                pos_error = torch.norm(ee_pos_b - target_ee_pos_b, dim=-1).item()
                root_height = root_pos_w[:, 2].mean().item()

                # Joint changes
                current_joints = robot.data.joint_pos[:, right_arm_joint_ids]
                default_joints = robot.data.default_joint_pos[:, right_arm_joint_ids]
                joint_changes = current_joints - default_joints

                status = "✓ STABLE" if root_height > 0.5 else "✗ FALLEN"

                print(f"[{step:3d}] EE Error: {pos_error:.4f}m | Base Z: {root_height:.3f}m {status}")
                print(f"      EE pos (base): [{ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}]")
                print(
                    f"      Target (base): [{target_ee_pos_b[0, 0]:.3f}, {target_ee_pos_b[0, 1]:.3f}, {target_ee_pos_b[0, 2]:.3f}]")

                # Print significant joint changes
                changes_str = []
                for i, name in enumerate(right_arm_joint_names):
                    change_deg = math.degrees(joint_changes[0, i].item())
                    if abs(change_deg) > 0.5:
                        short_name = name.replace("right_", "").replace("_joint", "")
                        changes_str.append(f"{short_name}: {change_deg:+.1f}°")
                if changes_str:
                    print(f"      Joints: {', '.join(changes_str)}")

            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step}")
                break

        # ============================================================
        # Summary
        # ============================================================

        final_ee_pos_b, final_ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3],
            robot.data.root_pose_w[:, 3:7],
            robot.data.body_pose_w[:, right_ee_body_idx, 0:3],
            robot.data.body_pose_w[:, right_ee_body_idx, 3:7]
        )

        movement = final_ee_pos_b - init_ee_pos_b
        final_error = torch.norm(final_ee_pos_b - target_ee_pos_b, dim=-1).item()

        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        print(f"  Initial EE (base): {init_ee_pos_b[0].tolist()}")
        print(f"  Final EE (base):   {final_ee_pos_b[0].tolist()}")
        print(f"  Target (base):     {target_ee_pos_b[0].tolist()}")
        print(f"  Actual movement:   {movement[0].tolist()}")
        print(f"  Expected movement: [0.0, 0.15, 0.0]")
        print(f"  Final error:       {final_error:.4f}m")

        y_movement = movement[0, 1].item()
        if y_movement > 0.10:
            print(f"\n  ✓ SUCCESS! Arm moved forward {y_movement:.3f}m")
        elif y_movement > 0.05:
            print(f"\n  ⚠ PARTIAL: Arm moved {y_movement:.3f}m (not full 0.15m)")
        elif y_movement > 0:
            print(f"\n  ⚠ MINIMAL: Arm moved only {y_movement:.3f}m")
        else:
            print(f"\n  ✗ FAILED: Arm moved backward or not at all")

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()