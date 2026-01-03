# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V14
# DiffIK DEBUG VERSION - See what DiffIK is actually doing

"""
G1 Pick-and-Place Demo V14 - DiffIK Debug

This version adds extensive debugging to understand why DiffIK isn't moving the arms.

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v14.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V14 - DiffIK Debug")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V14")
print("  DiffIK DEBUG VERSION")
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
        # Get DiffIK action term for debugging
        # ============================================================
        upper_body_action = env.action_manager._terms["upper_body_ik"]

        print(f"\n[DEBUG] DiffIK Action Term Info:")
        print(f"  - IK joint IDs: {upper_body_action._ik_joint_ids.tolist()}")
        print(f"  - Left arm DOF IDs: {upper_body_action._left_arm_dof_ids.tolist()}")
        print(f"  - Right arm DOF IDs: {upper_body_action._right_arm_dof_ids.tolist()}")
        print(f"  - Damping: {upper_body_action._damping}")
        print(f"  - Max joint delta: {upper_body_action._max_joint_delta}")

        # Get body indices
        left_ee_idx = robot.body_names.index("left_wrist_yaw_link")
        right_ee_idx = robot.body_names.index("right_wrist_yaw_link")

        # Get initial EE positions
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos
        init_right_offset = init_right_pos - init_base_pos

        print(f"\n[DEBUG] Initial positions:")
        print(f"  - Base: {init_base_pos[0].tolist()}")
        print(f"  - Left EE: {init_left_pos[0].tolist()}")
        print(f"  - Right EE: {init_right_pos[0].tolist()}")
        print(f"  - Right EE offset: {init_right_offset[0].tolist()}")

        # Get initial joint positions for right arm
        ik_joint_ids = upper_body_action._ik_joint_ids
        right_arm_local_ids = upper_body_action._right_arm_dof_ids

        init_ik_joints = robot.data.joint_pos[:, ik_joint_ids].clone()
        print(f"\n[DEBUG] Initial IK joint positions (radians):")
        for i, name in enumerate(upper_body_action._ik_joint_names):
            val = init_ik_joints[0, i].item()
            print(f"    {name}: {val:.4f} rad ({math.degrees(val):.2f}°)")

        # ============================================================
        # TEST 1: Send a target that's different from current
        # ============================================================
        print("\n" + "=" * 50)
        print("  TEST: Moving right EE forward by 0.15m")
        print("=" * 50)

        # Target: move right EE forward (Y+) by 0.15m
        target_right_offset = init_right_offset.clone()
        target_right_offset[:, 1] += 0.15  # Move forward

        print(f"\n[DEBUG] Target right EE offset: {target_right_offset[0].tolist()}")

        dt = env_cfg.sim.dt * env_cfg.decimation

        for step in range(200):
            current_base_pos = robot.data.root_pos_w[:, :3]

            # Create actions
            actions = torch.zeros(num_envs, action_dim, device=device)

            # Left arm - keep at initial
            target_left_pos = current_base_pos + init_left_offset
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

            # Right arm - move forward
            target_right_pos = current_base_pos + target_right_offset
            actions[:, 7:10] = target_right_pos
            actions[:, 10:14] = init_right_quat

            # Hands - neutral
            actions[:, 14:28] = 0.0

            # Lower body - stand still
            actions[:, 28:32] = 0.0

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Debug every 20 steps
            if step % 20 == 0:
                # Get current positions
                current_right_ee = robot.data.body_pos_w[:, right_ee_idx]
                ee_error = torch.norm(current_right_ee - target_right_pos, dim=-1).item()

                # Get current IK joints
                current_ik_joints = robot.data.joint_pos[:, ik_joint_ids]

                # Get joint targets from action term
                joint_targets = upper_body_action._ik_joint_targets[0].clone()

                # Calculate joint differences
                joint_diff = current_ik_joints[0] - init_ik_joints[0]
                target_diff = joint_targets - init_ik_joints[0]

                root_height = robot.data.root_pos_w[:, 2].mean().item()

                print(f"\n[Step {step:3d}] EE Error: {ee_error:.4f}m | Base Z: {root_height:.3f}m")
                print(f"  Current right EE: {current_right_ee[0].tolist()}")
                print(f"  Target right EE:  {target_right_pos[0].tolist()}")

                # Print joint changes for right arm
                print(f"  Right arm joint changes (actual):")
                for i, local_id in enumerate(right_arm_local_ids.tolist()):
                    name = upper_body_action._ik_joint_names[local_id]
                    diff = joint_diff[local_id].item()
                    if abs(diff) > 0.001:  # Only print if changed
                        print(f"    {name}: {math.degrees(diff):+.2f}°")

                print(f"  Right arm joint targets (from DiffIK):")
                for i, local_id in enumerate(right_arm_local_ids.tolist()):
                    name = upper_body_action._ik_joint_names[local_id]
                    tgt = target_diff[local_id].item()
                    if abs(tgt) > 0.001:  # Only print if changed
                        print(f"    {name}: {math.degrees(tgt):+.2f}° target")

            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step}")
                break

        # ============================================================
        # Summary
        # ============================================================
        final_right_ee = robot.data.body_pos_w[:, right_ee_idx]
        final_ee_offset = final_right_ee - robot.data.root_pos_w[:, :3]

        movement = final_ee_offset - init_right_offset

        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        print(f"  Initial right EE offset: {init_right_offset[0].tolist()}")
        print(f"  Final right EE offset:   {final_ee_offset[0].tolist()}")
        print(f"  Actual movement:         {movement[0].tolist()}")
        print(f"  Target movement:         [0.0, 0.15, 0.0]")

        total_movement = movement[0].norm().item()
        print(f"\n  Total movement magnitude: {total_movement:.4f}m")

        if total_movement < 0.01:
            print("\n  ⚠️  ARM DID NOT MOVE! Possible issues:")
            print("      1. Actuator stiffness too low")
            print("      2. DiffIK not computing deltas")
            print("      3. Joint limits blocking movement")
        elif total_movement < 0.05:
            print("\n  ⚠️  ARM MOVED SLIGHTLY but not enough")
            print("      DiffIK is working but needs more iterations or higher gains")
        else:
            print("\n  ✓  ARM MOVED! DiffIK is working")

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()