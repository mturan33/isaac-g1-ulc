# Copyright (c) 2025, VLM-RL G1 Project
# Test Differential IK Locomanipulation - V5 Tiny Wave

"""
Test script for G1 Locomanipulation with Differential IK + Locomotion Policy
V5: TINY wave motion that DiffIK can actually track

Problem: DiffIK has max_joint_delta limit, can't make big movements per step
Solution: Use MUCH smaller amplitudes (5cm instead of 20cm)

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_locomanipulation.py --mode locomanip
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test G1 Locomanipulation with Different Modes")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="locomanip",
                    choices=["stand", "walk", "wave", "locomanip"],
                    help="Control mode: stand, walk, wave, or locomanip (walk+wave)")
parser.add_argument("--walk_speed", type=float, default=0.3, help="Forward walking speed (m/s)")
parser.add_argument("--wave_freq", type=float, default=0.3, help="Wave frequency (Hz) - SLOW!")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv

MODE_DESCRIPTIONS = {
    "stand": "Standing still (zero velocity)",
    "walk": f"Walking forward at {args_cli.walk_speed} m/s",
    "wave": "Waving right hand while standing",
    "locomanip": f"FULL LOCOMANIPULATION: Walking at {args_cli.walk_speed} m/s + Waving hand"
}

print("\n" + "=" * 70)
print("  G1 Locomanipulation Test - V5 (Tiny Wave for DiffIK)")
print(f"  Mode: {args_cli.mode.upper()}")
print(f"  Description: {MODE_DESCRIPTIONS[args_cli.mode]}")
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

        left_ee_idx = 28
        right_ee_idx = 29

        # Store INITIAL EE poses
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        # Calculate initial offset from robot base
        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos
        init_right_offset = init_right_pos - init_base_pos

        print(f"\n[INFO] Initial configuration:")
        print(f"  - Base pos: {init_base_pos[0].cpu().numpy()}")
        print(f"  - Right EE offset from base: {init_right_offset[0].cpu().numpy()}")

        # TINY wave parameters - DiffIK can track these!
        wave_amp = 0.05  # Only 5cm! DiffIK can handle this

        do_walk = args_cli.mode in ["walk", "locomanip"]
        do_wave = args_cli.mode in ["wave", "locomanip"]

        print(f"\n[INFO] Control settings:")
        print(f"  - Walking: {'ON' if do_walk else 'OFF'}" + (f" (vx={args_cli.walk_speed} m/s)" if do_walk else ""))
        print(f"  - Waving:  {'ON' if do_wave else 'OFF'}" + (
            f" (freq={args_cli.wave_freq}Hz, amp={wave_amp}m - TINY for DiffIK!)" if do_wave else ""))

        print(f"\n[INFO] Running simulation for 2000 steps (~40 seconds)...")
        print("  Press Ctrl+C to stop.\n")

        step_count = 0
        max_steps = 2000

        start_pos = robot.data.root_pos_w[:, :2].clone()

        # Track IK performance
        total_error = 0.0
        error_count = 0

        while simulation_app.is_running() and step_count < max_steps:
            t = step_count * 0.02

            current_base_pos = robot.data.root_pos_w[:, :3]

            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== UPPER BODY CONTROL =====
            # Left arm: maintain relative offset from body
            target_left_pos = current_base_pos + init_left_offset
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

            # Calculate wave offset
            wave_phase = 2 * math.pi * args_cli.wave_freq * t

            if do_wave:
                # TINY sinusoidal motion in Y direction (side to side)
                side_offset = wave_amp * math.sin(wave_phase)

                # Target = base + initial_offset + tiny wave
                target_right_pos = current_base_pos + init_right_offset.clone()
                target_right_pos[:, 1] += side_offset  # Y: side to side only

                actions[:, 7:10] = target_right_pos
                actions[:, 10:14] = init_right_quat
            else:
                target_right_pos = current_base_pos + init_right_offset
                actions[:, 7:10] = target_right_pos
                actions[:, 10:14] = init_right_quat

            # Hands - neutral
            actions[:, 14:28] = 0.0

            # ===== LOWER BODY CONTROL =====
            if do_walk:
                actions[:, 28] = args_cli.walk_speed
                actions[:, 29] = 0.0
                actions[:, 30] = 0.0
                actions[:, 31] = 0.0
            else:
                actions[:, 28:32] = 0.0

            obs_dict, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            # Measure IK tracking error
            if do_wave:
                actual_right_pos = robot.data.body_pos_w[:, right_ee_idx]
                actual_offset_y = (actual_right_pos[:, 1] - current_base_pos[:, 1]).item()
                target_offset_y = init_right_offset[0, 1].item() + side_offset
                error = abs(actual_offset_y - target_offset_y)
                total_error += error
                error_count += 1

            # Debug every 25 steps for first 200
            if step_count % 25 == 0 and step_count <= 200 and do_wave:
                actual_right_pos = robot.data.body_pos_w[:, right_ee_idx]
                actual_offset_y = (actual_right_pos[:, 1] - current_base_pos[:, 1]).item()
                target_offset_y = init_right_offset[0, 1].item() + side_offset
                error = abs(actual_offset_y - target_offset_y)

                print(f"  [Step {step_count:3d}] Wave sin={math.sin(wave_phase):+.2f} | "
                      f"Target Y offset={target_offset_y:.3f} | "
                      f"Actual Y offset={actual_offset_y:.3f} | "
                      f"Error={error:.4f}m")

            # Log every 100 steps
            if step_count % 100 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                root_vel = robot.data.root_lin_vel_w[:, 0].mean().item()
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

                status = []
                if do_walk:
                    status.append(f"Vel: {root_vel:.2f}m/s")
                    status.append(f"Dist: {distance:.2f}m")
                if do_wave:
                    avg_error = total_error / max(error_count, 1)
                    status.append(f"Wave: sin={math.sin(wave_phase):+.2f}")
                    status.append(f"AvgErr: {avg_error:.4f}m")

                status_str = " | ".join(status) if status else "Holding"
                print(f"[Step {step_count:4d}] Height: {root_height:.3f}m | {status_str}")

            if terminated.any() or truncated.any():
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()
                print(f"\n[!] Episode ended at step {step_count}")
                print(f"    Total distance traveled: {distance:.2f}m")
                if do_wave:
                    avg_error = total_error / max(error_count, 1)
                    print(f"    Average IK tracking error: {avg_error:.4f}m")
                print("    Resetting...")
                obs_dict, _ = env.reset()
                start_pos = robot.data.root_pos_w[:, :2].clone()
                init_base_pos = robot.data.root_pos_w[:, :3].clone()
                total_error = 0.0
                error_count = 0

        distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

        print("\n" + "=" * 70)
        print(f"  ✓ Test completed!")
        print(f"  Mode: {args_cli.mode.upper()}")
        if do_walk:
            print(f"  Distance traveled: {distance:.2f}m")
        if do_wave:
            avg_error = total_error / max(error_count, 1)
            print(f"  Average IK tracking error: {avg_error:.4f}m")
            if avg_error < 0.01:
                print(f"  ✓ IK tracking GOOD!")
            elif avg_error < 0.03:
                print(f"  ~ IK tracking OK")
            else:
                print(f"  ✗ IK tracking POOR - try smaller amplitude")
        print("=" * 70)
        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()