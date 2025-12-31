# Copyright (c) 2025, VLM-RL G1 Project
# Test Differential IK Locomanipulation - Fixed Version

"""
Test script for G1 Locomanipulation with Differential IK + Locomotion Policy
FIXED: Proper wave motion + Upper body stability

Modes:
  - stand:     Robot stands still
  - walk:      Robot walks forward
  - wave:      Robot waves hand while standing
  - locomanip: Robot walks while waving hand (FULL LOCOMANIPULATION!)

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
parser.add_argument("--wave_freq", type=float, default=2.0, help="Wave frequency (Hz)")
parser.add_argument("--wave_amp", type=float, default=0.20, help="Wave amplitude (m)")
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
print("  G1 Locomanipulation Test - FIXED VERSION")
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

        # EE body indices
        left_ee_idx = 28
        right_ee_idx = 29

        # Store INITIAL EE poses (before any movement)
        # These will be our reference "neutral" poses
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        # Calculate initial offset from robot base (body frame reference)
        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos
        init_right_offset = init_right_pos - init_base_pos

        print(f"\n[INFO] Initial configuration:")
        print(f"  - Base pos: {init_base_pos[0].cpu().numpy()}")
        print(f"  - Left EE offset: {init_left_offset[0].cpu().numpy()}")
        print(f"  - Right EE offset: {init_right_offset[0].cpu().numpy()}")

        # Wave motion parameters - raise hand up and wave
        wave_raise_height = 0.25  # Raise hand 25cm from initial position

        do_walk = args_cli.mode in ["walk", "locomanip"]
        do_wave = args_cli.mode in ["wave", "locomanip"]

        print(f"\n[INFO] Control settings:")
        print(f"  - Walking: {'ON' if do_walk else 'OFF'}" + (f" (vx={args_cli.walk_speed} m/s)" if do_walk else ""))
        print(f"  - Waving:  {'ON' if do_wave else 'OFF'}" + (
            f" (freq={args_cli.wave_freq}Hz, amp={args_cli.wave_amp}m, raise={wave_raise_height}m)" if do_wave else ""))

        print(f"\n[INFO] Running simulation for 2000 steps (~40 seconds)...")
        print("  Press Ctrl+C to stop.\n")

        step_count = 0
        max_steps = 2000

        start_pos = robot.data.root_pos_w[:, :2].clone()

        while simulation_app.is_running() and step_count < max_steps:
            t = step_count * 0.02  # env step = 0.02s

            # Get current robot base position
            current_base_pos = robot.data.root_pos_w[:, :3]
            current_base_quat = robot.data.root_quat_w

            # Create actions
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== UPPER BODY CONTROL =====
            # Keep arms at RELATIVE position to body (body frame)
            # This prevents the weird leaning when robot moves

            # Left arm: maintain relative offset from body
            target_left_pos = current_base_pos + init_left_offset
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat  # Keep initial orientation

            if do_wave:
                # Right arm: WAVE MOTION
                # 1. Raise the hand up
                # 2. Add sinusoidal side-to-side motion

                wave_phase = math.sin(2 * math.pi * args_cli.wave_freq * t)

                # Target position: base + offset + raise + wave
                target_right_pos = current_base_pos + init_right_offset
                target_right_pos[:, 2] += wave_raise_height  # Raise hand UP (Z axis)
                target_right_pos[:, 1] += args_cli.wave_amp * wave_phase  # Wave side-to-side (Y axis)

                actions[:, 7:10] = target_right_pos
                actions[:, 10:14] = init_right_quat  # Keep orientation
            else:
                # No wave: maintain relative offset
                target_right_pos = current_base_pos + init_right_offset
                actions[:, 7:10] = target_right_pos
                actions[:, 10:14] = init_right_quat

            # Hands - neutral
            actions[:, 14:28] = 0.0

            # ===== LOWER BODY CONTROL (Locomotion Policy) =====
            if do_walk:
                actions[:, 28] = args_cli.walk_speed  # vx (forward velocity)
                actions[:, 29] = 0.0  # vy (lateral velocity)
                actions[:, 30] = 0.0  # wz (angular velocity)
                actions[:, 31] = 0.0  # height offset
            else:
                actions[:, 28:32] = 0.0  # Stand still

            obs_dict, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            # Log every 200 steps
            if step_count % 200 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                root_vel = robot.data.root_lin_vel_w[:, 0].mean().item()
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

                # Get actual right hand Z position for debugging
                actual_right_z = robot.data.body_pos_w[:, right_ee_idx, 2].mean().item()
                target_right_z = (current_base_pos[:, 2] + init_right_offset[
                    :, 2] + wave_raise_height).mean().item() if do_wave else 0

                status = []
                if do_walk:
                    status.append(f"Vel: {root_vel:.2f}m/s")
                    status.append(f"Dist: {distance:.2f}m")
                if do_wave:
                    wave_phase = math.sin(2 * math.pi * args_cli.wave_freq * t)
                    wave_dir = "→" if wave_phase > 0 else "←"
                    status.append(f"Wave: {wave_dir} ({wave_phase:.2f})")
                    status.append(f"HandZ: {actual_right_z:.2f}m")

                status_str = " | ".join(status) if status else "Holding"
                print(f"[Step {step_count:4d}] Height: {root_height:.3f}m | {status_str}")

            if terminated.any() or truncated.any():
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()
                print(f"\n[!] Episode ended at step {step_count}")
                print(f"    Total distance traveled: {distance:.2f}m")
                print("    Resetting...")
                obs_dict, _ = env.reset()
                start_pos = robot.data.root_pos_w[:, :2].clone()
                # Re-capture initial poses after reset
                init_base_pos = robot.data.root_pos_w[:, :3].clone()

        distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

        print("\n" + "=" * 70)
        print(f"  ✓ Test completed!")
        print(f"  Mode: {args_cli.mode.upper()}")
        if do_walk:
            print(f"  Distance traveled: {distance:.2f}m")
        if do_wave:
            print(f"  Wave cycles: ~{int(max_steps * 0.02 * args_cli.wave_freq)}")
        print("=" * 70)
        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()