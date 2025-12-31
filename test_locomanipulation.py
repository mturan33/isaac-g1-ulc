# Copyright (c) 2025, VLM-RL G1 Project
# Test Differential IK Locomanipulation Environment with Multiple Modes

"""
Test script for G1 Locomanipulation with Differential IK + Locomotion Policy

Modes:
  - stand:     Robot stands still
  - walk:      Robot walks forward
  - wave:      Robot waves hand while standing
  - locomanip: Robot walks while waving hand (FULL LOCOMANIPULATION!)

Usage:
    cd C:\IsaacLab

    # Stand still
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_locomanipulation.py --mode stand

    # Walk forward
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_locomanipulation.py --mode walk

    # Wave hand while standing
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_locomanipulation.py --mode wave

    # FULL LOCOMANIPULATION: Walk + Wave
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
parser.add_argument("--walk_speed", type=float, default=0.5, help="Forward walking speed (m/s)")
parser.add_argument("--wave_freq", type=float, default=1.5, help="Wave frequency (Hz)")
parser.add_argument("--wave_amp", type=float, default=0.15, help="Wave amplitude (m)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# Import environment components
from isaaclab.envs import ManagerBasedRLEnv

MODE_DESCRIPTIONS = {
    "stand": "Standing still (zero velocity)",
    "walk": f"Walking forward at {args_cli.walk_speed} m/s",
    "wave": "Waving right hand while standing",
    "locomanip": f"FULL LOCOMANIPULATION: Walking at {args_cli.walk_speed} m/s + Waving hand"
}

print("\n" + "=" * 70)
print("  G1 Locomanipulation Test")
print(f"  Mode: {args_cli.mode.upper()}")
print(f"  Description: {MODE_DESCRIPTIONS[args_cli.mode]}")
print("=" * 70 + "\n")


def main():
    try:
        # Import environment config
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
            LocomanipulationG1DiffIKEnvCfg
        )

        print("[INFO] Creating environment...")

        env_cfg = LocomanipulationG1DiffIKEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[SUCCESS] ✓ Environment created!")

        # Reset
        obs_dict, _ = env.reset()

        # Get dimensions
        action_dim = env.action_manager.total_action_dim
        num_envs = args_cli.num_envs
        device = env.device

        # Get robot articulation
        robot = env.scene["robot"]

        # EE body indices
        left_ee_idx = 28  # left_wrist_yaw_link
        right_ee_idx = 29  # right_wrist_yaw_link

        # Store initial EE poses for reference
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        # Base position for wave motion (relative offset from body)
        # Right hand default position (will add wave motion to Z)
        right_hand_base_offset = torch.tensor([0.15, -0.25, 0.3], device=device)  # Forward, right, up

        print(f"\n[INFO] Initial EE poses:")
        print(f"  - Left EE:  {init_left_pos[0].cpu().numpy()}")
        print(f"  - Right EE: {init_right_pos[0].cpu().numpy()}")

        # Mode flags
        do_walk = args_cli.mode in ["walk", "locomanip"]
        do_wave = args_cli.mode in ["wave", "locomanip"]

        print(f"\n[INFO] Control settings:")
        print(f"  - Walking: {'ON' if do_walk else 'OFF'}" + (f" (vx={args_cli.walk_speed} m/s)" if do_walk else ""))
        print(f"  - Waving:  {'ON' if do_wave else 'OFF'}" + (
            f" (freq={args_cli.wave_freq}Hz, amp={args_cli.wave_amp}m)" if do_wave else ""))

        # Run simulation
        print(f"\n[INFO] Running simulation for 2000 steps (~40 seconds)...")
        print("  Press Ctrl+C to stop.\n")

        step_count = 0
        max_steps = 2000

        # Track distance traveled
        start_pos = robot.data.root_pos_w[:, :2].clone()

        while simulation_app.is_running() and step_count < max_steps:
            # Time for wave motion
            t = step_count * 0.02  # env step = 0.02s

            # Get current robot base position for relative EE positioning
            base_pos = robot.data.root_pos_w[:, :3]
            base_quat = robot.data.root_quat_w

            # Get current EE poses
            current_left_pos = robot.data.body_pos_w[:, left_ee_idx]
            current_left_quat = robot.data.body_quat_w[:, left_ee_idx]
            current_right_pos = robot.data.body_pos_w[:, right_ee_idx]
            current_right_quat = robot.data.body_quat_w[:, right_ee_idx]

            # Create actions
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== UPPER BODY CONTROL =====
            if do_wave:
                # Left arm: Hold current position
                actions[:, 0:3] = current_left_pos
                actions[:, 3:7] = current_left_quat

                # Right arm: Wave motion (sinusoidal Z movement)
                wave_offset = args_cli.wave_amp * math.sin(2 * math.pi * args_cli.wave_freq * t)

                # Keep X, Y from current, add wave to Z
                actions[:, 7:10] = current_right_pos.clone()
                actions[:, 9] += wave_offset  # Add wave to Z position

                # Keep current orientation
                actions[:, 10:14] = current_right_quat
            else:
                # Hold both arms at current positions
                actions[:, 0:3] = current_left_pos
                actions[:, 3:7] = current_left_quat
                actions[:, 7:10] = current_right_pos
                actions[:, 10:14] = current_right_quat

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

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            # Log every 200 steps
            if step_count % 200 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                root_vel = robot.data.root_lin_vel_w[:, 0].mean().item()  # Forward velocity
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

                status = []
                if do_walk:
                    status.append(f"Vel: {root_vel:.2f}m/s")
                    status.append(f"Dist: {distance:.2f}m")
                if do_wave:
                    wave_phase = "↑" if math.sin(2 * math.pi * args_cli.wave_freq * t) > 0 else "↓"
                    status.append(f"Wave: {wave_phase}")

                status_str = " | ".join(status) if status else "Holding"
                print(f"[Step {step_count:4d}] Height: {root_height:.3f}m | {status_str}")

            # Reset if terminated
            if terminated.any() or truncated.any():
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()
                print(f"\n[!] Episode ended at step {step_count}")
                print(f"    Total distance traveled: {distance:.2f}m")
                print("    Resetting...")
                obs_dict, _ = env.reset()
                start_pos = robot.data.root_pos_w[:, :2].clone()

        # Final stats
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