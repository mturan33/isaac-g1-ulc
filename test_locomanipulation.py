# Copyright (c) 2025, VLM-RL G1 Project
# Test Direct Physics Override - V7

"""
Test script for G1 Locomanipulation with DIRECT PHYSICS OVERRIDE
V7: Writes joint positions directly to simulation, bypassing DiffIK completely!

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_locomanipulation.py --mode locomanip
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test G1 Locomanipulation with Direct Physics Override")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="locomanip",
                    choices=["stand", "walk", "wave", "locomanip"],
                    help="Control mode: stand, walk, wave, or locomanip (walk+wave)")
parser.add_argument("--walk_speed", type=float, default=0.3, help="Forward walking speed (m/s)")
parser.add_argument("--wave_freq", type=float, default=0.7,
                    help="Wave frequency (Hz) - non-0.5 to avoid sampling artifact")
parser.add_argument("--wave_amp", type=float, default=0.5, help="Wave amplitude (radians) - ~30 degrees")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv

MODE_DESCRIPTIONS = {
    "stand": "Standing still (zero velocity)",
    "walk": f"Walking forward at {args_cli.walk_speed} m/s",
    "wave": "Waving right hand while standing (DIRECT PHYSICS OVERRIDE)",
    "locomanip": f"FULL LOCOMANIPULATION: Walking at {args_cli.walk_speed} m/s + Waving hand"
}

print("\n" + "=" * 70)
print("  G1 Locomanipulation Test - V7 (Direct Physics Override)")
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

        # Find joint indices for direct control
        all_joint_names = robot.data.joint_names
        print(f"\n[INFO] Total joints: {len(all_joint_names)}")

        # Find wave joints
        right_shoulder_roll_idx = None
        right_shoulder_pitch_idx = None
        right_elbow_idx = None

        for i, name in enumerate(all_joint_names):
            if "right_shoulder_roll" in name:
                right_shoulder_roll_idx = i
            elif "right_shoulder_pitch" in name:
                right_shoulder_pitch_idx = i
            elif "right_elbow" in name:
                right_elbow_idx = i

        print(f"\n[INFO] Wave joints found:")
        print(f"  - right_shoulder_roll: idx={right_shoulder_roll_idx}")
        print(f"  - right_shoulder_pitch: idx={right_shoulder_pitch_idx}")
        print(f"  - right_elbow: idx={right_elbow_idx}")

        # Get default joint positions
        default_joint_pos = robot.data.default_joint_pos.clone()

        # Get EE positions for reference
        left_ee_idx = 28
        right_ee_idx = 29

        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos
        init_right_offset = init_right_pos - init_base_pos

        do_walk = args_cli.mode in ["walk", "locomanip"]
        do_wave = args_cli.mode in ["wave", "locomanip"]

        wave_amp = args_cli.wave_amp  # radians
        wave_freq = args_cli.wave_freq  # Hz

        print(f"\n[INFO] Control settings:")
        print(f"  - Walking: {'ON' if do_walk else 'OFF'}" + (f" (vx={args_cli.walk_speed} m/s)" if do_walk else ""))
        print(f"  - Waving:  {'ON' if do_wave else 'OFF'}" + (
            f" (freq={wave_freq}Hz, amp={wave_amp}rad = {math.degrees(wave_amp):.1f}°)" if do_wave else ""))

        # Get joint limits for shoulder roll
        if right_shoulder_roll_idx is not None:
            joint_limits = robot.data.soft_joint_pos_limits
            roll_lower = joint_limits[0, right_shoulder_roll_idx, 0].item()
            roll_upper = joint_limits[0, right_shoulder_roll_idx, 1].item()
            print(
                f"\n[INFO] Shoulder roll joint limits: [{math.degrees(roll_lower):.1f}°, {math.degrees(roll_upper):.1f}°]")

        print(f"\n[INFO] Running simulation for 2000 steps (~40 seconds)...")
        print("  Press Ctrl+C to stop.\n")

        step_count = 0
        max_steps = 2000

        start_pos = robot.data.root_pos_w[:, :2].clone()

        while simulation_app.is_running() and step_count < max_steps:
            t = step_count * 0.02

            current_base_pos = robot.data.root_pos_w[:, :3]

            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== UPPER BODY CONTROL (DiffIK input - keep EE at current position) =====
            target_left_pos = current_base_pos + init_left_offset
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

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

            # Step the environment (DiffIK + locomotion policy)
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # ===== DIRECT PHYSICS OVERRIDE FOR WAVING =====
            # After env.step(), we DIRECTLY WRITE joint position to simulation
            if do_wave and right_shoulder_roll_idx is not None:
                wave_phase = 2 * math.pi * wave_freq * t
                wave_offset = wave_amp * math.sin(wave_phase)

                # Get current joint positions
                current_joint_pos = robot.data.joint_pos.clone()

                # Override shoulder roll with wave motion
                default_roll = default_joint_pos[0, right_shoulder_roll_idx].item()
                new_roll = default_roll + wave_offset

                # Clamp to joint limits
                new_roll = max(roll_lower, min(roll_upper, new_roll))

                # Write directly to simulation
                current_joint_pos[:, right_shoulder_roll_idx] = new_roll

                # Get current velocities
                current_joint_vel = robot.data.joint_vel.clone()

                # Write joint state directly to simulation
                robot.write_joint_state_to_sim(
                    position=current_joint_pos,
                    velocity=current_joint_vel
                )

            step_count += 1

            # Debug every 30 steps (not 50, to avoid sampling artifact with 0.7Hz)
            if step_count % 30 == 0 and step_count <= 300 and do_wave and right_shoulder_roll_idx is not None:
                wave_phase = 2 * math.pi * wave_freq * t
                wave_val = math.sin(wave_phase)
                actual_roll = robot.data.joint_pos[0, right_shoulder_roll_idx].item()
                target_roll = default_joint_pos[0, right_shoulder_roll_idx].item() + wave_amp * wave_val

                print(f"  [Step {step_count:3d}] t={t:.2f}s | Wave sin={wave_val:+.2f} | "
                      f"Target roll={math.degrees(target_roll):+.1f}° | "
                      f"Actual roll={math.degrees(actual_roll):+.1f}°")

            # Log every 100 steps
            if step_count % 100 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                root_vel = robot.data.root_lin_vel_w[:, 0].mean().item()
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

                status = []
                if do_walk:
                    status.append(f"Vel: {root_vel:.2f}m/s")
                    status.append(f"Dist: {distance:.2f}m")
                if do_wave and right_shoulder_roll_idx is not None:
                    wave_phase = 2 * math.pi * wave_freq * t
                    actual_roll = robot.data.joint_pos[0, right_shoulder_roll_idx].item()
                    status.append(f"Wave: sin={math.sin(wave_phase):+.2f}")
                    status.append(f"ShoulderRoll={math.degrees(actual_roll):+.1f}°")

                status_str = " | ".join(status) if status else "Holding"
                print(f"[Step {step_count:4d}] Height: {root_height:.3f}m | {status_str}")

            if terminated.any() or truncated.any():
                distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()
                print(f"\n[!] Episode ended at step {step_count}")
                print(f"    Total distance traveled: {distance:.2f}m")
                print("    Resetting...")
                obs_dict, _ = env.reset()
                start_pos = robot.data.root_pos_w[:, :2].clone()
                init_base_pos = robot.data.root_pos_w[:, :3].clone()

        distance = (robot.data.root_pos_w[:, :2] - start_pos).norm(dim=-1).mean().item()

        print("\n" + "=" * 70)
        print(f"  ✓ Test completed!")
        print(f"  Mode: {args_cli.mode.upper()}")
        if do_walk:
            print(f"  Distance traveled: {distance:.2f}m")
        print("=" * 70)
        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()