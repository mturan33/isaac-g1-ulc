# Copyright (c) 2025, VLM-RL G1 Project
# G1 Stable Base Manipulation Test
# Alt gövde sabit kalırken kollar hareket eder

"""
G1 Stable Base Manipulation Test

Isaac-PickPlace-Locomanipulation mantığı kullanılıyor AMA:
- Lower body velocity commands HER ZAMAN SIFIR
- Bu sayede locomotion policy aktif, robot dengede kalıyor
- Ancak yürümüyor, yerinde duruyor
- Kollar DiffIK ile serbestçe hareket edebiliyor

Teleop'taki çalışma mantığı ile aynı: sürekli 0 velocity gönderiliyor.

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p g1_stable_base_manipulation.py --mode reach
    .\isaaclab.bat -p g1_stable_base_manipulation.py --mode wave
    .\isaaclab.bat -p g1_stable_base_manipulation.py --mode circle
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Stable Base Manipulation Test")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="wave",
                    choices=["stand", "wave", "reach", "circle", "random"],
                    help="Arm control mode")
parser.add_argument("--arm", type=str, default="right",
                    choices=["left", "right", "both"],
                    help="Which arm to control")
parser.add_argument("--max_steps", type=int, default=2000,
                    help="Maximum simulation steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

MODE_DESCRIPTIONS = {
    "stand": "Kollar default pozisyonda, alt gövde sabit",
    "wave": "El sallama hareketi, alt gövde SABİT",
    "reach": "İleri uzanma hareketi, alt gövde SABİT",
    "circle": "Dairesel hareket, alt gövde SABİT",
    "random": "Rastgele hedefler, alt gövde SABİT"
}

print("\n" + "=" * 70)
print("  G1 STABLE BASE MANIPULATION TEST")
print(f"  Mode: {args_cli.mode.upper()}")
print(f"  Description: {MODE_DESCRIPTIONS[args_cli.mode]}")
print("  Lower Body: ALWAYS ZERO VELOCITY (stable base)")
print("=" * 70 + "\n")


def main():
    try:
        # Import environment config
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
            LocomanipulationG1DiffIKEnvCfg
        )

        print("[INFO] Creating LocomanipulationG1DiffIKEnvCfg environment...")

        env_cfg = LocomanipulationG1DiffIKEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[SUCCESS] ✓ Environment created!")

        # Get environment info
        obs_dict, _ = env.reset()

        action_dim = env.action_manager.total_action_dim
        num_envs = args_cli.num_envs
        device = env.device

        print(f"\n[INFO] Environment Details:")
        print(f"  - Action dimension: {action_dim}")
        print(f"  - Device: {device}")

        # Get robot reference
        robot = env.scene["robot"]
        all_joint_names = robot.data.joint_names
        all_body_names = robot.data.body_names

        print(f"  - Total joints: {len(all_joint_names)}")
        print(f"  - Total bodies: {len(all_body_names)}")

        # ============================================================
        # ACTION SPACE LAYOUT (32D Total)
        # ============================================================
        # [0:3]   = Left EE target position (world frame)
        # [3:7]   = Left EE target quaternion
        # [7:10]  = Right EE target position (world frame)
        # [10:14] = Right EE target quaternion
        # [14:28] = Hand joint positions (14 DoF)
        # [28]    = vx (forward velocity) - ALWAYS 0 FOR STABLE BASE
        # [29]    = vy (side velocity) - ALWAYS 0 FOR STABLE BASE
        # [30]    = yaw (turning) - ALWAYS 0 FOR STABLE BASE
        # [31]    = height offset - ALWAYS 0 FOR STABLE BASE
        # ============================================================

        print(f"\n[INFO] Action Space Layout (32D):")
        print(f"  [0:3]   Left EE position")
        print(f"  [3:7]   Left EE quaternion")
        print(f"  [7:10]  Right EE position")
        print(f"  [10:14] Right EE quaternion")
        print(f"  [14:28] Hand joints")
        print(f"  [28:32] Lower body (vx, vy, yaw, height) - ALWAYS ZERO!")

        # Find end-effector body indices
        # Based on chat summary: left_wrist_yaw_link=28, right_wrist_yaw_link=29
        left_ee_idx = None
        right_ee_idx = None

        for i, name in enumerate(all_body_names):
            if "left_wrist_yaw_link" in name:
                left_ee_idx = i
            elif "right_wrist_yaw_link" in name:
                right_ee_idx = i

        # Fallback to indices from chat if names not found exactly
        if left_ee_idx is None:
            left_ee_idx = 28
            print(f"[WARN] left_wrist_yaw_link not found, using index {left_ee_idx}")
        if right_ee_idx is None:
            right_ee_idx = 29
            print(f"[WARN] right_wrist_yaw_link not found, using index {right_ee_idx}")

        print(f"\n[INFO] End-effector body indices:")
        print(f"  - Left EE: {left_ee_idx}")
        print(f"  - Right EE: {right_ee_idx}")

        # Get initial state
        init_root_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        # Compute offsets from base (for tracking base movement)
        init_left_offset = init_left_pos - init_root_pos
        init_right_offset = init_right_pos - init_root_pos

        print(f"\n[INFO] Initial State:")
        print(f"  Root pos: {init_root_pos[0].tolist()}")
        print(f"  Left EE: {init_left_pos[0].tolist()}")
        print(f"  Right EE: {init_right_pos[0].tolist()}")

        # ============================================================
        # SIMULATION LOOP
        # ============================================================
        print(f"\n[INFO] Running {args_cli.max_steps} steps...")
        print(f"[INFO] Mode: {args_cli.mode}, Arm: {args_cli.arm}")
        print("  Press Ctrl+C to stop.\n")

        step_count = 0
        dt = 0.02  # 50 Hz

        # Tracking for stats
        max_drift = 0.0
        max_height_change = 0.0
        initial_height = init_root_pos[0, 2].item()

        while simulation_app.is_running() and step_count < args_cli.max_steps:
            t = step_count * dt

            # Get current base position (might drift slightly)
            current_base_pos = robot.data.root_pos_w[:, :3]

            # Initialize action tensor
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ============================================================
            # LOWER BODY: ALWAYS ZERO VELOCITY (CRITICAL!)
            # ============================================================
            # This is the key to stable base - like teleop, we send zero
            actions[:, 28] = 0.0  # vx = 0 (no forward movement)
            actions[:, 29] = 0.0  # vy = 0 (no side movement)
            actions[:, 30] = 0.0  # yaw = 0 (no turning)
            actions[:, 31] = 0.0  # height = 0 (default height)

            # ============================================================
            # UPPER BODY: Arm Control Based on Mode
            # ============================================================

            # Default: Keep EE at current offset from base (stable)
            target_left_pos = current_base_pos + init_left_offset
            target_left_quat = init_left_quat.clone()
            target_right_pos = current_base_pos + init_right_offset
            target_right_quat = init_right_quat.clone()

            # Apply arm movement based on mode
            if args_cli.mode == "stand":
                # Just hold position - do nothing extra
                pass

            elif args_cli.mode == "wave":
                # Wave motion - sinusoidal in Y direction
                wave_freq = 0.7  # Hz
                wave_amp = 0.15  # meters
                wave_offset = wave_amp * math.sin(2 * math.pi * wave_freq * t)

                if args_cli.arm in ["right", "both"]:
                    target_right_pos[:, 1] += wave_offset  # Y axis wave
                    target_right_pos[:, 2] += 0.1  # Raise arm

                if args_cli.arm in ["left", "both"]:
                    target_left_pos[:, 1] -= wave_offset  # Opposite phase
                    target_left_pos[:, 2] += 0.1

            elif args_cli.mode == "reach":
                # Reach forward - sinusoidal in X direction
                reach_freq = 0.3  # Hz
                reach_amp = 0.1  # meters
                reach_offset = reach_amp * (1 - math.cos(2 * math.pi * reach_freq * t)) / 2

                if args_cli.arm in ["right", "both"]:
                    target_right_pos[:, 0] += reach_offset  # X axis forward

                if args_cli.arm in ["left", "both"]:
                    target_left_pos[:, 0] += reach_offset

            elif args_cli.mode == "circle":
                # Circular motion in YZ plane
                circle_freq = 0.4  # Hz
                circle_radius = 0.1  # meters
                angle = 2 * math.pi * circle_freq * t

                if args_cli.arm in ["right", "both"]:
                    target_right_pos[:, 1] += circle_radius * math.sin(angle)
                    target_right_pos[:, 2] += circle_radius * math.cos(angle)

                if args_cli.arm in ["left", "both"]:
                    target_left_pos[:, 1] -= circle_radius * math.sin(angle)
                    target_left_pos[:, 2] += circle_radius * math.cos(angle)

            elif args_cli.mode == "random":
                # Random target every 100 steps
                if step_count % 100 == 0:
                    random_offset = (torch.rand(3, device=device) - 0.5) * 0.2

                    if args_cli.arm in ["right", "both"]:
                        target_right_pos += random_offset.unsqueeze(0)

                    if args_cli.arm in ["left", "both"]:
                        target_left_pos += random_offset.unsqueeze(0)

            # Set action values
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = target_left_quat
            actions[:, 7:10] = target_right_pos
            actions[:, 10:14] = target_right_quat

            # Hands - keep neutral (half closed)
            actions[:, 14:28] = 0.0

            # Step the environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            step_count += 1

            # Calculate drift and height change
            current_root = robot.data.root_pos_w[0]
            xy_drift = torch.norm(current_root[:2] - init_root_pos[0, :2]).item()
            height_change = abs(current_root[2].item() - initial_height)

            max_drift = max(max_drift, xy_drift)
            max_height_change = max(max_height_change, height_change)

            # Log every 100 steps
            if step_count % 100 == 0:
                current_left_pos = robot.data.body_pos_w[:, left_ee_idx][0]
                current_right_pos = robot.data.body_pos_w[:, right_ee_idx][0]

                # Stability status
                stability = "✅ STABLE" if xy_drift < 0.05 else f"⚠️ DRIFT {xy_drift:.3f}m"

                print(f"[Step {step_count:4d}] t={t:.1f}s | "
                      f"XY Drift: {xy_drift:.4f}m | "
                      f"Height: {current_root[2].item():.3f}m | "
                      f"{stability}")

                if args_cli.arm in ["right", "both"]:
                    print(f"             Right EE: ({current_right_pos[0]:.3f}, "
                          f"{current_right_pos[1]:.3f}, {current_right_pos[2]:.3f})")

            # Handle episode termination
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}")
                print(f"    Max drift: {max_drift:.4f}m")
                print("    Resetting...")
                obs_dict, _ = env.reset()

                # Reset tracking
                init_root_pos = robot.data.root_pos_w[:, :3].clone()
                init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
                init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
                init_left_offset = init_left_pos - init_root_pos
                init_right_offset = init_right_pos - init_root_pos
                initial_height = init_root_pos[0, 2].item()

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        print(f"\n  Mode: {args_cli.mode.upper()}")
        print(f"  Arm: {args_cli.arm}")
        print(f"  Total steps: {step_count}")
        print(f"\n  STABILITY METRICS:")
        print(f"    Max XY drift: {max_drift:.4f}m")
        print(f"    Max height change: {max_height_change:.4f}m")

        if max_drift < 0.05:
            print(f"\n  🎉 SUCCESS! Base remained stable!")
            print(f"     Zero velocity command worked as expected.")
        elif max_drift < 0.1:
            print(f"\n  ⚠️ WARNING: Slight drift detected")
            print(f"     May need to tune locomotion policy.")
        else:
            print(f"\n  ❌ FAILED: Significant drift occurred")
            print(f"     Check locomotion policy integration.")

        print("\n" + "=" * 70)

        env.close()

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("\n[INFO] Make sure LocomanipulationG1DiffIKEnvCfg is available.")
        print("[INFO] Check if locomanipulation_g1_diffik_env_cfg.py is properly set up.")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()