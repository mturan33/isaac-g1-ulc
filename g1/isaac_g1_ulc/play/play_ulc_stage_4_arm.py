"""
G1 Arm - AGGRESSIVE Manual Test
================================

Kolu ZORLA hareket ettir - joint pozisyonlarını direkt ayarla.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_arm_manual_test.py --test forward
./isaaclab.bat -p play_arm_manual_test.py --test wave
./isaaclab.bat -p play_arm_manual_test.py --test up
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import time

parser = argparse.ArgumentParser(description="G1 Arm Manual Test")
parser.add_argument("--test", type=str, default="forward",
                    choices=["forward", "wave", "up", "side", "all"],
                    help="Test modu: forward, wave, up, side, all")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

# Environment import
env_path = "source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/envs"
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg


def main():
    print("\n" + "=" * 70)
    print(f"   G1 ARM MANUAL TEST - Mode: {args.test.upper()}")
    print("=" * 70)

    # Environment
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0

    env = G1ArmReachEnv(cfg=env_cfg)

    # Joint limitleri
    # 0: right_shoulder_pitch  (-2.97, 2.79)  - ileri/geri
    # 1: right_shoulder_roll   (-2.25, 1.59)  - yana açma
    # 2: right_shoulder_yaw    (-2.62, 2.62)  - döndürme
    # 3: right_elbow_pitch     (-0.23, 3.42)  - dirsek bükme
    # 4: right_elbow_roll      (-2.09, 2.09)  - bilek döndürme

    # Test pozisyonları (RADYAN)
    poses = {
        "home":    [-0.3, 0.0, 0.0, 0.5, 0.0],
        "forward": [1.5, 0.0, 0.0, 0.0, 0.0],   # Kol düz ileri
        "up":      [-1.5, 0.0, 0.0, 0.0, 0.0],  # Kol yukarı
        "side":    [0.0, 1.5, 0.0, 0.0, 0.0],   # Kol yana
        "bent":    [0.5, 0.0, 0.0, 2.0, 0.0],   # Dirsek bükülü
    }

    print("\n[INFO] Test pozisyonları:")
    for name, pos in poses.items():
        print(f"  {name:10s}: {pos}")

    print(f"\n[INFO] Seçilen test: {args.test}")
    print("[INFO] Ctrl+C ile çık\n")

    # Reset
    obs_dict, _ = env.reset()

    step = 0
    phase = 0
    phase_duration = 100  # Her pozisyonda 100 step kal

    if args.test == "all":
        sequence = ["home", "forward", "home", "up", "home", "side", "home", "bent", "home"]
    elif args.test == "wave":
        sequence = None  # Sinüsoidal
    else:
        sequence = ["home", args.test, "home", args.test]

    try:
        while simulation_app.is_running():
            step += 1

            # Hedef pozisyonu belirle
            if args.test == "wave":
                # Sinüsoidal hareket
                t = step * 0.02
                target_joints = torch.tensor([
                    0.8 * math.sin(t),           # shoulder_pitch
                    0.5 * math.sin(t * 0.7),     # shoulder_roll
                    0.3 * math.sin(t * 1.3),     # shoulder_yaw
                    1.0 + 0.8 * math.sin(t * 0.5),  # elbow_pitch (always positive)
                    0.5 * math.sin(t * 0.9),     # elbow_roll
                ], device="cuda:0")
            else:
                # Sequence modu
                phase = (step // phase_duration) % len(sequence)
                pose_name = sequence[phase]
                target_joints = torch.tensor(poses[pose_name], device="cuda:0")

            # ===== DİREKT JOINT POZİSYONU SET ET =====
            # Action kullanmadan, direkt joint target'ı ayarla

            joint_pos_targets = env.robot.data.joint_pos.clone()
            joint_pos_targets[0, env.arm_joint_indices] = target_joints

            # Joint limitleri uygula
            joint_pos_targets[0, env.arm_joint_indices] = torch.clamp(
                joint_pos_targets[0, env.arm_joint_indices],
                env.joint_lower,
                env.joint_upper
            )

            env.robot.set_joint_position_target(joint_pos_targets)

            # Simülasyonu ilerlet (action olmadan)
            # Dummy action gönder
            dummy_action = torch.zeros((1, 5), device="cuda:0")
            obs_dict, rewards, terminated, truncated, info = env.step(dummy_action)

            # Log (her 30 step)
            if step % 30 == 0:
                current_joints = env.robot.data.joint_pos[0, env.arm_joint_indices]
                ee_pos = env._compute_ee_pos()[0]

                if args.test != "wave":
                    pose_name = sequence[phase]
                else:
                    pose_name = "wave"

                print(f"[Step {step:4d}] Phase: {pose_name:10s}")
                print(f"  Target:  [{target_joints[0]:.2f}, {target_joints[1]:.2f}, "
                      f"{target_joints[2]:.2f}, {target_joints[3]:.2f}, {target_joints[4]:.2f}]")
                print(f"  Current: [{current_joints[0]:.2f}, {current_joints[1]:.2f}, "
                      f"{current_joints[2]:.2f}, {current_joints[3]:.2f}, {current_joints[4]:.2f}]")
                print(f"  EE pos:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

                # Hedef vs current farkı
                diff = (target_joints - current_joints).abs().max().item()
                if diff > 0.1:
                    print(f"  ⚠️  Max diff: {diff:.2f} rad - JOINT'LER HEDEFİ TAKİP ETMİYOR!")
                else:
                    print(f"  ✓  Hedef takip ediliyor (diff: {diff:.3f})")
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print("TEST TAMAMLANDI")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()