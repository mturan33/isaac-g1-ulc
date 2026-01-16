"""
G1 Arm - VISUAL DEBUG TEST
===========================

Kol hareket ederken target ve EE sphere'leri de güncelle.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_arm_visual_test.py
"""

from __future__ import annotations

import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(description="G1 Arm Visual Debug Test")
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
    print("   G1 ARM VISUAL DEBUG TEST")
    print("   Target ve EE sphere'ler HER STEP güncelleniyor!")
    print("=" * 70)

    # Environment
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0

    env = G1ArmReachEnv(cfg=env_cfg)

    # Reset
    obs_dict, _ = env.reset()

    # Test hedefleri (world koordinatlarında, robot root'a göre offset)
    # Robot root: [0, 0, 1.0]
    root_pos = env.robot.data.root_pos_w[0]  # [0, 0, 1]

    targets_rel = {
        "front":  torch.tensor([0.35, 0.20, 0.05], device="cuda:0"),   # İleri-sağ
        "right":  torch.tensor([0.10, 0.40, 0.10], device="cuda:0"),   # Sağ-yana
        "up":     torch.tensor([0.20, 0.20, 0.40], device="cuda:0"),   # Yukarı
        "down":   torch.tensor([0.20, 0.20, -0.20], device="cuda:0"),  # Aşağı
    }

    target_sequence = ["front", "right", "up", "down"]

    print(f"\n[INFO] Robot root: {root_pos.tolist()}")
    print("[INFO] Target pozisyonları (root-relative):")
    for name, pos in targets_rel.items():
        world_pos = root_pos + pos
        print(f"  {name:8s}: rel={pos.tolist()} → world={world_pos.tolist()}")

    print("\n[INFO] Test başlıyor...")
    print("[INFO] Yeşil küre = Target, Turuncu küre = End Effector")
    print("[INFO] Ctrl+C ile çık\n")

    step = 0
    phase_duration = 200  # Her hedefte 200 step
    current_target_name = "front"

    try:
        while simulation_app.is_running():
            step += 1

            # Faz belirleme
            phase_idx = (step // phase_duration) % len(target_sequence)
            current_target_name = target_sequence[phase_idx]
            target_rel = targets_rel[current_target_name]

            # Target'ı environment'a kaydet (root-relative)
            env.target_pos[0] = target_rel

            # ===== TARGET SPHERE GÜNCELLE =====
            target_world = root_pos + target_rel
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
            target_pose = torch.cat([target_world, default_quat]).unsqueeze(0)
            env.target_obj.write_root_pose_to_sim(target_pose)

            # ===== EE MARKER GÜNCELLE =====
            ee_pos_world = env._compute_ee_pos()
            palm_quat = env.robot.data.body_quat_w[:, env.palm_idx]
            ee_marker_pose = torch.cat([ee_pos_world[0], palm_quat[0]]).unsqueeze(0)
            env.ee_marker.write_root_pose_to_sim(ee_marker_pose)

            # ===== KOL KONTROLÜ =====
            # EE'yi target'a götürmek için basit P kontrolü
            ee_pos_rel = ee_pos_world[0] - root_pos
            error = target_rel - ee_pos_rel

            # Error'dan joint delta hesapla (çok basit yaklaşım)
            # Shoulder pitch: ileri/geri (x error)
            # Shoulder roll: yana (y error)
            # Elbow pitch: yukarı/aşağı (z error)

            current_joints = env.robot.data.joint_pos[0, env.arm_joint_indices]

            # Basit P controller
            Kp = 2.0
            joint_delta = torch.tensor([
                Kp * error[0],   # shoulder_pitch ← x error
                -Kp * error[1],  # shoulder_roll ← -y error (ters)
                0.0,              # shoulder_yaw
                -Kp * error[2],  # elbow_pitch ← -z error
                0.0,              # elbow_roll
            ], device="cuda:0")

            # Yeni hedef joint pozisyonları
            target_joints = current_joints + joint_delta * 0.02  # Küçük adımlar
            target_joints = torch.clamp(target_joints, env.joint_lower, env.joint_upper)

            # Joint target'ları set et
            all_joint_targets = env.robot.data.joint_pos.clone()
            all_joint_targets[0, env.arm_joint_indices] = target_joints
            env.robot.set_joint_position_target(all_joint_targets)

            # ===== SİMÜLASYON =====
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(env.sim.get_physics_dt())

            # Log
            if step % 50 == 0:
                distance = error.norm().item()
                reached = "✓ REACHED!" if distance < 0.05 else ""

                print(f"[Step {step:4d}] Target: {current_target_name:8s}")
                print(f"  Target (rel): [{target_rel[0]:.3f}, {target_rel[1]:.3f}, {target_rel[2]:.3f}]")
                print(f"  EE pos (rel): [{ee_pos_rel[0]:.3f}, {ee_pos_rel[1]:.3f}, {ee_pos_rel[2]:.3f}]")
                print(f"  Distance:     {distance:.3f}m {reached}")
                print(f"  Joints: [{current_joints[0]:.2f}, {current_joints[1]:.2f}, "
                      f"{current_joints[2]:.2f}, {current_joints[3]:.2f}, {current_joints[4]:.2f}]")
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()