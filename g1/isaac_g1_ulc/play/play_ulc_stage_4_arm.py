"""
G1 Arm - DIRECT PHYSICS TEST
=============================

Environment'ın _apply_action metodunu BYPASS ederek
direkt joint kontrolü test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_arm_physics_test.py
"""

from __future__ import annotations

import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(description="G1 Arm Direct Physics Test")
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
    print("   G1 ARM DIRECT PHYSICS TEST")
    print("   Environment _apply_action BYPASS edilecek!")
    print("=" * 70)

    # Environment
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0

    env = G1ArmReachEnv(cfg=env_cfg)

    # Reset
    obs_dict, _ = env.reset()

    # Joint bilgileri
    print(f"\n[INFO] Arm joint indices: {env.arm_joint_indices.tolist()}")
    print(f"[INFO] Joint names: {[env.robot.data.joint_names[i] for i in env.arm_joint_indices.tolist()]}")
    print(f"[INFO] Joint lower limits: {env.joint_lower.tolist()}")
    print(f"[INFO] Joint upper limits: {env.joint_upper.tolist()}")

    # Test pozisyonları
    poses = {
        "home":    torch.tensor([-0.3, 0.0, 0.0, 0.5, 0.0], device="cuda:0"),
        "forward": torch.tensor([1.5, 0.0, 0.0, 0.0, 0.0], device="cuda:0"),
        "up":      torch.tensor([-1.5, 0.0, 0.0, 0.0, 0.0], device="cuda:0"),
        "side":    torch.tensor([0.0, 1.0, 0.0, 0.5, 0.0], device="cuda:0"),
    }

    print("\n[INFO] Test başlıyor - joint'ler sırayla hareket edecek")
    print("[INFO] Ctrl+C ile çık\n")

    step = 0
    current_target = poses["home"].clone()

    try:
        while simulation_app.is_running():
            step += 1

            # Her 150 step'te hedef değiştir
            phase = (step // 150) % 4
            if phase == 0:
                target_name = "home"
            elif phase == 1:
                target_name = "forward"
            elif phase == 2:
                target_name = "home"
            else:
                target_name = "side"

            target_joints = poses[target_name]

            # ===== YÖNTEM 1: Tüm joint'lere target set et =====
            all_joint_targets = env.robot.data.joint_pos.clone()

            # Sadece arm joint'lerini güncelle
            for i, idx in enumerate(env.arm_joint_indices):
                all_joint_targets[0, idx] = target_joints[i]

            # Joint target'ı set et
            env.robot.set_joint_position_target(all_joint_targets)

            # ===== YÖNTEM 2: Write joint state (daha agresif) =====
            # Bu yöntem joint'i ANINDA hedefe taşır (teleport gibi)
            # if step % 150 == 1:  # Sadece faz değişiminde
            #     joint_pos = env.robot.data.joint_pos.clone()
            #     joint_vel = torch.zeros_like(env.robot.data.joint_vel)
            #     for i, idx in enumerate(env.arm_joint_indices):
            #         joint_pos[0, idx] = target_joints[i]
            #     env.robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # ===== Simülasyonu ilerlet (env.step KULLANMADAN!) =====
            # env.step() yerine direkt scene.write_data_to_sim ve simulate
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(env.sim.get_physics_dt())

            # Log (her 30 step)
            if step % 30 == 0:
                current_joints = env.robot.data.joint_pos[0, env.arm_joint_indices]
                ee_pos = env._compute_ee_pos()[0]

                print(f"[Step {step:4d}] Phase: {target_name:10s}")
                print(f"  Target:  [{target_joints[0]:.2f}, {target_joints[1]:.2f}, "
                      f"{target_joints[2]:.2f}, {target_joints[3]:.2f}, {target_joints[4]:.2f}]")
                print(f"  Current: [{current_joints[0]:.2f}, {current_joints[1]:.2f}, "
                      f"{current_joints[2]:.2f}, {current_joints[3]:.2f}, {current_joints[4]:.2f}]")
                print(f"  EE pos:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

                diff = (target_joints - current_joints).abs().max().item()
                if diff > 0.1:
                    print(f"  ⚠️  Max diff: {diff:.2f} rad")
                else:
                    print(f"  ✓  OK (diff: {diff:.3f})")
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()