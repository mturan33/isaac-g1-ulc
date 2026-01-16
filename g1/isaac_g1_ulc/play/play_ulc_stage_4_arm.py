"""
G1 Arm - IMPROVED VISUAL TEST
==============================

Daha hızlı controller + hedefe ulaşınca yeni hedef.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_arm_visual_test_v2.py
"""

from __future__ import annotations

import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(description="G1 Arm Visual Test v2")
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
    print("   G1 ARM VISUAL TEST v2 - FASTER CONTROLLER")
    print("   Hedefe ulaşınca (< 8cm) yeni hedef spawn!")
    print("=" * 70)

    # Environment - DAHA GÜÇLÜ ACTUATOR
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 300.0

    env = G1ArmReachEnv(cfg=env_cfg)

    # Reset
    obs_dict, _ = env.reset()

    root_pos = env.robot.data.root_pos_w[0]

    # Ulaşılabilir hedefler (daha yakın ve makul)
    targets_rel = [
        torch.tensor([0.25, 0.15, 0.05], device="cuda:0"),   # Front-right
        torch.tensor([0.15, 0.30, 0.10], device="cuda:0"),   # Right
        torch.tensor([0.20, 0.20, 0.25], device="cuda:0"),   # Up
        torch.tensor([0.30, 0.10, -0.05], device="cuda:0"),  # Front-down
        torch.tensor([0.10, 0.25, 0.15], device="cuda:0"),   # Side-up
    ]

    print(f"\n[INFO] Robot root: {root_pos.tolist()}")
    print(f"[INFO] {len(targets_rel)} hedef pozisyonu tanımlı")
    print("[INFO] Hedefe < 8cm yaklaşınca yeni hedef!")
    print("[INFO] Ctrl+C ile çık\n")

    step = 0
    target_idx = 0
    reach_count = 0
    reach_threshold = 0.08  # 8cm

    # İlk hedef
    target_rel = targets_rel[target_idx].clone()
    env.target_pos[0] = target_rel

    try:
        while simulation_app.is_running():
            step += 1

            # ===== EE POZİSYONU =====
            ee_pos_world = env._compute_ee_pos()
            ee_pos_rel = ee_pos_world[0] - root_pos

            # ===== MESAFE KONTROLÜ =====
            error = target_rel - ee_pos_rel
            distance = error.norm().item()

            # Hedefe ulaştı mı?
            if distance < reach_threshold:
                reach_count += 1
                target_idx = (target_idx + 1) % len(targets_rel)
                target_rel = targets_rel[target_idx].clone()
                env.target_pos[0] = target_rel
                print(f"  🎯 REACHED! Total: {reach_count} | New target #{target_idx}")

            # ===== TARGET SPHERE GÜNCELLE =====
            target_world = root_pos + target_rel
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
            target_pose = torch.cat([target_world, default_quat]).unsqueeze(0)
            env.target_obj.write_root_pose_to_sim(target_pose)

            # ===== EE MARKER GÜNCELLE =====
            palm_quat = env.robot.data.body_quat_w[:, env.palm_idx]
            ee_marker_pose = torch.cat([ee_pos_world[0], palm_quat[0]]).unsqueeze(0)
            env.ee_marker.write_root_pose_to_sim(ee_marker_pose)

            # ===== DAHA HIZLI P CONTROLLER =====
            current_joints = env.robot.data.joint_pos[0, env.arm_joint_indices]

            # Jacobian-like mapping (basitleştirilmiş)
            # x error → shoulder pitch (ileri/geri)
            # y error → shoulder roll (yana)
            # z error → elbow pitch + shoulder pitch

            Kp = 8.0  # ARTIRILDI (2.0 → 8.0)

            joint_delta = torch.tensor([
                Kp * error[0] - Kp * 0.3 * error[2],  # shoulder_pitch
                -Kp * error[1],                        # shoulder_roll
                0.0,                                   # shoulder_yaw
                -Kp * error[2] - Kp * 0.2 * error[0], # elbow_pitch
                0.0,                                   # elbow_roll
            ], device="cuda:0")

            # Step size ARTIRILDI (0.02 → 0.05)
            step_size = 0.05
            target_joints = current_joints + joint_delta * step_size
            target_joints = torch.clamp(target_joints, env.joint_lower, env.joint_upper)

            # Joint target set
            all_joint_targets = env.robot.data.joint_pos.clone()
            all_joint_targets[0, env.arm_joint_indices] = target_joints
            env.robot.set_joint_position_target(all_joint_targets)

            # ===== SİMÜLASYON =====
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(env.sim.get_physics_dt())

            # Log (her 100 step)
            if step % 100 == 0:
                print(f"[Step {step:4d}] Target #{target_idx} | Distance: {distance:.3f}m | Reaches: {reach_count}")
                print(f"  Target: [{target_rel[0]:.2f}, {target_rel[1]:.2f}, {target_rel[2]:.2f}]")
                print(f"  EE:     [{ee_pos_rel[0]:.2f}, {ee_pos_rel[1]:.2f}, {ee_pos_rel[2]:.2f}]")
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print(f"  Total Steps: {step}")
    print(f"  Total Reaches: {reach_count}")
    print(f"  Reach Rate: {reach_count / (step/100):.1f} per 100 steps")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()