#!/usr/bin/env python3
"""
G1 Arm Poses v2 - Corrected Kinematics
=======================================

G1 kinematik yapısı:
- shoulder_pitch: Kolu yukarı/aşağı (- = yukarı, + = aşağı)
- shoulder_roll:  Kolu yanlara aç (+ = sol dışa, - = sağ dışa)
- shoulder_yaw:   Kolu döndür/öne getir
- elbow_pitch:    Dirsek bük
- elbow_roll:     Bilek rotasyonu

ÖNE UZATMAK İÇİN:
1. shoulder_pitch = -1.57 (kolu yukarı kaldır)
2. shoulder_yaw = ±1.57 (kolu öne döndür)
VEYA
1. shoulder_roll ile yana aç
2. shoulder_yaw ile öne döndür

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/test_arm_v2.py --num_envs 1 --pose reach_forward
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Arm Poses v2")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--pose", type=str, default="reach_forward",
                    choices=["reach_forward", "reach_down", "reach_side",
                             "hands_up", "explore", "push"])

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

print("\n" + "=" * 70)
print("  G1 ARM POSES v2 - Corrected Kinematics")
print(f"  Pose: {args.pose.upper()}")
print("=" * 70 + "\n")


def main():
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 1.0])

    # Load robot
    print("[INFO] Loading G1 robot...")

    try:
        from isaaclab_assets import G1_MINIMAL_CFG
        robot_cfg = G1_MINIMAL_CFG.copy()
    except:
        from isaaclab_assets import G1_CFG
        robot_cfg = G1_CFG.copy()

    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)

    try:
        robot_cfg.spawn.articulation_props.fix_root_link = True
    except:
        pass

    # Scene
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    robot.update(sim.cfg.dt)

    # Find joints
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    # Find arm joints
    left_arm = {}
    right_arm = {}

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if "left" in name_lower:
            if "shoulder_pitch" in name_lower:
                left_arm["shoulder_pitch"] = i
            elif "shoulder_roll" in name_lower:
                left_arm["shoulder_roll"] = i
            elif "shoulder_yaw" in name_lower:
                left_arm["shoulder_yaw"] = i
            elif "elbow_pitch" in name_lower:
                left_arm["elbow_pitch"] = i
            elif "elbow_roll" in name_lower:
                left_arm["elbow_roll"] = i
        elif "right" in name_lower:
            if "shoulder_pitch" in name_lower:
                right_arm["shoulder_pitch"] = i
            elif "shoulder_roll" in name_lower:
                right_arm["shoulder_roll"] = i
            elif "shoulder_yaw" in name_lower:
                right_arm["shoulder_yaw"] = i
            elif "elbow_pitch" in name_lower:
                right_arm["elbow_pitch"] = i
            elif "elbow_roll" in name_lower:
                right_arm["elbow_roll"] = i

    print(f"[INFO] Left arm: {left_arm}")
    print(f"[INFO] Right arm: {right_arm}")

    # Find EE bodies
    left_ee_idx = 28
    right_ee_idx = 29
    for i, name in enumerate(body_names):
        if "left_palm" in name.lower():
            left_ee_idx = i
        elif "right_palm" in name.lower():
            right_ee_idx = i

    # Get initial positions
    init_joints = robot.data.joint_pos.clone()
    init_left_ee = robot.data.body_pos_w[0, left_ee_idx].clone()
    init_right_ee = robot.data.body_pos_w[0, right_ee_idx].clone()

    print(f"\n[INFO] Initial left EE: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")
    print(f"[INFO] Initial right EE: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")

    # New poses with correct understanding
    poses = {
        "reach_forward": {
            # Öne uzat: pitch ile yukarı, yaw ile öne döndür
            "left": {
                "shoulder_pitch": -1.2,  # Yukarı kaldır
                "shoulder_roll": 0.3,  # Hafif yana
                "shoulder_yaw": -1.2,  # Öne döndür (sol için negatif)
                "elbow_pitch": 0.3,  # Hafif bük
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": -1.2,
                "shoulder_roll": -0.3,
                "shoulder_yaw": 1.2,  # Öne döndür (sağ için pozitif)
                "elbow_pitch": 0.3,
                "elbow_roll": 0.0,
            }
        },
        "push": {
            # İtme pozu - kollar düz öne
            "left": {
                "shoulder_pitch": -1.57,  # 90 derece yukarı
                "shoulder_roll": 0.0,
                "shoulder_yaw": -1.57,  # 90 derece öne döndür
                "elbow_pitch": 0.0,  # Düz
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": -1.57,
                "shoulder_roll": 0.0,
                "shoulder_yaw": 1.57,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "reach_down": {
            # Aşağı uzan (bir şey al)
            "left": {
                "shoulder_pitch": 0.5,  # Aşağı
                "shoulder_roll": 0.3,
                "shoulder_yaw": -0.5,  # Hafif öne
                "elbow_pitch": 0.3,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": 0.5,
                "shoulder_roll": -0.3,
                "shoulder_yaw": 0.5,
                "elbow_pitch": 0.3,
                "elbow_roll": 0.0,
            }
        },
        "reach_side": {
            # Yanlara uzan (T-pose ama düz)
            "left": {
                "shoulder_pitch": 0.0,
                "shoulder_roll": 1.57,  # Tam yana
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": 0.0,
                "shoulder_roll": -1.57,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "hands_up": {
            # Eller yukarı
            "left": {
                "shoulder_pitch": -2.5,  # Tam yukarı
                "shoulder_roll": 0.3,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": -2.5,
                "shoulder_roll": -0.3,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "explore": {
            # Her joint'i test et
            "left": {
                "shoulder_pitch": -0.8,
                "shoulder_roll": 0.5,
                "shoulder_yaw": -0.8,
                "elbow_pitch": 0.8,
                "elbow_roll": 0.3,
            },
            "right": {
                "shoulder_pitch": -0.8,
                "shoulder_roll": -0.5,
                "shoulder_yaw": 0.8,
                "elbow_pitch": 0.8,
                "elbow_roll": -0.3,
            }
        },
    }

    target_pose = poses[args.pose]

    print(f"\n[INFO] Target pose: {args.pose.upper()}")
    print(f"[INFO] Left: pitch={target_pose['left']['shoulder_pitch']:.2f}, "
          f"roll={target_pose['left']['shoulder_roll']:.2f}, "
          f"yaw={target_pose['left']['shoulder_yaw']:.2f}")

    print("\n" + "=" * 60)
    print(f"  MOVING TO {args.pose.upper()}")
    print("=" * 60)

    extend_duration = 3.0
    hold_duration = 5.0
    steps = int((extend_duration + hold_duration) / sim.cfg.dt)

    for step in range(steps):
        t = step * sim.cfg.dt

        # Interpolation
        if t < extend_duration:
            alpha = t / extend_duration
            alpha = 0.5 * (1 - math.cos(math.pi * alpha))
        else:
            alpha = 1.0

        # Build targets
        joint_targets = init_joints.clone()

        # Left arm
        for joint_name, target_val in target_pose["left"].items():
            if joint_name in left_arm:
                idx = left_arm[joint_name]
                init_val = init_joints[0, idx].item()
                joint_targets[0, idx] = init_val * (1 - alpha) + target_val * alpha

        # Right arm
        for joint_name, target_val in target_pose["right"].items():
            if joint_name in right_arm:
                idx = right_arm[joint_name]
                init_val = init_joints[0, idx].item()
                joint_targets[0, idx] = init_val * (1 - alpha) + target_val * alpha

        # Apply
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim.cfg.dt)

        # Log
        if step % 100 == 0:
            left_ee = robot.data.body_pos_w[0, left_ee_idx]
            right_ee = robot.data.body_pos_w[0, right_ee_idx]

            left_delta = left_ee - init_left_ee
            right_delta = right_ee - init_right_ee

            print(f"  [{step:4d}] t={t:.1f}s | "
                  f"L: X{left_delta[0]:+.2f} Y{left_delta[1]:+.2f} Z{left_delta[2]:+.2f} | "
                  f"R: X{right_delta[0]:+.2f} Y{right_delta[1]:+.2f} Z{right_delta[2]:+.2f}")

    # Final result
    final_left = robot.data.body_pos_w[0, left_ee_idx]
    final_right = robot.data.body_pos_w[0, right_ee_idx]

    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)

    left_fwd = (final_left[0] - init_left_ee[0]).item()
    right_fwd = (final_right[0] - init_right_ee[0]).item()

    print(f"\n  Left arm:")
    print(f"    Initial: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")
    print(f"    Final:   ({final_left[0]:.3f}, {final_left[1]:.3f}, {final_left[2]:.3f})")
    print(f"    Delta X (forward): {left_fwd:+.3f}m")
    print(f"    Delta Y (side):    {(final_left[1] - init_left_ee[1]).item():+.3f}m")
    print(f"    Delta Z (up):      {(final_left[2] - init_left_ee[2]).item():+.3f}m")

    print(f"\n  Right arm:")
    print(f"    Initial: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"    Final:   ({final_right[0]:.3f}, {final_right[1]:.3f}, {final_right[2]:.3f})")
    print(f"    Delta X (forward): {right_fwd:+.3f}m")
    print(f"    Delta Y (side):    {(final_right[1] - init_right_ee[1]).item():+.3f}m")
    print(f"    Delta Z (up):      {(final_right[2] - init_right_ee[2]).item():+.3f}m")

    # Success check for forward poses
    if args.pose in ["reach_forward", "push"]:
        if left_fwd > 0.15 and right_fwd > 0.15:
            print(f"\n  🎉 SUCCESS! Kollar öne uzandı! (>{0.15}m)")
        elif left_fwd > 0.08 and right_fwd > 0.08:
            print(f"\n  ✅ GOOD! Kollar öne gitti (>{0.08}m)")
        elif left_fwd > 0.03 and right_fwd > 0.03:
            print(f"\n  ⚠️ Biraz öne gitti ama yetersiz")
        else:
            print(f"\n  ❌ FAIL - kollar öne gitmedi")

    print("=" * 60 + "\n")

    # Hold
    print("[INFO] Holding for observation (5 sec)...")
    for _ in range(500):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()