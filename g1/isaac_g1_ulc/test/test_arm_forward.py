#!/usr/bin/env python3
"""
G1 Arm Forward Extension - Corrected
=====================================

G1 kinematik yapısı:
- shoulder_pitch negatif = kollar YUKARI
- shoulder_pitch pozitif = kollar AŞAĞI/GERİ

Öne uzatmak için elbow ile kompanze etmemiz lazım.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/test_arm_forward.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Arm Forward")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--pose", type=str, default="forward",
                    choices=["forward", "up", "side", "down", "tpose", "zombie"])

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
print("  G1 ARM POSES TEST")
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

    # Define different poses
    # G1 kinematik:
    # - shoulder_pitch: + = aşağı/geri, - = yukarı
    # - shoulder_roll: + = sol için dışa, - = sağ için dışa
    # - shoulder_yaw: rotasyon
    # - elbow_pitch: + = bükme?, - = düzleştirme?

    poses = {
        "forward": {
            # Zombie pose - kollar düz öne
            "left": {
                "shoulder_pitch": 0.0,  # Nötr
                "shoulder_roll": 1.5,  # Kolu yanlara açmadan öne
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,  # Düz
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": 0.0,
                "shoulder_roll": -1.5,  # Sağ için ters
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "zombie": {
            # Klasik zombi - kollar dümdüz öne
            "left": {
                "shoulder_pitch": 0.3,  # Hafif aşağı
                "shoulder_roll": 1.57,  # 90 derece
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": 0.3,
                "shoulder_roll": -1.57,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "up": {
            # Eller yukarı
            "left": {
                "shoulder_pitch": -1.5,
                "shoulder_roll": 0.3,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": -1.5,
                "shoulder_roll": -0.3,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "side": {
            # T-pose benzeri - kollar yanlara
            "left": {
                "shoulder_pitch": 0.0,
                "shoulder_roll": 1.57,  # 90 derece dışa
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
        "tpose": {
            # Tam T-pose
            "left": {
                "shoulder_pitch": 0.0,
                "shoulder_roll": 2.0,  # Maksimum dışa
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": 0.0,
                "shoulder_roll": -2.0,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
        "down": {
            # Eller aşağı
            "left": {
                "shoulder_pitch": 1.0,
                "shoulder_roll": 0.2,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            },
            "right": {
                "shoulder_pitch": 1.0,
                "shoulder_roll": -0.2,
                "shoulder_yaw": 0.0,
                "elbow_pitch": 0.0,
                "elbow_roll": 0.0,
            }
        },
    }

    target_pose = poses[args.pose]

    print(f"\n[INFO] Target pose: {args.pose.upper()}")
    print(f"[INFO] Left arm targets: {target_pose['left']}")
    print(f"[INFO] Right arm targets: {target_pose['right']}")

    print("\n" + "=" * 60)
    print(f"  MOVING TO {args.pose.upper()} POSE")
    print("=" * 60)

    extend_duration = 2.0
    hold_duration = 5.0
    steps = int((extend_duration + hold_duration) / sim.cfg.dt)

    for step in range(steps):
        t = step * sim.cfg.dt

        # Interpolation
        if t < extend_duration:
            alpha = t / extend_duration
            alpha = 0.5 * (1 - math.cos(math.pi * alpha))  # Smooth
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

    print(f"\n  Left arm:")
    print(f"    Initial: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")
    print(f"    Final:   ({final_left[0]:.3f}, {final_left[1]:.3f}, {final_left[2]:.3f})")
    print(f"    Delta X (forward): {(final_left[0] - init_left_ee[0]).item():+.3f}m")
    print(f"    Delta Y (side):    {(final_left[1] - init_left_ee[1]).item():+.3f}m")
    print(f"    Delta Z (up):      {(final_left[2] - init_left_ee[2]).item():+.3f}m")

    print(f"\n  Right arm:")
    print(f"    Initial: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"    Final:   ({final_right[0]:.3f}, {final_right[1]:.3f}, {final_right[2]:.3f})")
    print(f"    Delta X (forward): {(final_right[0] - init_right_ee[0]).item():+.3f}m")
    print(f"    Delta Y (side):    {(final_right[1] - init_right_ee[1]).item():+.3f}m")
    print(f"    Delta Z (up):      {(final_right[2] - init_right_ee[2]).item():+.3f}m")

    # Success check
    left_fwd = (final_left[0] - init_left_ee[0]).item()
    right_fwd = (final_right[0] - init_right_ee[0]).item()

    if args.pose in ["forward", "zombie"]:
        if left_fwd > 0.15 and right_fwd > 0.15:
            print(f"\n  🎉 SUCCESS! Kollar öne uzandı!")
        elif left_fwd > 0.05 and right_fwd > 0.05:
            print(f"\n  ⚠️ Kısmen başarılı - kollar biraz öne gitti")
        else:
            print(f"\n  ❌ FAIL - kollar öne uzanmadı")

    print("=" * 60 + "\n")

    # Hold
    print("[INFO] Holding for observation...")
    for _ in range(500):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()