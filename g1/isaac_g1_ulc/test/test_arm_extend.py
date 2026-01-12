#!/usr/bin/env python3
"""
G1 Arm Extension Test
======================

Kolları öne doğru uzat - basit joint interpolation.
Jacobian kullanmıyor, direkt joint açıları ile çalışıyor.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/test_arm_extend.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Arm Extension")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=5.0)

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
print("  G1 ARM EXTENSION TEST")
print("  Kolları öne doğru uzat")
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

    print(f"[INFO] Joints: {len(joint_names)}, Bodies: {len(body_names)}")

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

    print(f"\n[INFO] Left arm joints: {left_arm}")
    print(f"[INFO] Right arm joints: {right_arm}")

    # Find EE bodies
    left_ee_idx = None
    right_ee_idx = None
    for i, name in enumerate(body_names):
        if "left_palm" in name.lower():
            left_ee_idx = i
        elif "right_palm" in name.lower():
            right_ee_idx = i

    print(f"\n[INFO] Left EE: [{left_ee_idx}]")
    print(f"[INFO] Right EE: [{right_ee_idx}]")

    # Get initial positions
    init_joints = robot.data.joint_pos.clone()
    init_left_ee = robot.data.body_pos_w[0, left_ee_idx].clone()
    init_right_ee = robot.data.body_pos_w[0, right_ee_idx].clone()

    print(f"\n[INFO] Initial left EE: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")
    print(f"[INFO] Initial right EE: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")

    # Define target poses for "arms extended forward"
    # shoulder_pitch: negative = forward
    # shoulder_roll: positive for left (outward), negative for right (outward)
    # elbow_pitch: slight bend

    target_poses = {
        "left": {
            "shoulder_pitch": -1.0,  # Forward
            "shoulder_roll": 0.3,  # Slightly outward
            "shoulder_yaw": 0.0,
            "elbow_pitch": -0.3,  # Slight bend
            "elbow_roll": 0.0,
        },
        "right": {
            "shoulder_pitch": -1.0,  # Forward
            "shoulder_roll": -0.3,  # Slightly outward (opposite for right)
            "shoulder_yaw": 0.0,
            "elbow_pitch": -0.3,  # Slight bend
            "elbow_roll": 0.0,
        }
    }

    print("\n" + "=" * 60)
    print("  EXTENDING ARMS FORWARD")
    print("=" * 60)

    steps = int(args.duration / sim.cfg.dt)
    extend_duration = 2.0  # Time to reach target
    hold_duration = args.duration - extend_duration

    for step in range(steps):
        t = step * sim.cfg.dt

        # Interpolation factor
        if t < extend_duration:
            alpha = t / extend_duration
        else:
            alpha = 1.0

        # Smooth interpolation (ease in-out)
        alpha = 0.5 * (1 - math.cos(math.pi * alpha))

        # Build target joint positions
        joint_targets = init_joints.clone()

        # Left arm
        for joint_name, target_val in target_poses["left"].items():
            if joint_name in left_arm:
                idx = left_arm[joint_name]
                init_val = init_joints[0, idx].item()
                joint_targets[0, idx] = init_val * (1 - alpha) + target_val * alpha

        # Right arm
        for joint_name, target_val in target_poses["right"].items():
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
        if step % 50 == 0:
            left_ee = robot.data.body_pos_w[0, left_ee_idx]
            right_ee = robot.data.body_pos_w[0, right_ee_idx]

            left_fwd = left_ee[0].item() - init_left_ee[0].item()
            right_fwd = right_ee[0].item() - init_right_ee[0].item()

            print(f"  [{step:4d}] t={t:.2f}s | alpha={alpha:.2f} | "
                  f"L_fwd={left_fwd:+.3f}m | R_fwd={right_fwd:+.3f}m")

    # Final positions
    final_left_ee = robot.data.body_pos_w[0, left_ee_idx]
    final_right_ee = robot.data.body_pos_w[0, right_ee_idx]

    left_movement = (final_left_ee - init_left_ee).norm().item()
    right_movement = (final_right_ee - init_right_ee).norm().item()

    left_forward = final_left_ee[0].item() - init_left_ee[0].item()
    right_forward = final_right_ee[0].item() - init_right_ee[0].item()

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"\n  Left arm:")
    print(f"    Initial: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")
    print(f"    Final:   ({final_left_ee[0]:.3f}, {final_left_ee[1]:.3f}, {final_left_ee[2]:.3f})")
    print(f"    Forward: {left_forward:+.3f}m")
    print(f"    Total:   {left_movement:.3f}m")

    print(f"\n  Right arm:")
    print(f"    Initial: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"    Final:   ({final_right_ee[0]:.3f}, {final_right_ee[1]:.3f}, {final_right_ee[2]:.3f})")
    print(f"    Forward: {right_forward:+.3f}m")
    print(f"    Total:   {right_movement:.3f}m")

    if left_forward > 0.1 and right_forward > 0.1:
        print(f"\n  🎉 SUCCESS! Both arms extended forward!")
    elif left_forward > 0.05 or right_forward > 0.05:
        print(f"\n  ⚠️ Partial success - arms moved but not fully extended")
    else:
        print(f"\n  ❌ FAIL - arms didn't extend")

    print("=" * 60 + "\n")

    # Hold for observation
    print("[INFO] Holding pose for 3 seconds...")
    for _ in range(300):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()