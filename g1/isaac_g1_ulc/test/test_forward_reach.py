#!/usr/bin/env python3
"""
G1 Forward Reach Test - NEGATIVE shoulder_pitch ile
====================================================

Test sonuçlarına göre:
- shoulder_pitch NEGATIVE = öne (+X) ve yukarı (+Z)
- shoulder_pitch POSITIVE = arkaya (-X) ve aşağı (-Z)

Öne uzanmak için: kolu yukarı kaldırıp, dirsekle uzatmak lazım!

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_forward_reach.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Forward Reach Test")
parser.add_argument("--num_envs", type=int, default=1)

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
print("  G1 FORWARD REACH TEST")
print("  shoulder_pitch NEG + elbow combination")
print("=" * 70 + "\n")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 1.0])

    # Load robot
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

    # Joint indices
    joints = {}
    for i, name in enumerate(joint_names):
        joints[name] = i

    # RIGHT arm joints
    r_pitch = None
    r_roll = None
    r_yaw = None
    r_elbow_pitch = None
    r_elbow_roll = None

    for name, idx in joints.items():
        nl = name.lower()
        if "right" in nl:
            if "shoulder_pitch" in nl:
                r_pitch = idx
            elif "shoulder_roll" in nl:
                r_roll = idx
            elif "shoulder_yaw" in nl:
                r_yaw = idx
            elif "elbow_pitch" in nl:
                r_elbow_pitch = idx
            elif "elbow_roll" in nl:
                r_elbow_roll = idx

    # LEFT arm joints
    l_pitch = None
    l_roll = None
    l_yaw = None
    l_elbow_pitch = None
    l_elbow_roll = None

    for name, idx in joints.items():
        nl = name.lower()
        if "left" in nl:
            if "shoulder_pitch" in nl:
                l_pitch = idx
            elif "shoulder_roll" in nl:
                l_roll = idx
            elif "shoulder_yaw" in nl:
                l_yaw = idx
            elif "elbow_pitch" in nl:
                l_elbow_pitch = idx
            elif "elbow_roll" in nl:
                l_elbow_roll = idx

    print(f"[INFO] Right arm: pitch={r_pitch}, roll={r_roll}, yaw={r_yaw}, elbow={r_elbow_pitch}")
    print(f"[INFO] Left arm: pitch={l_pitch}, roll={l_roll}, yaw={l_yaw}, elbow={l_elbow_pitch}")

    # EE bodies
    right_ee_idx = 29
    left_ee_idx = 28
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            right_ee_idx = i
        if "left_palm" in name.lower():
            left_ee_idx = i

    init_joints = robot.data.joint_pos.clone()
    init_right_ee = robot.data.body_pos_w[0, right_ee_idx].clone()
    init_left_ee = robot.data.body_pos_w[0, left_ee_idx].clone()

    print(f"\n[INFO] Initial Right EE: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"[INFO] Initial Left EE: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")

    # Forward reach pose configurations to test
    # Key insight: shoulder_pitch NEG = arm goes UP and FORWARD
    # We need to combine with elbow to extend forward

    poses = {
        "forward_1": {
            # Shoulder pitch negative to raise arm, elbow positive to extend
            "shoulder_pitch": -0.8,  # Raise arm ~45 degrees
            "shoulder_roll": 0.0,  # No lateral
            "shoulder_yaw": 0.0,  # No rotation
            "elbow_pitch": 1.2,  # Extend elbow forward
        },
        "forward_2": {
            # More horizontal reach
            "shoulder_pitch": -1.2,  # Raise more
            "shoulder_roll": 0.0,
            "shoulder_yaw": 0.0,
            "elbow_pitch": 1.5,  # More extension
        },
        "forward_3": {
            # Maximum forward
            "shoulder_pitch": -1.57,  # ~90 degrees up
            "shoulder_roll": 0.0,
            "shoulder_yaw": 0.0,
            "elbow_pitch": 1.57,  # Straight elbow
        },
        "forward_with_roll": {
            # Try adding roll for forward motion
            "shoulder_pitch": -1.0,
            "shoulder_roll": -0.5,  # Negative roll for right arm
            "shoulder_yaw": 0.0,
            "elbow_pitch": 1.2,
        },
        "punch": {
            # Punching motion
            "shoulder_pitch": -1.57,  # Arm up
            "shoulder_roll": 0.0,
            "shoulder_yaw": 0.0,
            "elbow_pitch": 0.0,  # Straight arm
        },
    }

    results = {}

    for pose_name, pose_cfg in poses.items():
        print(f"\n{'=' * 60}")
        print(f"  TESTING: {pose_name}")
        print(f"{'=' * 60}")

        # Reset
        robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        # Apply pose gradually
        for step in range(200):
            alpha = min(1.0, step / 150.0)
            alpha = 0.5 * (1 - math.cos(math.pi * alpha))

            joint_targets = init_joints.clone()

            # Right arm
            if r_pitch is not None:
                joint_targets[0, r_pitch] = pose_cfg["shoulder_pitch"] * alpha
            if r_roll is not None:
                joint_targets[0, r_roll] = pose_cfg["shoulder_roll"] * alpha
            if r_yaw is not None:
                joint_targets[0, r_yaw] = pose_cfg["shoulder_yaw"] * alpha
            if r_elbow_pitch is not None:
                joint_targets[0, r_elbow_pitch] = pose_cfg["elbow_pitch"] * alpha

            # Left arm (mirror for roll)
            if l_pitch is not None:
                joint_targets[0, l_pitch] = pose_cfg["shoulder_pitch"] * alpha
            if l_roll is not None:
                # Roll is mirrored for left arm
                joint_targets[0, l_roll] = -pose_cfg["shoulder_roll"] * alpha
            if l_yaw is not None:
                joint_targets[0, l_yaw] = pose_cfg["shoulder_yaw"] * alpha
            if l_elbow_pitch is not None:
                joint_targets[0, l_elbow_pitch] = pose_cfg["elbow_pitch"] * alpha

            robot.set_joint_position_target(joint_targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        # Stabilize
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        # Measure
        final_right_ee = robot.data.body_pos_w[0, right_ee_idx]
        final_left_ee = robot.data.body_pos_w[0, left_ee_idx]

        delta_r = final_right_ee - init_right_ee
        delta_l = final_left_ee - init_left_ee

        print(f"  Config: pitch={pose_cfg['shoulder_pitch']:.2f}, "
              f"roll={pose_cfg['shoulder_roll']:.2f}, "
              f"yaw={pose_cfg['shoulder_yaw']:.2f}, "
              f"elbow={pose_cfg['elbow_pitch']:.2f}")
        print(f"\n  Right arm delta:")
        print(f"    X (forward): {delta_r[0].item():+.3f}m")
        print(f"    Y (side):    {delta_r[1].item():+.3f}m")
        print(f"    Z (up):      {delta_r[2].item():+.3f}m")

        print(f"\n  Left arm delta:")
        print(f"    X (forward): {delta_l[0].item():+.3f}m")
        print(f"    Y (side):    {delta_l[1].item():+.3f}m")
        print(f"    Z (up):      {delta_l[2].item():+.3f}m")

        # Check success
        forward_r = delta_r[0].item()
        forward_l = delta_l[0].item()

        if forward_r > 0.15:
            print(f"\n  🎉 RIGHT ARM: Excellent forward reach!")
        elif forward_r > 0.08:
            print(f"\n  ✅ RIGHT ARM: Good forward reach")
        elif forward_r > 0:
            print(f"\n  ⚠️ RIGHT ARM: Some forward motion")
        else:
            print(f"\n  ❌ RIGHT ARM: No forward motion")

        results[pose_name] = {
            "right_x": forward_r,
            "right_y": delta_r[1].item(),
            "right_z": delta_r[2].item(),
            "left_x": forward_l,
            "left_y": delta_l[1].item(),
            "left_z": delta_l[2].item(),
        }

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY - Forward Reach Results")
    print("=" * 70)

    print(f"\n  {'Pose':<20} {'Right X':>10} {'Right Z':>10} {'Left X':>10} {'Left Z':>10}")
    print("  " + "-" * 62)

    best_pose = None
    best_forward = -999

    for pose_name, data in results.items():
        print(f"  {pose_name:<20} {data['right_x']:>+10.3f} {data['right_z']:>+10.3f} "
              f"{data['left_x']:>+10.3f} {data['left_z']:>+10.3f}")

        avg_forward = (data['right_x'] + data['left_x']) / 2
        if avg_forward > best_forward:
            best_forward = avg_forward
            best_pose = pose_name

    print(f"\n  BEST POSE: {best_pose} (avg forward: {best_forward:+.3f}m)")
    print("=" * 70)

    # Hold best pose
    print(f"\n[INFO] Showing best pose: {best_pose}")
    best_cfg = poses[best_pose]

    robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
    for step in range(200):
        alpha = min(1.0, step / 150.0)
        alpha = 0.5 * (1 - math.cos(math.pi * alpha))

        joint_targets = init_joints.clone()

        if r_pitch is not None:
            joint_targets[0, r_pitch] = best_cfg["shoulder_pitch"] * alpha
        if r_roll is not None:
            joint_targets[0, r_roll] = best_cfg["shoulder_roll"] * alpha
        if r_yaw is not None:
            joint_targets[0, r_yaw] = best_cfg["shoulder_yaw"] * alpha
        if r_elbow_pitch is not None:
            joint_targets[0, r_elbow_pitch] = best_cfg["elbow_pitch"] * alpha

        if l_pitch is not None:
            joint_targets[0, l_pitch] = best_cfg["shoulder_pitch"] * alpha
        if l_roll is not None:
            joint_targets[0, l_roll] = -best_cfg["shoulder_roll"] * alpha
        if l_yaw is not None:
            joint_targets[0, l_yaw] = best_cfg["shoulder_yaw"] * alpha
        if l_elbow_pitch is not None:
            joint_targets[0, l_elbow_pitch] = best_cfg["elbow_pitch"] * alpha

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("[INFO] Holding for observation (10 sec)...")
    for _ in range(1000):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()