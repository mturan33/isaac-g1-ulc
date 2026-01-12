#!/usr/bin/env python3
"""
G1 Joint Explorer - Her Joint'i Tek Tek Test Et
================================================

G1'in kinematik yapısını anlamak için her kol joint'ini
ayrı ayrı hareket ettirip EE'nin nasıl değiştiğini gözlemle.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_joint_explorer.py --num_envs 1
"""

import argparse
import math
import time

parser = argparse.ArgumentParser(description="G1 Joint Explorer")
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
print("  G1 JOINT EXPLORER")
print("  Her joint'i tek tek test et")
print("=" * 70 + "\n")


def main():
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 1.0])

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

    # Find RIGHT arm joints (test one arm)
    right_arm_joints = {}
    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if "right" in name_lower:
            if "shoulder_pitch" in name_lower:
                right_arm_joints["right_shoulder_pitch"] = i
            elif "shoulder_roll" in name_lower:
                right_arm_joints["right_shoulder_roll"] = i
            elif "shoulder_yaw" in name_lower:
                right_arm_joints["right_shoulder_yaw"] = i
            elif "elbow_pitch" in name_lower:
                right_arm_joints["right_elbow_pitch"] = i
            elif "elbow_roll" in name_lower:
                right_arm_joints["right_elbow_roll"] = i

    print(f"[INFO] Right arm joints: {right_arm_joints}")

    # Find EE
    right_ee_idx = 29
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            right_ee_idx = i
            break

    print(f"[INFO] Right EE body index: {right_ee_idx}")

    # Get joint limits
    joint_limits = robot.data.joint_limits[0]  # (num_joints, 2)

    print("\n[INFO] Joint limits:")
    for name, idx in right_arm_joints.items():
        low = joint_limits[idx, 0].item()
        high = joint_limits[idx, 1].item()
        print(f"  {name}: [{low:.2f}, {high:.2f}] rad = [{math.degrees(low):.1f}°, {math.degrees(high):.1f}°]")

    # Test each joint
    print("\n" + "=" * 70)
    print("  TESTING EACH JOINT INDIVIDUALLY")
    print("=" * 70)

    init_joints = robot.data.joint_pos.clone()
    init_ee = robot.data.body_pos_w[0, right_ee_idx].clone()

    print(f"\n[INFO] Initial EE position: ({init_ee[0]:.3f}, {init_ee[1]:.3f}, {init_ee[2]:.3f})")

    results = {}

    for joint_name, joint_idx in right_arm_joints.items():
        print(f"\n{'=' * 60}")
        print(f"  Testing: {joint_name} (index {joint_idx})")
        print(f"{'=' * 60}")

        # Get limits
        low = joint_limits[joint_idx, 0].item()
        high = joint_limits[joint_idx, 1].item()

        # Test values: negative, zero, positive
        test_values = [
            ("NEG", max(low, -1.0)),
            ("ZERO", 0.0),
            ("POS", min(high, 1.0)),
        ]

        joint_results = {}

        for label, target_val in test_values:
            # Reset to initial
            robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))

            # Smoothly move to target
            for step in range(150):  # 1.5 sec
                alpha = min(1.0, step / 100.0)
                alpha = 0.5 * (1 - math.cos(math.pi * alpha))

                joint_targets = init_joints.clone()
                init_val = init_joints[0, joint_idx].item()
                joint_targets[0, joint_idx] = init_val * (1 - alpha) + target_val * alpha

                robot.set_joint_position_target(joint_targets)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.cfg.dt)

            # Get final EE
            final_ee = robot.data.body_pos_w[0, right_ee_idx]
            delta = final_ee - init_ee

            joint_results[label] = {
                "value": target_val,
                "delta_x": delta[0].item(),
                "delta_y": delta[1].item(),
                "delta_z": delta[2].item(),
            }

            print(f"  {label} ({target_val:+.2f} rad): "
                  f"X{delta[0].item():+.3f} Y{delta[1].item():+.3f} Z{delta[2].item():+.3f}")

        # Analyze which direction this joint moves
        neg = joint_results["NEG"]
        pos = joint_results["POS"]

        # Calculate dominant direction
        dx = pos["delta_x"] - neg["delta_x"]
        dy = pos["delta_y"] - neg["delta_y"]
        dz = pos["delta_z"] - neg["delta_z"]

        dominant = "X (forward)" if abs(dx) > abs(dy) and abs(dx) > abs(dz) else \
            "Y (side)" if abs(dy) > abs(dz) else "Z (up)"

        print(f"\n  → Dominant motion: {dominant}")
        print(f"  → Effect: POS-NEG = X{dx:+.3f} Y{dy:+.3f} Z{dz:+.3f}")

        # Determine what positive value does
        if abs(dx) > 0.02:
            if dx > 0:
                print(f"  → POS (+1.0) moves EE FORWARD (+X)")
            else:
                print(f"  → POS (+1.0) moves EE BACKWARD (-X)")
        if abs(dy) > 0.02:
            if dy > 0:
                print(f"  → POS (+1.0) moves EE LEFT (+Y)")
            else:
                print(f"  → POS (+1.0) moves EE RIGHT (-Y)")
        if abs(dz) > 0.02:
            if dz > 0:
                print(f"  → POS (+1.0) moves EE UP (+Z)")
            else:
                print(f"  → POS (+1.0) moves EE DOWN (-Z)")

        results[joint_name] = {
            "tests": joint_results,
            "effect": {"dx": dx, "dy": dy, "dz": dz}
        }

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY - How to reach FORWARD (+X)")
    print("=" * 70)

    forward_joints = []
    for joint_name, data in results.items():
        dx = data["effect"]["dx"]
        if abs(dx) > 0.02:
            if dx > 0:
                forward_joints.append((joint_name, "POSITIVE", dx))
            else:
                forward_joints.append((joint_name, "NEGATIVE", -dx))

    forward_joints.sort(key=lambda x: x[2], reverse=True)

    print("\n  To move EE forward (+X), use:")
    for joint, direction, effect in forward_joints:
        print(f"    • {joint}: {direction} value (effect: {effect:.3f}m per rad)")

    # Best combination for forward
    print("\n" + "=" * 70)
    print("  RECOMMENDED FORWARD POSE")
    print("=" * 70)

    print("\n  Based on analysis, try these joint values:")
    for joint, direction, effect in forward_joints:
        idx = right_arm_joints[joint]
        low = joint_limits[idx, 0].item()
        high = joint_limits[idx, 1].item()

        if direction == "POSITIVE":
            recommended = min(high * 0.8, 1.5)
        else:
            recommended = max(low * 0.8, -1.5)

        print(f"    {joint}: {recommended:+.2f} rad ({direction})")

    print("\n" + "=" * 70)
    print("  TESTING COMBINED FORWARD POSE")
    print("=" * 70)

    # Reset
    robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))

    # Apply all forward-moving joints
    for step in range(200):
        alpha = min(1.0, step / 150.0)
        alpha = 0.5 * (1 - math.cos(math.pi * alpha))

        joint_targets = init_joints.clone()

        for joint, direction, effect in forward_joints:
            idx = right_arm_joints[joint]
            low = joint_limits[idx, 0].item()
            high = joint_limits[idx, 1].item()

            if direction == "POSITIVE":
                target = min(high * 0.7, 1.2)
            else:
                target = max(low * 0.7, -1.2)

            init_val = init_joints[0, idx].item()
            joint_targets[0, idx] = init_val * (1 - alpha) + target * alpha

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # Final result
    final_ee = robot.data.body_pos_w[0, right_ee_idx]
    delta = final_ee - init_ee

    print(f"\n  Combined forward pose result:")
    print(f"    Initial: ({init_ee[0]:.3f}, {init_ee[1]:.3f}, {init_ee[2]:.3f})")
    print(f"    Final:   ({final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f})")
    print(f"    Delta X (forward): {delta[0].item():+.3f}m")
    print(f"    Delta Y (side):    {delta[1].item():+.3f}m")
    print(f"    Delta Z (up):      {delta[2].item():+.3f}m")

    if delta[0].item() > 0.15:
        print(f"\n  🎉 SUCCESS! Forward reach > 15cm!")
    elif delta[0].item() > 0.08:
        print(f"\n  ✅ Good progress - forward reach > 8cm")
    else:
        print(f"\n  ⚠️ Need more exploration")

    print("\n" + "=" * 70)

    # Hold for observation
    print("[INFO] Holding pose for observation...")
    for _ in range(500):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()