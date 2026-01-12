#!/usr/bin/env python3
"""
G1 Arm Movement Test - NO JACOBIAN
===================================

Jacobian kullanmadan direkt joint açılarını değiştirerek
actuator'ların çalışıp çalışmadığını test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/test_arm_simple.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Simple Arm Test")
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
print("  G1 SIMPLE ARM TEST - NO JACOBIAN")
print("  Direkt joint açılarını değiştirerek actuator test")
print("=" * 70 + "\n")


def main():
    # Setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 1.0])

    # Load robot
    print("[INFO] Loading G1 robot...")

    try:
        from isaaclab_assets import G1_MINIMAL_CFG
        robot_cfg = G1_MINIMAL_CFG.copy()
        print("[INFO] Using G1_MINIMAL_CFG")
    except:
        from isaaclab_assets import G1_CFG
        robot_cfg = G1_CFG.copy()
        print("[INFO] Using G1_CFG")

    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)

    # Fix base
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

    print(f"\n[INFO] Robot: {robot.num_joints} joints, {robot.num_bodies} bodies")

    # Find right arm joints
    arm_joint_indices = []
    arm_joint_names = []
    for i, name in enumerate(joint_names):
        if "right" in name.lower():
            if any(x in name.lower() for x in ["shoulder", "elbow"]):
                arm_joint_indices.append(i)
                arm_joint_names.append(name)

    print(f"\n[INFO] Right arm joints:")
    for i, (idx, name) in enumerate(zip(arm_joint_indices, arm_joint_names)):
        print(f"  [{idx}] {name}")

    # Find EE
    ee_idx = None
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            ee_idx = i
            print(f"\n[INFO] EE body: [{i}] {name}")
            break

    if ee_idx is None:
        ee_idx = 29
        print(f"\n[INFO] Using default EE index: {ee_idx}")

    # Get initial state
    init_joint_pos = robot.data.joint_pos.clone()
    init_ee_pos = robot.data.body_pos_w[:, ee_idx].clone()

    print(f"\n[INFO] Initial arm joint positions:")
    for idx, name in zip(arm_joint_indices, arm_joint_names):
        print(f"  {name}: {init_joint_pos[0, idx].item():.4f} rad")

    print(f"\n[INFO] Initial EE position: ({init_ee_pos[0, 0]:.3f}, {init_ee_pos[0, 1]:.3f}, {init_ee_pos[0, 2]:.3f})")

    # ============================================================
    # TEST 1: Move shoulder_pitch forward
    # ============================================================
    print("\n" + "=" * 50)
    print("  TEST 1: Move right_shoulder_pitch forward")
    print("=" * 50)

    # Find shoulder_pitch index
    shoulder_pitch_idx = None
    for idx, name in zip(arm_joint_indices, arm_joint_names):
        if "shoulder_pitch" in name.lower():
            shoulder_pitch_idx = idx
            break

    if shoulder_pitch_idx is None:
        print("[ERROR] Cannot find shoulder_pitch!")
        return

    print(f"[INFO] Shoulder pitch index: {shoulder_pitch_idx}")
    print(f"[INFO] Initial value: {init_joint_pos[0, shoulder_pitch_idx].item():.4f} rad")

    # Move shoulder forward (negative pitch usually raises arm forward)
    target_angle = -0.5  # radians
    print(f"[INFO] Target value: {target_angle:.4f} rad")

    for step in range(200):
        # Interpolate
        t = min(1.0, step / 100.0)
        current_target = init_joint_pos[0, shoulder_pitch_idx].item() * (1 - t) + target_angle * t

        # Set joint target
        joint_targets = robot.data.joint_pos.clone()
        joint_targets[:, shoulder_pitch_idx] = current_target
        robot.set_joint_position_target(joint_targets)

        # IMPORTANT: Write to sim!
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim.cfg.dt)

        if step % 50 == 0:
            current_ee = robot.data.body_pos_w[0, ee_idx]
            current_joint = robot.data.joint_pos[0, shoulder_pitch_idx].item()
            print(f"  [{step:3d}] Joint: {current_joint:.4f} rad | "
                  f"EE: ({current_ee[0]:.3f}, {current_ee[1]:.3f}, {current_ee[2]:.3f})")

    final_ee = robot.data.body_pos_w[0, ee_idx]
    ee_movement = (final_ee - init_ee_pos[0]).norm().item()

    print(f"\n  Final EE: ({final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f})")
    print(f"  EE movement: {ee_movement:.4f}m")

    if ee_movement > 0.05:
        print("  ✅ PASS - Arm moved!")
    else:
        print("  ❌ FAIL - Arm didn't move much")

    # ============================================================
    # TEST 2: Move multiple joints (wave motion)
    # ============================================================
    print("\n" + "=" * 50)
    print("  TEST 2: Wave motion (multiple joints)")
    print("=" * 50)

    # Reset
    robot.write_joint_state_to_sim(init_joint_pos, torch.zeros_like(init_joint_pos))
    for _ in range(20):
        sim.step()
        robot.update(sim.cfg.dt)

    init_ee_pos2 = robot.data.body_pos_w[:, ee_idx].clone()

    # Wave: move shoulder_roll outward
    shoulder_roll_idx = None
    elbow_pitch_idx = None

    for idx, name in zip(arm_joint_indices, arm_joint_names):
        if "shoulder_roll" in name.lower():
            shoulder_roll_idx = idx
        elif "elbow_pitch" in name.lower():
            elbow_pitch_idx = idx

    print(f"[INFO] Shoulder roll idx: {shoulder_roll_idx}")
    print(f"[INFO] Elbow pitch idx: {elbow_pitch_idx}")

    max_ee_y = init_ee_pos2[0, 1].item()

    for step in range(300):
        t = step * 0.01

        joint_targets = init_joint_pos.clone()

        # Shoulder roll: move arm to side
        if shoulder_roll_idx is not None:
            # Negative roll usually moves arm outward for right arm
            roll_target = -0.8 * math.sin(2 * math.pi * 0.3 * t)
            joint_targets[:, shoulder_roll_idx] = roll_target

        # Elbow bend
        if elbow_pitch_idx is not None:
            elbow_target = -0.5 * (1 - math.cos(2 * math.pi * 0.3 * t)) / 2
            joint_targets[:, elbow_pitch_idx] = elbow_target

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim.cfg.dt)

        current_ee = robot.data.body_pos_w[0, ee_idx]
        if current_ee[1].item() > max_ee_y:
            max_ee_y = current_ee[1].item()

        if step % 50 == 0:
            print(f"  [{step:3d}] EE: ({current_ee[0]:.3f}, {current_ee[1]:.3f}, {current_ee[2]:.3f})")

    y_movement = max_ee_y - init_ee_pos2[0, 1].item()
    print(f"\n  Max Y movement: {y_movement:.4f}m")

    if y_movement > 0.05:
        print("  ✅ PASS - Arm waved!")
    else:
        print("  ❌ FAIL - Arm didn't wave")

    # ============================================================
    # TEST 3: Check joint limits and actuator strength
    # ============================================================
    print("\n" + "=" * 50)
    print("  TEST 3: Check actuator response")
    print("=" * 50)

    # Reset
    robot.write_joint_state_to_sim(init_joint_pos, torch.zeros_like(init_joint_pos))
    for _ in range(20):
        sim.step()
        robot.update(sim.cfg.dt)

    print("\n[INFO] Testing each arm joint individually:")

    for idx, name in zip(arm_joint_indices, arm_joint_names):
        # Try to move this joint
        init_val = init_joint_pos[0, idx].item()
        target_val = init_val + 0.5  # Try +0.5 rad

        for step in range(100):
            t = min(1.0, step / 50.0)
            joint_targets = init_joint_pos.clone()
            joint_targets[:, idx] = init_val * (1 - t) + target_val * t
            robot.set_joint_position_target(joint_targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        final_val = robot.data.joint_pos[0, idx].item()
        movement = abs(final_val - init_val)

        status = "✅" if movement > 0.1 else "❌"
        print(f"  {status} {name}: {init_val:.3f} → {final_val:.3f} (Δ={movement:.3f} rad)")

        # Reset for next test
        robot.write_joint_state_to_sim(init_joint_pos, torch.zeros_like(init_joint_pos))
        for _ in range(10):
            sim.step()
            robot.update(sim.cfg.dt)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (shoulder pitch): {'✅' if ee_movement > 0.05 else '❌'}")
    print(f"  Test 2 (wave motion):    {'✅' if y_movement > 0.05 else '❌'}")
    print("\n  If tests PASS → Actuators work, Jacobian calculation has bug")
    print("  If tests FAIL → Actuator/config problem")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()