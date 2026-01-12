#!/usr/bin/env python3
"""
G1 Jacobian Debug Test
=======================

Jacobian hesabını debug et - indexing hatası mı var?

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/test_jacobian_debug.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Jacobian Debug")
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
print("  G1 JACOBIAN DEBUG TEST")
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

    # Find joints and bodies
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    print(f"[INFO] Joints: {len(joint_names)}, Bodies: {len(body_names)}")

    # Find arm joints
    arm_indices = []
    arm_names = []
    for i, name in enumerate(joint_names):
        if "right" in name.lower() and any(x in name.lower() for x in ["shoulder", "elbow"]):
            arm_indices.append(i)
            arm_names.append(name)

    print(f"\n[INFO] Right arm joints:")
    for i, (idx, name) in enumerate(zip(arm_indices, arm_names)):
        print(f"  [{idx}] {name}")

    # Find EE
    ee_idx = None
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            ee_idx = i
            break

    if ee_idx is None:
        ee_idx = 29

    print(f"\n[INFO] EE body index: {ee_idx} ({body_names[ee_idx]})")

    # ============================================================
    # Get Jacobian and analyze
    # ============================================================
    print("\n" + "=" * 50)
    print("  JACOBIAN ANALYSIS")
    print("=" * 50)

    # Get Jacobian
    jacobians = robot.root_physx_view.get_jacobians()
    print(f"\n[INFO] Jacobians shape: {jacobians.shape}")
    print(f"  Expected: (num_envs, num_bodies, 6, num_dofs)")
    print(f"  Got:      ({jacobians.shape[0]}, {jacobians.shape[1]}, {jacobians.shape[2]}, {jacobians.shape[3]})")

    # Full Jacobian for EE
    J_full = jacobians[0, ee_idx, :, :]  # (6, num_dofs)
    print(f"\n[INFO] Full EE Jacobian shape: {J_full.shape}")

    # Position Jacobian (first 3 rows)
    J_pos = jacobians[0, ee_idx, :3, :]  # (3, num_dofs)
    print(f"[INFO] Position Jacobian shape: {J_pos.shape}")

    # Check which columns have non-zero values
    print(f"\n[INFO] Checking Jacobian columns (non-zero = affects EE):")
    arm_idx_tensor = torch.tensor(arm_indices, device="cuda:0")

    for i in range(J_pos.shape[1]):  # For each DOF
        col_norm = J_pos[:, i].norm().item()
        is_arm = i in arm_indices
        marker = "★ ARM" if is_arm else ""
        if col_norm > 0.001 or is_arm:
            print(f"  Joint [{i:2d}] {joint_names[i]:35s}: norm={col_norm:.4f} {marker}")

    # ============================================================
    # Test: Extract arm Jacobian correctly
    # ============================================================
    print("\n" + "=" * 50)
    print("  ARM JACOBIAN EXTRACTION TEST")
    print("=" * 50)

    # Method 1: Direct indexing (what we were doing)
    print("\n[METHOD 1] Direct indexing: J[:, :, arm_indices]")
    try:
        J_arm_1 = jacobians[:, ee_idx, :3, :][:, :, arm_idx_tensor]
        print(f"  Shape: {J_arm_1.shape}")
        print(f"  Values:\n{J_arm_1[0]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Method 2: Manual extraction
    print("\n[METHOD 2] Manual extraction column by column:")
    J_arm_2 = torch.zeros(1, 3, len(arm_indices), device="cuda:0")
    for i, idx in enumerate(arm_indices):
        J_arm_2[0, :, i] = J_pos[:, idx]
    print(f"  Shape: {J_arm_2.shape}")
    print(f"  Values:\n{J_arm_2[0]}")

    # Check if they match
    if J_arm_1 is not None:
        diff = (J_arm_1 - J_arm_2).abs().max().item()
        print(f"\n[INFO] Difference between methods: {diff}")

    # ============================================================
    # Test: Verify Jacobian with finite differences
    # ============================================================
    print("\n" + "=" * 50)
    print("  FINITE DIFFERENCE VERIFICATION")
    print("=" * 50)

    # Get current EE position
    ee_pos_init = robot.data.body_pos_w[0, ee_idx].clone()
    print(f"\n[INFO] Initial EE position: ({ee_pos_init[0]:.4f}, {ee_pos_init[1]:.4f}, {ee_pos_init[2]:.4f})")

    # For each arm joint, perturb and measure EE change
    delta_q = 0.01  # Small perturbation

    print(f"\n[INFO] Testing numerical Jacobian (delta_q = {delta_q} rad):")

    numerical_J = torch.zeros(3, len(arm_indices), device="cuda:0")

    for i, (joint_idx, joint_name) in enumerate(zip(arm_indices, arm_names)):
        # Save current state
        init_joints = robot.data.joint_pos.clone()

        # Perturb joint
        new_joints = init_joints.clone()
        new_joints[0, joint_idx] += delta_q
        robot.write_joint_state_to_sim(new_joints, torch.zeros_like(new_joints))

        # Step simulation
        for _ in range(5):
            sim.step()
            robot.update(sim.cfg.dt)

        # Measure EE change
        ee_pos_new = robot.data.body_pos_w[0, ee_idx]
        delta_ee = (ee_pos_new - ee_pos_init) / delta_q
        numerical_J[:, i] = delta_ee

        # Analytical Jacobian column
        analytical_J_col = J_pos[:, joint_idx]

        # Compare
        diff = (delta_ee - analytical_J_col).norm().item()
        match = "✅" if diff < 0.1 else "❌"

        print(f"\n  {joint_name}:")
        print(f"    Numerical:   ({delta_ee[0]:.4f}, {delta_ee[1]:.4f}, {delta_ee[2]:.4f})")
        print(f"    Analytical:  ({analytical_J_col[0]:.4f}, {analytical_J_col[1]:.4f}, {analytical_J_col[2]:.4f})")
        print(f"    Difference:  {diff:.4f} {match}")

        # Reset
        robot.write_joint_state_to_sim(init_joints, torch.zeros_like(init_joints))
        for _ in range(5):
            sim.step()
            robot.update(sim.cfg.dt)

    # ============================================================
    # Test: Simple IK step
    # ============================================================
    print("\n" + "=" * 50)
    print("  SIMPLE IK TEST")
    print("=" * 50)

    # Get current state
    robot.update(sim.cfg.dt)
    ee_pos = robot.data.body_pos_w[0, ee_idx]
    target_offset = torch.tensor([0.0, 0.1, 0.0], device="cuda:0")  # Move Y+ by 10cm
    target_pos = ee_pos + target_offset

    print(f"\n[INFO] Current EE: ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
    print(f"[INFO] Target EE:  ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")

    # Get fresh Jacobian
    jacobians = robot.root_physx_view.get_jacobians()

    # Extract arm Jacobian - TRY DIFFERENT METHODS
    print("\n[INFO] Testing IK with different Jacobian extractions:")

    # Method A: Using all DOFs, then extracting
    J_all = jacobians[0, ee_idx, :3, :]  # (3, 37)
    J_arm_a = J_all[:, arm_idx_tensor]  # (3, 5)

    # Method B: Batch extraction
    J_arm_b = jacobians[:, ee_idx, :3, :][:, :, arm_idx_tensor]  # (1, 3, 5)

    print(f"  Method A shape: {J_arm_a.shape}")
    print(f"  Method B shape: {J_arm_b.shape}")

    # Compute IK with Method A
    error = target_offset.unsqueeze(0)  # (1, 3)
    J = J_arm_a.unsqueeze(0)  # (1, 3, 5)

    damping = 0.05
    JJT = torch.bmm(J, J.transpose(1, 2))  # (1, 3, 3)
    damping_eye = (damping ** 2) * torch.eye(3, device="cuda:0").unsqueeze(0)

    print(f"\n[INFO] J J^T:\n{JJT[0]}")
    print(f"\n[INFO] J J^T + λ²I:\n{(JJT + damping_eye)[0]}")

    # Check if singular
    det = torch.linalg.det(JJT + damping_eye)
    print(f"\n[INFO] Determinant: {det.item():.6f}")

    if det.abs().item() < 1e-6:
        print("  ⚠️ Matrix is near-singular!")

    # Solve
    try:
        x = torch.linalg.solve(JJT + damping_eye, error.unsqueeze(-1))  # (1, 3, 1)
        delta_q_ik = torch.bmm(J.transpose(1, 2), x).squeeze()  # (5,)

        print(f"\n[INFO] IK solution delta_q:")
        for i, (name, dq) in enumerate(zip(arm_names, delta_q_ik)):
            print(f"  {name}: {dq.item():.4f} rad")

        # Apply IK
        print("\n[INFO] Applying IK solution...")

        init_joints = robot.data.joint_pos.clone()
        new_joints = init_joints.clone()
        for i, idx in enumerate(arm_indices):
            new_joints[0, idx] += delta_q_ik[i].item() * 0.5  # Scale down

        robot.set_joint_position_target(new_joints)
        robot.write_data_to_sim()

        for _ in range(50):
            sim.step()
            robot.update(sim.cfg.dt)

        final_ee = robot.data.body_pos_w[0, ee_idx]
        print(f"\n[INFO] Final EE: ({final_ee[0]:.4f}, {final_ee[1]:.4f}, {final_ee[2]:.4f})")
        print(f"[INFO] Target:   ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")
        print(f"[INFO] Error:    {(final_ee - target_pos).norm().item():.4f}m")

        movement = (final_ee - ee_pos).norm().item()
        print(f"[INFO] Movement: {movement:.4f}m")

        if movement > 0.01:
            print("  ✅ IK produced movement!")
        else:
            print("  ❌ IK didn't produce movement")

    except Exception as e:
        print(f"[ERROR] IK failed: {e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n  Check the numerical vs analytical Jacobian comparison above.")
    print("  If they don't match, there's a coordinate frame issue.")
    print("  If they match but IK fails, check the damping/scaling.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()