# Copyright (c) 2025, VLM-RL G1 Project
# G1 DiffIK Test - V26
# Pure Standalone - No gym wrapper needed

"""
G1 DiffIK Test V26 - Pure Standalone

Gym environment'lar Pink IK dependency yüzünden kayıtlı değil.
Bu script doğrudan isaaclab_assets'tan G1 robot config kullanıyor.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 DiffIK V26 - Pure Standalone")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

print("\n" + "=" * 70)
print("  G1 DiffIK Test - V26")
print("  Pure Standalone - No gym wrapper")
print("=" * 70 + "\n")


def main():
    # ============================================================
    # 1. Create Simulation Context
    # ============================================================
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])

    # ============================================================
    # 2. Try to import G1 config from isaaclab_assets
    # ============================================================
    print("[INFO] Importing G1 robot configuration...")

    # Try different import paths
    robot_cfg = None
    import_path = None

    try:
        from isaaclab_assets import G1_MINIMAL_CFG
        robot_cfg = G1_MINIMAL_CFG.copy()
        import_path = "isaaclab_assets.G1_MINIMAL_CFG"
        print(f"[INFO] ✅ Found: {import_path}")
    except ImportError:
        pass

    if robot_cfg is None:
        try:
            from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
            robot_cfg = G1_MINIMAL_CFG.copy()
            import_path = "isaaclab_assets.robots.unitree.G1_MINIMAL_CFG"
            print(f"[INFO] ✅ Found: {import_path}")
        except ImportError:
            pass

    if robot_cfg is None:
        try:
            from isaaclab_assets import G1_CFG
            robot_cfg = G1_CFG.copy()
            import_path = "isaaclab_assets.G1_CFG"
            print(f"[INFO] ✅ Found: {import_path}")
        except ImportError:
            pass

    if robot_cfg is None:
        try:
            from isaaclab_assets.robots.unitree import G1_CFG
            robot_cfg = G1_CFG.copy()
            import_path = "isaaclab_assets.robots.unitree.G1_CFG"
            print(f"[INFO] ✅ Found: {import_path}")
        except ImportError:
            pass

    if robot_cfg is None:
        # Last resort - list what's available
        print("[ERROR] Could not find G1 config. Listing available assets...")
        import isaaclab_assets
        print(f"[DEBUG] isaaclab_assets contents: {dir(isaaclab_assets)}")

        # Try to find anything with G1
        g1_items = [x for x in dir(isaaclab_assets) if 'G1' in x or 'g1' in x]
        print(f"[DEBUG] G1-related items: {g1_items}")

        if g1_items:
            first_item = g1_items[0]
            robot_cfg = getattr(isaaclab_assets, first_item).copy()
            import_path = f"isaaclab_assets.{first_item}"
            print(f"[INFO] Using: {import_path}")
        else:
            print("[ERROR] No G1 configuration found!")
            return

    # ============================================================
    # 3. Configure robot for FIXED BASE
    # ============================================================
    print(f"\n[INFO] Configuring robot with fixed base...")

    robot_cfg.prim_path = "/World/Robot"

    # Set initial position (raised for fixed base)
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)

    # Try to set fixed base
    if hasattr(robot_cfg.spawn, 'articulation_props'):
        robot_cfg.spawn.articulation_props.fix_root_link = True
        print("[INFO] Set fix_root_link = True via articulation_props")
    else:
        print("[WARN] articulation_props not found, trying alternative method")

    print(f"[INFO] USD path: {robot_cfg.spawn.usd_path}")

    # ============================================================
    # 4. Create scene elements
    # ============================================================
    print("[INFO] Creating scene...")

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Create robot
    print("[INFO] Creating G1 robot...")
    robot = Articulation(cfg=robot_cfg)

    # ============================================================
    # 5. Initialize simulation
    # ============================================================
    print("[INFO] Resetting simulation...")
    sim.reset()

    # Initial update
    robot.update(sim.cfg.dt)

    print(f"\n[INFO] ✅ Robot created successfully!")
    print(f"[INFO] Number of joints: {robot.num_joints}")
    print(f"[INFO] Number of bodies: {robot.num_bodies}")

    # Print joint info
    print(f"\n[INFO] Joint names ({len(robot.data.joint_names)}):")
    for i, name in enumerate(robot.data.joint_names[:10]):
        print(f"  [{i:2d}] {name}")
    if len(robot.data.joint_names) > 10:
        print(f"  ... and {len(robot.data.joint_names) - 10} more")

    # Print body info
    print(f"\n[INFO] Body names ({len(robot.data.body_names)}):")
    for i, name in enumerate(robot.data.body_names[:10]):
        print(f"  [{i:2d}] {name}")
    if len(robot.data.body_names) > 10:
        print(f"  ... and {len(robot.data.body_names) - 10} more")

    # ============================================================
    # 6. Find right arm joints
    # ============================================================
    print("\n[INFO] Finding right arm joints...")

    right_arm_joint_names = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    right_arm_indices = []
    for jname in right_arm_joint_names:
        if jname in robot.data.joint_names:
            idx = robot.data.joint_names.index(jname)
            right_arm_indices.append(idx)
            print(f"  ✅ '{jname}' -> index {idx}")
        else:
            print(f"  ❌ '{jname}' not found")

    if len(right_arm_indices) < 4:
        print("[ERROR] Not enough arm joints found!")
        print("[INFO] Available joints:")
        for i, name in enumerate(robot.data.joint_names):
            print(f"  [{i:2d}] {name}")
        return

    # ============================================================
    # 7. Find end-effector body
    # ============================================================
    print("\n[INFO] Finding end-effector body...")

    ee_candidates = [
        "right_wrist_yaw_link",
        "right_hand_link",
        "right_wrist_link",
        "right_palm_link",
    ]

    ee_idx = None
    ee_name = None
    for candidate in ee_candidates:
        if candidate in robot.data.body_names:
            ee_idx = robot.data.body_names.index(candidate)
            ee_name = candidate
            print(f"  ✅ Found EE: '{ee_name}' -> body index {ee_idx}")
            break

    if ee_idx is None:
        print("[ERROR] No end-effector body found!")
        print("[INFO] Available bodies:")
        for i, name in enumerate(robot.data.body_names):
            print(f"  [{i:2d}] {name}")
        return

    # ============================================================
    # 8. Get initial state
    # ============================================================
    robot.update(sim.cfg.dt)

    init_root_pos = robot.data.root_pos_w.clone()
    init_root_quat = robot.data.root_quat_w.clone()
    init_ee_pos = robot.data.body_pos_w[:, ee_idx].clone()
    init_ee_quat = robot.data.body_quat_w[:, ee_idx].clone()
    init_joint_pos = robot.data.joint_pos.clone()

    print(f"\n[INFO] Initial state:")
    print(f"  Root pos: {init_root_pos[0].tolist()}")
    print(f"  EE pos:   {init_ee_pos[0].tolist()}")
    print(f"  Right arm joints: {init_joint_pos[0, right_arm_indices].tolist()}")

    # ============================================================
    # 9. Set target position (move arm forward)
    # ============================================================
    target_offset = torch.tensor([0.0, 0.15, 0.0], device=sim.device)  # Y+ 15cm
    target_pos = init_ee_pos + target_offset
    target_quat = init_ee_quat.clone()

    print(f"\n[INFO] Target:")
    print(f"  EE target pos: {target_pos[0].tolist()}")
    print(f"  Movement: Y +0.15m")

    # ============================================================
    # 10. Simple IK Control Loop (Jacobian Pseudoinverse)
    # ============================================================
    print("\n" + "=" * 60)
    print("  Control Loop - Jacobian Pseudoinverse IK")
    print("=" * 60 + "\n")

    # IK parameters
    damping = 0.05
    max_delta = 0.1  # Max joint change per step

    best_error = float('inf')
    best_step = 0

    for step in range(300):
        # Update robot state
        robot.update(sim.cfg.dt)

        # Get current poses
        current_root_pos = robot.data.root_pos_w
        current_root_quat = robot.data.root_quat_w
        current_ee_pos = robot.data.body_pos_w[:, ee_idx]
        current_ee_quat = robot.data.body_quat_w[:, ee_idx]
        current_joint_pos = robot.data.joint_pos

        # Compute position error (world frame)
        pos_error = target_pos - current_ee_pos  # (1, 3)

        # Get Jacobian
        jacobian_full = robot.root_physx_view.get_jacobians()  # (num_envs, num_bodies, 6, num_dofs)

        # Extract EE jacobian for right arm joints (position rows only)
        # Jacobian shape: (1, num_bodies, 6, num_dofs)
        # We want: (1, 3, 7) for position control of 7 arm joints
        ee_jacobian = jacobian_full[:, ee_idx, :3, :]  # (1, 3, num_dofs) - position rows
        arm_jacobian = ee_jacobian[:, :, right_arm_indices]  # (1, 3, 7)

        # Damped Least Squares IK
        # J^T (J J^T + λ²I)^{-1} e
        JJT = torch.bmm(arm_jacobian, arm_jacobian.transpose(1, 2))  # (1, 3, 3)
        JJT_damped = JJT + damping ** 2 * torch.eye(3, device=sim.device).unsqueeze(0)

        # Solve for delta_q
        JJT_inv = torch.linalg.inv(JJT_damped)  # (1, 3, 3)
        delta_x = pos_error  # (1, 3)
        delta_q = torch.bmm(arm_jacobian.transpose(1, 2), torch.bmm(JJT_inv, delta_x.unsqueeze(-1)))  # (1, 7, 1)
        delta_q = delta_q.squeeze(-1)  # (1, 7)

        # Clamp delta
        delta_q = torch.clamp(delta_q, -max_delta, max_delta)

        # Apply to right arm joints
        new_joint_pos = current_joint_pos.clone()
        new_joint_pos[:, right_arm_indices] += delta_q

        # Set joint targets
        robot.set_joint_position_target(new_joint_pos)
        robot.write_data_to_sim()

        # Step simulation
        sim.step()

        # Compute metrics
        ee_error = torch.norm(pos_error, dim=-1).item()
        root_drift = torch.norm(current_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
        y_movement = (current_ee_pos[0, 1] - init_ee_pos[0, 1]).item()

        if ee_error < best_error:
            best_error = ee_error
            best_step = step

        # Log every 30 steps
        if step % 30 == 0:
            fixed_status = "✅ FIXED" if root_drift < 0.01 else f"⚠️ drift {root_drift:.4f}m"
            success = " ← SUCCESS!" if ee_error < 0.02 else ""

            print(f"[{step:3d}] EE err: {ee_error:.4f}m | Y move: {y_movement:+.3f}m | "
                  f"Root: {fixed_status}{success}")

        # Early success
        if ee_error < 0.02 and step > 50:
            print(f"\n*** SUCCESS at step {step}! ***")
            break

    # ============================================================
    # 11. Final Summary
    # ============================================================
    robot.update(sim.cfg.dt)
    final_ee_pos = robot.data.body_pos_w[:, ee_idx]
    final_root_pos = robot.data.root_pos_w
    final_ee_error = torch.norm(target_pos - final_ee_pos, dim=-1).item()
    final_root_drift = torch.norm(final_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
    final_y_movement = (final_ee_pos[0, 1] - init_ee_pos[0, 1]).item()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(f"\n  🤖 DiffIK ARM CONTROL:")
    print(f"     Best EE error: {best_error:.4f}m at step {best_step}")
    print(f"     Final EE error: {final_ee_error:.4f}m")
    print(f"     Y movement: {final_y_movement:+.4f}m (target: +0.15m)")
    print(f"     Accuracy: {(final_y_movement / 0.15) * 100:.1f}%")

    print(f"\n  🔒 FIXED BASE STATUS:")
    print(f"     Root drift: {final_root_drift:.6f}m")
    if final_root_drift < 0.01:
        print(f"     Status: ✅ TRULY FIXED!")
    else:
        print(f"     Status: ⚠️ Some drift detected")

    if best_error < 0.03 and final_root_drift < 0.05:
        print(f"\n  🎉 TEST PASSED!")
        print(f"     DiffIK working on G1 robot!")
    else:
        print(f"\n  ⚠️ Test needs tuning")
        print(f"     Check IK parameters or joint limits")


if __name__ == "__main__":
    main()
    simulation_app.close()