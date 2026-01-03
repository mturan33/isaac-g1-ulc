# Copyright (c) 2025, VLM-RL G1 Project
# G1 DiffIK Test - V21
# MINIMAL FIXED BASE - Just test if DiffIK works at all

"""
G1 DiffIK Test V21 - Minimal Fixed Base

This is the SIMPLEST possible test:
- Spawn G1 with fixed base
- Move one arm with DiffIK
- No locomotion, no complexity
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 DiffIK V21 - Minimal Fixed Base")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg  # Correct import!
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

print("\n" + "=" * 70)
print("  G1 DiffIK Test - V21 MINIMAL")
print("  Fixed Base + Simple DiffIK")
print("=" * 70 + "\n")


def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    # ============================================================
    # Spawn G1 with FIXED BASE
    # ============================================================

    # G1 robot config - FIXED BASE!
    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,  # ★★★ THIS IS THE KEY! ★★★
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),  # Higher since fixed
            joint_pos={
                # Legs - default standing pose
                ".*_hip_pitch_joint": -0.1,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                ".*_ankle_roll_joint": 0.0,
                # Arms - default pose
                ".*_shoulder_pitch_joint": 0.0,
                ".*_shoulder_roll_joint": 0.3,
                ".*_shoulder_yaw_joint": 0.0,
                ".*_elbow_joint": 0.5,
                ".*_wrist_roll_joint": 0.0,
                ".*_wrist_pitch_joint": 0.0,
                ".*_wrist_yaw_joint": 0.0,
                # Waist
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                stiffness=100.0,
                damping=5.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
                stiffness=40.0,
                damping=10.0,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=robot_cfg)

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Play simulation
    sim.reset()

    print(f"[INFO] Robot spawned with fix_root_link=True")
    print(f"[INFO] Joint names: {robot.data.joint_names}")

    # ============================================================
    # Setup DiffIK Controller for RIGHT arm
    # ============================================================

    # Find right arm joints
    right_arm_joints = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    right_arm_indices = []
    for jname in right_arm_joints:
        if jname in robot.data.joint_names:
            idx = robot.data.joint_names.index(jname)
            right_arm_indices.append(idx)
            print(f"  Joint '{jname}' -> index {idx}")

    print(f"\n[INFO] Right arm joint indices: {right_arm_indices}")

    # Find end-effector
    ee_name = "right_wrist_yaw_link"
    ee_idx = robot.data.body_names.index(ee_name)
    print(f"[INFO] End-effector '{ee_name}' -> body index {ee_idx}")

    # DiffIK Controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)

    # Get initial EE pose
    robot.update(sim.cfg.dt)
    init_ee_pos = robot.data.body_pos_w[:, ee_idx].clone()
    init_ee_quat = robot.data.body_quat_w[:, ee_idx].clone()
    init_root_pos = robot.data.root_pos_w.clone()

    print(f"\n[INFO] Initial EE position: {init_ee_pos[0].tolist()}")
    print(f"[INFO] Initial Root position: {init_root_pos[0].tolist()}")

    # Target: Move arm forward (Y+) by 0.15m
    target_pos = init_ee_pos.clone()
    target_pos[:, 1] += 0.15
    target_quat = init_ee_quat.clone()

    print(f"[INFO] Target EE position: {target_pos[0].tolist()}")

    # ============================================================
    # Control Loop
    # ============================================================

    print("\n" + "=" * 60)
    print("  Control Loop - Fixed Base DiffIK")
    print("=" * 60 + "\n")

    best_error = float('inf')
    best_step = 0

    for step in range(300):
        # Get current state
        robot.update(sim.cfg.dt)

        current_ee_pos = robot.data.body_pos_w[:, ee_idx]
        current_ee_quat = robot.data.body_quat_w[:, ee_idx]
        current_root_pos = robot.data.root_pos_w

        # Compute EE pose in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            current_ee_pos,
            current_ee_quat
        )

        # Target in root frame
        target_pos_b, target_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            target_pos,
            target_quat
        )

        # Reset IK controller
        diff_ik.reset()
        diff_ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        # Get Jacobian
        jacobian = robot.root_physx_view.get_jacobians()

        # Extract right arm jacobian
        # Jacobian shape: (num_envs, num_bodies, 6, num_dofs)
        ee_jacobi = jacobian[:, ee_idx, :, :]  # (1, 6, num_dofs)

        # Get only right arm columns
        right_arm_jacobi = ee_jacobi[:, :, right_arm_indices]  # (1, 6, 7)

        # Current joint positions for right arm
        right_arm_pos = robot.data.joint_pos[:, right_arm_indices]

        # Compute IK
        joint_pos_des = diff_ik.compute(ee_pos_b, ee_quat_b, right_arm_jacobi, right_arm_pos)

        # Apply to robot (only right arm)
        current_joint_pos = robot.data.joint_pos.clone()
        current_joint_pos[:, right_arm_indices] = joint_pos_des

        robot.set_joint_position_target(current_joint_pos)
        robot.write_data_to_sim()

        # Step simulation
        sim.step()

        # Metrics
        ee_error = torch.norm(current_ee_pos - target_pos, dim=-1).item()
        root_drift = torch.norm(current_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
        movement = current_ee_pos - init_ee_pos

        if ee_error < best_error:
            best_error = ee_error
            best_step = step

        # Log every 30 steps
        if step % 30 == 0:
            fixed_status = "✅ FIXED" if root_drift < 0.01 else f"❌ DRIFT {root_drift:.3f}m"
            success_mark = " <-- SUCCESS!" if ee_error < 0.02 else ""

            print(f"[{step:3d}] EE Err: {ee_error:.4f}m | Y move: {movement[0, 1]:.3f}m | "
                  f"Root: {fixed_status}{success_mark}")

        # Success check
        if ee_error < 0.02:
            print(f"\n*** SUCCESS at step {step}! EE Error: {ee_error:.4f}m ***")
            # Continue for a bit to verify stability
            if step > 50:
                break

    # Final summary
    robot.update(sim.cfg.dt)
    final_ee_pos = robot.data.body_pos_w[:, ee_idx]
    final_root_pos = robot.data.root_pos_w
    final_ee_error = torch.norm(final_ee_pos - target_pos, dim=-1).item()
    final_root_drift = torch.norm(final_root_pos[:, :2] - init_root_pos[:, :2], dim=-1).item()
    final_movement = final_ee_pos - init_ee_pos

    print("\n" + "=" * 60)
    print("  SUMMARY - MINIMAL FIXED BASE TEST")
    print("=" * 60)
    print(f"\n  DiffIK ARM CONTROL:")
    print(f"    Best EE error: {best_error:.4f}m at step {best_step}")
    print(f"    Final EE error: {final_ee_error:.4f}m")
    print(f"    Y movement: {final_movement[0, 1]:.4f}m (target: 0.15m)")

    print(f"\n  FIXED BASE STATUS:")
    print(f"    Root drift: {final_root_drift:.6f}m")
    print(f"    Status: {'✅ FIXED!' if final_root_drift < 0.01 else '❌ DRIFTED'}")

    if best_error < 0.02 and final_root_drift < 0.01:
        print(f"\n  🎉 COMPLETE SUCCESS! DiffIK works with Fixed Base!")
    elif final_root_drift < 0.01:
        print(f"\n  ✅ Fixed Base works! DiffIK needs tuning.")
    else:
        print(f"\n  ❌ Something is wrong with Fixed Base config.")


if __name__ == "__main__":
    main()
    simulation_app.close()