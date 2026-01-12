#!/usr/bin/env python3
"""
G1 Online Jacobian IK
=====================

Simülasyondan gerçek zamanlı Jacobian ölçümü.
Tahmin yok - gerçek EE hareketini ölçer.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_online_jacobian.py --num_envs 1
"""

import argparse
import math
import numpy as np

parser = argparse.ArgumentParser(description="G1 Online Jacobian IK")
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
print("  G1 ONLINE JACOBIAN IK")
print("  Real-time Jacobian measurement from simulation")
print("=" * 70 + "\n")


class OnlineJacobianIK:
    """
    IK with Jacobian measured directly from simulation.
    No guessing - measures actual EE response to joint changes.
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        self.num_joints = 5

        # Measured Jacobian (will be updated)
        self.jacobian = None

        # IK parameters
        self.damping = 0.01
        self.gain = 5.0
        self.max_delta = 0.5

        # Joint limits
        self.limits_low = torch.tensor([-2.97, -2.25, -2.62, -0.23, -2.09], device=device)
        self.limits_high = torch.tensor([2.79, 1.59, 2.62, 3.42, 2.09], device=device)

    def set_jacobian(self, jacobian):
        """Set measured Jacobian [3, 5]: d(xyz)/d(joints)"""
        self.jacobian = jacobian.to(self.device)

    def compute(self, target_pos, current_ee, current_joints):
        """Compute joint deltas to reach target"""
        if self.jacobian is None:
            return torch.zeros(5, device=self.device)

        # Position error
        error = target_pos - current_ee
        error_norm = torch.norm(error)

        # Clip error for stability
        if error_norm > 0.2:
            error = error * 0.2 / error_norm

        # Damped least squares: J_pinv = J^T (J J^T + λ²I)^-1
        J = self.jacobian  # [3, 5]
        JJT = J @ J.T  # [3, 3]
        damped = JJT + (self.damping ** 2) * torch.eye(3, device=self.device)
        J_pinv = J.T @ torch.linalg.inv(damped)  # [5, 3]

        # Joint delta
        dq = self.gain * J_pinv @ error

        # Clip
        dq = torch.clamp(dq, -self.max_delta, self.max_delta)

        # Enforce limits
        new_joints = current_joints + dq
        new_joints = torch.clamp(new_joints, self.limits_low, self.limits_high)
        dq = new_joints - current_joints

        return dq


def measure_jacobian(robot, sim, arm_indices, ee_idx, epsilon=0.02):
    """
    Measure Jacobian by perturbing each joint and observing EE change.

    Returns: Jacobian [3, 5] where J[i,j] = d(ee_i)/d(joint_j)
    """
    device = robot.data.joint_pos.device
    num_joints = len(arm_indices)

    # Save original state
    original_joint_pos = robot.data.joint_pos.clone()
    original_joint_vel = robot.data.joint_vel.clone()

    # Get baseline EE position
    robot.write_joint_state_to_sim(original_joint_pos, original_joint_vel)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.cfg.dt)
    baseline_ee = robot.data.body_pos_w[0, ee_idx].clone()

    # Measure each joint's effect
    jacobian = torch.zeros(3, num_joints, device=device)

    for j in range(num_joints):
        joint_idx = arm_indices[j]

        # Perturb joint positively
        perturbed_pos = original_joint_pos.clone()
        perturbed_pos[0, joint_idx] += epsilon

        robot.write_joint_state_to_sim(perturbed_pos, original_joint_vel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        ee_plus = robot.data.body_pos_w[0, ee_idx].clone()

        # Perturb joint negatively
        perturbed_pos = original_joint_pos.clone()
        perturbed_pos[0, joint_idx] -= epsilon

        robot.write_joint_state_to_sim(perturbed_pos, original_joint_vel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        ee_minus = robot.data.body_pos_w[0, ee_idx].clone()

        # Central difference: d(ee)/d(joint) = (ee+ - ee-) / (2*epsilon)
        jacobian[:, j] = (ee_plus - ee_minus) / (2 * epsilon)

    # Restore original state
    robot.write_joint_state_to_sim(original_joint_pos, original_joint_vel)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.cfg.dt)

    return jacobian


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 1.0])

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

    # Find indices
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    right_joint_order = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]

    right_indices = []
    for order_name in right_joint_order:
        for i, name in enumerate(joint_names):
            if "right" in name.lower() and order_name in name.lower():
                right_indices.append(i)
                break

    right_indices_tensor = torch.tensor(right_indices, device="cuda:0")

    # EE body index
    right_ee_idx = 29
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            right_ee_idx = i
            break

    print(f"[INFO] Right arm indices: {right_indices}")
    print(f"[INFO] Right EE body: {right_ee_idx}")

    # Get initial EE
    init_ee = robot.data.body_pos_w[0, right_ee_idx].clone()
    print(f"\n[INFO] Initial EE: ({init_ee[0]:.3f}, {init_ee[1]:.3f}, {init_ee[2]:.3f})")

    # STEP 1: Measure Jacobian at current pose
    print("\n" + "=" * 60)
    print("  STEP 1: MEASURING JACOBIAN FROM SIMULATION")
    print("=" * 60)

    print("\n[INFO] Perturbing each joint and measuring EE response...")
    jacobian = measure_jacobian(robot, sim, right_indices, right_ee_idx, epsilon=0.05)

    print(f"\n[INFO] Measured Jacobian [d(XYZ)/d(joint)]:")
    joint_names_short = ["sh_pitch", "sh_roll", "sh_yaw", "el_pitch", "el_roll"]
    print(f"{'':12} {'X':>8} {'Y':>8} {'Z':>8}")
    for j, name in enumerate(joint_names_short):
        print(f"{name:12} {jacobian[0, j]:+8.4f} {jacobian[1, j]:+8.4f} {jacobian[2, j]:+8.4f}")

    # Verify: positive X means joint increase -> EE moves forward
    print(f"\n[INFO] Analysis:")
    for j, name in enumerate(joint_names_short):
        x_effect = jacobian[0, j].item()
        if abs(x_effect) > 0.05:
            direction = "FORWARD" if x_effect > 0 else "BACKWARD"
            print(f"  {name}: {direction} ({x_effect:+.3f} m/rad)")

    # STEP 2: Create IK controller with measured Jacobian
    print("\n" + "=" * 60)
    print("  STEP 2: IK WITH MEASURED JACOBIAN")
    print("=" * 60)

    ik = OnlineJacobianIK(device="cuda:0")
    ik.set_jacobian(jacobian)

    # Target: 15cm forward
    target = init_ee.clone()
    target[0] += 0.15

    print(f"\n[INFO] Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
    print("[INFO] Running IK tracking for 5 seconds...")

    errors = []

    for step in range(500):
        # Get current state
        current_ee = robot.data.body_pos_w[0, right_ee_idx]
        current_joints = robot.data.joint_pos[0, right_indices_tensor]

        # Compute IK
        dq = ik.compute(target, current_ee, current_joints)

        # Apply
        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices_tensor] += dq

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        # Log
        error = torch.norm(current_ee - target).item()
        errors.append(error)

        # Re-measure Jacobian periodically (Jacobian changes with pose)
        if step > 0 and step % 100 == 0:
            jacobian = measure_jacobian(robot, sim, right_indices, right_ee_idx)
            ik.set_jacobian(jacobian)

            forward = (current_ee[0] - init_ee[0]).item()
            print(f"[{step / 100:.0f}s] Error: {error:.3f}m, Forward: {forward:+.3f}m (Jacobian updated)")

    # Final results
    final_ee = robot.data.body_pos_w[0, right_ee_idx]
    forward = (final_ee[0] - init_ee[0]).item()
    final_error = torch.norm(final_ee - target).item()

    print(f"\n{'=' * 60}")
    print(f"  ONLINE JACOBIAN IK RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Forward movement: {forward:+.3f}m (target: +0.150m)")
    print(f"  Final error: {final_error:.3f}m ({final_error * 100:.1f}cm)")
    print(f"  Best error: {min(errors):.3f}m")

    if forward > 0.12:
        print(f"\n  🎉 EXCELLENT: >80% of target reached!")
    elif forward > 0.08:
        print(f"\n  ✅ GOOD: >50% of target reached")
    elif forward > 0:
        print(f"\n  ⚠️ PARTIAL: Some forward movement")
    else:
        print(f"\n  ❌ FAILED: Moved backward")

    # Show final joints
    final_joints = robot.data.joint_pos[0, right_indices_tensor]
    print(f"\n  Final joints (deg):")
    for j, name in enumerate(joint_names_short):
        print(f"    {name}: {np.rad2deg(final_joints[j].item()):+.1f}°")

    # STEP 3: Compare with direct joint control
    print("\n" + "=" * 60)
    print("  COMPARISON WITH DIRECT JOINT CONTROL")
    print("=" * 60)

    # Reset
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel)
    )
    for _ in range(50):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # Direct control
    for step in range(200):
        alpha = min(1.0, step / 150.0)
        alpha = 0.5 * (1 - math.cos(math.pi * alpha))

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices[0]] = -1.57 * alpha
        joint_targets[0, right_indices[3]] = 1.57 * alpha

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    direct_ee = robot.data.body_pos_w[0, right_ee_idx]
    direct_forward = (direct_ee[0] - init_ee[0]).item()

    print(f"\n  Online Jacobian IK: {forward:+.3f}m")
    print(f"  Direct joint control: {direct_forward:+.3f}m")

    ratio = forward / direct_forward if direct_forward > 0 else 0
    print(f"\n  IK achieved {ratio * 100:.0f}% of direct control performance")

    if ratio > 0.7:
        print(f"  ✅ IK is working well!")
    else:
        print(f"  ⚠️ IK needs more tuning")

    # Hold
    print("\n[INFO] Holding for observation (5 sec)...")
    for _ in range(500):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()