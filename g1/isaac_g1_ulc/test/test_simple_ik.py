#!/usr/bin/env python3
"""
G1 Simple IK Isaac Lab Test
============================

Bu script g1_simple_ik controller'ını Isaac Lab'da test eder.
Kollar hedef pozisyonlara doğru hareket eder.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_simple_ik.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Simple IK Test")
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

# Import our custom IK controller
import sys

sys.path.insert(0, "/home/claude")  # For running standalone test
# For actual use, copy g1_simple_ik.py to your project

print("\n" + "=" * 70)
print("  G1 SIMPLE IK TEST")
print("  Windows-compatible IK controller")
print("=" * 70 + "\n")


# ============================================================
# Inline G1 Simple IK (for standalone test)
# ============================================================

class G1SimpleIK:
    """Simple Jacobian-based IK for G1 arms (inline version for testing)"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.damping = 0.05

        # Empirical Jacobian from our tests
        # Sign: what happens when joint value INCREASES
        self.right_jacobian = torch.tensor([
            [-0.31, -0.09, -0.31],  # shoulder_pitch
            [0.00, 0.30, 0.01],  # shoulder_roll
            [0.01, 0.35, 0.00],  # shoulder_yaw
            [-0.10, 0.00, -0.21],  # elbow_pitch
            [0.00, 0.00, 0.00],  # elbow_roll
        ], device=device, dtype=torch.float32)

        self.left_jacobian = torch.tensor([
            [-0.31, 0.09, -0.31],  # shoulder_pitch
            [0.00, -0.30, 0.01],  # shoulder_roll (Y mirrored)
            [0.01, -0.35, 0.00],  # shoulder_yaw (Y mirrored)
            [-0.10, 0.00, -0.21],  # elbow_pitch
            [0.00, 0.00, 0.00],  # elbow_roll
        ], device=device, dtype=torch.float32)

        # Joint limits
        self.limits_low = torch.tensor([-2.97, -2.25, -2.62, -0.23, -2.09], device=device)
        self.limits_high = torch.tensor([2.79, 1.59, 2.62, 3.42, 2.09], device=device)

    def compute(self, current_joints, target_ee, current_ee, arm="right", dt=0.02):
        """Compute joint deltas to reach target"""
        if current_joints.dim() == 1:
            current_joints = current_joints.unsqueeze(0)
            target_ee = target_ee.unsqueeze(0)
            current_ee = current_ee.unsqueeze(0)

        batch_size = current_joints.shape[0]
        jacobian = self.right_jacobian if arm == "right" else self.left_jacobian
        jacobian = jacobian.unsqueeze(0).expand(batch_size, -1, -1)

        # Position error
        pos_error = target_ee - current_ee
        error_norm = torch.norm(pos_error, dim=-1, keepdim=True)
        pos_error = torch.where(error_norm > 0.5, pos_error * 0.5 / (error_norm + 1e-6), pos_error)

        # Damped pseudo-inverse
        JJT = torch.bmm(jacobian, jacobian.transpose(-2, -1))
        damping_matrix = (self.damping ** 2) * torch.eye(5, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        JJT_inv = torch.linalg.inv(JJT + damping_matrix)
        J_pinv = torch.bmm(jacobian.transpose(-2, -1), JJT_inv)

        # Joint velocity
        joint_vel = torch.bmm(J_pinv.transpose(-2, -1), pos_error.unsqueeze(-1)).squeeze(-1)

        # Joint deltas
        gain = 5.0
        joint_deltas = gain * joint_vel * dt
        joint_deltas = torch.clamp(joint_deltas, -0.5, 0.5)

        # Apply limits
        new_joints = current_joints + joint_deltas
        new_joints = torch.clamp(new_joints, self.limits_low, self.limits_high)
        joint_deltas = new_joints - current_joints

        return joint_deltas.squeeze(0)


# ============================================================
# Main Test
# ============================================================

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

    # Create IK controller
    ik = G1SimpleIK(device="cuda:0")

    # Joint indices
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    # Find arm joints
    right_indices = []
    left_indices = []

    for i, name in enumerate(joint_names):
        nl = name.lower()
        if "right" in nl:
            if any(x in nl for x in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]):
                right_indices.append(i)
        if "left" in nl:
            if any(x in nl for x in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]):
                left_indices.append(i)

    # Sort by joint order
    right_joint_order = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]
    left_joint_order = right_joint_order

    def sort_indices(indices, joint_names, order):
        sorted_indices = []
        for o in order:
            for idx in indices:
                if o in joint_names[idx].lower():
                    sorted_indices.append(idx)
                    break
        return sorted_indices

    right_indices = sort_indices(right_indices, joint_names, right_joint_order)
    left_indices = sort_indices(left_indices, joint_names, left_joint_order)

    right_indices = torch.tensor(right_indices, device="cuda:0")
    left_indices = torch.tensor(left_indices, device="cuda:0")

    print(f"[INFO] Right arm indices: {right_indices.tolist()}")
    print(f"[INFO] Left arm indices: {left_indices.tolist()}")

    # EE body indices
    right_ee_idx = 29
    left_ee_idx = 28
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            right_ee_idx = i
        if "left_palm" in name.lower():
            left_ee_idx = i

    print(f"[INFO] Right EE body: {right_ee_idx}")
    print(f"[INFO] Left EE body: {left_ee_idx}")

    # Get initial positions
    init_right_ee = robot.data.body_pos_w[0, right_ee_idx].clone()
    init_left_ee = robot.data.body_pos_w[0, left_ee_idx].clone()

    print(f"\n[INFO] Initial Right EE: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"[INFO] Initial Left EE: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")

    # Define target trajectory (circular motion in front of robot)
    def get_targets(t):
        # Right arm: circular motion
        radius = 0.15
        cx, cy, cz = 0.35, -0.15, 1.15  # center
        right_target = torch.tensor([
            cx + radius * math.cos(t * 0.5),
            cy,
            cz + radius * math.sin(t * 0.5),
        ], device="cuda:0")

        # Left arm: mirrored
        left_target = torch.tensor([
            cx + radius * math.cos(t * 0.5 + math.pi),
            -cy,
            cz + radius * math.sin(t * 0.5 + math.pi),
        ], device="cuda:0")

        return right_target, left_target

    # Run simulation
    print("\n[INFO] Starting IK tracking test...")
    print("[INFO] Arms will track circular targets")
    print("[INFO] Press Ctrl+C to stop\n")

    sim_time = 0.0
    tracking_errors = []

    for step in range(3000):  # 30 seconds
        # Get current state
        right_ee = robot.data.body_pos_w[0, right_ee_idx]
        left_ee = robot.data.body_pos_w[0, left_ee_idx]

        right_joints = robot.data.joint_pos[0, right_indices]
        left_joints = robot.data.joint_pos[0, left_indices]

        # Get targets
        right_target, left_target = get_targets(sim_time)

        # Compute IK
        right_deltas = ik.compute(right_joints, right_target, right_ee, "right")
        left_deltas = ik.compute(left_joints, left_target, left_ee, "left")

        # Apply joint targets
        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] += right_deltas
        joint_targets[0, left_indices] += left_deltas

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        sim_time += sim.cfg.dt

        # Calculate tracking error
        right_error = torch.norm(right_ee - right_target).item()
        left_error = torch.norm(left_ee - left_target).item()

        # Log every second
        if step % 100 == 0:
            print(f"[{sim_time:5.1f}s] Right error: {right_error:.4f}m, Left error: {left_error:.4f}m")
            tracking_errors.append((right_error, left_error))

    # Summary
    print("\n" + "=" * 70)
    print("  IK TRACKING TEST COMPLETE")
    print("=" * 70)

    avg_right = sum(e[0] for e in tracking_errors) / len(tracking_errors)
    avg_left = sum(e[1] for e in tracking_errors) / len(tracking_errors)

    print(f"\n  Average tracking error:")
    print(f"    Right arm: {avg_right:.4f}m ({avg_right * 100:.2f}cm)")
    print(f"    Left arm:  {avg_left:.4f}m ({avg_left * 100:.2f}cm)")

    if avg_right < 0.05 and avg_left < 0.05:
        print("\n  🎉 EXCELLENT: IK tracking working well!")
    elif avg_right < 0.10 and avg_left < 0.10:
        print("\n  ✅ GOOD: IK tracking acceptable")
    else:
        print("\n  ⚠️ NEEDS TUNING: Large tracking error")

    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()