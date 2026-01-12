#!/usr/bin/env python3
"""
G1 IK with IKPY Library
=======================

IKPY kullanarak G1 kolları için doğru IK hesaplama.
Pure Python - Windows'ta çalışır.

KURULUM:
pip install ikpy --break-system-packages

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_ikpy.py --num_envs 1
"""

import argparse
import math
import numpy as np

parser = argparse.ArgumentParser(description="G1 IKPY Test")
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

# Try to import ikpy
try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink

    IKPY_AVAILABLE = True
    print("\n✅ IKPY library loaded successfully!")
except ImportError:
    IKPY_AVAILABLE = False
    print("\n⚠️ IKPY not installed. Run: pip install ikpy --break-system-packages")

print("\n" + "=" * 70)
print("  G1 IK TEST WITH IKPY")
print("  Pure Python IK library")
print("=" * 70 + "\n")


def create_g1_right_arm_chain():
    """
    Create G1 right arm kinematic chain using IKPY.
    Based on our joint explorer test results.

    G1 Right Arm Structure:
    - torso -> shoulder_pitch -> shoulder_roll -> shoulder_yaw
            -> elbow_pitch -> elbow_roll -> palm

    Link lengths (approximate, in meters):
    - Shoulder to elbow: ~0.25m
    - Elbow to palm: ~0.22m
    """

    right_arm_chain = Chain(name='g1_right_arm', links=[
        # Origin (torso/shoulder mount point)
        OriginLink(),

        # Shoulder Pitch - rotates around Y axis (pitch = nodding motion)
        # Moves arm forward/backward and up/down
        URDFLink(
            name="right_shoulder_pitch",
            origin_translation=[0, -0.15, 0],  # Shoulder offset from torso
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Pitch axis
            bounds=(-2.97, 2.79),
        ),

        # Shoulder Roll - rotates around X axis (roll = tilting motion)
        # Moves arm to side
        URDFLink(
            name="right_shoulder_roll",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],  # Roll axis
            bounds=(-2.25, 1.59),
        ),

        # Shoulder Yaw - rotates around Z axis (yaw = twisting motion)
        URDFLink(
            name="right_shoulder_yaw",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],  # Yaw axis
            bounds=(-2.62, 2.62),
        ),

        # Upper arm length
        URDFLink(
            name="right_upper_arm",
            origin_translation=[0.25, 0, 0],  # Upper arm length
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],  # Fixed link
        ),

        # Elbow Pitch
        URDFLink(
            name="right_elbow_pitch",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Pitch axis
            bounds=(-0.23, 3.42),
        ),

        # Elbow Roll
        URDFLink(
            name="right_elbow_roll",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],  # Roll axis
            bounds=(-2.09, 2.09),
        ),

        # Forearm to palm (end effector)
        URDFLink(
            name="right_palm",
            origin_translation=[0.22, 0, 0],  # Forearm length
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],  # Fixed link (end effector)
        ),
    ])

    # Active links mask: Origin and fixed links are inactive
    # [Origin, sh_pitch, sh_roll, sh_yaw, upper_arm, el_pitch, el_roll, palm]
    active_links_mask = [False, True, True, True, False, True, True, False]

    return right_arm_chain, active_links_mask


def create_g1_left_arm_chain():
    """Create G1 left arm kinematic chain (mirrored)"""

    left_arm_chain = Chain(name='g1_left_arm', links=[
        OriginLink(),

        URDFLink(
            name="left_shoulder_pitch",
            origin_translation=[0, 0.15, 0],  # Mirrored Y
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-2.97, 2.79),
        ),

        URDFLink(
            name="left_shoulder_roll",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],
            bounds=(-1.59, 2.25),  # Mirrored limits
        ),

        URDFLink(
            name="left_shoulder_yaw",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-2.62, 2.62),
        ),

        URDFLink(
            name="left_upper_arm",
            origin_translation=[0.25, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
        ),

        URDFLink(
            name="left_elbow_pitch",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-0.23, 3.42),
        ),

        URDFLink(
            name="left_elbow_roll",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],
            bounds=(-2.09, 2.09),
        ),

        URDFLink(
            name="left_palm",
            origin_translation=[0.22, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
        ),
    ])

    active_links_mask = [False, True, True, True, False, True, True, False]

    return left_arm_chain, active_links_mask


class G1IKPYController:
    """G1 IK Controller using IKPY library"""

    def __init__(self, device="cuda:0"):
        self.device = device

        # Create chains
        self.right_chain, self.right_mask = create_g1_right_arm_chain()
        self.left_chain, self.left_mask = create_g1_left_arm_chain()

        # Base position (shoulder mount in world frame)
        # This needs to be set based on robot's position
        self.right_base_pos = np.array([0.0, -0.15, 1.20])  # Approximate
        self.left_base_pos = np.array([0.0, 0.15, 1.20])

    def compute(self, target_world_pos, arm="right", initial_joints=None):
        """
        Compute IK for target position in world frame.

        Args:
            target_world_pos: Target position [x, y, z] in world frame
            arm: "right" or "left"
            initial_joints: Initial joint positions for warm start

        Returns:
            joint_angles: Array of 5 joint angles [sh_pitch, sh_roll, sh_yaw, el_pitch, el_roll]
        """
        if isinstance(target_world_pos, torch.Tensor):
            target_world_pos = target_world_pos.cpu().numpy()

        if arm == "right":
            chain = self.right_chain
            base_pos = self.right_base_pos
        else:
            chain = self.left_chain
            base_pos = self.left_base_pos

        # Convert target from world frame to chain frame
        target_local = target_world_pos - base_pos

        # Prepare initial position (full chain including fixed links)
        if initial_joints is not None:
            if isinstance(initial_joints, torch.Tensor):
                initial_joints = initial_joints.cpu().numpy()
            # Expand to full chain: [origin, sh_pitch, sh_roll, sh_yaw, upper_arm, el_pitch, el_roll, palm]
            init_full = np.zeros(8)
            init_full[1] = initial_joints[0]  # shoulder_pitch
            init_full[2] = initial_joints[1]  # shoulder_roll
            init_full[3] = initial_joints[2]  # shoulder_yaw
            init_full[5] = initial_joints[3]  # elbow_pitch
            init_full[6] = initial_joints[4]  # elbow_roll
        else:
            init_full = None

        # Compute IK
        ik_result = chain.inverse_kinematics(
            target_position=target_local,
            initial_position=init_full,
        )

        # Extract active joint angles
        # [origin, sh_pitch, sh_roll, sh_yaw, upper_arm, el_pitch, el_roll, palm]
        joint_angles = np.array([
            ik_result[1],  # shoulder_pitch
            ik_result[2],  # shoulder_roll
            ik_result[3],  # shoulder_yaw
            ik_result[5],  # elbow_pitch
            ik_result[6],  # elbow_roll
        ])

        return joint_angles

    def forward_kinematics(self, joint_angles, arm="right"):
        """Compute FK to get end effector position"""
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()

        if arm == "right":
            chain = self.right_chain
            base_pos = self.right_base_pos
        else:
            chain = self.left_chain
            base_pos = self.left_base_pos

        # Expand to full chain
        full_joints = np.zeros(8)
        full_joints[1] = joint_angles[0]
        full_joints[2] = joint_angles[1]
        full_joints[3] = joint_angles[2]
        full_joints[5] = joint_angles[3]
        full_joints[6] = joint_angles[4]

        # Compute FK
        transform = chain.forward_kinematics(full_joints)
        local_pos = transform[:3, 3]
        world_pos = local_pos + base_pos

        return world_pos

    def update_base_position(self, right_shoulder_pos, left_shoulder_pos):
        """Update base positions from simulation"""
        if isinstance(right_shoulder_pos, torch.Tensor):
            right_shoulder_pos = right_shoulder_pos.cpu().numpy()
        if isinstance(left_shoulder_pos, torch.Tensor):
            left_shoulder_pos = left_shoulder_pos.cpu().numpy()

        self.right_base_pos = right_shoulder_pos
        self.left_base_pos = left_shoulder_pos


def main():
    if not IKPY_AVAILABLE:
        print("\n❌ Cannot run test without IKPY. Install it first:")
        print("   pip install ikpy --break-system-packages")
        return

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

    left_indices = []
    for order_name in right_joint_order:
        for i, name in enumerate(joint_names):
            if "left" in name.lower() and order_name in name.lower():
                left_indices.append(i)
                break

    right_indices = torch.tensor(right_indices, device="cuda:0")
    left_indices = torch.tensor(left_indices, device="cuda:0")

    # EE body indices
    right_ee_idx = 29
    left_ee_idx = 28
    for i, name in enumerate(body_names):
        if "right_palm" in name.lower():
            right_ee_idx = i
        if "left_palm" in name.lower():
            left_ee_idx = i

    print(f"[INFO] Right arm indices: {right_indices.tolist()}")
    print(f"[INFO] Left arm indices: {left_indices.tolist()}")

    # Get initial positions
    init_right_ee = robot.data.body_pos_w[0, right_ee_idx].clone()
    init_left_ee = robot.data.body_pos_w[0, left_ee_idx].clone()

    print(f"\n[INFO] Initial Right EE: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"[INFO] Initial Left EE: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")

    # Create IK controller
    ik = G1IKPYController(device="cuda:0")

    # Update base positions from actual robot
    # Find shoulder body positions
    shoulder_offset = torch.tensor([0.0, 0.0, 0.0], device="cuda:0")
    for i, name in enumerate(body_names):
        if "right_shoulder" in name.lower():
            shoulder_pos = robot.data.body_pos_w[0, i]
            ik.right_base_pos = shoulder_pos.cpu().numpy()
            print(f"[INFO] Right shoulder at: {shoulder_pos.tolist()}")
            break

    for i, name in enumerate(body_names):
        if "left_shoulder" in name.lower():
            shoulder_pos = robot.data.body_pos_w[0, i]
            ik.left_base_pos = shoulder_pos.cpu().numpy()
            print(f"[INFO] Left shoulder at: {shoulder_pos.tolist()}")
            break

    # TEST: Stationary forward reach target
    print("\n" + "=" * 60)
    print("  TEST: FORWARD REACH WITH IKPY")
    print("=" * 60)

    # Target: 15cm forward from initial EE
    right_target = init_right_ee.clone()
    right_target[0] += 0.15

    left_target = init_left_ee.clone()
    left_target[0] += 0.15

    print(f"\n[INFO] Right target: ({right_target[0]:.3f}, {right_target[1]:.3f}, {right_target[2]:.3f})")
    print(f"[INFO] Left target: ({left_target[0]:.3f}, {left_target[1]:.3f}, {left_target[2]:.3f})")

    print("\n[INFO] Computing IK solution...")

    # Get current joint positions
    right_joints = robot.data.joint_pos[0, right_indices].cpu().numpy()
    left_joints = robot.data.joint_pos[0, left_indices].cpu().numpy()

    # Compute IK
    right_ik_joints = ik.compute(right_target, "right", right_joints)
    left_ik_joints = ik.compute(left_target, "left", left_joints)

    print(f"\n[INFO] IK Solution (right): {np.rad2deg(right_ik_joints).round(1)} deg")
    print(f"[INFO] IK Solution (left):  {np.rad2deg(left_ik_joints).round(1)} deg")

    # Apply IK solution gradually
    print("\n[INFO] Applying IK solution...")

    for step in range(300):
        alpha = min(1.0, step / 200.0)
        alpha = 0.5 * (1 - math.cos(math.pi * alpha))  # Smooth

        # Interpolate
        target_right = right_joints * (1 - alpha) + right_ik_joints * alpha
        target_left = left_joints * (1 - alpha) + left_ik_joints * alpha

        # Apply
        joint_targets = robot.data.joint_pos.clone()
        for i, idx in enumerate(right_indices):
            joint_targets[0, idx] = target_right[i]
        for i, idx in enumerate(left_indices):
            joint_targets[0, idx] = target_left[i]

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        if step % 100 == 0:
            right_ee = robot.data.body_pos_w[0, right_ee_idx]
            left_ee = robot.data.body_pos_w[0, left_ee_idx]
            right_error = torch.norm(right_ee - right_target).item()
            left_error = torch.norm(left_ee - left_target).item()
            print(f"[{step / 100:.0f}s] Right err: {right_error:.3f}m, Left err: {left_error:.3f}m")

    # Final measurement
    final_right_ee = robot.data.body_pos_w[0, right_ee_idx]
    final_left_ee = robot.data.body_pos_w[0, left_ee_idx]

    right_forward = (final_right_ee[0] - init_right_ee[0]).item()
    left_forward = (final_left_ee[0] - init_left_ee[0]).item()
    right_error = torch.norm(final_right_ee - right_target).item()
    left_error = torch.norm(final_left_ee - left_target).item()

    print(f"\n{'=' * 60}")
    print(f"  IKPY RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Right arm:")
    print(f"    Forward movement: {right_forward:+.3f}m (target: +0.150m)")
    print(f"    Final error: {right_error:.3f}m ({right_error * 100:.1f}cm)")
    print(f"\n  Left arm:")
    print(f"    Forward movement: {left_forward:+.3f}m (target: +0.150m)")
    print(f"    Final error: {left_error:.3f}m ({left_error * 100:.1f}cm)")

    if right_error < 0.05 and left_error < 0.05:
        print(f"\n  🎉 EXCELLENT: Error < 5cm!")
    elif right_error < 0.10 and left_error < 0.10:
        print(f"\n  ✅ GOOD: Error < 10cm")
    else:
        print(f"\n  ⚠️ NEEDS TUNING: Error > 10cm")
        print(f"     Kinematic chain may need adjustment")

    # Hold for observation
    print("\n[INFO] Holding pose for observation (5 sec)...")
    for _ in range(500):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()