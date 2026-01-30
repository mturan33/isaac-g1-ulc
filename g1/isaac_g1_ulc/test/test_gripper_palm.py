"""
G1 Gripper & Palm Test Script
- Shows palm position marker (green)
- Opens and closes fingers continuously
- 1 env for visual inspection
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import math
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

# Finger joint names for right hand
RIGHT_FINGER_JOINTS = [
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
    "right_three_joint",
    "right_four_joint",
    "right_five_joint",
    "right_six_joint",
]

# Arm joints (for holding position)
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]


class TestSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, -0.5, 1.2], [0.0, 0.0, 0.8])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Create markers for palm visualization
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={
            "palm": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
            ),
            "palm_orientation": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red - forward direction
            ),
        }
    )
    markers = VisualizationMarkers(marker_cfg)

    sim.reset()

    robot = scene["robot"]
    device = robot.device

    # Find joint indices
    finger_indices = []
    for name in RIGHT_FINGER_JOINTS:
        try:
            idx = robot.joint_names.index(name)
            finger_indices.append(idx)
            print(f"Found finger joint: {name} at index {idx}")
        except ValueError:
            print(f"WARNING: Could not find joint {name}")

    arm_indices = []
    for name in RIGHT_ARM_JOINTS:
        try:
            idx = robot.joint_names.index(name)
            arm_indices.append(idx)
        except ValueError:
            print(f"WARNING: Could not find joint {name}")

    # Find palm body index
    palm_idx = robot.body_names.index("right_palm_link")
    print(f"\nPalm body index: {palm_idx}")

    # Get joint limits for fingers
    joint_limits = robot.root_physx_view.get_dof_limits()
    print(f"\nJoint limits shape: {joint_limits.shape}")

    finger_lower = []
    finger_upper = []
    for idx in finger_indices:
        lower = joint_limits[0, idx, 0].item()
        upper = joint_limits[0, idx, 1].item()
        finger_lower.append(lower)
        finger_upper.append(upper)
        print(f"  {robot.joint_names[idx]}: [{lower:.3f}, {upper:.3f}] rad")

    finger_lower = torch.tensor(finger_lower, device=device)
    finger_upper = torch.tensor(finger_upper, device=device)

    print("\n" + "=" * 60)
    print("TEST STARTING: Watch the robot hand open and close")
    print("Green sphere = Palm position")
    print("Red sphere = Palm forward direction")
    print("=" * 60 + "\n")

    # Animation parameters
    cycle_duration = 2.0  # seconds for full open-close cycle
    step = 0

    while simulation_app.is_running():
        # Calculate phase (0 to 1 to 0)
        t = (step * sim_cfg.dt) % cycle_duration
        phase = abs(2.0 * t / cycle_duration - 1.0)  # 0->1->0 triangle wave

        # Interpolate finger positions between open and closed
        # Open = lower limit, Closed = upper limit
        finger_targets = finger_lower + phase * (finger_upper - finger_lower)

        # Get current joint positions
        joint_pos = robot.data.joint_pos.clone()

        # Set finger targets
        for i, idx in enumerate(finger_indices):
            joint_pos[0, idx] = finger_targets[i]

        # Hold arm in a visible position (slightly forward and to the side)
        arm_pose = [0.3, 0.3, 0.0, 0.5, 0.0]  # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll
        for i, idx in enumerate(arm_indices):
            joint_pos[0, idx] = arm_pose[i]

        # Apply joint targets
        robot.set_joint_position_target(joint_pos)

        # Get palm position and orientation
        palm_pos = robot.data.body_pos_w[:, palm_idx, :]  # (1, 3)
        palm_quat = robot.data.body_quat_w[:, palm_idx, :]  # (1, 4) wxyz

        # Calculate forward direction from quaternion
        # Forward direction in palm frame is typically +X or -Z
        w, x, y, z = palm_quat[0, 0], palm_quat[0, 1], palm_quat[0, 2], palm_quat[0, 3]
        # Rotate unit vector [1, 0, 0] by quaternion
        forward_x = 1 - 2 * (y * y + z * z)
        forward_y = 2 * (x * y + w * z)
        forward_z = 2 * (x * z - w * y)
        forward = torch.tensor([[forward_x, forward_y, forward_z]], device=device)
        forward_pos = palm_pos + 0.08 * forward  # 8cm in front of palm

        # Update markers
        marker_pos = torch.cat([palm_pos, forward_pos], dim=0)  # (2, 3)
        marker_indices = torch.tensor([0, 1], device=device)
        markers.visualize(marker_pos, marker_indices=marker_indices)

        # Print status every 100 steps
        if step % 100 == 0:
            state = "CLOSING" if t < cycle_duration / 2 else "OPENING"
            print(f"Step {step:5d} | Phase: {phase:.2f} | {state}")
            print(f"  Palm pos: [{palm_pos[0, 0]:.3f}, {palm_pos[0, 1]:.3f}, {palm_pos[0, 2]:.3f}]")
            print(f"  Palm quat (wxyz): [{w:.3f}, {x:.3f}, {y:.3f}, {z:.3f}]")

        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

        step += 1

        if step > 2000:  # Run for ~10 seconds
            break

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    simulation_app.close()


if __name__ == "__main__":
    main()