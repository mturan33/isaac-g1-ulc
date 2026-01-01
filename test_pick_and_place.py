# Copyright (c) 2025, VLM-RL G1 Project
# Pick and Place using Official DifferentialIKController Tutorial Approach

"""
G1 Pick-and-Place Demo - V2 (Based on Official Tutorial)

This script follows the official Isaac Lab DifferentialIKController tutorial:
https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_diff_ik.html

Key differences from V1:
1. Uses DifferentialIKController directly (not through action term)
2. Properly transforms EE pose to body frame using subtract_frame_transforms
3. Correctly handles Jacobian indexing for floating-base robots
4. Manual control loop instead of relying on action manager

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v2.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick and Place - Tutorial Based")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Import G1 robot config
from isaaclab_assets import G1_MINIMAL_CFG

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V2 (Official Tutorial Based)")
print("  Using DifferentialIKController directly with proper body frame transforms")
print("=" * 70 + "\n")


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 pick and place task."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.4)),
    )

    # Target object (cube to pick)
    target_object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.2, 0.45)),
    )

    # G1 Robot
    robot: AssetBaseCfg = G1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                # Legs - standing pose
                "left_hip_pitch_joint": -0.1,
                "right_hip_pitch_joint": -0.1,
                "left_knee_joint": 0.3,
                "right_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "right_ankle_pitch_joint": -0.2,
                # Arms - neutral forward pose
                "left_shoulder_pitch_joint": 0.3,
                "right_shoulder_pitch_joint": 0.3,
                "left_elbow_joint": 0.5,
                "right_elbow_joint": 0.5,
            },
        ),
    )


# ============================================================================
# STATE MACHINE
# ============================================================================

class PickPlaceStateMachine:
    """State machine for pick-and-place task."""

    def __init__(self, device: str):
        self.device = device
        self.current_state = 0
        self.state_timer = 0.0

        # Target positions (world frame)
        # Object is at (0.5, 0.2, 0.45), table at z=0.425
        OBJECT_POS = [0.5, 0.2, 0.50]  # Slightly above object
        OBJECT_GRASP = [0.5, 0.2, 0.47]  # At object level
        LIFT_POS = [0.5, 0.2, 0.65]  # Lifted
        DROP_POS = [0.5, -0.2, 0.65]  # Other side of table
        DROP_LOW = [0.5, -0.2, 0.50]  # Lower to drop

        # Gripper: 0 = open, 1 = closed (we'll use hand joints)
        self.states = [
            {"name": "INIT", "pos": [0.3, 0.2, 0.6], "grip": 0, "dur": 2.0},
            {"name": "APPROACH", "pos": OBJECT_POS, "grip": 0, "dur": 3.0},
            {"name": "REACH", "pos": OBJECT_GRASP, "grip": 0, "dur": 2.0},
            {"name": "GRASP", "pos": OBJECT_GRASP, "grip": 1, "dur": 1.0},
            {"name": "LIFT", "pos": LIFT_POS, "grip": 1, "dur": 2.0},
            {"name": "MOVE", "pos": DROP_POS, "grip": 1, "dur": 3.0},
            {"name": "LOWER", "pos": DROP_LOW, "grip": 1, "dur": 2.0},
            {"name": "RELEASE", "pos": DROP_LOW, "grip": 0, "dur": 1.0},
            {"name": "RETRACT", "pos": [0.3, -0.2, 0.6], "grip": 0, "dur": 2.0},
            {"name": "DONE", "pos": [0.3, 0.0, 0.6], "grip": 0, "dur": 999.0},
        ]

        # Default orientation (palm down, fingers forward)
        # quaternion: w, x, y, z
        self.default_quat = torch.tensor([0.0, 0.707, 0.707, 0.0], device=device)

        print("[StateMachine] States:")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s['name']}: pos={s['pos']}, grip={s['grip']}, dur={s['dur']}s")

    def update(self, dt: float) -> bool:
        """Update state, return True if state changed."""
        self.state_timer += dt
        state = self.states[self.current_state]

        if self.state_timer >= state["dur"] and self.current_state < len(self.states) - 1:
            self.current_state += 1
            self.state_timer = 0.0
            new_state = self.states[self.current_state]
            print(f"\n[State Change] → {new_state['name']}")
            return True
        return False

    def get_ee_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get target EE position and orientation (world frame)."""
        state = self.states[self.current_state]
        pos = torch.tensor(state["pos"], device=self.device, dtype=torch.float32)
        return pos, self.default_quat.clone()

    def get_gripper(self) -> float:
        """Get gripper state: 0=open, 1=closed."""
        return self.states[self.current_state]["grip"]

    def reset(self):
        self.current_state = 0
        self.state_timer = 0.0


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    # Simulation settings
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.5], [0.5, 0.0, 0.5])

    # Create scene
    scene_cfg = G1PickPlaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Get robot
    robot: Articulation = scene["robot"]

    # Configure robot entity for IK
    # G1 right arm joints and end-effector
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        body_names=["right_wrist_yaw_link"],  # End-effector
    )
    robot_entity_cfg.resolve(scene)

    # Get joint and body indices
    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_id = robot_entity_cfg.body_ids[0]

    print(f"\n[INFO] Right arm joint IDs: {arm_joint_ids}")
    print(f"[INFO] Right arm joint names: {[robot.joint_names[i] for i in arm_joint_ids]}")
    print(f"[INFO] EE body ID: {ee_body_id} ({robot.body_names[ee_body_id]})")

    # For floating-base robot, Jacobian index equals body index
    # (For fixed-base, it would be body_id - 1)
    if robot.is_fixed_base:
        ee_jacobi_idx = ee_body_id - 1
    else:
        ee_jacobi_idx = ee_body_id

    print(f"[INFO] Robot is_fixed_base: {robot.is_fixed_base}")
    print(f"[INFO] Jacobian EE index: {ee_jacobi_idx}")

    # Create Differential IK Controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,  # Absolute pose commands
        ik_method="dls",  # Damped Least Squares
        ik_params={"lambda_val": 0.1},  # Damping factor
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=scene.num_envs,
        device=sim.device
    )

    print(f"[INFO] IK Controller created with method: {diff_ik_cfg.ik_method}")

    # Create visualization markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Create state machine
    state_machine = PickPlaceStateMachine(device=sim.device)

    # Reset simulation
    sim.reset()

    # Store initial joint positions for arm
    joint_pos_des = robot.data.joint_pos[:, arm_joint_ids].clone()

    print("\n[INFO] Starting simulation...")
    print("[INFO] Press Ctrl+C to stop.\n")
    print(f"[State] → {state_machine.states[0]['name']}")

    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Update state machine
        state_machine.update(sim_dt)

        # Get target from state machine (world frame)
        target_pos_w, target_quat_w = state_machine.get_ee_target()
        target_pos_w = target_pos_w.unsqueeze(0)  # Add batch dim
        target_quat_w = target_quat_w.unsqueeze(0)

        # Add environment origin offset
        target_pos_w = target_pos_w + scene.env_origins

        # Set IK command (world frame target)
        ik_command = torch.cat([target_pos_w, target_quat_w], dim=-1)
        diff_ik_controller.set_command(ik_command)

        # Get current robot state
        root_pose_w = robot.data.root_state_w[:, 0:7]  # [pos(3), quat(4)]
        ee_pose_w = robot.data.body_state_w[:, ee_body_id, 0:7]
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]

        # Transform EE pose from world frame to body (root) frame
        # This is CRITICAL for proper IK computation!
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],  # Root frame
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]  # EE in world
        )

        # Get Jacobian from PhysX
        # Shape: (num_envs, num_bodies, 6, num_joints)
        jacobian = robot.root_physx_view.get_jacobians()

        # Extract Jacobian for EE body and arm joints only
        # Shape: (num_envs, 6, num_arm_joints)
        jacobian_ee = jacobian[:, ee_jacobi_idx, :, arm_joint_ids]

        # Compute IK: get desired joint positions
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian_ee, joint_pos
        )

        # Apply joint position targets (only for arm joints)
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)

        # Write data and step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1

        # Update scene buffers
        scene.update(sim_dt)

        # Update visualization markers
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(target_pos_w, target_quat_w)

        # Log every 50 steps
        if count % 50 == 0:
            state = state_machine.states[state_machine.current_state]
            ee_pos = ee_pose_w[0, 0:3].cpu().numpy()
            tgt_pos = target_pos_w[0].cpu().numpy()
            error = ((ee_pos[0] - tgt_pos[0]) ** 2 + (ee_pos[1] - tgt_pos[1]) ** 2 + (
                        ee_pos[2] - tgt_pos[2]) ** 2) ** 0.5

            print(f"[Step {count:4d}] State: {state['name']:10s} | "
                  f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                  f"Target: [{tgt_pos[0]:.3f}, {tgt_pos[1]:.3f}, {tgt_pos[2]:.3f}] | "
                  f"Error: {error:.4f}m")

        # Reset after some time
        if count % 2000 == 0:
            print("\n[INFO] Resetting simulation...")

            # Reset joint state
            joint_pos_reset = robot.data.default_joint_pos.clone()
            joint_vel_reset = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos_reset, joint_vel_reset)
            robot.reset()

            # Reset controller
            diff_ik_controller.reset()

            # Reset state machine
            state_machine.reset()

            joint_pos_des = robot.data.joint_pos[:, arm_joint_ids].clone()
            print(f"[State] → {state_machine.states[0]['name']}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()