# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V3
# Uses full G1_CFG (with wrist joints) instead of G1_MINIMAL_CFG

"""
G1 Pick-and-Place Demo V3
- Uses G1_CFG (full robot with wrist joints = 7-DOF arm)
- Proper scene setup with table and objects
- Correct robot spawn position
- DifferentialIKController with full arm control

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v3.py
"""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V3")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after app launch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.shapes import ShapeCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Use FULL G1_CFG (not MINIMAL) - this includes wrist joints!
from isaaclab_assets.robots.unitree import G1_CFG

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V3")
print("  Using FULL G1_CFG (with wrist joints) + Proper Scene")
print("=" * 70 + "\n")


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene with G1 robot, table, and manipulation objects."""

    # Ground plane
    ground = sim_utils.GroundPlaneCfg()

    # Lighting
    dome_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.9, 0.9, 0.9),
    )

    # Table - positioned so robot can reach
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.4),  # width, depth, height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.2),  # Table center at x=0.5, z=0.2 means top at z=0.4
        ),
    )

    # Target object (red cube) - on the table
    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.15, 0.45),  # On table, slightly to the right
        ),
    )

    # Secondary object (blue cylinder) - for variety
    secondary_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.15, 0.45),  # On table, slightly to the left
        ),
    )

    # G1 Robot - FULL CONFIG (with wrist joints!)
    robot: ArticulationCfg = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),  # Standing height
            joint_pos={
                # Legs - standing pose
                "left_hip_pitch_joint": -0.1,
                "right_hip_pitch_joint": -0.1,
                "left_hip_roll_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.3,
                "right_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "right_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                # Arms - ready pose (slightly forward)
                "left_shoulder_pitch_joint": 0.3,
                "right_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.5,
                "right_elbow_pitch_joint": 0.5,
                "left_elbow_roll_joint": 0.0,
                "right_elbow_roll_joint": 0.0,
                # Wrist joints (FULL G1_CFG has these!)
                "left_wrist_pitch_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            },
        ),
    )


# ============================================================================
# STATE MACHINE
# ============================================================================

class PickPlaceStateMachine:
    """Simple state machine for pick and place sequence."""

    def __init__(self, device: str):
        self.device = device
        self.current_state = 0
        self.state_timer = 0.0

        # Target positions relative to robot (robot at origin)
        # Table top is at z=0.4, object at z=0.45
        # Robot base is at z=0.74, so relative z = 0.45 - 0.74 = -0.29 (below base)
        # But we need world frame positions for IK

        # World frame targets (robot at origin, table at x=0.5)
        ABOVE_OBJECT = [0.45, 0.15, 0.55]  # Above red cube
        AT_OBJECT = [0.45, 0.15, 0.47]  # At red cube level
        LIFTED = [0.45, 0.15, 0.60]  # Lifted
        ABOVE_DROP = [0.45, -0.15, 0.60]  # Above blue cylinder area
        AT_DROP = [0.45, -0.15, 0.50]  # Drop position
        HOME = [0.35, 0.0, 0.55]  # Home position

        self.states = [
            {"name": "HOME", "pos": HOME, "grip": 0, "dur": 2.0},
            {"name": "APPROACH", "pos": ABOVE_OBJECT, "grip": 0, "dur": 2.0},
            {"name": "REACH", "pos": AT_OBJECT, "grip": 0, "dur": 1.5},
            {"name": "GRASP", "pos": AT_OBJECT, "grip": 1, "dur": 1.0},
            {"name": "LIFT", "pos": LIFTED, "grip": 1, "dur": 1.5},
            {"name": "MOVE", "pos": ABOVE_DROP, "grip": 1, "dur": 2.0},
            {"name": "LOWER", "pos": AT_DROP, "grip": 1, "dur": 1.5},
            {"name": "RELEASE", "pos": AT_DROP, "grip": 0, "dur": 1.0},
            {"name": "RETRACT", "pos": HOME, "grip": 0, "dur": 2.0},
            {"name": "DONE", "pos": HOME, "grip": 0, "dur": 999.0},
        ]

        # Default orientation (palm down)
        self.default_quat = torch.tensor([0.0, 0.707, 0.707, 0.0], device=device)

        print("[StateMachine] States:")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s['name']}: pos={s['pos']}, grip={s['grip']}, dur={s['dur']}s")

    def update(self, dt: float) -> bool:
        """Update state, return True if state changed."""
        self.state_timer += dt

        if self.state_timer >= self.states[self.current_state]["dur"]:
            if self.current_state < len(self.states) - 1:
                self.current_state += 1
                self.state_timer = 0.0
                print(f"\n[State Change] → {self.states[self.current_state]['name']}")
                return True
        return False

    def get_ee_target(self):
        """Get current end-effector target (position, quaternion)."""
        state = self.states[self.current_state]
        pos = torch.tensor(state["pos"], device=self.device)
        return pos, self.default_quat

    def get_gripper_state(self) -> int:
        """Get gripper state: 0=open, 1=closed."""
        return self.states[self.current_state]["grip"]

    def reset(self):
        self.current_state = 0
        self.state_timer = 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Simulation settings
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.0, 1.2], [0.4, 0.0, 0.5])

    # Create scene
    scene_cfg = G1PickPlaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Get robot
    robot: Articulation = scene["robot"]

    # Create visualization markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Create state machine
    state_machine = PickPlaceStateMachine(device=sim.device)

    # IMPORTANT: Reset simulation FIRST - PhysX view is only available after reset!
    sim.reset()
    print("[INFO] Simulation reset complete.")

    # Print available joints and bodies for debugging
    print(f"\n[DEBUG] Available joints ({len(robot.joint_names)}):")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    print(f"\n[DEBUG] Available bodies ({len(robot.body_names)}):")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {name}")

    # Find right arm joints (7-DOF with wrist)
    # G1_CFG should have: shoulder_pitch, shoulder_roll, shoulder_yaw,
    #                     elbow_pitch, elbow_roll, wrist_pitch, wrist_roll, wrist_yaw
    arm_joint_names = []
    arm_joint_pattern = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ]

    # Check if wrist joints exist
    wrist_joints = ["right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint"]
    for wj in wrist_joints:
        if wj in robot.joint_names:
            arm_joint_pattern.append(wj)

    print(f"\n[INFO] Arm joints to use: {arm_joint_pattern}")

    # Find end-effector body
    ee_body_name = None
    ee_candidates = ["right_wrist_yaw_link", "right_palm_link", "right_hand_link",
                     "right_elbow_roll_link", "right_wrist_roll_link"]
    for candidate in ee_candidates:
        if candidate in robot.body_names:
            ee_body_name = candidate
            break

    if ee_body_name is None:
        # Use last body in right arm chain
        for name in robot.body_names:
            if "right" in name and ("wrist" in name or "palm" in name or "hand" in name):
                ee_body_name = name
                break

    if ee_body_name is None:
        print("[WARNING] Could not find right arm end-effector, using right_elbow_roll_link")
        ee_body_name = "right_elbow_roll_link"

    print(f"[INFO] Using end-effector: {ee_body_name}")

    # Configure robot entity for IK
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=arm_joint_pattern,
        body_names=[ee_body_name],
    )
    robot_entity_cfg.resolve(scene)

    # Get joint and body indices
    arm_joint_ids = robot_entity_cfg.joint_ids
    ee_body_id = robot_entity_cfg.body_ids[0]

    print(f"\n[INFO] Right arm joint IDs: {arm_joint_ids}")
    print(f"[INFO] Right arm joint names: {[robot.joint_names[i] for i in arm_joint_ids]}")
    print(f"[INFO] EE body ID: {ee_body_id} ({robot.body_names[ee_body_id]})")

    # For floating-base robot, Jacobian index equals body index
    if robot.is_fixed_base:
        ee_jacobi_idx = ee_body_id - 1
    else:
        ee_jacobi_idx = ee_body_id

    print(f"[INFO] Robot is_fixed_base: {robot.is_fixed_base}")
    print(f"[INFO] Jacobian EE index: {ee_jacobi_idx}")

    # Create Differential IK Controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=scene.num_envs,
        device=sim.device
    )

    print(f"[INFO] IK Controller created with method: {diff_ik_cfg.ik_method}")

    # Store initial joint positions for arm
    joint_pos_des = robot.data.joint_pos[:, arm_joint_ids].clone()

    print("\n[INFO] Starting simulation...")
    print("[INFO] Press Ctrl+C to stop.\n")
    print(f"[State] → {state_machine.states[0]['name']}")

    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0
    reset_count = 0

    while simulation_app.is_running():
        # Update state machine
        state_machine.update(sim_dt)

        # Get target from state machine (world frame)
        target_pos_w, target_quat_w = state_machine.get_ee_target()
        target_pos_w = target_pos_w.unsqueeze(0)  # Add batch dim
        target_quat_w = target_quat_w.unsqueeze(0)

        # Get current robot state
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]

        # Get Jacobian
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]

        # Transform EE pose to base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Transform target to base frame
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_w, target_quat_w
        )

        # Create IK command (position + quaternion in base frame)
        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)

        # Set command and compute
        diff_ik_controller.reset()
        diff_ik_controller.set_command(ik_command)
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Apply joint positions
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)

        # Write to sim and step
        scene.write_data_to_sim()
        sim.step()
        count += 1

        # Update scene
        scene.update(sim_dt)

        # Update visualization markers
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(target_pos_w, target_quat_w)

        # Log every 50 steps
        if count % 50 == 0:
            ee_pos = ee_pose_w[0, :3]
            target = target_pos_w[0]
            error = (ee_pos - target).norm().item()
            state_name = state_machine.states[state_machine.current_state]["name"]
            print(f"[Step {count:4d}] State: {state_name:10s} | "
                  f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] | "
                  f"Target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] | "
                  f"Error: {error:.4f}m")

        # Reset after reaching DONE state for a while
        if state_machine.current_state == len(state_machine.states) - 1:
            if state_machine.state_timer > 3.0:
                reset_count += 1
                if reset_count >= 2:
                    print("\n[INFO] Demo completed. Exiting...")
                    break
                print(f"\n[INFO] Resetting simulation (cycle {reset_count})...")
                state_machine.reset()
                print(f"[State] → {state_machine.states[0]['name']}")

    print("\n" + "=" * 70)
    print("  G1 Pick-and-Place Demo V3 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()