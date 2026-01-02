# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V4 (Agile Locomotion + DiffIK)
# Lower body: Agile Policy (PPO pre-trained)
# Upper body: Differential IK

"""
G1 Locomanipulation Demo V4
- Lower Body: Agile Locomotion Policy (standing balance)
- Upper Body: Differential IK (arm manipulation)
- Full floating-base humanoid control

This is a simplified direct workflow implementation of NVIDIA's
official locomanipulation environment.

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v4.py
"""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Locomanipulation V4")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after app launch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.io.torchscript import load_torchscript_model

# Use G1_29DOF_CFG (with wrist joints)
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

print("\n" + "=" * 70)
print("  G1 Locomanipulation Demo - V4")
print("  Lower Body: Agile Locomotion Policy (PPO)")
print("  Upper Body: Differential IK")
print("=" * 70 + "\n")


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1LocomanipSceneCfg(InteractiveSceneCfg):
    """Scene with G1 robot for locomanipulation."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Packing Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Steering wheel
    steering_wheel = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SteeringWheel",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Red cube - manipulation target
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.45, 0.75)),
    )

    # Blue cylinder
    blue_cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.45, 0.75)),
    )

    # G1 Robot - 29 DOF (floating base for locomotion)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),  # Standing height
            joint_pos={
                # Legs - slightly bent for stability
                "left_hip_pitch_joint": -0.1,
                "right_hip_pitch_joint": -0.1,
                "left_hip_roll_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.2,
                "right_knee_joint": 0.2,
                "left_ankle_pitch_joint": -0.1,
                "right_ankle_pitch_joint": -0.1,
                "left_ankle_roll_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                # Arms in neutral position
                "left_shoulder_pitch_joint": 0.3,
                "right_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "left_shoulder_yaw_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.5,
                "right_elbow_joint": 0.5,
                # Wrists neutral
                "left_wrist_roll_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Waist straight
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
            },
        ),
    )


# ============================================================================
# AGILE LOCOMOTION POLICY WRAPPER
# ============================================================================

class AgileLocomotionPolicy:
    """Wrapper for the pre-trained Agile locomotion policy."""

    def __init__(self, robot: Articulation, device: str):
        self.robot = robot
        self.device = device
        self.num_envs = robot.num_instances

        # Load the Agile policy
        policy_path = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
        try:
            resolved_path = retrieve_file_path(policy_path)
            self.policy = load_torchscript_model(resolved_path, device=device)
            print(f"[INFO] Loaded Agile policy from: {resolved_path}")
            self._policy_loaded = True
        except Exception as e:
            print(f"[WARNING] Could not load Agile policy: {e}")
            print("[WARNING] Using fallback standing controller")
            self._policy_loaded = False

        # Lower body joint names (same order as Agile policy expects)
        self.lower_body_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]

        # Find joint IDs
        self.lower_body_joint_ids = []
        all_joint_names = robot.joint_names
        for name in self.lower_body_joint_names:
            if name in all_joint_names:
                self.lower_body_joint_ids.append(all_joint_names.index(name))

        print(f"[INFO] Lower body joints: {len(self.lower_body_joint_ids)}")

        # Default standing pose
        self.default_lower_body_pos = torch.tensor([
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # Left leg
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,  # Right leg
        ], device=device).unsqueeze(0).repeat(self.num_envs, 1)

        # Scale for policy output (from config)
        self.policy_output_scale = 0.25

        # Store last action for observation
        self.last_action = torch.zeros(self.num_envs, 12, device=device)

    def compute_observations(self) -> torch.Tensor:
        """Compute observations for Agile policy.

        Observation structure (from AgileTeacherPolicyObservationsCfg):
        - base_lin_vel (3)
        - base_ang_vel (3)
        - projected_gravity (3)
        - joint_pos_rel (all joints ~29)
        - joint_vel_rel (all joints ~29)
        - last_action (12)

        Plus command input [vx, vy, wz, hip_height] prepended.
        """
        # Base velocities (in base frame)
        base_lin_vel = self.robot.data.root_lin_vel_b  # (N, 3)
        base_ang_vel = self.robot.data.root_ang_vel_b  # (N, 3)

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        root_quat = self.robot.data.root_quat_w
        # Transform gravity to body frame
        projected_gravity = self._quat_rotate_inverse(root_quat, gravity_vec.unsqueeze(0).repeat(self.num_envs, 1))

        # Joint positions and velocities (relative to default)
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel * 0.1  # scaled
        default_joint_pos = self.robot.data.default_joint_pos
        joint_pos_rel = joint_pos - default_joint_pos

        # Concatenate observations
        obs = torch.cat([
            base_lin_vel,  # 3
            base_ang_vel,  # 3
            projected_gravity,  # 3
            joint_pos_rel,  # ~43
            joint_vel,  # ~43
            self.last_action,  # 12
        ], dim=-1)

        return obs

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse of quaternion."""
        q_w = q[:, 0:1]
        q_vec = q[:, 1:4]
        a = v * (2.0 * q_w * q_w - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def get_joint_targets(self, command: torch.Tensor = None) -> torch.Tensor:
        """Get lower body joint position targets.

        Args:
            command: [vx, vy, wz, hip_height] velocity commands (optional)
                     Default is standing still: [0, 0, 0, 0]

        Returns:
            Joint position targets for lower body (12 joints)
        """
        if not self._policy_loaded:
            # Fallback: return default standing pose
            return self.default_lower_body_pos

        # Default command: stand still
        if command is None:
            command = torch.zeros(self.num_envs, 4, device=self.device)

        # Get observations
        obs = self.compute_observations()

        # Compose policy input: [command (repeated), observations]
        # The policy expects command to be prepended
        policy_input = torch.cat([command, obs], dim=-1)

        try:
            # Run policy
            with torch.no_grad():
                joint_actions = self.policy.forward(policy_input)

            # Apply scaling and offset
            joint_targets = joint_actions * self.policy_output_scale + self.default_lower_body_pos

            # Store for next observation
            self.last_action = joint_actions.clone()

            return joint_targets

        except Exception as e:
            print(f"[WARNING] Policy forward failed: {e}")
            return self.default_lower_body_pos


# ============================================================================
# PICK AND PLACE STATE MACHINE
# ============================================================================

class PickPlaceStateMachine:
    """State machine for pick-and-place task."""

    def __init__(self, dt: float, device: str):
        self.dt = dt
        self.device = device

        # Target positions in world frame (reachable by arm)
        HOME = [0.15, 0.25, 0.85]
        ABOVE_CUBE = [0.0, 0.35, 0.85]
        AT_CUBE = [0.0, 0.38, 0.78]
        LIFTED = [0.0, 0.35, 0.90]
        ABOVE_DROP = [0.12, 0.35, 0.90]
        AT_DROP = [0.12, 0.38, 0.82]

        self.states = [
            {"name": "HOME", "pos": HOME, "grip": 0, "dur": 3.0},
            {"name": "APPROACH", "pos": ABOVE_CUBE, "grip": 0, "dur": 3.0},
            {"name": "REACH", "pos": AT_CUBE, "grip": 0, "dur": 2.0},
            {"name": "GRASP", "pos": AT_CUBE, "grip": 1, "dur": 1.5},
            {"name": "LIFT", "pos": LIFTED, "grip": 1, "dur": 2.0},
            {"name": "MOVE", "pos": ABOVE_DROP, "grip": 1, "dur": 2.5},
            {"name": "LOWER", "pos": AT_DROP, "grip": 1, "dur": 2.0},
            {"name": "RELEASE", "pos": AT_DROP, "grip": 0, "dur": 1.5},
            {"name": "RETRACT", "pos": HOME, "grip": 0, "dur": 3.0},
            {"name": "DONE", "pos": HOME, "grip": 0, "dur": 999.0},
        ]

        self.current_state = 0
        self.state_timer = 0.0

        print("[StateMachine] States:")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s['name']}: pos={s['pos']}, grip={s['grip']}, dur={s['dur']}s")

    def reset(self):
        self.current_state = 0
        self.state_timer = 0.0
        print(f"\n[State] → {self.states[0]['name']}")

    def step(self):
        self.state_timer += self.dt
        state = self.states[self.current_state]

        if self.state_timer >= state["dur"] and self.current_state < len(self.states) - 1:
            self.current_state += 1
            self.state_timer = 0.0
            print(f"\n[State Change] → {self.states[self.current_state]['name']}")

    def get_target(self) -> tuple:
        state = self.states[self.current_state]
        return (
            torch.tensor([state["pos"]], device=self.device),
            state["grip"],
            state["name"]
        )


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Simulation parameters
    sim_dt = 0.005
    decimation = 4
    control_dt = sim_dt * decimation

    # Configure simulation
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    # Create scene
    scene_cfg = G1LocomanipSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Get robot
    robot: Articulation = scene["robot"]

    # Reset simulation
    sim.reset()
    scene.reset()
    print("[INFO] Simulation reset complete.")

    # Print robot info
    print(f"\n[DEBUG] Available joints ({len(robot.joint_names)}):")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    # ========================================================================
    # Setup Agile Locomotion Policy (Lower Body)
    # ========================================================================
    agile_policy = AgileLocomotionPolicy(robot, args_cli.device)

    # ========================================================================
    # Setup Differential IK (Upper Body - Right Arm)
    # ========================================================================

    # Find right arm joints
    possible_arm_joints = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    arm_joint_ids = []
    arm_joint_names = []
    for name in possible_arm_joints:
        if name in robot.joint_names:
            arm_joint_ids.append(robot.joint_names.index(name))
            arm_joint_names.append(name)

    print(f"\n[INFO] Found arm joints ({len(arm_joint_ids)}):")
    for name in arm_joint_names:
        print(f"  - {name}")

    # Find end-effector
    ee_name = "right_wrist_yaw_link"
    ee_body_id = robot.body_names.index(ee_name) if ee_name in robot.body_names else None
    print(f"[INFO] Found end-effector: {ee_name} (ID: {ee_body_id})")

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1}
    )

    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=args_cli.num_envs, device=args_cli.device)

    # Jacobian index (for floating base: body_id - 1)
    jacobian_ee_idx = ee_body_id - 1 if not robot.is_fixed_base else ee_body_id
    print(f"[INFO] Jacobian EE index: {jacobian_ee_idx}")

    # ========================================================================
    # Visualization Markers
    # ========================================================================
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # ========================================================================
    # State Machine
    # ========================================================================
    state_machine = PickPlaceStateMachine(control_dt, args_cli.device)
    state_machine.reset()

    # ========================================================================
    # Main Loop
    # ========================================================================
    print("\n[INFO] Starting simulation...")
    print("[INFO] Lower body: Agile Policy | Upper body: Differential IK")
    print("[INFO] Press Ctrl+C to stop.\n")

    step_count = 0
    cycle_count = 0
    max_cycles = 2

    while simulation_app.is_running():
        # Get current state target
        target_pos, grip_state, state_name = state_machine.get_target()

        # Update state machine
        state_machine.step()

        # ====================================================================
        # Lower Body Control (Agile Policy)
        # ====================================================================

        # Standing command: [vx, vy, wz, hip_height] = [0, 0, 0, 0]
        lower_body_command = torch.zeros(args_cli.num_envs, 4, device=args_cli.device)

        # Get lower body joint targets from Agile policy
        lower_body_targets = agile_policy.get_joint_targets(lower_body_command)

        # Apply to lower body joints
        if len(agile_policy.lower_body_joint_ids) > 0:
            robot.set_joint_position_target(
                lower_body_targets,
                joint_ids=agile_policy.lower_body_joint_ids
            )

        # ====================================================================
        # Upper Body Control (Differential IK)
        # ====================================================================

        # Get current EE pose
        ee_pos_w = robot.data.body_pos_w[:, ee_body_id, :]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_id, :]

        # Get root pose
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w

        # Transform target to body frame
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w,
            target_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args_cli.device)
        )

        # Reset IK controller
        diff_ik.reset()
        diff_ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        # Get Jacobian and current joint state
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_ee_idx, :, :]
        arm_jacobian = jacobian[:, :, arm_joint_ids]

        # EE pose in body frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # Current arm joint positions
        current_arm_pos = robot.data.joint_pos[:, arm_joint_ids]

        # Compute IK
        arm_joint_targets = diff_ik.compute(ee_pos_b, ee_quat_b, arm_jacobian, current_arm_pos)

        # Apply to arm joints
        robot.set_joint_position_target(arm_joint_targets, joint_ids=arm_joint_ids)

        # ====================================================================
        # Simulation Step
        # ====================================================================

        # Write commands
        robot.write_data_to_sim()
        scene.write_data_to_sim()

        # Step simulation
        for _ in range(decimation):
            sim.step(render=False)

        # Update scene
        scene.update(sim_dt * decimation)

        # Render
        sim.render()

        # Update visualization
        ee_marker.visualize(ee_pos_w, ee_quat_w)
        goal_marker.visualize(target_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args_cli.device))

        # Logging
        step_count += 1
        if step_count % 50 == 0:
            error = torch.norm(ee_pos_w - target_pos, dim=-1).item()
            base_height = robot.data.root_pos_w[0, 2].item()
            print(f"[Step {step_count:4d}] State: {state_name:10s} | "
                  f"EE: [{ee_pos_w[0, 0]:.3f}, {ee_pos_w[0, 1]:.3f}, {ee_pos_w[0, 2]:.3f}] | "
                  f"Error: {error:.4f}m | Base Z: {base_height:.3f}m")

        # Reset after cycle
        if state_name == "DONE" and state_machine.state_timer > 3.0:
            cycle_count += 1
            if cycle_count >= max_cycles:
                print(f"\n[INFO] Demo completed ({max_cycles} cycles). Exiting...")
                break
            print(f"\n[INFO] Resetting simulation (cycle {cycle_count})...")
            state_machine.reset()

    print("\n" + "=" * 70)
    print("  G1 Locomanipulation Demo V4 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()