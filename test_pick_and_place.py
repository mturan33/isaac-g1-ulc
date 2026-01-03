# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V7 (Simple Standing Controller)
# Strategy: Use simple PD controller for standing, focus on working IK

"""
G1 Locomanipulation Demo V7
- Lower Body: Simple PD standing controller (reliable)
- Upper Body: Differential IK (working)

This version prioritizes stability over complexity.
We can add Agile locomotion later once the basic demo works.

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v7.py
"""

import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Locomanipulation V7")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

print("\n" + "=" * 70)
print("  G1 Locomanipulation Demo - V7")
print("  Lower Body: Simple PD Standing Controller")
print("  Upper Body: Differential IK")
print("=" * 70 + "\n")


# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

@configclass
class G1LocomanipSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

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

    # G1 Robot with custom standing pose
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
                # Legs - stable standing pose (slightly bent)
                "left_hip_pitch_joint": -0.15,
                "right_hip_pitch_joint": -0.15,
                "left_hip_roll_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.3,
                "right_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.15,
                "right_ankle_pitch_joint": -0.15,
                "left_ankle_roll_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                # Arms
                "left_shoulder_pitch_joint": 0.3,
                "right_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "left_shoulder_yaw_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.5,
                "right_elbow_joint": 0.5,
                "left_wrist_roll_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                # Waist
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
            },
        ),
    )


# ============================================================================
# SIMPLE STANDING CONTROLLER
# ============================================================================

class SimpleStandingController:
    """Simple PD controller to maintain standing pose.

    This is a basic controller that tries to keep the robot in a
    stable standing position by holding leg joints at target positions.
    """

    def __init__(self, robot: Articulation, device: str):
        self.robot = robot
        self.device = device
        self.num_envs = robot.num_instances

        # Lower body joint names
        self.lower_body_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]

        # Waist joint names (also controlled for stability)
        self.waist_joint_names = [
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
        ]

        # Find joint IDs
        all_joints = list(robot.joint_names)

        self.lower_body_joint_ids = []
        for name in self.lower_body_joint_names:
            if name in all_joints:
                self.lower_body_joint_ids.append(all_joints.index(name))

        self.waist_joint_ids = []
        for name in self.waist_joint_names:
            if name in all_joints:
                self.waist_joint_ids.append(all_joints.index(name))

        # All controlled joints
        self.controlled_joint_ids = self.lower_body_joint_ids + self.waist_joint_ids

        print(f"[INFO] Standing controller - Lower body joints: {len(self.lower_body_joint_ids)}")
        print(f"[INFO] Standing controller - Waist joints: {len(self.waist_joint_ids)}")

        # Target positions (stable standing pose)
        # Legs: slightly bent for stability
        self.target_lower_body = torch.tensor([
            -0.15, 0.0, 0.0, 0.3, -0.15, 0.0,  # Left leg
            -0.15, 0.0, 0.0, 0.3, -0.15, 0.0,  # Right leg
        ], device=device).unsqueeze(0).repeat(self.num_envs, 1)

        # Waist: straight
        self.target_waist = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(self.num_envs, 1)

    def get_joint_targets(self) -> tuple:
        """Get target joint positions for standing.

        Returns:
            Tuple of (lower_body_targets, waist_targets)
        """
        return self.target_lower_body, self.target_waist


# ============================================================================
# STATE MACHINE
# ============================================================================

class PickPlaceStateMachine:
    def __init__(self, dt: float, device: str):
        self.dt = dt
        self.device = device

        # Targets are closer to the robot for better reachability
        HOME = [0.15, 0.25, 0.85]
        ABOVE_CUBE = [0.0, 0.35, 0.85]
        AT_CUBE = [0.0, 0.40, 0.78]
        LIFTED = [0.0, 0.35, 0.90]
        ABOVE_DROP = [0.12, 0.35, 0.90]
        AT_DROP = [0.12, 0.40, 0.82]

        self.states = [
            {"name": "HOME", "pos": HOME, "dur": 3.0},
            {"name": "APPROACH", "pos": ABOVE_CUBE, "dur": 3.0},
            {"name": "REACH", "pos": AT_CUBE, "dur": 2.0},
            {"name": "GRASP", "pos": AT_CUBE, "dur": 1.5},
            {"name": "LIFT", "pos": LIFTED, "dur": 2.0},
            {"name": "MOVE", "pos": ABOVE_DROP, "dur": 2.5},
            {"name": "LOWER", "pos": AT_DROP, "dur": 2.0},
            {"name": "RELEASE", "pos": AT_DROP, "dur": 1.5},
            {"name": "RETRACT", "pos": HOME, "dur": 3.0},
            {"name": "DONE", "pos": HOME, "dur": 999.0},
        ]

        self.current_state = 0
        self.state_timer = 0.0

    def reset(self):
        self.current_state = 0
        self.state_timer = 0.0
        print(f"\n[State] → {self.states[0]['name']}")

    def step(self):
        self.state_timer += self.dt
        if self.state_timer >= self.states[self.current_state]["dur"]:
            if self.current_state < len(self.states) - 1:
                self.current_state += 1
                self.state_timer = 0.0
                print(f"\n[State] → {self.states[self.current_state]['name']}")

    def get_target(self):
        state = self.states[self.current_state]
        return (
            torch.tensor([state["pos"]], device=self.device),
            state["name"]
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    sim_dt = 0.005
    decimation = 4
    control_dt = sim_dt * decimation

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    scene_cfg = G1LocomanipSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    robot: Articulation = scene["robot"]

    sim.reset()
    scene.reset()
    print("[INFO] Simulation reset.\n")

    # Print joint info
    print(f"[INFO] Robot has {len(robot.joint_names)} joints")

    # Simple standing controller
    standing_controller = SimpleStandingController(robot, args_cli.device)

    # DiffIK for arm
    arm_joint_names = [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    arm_joint_ids = [robot.joint_names.index(n) for n in arm_joint_names]

    ee_name = "right_wrist_yaw_link"
    ee_body_id = robot.body_names.index(ee_name)
    jacobian_ee_idx = ee_body_id - 1

    print(f"\n[INFO] Arm joints: {len(arm_joint_ids)}")
    print(f"[INFO] EE: {ee_name} (body={ee_body_id}, jac_idx={jacobian_ee_idx})")

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1}
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=args_cli.num_envs, device=args_cli.device)

    # Markers
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/ee"))
    goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/goal"))

    state_machine = PickPlaceStateMachine(control_dt, args_cli.device)
    state_machine.reset()

    print("\n[INFO] Starting simulation...")
    print("[INFO] Lower body: PD Standing | Upper body: DiffIK\n")

    step_count = 0
    max_cycles = 2
    cycle_count = 0

    # Track stability
    stable_count = 0
    unstable_count = 0

    while simulation_app.is_running():
        target_pos, state_name = state_machine.get_target()
        state_machine.step()

        # Lower body + waist - Simple standing controller
        lower_targets, waist_targets = standing_controller.get_joint_targets()
        robot.set_joint_position_target(lower_targets, joint_ids=standing_controller.lower_body_joint_ids)
        robot.set_joint_position_target(waist_targets, joint_ids=standing_controller.waist_joint_ids)

        # Upper body - DiffIK
        ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w

        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos, root_quat,
            target_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args_cli.device)
        )

        diff_ik.reset()
        diff_ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_ee_idx, :, :]
        arm_jac = jacobian[:, :, arm_joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pos_w, ee_quat_w)
        arm_pos = robot.data.joint_pos[:, arm_joint_ids]

        arm_targets = diff_ik.compute(ee_pos_b, ee_quat_b, arm_jac, arm_pos)
        robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)

        # Step simulation
        robot.write_data_to_sim()
        scene.write_data_to_sim()

        for _ in range(decimation):
            sim.step(render=False)

        scene.update(control_dt)
        sim.render()

        ee_marker.visualize(ee_pos_w, ee_quat_w)
        goal_marker.visualize(target_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args_cli.device))

        # Check stability
        base_z = robot.data.root_pos_w[0, 2].item()
        if base_z > 0.6:
            stable_count += 1
        else:
            unstable_count += 1

        step_count += 1
        if step_count % 50 == 0:
            error = torch.norm(ee_pos_w - target_pos, dim=-1).item()
            stability = "✓" if base_z > 0.6 else "✗"
            print(f"[{step_count:4d}] {state_name:10s} | Error: {error:.4f}m | "
                  f"Base Z: {base_z:.3f}m {stability}")

        if state_name == "DONE" and state_machine.state_timer > 3.0:
            cycle_count += 1
            if cycle_count >= max_cycles:
                print(f"\n[INFO] Completed {max_cycles} cycles.")
                print(f"[INFO] Stability: {stable_count}/{stable_count + unstable_count} steps stable")
                break
            print(f"\n[INFO] Resetting (cycle {cycle_count})...")
            state_machine.reset()

    print("\n" + "=" * 70)
    print("  V7 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()