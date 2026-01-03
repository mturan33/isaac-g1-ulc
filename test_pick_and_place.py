# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V6 (Correct Joint Ordering)
# Key fix: Joint order in observations must match Agile training order

"""
G1 Locomanipulation Demo V6
- CRITICAL FIX: Joint ordering in observations matches Agile training
- The Agile policy expects joints in pattern-group order, not articulation order

Joint order expected by Agile:
1. Shoulder joints (6): left/right × pitch/roll/yaw
2. Elbow joints (2): left/right
3. Wrist joints (6): left/right × roll/pitch/yaw
4. Hip joints (6): left/right × pitch/roll/yaw
5. Knee joints (2): left/right
6. Ankle joints (4): left/right × pitch/roll
7. Waist joints (3): yaw/roll/pitch

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v6.py
"""

import argparse
import re
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Locomanipulation V6")
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.io.torchscript import load_torchscript_model

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

print("\n" + "=" * 70)
print("  G1 Locomanipulation Demo - V6")
print("  CRITICAL FIX: Correct joint ordering for Agile policy")
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

    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={
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
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
            },
        ),
    )


# ============================================================================
# AGILE LOCOMOTION POLICY (CORRECT JOINT ORDERING)
# ============================================================================

class AgileLocomotionPolicy:
    """Agile policy wrapper with CORRECT joint ordering.

    The key insight: Agile policy expects joints in the ORDER the regex patterns
    are listed, NOT the order they appear in the articulation.

    Pattern order from AgileTeacherPolicyObservationsCfg:
    1. .*_shoulder_.*_joint  (6 joints)
    2. .*_elbow_joint        (2 joints)
    3. .*_wrist_.*_joint     (6 joints)
    4. .*_hip_.*_joint       (6 joints)
    5. .*_knee_joint         (2 joints)
    6. .*_ankle_.*_joint     (4 joints)
    7. waist_.*_joint        (3 joints)
    """

    # EXACT joint patterns in the ORDER Agile expects
    AGILE_OBS_JOINT_PATTERNS = [
        r".*_shoulder_.*_joint",
        r".*_elbow_joint",
        r".*_wrist_.*_joint",
        r".*_hip_.*_joint",
        r".*_knee_joint",
        r".*_ankle_.*_joint",
        r"waist_.*_joint",
    ]

    def __init__(self, robot: Articulation, device: str):
        self.robot = robot
        self.device = device
        self.num_envs = robot.num_instances

        # Find joints in PATTERN ORDER (not articulation order!)
        self.obs_joint_ids = []
        self.obs_joint_names = []

        all_joint_names = list(robot.joint_names)

        # Process each pattern in order
        for pattern in self.AGILE_OBS_JOINT_PATTERNS:
            for i, name in enumerate(all_joint_names):
                if re.match(pattern, name) and i not in self.obs_joint_ids:
                    self.obs_joint_ids.append(i)
                    self.obs_joint_names.append(name)

        print(f"[INFO] Observation joints ({len(self.obs_joint_ids)}) in pattern order:")
        for i, (idx, name) in enumerate(zip(self.obs_joint_ids, self.obs_joint_names)):
            print(f"  [{i:2d}] idx={idx:2d} {name}")

        # Lower body joints for control output (12 joints)
        self.lower_body_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]

        self.lower_body_joint_ids = []
        for name in self.lower_body_joint_names:
            if name in all_joint_names:
                self.lower_body_joint_ids.append(all_joint_names.index(name))

        print(f"\n[INFO] Lower body control joints: {len(self.lower_body_joint_ids)}")

        # Load Agile policy
        policy_path = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
        try:
            resolved_path = retrieve_file_path(policy_path)
            self.policy = load_torchscript_model(resolved_path, device=device)
            print(f"[INFO] Loaded Agile policy from: {resolved_path}")
            self._policy_loaded = True
        except Exception as e:
            print(f"[WARNING] Could not load Agile policy: {e}")
            self._policy_loaded = False

        # Default positions
        self.default_obs_joint_pos = robot.data.default_joint_pos[:, self.obs_joint_ids].clone()
        self.default_lower_body_pos = robot.data.default_joint_pos[:, self.lower_body_joint_ids].clone()

        # Policy parameters
        self.policy_output_scale = 0.25
        self.last_action = torch.zeros(self.num_envs, 12, device=device)

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse of quaternion."""
        q_w = q[:, 0:1]
        q_vec = q[:, 1:4]
        a = v * (2.0 * q_w * q_w - 1.0)
        b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * (torch.sum(q_vec * v, dim=-1, keepdim=True)) * 2.0
        return a - b + c

    def compute_observations(self) -> torch.Tensor:
        """Compute observations in CORRECT order for Agile policy."""

        # Base velocities
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b

        # Projected gravity
        gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).repeat(self.num_envs, 1)
        root_quat = self.robot.data.root_quat_w
        projected_gravity = self._quat_rotate_inverse(root_quat, gravity_vec)

        # Joint positions and velocities - IN PATTERN ORDER!
        joint_pos = self.robot.data.joint_pos[:, self.obs_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.obs_joint_ids] * 0.1
        joint_pos_rel = joint_pos - self.default_obs_joint_pos

        # Concatenate: [lin_vel(3), ang_vel(3), gravity(3), pos(29), vel(29), action(12)] = 79
        obs = torch.cat([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos_rel,
            joint_vel,
            self.last_action,
        ], dim=-1)

        return obs

    def get_joint_targets(self, command: torch.Tensor = None) -> torch.Tensor:
        """Get lower body joint targets from Agile policy."""

        if not self._policy_loaded:
            return self.default_lower_body_pos

        if command is None:
            command = torch.zeros(self.num_envs, 4, device=self.device)

        try:
            obs = self.compute_observations()

            # Policy input: [command(4), obs(79)] = 83
            policy_input = torch.cat([command, obs], dim=-1)

            if not hasattr(self, '_debug_done'):
                print(f"\n[DEBUG] Observation shape: {obs.shape}")
                print(f"[DEBUG] Policy input shape: {policy_input.shape}")
                self._debug_done = True

            with torch.no_grad():
                joint_actions = self.policy.forward(policy_input)

            # Scale and offset
            joint_targets = joint_actions * self.policy_output_scale + self.default_lower_body_pos
            self.last_action = joint_actions.clone()

            return joint_targets

        except Exception as e:
            if not hasattr(self, '_error_done'):
                print(f"[WARNING] Policy failed: {e}")
                self._error_done = True
            return self.default_lower_body_pos


# ============================================================================
# STATE MACHINE
# ============================================================================

class PickPlaceStateMachine:
    def __init__(self, dt: float, device: str):
        self.dt = dt
        self.device = device

        HOME = [0.15, 0.25, 0.85]
        ABOVE_CUBE = [0.0, 0.35, 0.85]
        AT_CUBE = [0.0, 0.38, 0.78]
        LIFTED = [0.0, 0.35, 0.90]
        ABOVE_DROP = [0.12, 0.35, 0.90]
        AT_DROP = [0.12, 0.38, 0.82]

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

    # Agile policy with correct joint ordering
    agile_policy = AgileLocomotionPolicy(robot, args_cli.device)

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
    print("[INFO] Lower body: Agile (correct order) | Upper body: DiffIK\n")

    step_count = 0
    max_cycles = 2
    cycle_count = 0

    while simulation_app.is_running():
        target_pos, state_name = state_machine.get_target()
        state_machine.step()

        # Lower body - Agile policy
        command = torch.zeros(args_cli.num_envs, 4, device=args_cli.device)
        lower_targets = agile_policy.get_joint_targets(command)
        robot.set_joint_position_target(lower_targets, joint_ids=agile_policy.lower_body_joint_ids)

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

        # Step
        robot.write_data_to_sim()
        scene.write_data_to_sim()

        for _ in range(decimation):
            sim.step(render=False)

        scene.update(control_dt)
        sim.render()

        ee_marker.visualize(ee_pos_w, ee_quat_w)
        goal_marker.visualize(target_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args_cli.device))

        step_count += 1
        if step_count % 50 == 0:
            error = torch.norm(ee_pos_w - target_pos, dim=-1).item()
            base_z = robot.data.root_pos_w[0, 2].item()
            print(f"[{step_count:4d}] {state_name:10s} | Error: {error:.4f}m | Base Z: {base_z:.3f}m")

        if state_name == "DONE" and state_machine.state_timer > 3.0:
            cycle_count += 1
            if cycle_count >= max_cycles:
                print(f"\n[INFO] Completed {max_cycles} cycles.")
                break
            print(f"\n[INFO] Resetting (cycle {cycle_count})...")
            state_machine.reset()

    print("\n" + "=" * 70)
    print("  V6 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()