# Copyright (c) 2025, VLM-RL G1 Project
# G1 Hierarchical Control: PPO + WORKING DifferentialIK
#
# KEY FIXES:
# 1. command_type="position" (not "pose") - we only care about 3D position
# 2. Realistic workspace targets - within G1 arm reach
# 3. Better IK gains and convergence

"""
G1 Hierarchical Control with WORKING IK
========================================

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\g1_hierarchical_ik_working.py --num_envs 4 --load_run 2025-12-27_00-29-54
"""

import argparse
import os
import math
import torch
import torch.nn as nn
from typing import List

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Hierarchical Control: PPO + Working IK")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--ik_method", type=str, default="dls", choices=["dls", "pinv", "svd", "trans"])
parser.add_argument("--target_mode", type=str, default="static",
                    choices=["circle", "static", "wave", "front_reach"])
parser.add_argument("--arm", type=str, default="right", choices=["left", "right"])
parser.add_argument("--debug", action="store_true", default=True)
parser.add_argument("--max_joint_delta", type=float, default=0.15, help="Max joint change per step")
parser.add_argument("--ik_gain", type=float, default=1.0, help="IK gain multiplier")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_conjugate
from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks  # noqa: F401

##############################################################################
# G1 ARM CONFIGURATION
##############################################################################

G1_ARM_JOINTS = {
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ],
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
    ],
}

G1_EE_BODIES = {
    "right": "right_palm_link",
    "left": "left_palm_link",
}

ARM_JOINT_INDICES = {
    "right": [6, 10, 14, 18, 22],
    "left": [5, 9, 13, 17, 21],
}

# G1 Joint Limits
G1_ARM_JOINT_LIMITS = {
    "right": {
        "shoulder_pitch": [-3.1, 2.6],
        "shoulder_roll": [-1.6, 2.6],
        "shoulder_yaw": [-2.6, 2.6],
        "elbow_pitch": [-1.6, 1.6],
        "elbow_roll": [-1.6, 1.6],
    },
}

# Default arm pose
G1_ARM_DEFAULT_POS = {
    "right": [0.35, -0.16, 0.0, 0.87, 0.0],
    "left": [0.35, 0.16, 0.0, 0.87, 0.0],
}


##############################################################################
# CUSTOM ACTOR NETWORK
##############################################################################

class CustomActorCritic(nn.Module):
    def __init__(self, num_obs: int, num_actions: int,
                 actor_hidden_dims: List[int] = [256, 128, 128],
                 activation: str = "elu"):
        super().__init__()
        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()

        actor_layers = []
        prev_dim = num_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(prev_dim, hidden_dim), act_fn])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)
        self.std = nn.Parameter(torch.ones(num_actions))

    def act_inference(self, obs):
        with torch.no_grad():
            return self.actor(obs)

    def load_rsl_rl_checkpoint(self, checkpoint_path: str, device: str):
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = data["model_state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("actor."):
                new_state_dict[key] = value
            elif key == "std":
                new_state_dict["std"] = value
            elif key == "log_std":
                new_state_dict["std"] = torch.exp(value)
        self.load_state_dict(new_state_dict, strict=False)
        return True


def find_checkpoint(run_dir: str, checkpoint_name: str = None) -> str:
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if checkpoint_name:
        path = os.path.join(run_dir, checkpoint_name)
        if os.path.exists(path):
            return path
    checkpoints = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return os.path.join(run_dir, checkpoints[-1])


##############################################################################
# WORKING ARM IK CONTROLLER
##############################################################################

class G1ArmIKControllerWorking:
    """
    DifferentialIK controller that actually works!

    KEY FIXES:
    1. command_type="position" - only control 3D position, ignore orientation
    2. Proper workspace - targets within arm reach
    3. Better IK params - higher gain, less damping for faster convergence
    """

    def __init__(
            self,
            num_envs: int,
            device: str,
            arm: str = "right",
            ik_method: str = "dls",
            max_joint_delta: float = 0.15,
            ik_gain: float = 1.0,
            debug: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm
        self.debug = debug
        self.max_joint_delta = max_joint_delta
        self.ik_gain = ik_gain

        # KEY FIX 1: Use "position" not "pose"!
        # "pose" = 7D (position + quaternion) - causes issues when we only care about position
        # "position" = 3D position only
        self.ik_cfg = DifferentialIKControllerCfg(
            command_type="position",  # <-- CRITICAL FIX!
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params={"lambda_val": 0.05} if ik_method == "dls" else {"k_val": 1.0},
        )

        self.controller = DifferentialIKController(
            self.ik_cfg,
            num_envs=num_envs,
            device=device,
        )

        # Indices
        self.ee_body_idx = None
        self.arm_joint_ids = None
        self.jacobian_col_ids = None
        self.ee_jacobi_idx = None

        # Joint limits
        limits = G1_ARM_JOINT_LIMITS["right"]
        self.joint_lower = torch.tensor([
            limits["shoulder_pitch"][0],
            limits["shoulder_roll"][0],
            limits["shoulder_yaw"][0],
            limits["elbow_pitch"][0],
            limits["elbow_roll"][0],
        ], device=device)
        self.joint_upper = torch.tensor([
            limits["shoulder_pitch"][1],
            limits["shoulder_roll"][1],
            limits["shoulder_yaw"][1],
            limits["elbow_pitch"][1],
            limits["elbow_roll"][1],
        ], device=device)

        # Target (position only for "position" mode)
        self.target_pos = torch.zeros(num_envs, 3, device=device)

        # Previous joint positions for smoothing
        self.prev_joint_pos = None

        self.initialized = False
        self.step_count = 0

        print(f"\n{'=' * 60}")
        print(f"[IK WORKING] Position-only DifferentialIK")
        print(f"[IK WORKING] Arm: {arm}, Method: {ik_method}")
        print(f"[IK WORKING] Max delta: {max_joint_delta} rad/step")
        print(f"[IK WORKING] IK gain: {ik_gain}")
        print(f"{'=' * 60}")

    def initialize_from_robot(self, robot, scene):
        """Initialize indices from robot."""
        print(f"\n[IK INIT] Initializing...")

        try:
            # Print ALL joint names for debugging
            joint_names = robot.joint_names if hasattr(robot, 'joint_names') else []
            print(f"[IK INIT] Total joints: {len(joint_names)}")
            print(f"[IK INIT] All joint names:")
            for i, name in enumerate(joint_names):
                marker = " <-- ARM?" if "shoulder" in name or "elbow" in name else ""
                print(f"  [{i:2d}] {name}{marker}")

            body_names = robot.body_names if hasattr(robot, 'body_names') else []
            ee_name = G1_EE_BODIES[self.arm]

            if ee_name in body_names:
                self.ee_body_idx = body_names.index(ee_name)
            else:
                self.ee_body_idx = 29 if self.arm == "right" else 28

            print(f"[IK INIT] EE body index: {self.ee_body_idx}")

            # Floating-base: use body index directly
            self.ee_jacobi_idx = self.ee_body_idx

            joint_names = robot.joint_names if hasattr(robot, 'joint_names') else []
            self.arm_joint_ids = []

            # Try exact match first
            for jname in G1_ARM_JOINTS[self.arm]:
                if jname in joint_names:
                    self.arm_joint_ids.append(joint_names.index(jname))

            # If not found, try searching by pattern
            if len(self.arm_joint_ids) < 5:
                print(f"[IK INIT] Exact match failed, searching by pattern...")
                self.arm_joint_ids = []

                # Search patterns for right arm
                if self.arm == "right":
                    patterns = ["right_shoulder_pitch", "right_shoulder_roll",
                                "right_shoulder_yaw", "right_elbow_pitch", "right_elbow_roll"]
                else:
                    patterns = ["left_shoulder_pitch", "left_shoulder_roll",
                                "left_shoulder_yaw", "left_elbow_pitch", "left_elbow_roll"]

                for pattern in patterns:
                    for i, jname in enumerate(joint_names):
                        if pattern in jname.lower().replace("_joint", "").replace("-", "_"):
                            self.arm_joint_ids.append(i)
                            print(f"[IK INIT] Found: {jname} -> index {i}")
                            break

            if len(self.arm_joint_ids) < 5:
                print(f"[IK INIT] WARNING: Only found {len(self.arm_joint_ids)} joints, using fallback!")
                self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]

            # +6 offset for floating-base Jacobian
            self.jacobian_col_ids = [idx + 6 for idx in self.arm_joint_ids]

            print(f"[IK INIT] Arm joint indices: {self.arm_joint_ids}")
            print(f"[IK INIT] Jacobian columns: {self.jacobian_col_ids}")

            # Initialize prev joint pos
            self.prev_joint_pos = robot.data.joint_pos[:, self.arm_joint_ids].clone()

            # Verify Jacobian
            full_jac = robot.root_physx_view.get_jacobians()
            arm_jac = full_jac[:, self.ee_jacobi_idx, :3, :][:, :, self.jacobian_col_ids]  # Only position rows (0:3)
            jac_norm = torch.norm(arm_jac).item()
            print(f"[IK INIT] Position Jacobian norm: {jac_norm:.4f}")

            # Print initial EE position
            ee_pos_w = robot.data.body_state_w[:, self.ee_body_idx, 0:3]
            root_pos_w = robot.data.root_state_w[:, 0:3]
            ee_pos_b = ee_pos_w - root_pos_w  # Simple approximation
            print(
                f"[IK INIT] Initial EE pos (base): [{ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}]")

            self.initialized = True
            print(f"[IK INIT] ✓ Initialization complete!")

        except Exception as e:
            print(f"[IK INIT] ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.ee_body_idx = 29
            self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]
            self.jacobian_col_ids = [idx + 6 for idx in self.arm_joint_ids]
            self.ee_jacobi_idx = 29
            self.initialized = True

    def set_target(self, target_pos: torch.Tensor, ee_quat: torch.Tensor | None = None):
        """Set target position in base frame (3D only).

        Args:
            target_pos: Target position in base frame (N, 3)
            ee_quat: Current end-effector quaternion (N, 4) - required for position mode
        """
        self.target_pos = target_pos.clone()
        # For position mode:
        # - command = 3D position
        # - ee_quat = current EE orientation (required even though we only care about position)
        if ee_quat is None:
            # Use identity quaternion if not provided
            ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
            ee_quat[:, 0] = 1.0  # [w, x, y, z]
        self.controller.set_command(target_pos, ee_quat=ee_quat)

    def _transform_jacobian_to_base_frame(self, jacobian_w: torch.Tensor,
                                          root_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform Jacobian from world to base frame."""
        root_quat_conj = quat_conjugate(root_quat_w)
        rot_matrix = matrix_from_quat(root_quat_conj)

        # Transform both linear (0:3) and angular (3:6) parts
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(rot_matrix, jacobian_w[:, 3:, :])
        return jacobian_b

    def compute(self, robot) -> torch.Tensor:
        """Compute IK solution."""
        if not self.initialized:
            return torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

        self.step_count += 1
        debug_this_step = self.debug and (self.step_count % 50 == 1)

        try:
            # Current joint positions
            current_joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]

            # EE pose in world frame
            ee_pos_w = robot.data.body_state_w[:, self.ee_body_idx, 0:3]
            ee_quat_w = robot.data.body_state_w[:, self.ee_body_idx, 3:7]

            # Root pose
            root_pos_w = robot.data.root_state_w[:, 0:3]
            root_quat_w = robot.data.root_state_w[:, 3:7]

            # Transform EE to base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            # Get Jacobian - full 6 rows (position + orientation)
            # Even for position mode, controller may need full Jacobian
            full_jacobian = robot.root_physx_view.get_jacobians()
            jacobian_w = full_jacobian[:, self.ee_jacobi_idx, :, :]  # All 6 rows
            jacobian_arm_w = jacobian_w[:, :, self.jacobian_col_ids]

            # Transform to base frame
            jacobian_arm_b = self._transform_jacobian_to_base_frame(jacobian_arm_w, root_quat_w)

            # Compute IK - for position mode, ee_quat_b is ignored
            joint_pos_ik = self.controller.compute(ee_pos_b, ee_quat_b, jacobian_arm_b, current_joint_pos)

            # Calculate delta
            raw_delta = joint_pos_ik - current_joint_pos

            # Apply gain
            scaled_delta = raw_delta * self.ik_gain

            # Clamp delta
            clamped_delta = torch.clamp(scaled_delta, -self.max_joint_delta, self.max_joint_delta)

            # Apply delta
            new_joint_pos = current_joint_pos + clamped_delta

            # Enforce joint limits
            new_joint_pos = torch.clamp(new_joint_pos, self.joint_lower, self.joint_upper)

            # Debug
            if debug_this_step:
                pos_error = self.target_pos - ee_pos_b
                error_mag = torch.norm(pos_error, dim=1)

                print(f"\n{'=' * 60}")
                print(f"[IK WORKING] Step {self.step_count}")
                print(f"{'=' * 60}")
                print(f"[TARGET]")
                print(
                    f"  Target:    [{self.target_pos[0, 0]:.3f}, {self.target_pos[0, 1]:.3f}, {self.target_pos[0, 2]:.3f}]")
                print(f"  Current:   [{ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}]")
                print(f"  Error:     {error_mag[0]:.4f} m")
                print(f"  Direction: [{pos_error[0, 0]:.3f}, {pos_error[0, 1]:.3f}, {pos_error[0, 2]:.3f}]")

                print(f"\n[JACOBIAN]")
                print(f"  Shape: {jacobian_arm_b.shape}")
                print(f"  Norm:  {torch.norm(jacobian_arm_b[0]):.4f}")

                print(f"\n[JOINT CONTROL]")
                print(
                    f"  Current:    [{current_joint_pos[0, 0]:.3f}, {current_joint_pos[0, 1]:.3f}, {current_joint_pos[0, 2]:.3f}, {current_joint_pos[0, 3]:.3f}, {current_joint_pos[0, 4]:.3f}]")
                print(
                    f"  IK output:  [{joint_pos_ik[0, 0]:.3f}, {joint_pos_ik[0, 1]:.3f}, {joint_pos_ik[0, 2]:.3f}, {joint_pos_ik[0, 3]:.3f}, {joint_pos_ik[0, 4]:.3f}]")
                print(
                    f"  Raw delta:  [{raw_delta[0, 0]:.4f}, {raw_delta[0, 1]:.4f}, {raw_delta[0, 2]:.4f}, {raw_delta[0, 3]:.4f}, {raw_delta[0, 4]:.4f}]")
                print(
                    f"  Final:      [{new_joint_pos[0, 0]:.3f}, {new_joint_pos[0, 1]:.3f}, {new_joint_pos[0, 2]:.3f}, {new_joint_pos[0, 3]:.3f}, {new_joint_pos[0, 4]:.3f}]")

                # Check if error is decreasing
                if hasattr(self, 'prev_error'):
                    if error_mag[0] < self.prev_error:
                        print(f"\n  ✓ Error DECREASING ({self.prev_error:.4f} → {error_mag[0]:.4f})")
                    else:
                        print(f"\n  ⚠️ Error NOT decreasing ({self.prev_error:.4f} → {error_mag[0]:.4f})")
                self.prev_error = error_mag[0].item()

                print(f"{'=' * 60}\n")

            # Update prev
            self.prev_joint_pos = new_joint_pos.clone()

            return new_joint_pos

        except Exception as e:
            print(f"[IK ERROR] {e}")
            import traceback
            traceback.print_exc()
            return robot.data.joint_pos[:, self.arm_joint_ids]

    def get_ee_pos_base(self, robot) -> torch.Tensor:
        """Get EE position in base frame."""
        ee_pos_w = robot.data.body_state_w[:, self.ee_body_idx, 0:3]
        ee_quat_w = robot.data.body_state_w[:, self.ee_body_idx, 3:7]
        root_pos_w = robot.data.root_state_w[:, 0:3]
        root_quat_w = robot.data.root_state_w[:, 3:7]
        ee_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        return ee_pos_b

    def reset(self, env_ids: torch.Tensor = None):
        """Reset controller."""
        if env_ids is None:
            self.target_pos.zero_()
            if self.prev_joint_pos is not None:
                default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)
                self.prev_joint_pos = default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        else:
            self.target_pos[env_ids] = 0.0
            if self.prev_joint_pos is not None:
                default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)
                self.prev_joint_pos[env_ids] = default_pos

        self.controller.reset(env_ids)


##############################################################################
# TARGET GENERATOR - Realistic workspace targets
##############################################################################

class TargetGenerator:
    """Generate targets within G1's reachable workspace."""

    def __init__(self, num_envs: int, device: str, mode: str = "static", arm: str = "right"):
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.arm = arm
        self.initialized = False
        self.initial_ee_pos = None

        # Movement parameters
        self.radius = 0.05  # Small radius for stability
        self.freq = 0.1  # Slow movement

    def initialize_from_ee(self, ee_pos: torch.Tensor):
        """Initialize target from current EE position."""
        self.initial_ee_pos = ee_pos.clone()
        self.initialized = True
        print(f"[TARGET] Initialized from EE: [{ee_pos[0, 0]:.3f}, {ee_pos[0, 1]:.3f}, {ee_pos[0, 2]:.3f}]")

    def get_target(self, time: float) -> torch.Tensor:
        if not self.initialized:
            # Return a safe default until initialized
            default = torch.tensor([0.1, -0.2, 0.0], device=self.device)
            return default.unsqueeze(0).expand(self.num_envs, -1).clone()

        pos = self.initial_ee_pos.clone()

        if self.mode == "circle":
            angle = 2 * math.pi * self.freq * time
            pos[:, 0] += self.radius * math.cos(angle)
            pos[:, 2] += self.radius * math.sin(angle)
        elif self.mode == "wave":
            wave = math.sin(2 * math.pi * self.freq * time)
            pos[:, 2] += wave * self.radius
        elif self.mode == "front_reach":
            # Gradually reach forward
            t = min(time / 10.0, 1.0)
            pos[:, 0] += t * 0.1  # Reach 10cm forward
        # static: just initial position (hold still)

        return pos


##############################################################################
# MAIN
##############################################################################

def main():
    print("\n" + "=" * 70)
    print("  G1 Hierarchical Control - WORKING IK")
    print("  ")
    print("  KEY FIXES:")
    print("  1. command_type='position' (not 'pose')")
    print("  2. Realistic workspace targets")
    print("  3. Better IK convergence")
    print("=" * 70 + "\n")

    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.action_manager.total_action_dim

    print(f"[Env] Obs: {obs_dim}, Actions: {action_dim}")

    robot = None
    scene = env.scene
    if hasattr(scene, 'articulations') and 'robot' in scene.articulations:
        robot = scene.articulations['robot']
        print(f"[Env] ✓ Robot found")

    # Load locomotion policy
    policy = None
    try:
        run_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
        checkpoint_path = find_checkpoint(run_dir, args_cli.checkpoint)
        print(f"[Policy] Loading: {checkpoint_path}")

        policy = CustomActorCritic(
            num_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[256, 128, 128],
        ).to(env.device)

        policy.load_rsl_rl_checkpoint(checkpoint_path, env.device)
        policy.eval()
        print("[Policy] ✓ Loaded!")

    except Exception as e:
        print(f"[Policy] ✗ Error: {e}")
        policy = None

    # Create working IK controller
    arm_ik = G1ArmIKControllerWorking(
        env.num_envs, env.device,
        arm=args_cli.arm,
        ik_method=args_cli.ik_method,
        max_joint_delta=args_cli.max_joint_delta,
        ik_gain=args_cli.ik_gain,
        debug=args_cli.debug
    )
    if robot is not None:
        arm_ik.initialize_from_robot(robot, scene)

    # Target generator with realistic workspace
    target_gen = TargetGenerator(env.num_envs, env.device, args_cli.target_mode, arm=args_cli.arm)

    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)

    print("\n" + "-" * 70)
    print(f"[Info] Starting simulation...")
    print(f"[Info] Target mode: {args_cli.target_mode}")
    print("-" * 70 + "\n")

    sim_time = 0.0
    dt = 0.02
    step_count = 0

    # Track convergence
    error_history = []
    min_error = float('inf')

    # Initialize target from current EE position after first step
    target_initialized = False

    try:
        while simulation_app.is_running():
            # Initialize target generator from current EE pos
            if not target_initialized and robot is not None and arm_ik.initialized:
                ee_pos_b = arm_ik.get_ee_pos_base(robot)
                target_gen.initialize_from_ee(ee_pos_b)
                target_initialized = True

            # Get target
            target_pos = target_gen.get_target(sim_time)

            # Lower body: PPO
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs)
            else:
                actions.zero_()

            # Upper body: IK
            if robot is not None and arm_ik.initialized:
                # Get current EE quaternion in base frame
                ee_quat_w = robot.data.body_state_w[:, arm_ik.ee_body_idx, 3:7]
                root_quat_w = robot.data.root_state_w[:, 3:7]
                # Transform to base frame (simplified - just pass world frame quat)
                # The controller uses this for display, so approximation is OK
                ee_quat_b = ee_quat_w  # Simplified

                arm_ik.set_target(target_pos, ee_quat=ee_quat_b)
                joint_pos_des = arm_ik.compute(robot)

                # Apply to actions
                for i, joint_idx in enumerate(arm_ik.arm_joint_ids):
                    actions[:, joint_idx] = joint_pos_des[:, i]

            # Step
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]

            # Handle resets
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                arm_ik.reset(reset_ids)

            sim_time += dt
            step_count += 1

            # Logging
            if step_count % 100 == 0:
                mean_reward = rewards.mean().item()
                alive = (~terminated).float().mean().item() * 100

                if robot is not None:
                    ee_pos_b = arm_ik.get_ee_pos_base(robot)[0]
                    target_b = target_pos[0]
                    error = torch.norm(target_b - ee_pos_b).item()
                    error_history.append(error)

                    if error < min_error:
                        min_error = error

                    # Trend
                    if len(error_history) > 3:
                        recent = sum(error_history[-3:]) / 3
                        older = sum(error_history[-6:-3]) / 3 if len(error_history) > 6 else recent
                        trend = "↓" if recent < older - 0.01 else "↑" if recent > older + 0.01 else "→"
                    else:
                        trend = "→"

                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                          f"Error: {error:.3f}m {trend} (min: {min_error:.3f}m)")
                else:
                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}%")

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user")

        if error_history:
            print(f"\n[Summary]")
            print(f"  Initial error: {error_history[0]:.4f}m")
            print(f"  Final error:   {error_history[-1]:.4f}m")
            print(f"  Min error:     {min_error:.4f}m")
            print(f"  Improvement:   {(error_history[0] - min_error) / error_history[0] * 100:.1f}%")

    finally:
        env.close()
        print("[Info] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()