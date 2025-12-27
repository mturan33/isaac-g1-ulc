# Copyright (c) 2025, VLM-RL G1 Project
# G1 Hierarchical Control: PPO Locomotion + DifferentialIK (FIXED + SMOOTHED)
#
# FIXES APPLIED:
# 1. Jacobian column index: +6 offset for floating-base (PR #1033)
# 2. Body index: Use ee_body_idx directly for floating-base
# 3. Jacobian frame transform: World -> Base frame (PR #967)
# 4. NEW: Joint change clamping to prevent wild movements
# 5. NEW: Joint limit enforcement
# 6. NEW: Exponential smoothing for stable control

"""
G1 Hierarchical Control with STABLE DifferentialIK
===================================================

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\g1_hierarchical_ik_stable.py --num_envs 4 --load_run 2025-12-27_00-29-54
"""

import argparse
import os
import math
import torch
import torch.nn as nn
from typing import List

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Hierarchical Control: PPO + Stable IK")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--ik_method", type=str, default="dls", choices=["dls", "pinv", "svd", "trans"])
parser.add_argument("--target_mode", type=str, default="static",
                    choices=["circle", "static", "wave", "reach"])
parser.add_argument("--arm", type=str, default="right", choices=["left", "right"])
parser.add_argument("--debug", action="store_true", default=True)
# Stability parameters
parser.add_argument("--max_joint_delta", type=float, default=0.1,
                    help="Max joint change per step (rad)")
parser.add_argument("--smoothing", type=float, default=0.3,
                    help="Smoothing factor (0=no smoothing, 1=full smoothing)")
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

# G1 Joint Limits (from URDF) - approximate values
G1_ARM_JOINT_LIMITS = {
    "right": {
        # [lower, upper] in radians
        "shoulder_pitch": [-3.1, 2.6],  # right_shoulder_pitch_joint
        "shoulder_roll": [-1.6, 2.6],  # right_shoulder_roll_joint
        "shoulder_yaw": [-2.6, 2.6],  # right_shoulder_yaw_joint
        "elbow_pitch": [-1.6, 1.6],  # right_elbow_pitch_joint
        "elbow_roll": [-1.6, 1.6],  # right_elbow_roll_joint
    },
    "left": {
        "shoulder_pitch": [-3.1, 2.6],
        "shoulder_roll": [-2.6, 1.6],
        "shoulder_yaw": [-2.6, 2.6],
        "elbow_pitch": [-1.6, 1.6],
        "elbow_roll": [-1.6, 1.6],
    },
}

# Default arm pose (comfortable position)
G1_ARM_DEFAULT_POS = {
    "right": [0.35, -0.16, 0.0, 0.87, 0.0],  # From observation
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
# STABLE ARM IK CONTROLLER
##############################################################################

class G1ArmIKControllerStable:
    """
    DifferentialIK controller with stability improvements:
    - Joint change clamping
    - Joint limit enforcement
    - Exponential smoothing
    - Gradual convergence
    """

    def __init__(
            self,
            num_envs: int,
            device: str,
            arm: str = "right",
            ik_method: str = "dls",
            max_joint_delta: float = 0.1,  # Max change per step
            smoothing_factor: float = 0.3,  # 0=instant, 1=no change
            debug: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm
        self.debug = debug
        self.max_joint_delta = max_joint_delta
        self.smoothing_factor = smoothing_factor

        # IK Controller with higher damping for stability
        self.ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params={"lambda_val": 0.15} if ik_method == "dls" else {"k_val": 0.5},
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
        limits = G1_ARM_JOINT_LIMITS[arm]
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

        # Target pose
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_quat = torch.zeros(num_envs, 4, device=device)
        self.target_quat[:, 0] = 1.0

        # Smoothed output (for filtering)
        self.smoothed_joint_pos = None

        self.initialized = False
        self.step_count = 0

        print(f"\n{'=' * 60}")
        print(f"[IK STABLE] G1ArmIKController - STABILITY ENHANCED")
        print(f"[IK STABLE] Arm: {arm}, Method: {ik_method}")
        print(f"[IK STABLE] Max delta: {max_joint_delta} rad/step")
        print(f"[IK STABLE] Smoothing: {smoothing_factor}")
        print(f"{'=' * 60}")

    def initialize_from_robot(self, robot, scene):
        """Initialize indices from robot articulation."""
        print(f"\n[IK INIT] Initializing...")

        try:
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
            for jname in G1_ARM_JOINTS[self.arm]:
                if jname in joint_names:
                    self.arm_joint_ids.append(joint_names.index(jname))

            if len(self.arm_joint_ids) < 5:
                self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]

            # +6 offset for floating-base Jacobian columns
            self.jacobian_col_ids = [idx + 6 for idx in self.arm_joint_ids]

            print(f"[IK INIT] Arm joint indices: {self.arm_joint_ids}")
            print(f"[IK INIT] Jacobian columns (+6): {self.jacobian_col_ids}")

            # Initialize smoothed position from default
            default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)
            self.smoothed_joint_pos = default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()

            # Verify Jacobian
            full_jac = robot.root_physx_view.get_jacobians()
            arm_jac = full_jac[:, self.ee_jacobi_idx, :, :][:, :, self.jacobian_col_ids]
            jac_norm = torch.norm(arm_jac).item()
            print(f"[IK INIT] Jacobian norm: {jac_norm:.4f}")

            if jac_norm < 0.1:
                print(f"[IK INIT] ⚠️ WARNING: Jacobian norm very small!")
            else:
                print(f"[IK INIT] ✓ Jacobian looks valid")

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
            default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)
            self.smoothed_joint_pos = default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
            self.initialized = True

    def set_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor = None):
        """Set target pose in base frame."""
        self.target_pos = target_pos.clone()

        if target_quat is None:
            target_quat = torch.zeros(self.num_envs, 4, device=self.device)
            target_quat[:, 0] = 1.0

        self.target_quat = target_quat.clone()
        pose_command = torch.cat([target_pos, target_quat], dim=-1)
        self.controller.set_command(pose_command)

    def _transform_jacobian_to_base_frame(self, jacobian_w: torch.Tensor,
                                          root_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform Jacobian from world to base frame."""
        root_quat_conj = quat_conjugate(root_quat_w)
        rot_matrix = matrix_from_quat(root_quat_conj)

        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(rot_matrix, jacobian_w[:, 3:, :])

        return jacobian_b

    def _clamp_joint_delta(self, joint_delta: torch.Tensor) -> torch.Tensor:
        """Clamp joint changes to maximum allowed per step."""
        return torch.clamp(joint_delta, -self.max_joint_delta, self.max_joint_delta)

    def _enforce_joint_limits(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Clamp joint positions to valid range."""
        return torch.clamp(joint_pos, self.joint_lower, self.joint_upper)

    def _apply_smoothing(self, new_pos: torch.Tensor, current_pos: torch.Tensor) -> torch.Tensor:
        """Apply exponential smoothing."""
        # smoothed = alpha * current + (1-alpha) * new
        alpha = self.smoothing_factor
        return alpha * current_pos + (1 - alpha) * new_pos

    def compute(self, robot) -> torch.Tensor:
        """Compute stable IK solution."""
        if not self.initialized:
            return torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

        self.step_count += 1
        debug_this_step = self.debug and (self.step_count % 100 == 1)

        try:
            # Get current joint positions
            current_joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]

            # Get EE pose in world frame
            ee_pose_w = robot.data.body_state_w[:, self.ee_body_idx, 0:7]
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            # Get root pose
            root_pose_w = robot.data.root_state_w[:, 0:7]
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]

            # Transform EE to base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            # Get Jacobian with correct indices
            full_jacobian = robot.root_physx_view.get_jacobians()
            jacobian_w = full_jacobian[:, self.ee_jacobi_idx, :, :]
            jacobian_arm_w = jacobian_w[:, :, self.jacobian_col_ids]

            # Transform Jacobian to base frame
            jacobian_arm_b = self._transform_jacobian_to_base_frame(jacobian_arm_w, root_quat_w)

            # Compute raw IK
            joint_pos_ik = self.controller.compute(ee_pos_b, ee_quat_b, jacobian_arm_b, current_joint_pos)

            # Calculate delta
            raw_delta = joint_pos_ik - current_joint_pos

            # STABILITY 1: Clamp delta
            clamped_delta = self._clamp_joint_delta(raw_delta)

            # Apply clamped delta
            new_joint_pos = current_joint_pos + clamped_delta

            # STABILITY 2: Enforce joint limits
            limited_joint_pos = self._enforce_joint_limits(new_joint_pos)

            # STABILITY 3: Apply smoothing
            if self.smoothed_joint_pos is None:
                self.smoothed_joint_pos = limited_joint_pos.clone()
            else:
                self.smoothed_joint_pos = self._apply_smoothing(limited_joint_pos, self.smoothed_joint_pos)

            # Final output
            final_joint_pos = self.smoothed_joint_pos.clone()

            # Debug output
            if debug_this_step:
                pos_error = self.target_pos - ee_pos_b
                error_mag = torch.norm(pos_error, dim=1)
                final_delta = final_joint_pos - current_joint_pos

                print(f"\n{'=' * 60}")
                print(f"[IK STABLE] Step {self.step_count}")
                print(f"{'=' * 60}")
                print(f"[TARGET]")
                print(
                    f"  Target:  [{self.target_pos[0, 0]:.3f}, {self.target_pos[0, 1]:.3f}, {self.target_pos[0, 2]:.3f}]")
                print(f"  Current: [{ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}]")
                print(f"  Error:   {error_mag[0]:.4f} m")

                print(f"\n[JOINT CONTROL]")
                print(
                    f"  Raw IK delta:     [{raw_delta[0, 0]:.4f}, {raw_delta[0, 1]:.4f}, {raw_delta[0, 2]:.4f}, {raw_delta[0, 3]:.4f}, {raw_delta[0, 4]:.4f}]")
                print(
                    f"  Clamped delta:    [{clamped_delta[0, 0]:.4f}, {clamped_delta[0, 1]:.4f}, {clamped_delta[0, 2]:.4f}, {clamped_delta[0, 3]:.4f}, {clamped_delta[0, 4]:.4f}]")
                print(
                    f"  Final delta:      [{final_delta[0, 0]:.4f}, {final_delta[0, 1]:.4f}, {final_delta[0, 2]:.4f}, {final_delta[0, 3]:.4f}, {final_delta[0, 4]:.4f}]")
                print(
                    f"  Final joint pos:  [{final_joint_pos[0, 0]:.3f}, {final_joint_pos[0, 1]:.3f}, {final_joint_pos[0, 2]:.3f}, {final_joint_pos[0, 3]:.3f}, {final_joint_pos[0, 4]:.3f}]")

                raw_norm = torch.norm(raw_delta[0]).item()
                final_norm = torch.norm(final_delta[0]).item()
                print(f"\n[STABILITY]")
                print(f"  Raw delta norm:   {raw_norm:.4f}")
                print(f"  Final delta norm: {final_norm:.4f}")
                print(f"  Reduction:        {(1 - final_norm / max(raw_norm, 1e-6)) * 100:.1f}%")
                print(f"{'=' * 60}\n")

            return final_joint_pos

        except Exception as e:
            print(f"[IK ERROR] {e}")
            import traceback
            traceback.print_exc()
            return robot.data.joint_pos[:, self.arm_joint_ids]

    def get_ee_pos_base(self, robot) -> torch.Tensor:
        """Get EE position in base frame."""
        ee_pos_w = robot.data.body_state_w[:, self.ee_body_idx, 0:3]
        root_pos_w = robot.data.root_state_w[:, 0:3]
        root_quat_w = robot.data.root_state_w[:, 3:7]
        ee_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w,
            torch.zeros(self.num_envs, 4, device=self.device)
        )
        return ee_pos_b

    def reset(self, env_ids: torch.Tensor = None):
        """Reset controller."""
        default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)

        if env_ids is None:
            self.target_pos.zero_()
            self.target_quat.zero_()
            self.target_quat[:, 0] = 1.0
            self.smoothed_joint_pos = default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        else:
            self.target_pos[env_ids] = 0.0
            self.target_quat[env_ids] = 0.0
            self.target_quat[env_ids, 0] = 1.0
            self.smoothed_joint_pos[env_ids] = default_pos

        self.controller.reset(env_ids)


##############################################################################
# TARGET GENERATOR
##############################################################################

class TargetGenerator:
    """Generate smooth target trajectories."""

    def __init__(self, num_envs: int, device: str, mode: str = "static", arm: str = "right"):
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.arm = arm

        # Reachable workspace for G1 arm (conservative)
        y_offset = -0.3 if arm == "right" else 0.3
        self.base_position = torch.tensor([0.25, y_offset, 0.3], device=device)
        self.radius = 0.08  # Smaller radius for stability
        self.freq = 0.15  # Slower for better tracking

    def get_target(self, time: float) -> torch.Tensor:
        pos = self.base_position.unsqueeze(0).expand(self.num_envs, -1).clone()

        if self.mode == "circle":
            angle = 2 * math.pi * self.freq * time
            pos[:, 0] += self.radius * math.cos(angle)
            pos[:, 2] += self.radius * math.sin(angle)
        elif self.mode == "wave":
            wave = math.sin(2 * math.pi * self.freq * time)
            pos[:, 2] += wave * self.radius
        elif self.mode == "reach":
            # Gradual reach forward
            t = min(time / 5.0, 1.0)  # Reach over 5 seconds
            pos[:, 0] += t * 0.15
            pos[:, 2] += t * 0.1

        return pos


##############################################################################
# MAIN
##############################################################################

def main():
    print("\n" + "=" * 70)
    print("  G1 Hierarchical Control - STABLE DifferentialIK")
    print("  ")
    print("  STABILITY FEATURES:")
    print(f"  - Max joint delta: {args_cli.max_joint_delta} rad/step")
    print(f"  - Smoothing factor: {args_cli.smoothing}")
    print("  - Joint limit enforcement")
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
        print(f"[Env] ✓ Robot found (floating_base={not robot.is_fixed_base})")

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

    # Create stable IK controller
    arm_ik = G1ArmIKControllerStable(
        env.num_envs, env.device,
        arm=args_cli.arm,
        ik_method=args_cli.ik_method,
        max_joint_delta=args_cli.max_joint_delta,
        smoothing_factor=args_cli.smoothing,
        debug=args_cli.debug
    )
    if robot is not None:
        arm_ik.initialize_from_robot(robot, scene)

    # Target generator
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

    # Tracking metrics
    error_history = []

    try:
        while simulation_app.is_running():
            # Get target
            target_pos = target_gen.get_target(sim_time)

            # Lower body: PPO
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs)
            else:
                actions.zero_()

            # Upper body: Stable IK
            if robot is not None and arm_ik.initialized:
                arm_ik.set_target(target_pos)
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
            if step_count % 200 == 0:
                mean_reward = rewards.mean().item()
                alive = (~terminated).float().mean().item() * 100

                if robot is not None:
                    ee_pos_b = arm_ik.get_ee_pos_base(robot)[0]
                    target_b = target_pos[0]
                    error = torch.norm(target_b - ee_pos_b).item()
                    error_history.append(error)

                    # Calculate trend
                    if len(error_history) > 5:
                        recent_avg = sum(error_history[-5:]) / 5
                        older_avg = sum(error_history[-10:-5]) / 5 if len(error_history) > 10 else recent_avg
                        trend = "↓" if recent_avg < older_avg else "↑" if recent_avg > older_avg else "→"
                    else:
                        trend = "→"

                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                          f"Error: {error:.3f}m {trend}")
                else:
                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}%")

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user")

        # Print summary
        if error_history:
            print(f"\n[Summary]")
            print(f"  Min error: {min(error_history):.4f}m")
            print(f"  Max error: {max(error_history):.4f}m")
            print(f"  Avg error: {sum(error_history) / len(error_history):.4f}m")

    finally:
        env.close()
        print("[Info] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()