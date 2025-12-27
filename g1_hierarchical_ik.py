# Copyright (c) 2025, VLM-RL G1 Project
# G1 Hierarchical Control: PPO Locomotion + DifferentialIK (FIXED)
#
# BUG FIXES APPLIED:
# 1. Jacobian column index: +6 offset for floating-base robots (PR #1033)
# 2. Body index: Use ee_body_idx directly for floating-base (not -1)
# 3. Jacobian frame transform: World -> Base frame rotation (PR #967)

"""
G1 Hierarchical Control with FIXED DifferentialIK
==================================================

3 Critical Fixes Applied:
1. Jacobian column indices need +6 offset for floating-base robots
2. ee_jacobi_idx should equal ee_body_idx (not ee_body_idx - 1)
3. Jacobian must be rotated from world frame to base frame

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\g1_hierarchical_ik_fixed.py --num_envs 4 --load_run 2025-12-27_00-29-54
"""

import argparse
import os
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from collections import deque

# ==== Isaac Lab App Launcher ====
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Hierarchical Control: PPO + IK (FIXED)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, required=True, help="Locomotion policy run folder")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--ik_method", type=str, default="dls", choices=["dls", "pinv", "svd", "trans"])
parser.add_argument("--target_mode", type=str, default="circle",
                    choices=["circle", "static", "wave", "reach"])
parser.add_argument("--arm", type=str, default="right", choices=["left", "right"])
parser.add_argument("--debug", action="store_true", default=True, help="Enable detailed debug output")
parser.add_argument("--use_analytical", action="store_true", default=False,
                    help="Use analytical control instead of IK (fallback)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==== Post-Launch Imports ====
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv, quat_conjugate
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

# Joint indices in the 37-DOF action space
ARM_JOINT_INDICES = {
    "right": [6, 10, 14, 18, 22],
    "left": [5, 9, 13, 17, 21],
}


##############################################################################
# CUSTOM ACTOR NETWORK
##############################################################################

class CustomActorCritic(nn.Module):
    """Custom ActorCritic compatible with RSL-RL checkpoints."""

    def __init__(
            self,
            num_obs: int,
            num_actions: int,
            actor_hidden_dims: List[int] = [512, 256, 128],
            critic_hidden_dims: List[int] = [512, 256, 128],
            activation: str = "elu",
    ):
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions

        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()

        # Build actor
        actor_layers = []
        prev_dim = num_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(act_fn)
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic
        critic_layers = []
        prev_dim = num_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(act_fn)
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(torch.ones(num_actions))
        print(f"[Policy] Created CustomActorCritic: obs={num_obs}, actions={num_actions}")

    def forward(self, obs):
        return self.actor(obs)

    def act_inference(self, obs):
        with torch.no_grad():
            return self.actor(obs)

    def load_rsl_rl_checkpoint(self, checkpoint_path: str, device: str):
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = data["model_state_dict"]

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("actor.") or key.startswith("critic."):
                new_state_dict[key] = value
            elif key == "std":
                new_state_dict["std"] = value
            elif key == "log_std":
                new_state_dict["std"] = torch.exp(value)

        self.load_state_dict(new_state_dict, strict=True)
        print("[Policy] ✓ Checkpoint loaded successfully")
        return True


##############################################################################
# CHECKPOINT FINDER
##############################################################################

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
# FIXED ARM IK CONTROLLER - ALL 3 BUGS FIXED!
##############################################################################

class G1ArmIKControllerFixed:
    """
    DifferentialIK controller for G1 arm with ALL FIXES APPLIED.

    FIX 1: Jacobian column indices need +6 offset for floating-base
           PhysX includes 6 DOFs for root pose at the beginning

    FIX 2: ee_jacobi_idx = ee_body_idx (not ee_body_idx - 1)
           For floating-base, don't subtract 1

    FIX 3: Transform Jacobian from world frame to base frame
           PhysX returns world-frame Jacobian, IK expects base-frame
    """

    def __init__(
            self,
            num_envs: int,
            device: str,
            arm: str = "right",
            ik_method: str = "dls",
            debug: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm
        self.debug = debug

        # IK Controller config
        self.ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params={"lambda_val": 0.05} if ik_method == "dls" else {"k_val": 1.0},
        )

        self.controller = DifferentialIKController(
            self.ik_cfg,
            num_envs=num_envs,
            device=device,
        )

        # Will be initialized from robot
        self.ee_body_idx = None
        self.arm_joint_ids = None  # Indices for joint_pos array
        self.jacobian_col_ids = None  # Indices for Jacobian columns (WITH +6 offset!)
        self.ee_jacobi_idx = None  # Row index for Jacobian
        self.is_floating_base = True  # G1 is floating-base

        # Target pose
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_quat = torch.zeros(num_envs, 4, device=device)
        self.target_quat[:, 0] = 1.0  # w=1 for identity quaternion

        self.initialized = False
        self.step_count = 0

        print(f"\n{'=' * 60}")
        print(f"[IK FIXED] G1ArmIKController - ALL BUGS FIXED")
        print(f"[IK FIXED] Arm: {arm}, Method: {ik_method}")
        print(f"{'=' * 60}")

    def initialize_from_robot(self, robot, scene):
        """Initialize indices from robot articulation."""
        print(f"\n[IK INIT] Initializing from robot...")

        try:
            # === Get body names and find end-effector ===
            body_names = robot.body_names if hasattr(robot, 'body_names') else []
            ee_name = G1_EE_BODIES[self.arm]

            if self.debug:
                print(f"[IK INIT] Total bodies: {len(body_names)}")
                print(f"[IK INIT] Looking for EE: {ee_name}")

            if ee_name in body_names:
                self.ee_body_idx = body_names.index(ee_name)
            else:
                self.ee_body_idx = 29 if self.arm == "right" else 28
                print(f"[IK INIT] WARNING: Using fallback EE index {self.ee_body_idx}")

            print(f"[IK INIT] ✓ EE body index: {self.ee_body_idx}")

            # === FIX 2: ee_jacobi_idx for floating-base ===
            # For floating-base robots, use ee_body_idx directly (NOT -1!)
            self.is_floating_base = not robot.is_fixed_base if hasattr(robot, 'is_fixed_base') else True

            if self.is_floating_base:
                self.ee_jacobi_idx = self.ee_body_idx  # FIX 2: No subtraction!
                print(f"[IK INIT] ✓ Floating-base detected, ee_jacobi_idx = {self.ee_jacobi_idx}")
            else:
                self.ee_jacobi_idx = self.ee_body_idx - 1
                print(f"[IK INIT] Fixed-base, ee_jacobi_idx = {self.ee_jacobi_idx}")

            # === Get joint names and find arm joints ===
            joint_names = robot.joint_names if hasattr(robot, 'joint_names') else []

            if self.debug:
                print(f"[IK INIT] Total joints: {len(joint_names)}")

            # Find arm joint indices in joint_pos array
            self.arm_joint_ids = []
            for jname in G1_ARM_JOINTS[self.arm]:
                if jname in joint_names:
                    idx = joint_names.index(jname)
                    self.arm_joint_ids.append(idx)
                    if self.debug:
                        print(f"[IK INIT]   {jname} -> joint index {idx}")

            if len(self.arm_joint_ids) < 5:
                self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]
                print(f"[IK INIT] WARNING: Using default arm_joint_ids: {self.arm_joint_ids}")

            print(f"[IK INIT] ✓ Arm joint indices (for joint_pos): {self.arm_joint_ids}")

            # === FIX 1: Jacobian column indices with +6 offset ===
            # PhysX Jacobian shape: (num_envs, num_bodies, 6, num_joints + 6)
            # The +6 is for floating-base root DOFs at the BEGINNING
            if self.is_floating_base:
                self.jacobian_col_ids = [idx + 6 for idx in self.arm_joint_ids]  # FIX 1!
                print(f"[IK INIT] ✓ Jacobian column indices (+6 offset): {self.jacobian_col_ids}")
            else:
                self.jacobian_col_ids = self.arm_joint_ids
                print(f"[IK INIT] Fixed-base: Jacobian columns = {self.jacobian_col_ids}")

            # === Verify Jacobian shape ===
            try:
                full_jac = robot.root_physx_view.get_jacobians()
                print(f"\n[IK INIT] Jacobian shape verification:")
                print(f"[IK INIT]   Full Jacobian shape: {full_jac.shape}")
                print(f"[IK INIT]   Expected: (num_envs, num_bodies, 6, num_joints+6)")
                print(f"[IK INIT]   Actual last dim: {full_jac.shape[-1]}")
                print(f"[IK INIT]   num_joints: {len(joint_names)}, +6 = {len(joint_names) + 6}")

                # Extract test slice
                test_jac = full_jac[:, self.ee_jacobi_idx, :, :]
                print(f"[IK INIT]   EE Jacobian slice shape: {test_jac.shape}")

                arm_jac = test_jac[:, :, self.jacobian_col_ids]
                print(f"[IK INIT]   Arm Jacobian shape: {arm_jac.shape}")
                print(f"[IK INIT]   Arm Jacobian norm: {torch.norm(arm_jac).item():.4f}")

                # Check if Jacobian is all zeros (indicates wrong indices)
                if torch.norm(arm_jac) < 1e-6:
                    print(f"[IK INIT] ⚠️ WARNING: Arm Jacobian is near-zero! Check indices!")
                else:
                    print(f"[IK INIT] ✓ Arm Jacobian looks valid (non-zero)")

            except Exception as e:
                print(f"[IK INIT] Could not verify Jacobian: {e}")

            self.initialized = True
            print(f"\n[IK INIT] ✓ Initialization complete!")
            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"[IK INIT] ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            self.ee_body_idx = 29
            self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]
            self.jacobian_col_ids = [idx + 6 for idx in self.arm_joint_ids]
            self.ee_jacobi_idx = 29
            self.initialized = True

    def set_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor = None):
        """Set target pose in BASE FRAME."""
        self.target_pos = target_pos.clone()

        if target_quat is None:
            target_quat = torch.zeros(self.num_envs, 4, device=self.device)
            target_quat[:, 0] = 1.0  # Identity quaternion (w, x, y, z)

        self.target_quat = target_quat.clone()

        # Set command for DifferentialIKController
        pose_command = torch.cat([target_pos, target_quat], dim=-1)
        self.controller.set_command(pose_command)

    def _transform_jacobian_to_base_frame(self, jacobian_w: torch.Tensor,
                                          root_quat_w: torch.Tensor) -> torch.Tensor:
        """
        FIX 3: Transform Jacobian from world frame to base frame.

        PhysX returns Jacobian in world frame, but DifferentialIKController
        expects everything in a consistent frame (base frame).

        J_base = R_base^T @ J_world
        where R_base is the rotation matrix from base to world
        """
        # Get rotation matrix from base to world
        # quat_conjugate gives us world_to_base rotation
        root_quat_conj = quat_conjugate(root_quat_w)  # (num_envs, 4)

        # Convert quaternion to rotation matrix
        # matrix_from_quat expects (w, x, y, z) format
        rot_matrix = matrix_from_quat(root_quat_conj)  # (num_envs, 3, 3)

        # Transform linear part (first 3 rows)
        jacobian_b = jacobian_w.clone()

        # J_b[:3, :] = R @ J_w[:3, :]
        linear_part = jacobian_w[:, :3, :]  # (num_envs, 3, num_joints)
        jacobian_b[:, :3, :] = torch.bmm(rot_matrix, linear_part)

        # J_b[3:, :] = R @ J_w[3:, :]
        angular_part = jacobian_w[:, 3:, :]  # (num_envs, 3, num_joints)
        jacobian_b[:, 3:, :] = torch.bmm(rot_matrix, angular_part)

        return jacobian_b

    def compute(self, robot) -> torch.Tensor:
        """
        Compute IK solution with ALL FIXES applied.

        Returns:
            Desired joint positions for arm joints
        """
        if not self.initialized:
            return torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

        self.step_count += 1
        debug_this_step = self.debug and (self.step_count % 50 == 1)

        try:
            # === Get current joint positions ===
            joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]

            # === Get current EE pose in world frame ===
            ee_pose_w = robot.data.body_state_w[:, self.ee_body_idx, 0:7]
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            # === Get root pose in world frame ===
            root_pose_w = robot.data.root_state_w[:, 0:7]
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]

            # === Transform EE pose to base frame ===
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            # === Get Jacobian with CORRECT indices (FIX 1 & 2) ===
            full_jacobian = robot.root_physx_view.get_jacobians()
            # Shape: (num_envs, num_bodies, 6, num_dofs+6)

            # FIX 2: Use ee_body_idx directly for floating-base
            jacobian_w = full_jacobian[:, self.ee_jacobi_idx, :, :]
            # Shape: (num_envs, 6, num_dofs+6)

            # FIX 1: Use +6 offset indices for arm joints
            jacobian_arm_w = jacobian_w[:, :, self.jacobian_col_ids]
            # Shape: (num_envs, 6, 5) for 5 arm joints

            # === FIX 3: Transform Jacobian to base frame ===
            jacobian_arm_b = self._transform_jacobian_to_base_frame(jacobian_arm_w, root_quat_w)

            # === Debug output ===
            if debug_this_step:
                print(f"\n{'=' * 60}")
                print(f"[IK DEBUG] Step {self.step_count}")
                print(f"{'=' * 60}")

                # Target vs Current
                print(f"\n[POSES]")
                print(
                    f"  Target pos (base): [{self.target_pos[0, 0]:.3f}, {self.target_pos[0, 1]:.3f}, {self.target_pos[0, 2]:.3f}]")
                print(f"  Current EE (base): [{ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}]")

                pos_error = self.target_pos - ee_pos_b
                error_mag = torch.norm(pos_error, dim=1)
                print(f"  Position error:    [{pos_error[0, 0]:.3f}, {pos_error[0, 1]:.3f}, {pos_error[0, 2]:.3f}]")
                print(f"  Error magnitude:   {error_mag[0]:.4f} m")

                # Joint positions
                print(f"\n[JOINTS]")
                print(f"  Current joint pos: {joint_pos[0].tolist()}")

                # Jacobian info
                print(f"\n[JACOBIAN]")
                print(f"  Full Jacobian shape:    {full_jacobian.shape}")
                print(f"  Arm Jacobian (world):   norm={torch.norm(jacobian_arm_w[0]):.4f}")
                print(f"  Arm Jacobian (base):    norm={torch.norm(jacobian_arm_b[0]):.4f}")

                # Check for near-zero Jacobian
                jac_norm = torch.norm(jacobian_arm_b[0])
                if jac_norm < 0.01:
                    print(f"  ⚠️ WARNING: Jacobian norm very small! ({jac_norm:.6f})")
                    print(f"  This may indicate wrong column indices!")

                # Show Jacobian values for first environment
                print(f"\n[JACOBIAN VALUES (env 0)]")
                for i, col_idx in enumerate(self.jacobian_col_ids):
                    col = jacobian_arm_b[0, :, i]
                    print(
                        f"  Col {i} (jac_idx={col_idx}): [{col[0]:.3f}, {col[1]:.3f}, {col[2]:.3f}, {col[3]:.3f}, {col[4]:.3f}, {col[5]:.3f}]")

            # === Compute IK ===
            joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian_arm_b, joint_pos)

            # === More debug output ===
            if debug_this_step:
                joint_diff = joint_pos_des - joint_pos
                print(f"\n[IK OUTPUT]")
                print(f"  Desired joint pos: {joint_pos_des[0].tolist()}")
                print(f"  Joint diff:        {joint_diff[0].tolist()}")
                print(f"  Joint diff norm:   {torch.norm(joint_diff[0]):.6f}")

                if torch.norm(joint_diff[0]) < 1e-6:
                    print(f"  ⚠️ WARNING: joint_diff is ZERO! IK not working!")
                    print(f"  Check: 1) Jacobian indices 2) Frame transforms 3) Target setting")
                else:
                    print(f"  ✓ IK producing non-zero joint changes")

                print(f"{'=' * 60}\n")

            return joint_pos_des

        except Exception as e:
            print(f"[IK ERROR] {e}")
            import traceback
            traceback.print_exc()
            return robot.data.joint_pos[:, self.arm_joint_ids]

    def get_ee_pos_world(self, robot) -> torch.Tensor:
        """Get current end-effector position in world frame."""
        return robot.data.body_state_w[:, self.ee_body_idx, 0:3]

    def get_ee_pos_base(self, robot) -> torch.Tensor:
        """Get current end-effector position in base frame."""
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
        if env_ids is None:
            self.target_pos.zero_()
            self.target_quat.zero_()
            self.target_quat[:, 0] = 1.0
        else:
            self.target_pos[env_ids] = 0.0
            self.target_quat[env_ids] = 0.0
            self.target_quat[env_ids, 0] = 1.0
        self.controller.reset(env_ids)


##############################################################################
# TARGET GENERATOR
##############################################################################

class TargetGenerator:
    """Generate target trajectories for end-effector in BASE FRAME."""

    def __init__(self, num_envs: int, device: str, mode: str = "circle", arm: str = "right"):
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.arm = arm

        # Base position in robot's base frame
        y_offset = -0.25 if arm == "right" else 0.25
        self.base_position = torch.tensor([0.35, y_offset, 0.55], device=device)
        self.radius = 0.12
        self.freq = 0.3  # Slower for better tracking

    def get_target(self, time: float) -> torch.Tensor:
        """Get target position at given time (in BASE FRAME)."""
        pos = self.base_position.unsqueeze(0).expand(self.num_envs, -1).clone()

        if self.mode == "circle":
            angle = 2 * math.pi * self.freq * time
            pos[:, 0] += self.radius * math.cos(angle)
            pos[:, 2] += self.radius * math.sin(angle)
        elif self.mode == "wave":
            wave = math.sin(2 * math.pi * self.freq * time)
            pos[:, 2] += wave * self.radius
        elif self.mode == "reach":
            wave = math.sin(2 * math.pi * 0.3 * time)
            pos[:, 2] += wave * 0.15
            pos[:, 0] += (1 + wave) * 0.05
        # static mode: just return base_position

        return pos


##############################################################################
# MAIN
##############################################################################

def main():
    """Main simulation loop."""

    print("\n" + "=" * 70)
    print("  G1 Hierarchical Control - FIXED DifferentialIK")
    print("  ")
    print("  FIXES APPLIED:")
    print("  1. Jacobian column indices: +6 offset for floating-base")
    print("  2. Body index: Use ee_body_idx directly")
    print("  3. Frame transform: World -> Base for Jacobian")
    print("=" * 70 + "\n")

    # ==== Environment Setup ====
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.action_manager.total_action_dim

    print(f"[Env] Obs dim: {obs_dim}, Action dim: {action_dim}")

    # ==== Get Robot ====
    robot = None
    scene = env.scene
    if hasattr(scene, 'articulations') and 'robot' in scene.articulations:
        robot = scene.articulations['robot']
        print("[Env] ✓ Robot articulation found!")

        # Print robot info
        if hasattr(robot, 'is_fixed_base'):
            print(f"[Env] Robot is_fixed_base: {robot.is_fixed_base}")
        if hasattr(robot, 'num_bodies'):
            print(f"[Env] Robot num_bodies: {robot.num_bodies}")
        if hasattr(robot, 'num_joints'):
            print(f"[Env] Robot num_joints: {robot.num_joints}")

    # ==== Load Policy ====
    policy = None
    try:
        run_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
        checkpoint_path = find_checkpoint(run_dir, args_cli.checkpoint)
        print(f"\n[Policy] Loading: {checkpoint_path}")

        policy = CustomActorCritic(
            num_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[256, 128, 128],
            critic_hidden_dims=[256, 128, 128],
            activation="elu",
        ).to(env.device)

        policy.load_rsl_rl_checkpoint(checkpoint_path, env.device)
        policy.eval()
        print("[Policy] ✓ Locomotion policy loaded!")

    except Exception as e:
        print(f"[Policy] ✗ Error: {e}")
        policy = None

    # ==== Create FIXED IK Controller ====
    arm = args_cli.arm
    arm_ik = G1ArmIKControllerFixed(
        env.num_envs, env.device,
        arm=arm,
        ik_method=args_cli.ik_method,
        debug=args_cli.debug
    )
    if robot is not None:
        arm_ik.initialize_from_robot(robot, scene)

    # ==== Create Target Generator ====
    target_gen = TargetGenerator(env.num_envs, env.device, args_cli.target_mode, arm=arm)

    # ==== Reset ====
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)

    print("\n" + "-" * 70)
    print("[Info] Starting simulation... Press Ctrl+C to stop")
    print("[Info] Lower body: PPO locomotion policy")
    print(f"[Info] Upper body: DifferentialIK ({args_cli.ik_method})")
    print(f"[Info] Target mode: {args_cli.target_mode}")
    print(f"[Info] Debug: {args_cli.debug}")
    print("-" * 70 + "\n")

    sim_time = 0.0
    dt = 0.02
    step_count = 0

    try:
        while simulation_app.is_running():
            # ==== Get Target ====
            target_pos = target_gen.get_target(sim_time)

            # ==== Lower Body: PPO Policy ====
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs)
            else:
                actions.zero_()

            # ==== Upper Body: IK Control ====
            if robot is not None and arm_ik.initialized and not args_cli.use_analytical:
                # Set target pose
                arm_ik.set_target(target_pos)

                # Compute IK
                joint_pos_des = arm_ik.compute(robot)

                # Apply to actions
                for i, joint_idx in enumerate(arm_ik.arm_joint_ids):
                    actions[:, joint_idx] = joint_pos_des[:, i]

            elif args_cli.use_analytical:
                # Fallback: Simple analytical control
                target_b = target_pos[0]
                shoulder_pitch = -1.5 - (target_b[0].item() - 0.35) * 2.0
                shoulder_roll = -0.3
                elbow_pitch = -1.2 + (target_b[2].item() - 0.5) * 1.5

                actions[:, 6] = shoulder_pitch
                actions[:, 10] = shoulder_roll
                actions[:, 14] = 0.0
                actions[:, 18] = elbow_pitch
                actions[:, 22] = 0.0

            # ==== Step ====
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]

            # ==== Handle Resets ====
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                arm_ik.reset(reset_ids)

            sim_time += dt
            step_count += 1

            # ==== Periodic Logging ====
            if step_count % 200 == 0:
                mean_reward = rewards.mean().item()
                alive = (~terminated).float().mean().item() * 100

                if robot is not None:
                    ee_pos_b = arm_ik.get_ee_pos_base(robot)[0]
                    target_b = target_pos[0]
                    error = torch.norm(target_b - ee_pos_b).item()

                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                          f"TrackErr: {error:.3f}m")
                else:
                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}%")

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user")

    finally:
        env.close()
        print("[Info] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()