# Copyright (c) 2025, VLM-RL G1 Project
# G1 Hierarchical Control V2: PPO Locomotion + Real DifferentialIK
# Uses PhysX Jacobians for accurate task-space control

"""
G1 Hierarchical Control V2 - Advanced IK Integration
=====================================================

Bu script, Isaac Lab'ın DifferentialIKController'ını gerçek PhysX Jacobian'ları
ile kullanarak task-space arm kontrolü sağlar.

Lower Body: PPO Locomotion Policy (trained 20K iterations)
Upper Body: DifferentialIKController with PhysX Jacobians

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\g1_hierarchical_ik_v2.py --num_envs 4 --load_run 2025-12-27_00-29-54
"""

import argparse
import os
import math
import torch
from typing import Tuple, Optional

# ==== Isaac Lab App Launcher ====
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Hierarchical Control V2: PPO + Real IK")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, required=True, help="Locomotion policy run folder")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--ik_method", type=str, default="dls", choices=["dls", "pinv", "svd", "trans"])
parser.add_argument("--target_mode", type=str, default="circle",
                    choices=["circle", "static", "wave", "reach", "track_object"])
parser.add_argument("--arm", type=str, default="right", choices=["left", "right", "both"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==== Post-Launch Imports ====
import gymnasium as gym
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz

# RSL-RL
try:
    from rsl_rl.modules import ActorCritic

    RSL_RL_AVAILABLE = True
except ImportError:
    RSL_RL_AVAILABLE = False
    print("[Warning] RSL-RL not available")

# Isaac Lab tasks
import isaaclab_tasks  # noqa: F401

##############################################################################
# G1 ARM CONFIGURATION
##############################################################################

# G1 Joint names for arms (from URDF)
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

# End-effector body names
G1_EE_BODIES = {
    "right": "right_palm_link",
    "left": "left_palm_link",
}

# Arm joint indices in the 37 DoF action space
# These will be dynamically determined from joint names
ARM_JOINT_INDICES = {
    "right": [6, 10, 14, 18, 22],  # Approximate - will verify
    "left": [5, 9, 13, 17, 21],
}


##############################################################################
# CHECKPOINT FINDER
##############################################################################

def find_checkpoint(run_dir: str, checkpoint_name: str = None) -> str:
    """Find checkpoint file in run directory."""
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if checkpoint_name:
        path = os.path.join(run_dir, checkpoint_name)
        if os.path.exists(path):
            return path

    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return os.path.join(run_dir, checkpoints[-1])


##############################################################################
# ARM IK CONTROLLER WRAPPER
##############################################################################

class G1ArmIKController:
    """
    Wrapper for DifferentialIKController specialized for G1 arm control.

    Handles:
    - Jacobian extraction from robot
    - Frame transformations (world -> base)
    - Joint index mapping
    """

    def __init__(
            self,
            num_envs: int,
            device: str,
            arm: str = "right",
            ik_method: str = "dls",
    ):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm

        # Create DifferentialIK controller
        self.ik_cfg = DifferentialIKControllerCfg(
            command_type="position",  # 3D position only (no orientation)
            use_relative_mode=False,  # Absolute targets
            ik_method=ik_method,
            ik_params={"lambda_val": 0.1} if ik_method == "dls" else {"k_val": 1.0},
        )

        self.controller = DifferentialIKController(
            self.ik_cfg,
            num_envs=num_envs,
            device=device,
        )

        # Body and joint indices (will be set when robot is available)
        self.ee_body_idx = None
        self.arm_joint_ids = None
        self.ee_jacobi_idx = None  # Jacobian index for end-effector

        # State buffers
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.initialized = False

        print(f"[IK] Created G1ArmIKController for {arm} arm (method: {ik_method})")

    def initialize_from_robot(self, robot, scene):
        """
        Initialize body and joint indices from robot articulation.

        Args:
            robot: Isaac Lab Articulation object
            scene: InteractiveScene containing the robot
        """
        try:
            # Find end-effector body index
            ee_name = G1_EE_BODIES[self.arm]

            # Try to find the body
            body_names = robot.body_names if hasattr(robot, 'body_names') else []
            print(f"[IK] Robot body names: {body_names[:10]}..." if len(
                body_names) > 10 else f"[IK] Robot body names: {body_names}")

            # Find EE body index
            if ee_name in body_names:
                self.ee_body_idx = body_names.index(ee_name)
                print(f"[IK] Found {ee_name} at body index {self.ee_body_idx}")
            else:
                # Try alternative names
                alt_names = [f"{self.arm}_five_link", f"{self.arm}_hand_link", f"{self.arm}_wrist_link"]
                for alt in alt_names:
                    if alt in body_names:
                        self.ee_body_idx = body_names.index(alt)
                        print(f"[IK] Using alternative {alt} at index {self.ee_body_idx}")
                        break

                if self.ee_body_idx is None:
                    print(f"[IK] Warning: Could not find end-effector body, using fallback")
                    self.ee_body_idx = len(body_names) - 1  # Use last body as fallback

            # Find arm joint indices
            joint_names = robot.joint_names if hasattr(robot, 'joint_names') else []
            print(f"[IK] Robot joint names: {joint_names[:10]}..." if len(
                joint_names) > 10 else f"[IK] Robot joint names: {joint_names}")

            self.arm_joint_ids = []
            for jname in G1_ARM_JOINTS[self.arm]:
                # Try exact match first
                if jname in joint_names:
                    self.arm_joint_ids.append(joint_names.index(jname))
                else:
                    # Try partial match
                    for i, name in enumerate(joint_names):
                        if jname.replace("_joint", "") in name:
                            self.arm_joint_ids.append(i)
                            break

            if len(self.arm_joint_ids) < 5:
                print(f"[IK] Warning: Only found {len(self.arm_joint_ids)} arm joints, using defaults")
                self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]

            print(f"[IK] Arm joint indices: {self.arm_joint_ids}")

            # Jacobian index for EE
            self.ee_jacobi_idx = self.ee_body_idx - 1  # Usually body_idx - 1 for Jacobian

            self.initialized = True
            print(f"[IK] Initialization complete!")

        except Exception as e:
            print(f"[IK] Error during initialization: {e}")
            # Use fallback indices
            self.ee_body_idx = 20  # Approximate
            self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]
            self.ee_jacobi_idx = self.ee_body_idx - 1
            self.initialized = True

    def set_target(self, target_pos: torch.Tensor):
        """Set target end-effector position in base frame."""
        self.target_pos = target_pos.clone()
        self.controller.set_command(target_pos)

    def compute(
            self,
            robot,
            jacobian: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute arm joint positions using IK.

        Args:
            robot: Articulation object with current state
            jacobian: Optional pre-computed Jacobian

        Returns:
            Joint positions for arm joints (num_envs, num_arm_joints)
        """
        if not self.initialized:
            print("[IK] Warning: Controller not initialized!")
            return torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

        try:
            # Get current joint positions
            joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]

            # Get current EE pose in world frame
            ee_pose_w = robot.data.body_state_w[:, self.ee_body_idx, 0:7]
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            # Get root pose
            root_pose_w = robot.data.root_state_w[:, 0:7]
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]

            # Transform EE pose to base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                ee_pos_w, ee_quat_w
            )

            # Get or compute Jacobian
            if jacobian is None:
                # Extract Jacobian from PhysX
                full_jacobian = robot.root_physx_view.get_jacobians()
                # Shape: (num_envs, num_bodies, 6, num_joints)
                # Extract EE Jacobian for arm joints only
                jacobian = full_jacobian[:, self.ee_jacobi_idx, :3, :]  # Position only
                jacobian = jacobian[:, :, self.arm_joint_ids]  # Arm joints only

            # Compute IK
            joint_pos_des = self.controller.compute(
                ee_pos_b,
                ee_quat_b,
                jacobian,
                joint_pos,
            )

            return joint_pos_des

        except Exception as e:
            print(f"[IK] Compute error: {e}")
            # Return current positions as fallback
            return robot.data.joint_pos[:, self.arm_joint_ids] if hasattr(robot.data, 'joint_pos') else \
                torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

    def reset(self, env_ids: torch.Tensor = None):
        """Reset controller state."""
        if env_ids is None:
            self.target_pos.zero_()
        else:
            self.target_pos[env_ids] = 0.0
        self.controller.reset(env_ids)


##############################################################################
# SIMPLE ARM IK (FALLBACK)
##############################################################################

class SimpleArmIK:
    """
    Simplified geometric IK fallback for G1 arm.
    Used when DifferentialIKController initialization fails.
    """

    def __init__(self, num_envs: int, device: str, arm: str = "right"):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm
        self.arm_joint_ids = ARM_JOINT_INDICES[arm]

        # Joint position buffer
        self.joint_pos = torch.zeros(num_envs, 5, device=device)

        # Default position (arm slightly forward)
        self.default_pos = torch.tensor([0.3, -0.3 if arm == "right" else 0.3, 0.5], device=device)

    def set_target(self, target_pos: torch.Tensor):
        """Set target position."""
        self.target_pos = target_pos.clone()

    def compute(self, current_ee_pos: torch.Tensor, target_pos: torch.Tensor, dt: float = 0.02) -> torch.Tensor:
        """Compute joint deltas using geometric approximation."""
        error = target_pos - current_ee_pos
        gain = 3.0

        # Simplified Jacobian-like mapping
        deltas = torch.zeros(self.num_envs, 5, device=self.device)

        # Shoulder pitch: forward/back + up/down
        deltas[:, 0] = -gain * (error[:, 0] * 0.4 + error[:, 2] * 0.6) * dt

        # Shoulder roll: left/right
        sign = -1.0 if self.arm == "right" else 1.0
        deltas[:, 1] = sign * gain * error[:, 1] * 0.5 * dt

        # Shoulder yaw: rotation
        deltas[:, 2] = gain * (error[:, 0] * 0.1 - error[:, 1] * 0.1) * dt

        # Elbow pitch: reach
        deltas[:, 3] = -gain * (error[:, 0] * 0.3 + error[:, 2] * 0.4) * dt

        # Elbow roll: minimal
        deltas[:, 4] = 0.0

        # Clamp and integrate
        deltas = torch.clamp(deltas, -0.15, 0.15)
        self.joint_pos += deltas

        # Apply joint limits
        self.joint_pos[:, 0] = torch.clamp(self.joint_pos[:, 0], -2.5, 2.0)
        self.joint_pos[:, 1] = torch.clamp(self.joint_pos[:, 1], -1.2, 1.0)
        self.joint_pos[:, 2] = torch.clamp(self.joint_pos[:, 2], -1.2, 1.2)
        self.joint_pos[:, 3] = torch.clamp(self.joint_pos[:, 3], -2.5, 0.0)
        self.joint_pos[:, 4] = torch.clamp(self.joint_pos[:, 4], -1.0, 1.0)

        return self.joint_pos.clone()

    def reset(self, env_ids: torch.Tensor = None):
        if env_ids is None:
            self.joint_pos.zero_()
        else:
            self.joint_pos[env_ids] = 0.0


##############################################################################
# TARGET TRAJECTORY GENERATOR
##############################################################################

class TargetGenerator:
    """Generate target trajectories for end-effector."""

    def __init__(
            self,
            num_envs: int,
            device: str,
            mode: str = "circle",
            arm: str = "right",
    ):
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.arm = arm

        # Base position for right/left arm
        y_offset = -0.25 if arm == "right" else 0.25
        self.base_position = torch.tensor([0.35, y_offset, 0.55], device=device)

        # Motion parameters
        self.radius = 0.12
        self.freq = 0.4

    def get_target(self, time: float) -> torch.Tensor:
        """Get target position at given time."""
        pos = self.base_position.unsqueeze(0).expand(self.num_envs, -1).clone()

        if self.mode == "circle":
            angle = 2 * math.pi * self.freq * time
            pos[:, 0] += self.radius * math.cos(angle)
            pos[:, 2] += self.radius * math.sin(angle)

        elif self.mode == "wave":
            wave = math.sin(2 * math.pi * self.freq * time)
            pos[:, 1] += wave * self.radius

        elif self.mode == "reach":
            wave = math.sin(2 * math.pi * 0.3 * time)
            pos[:, 2] += wave * 0.15
            pos[:, 0] += (1 + wave) * 0.05

        elif self.mode == "track_object":
            # Future: will be replaced with VLM-detected object position
            angle = 2 * math.pi * 0.2 * time
            pos[:, 0] += 0.1 * math.cos(angle)
            pos[:, 1] += 0.05 * math.sin(2 * angle)
            pos[:, 2] += 0.08 * math.sin(angle)

        return pos


##############################################################################
# MAIN SIMULATION
##############################################################################

def main():
    """Main simulation loop with hierarchical control."""

    print("=" * 70)
    print("  G1 Hierarchical Control V2")
    print("  Lower Body: PPO Locomotion (20K iterations)")
    print("  Upper Body: DifferentialIK with PhysX Jacobians")
    print("=" * 70)

    # ==== Environment Setup ====
    env_cfg_entry = "Isaac-Velocity-Flat-G1-v0"

    env = gym.make(
        env_cfg_entry,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # Get dimensions
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else 123
    action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 37

    print(f"\n[Env] Task: {env_cfg_entry}")
    print(f"[Env] Num envs: {args_cli.num_envs}")
    print(f"[Env] Obs dim: {obs_dim}")
    print(f"[Env] Action dim: {action_dim}")
    print(f"[Env] Device: {env.device}")

    # ==== Try to access robot from environment ====
    robot = None
    scene = None
    use_real_ik = False

    try:
        # Access underlying scene
        unwrapped_env = env.unwrapped

        if hasattr(unwrapped_env, 'scene'):
            scene = unwrapped_env.scene
            print(f"[Env] Found scene: {type(scene)}")

            # Try to get robot articulation
            if hasattr(scene, 'articulations'):
                articulations = scene.articulations
                print(
                    f"[Env] Articulations: {list(articulations.keys()) if hasattr(articulations, 'keys') else articulations}")

                # Get robot (usually named "robot")
                if 'robot' in articulations:
                    robot = articulations['robot']
                    print(f"[Env] Found robot articulation!")
                    use_real_ik = True

        if robot is None and hasattr(unwrapped_env, 'robot'):
            robot = unwrapped_env.robot
            print(f"[Env] Found robot from env.robot")
            use_real_ik = True

    except Exception as e:
        print(f"[Env] Could not access robot: {e}")

    # ==== Load PPO Policy ====
    policy = None
    if RSL_RL_AVAILABLE:
        try:
            run_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
            checkpoint_path = find_checkpoint(run_dir, args_cli.checkpoint)

            print(f"\n[Policy] Loading: {checkpoint_path}")

            loaded = torch.load(checkpoint_path, map_location=env.device)

            if "model_state_dict" in loaded:
                policy = ActorCritic(
                    num_obs=obs_dim,
                    num_actions=action_dim,
                    actor_hidden_dims=[512, 256, 128],
                    critic_hidden_dims=[512, 256, 128],
                ).to(env.device)

                policy.load_state_dict(loaded["model_state_dict"])
                policy.eval()

                print("[Policy] ✓ PPO locomotion policy loaded!")

        except Exception as e:
            print(f"[Policy] ✗ Error loading policy: {e}")
            policy = None

    if policy is None:
        print("[Policy] Using simple standing control")

    # ==== Create Arm IK Controller ====
    arm = args_cli.arm

    if use_real_ik and robot is not None:
        arm_ik = G1ArmIKController(
            args_cli.num_envs,
            env.device,
            arm=arm,
            ik_method=args_cli.ik_method,
        )
        arm_ik.initialize_from_robot(robot, scene)
        print(f"[IK] Using DifferentialIK with PhysX Jacobians")
    else:
        arm_ik = SimpleArmIK(args_cli.num_envs, env.device, arm=arm)
        print(f"[IK] Using SimpleArmIK fallback")
        use_real_ik = False

    # ==== Create Target Generator ====
    target_gen = TargetGenerator(
        args_cli.num_envs,
        env.device,
        args_cli.target_mode,
        arm=arm,
    )
    print(f"[Target] Mode: {args_cli.target_mode}")

    # ==== Reset Environment ====
    obs, _ = env.reset()

    # Initial values
    actions = torch.zeros(args_cli.num_envs, action_dim, device=env.device)
    arm_joint_ids = ARM_JOINT_INDICES[arm]

    # Approximate EE position tracker (for fallback)
    approx_ee_pos = target_gen.base_position.unsqueeze(0).expand(args_cli.num_envs, -1).clone()

    # ==== Simulation Loop ====
    print("\n" + "-" * 70)
    print("[Info] Starting simulation... Press Ctrl+C to stop")
    print("-" * 70)

    sim_time = 0.0
    dt = 0.02
    step_count = 0

    try:
        while simulation_app.is_running():

            # ==== Get Target ====
            target_pos = target_gen.get_target(sim_time)

            # ==== Compute Actions ====

            # --- Lower Body: PPO Policy ---
            if policy is not None:
                with torch.no_grad():
                    policy_actions = policy.act_inference(obs)
                    actions = policy_actions.clone()
            else:
                actions.zero_()

            # --- Upper Body: IK Control ---
            if use_real_ik and isinstance(arm_ik, G1ArmIKController):
                # Set target
                arm_ik.set_target(target_pos)

                # Compute IK
                arm_joints = arm_ik.compute(robot)

                # Override arm joints
                for i, idx in enumerate(arm_joint_ids):
                    if i < arm_joints.shape[1]:
                        actions[:, idx] = arm_joints[:, i]

            else:
                # Use simple IK
                arm_joints = arm_ik.compute(approx_ee_pos, target_pos, dt)

                # Override arm joints
                for i, idx in enumerate(arm_joint_ids):
                    actions[:, idx] = arm_joints[:, i]

                # Update approximate EE position
                approx_ee_pos = 0.9 * approx_ee_pos + 0.1 * target_pos

            # ==== Step Environment ====
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Handle resets
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                arm_ik.reset(reset_ids)
                if not use_real_ik:
                    approx_ee_pos[reset_ids] = target_gen.base_position

            # ==== Update Time ====
            sim_time += dt
            step_count += 1

            # ==== Logging ====
            if step_count % 200 == 0:
                mean_reward = rewards.mean().item()
                alive = (~terminated).float().mean().item() * 100

                # Target position
                target_str = f"[{target_pos[0, 0]:.2f}, {target_pos[0, 1]:.2f}, {target_pos[0, 2]:.2f}]"

                print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                      f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                      f"Target: {target_str}")

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user")

    finally:
        env.close()
        print("[Info] Environment closed")


##############################################################################
# ENTRY POINT
##############################################################################

if __name__ == "__main__":
    main()
    simulation_app.close()