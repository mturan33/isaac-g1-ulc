# Copyright (c) 2025, VLM-RL G1 Project
# G1 IK Debug with Marker Prims (Isaac Sim 5.x compatible)

"""
G1 IK Debug with Visual Marker Prims

Uses USD sphere prims for visualization (more reliable than debug draw).

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\g1_ik_marker_debug.py --num_envs 4 --load_run 2025-12-27_00-29-54 --target_mode interactive
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 IK Debug with Marker Prims")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--target_mode", type=str, default="interactive", choices=["interactive", "wave", "static"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import os

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms

# For creating visual markers - Isaac Sim 5.x API
from pxr import UsdGeom, Gf, Sdf
import omni.usd

print("\n" + "=" * 70)
print("  G1 IK Debug with Marker Prims")
print("=" * 70)


class MarkerVisualization:
    """Visual markers using USD prims instead of debug draw."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.stage = omni.usd.get_context().get_stage()
        self.target_paths = []
        self.ee_paths = []

        # Create marker parent
        parent_path = "/World/Markers"
        if not self.stage.GetPrimAtPath(parent_path):
            UsdGeom.Xform.Define(self.stage, parent_path)

        # Create marker prims for each environment
        for i in range(num_envs):
            # Target marker (blue)
            target_path = f"/World/Markers/target_{i}"
            self._create_sphere(target_path, radius=0.03, color=(0.2, 0.4, 1.0))
            self.target_paths.append(target_path)

            # EE marker (green)
            ee_path = f"/World/Markers/ee_{i}"
            self._create_sphere(ee_path, radius=0.025, color=(0.2, 1.0, 0.4))
            self.ee_paths.append(ee_path)

        print(f"[MARKERS] Created {num_envs * 2} visual markers")

    def _create_sphere(self, path: str, radius: float, color: tuple):
        """Create a sphere prim at the given path."""
        sphere = UsdGeom.Sphere.Define(self.stage, path)
        sphere.GetRadiusAttr().Set(radius)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def update_target(self, env_idx: int, pos_w: torch.Tensor):
        """Update target marker position (world frame)."""
        if env_idx < len(self.target_paths):
            pos = pos_w.cpu().numpy()
            prim = self.stage.GetPrimAtPath(self.target_paths[env_idx])
            if prim:
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def update_ee(self, env_idx: int, pos_w: torch.Tensor):
        """Update EE marker position (world frame)."""
        if env_idx < len(self.ee_paths):
            pos = pos_w.cpu().numpy()
            prim = self.stage.GetPrimAtPath(self.ee_paths[env_idx])
            if prim:
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def update_all(self, target_positions_w: torch.Tensor, ee_positions_w: torch.Tensor):
        """Update all markers."""
        for i in range(min(self.num_envs, target_positions_w.shape[0])):
            self.update_target(i, target_positions_w[i])
            self.update_ee(i, ee_positions_w[i])


def main():
    # Import environment config
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    # Create environment
    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    device = env.device
    num_envs = env.num_envs

    print(
        f"[Env] Obs: {env.observation_manager.group_obs_dim['policy']}, Actions: {env.action_manager.total_action_dim}")

    # Get robot
    robot = env.scene.articulations.get("robot", None)
    if robot is None:
        print("[ERROR] Robot not found!")
        return
    print("[Env] ✓ Robot found")

    # Load policy
    policy = None
    if args_cli.load_run:
        log_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
        if args_cli.checkpoint:
            checkpoint_path = os.path.join(log_dir, args_cli.checkpoint)
        else:
            checkpoint_path = os.path.join(log_dir, "model_19999.pt")

        if os.path.exists(checkpoint_path):
            print(f"[Policy] Loading: {checkpoint_path}")
            loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)

            from rsl_rl.modules import ActorCritic

            obs_dim = env.observation_manager.group_obs_dim['policy']
            act_dim = env.action_manager.total_action_dim

            policy = ActorCritic(
                num_actor_obs=obs_dim,
                num_critic_obs=obs_dim,
                num_actions=act_dim,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
            ).to(device)

            policy.load_state_dict(loaded['model_state_dict'])
            policy.eval()
            print("[Policy] ✓ Loaded!")
        else:
            print(f"[Policy] Checkpoint not found: {checkpoint_path}")

    # Setup IK for right arm
    ARM = "right"
    EE_BODY = "right_wrist_yaw_link"
    ARM_JOINTS = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
    ]

    # Find indices
    ee_idx = robot.find_bodies(EE_BODY)[0][0]
    arm_joint_ids, _ = robot.find_joints(ARM_JOINTS)
    base_idx = robot.find_bodies("pelvis")[0][0]

    # Jacobian indices (add 6 for floating base)
    jacobian_ids = torch.tensor(arm_joint_ids, device=device) + 6

    # Create IK controller
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_controller = DifferentialIKController(ik_cfg, len(arm_joint_ids), device)

    print(f"\n[IK] EE body: {EE_BODY} (idx={ee_idx})")
    print(f"[IK] Arm joints: {ARM_JOINTS}")
    print(f"[IK] Joint indices: {arm_joint_ids}")

    # Create marker visualization
    markers = MarkerVisualization(num_envs)

    # Get initial EE position
    env.reset()
    env.sim.step()

    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    base_pos_w = robot.data.body_pos_w[:, base_idx, :]
    base_quat_w = robot.data.body_quat_w[:, base_idx, :]

    ee_pos_b, _ = subtract_frame_transforms(
        base_pos_w, base_quat_w,
        ee_pos_w, robot.data.body_quat_w[:, ee_idx, :]
    )

    print(f"[IK] Initial EE pos (base frame): {ee_pos_b[0].cpu().numpy()}")

    # Initialize target at current EE position
    target_pos_b = ee_pos_b.clone()

    # Workspace limits (base frame)
    WORKSPACE = {
        "x": (0.1, 0.5),
        "y": (-0.5, 0.0) if ARM == "right" else (0.0, 0.5),
        "z": (-0.3, 0.3),
    }

    # IK parameters
    IK_GAIN = 1.0
    MAX_DELTA = 0.15
    ACTION_SCALE = 0.5

    # Simulation loop
    print("\n" + "-" * 70)
    print("[Info] Starting simulation...")
    print(f"[Info] Target mode: {args_cli.target_mode}")
    print(f"[Info] Marker visualization: Enabled (USD prims)")
    print("-" * 70 + "\n")

    step = 0
    min_error = float('inf')

    obs_dict, _ = env.reset()

    while simulation_app.is_running():
        step += 1
        t = step * 0.02  # 50 Hz

        # Update target based on mode
        if args_cli.target_mode == "wave":
            target_pos_b[:, 0] = 0.3 + 0.1 * math.sin(2 * math.pi * 0.1 * t)
            target_pos_b[:, 1] = -0.25 + 0.1 * math.cos(2 * math.pi * 0.1 * t)
            target_pos_b[:, 2] = 0.0 + 0.1 * math.sin(2 * math.pi * 0.2 * t)
        elif args_cli.target_mode == "interactive":
            # Slow circular motion for visualization
            target_pos_b[:, 0] = 0.35 + 0.08 * math.sin(2 * math.pi * 0.05 * t)
            target_pos_b[:, 1] = -0.25 + 0.08 * math.cos(2 * math.pi * 0.05 * t)
            target_pos_b[:, 2] = 0.0 + 0.05 * math.sin(2 * math.pi * 0.1 * t)

        # Clamp to workspace
        target_pos_b[:, 0].clamp_(*WORKSPACE["x"])
        target_pos_b[:, 1].clamp_(*WORKSPACE["y"])
        target_pos_b[:, 2].clamp_(*WORKSPACE["z"])

        # Get current state
        ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
        ee_quat_w = robot.data.body_quat_w[:, ee_idx, :]
        base_pos_w = robot.data.body_pos_w[:, base_idx, :]
        base_quat_w = robot.data.body_quat_w[:, base_idx, :]

        # Transform to base frame
        ee_pos_b_curr, ee_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w, ee_pos_w, ee_quat_w
        )

        # Compute IK
        jacobian = robot.root_physx_view.get_jacobians()
        arm_jacobian = jacobian[:, ee_idx - 1, :3, :][:, :, jacobian_ids]

        current_arm_pos = robot.data.joint_pos[:, arm_joint_ids]

        ik_controller.set_command(target_pos_b)
        joint_delta = ik_controller.compute(
            ee_pos_b_curr, ee_quat_b,
            arm_jacobian,
            current_arm_pos
        )

        # Clamp delta
        joint_delta = torch.clamp(joint_delta * IK_GAIN, -MAX_DELTA, MAX_DELTA)
        new_arm_pos = current_arm_pos + joint_delta

        # Get policy action for locomotion
        if policy is not None:
            with torch.no_grad():
                obs = obs_dict["policy"]
                actions = policy.act_inference(obs)
        else:
            actions = torch.zeros(num_envs, env.action_manager.total_action_dim, device=device)

        # Override arm joints
        for i, joint_id in enumerate(arm_joint_ids):
            default_pos = robot.data.default_joint_pos[0, joint_id].item()
            actions[:, joint_id] = (new_arm_pos[:, i] - default_pos) / ACTION_SCALE

        # Step
        obs_dict, reward, terminated, truncated, info = env.step(actions)

        # Convert target to world frame for visualization
        identity_quat = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).expand(num_envs, -1)
        target_pos_w, _ = combine_frame_transforms(
            base_pos_w, base_quat_w,
            target_pos_b, identity_quat
        )

        # Update markers
        markers.update_all(target_pos_w, ee_pos_w)

        # Compute error
        error = torch.norm(ee_pos_b_curr - target_pos_b, dim=-1)
        mean_error = error.mean().item()
        min_error = min(min_error, mean_error)

        # Log
        if step % 100 == 0:
            alive = (1 - terminated.float()).mean().item() * 100
            print(f"[Step {step:5d}] t={t:6.2f}s | Reward: {reward.mean().item():7.3f} | "
                  f"Alive: {alive:5.1f}% | Error: {mean_error:.3f}m (min: {min_error:.3f}m)")

        # Reset if needed
        if terminated.any() or truncated.any():
            obs_dict, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()