# Copyright (c) 2025, VLM-RL G1 Project
# G1 Hierarchical Control with Debug Visualization
#
# Features:
# - Blue sphere: Target position
# - Green sphere: Current EE position
# - Red line: Error vector
# - Workspace clamping: If target is unreachable, move to closest point

"""
G1 IK with Debug Visualization
==============================

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\g1_ik_debug_sphere.py --num_envs 4 --load_run 2025-12-27_00-29-54 --target_mode interactive
"""

import argparse
import os
import math
import torch
import torch.nn as nn
from typing import List, Tuple

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 IK with Debug Visualization")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--target_mode", type=str, default="interactive",
                    choices=["interactive", "circle", "static", "reach_test"])
parser.add_argument("--arm", type=str, default="right", choices=["left", "right"])
parser.add_argument("--max_joint_delta", type=float, default=0.15)
parser.add_argument("--ik_gain", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_conjugate
from isaaclab.envs import ManagerBasedRLEnv

# Debug visualization
try:
    from isaacsim.util.debug_draw import _debug_draw

    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    try:
        import omni.isaac.debug_draw._debug_draw as _debug_draw

        DEBUG_DRAW_AVAILABLE = True
    except ImportError:
        DEBUG_DRAW_AVAILABLE = False
        print("[WARNING] Debug draw not available")

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

G1_ARM_DEFAULT_POS = {
    "right": [0.35, -0.16, 0.0, 0.87, 0.0],
    "left": [0.35, 0.16, 0.0, 0.87, 0.0],
}

# Workspace limits (in base frame) - conservative estimates
G1_WORKSPACE = {
    "right": {
        "x_min": -0.1, "x_max": 0.5,  # Forward/backward
        "y_min": -0.5, "y_max": 0.1,  # Left/right (right arm)
        "z_min": -0.4, "z_max": 0.5,  # Up/down
        "reach_max": 0.55,  # Maximum reach from shoulder
    },
    "left": {
        "x_min": -0.1, "x_max": 0.5,
        "y_min": -0.1, "y_max": 0.5,
        "z_min": -0.4, "z_max": 0.5,
        "reach_max": 0.55,
    }
}

# Shoulder position approximation (in base frame)
SHOULDER_POS = {
    "right": torch.tensor([0.0, -0.16, 0.3]),
    "left": torch.tensor([0.0, 0.16, 0.3]),
}


##############################################################################
# DEBUG VISUALIZATION
##############################################################################

class DebugVisualizer:
    """Draw debug spheres and lines for IK visualization."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.draw = None

        if DEBUG_DRAW_AVAILABLE:
            self.draw = _debug_draw.acquire_debug_draw_interface()
            print("[DEBUG VIS] Debug draw interface acquired!")
        else:
            print("[DEBUG VIS] Debug draw not available - visual only")

    def clear(self):
        """Clear all debug drawings."""
        if self.draw:
            self.draw.clear_lines()
            self.draw.clear_points()

    def draw_target_sphere(self, pos_world: torch.Tensor,
                           color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0), size: float = 0.03):
        """Draw target position as a sphere (blue by default)."""
        if not self.draw:
            return

        # Draw for first environment only
        pos = pos_world[0].cpu().numpy()
        self.draw.draw_points(
            [pos.tolist()],
            [color],
            [size * 100]  # Size in pixels
        )

    def draw_ee_sphere(self, pos_world: torch.Tensor, color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                       size: float = 0.02):
        """Draw current EE position as a sphere (green by default)."""
        if not self.draw:
            return

        pos = pos_world[0].cpu().numpy()
        self.draw.draw_points(
            [pos.tolist()],
            [color],
            [size * 100]
        )

    def draw_error_line(self, start: torch.Tensor, end: torch.Tensor,
                        color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)):
        """Draw line from current to target (red by default)."""
        if not self.draw:
            return

        start_pos = start[0].cpu().numpy().tolist()
        end_pos = end[0].cpu().numpy().tolist()
        self.draw.draw_lines(
            [start_pos],
            [end_pos],
            [color],
            [2.0]  # Line width
        )

    def draw_workspace_bounds(self, arm: str, root_pos: torch.Tensor):
        """Draw workspace boundary box."""
        if not self.draw:
            return

        ws = G1_WORKSPACE[arm]
        root = root_pos[0].cpu().numpy()

        # Draw corners of workspace box
        corners = [
            [root[0] + ws["x_min"], root[1] + ws["y_min"], root[2] + ws["z_min"]],
            [root[0] + ws["x_max"], root[1] + ws["y_min"], root[2] + ws["z_min"]],
            [root[0] + ws["x_max"], root[1] + ws["y_max"], root[2] + ws["z_min"]],
            [root[0] + ws["x_min"], root[1] + ws["y_max"], root[2] + ws["z_min"]],
            [root[0] + ws["x_min"], root[1] + ws["y_min"], root[2] + ws["z_max"]],
            [root[0] + ws["x_max"], root[1] + ws["y_min"], root[2] + ws["z_max"]],
            [root[0] + ws["x_max"], root[1] + ws["y_max"], root[2] + ws["z_max"]],
            [root[0] + ws["x_min"], root[1] + ws["y_max"], root[2] + ws["z_max"]],
        ]

        # Draw edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical
        ]

        for i, j in edges:
            self.draw.draw_lines(
                [corners[i]],
                [corners[j]],
                [(0.5, 0.5, 0.5, 0.5)],  # Gray, semi-transparent
                [1.0]
            )


##############################################################################
# WORKSPACE CLAMPING
##############################################################################

def clamp_to_workspace(target_base: torch.Tensor, arm: str, device: str) -> Tuple[torch.Tensor, bool]:
    """
    Clamp target position to reachable workspace.
    Returns (clamped_target, was_clamped).
    """
    ws = G1_WORKSPACE[arm]
    shoulder = SHOULDER_POS[arm].to(device)

    clamped = target_base.clone()
    was_clamped = False

    # Box constraints
    x_clamped = torch.clamp(clamped[:, 0], ws["x_min"], ws["x_max"])
    y_clamped = torch.clamp(clamped[:, 1], ws["y_min"], ws["y_max"])
    z_clamped = torch.clamp(clamped[:, 2], ws["z_min"], ws["z_max"])

    if not torch.allclose(clamped[:, 0], x_clamped) or \
            not torch.allclose(clamped[:, 1], y_clamped) or \
            not torch.allclose(clamped[:, 2], z_clamped):
        was_clamped = True

    clamped[:, 0] = x_clamped
    clamped[:, 1] = y_clamped
    clamped[:, 2] = z_clamped

    # Spherical reach constraint
    to_target = clamped - shoulder.unsqueeze(0)
    dist = torch.norm(to_target, dim=1, keepdim=True)

    if (dist > ws["reach_max"]).any():
        # Scale back to max reach
        scale = ws["reach_max"] / dist.clamp(min=0.01)
        mask = dist > ws["reach_max"]
        to_target_scaled = to_target * torch.where(mask, scale, torch.ones_like(scale))
        clamped = shoulder.unsqueeze(0) + to_target_scaled
        was_clamped = True

    return clamped, was_clamped


##############################################################################
# WORKING IK CONTROLLER
##############################################################################

class G1ArmIKController:
    """Position-only IK controller for G1 arm with workspace clamping."""

    def __init__(self, num_envs: int, device: str, arm: str = "right",
                 ik_method: str = "dls", max_joint_delta: float = 0.15, ik_gain: float = 1.0):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm
        self.ik_method = ik_method
        self.max_joint_delta = max_joint_delta
        self.ik_gain = ik_gain

        self.ee_body_name = G1_EE_BODIES[arm]
        self.arm_joint_names = G1_ARM_JOINTS[arm]

        self.ee_body_idx = None
        self.arm_joint_ids = []
        self.jacobian_cols = []

        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_pos_clamped = torch.zeros(num_envs, 3, device=device)
        self.was_clamped = False

        self.joint_lower = None
        self.joint_upper = None
        self.prev_joint_pos = None
        self.initialized = False
        self.step_count = 0

        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=False,
                ik_method=ik_method,
                ik_params={
                    "k_val": 1.0,
                    "lambda_val": 0.05,
                },
            ),
            num_envs=num_envs,
            device=device,
        )

        print(f"\n{'=' * 60}")
        print(f"[IK] Position-only DifferentialIK with Workspace Clamping")
        print(f"[IK] Arm: {arm}, Method: {ik_method}")
        print(f"[IK] Max delta: {max_joint_delta} rad/step")
        print(f"[IK] IK gain: {ik_gain}")
        print(f"{'=' * 60}")

    def initialize(self, robot):
        """Initialize from robot data."""
        if self.initialized:
            return

        # Find EE body
        body_names = robot.data.body_names
        if self.ee_body_name in body_names:
            self.ee_body_idx = body_names.index(self.ee_body_name)
        else:
            print(f"[ERROR] EE body '{self.ee_body_name}' not found!")
            return

        # Find arm joints
        joint_names = robot.data.joint_names
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            if name in joint_names:
                idx = joint_names.index(name)
                self.arm_joint_ids.append(idx)
                self.jacobian_cols.append(idx + 6)  # +6 for floating base

        if len(self.arm_joint_ids) != 5:
            print(f"[ERROR] Expected 5 arm joints, found {len(self.arm_joint_ids)}")
            return

        # Joint limits
        self.joint_lower = robot.data.soft_joint_pos_limits[0, self.arm_joint_ids, 0].clone()
        self.joint_upper = robot.data.soft_joint_pos_limits[0, self.arm_joint_ids, 1].clone()

        # Initialize prev position
        self.prev_joint_pos = robot.data.joint_pos[:, self.arm_joint_ids].clone()

        # Get initial EE position
        ee_pos_b = self.get_ee_pos_base(robot)

        print(f"[IK INIT] EE body index: {self.ee_body_idx}")
        print(f"[IK INIT] Arm joint indices: {self.arm_joint_ids}")
        print(f"[IK INIT] Initial EE pos (base): [{ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}]")
        print(f"[IK INIT] ✓ Initialization complete!")

        self.initialized = True

    def set_target(self, target_pos_base: torch.Tensor, clamp: bool = True):
        """Set target position with optional workspace clamping."""
        self.target_pos = target_pos_base.clone()

        if clamp:
            self.target_pos_clamped, self.was_clamped = clamp_to_workspace(
                target_pos_base, self.arm, self.device
            )
        else:
            self.target_pos_clamped = target_pos_base.clone()
            self.was_clamped = False

    def get_clamped_target(self) -> torch.Tensor:
        """Get workspace-clamped target."""
        return self.target_pos_clamped

    def _transform_jacobian_to_base_frame(self, jacobian_w, root_quat_w):
        """Transform Jacobian from world to base frame."""
        rot_matrix = matrix_from_quat(quat_conjugate(root_quat_w))
        batch_size = jacobian_w.shape[0]

        jacobian_b = jacobian_w.clone()
        for i in range(batch_size):
            jacobian_b[i, :3, :] = rot_matrix[i] @ jacobian_w[i, :3, :]
            jacobian_b[i, 3:6, :] = rot_matrix[i] @ jacobian_w[i, 3:6, :]

        return jacobian_b

    def compute(self, robot) -> torch.Tensor:
        """Compute arm joint positions using IK."""
        if not self.initialized:
            self.initialize(robot)
            if not self.initialized:
                return robot.data.joint_pos[:, self.arm_joint_ids]

        self.step_count += 1

        try:
            # Get Jacobian
            jacobian_w = robot.root_physx_view.get_jacobians()[:, self.ee_body_idx - 1, :, :]
            jacobian_arm_w = jacobian_w[:, :, self.jacobian_cols]

            # Get current state
            ee_pos_w = robot.data.body_state_w[:, self.ee_body_idx, 0:3]
            ee_quat_w = robot.data.body_state_w[:, self.ee_body_idx, 3:7]
            root_pos_w = robot.data.root_state_w[:, 0:3]
            root_quat_w = robot.data.root_state_w[:, 3:7]

            # Transform to base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            current_joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]
            jacobian_arm_b = self._transform_jacobian_to_base_frame(jacobian_arm_w, root_quat_w)

            # Use clamped target
            joint_pos_ik = self.controller.compute(
                self.target_pos_clamped, ee_quat_b, jacobian_arm_b, current_joint_pos
            )

            # Calculate and clamp delta
            raw_delta = joint_pos_ik - current_joint_pos
            scaled_delta = raw_delta * self.ik_gain
            clamped_delta = torch.clamp(scaled_delta, -self.max_joint_delta, self.max_joint_delta)

            # Apply delta
            new_joint_pos = current_joint_pos + clamped_delta
            new_joint_pos = torch.clamp(new_joint_pos, self.joint_lower, self.joint_upper)

            self.prev_joint_pos = new_joint_pos.clone()
            return new_joint_pos

        except Exception as e:
            print(f"[IK ERROR] {e}")
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

    def get_ee_pos_world(self, robot) -> torch.Tensor:
        """Get EE position in world frame."""
        return robot.data.body_state_w[:, self.ee_body_idx, 0:3]

    def reset(self, env_ids: torch.Tensor = None):
        """Reset controller."""
        if env_ids is None:
            self.target_pos.zero_()
            self.target_pos_clamped.zero_()
            if self.prev_joint_pos is not None:
                default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)
                self.prev_joint_pos = default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        else:
            self.target_pos[env_ids] = 0.0
            self.target_pos_clamped[env_ids] = 0.0
            if self.prev_joint_pos is not None:
                default_pos = torch.tensor(G1_ARM_DEFAULT_POS[self.arm], device=self.device)
                self.prev_joint_pos[env_ids] = default_pos

        self.controller.reset(env_ids)


##############################################################################
# INTERACTIVE TARGET GENERATOR
##############################################################################

class InteractiveTargetGenerator:
    """Generate targets with debug visualization and workspace clamping."""

    def __init__(self, num_envs: int, device: str, mode: str = "interactive", arm: str = "right"):
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.arm = arm
        self.initial_ee_pos = None
        self.initialized = False

    def initialize_from_ee(self, ee_pos: torch.Tensor):
        """Initialize from current EE position."""
        self.initial_ee_pos = ee_pos.clone()
        self.initialized = True
        print(f"[TARGET] Initial EE: [{ee_pos[0, 0]:.3f}, {ee_pos[0, 1]:.3f}, {ee_pos[0, 2]:.3f}]")

    def get_target(self, time: float) -> torch.Tensor:
        """Get target position based on mode."""
        if not self.initialized:
            return torch.tensor([[0.1, -0.2, 0.0]], device=self.device).expand(self.num_envs, -1)

        pos = self.initial_ee_pos.clone()

        if self.mode == "interactive":
            # Interactive mode: cycle through different reaches
            cycle = int(time / 10.0) % 4  # Change every 10 seconds
            phase = (time % 10.0) / 10.0  # 0-1 within cycle

            if cycle == 0:
                # Forward reach
                pos[:, 0] += phase * 0.15
                pos[:, 2] += phase * 0.1
            elif cycle == 1:
                # Side reach
                pos[:, 1] += phase * 0.1 * (-1 if self.arm == "right" else 1)
            elif cycle == 2:
                # Up reach
                pos[:, 2] += phase * 0.15
            else:
                # Circle
                angle = 2 * math.pi * phase
                pos[:, 0] += 0.08 * math.cos(angle)
                pos[:, 2] += 0.08 * math.sin(angle)

        elif self.mode == "circle":
            angle = 2 * math.pi * 0.2 * time
            pos[:, 0] += 0.10 * math.cos(angle)
            pos[:, 2] += 0.10 * math.sin(angle)

        elif self.mode == "reach_test":
            # Progressive reach test - goes beyond workspace
            t = min(time / 20.0, 1.0)  # 20 seconds to max
            pos[:, 0] += t * 0.6  # Will exceed workspace!
            pos[:, 2] += t * 0.3

        return pos


##############################################################################
# POLICY LOADER
##############################################################################

class PolicyWrapper(nn.Module):
    def __init__(self, actor_mlp):
        super().__init__()
        self.actor_mlp = actor_mlp

    def forward(self, obs):
        return self.actor_mlp(obs)

    @torch.no_grad()
    def act_inference(self, obs):
        return self.forward(obs)


def load_policy(run_dir: str, checkpoint: str = None, device: str = "cuda:0"):
    """Load trained PPO policy."""
    if checkpoint:
        model_path = os.path.join(run_dir, checkpoint)
    else:
        model_path = os.path.join(run_dir, "model_19999.pt")

    if not os.path.exists(model_path):
        return None

    print(f"[Policy] Loading: {model_path}")
    checkpoint_data = torch.load(model_path, map_location=device, weights_only=True)

    actor_state = checkpoint_data["model_state_dict"]
    actor_keys = [k for k in actor_state.keys() if k.startswith("actor")]

    if not actor_keys:
        return None

    first_key = actor_keys[0]
    input_dim = actor_state[first_key].shape[1] if len(actor_state[first_key].shape) > 1 else None

    # Build MLP
    layers = []
    layer_pairs = []

    for key in sorted(actor_keys):
        if "weight" in key:
            layer_num = int(key.split(".")[1])
            weight_key = f"actor.{layer_num}.weight"
            bias_key = f"actor.{layer_num}.bias"
            if weight_key in actor_state and bias_key in actor_state:
                layer_pairs.append((layer_num, weight_key, bias_key))

    layer_pairs.sort(key=lambda x: x[0])

    for i, (layer_num, weight_key, bias_key) in enumerate(layer_pairs):
        weight = actor_state[weight_key]
        bias = actor_state[bias_key]

        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data = weight
        linear.bias.data = bias
        layers.append(linear)

        if i < len(layer_pairs) - 1:
            layers.append(nn.ELU())

    if not layers:
        return None

    actor_mlp = nn.Sequential(*layers).to(device)
    actor_mlp.eval()

    policy = PolicyWrapper(actor_mlp)
    print("[Policy] ✓ Loaded!")
    return policy


##############################################################################
# MAIN
##############################################################################

def main():
    print("\n" + "=" * 70)
    print("  G1 IK with Debug Visualization")
    print("  ")
    print("  Features:")
    print("  - Blue sphere: Target position")
    print("  - Green sphere: Current EE position")
    print("  - Red line: Error vector")
    print("  - Workspace clamping: Auto-limit to reachable area")
    print("=" * 70)

    # Create environment using isaaclab method
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dict, info = env.reset()
    obs = obs_dict["policy"]

    device = env.device
    num_envs = env.num_envs

    print(f"[Env] Obs: {obs.shape[1]}, Actions: {env.action_space.shape[1]}")

    # Get robot
    robot = None
    scene = env.scene
    if hasattr(scene, 'articulations') and 'robot' in scene.articulations:
        robot = scene.articulations['robot']
        print(f"[Env] ✓ Robot found")

    # Load policy
    log_root = os.path.join(os.getcwd(), "logs", "rsl_rl", "g1_flat")
    run_dir = os.path.join(log_root, args_cli.load_run)
    policy = load_policy(run_dir, args_cli.checkpoint, device)

    # Create IK controller
    arm_ik = G1ArmIKController(
        num_envs=num_envs,
        device=device,
        arm=args_cli.arm,
        ik_method="dls",
        max_joint_delta=args_cli.max_joint_delta,
        ik_gain=args_cli.ik_gain,
    )

    # Create target generator
    target_gen = InteractiveTargetGenerator(
        num_envs=num_envs,
        device=device,
        mode=args_cli.target_mode,
        arm=args_cli.arm,
    )

    # Create debug visualizer
    debug_vis = DebugVisualizer(num_envs, device)

    # Initialize
    if robot is not None:
        arm_ik.initialize(robot)
        ee_pos_b = arm_ik.get_ee_pos_base(robot)
        target_gen.initialize_from_ee(ee_pos_b)

    print("\n" + "-" * 70)
    print(f"[Info] Starting simulation...")
    print(f"[Info] Target mode: {args_cli.target_mode}")
    print(f"[Info] Debug visualization: {'Enabled' if DEBUG_DRAW_AVAILABLE else 'Disabled'}")
    print("-" * 70 + "\n")

    # Simulation loop
    actions = torch.zeros(num_envs, env.action_space.shape[1], device=device)
    sim_time = 0.0
    step_count = 0
    dt = 0.02

    error_history = []
    min_error = float('inf')

    try:
        while simulation_app.is_running():
            # Clear previous debug drawings
            debug_vis.clear()

            # Get target
            target_pos_base = target_gen.get_target(sim_time)

            # PPO for lower body
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs)
                    if robot is not None and arm_ik.initialized:
                        for joint_idx in arm_ik.arm_joint_ids:
                            actions[:, joint_idx] = 0.0
            else:
                actions.zero_()

            # IK for upper body
            if robot is not None and arm_ik.initialized:
                # Set target with workspace clamping
                arm_ik.set_target(target_pos_base, clamp=True)

                # Compute IK
                joint_pos_des = arm_ik.compute(robot)

                # Get clamped target for visualization
                target_clamped = arm_ik.get_clamped_target()

                # Convert to action space
                ACTION_SCALE = 0.5
                default_arm_pos = torch.tensor(
                    G1_ARM_DEFAULT_POS[args_cli.arm],
                    device=device
                ).unsqueeze(0).expand(num_envs, -1)

                arm_action = (joint_pos_des - default_arm_pos) / ACTION_SCALE

                for i, joint_idx in enumerate(arm_ik.arm_joint_ids):
                    actions[:, joint_idx] = arm_action[:, i]

                # Debug visualization
                ee_pos_w = arm_ik.get_ee_pos_world(robot)
                root_pos_w = robot.data.root_state_w[:, 0:3]
                root_quat_w = robot.data.root_state_w[:, 3:7]

                # Convert target to world frame for visualization
                from isaaclab.utils.math import quat_apply
                target_world = root_pos_w + quat_apply(root_quat_w, target_clamped)
                original_target_world = root_pos_w + quat_apply(root_quat_w, target_pos_base)

                # Draw spheres
                debug_vis.draw_target_sphere(target_world, color=(0.0, 0.0, 1.0, 1.0),
                                             size=0.04)  # Blue - clamped target
                debug_vis.draw_ee_sphere(ee_pos_w, color=(0.0, 1.0, 0.0, 1.0), size=0.03)  # Green - current EE

                # If target was clamped, also show original (orange)
                if arm_ik.was_clamped:
                    debug_vis.draw_target_sphere(original_target_world, color=(1.0, 0.5, 0.0, 0.5),
                                                 size=0.03)  # Orange - original

                # Draw error line
                debug_vis.draw_error_line(ee_pos_w, target_world, color=(1.0, 0.0, 0.0, 1.0))  # Red line

            # Step environment
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

                if robot is not None and arm_ik.initialized:
                    ee_pos_b = arm_ik.get_ee_pos_base(robot)[0]
                    target_clamped = arm_ik.get_clamped_target()[0]
                    error = torch.norm(target_clamped - ee_pos_b).item()
                    error_history.append(error)

                    if error < min_error:
                        min_error = error

                    clamped_str = " [CLAMPED]" if arm_ik.was_clamped else ""
                    print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                          f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                          f"Error: {error:.3f}m (min: {min_error:.3f}m){clamped_str}")
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