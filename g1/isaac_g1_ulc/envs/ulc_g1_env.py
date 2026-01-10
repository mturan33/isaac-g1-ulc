"""
ULC G1 Environment - STAGE 4 COMPATIBLE OBSERVATIONS
=====================================================

Stage 4 Observation (77 total) - MUST MATCH CHECKPOINT:
- lin_vel_b: 3
- ang_vel_b: 3
- proj_gravity: 3
- leg_pos: 12
- leg_vel: 12
- height_cmd: 1
- vel_cmd: 3
- gait_phase: 2
- prev_actions[:, :12]: 12
- torso_cmd: 3
- torso_euler: 3
- left_arm_pos: 5
- right_arm_pos: 5
- left_arm_cmd: 5
- right_arm_cmd: 5

Total: 3+3+3+12+12+1+3+2+12+3+3+5+5+5+5 = 77 ✓

Joint Configuration:
- Legs: 12 joints (hip_pitch/roll/yaw, knee, ankle_pitch/roll x2)
- Arms: 10 joints (shoulder_pitch/roll/yaw, elbow_pitch/roll x2)
- Torso: 1 joint (torso_joint)

Stage 4: 22 actions (12 legs + 10 arms)
"""

from __future__ import annotations

import math
import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from .config.ulc_g1_env_cfg import ULC_G1_EnvCfg


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


class ULC_G1_Env(DirectRLEnv):
    """
    ULC G1 Environment - Unified Loco-Manipulation Controller.

    IMPORTANT: Observation structure MUST match Stage 4 training script exactly!
    """

    cfg: ULC_G1_EnvCfg

    def __init__(self, cfg: ULC_G1_EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Current curriculum stage
        self.current_stage = 1

        # Joint indices setup
        self._setup_joint_indices()

        # Commands buffer - MUST match Stage 4 structure
        self._init_commands()

        # Previous actions for observation (CRITICAL - Stage 4 uses this!)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # Gait phase
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.gait_frequency = 1.5  # Hz

        # Default positions
        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )
        self.default_arm = torch.tensor(
            [0.0, 0.0, 0.0, -0.2, 0.0] * 2,  # Left + Right
            device=self.device
        )

        # Residual scales for arms
        self.residual_scales = torch.tensor(
            [0.5, 0.3, 0.3, 0.4, 0.3] * 2,  # Left + Right
            device=self.device
        )

        # Logging buffers
        self.episode_sums = {}
        for key in ["height_tracking", "orientation", "com_stability",
                    "joint_acceleration", "action_rate", "termination",
                    "vx", "vy", "vyaw", "left_arm", "right_arm"]:
            self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)

        print(f"[ULC_G1_Env] Initialized - Stage {self.current_stage}")
        print(f"[ULC_G1_Env] Observations: {self.cfg.num_observations}, Actions: {self.cfg.num_actions}")

    def _setup_joint_indices(self):
        """Setup joint indices for different body parts - MUST match Stage 4 order."""
        robot = self.scene["robot"]
        joint_names = robot.data.joint_names

        # Leg joint names in EXACT order Stage 4 expects
        leg_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]

        # Left arm joint names
        left_arm_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
        ]

        # Right arm joint names
        right_arm_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ]

        # Build indices
        self.leg_joint_indices = torch.tensor(
            [joint_names.index(n) for n in leg_names if n in joint_names],
            device=self.device, dtype=torch.long
        )

        self.left_arm_indices = torch.tensor(
            [joint_names.index(n) for n in left_arm_names if n in joint_names],
            device=self.device, dtype=torch.long
        )

        self.right_arm_indices = torch.tensor(
            [joint_names.index(n) for n in right_arm_names if n in joint_names],
            device=self.device, dtype=torch.long
        )

        self.arm_joint_indices = torch.cat([self.left_arm_indices, self.right_arm_indices])

        # Torso joint
        self.torso_joint_indices = []
        for i, name in enumerate(joint_names):
            if "torso" in name.lower():
                self.torso_joint_indices.append(i)
        self.torso_joint_indices = torch.tensor(self.torso_joint_indices, device=self.device, dtype=torch.long)

        print(f"[ULC_G1_Env] Leg joints: {len(self.leg_joint_indices)}")
        print(f"[ULC_G1_Env] Left arm joints: {len(self.left_arm_indices)}")
        print(f"[ULC_G1_Env] Right arm joints: {len(self.right_arm_indices)}")
        print(f"[ULC_G1_Env] Torso joints: {len(self.torso_joint_indices)}")

    def _init_commands(self):
        """Initialize command buffers - MUST match Stage 4 structure."""
        # Height command
        self.height_command = torch.ones(self.num_envs, device=self.device) * 0.72

        # Velocity commands (vx, vy, vyaw)
        self.velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Torso commands (roll, pitch, yaw)
        self.torso_commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Arm commands - SEPARATE left and right (Stage 4 format)
        self.left_arm_cmd = torch.zeros(self.num_envs, 5, device=self.device)
        self.right_arm_cmd = torch.zeros(self.num_envs, 5, device=self.device)

        # Combined arm_commands for backward compatibility
        self.arm_commands = torch.zeros(self.num_envs, 10, device=self.device)

    def _setup_scene(self):
        """Setup scene with robot."""
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot

        # Add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and filter
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting - check if exists first
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/Light").IsValid():
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

    def get_torso_euler(self) -> torch.Tensor:
        """Get torso orientation as euler angles."""
        quat = self.robot.data.root_quat_w
        return quat_to_euler_xyz(quat)

    # =========================================================================
    # PRE/POST PHYSICS STEP
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self.actions = torch.clamp(actions, -1.0, 1.0)

        # Split actions: legs (0:12) and arms (12:22)
        leg_actions = self.actions[:, :12]
        arm_actions = self.actions[:, 12:]

        # Compute targets
        target_pos = self.robot.data.default_joint_pos.clone()

        # Legs: direct action
        target_pos[:, self.leg_joint_indices] = self.default_leg + leg_actions * 0.4

        # Arms: RESIDUAL actions (arm_cmd + scale * tanh(residual))
        arm_cmd = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)
        arm_residual = arm_actions * self.residual_scales
        arm_target = arm_cmd + torch.tanh(arm_residual) * self.residual_scales

        target_pos[:, self.arm_joint_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)

        # Update gait phase
        self.gait_phase = (self.gait_phase + self.gait_frequency * self.step_dt) % 1.0

        # Store action history
        self._prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()

    def _apply_action(self):
        """Apply actions - handled by set_joint_position_target."""
        pass

    def _get_observations(self) -> dict:
        """
        Build observation vector - MUST MATCH Stage 4 EXACTLY!

        Stage 4 Observation (77 total):
        - lin_vel_b: 3
        - ang_vel_b: 3
        - proj_gravity: 3
        - leg_pos: 12
        - leg_vel: 12
        - height_cmd: 1
        - vel_cmd: 3
        - gait_phase: 2
        - prev_actions[:, :12]: 12
        - torso_cmd: 3
        - torso_euler: 3
        - left_arm_pos: 5
        - right_arm_pos: 5
        - left_arm_cmd: 5
        - right_arm_cmd: 5

        Total: 77
        """
        robot = self.robot
        quat = robot.data.root_quat_w

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # Projected gravity
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)

        # Leg joint states
        leg_pos = robot.data.joint_pos[:, self.leg_joint_indices]
        leg_vel = robot.data.joint_vel[:, self.leg_joint_indices]

        # Arm joint states
        left_arm_pos = robot.data.joint_pos[:, self.left_arm_indices]
        right_arm_pos = robot.data.joint_pos[:, self.right_arm_indices]

        # Gait phase encoding
        gait_phase = torch.stack([
            torch.sin(2 * np.pi * self.gait_phase),
            torch.cos(2 * np.pi * self.gait_phase)
        ], dim=-1)

        # Torso orientation
        torso_euler = self.get_torso_euler()

        # Build observation - EXACT Stage 4 order!
        obs = torch.cat([
            lin_vel_b,                          # 3
            ang_vel_b,                          # 3
            proj_gravity,                       # 3
            leg_pos,                            # 12
            leg_vel,                            # 12
            self.height_command.unsqueeze(-1),  # 1
            self.velocity_commands,             # 3
            gait_phase,                         # 2
            self.prev_actions[:, :12],          # 12 (leg actions only!)
            self.torso_commands,                # 3
            torso_euler,                        # 3
            left_arm_pos,                       # 5
            right_arm_pos,                      # 5
            self.left_arm_cmd,                  # 5
            self.right_arm_cmd,                 # 5
        ], dim=-1)

        # Verify size
        assert obs.shape[1] == 77, f"Observation size mismatch: {obs.shape[1]} != 77"

        return {"policy": obs.clamp(-10, 10).nan_to_num()}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        robot = self.robot
        quat = robot.data.root_quat_w
        pos = robot.data.root_pos_w

        # Body-frame velocities
        lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

        # Joint states
        leg_pos = robot.data.joint_pos[:, self.leg_joint_indices]
        leg_vel = robot.data.joint_vel[:, self.leg_joint_indices]
        left_arm_pos = robot.data.joint_pos[:, self.left_arm_indices]
        right_arm_pos = robot.data.joint_pos[:, self.right_arm_indices]

        # Torso orientation
        torso_euler = self.get_torso_euler()

        rewards = {}

        # ==================== VELOCITY TRACKING ====================
        rewards["vx"] = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.velocity_commands[:, 0]) ** 2)
        rewards["vy"] = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.velocity_commands[:, 1]) ** 2)
        rewards["vyaw"] = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.velocity_commands[:, 2]) ** 2)

        # ==================== HEIGHT TRACKING ====================
        rewards["height_tracking"] = torch.exp(-10.0 * (pos[:, 2] - self.height_command) ** 2)

        # ==================== BASE ORIENTATION ====================
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(quat, gravity_vec)
        base_tilt_error = proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2
        rewards["orientation"] = torch.exp(-3.0 * base_tilt_error)

        # ==================== TORSO TRACKING ====================
        roll_err = (torso_euler[:, 0] - self.torso_commands[:, 0]) ** 2
        pitch_err = (torso_euler[:, 1] - self.torso_commands[:, 1]) ** 2
        yaw_err = (torso_euler[:, 2] - self.torso_commands[:, 2]) ** 2

        rewards["torso_roll"] = torch.exp(-5.0 * roll_err)
        rewards["torso_pitch"] = torch.exp(-5.0 * pitch_err)
        rewards["torso_yaw"] = torch.exp(-3.0 * yaw_err)

        # ==================== ARM TRACKING ====================
        left_arm_err = (left_arm_pos - self.left_arm_cmd).pow(2).sum(-1)
        right_arm_err = (right_arm_pos - self.right_arm_cmd).pow(2).sum(-1)

        rewards["left_arm"] = torch.exp(-3.0 * left_arm_err)
        rewards["right_arm"] = torch.exp(-3.0 * right_arm_err)

        # ==================== GAIT QUALITY ====================
        left_knee, right_knee = leg_pos[:, 6], leg_pos[:, 7]
        phase = self.gait_phase
        left_swing = (phase < 0.5).float()
        right_swing = (phase >= 0.5).float()

        knee_target_swing = 0.6
        knee_target_stance = 0.3
        knee_err = (
            (left_knee - (left_swing * knee_target_swing + (1 - left_swing) * knee_target_stance)) ** 2 +
            (right_knee - (right_swing * knee_target_swing + (1 - right_swing) * knee_target_stance)) ** 2
        )
        rewards["gait"] = torch.exp(-3.0 * knee_err)

        # ==================== CoM STABILITY ====================
        xy_velocity = torch.norm(lin_vel_b[:, :2], dim=-1)
        rewards["com_stability"] = torch.exp(-2.0 * xy_velocity ** 2)

        # ==================== PENALTIES ====================
        # Leg action smoothness
        leg_action_diff = self.actions[:, :12] - self._prev_actions[:, :12]
        rewards["smooth_legs"] = leg_action_diff.pow(2).sum(-1)

        # Arm action smoothness (gentler)
        arm_action_diff = self.actions[:, 12:] - self._prev_actions[:, 12:]
        rewards["smooth_arms"] = arm_action_diff.pow(2).sum(-1)

        # Torque penalty
        rewards["torque"] = (leg_vel.abs() * self.actions[:, :12].abs()).sum(-1)

        # ==================== TOTAL REWARD ====================
        reward_weights = {
            "vx": 2.5,
            "vy": 1.5,
            "vyaw": 1.5,
            "gait": 2.0,
            "height_tracking": 2.0,
            "orientation": 1.5,
            "torso_pitch": 3.0,
            "torso_roll": 2.0,
            "torso_yaw": 1.5,
            "left_arm": 4.0,
            "right_arm": 4.0,
            "com_stability": 4.0,
            "smooth_legs": -0.01,
            "smooth_arms": -0.005,
            "torque": -0.0003,
        }

        total_reward = torch.zeros(self.num_envs, device=self.device)
        for key, weight in reward_weights.items():
            if key in rewards:
                total_reward += weight * rewards[key]

        # Alive bonus
        total_reward += 0.5

        # Store extras for logging
        self.extras = {
            "R/vx": rewards["vx"].mean().item(),
            "R/height": rewards["height_tracking"].mean().item(),
            "R/left_arm": rewards["left_arm"].mean().item(),
            "R/right_arm": rewards["right_arm"].mean().item(),
            "R/com_stability": rewards["com_stability"].mean().item(),
            "M/height": pos[:, 2].mean().item(),
            "M/vx": lin_vel_b[:, 0].mean().item(),
            "M/pitch": torso_euler[:, 1].mean().item(),
            "M/left_arm_err": left_arm_err.mean().item(),
            "M/right_arm_err": right_arm_err.mean().item(),
        }

        return total_reward.clamp(-10, 35)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        robot = self.robot
        base_pos = robot.data.root_pos_w
        base_quat = robot.data.root_quat_w

        # Height termination
        height = base_pos[:, 2]
        too_low = height < self.cfg.termination["base_height_min"]
        too_high = height > self.cfg.termination["base_height_max"]

        # Orientation termination
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        proj_gravity = quat_apply_inverse(base_quat, gravity_vec)

        bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

        # Combine terminations
        terminated = too_low | too_high | bad_orientation

        # Time-out
        time_out = self.episode_length_buf >= self.max_episode_length

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        robot = self.robot
        n = len(env_ids)

        # Reset pose
        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

        robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        # Reset joint positions
        default_joint_pos = robot.data.default_joint_pos[env_ids]
        robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

        # Reset commands
        self.height_command[env_ids] = 0.72
        self.velocity_commands[env_ids] = 0.0
        self.torso_commands[env_ids] = 0.0
        self.left_arm_cmd[env_ids] = 0.0
        self.right_arm_cmd[env_ids] = 0.0
        self.arm_commands[env_ids] = 0.0

        # Reset gait phase
        self.gait_phase[env_ids] = torch.rand(n, device=self.device)

        # Reset action history
        self.prev_actions[env_ids] = 0
        self._prev_actions[env_ids] = 0

    # =========================================================================
    # CURRICULUM METHODS
    # =========================================================================

    def set_stage(self, stage: int):
        """Set curriculum stage."""
        if stage != self.current_stage:
            print(f"[ULC_G1_Env] Advancing to Stage {stage}")
            self.current_stage = stage

    def sample_commands(self):
        """Sample new commands based on current stage."""
        if self.current_stage >= 2:
            cmd_cfg = self.cfg.commands["velocity_range"]
            self.velocity_commands[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["vx"])
            self.velocity_commands[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["vy"])
            self.velocity_commands[:, 2] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["yaw_rate"])

        if self.current_stage >= 3:
            cmd_cfg = self.cfg.commands["torso_range"]
            self.torso_commands[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["roll"])
            self.torso_commands[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["pitch"])
            self.torso_commands[:, 2] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["yaw"])

        if self.current_stage >= 4:
            arm_range = self.cfg.commands["arm_range"]
            self.left_arm_cmd = torch.empty(self.num_envs, 5, device=self.device).uniform_(*arm_range)
            self.right_arm_cmd = torch.empty(self.num_envs, 5, device=self.device).uniform_(*arm_range)
            self.arm_commands = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)