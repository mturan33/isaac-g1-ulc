"""
ULC G1 Environment
==================

Unified Loco-Manipulation Controller için ana environment.
Stage-based training ile Standing → Locomotion → Torso → Arms sırası.

IMPORTANT: Bu dosya Stage 1 (Standing) için optimize edilmiş.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from .config.ulc_g1_env_cfg import ULC_G1_EnvCfg


class ULC_G1_Env(DirectRLEnv):
    """
    ULC G1 Environment - Unified Loco-Manipulation Controller.

    Stage 1: Standing - Temel denge, height tracking
    Stage 2: Locomotion - Velocity tracking eklenir
    Stage 3: Torso - Gövde orientation tracking
    Stage 4: Arms - Dual-arm position tracking
    Stage 5: Full - Tüm komutlar + domain randomization
    """

    cfg: ULC_G1_EnvCfg

    def __init__(self, cfg: ULC_G1_EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Current curriculum stage
        self.current_stage = 1

        # Commands buffer
        self._init_commands()

        # Previous actions for smoothness reward
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # Joint indices for different body parts
        self._setup_joint_indices()

        # CoM tracker (simplified for Stage 1)
        self._setup_com_tracker()

        # Logging buffers
        self.episode_sums = {}
        for key in ["height_tracking", "orientation", "com_stability",
                    "joint_acceleration", "action_rate", "termination"]:
            self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)

        print(f"[ULC_G1_Env] Initialized - Stage {self.current_stage}")
        print(f"[ULC_G1_Env] Observations: {self.cfg.num_observations}, Actions: {self.cfg.num_actions}")

    def _setup_joint_indices(self):
        """Setup joint indices for different body parts."""
        robot = self.scene["robot"]
        joint_names = robot.data.joint_names

        # Find joint indices by pattern matching
        self.leg_joint_indices = []
        self.arm_joint_indices = []
        self.waist_joint_indices = []

        for i, name in enumerate(joint_names):
            name_lower = name.lower()
            if any(x in name_lower for x in ["hip", "knee", "ankle"]):
                self.leg_joint_indices.append(i)
            elif any(x in name_lower for x in ["shoulder", "elbow", "wrist"]):
                self.arm_joint_indices.append(i)
            elif "waist" in name_lower:
                self.waist_joint_indices.append(i)

        self.leg_joint_indices = torch.tensor(self.leg_joint_indices, device=self.device, dtype=torch.long)
        self.arm_joint_indices = torch.tensor(self.arm_joint_indices, device=self.device, dtype=torch.long)
        self.waist_joint_indices = torch.tensor(self.waist_joint_indices, device=self.device, dtype=torch.long)

        print(f"[ULC_G1_Env] Leg joints: {len(self.leg_joint_indices)}")
        print(f"[ULC_G1_Env] Arm joints: {len(self.arm_joint_indices)}")
        print(f"[ULC_G1_Env] Waist joints: {len(self.waist_joint_indices)}")

    def _setup_com_tracker(self):
        """Setup CoM tracker."""
        # Simple CoM tracking using base position
        self.target_height = 0.75  # meters
        self.target_com_xy = torch.zeros(self.num_envs, 2, device=self.device)

    def _init_commands(self):
        """Initialize command buffers."""
        # Height command (Stage 1+)
        self.height_command = torch.ones(self.num_envs, 1, device=self.device) * 0.75

        # Velocity commands (Stage 2+)
        self.velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)  # vx, vy, yaw_rate

        # Torso commands (Stage 3+)
        self.torso_commands = torch.zeros(self.num_envs, 3, device=self.device)  # roll, pitch, yaw

        # Arm commands (Stage 4+)
        self.arm_commands = torch.zeros(self.num_envs, 14, device=self.device)  # 7 left + 7 right

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

        # Lighting - check if exists
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/Light").IsValid():
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

    # =========================================================================
    # PRE/POST PHYSICS STEP
    # =========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        # Clamp actions
        self.actions = torch.clamp(actions, -1.0, 1.0)

        # Scale actions to joint targets
        action_scale = 0.5  # radians

        # Get current joint positions
        current_pos = self.robot.data.joint_pos

        # Compute targets based on stage
        if self.current_stage == 1:
            # Stage 1: Only leg joints
            targets = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
            if len(self.leg_joint_indices) > 0:
                leg_actions = self.actions[:, :len(self.leg_joint_indices)]
                targets[:, self.leg_joint_indices] = leg_actions * action_scale
        else:
            # Higher stages: all joints (based on action dim)
            targets = self.actions * action_scale

        # Apply as position targets
        self.robot.set_joint_position_target(targets)

        # Store for action rate reward
        self.previous_actions = self.actions.clone()

    def _apply_action(self):
        """Apply actions - handled by set_joint_position_target."""
        pass

    def _get_observations(self) -> dict:
        """Compute observations."""
        robot = self.robot

        # Base state
        base_pos = robot.data.root_pos_w
        base_quat = robot.data.root_quat_w
        base_lin_vel = robot.data.root_lin_vel_w
        base_ang_vel = robot.data.root_ang_vel_w

        # Transform velocities to base frame
        base_lin_vel_b = quat_rotate_inverse(base_quat, base_lin_vel)
        base_ang_vel_b = quat_rotate_inverse(base_quat, base_ang_vel)

        # Projected gravity (in base frame)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

        # Joint state
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        # Build observation based on stage
        if self.current_stage == 1:
            # Stage 1: Simplified observation
            # Base velocities (6) + gravity (3) + leg joints (12) + leg velocities (12) + height cmd (1) + prev actions (12)
            obs_list = [
                base_lin_vel_b,  # 3
                base_ang_vel_b,  # 3
                projected_gravity,  # 3
                joint_pos[:, self.leg_joint_indices] if len(self.leg_joint_indices) > 0 else joint_pos[:, :12],  # 12
                joint_vel[:, self.leg_joint_indices] if len(self.leg_joint_indices) > 0 else joint_vel[:, :12],  # 12
                self.height_command,  # 1
                self.previous_actions[:, :12] if self.previous_actions.shape[1] >= 12 else self.previous_actions,  # 12
            ]
        else:
            # Full observation for higher stages
            obs_list = [
                base_lin_vel_b,
                base_ang_vel_b,
                projected_gravity,
                joint_pos,
                joint_vel,
                self.height_command,
                self.velocity_commands,
                self.torso_commands,
                self.arm_commands,
                self.previous_actions,
            ]

        obs = torch.cat(obs_list, dim=-1)

        # Clamp and handle NaN
        obs = torch.clamp(obs, -100.0, 100.0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        robot = self.robot

        # Base state
        base_pos = robot.data.root_pos_w
        base_quat = robot.data.root_quat_w
        base_lin_vel = robot.data.root_lin_vel_w

        # Joint state
        joint_vel = robot.data.joint_vel
        joint_acc = (joint_vel - getattr(self, '_prev_joint_vel', joint_vel)) / self.step_dt
        self._prev_joint_vel = joint_vel.clone()

        rewards = {}

        # =====================================================================
        # HEIGHT TRACKING REWARD
        # =====================================================================
        height = base_pos[:, 2]
        target_height = self.height_command.squeeze(-1)
        height_error = torch.abs(height - target_height)
        rewards["height_tracking"] = torch.exp(-10.0 * height_error ** 2)

        # =====================================================================
        # ORIENTATION REWARD (stay upright)
        # =====================================================================
        # Extract roll and pitch from quaternion
        # Simplified: use projected gravity alignment
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

        # Reward for gravity pointing down in base frame
        orientation_error = torch.sum(projected_gravity[:, :2] ** 2, dim=-1)  # Should be [0,0,-1]
        rewards["orientation"] = torch.exp(-5.0 * orientation_error)

        # =====================================================================
        # COM STABILITY REWARD
        # =====================================================================
        # XY velocity should be low for standing
        xy_velocity = torch.norm(base_lin_vel[:, :2], dim=-1)
        rewards["com_stability"] = torch.exp(-2.0 * xy_velocity ** 2)

        # =====================================================================
        # REGULARIZATION REWARDS
        # =====================================================================
        # Joint acceleration penalty
        rewards["joint_acceleration"] = torch.sum(joint_acc ** 2, dim=-1)

        # Action rate penalty
        if hasattr(self, '_prev_actions'):
            action_diff = self.actions - self._prev_actions
            rewards["action_rate"] = torch.sum(action_diff ** 2, dim=-1)
        else:
            rewards["action_rate"] = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = self.actions.clone()

        # =====================================================================
        # COMPUTE TOTAL REWARD
        # =====================================================================
        reward_scales = self.cfg.reward_scales
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for key, value in rewards.items():
            if key in reward_scales:
                scaled_reward = reward_scales[key] * value
                total_reward += scaled_reward
                self.episode_sums[key] += scaled_reward

        return total_reward

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
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

        # Check if tilted too much (gravity not pointing down)
        tilt_x = torch.abs(projected_gravity[:, 0]) > self.cfg.termination["max_roll"]
        tilt_y = torch.abs(projected_gravity[:, 1]) > self.cfg.termination["max_pitch"]

        # Combine terminations
        terminated = too_low | too_high | tilt_x | tilt_y

        # Time-out
        time_out = self.episode_length_buf >= self.max_episode_length

        # Add termination penalty
        self.episode_sums["termination"] += terminated.float() * self.cfg.reward_scales.get("termination", -100.0)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        # Get default state
        robot = self.robot

        # Random initial position
        default_pos = torch.tensor([0.0, 0.0, 0.8], device=self.device)
        pos_noise = self.cfg.randomization["init_pos_noise"]
        init_pos = default_pos + torch.randn(len(env_ids), 3, device=self.device) * pos_noise
        init_pos[:, 2] = 0.8  # Keep height fixed

        # Random initial orientation (small yaw only)
        init_rot = torch.zeros(len(env_ids), 4, device=self.device)
        init_rot[:, 3] = 1.0  # w=1 for identity quaternion

        # Apply initial state
        robot.write_root_pose_to_sim(
            torch.cat([init_pos, init_rot], dim=-1),
            env_ids
        )

        # Zero velocity
        robot.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device),
            env_ids
        )

        # Reset joint positions to default (with small noise)
        joint_noise = self.cfg.randomization["init_joint_noise"]
        default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
        default_joint_pos += torch.randn_like(default_joint_pos) * joint_noise
        robot.write_joint_state_to_sim(
            default_joint_pos,
            torch.zeros_like(default_joint_pos),
            None,
            env_ids
        )

        # Reset commands
        self.height_command[env_ids] = 0.75
        self.velocity_commands[env_ids] = 0.0
        self.torso_commands[env_ids] = 0.0
        self.arm_commands[env_ids] = 0.0

        # Reset tracking buffers
        self.previous_actions[env_ids] = 0.0
        if hasattr(self, '_prev_actions'):
            self._prev_actions[env_ids] = 0.0
        if hasattr(self, '_prev_joint_vel'):
            self._prev_joint_vel[env_ids] = 0.0

        # Log episode stats
        for key in self.episode_sums:
            if key != "termination":
                extras_key = f"Episode_Reward/{key}"
                if extras_key not in self.extras:
                    self.extras[extras_key] = torch.zeros(self.num_envs, device=self.device)
                self.extras[extras_key][env_ids] = self.episode_sums[key][env_ids] / (
                            self.episode_length_buf[env_ids] + 1)
            self.episode_sums[key][env_ids] = 0.0

    # =========================================================================
    # CURRICULUM METHODS
    # =========================================================================

    def set_stage(self, stage: int):
        """Set curriculum stage."""
        if stage != self.current_stage:
            print(f"[ULC_G1_Env] Advancing to Stage {stage}")
            self.current_stage = stage
            # Could update action/observation dimensions here
            # For now, we use fixed dimensions

    def sample_commands(self):
        """Sample new commands based on current stage."""
        if self.current_stage >= 2:
            # Sample velocity commands
            cmd_cfg = self.cfg.commands["velocity_range"]
            self.velocity_commands[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["vx"])
            self.velocity_commands[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["vy"])
            self.velocity_commands[:, 2] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["yaw_rate"])

        if self.current_stage >= 3:
            # Sample torso commands
            cmd_cfg = self.cfg.commands["torso_range"]
            self.torso_commands[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["roll"])
            self.torso_commands[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["pitch"])
            self.torso_commands[:, 2] = torch.empty(self.num_envs, device=self.device).uniform_(*cmd_cfg["yaw"])

        if self.current_stage >= 4:
            # Sample arm targets
            arm_range = self.cfg.commands["arm_range"]
            self.arm_commands = torch.empty(self.num_envs, 14, device=self.device).uniform_(*arm_range)