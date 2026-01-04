"""
Tracking Rewards
================

Velocity, arm position, torso orientation tracking reward'ları.
Stage 2+ için kullanılır.
"""

from __future__ import annotations

import torch
from typing import Optional


def compute_velocity_tracking_reward(
        current_vel: torch.Tensor,
        target_vel: torch.Tensor,
        scale: float = 5.0,
        weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reward for tracking target velocity.

    Args:
        current_vel: [num_envs, 3] Current velocity (vx, vy, yaw_rate)
        target_vel: [num_envs, 3] Target velocity
        scale: Exponential decay scale
        weights: [3] Optional weights for each component

    Returns:
        reward: [num_envs] velocity tracking reward
    """
    error = current_vel - target_vel

    if weights is not None:
        error = error * weights

    error_norm = torch.sum(error ** 2, dim=-1)
    return torch.exp(-scale * error_norm)


def compute_arm_tracking_reward(
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        scale: float = 3.0
) -> torch.Tensor:
    """
    Reward for tracking arm joint positions.

    Args:
        current_pos: [num_envs, 14] Current arm joint positions (7 left + 7 right)
        target_pos: [num_envs, 14] Target positions
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] arm tracking reward
    """
    error = current_pos - target_pos
    error_norm = torch.sum(error ** 2, dim=-1)
    return torch.exp(-scale * error_norm)


def compute_torso_tracking_reward(
        current_ori: torch.Tensor,
        target_ori: torch.Tensor,
        scale: float = 4.0
) -> torch.Tensor:
    """
    Reward for tracking torso orientation.

    Args:
        current_ori: [num_envs, 3] Current torso orientation (roll, pitch, yaw)
        target_ori: [num_envs, 3] Target orientation
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] torso tracking reward
    """
    error = current_ori - target_ori
    error_norm = torch.sum(error ** 2, dim=-1)
    return torch.exp(-scale * error_norm)


def compute_height_tracking_reward(
        current_height: torch.Tensor,
        target_height: torch.Tensor,
        scale: float = 10.0
) -> torch.Tensor:
    """
    Reward for tracking target height.

    Args:
        current_height: [num_envs] Current base height
        target_height: [num_envs] or float Target height
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] height tracking reward
    """
    if isinstance(target_height, float):
        target_height = torch.full_like(current_height, target_height)

    error = torch.abs(current_height - target_height)
    return torch.exp(-scale * error ** 2)


def compute_orientation_reward(
        base_quat: torch.Tensor,
        scale: float = 5.0
) -> torch.Tensor:
    """
    Reward for staying upright.

    Args:
        base_quat: [num_envs, 4] Base quaternion (w, x, y, z) or (x, y, z, w)
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] orientation reward
    """
    from isaaclab.utils.math import quat_rotate_inverse

    num_envs = base_quat.shape[0]
    device = base_quat.device

    # Project gravity to base frame
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device).expand(num_envs, -1)
    proj_gravity = quat_rotate_inverse(base_quat, gravity)

    # Error: deviation from pointing down
    # proj_gravity should be [0, 0, -1] when upright
    orientation_error = torch.sum(proj_gravity[:, :2] ** 2, dim=-1)

    return torch.exp(-scale * orientation_error)


def compute_gait_frequency_reward(
        foot_contacts: torch.Tensor,
        target_frequency: float = 2.0,
        dt: float = 0.02
) -> torch.Tensor:
    """
    Reward for maintaining target gait frequency.

    Args:
        foot_contacts: [num_envs, 2] Binary contact state (left, right)
        target_frequency: Target step frequency (Hz)
        dt: Time step

    Returns:
        reward: [num_envs] gait frequency reward

    Note: This needs history tracking, placeholder implementation.
    """
    # Placeholder - actual implementation needs step history
    return torch.ones(foot_contacts.shape[0], device=foot_contacts.device)


# =============================================================================
# REGULARIZATION REWARDS (Penalties)
# =============================================================================

def compute_joint_acceleration_penalty(
        current_vel: torch.Tensor,
        prev_vel: torch.Tensor,
        dt: float = 0.02,
        scale: float = 0.0005
) -> torch.Tensor:
    """
    Penalty for high joint accelerations.

    Args:
        current_vel: [num_envs, num_joints] Current joint velocities
        prev_vel: [num_envs, num_joints] Previous joint velocities
        dt: Time step
        scale: Penalty scale

    Returns:
        penalty: [num_envs] acceleration penalty (negative)
    """
    acceleration = (current_vel - prev_vel) / dt
    penalty = torch.sum(acceleration ** 2, dim=-1)
    return -scale * penalty


def compute_action_rate_penalty(
        current_action: torch.Tensor,
        prev_action: torch.Tensor,
        scale: float = 0.01
) -> torch.Tensor:
    """
    Penalty for large action changes.

    Args:
        current_action: [num_envs, num_actions] Current actions
        prev_action: [num_envs, num_actions] Previous actions
        scale: Penalty scale

    Returns:
        penalty: [num_envs] action rate penalty (negative)
    """
    diff = current_action - prev_action
    penalty = torch.sum(diff ** 2, dim=-1)
    return -scale * penalty


def compute_energy_penalty(
        joint_vel: torch.Tensor,
        joint_torque: torch.Tensor,
        scale: float = 0.001
) -> torch.Tensor:
    """
    Penalty for energy consumption.

    Args:
        joint_vel: [num_envs, num_joints] Joint velocities
        joint_torque: [num_envs, num_joints] Joint torques
        scale: Penalty scale

    Returns:
        penalty: [num_envs] energy penalty (negative)
    """
    power = torch.abs(joint_vel * joint_torque)
    total_power = torch.sum(power, dim=-1)
    return -scale * total_power


def compute_arm_smoothness_penalty(
        arm_vel: torch.Tensor,
        scale: float = 0.02
) -> torch.Tensor:
    """
    Penalty for jerky arm movements.

    Args:
        arm_vel: [num_envs, 14] Arm joint velocities
        scale: Penalty scale

    Returns:
        penalty: [num_envs] smoothness penalty (negative)
    """
    penalty = torch.sum(arm_vel ** 2, dim=-1)
    return -scale * penalty


class TrackingRewardComputer:
    """
    Class-based tracking reward computation.

    Maintains history for rewards that need previous states.
    """

    def __init__(self, num_envs: int, num_actions: int, num_joints: int, device: str = "cuda"):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.num_joints = num_joints
        self.device = device

        # History buffers
        self.prev_actions = torch.zeros(num_envs, num_actions, device=device)
        self.prev_joint_vel = torch.zeros(num_envs, num_joints, device=device)

    def update_history(self, actions: torch.Tensor, joint_vel: torch.Tensor):
        """Update history buffers."""
        self.prev_actions = actions.clone()
        self.prev_joint_vel = joint_vel.clone()

    def compute_penalties(
            self,
            actions: torch.Tensor,
            joint_vel: torch.Tensor,
            dt: float = 0.02
    ) -> dict:
        """
        Compute all penalty terms.

        Returns:
            penalties: Dictionary of penalty components
        """
        penalties = {}

        # Action rate
        penalties["action_rate"] = compute_action_rate_penalty(
            actions, self.prev_actions
        )

        # Joint acceleration
        penalties["joint_acceleration"] = compute_joint_acceleration_penalty(
            joint_vel, self.prev_joint_vel, dt
        )

        # Update history
        self.update_history(actions, joint_vel)

        return penalties

    def reset(self, env_ids: torch.Tensor):
        """Reset history for given environments."""
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0