"""
Center of Mass (CoM) Rewards
============================

ULC'nin en kritik reward bileşenleri - denge ve stabilite.
Bu reward'lar OLMADAN robot DÜŞER!
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def compute_com_position_reward(
        com_xy: torch.Tensor,
        support_center: torch.Tensor,
        scale: float = 5.0
) -> torch.Tensor:
    """
    Reward for keeping CoM centered over support polygon.

    Args:
        com_xy: [num_envs, 2] CoM position in XY plane
        support_center: [num_envs, 2] Support polygon center
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] position reward
    """
    error = torch.norm(com_xy - support_center, dim=-1)
    return torch.exp(-scale * error ** 2)


def compute_com_velocity_reward(
        com_vel_xy: torch.Tensor,
        target_vel: torch.Tensor = None,
        scale: float = 2.0
) -> torch.Tensor:
    """
    Reward for CoM velocity tracking or stability.

    Args:
        com_vel_xy: [num_envs, 2] CoM velocity in XY plane
        target_vel: [num_envs, 2] Target velocity (None for standing)
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] velocity reward
    """
    if target_vel is None:
        # Standing: penalize any XY velocity
        vel_error = torch.norm(com_vel_xy, dim=-1)
    else:
        vel_error = torch.norm(com_vel_xy - target_vel, dim=-1)

    return torch.exp(-scale * vel_error ** 2)


def compute_com_height_reward(
        com_height: torch.Tensor,
        target_height: float = 0.75,
        scale: float = 10.0
) -> torch.Tensor:
    """
    Reward for maintaining target CoM height.

    Args:
        com_height: [num_envs] CoM height
        target_height: Target height (meters)
        scale: Exponential decay scale

    Returns:
        reward: [num_envs] height reward
    """
    error = torch.abs(com_height - target_height)
    return torch.exp(-scale * error ** 2)


def compute_com_margin_reward(
        com_xy: torch.Tensor,
        left_foot_xy: torch.Tensor,
        right_foot_xy: torch.Tensor,
        max_margin: float = 0.12
) -> torch.Tensor:
    """
    Reward for staying safely within support polygon.

    Args:
        com_xy: [num_envs, 2] CoM position
        left_foot_xy: [num_envs, 2] Left foot position
        right_foot_xy: [num_envs, 2] Right foot position
        max_margin: Maximum safe distance from center

    Returns:
        reward: [num_envs] margin reward (0-1)
    """
    # Support center
    center = (left_foot_xy + right_foot_xy) / 2.0

    # Distance from center
    distance = torch.norm(com_xy - center, dim=-1)

    # Margin (how much room before edge)
    margin = max_margin - distance

    # Normalize to [0, 1]
    normalized = torch.clamp(margin / max_margin, 0.0, 1.0)

    return normalized


def compute_stability_reward(
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        base_lin_vel: torch.Tensor,
        target_height: float = 0.75
) -> torch.Tensor:
    """
    Combined stability reward using base state.

    Simpler alternative to full CoM computation.

    Args:
        base_pos: [num_envs, 3] Base position
        base_quat: [num_envs, 4] Base orientation (quaternion)
        base_lin_vel: [num_envs, 3] Base linear velocity
        target_height: Target standing height

    Returns:
        reward: [num_envs] combined stability reward
    """
    from isaaclab.utils.math import quat_rotate_inverse

    num_envs = base_pos.shape[0]
    device = base_pos.device

    # Height reward
    height = base_pos[:, 2]
    height_error = torch.abs(height - target_height)
    r_height = torch.exp(-10.0 * height_error ** 2)

    # Orientation reward (upright)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device).expand(num_envs, -1)
    proj_gravity = quat_rotate_inverse(base_quat, gravity)
    orientation_error = torch.sum(proj_gravity[:, :2] ** 2, dim=-1)
    r_orientation = torch.exp(-5.0 * orientation_error)

    # Velocity reward (low for standing)
    xy_vel = torch.norm(base_lin_vel[:, :2], dim=-1)
    r_velocity = torch.exp(-2.0 * xy_vel ** 2)

    # Combined
    reward = 0.4 * r_height + 0.4 * r_orientation + 0.2 * r_velocity

    return reward


class CoMRewardComputer:
    """
    Class-based CoM reward computation with caching.

    More efficient for repeated computation within episode.
    """

    def __init__(self, env: DirectRLEnv, target_height: float = 0.75):
        self.env = env
        self.target_height = target_height
        self.device = env.device
        self.num_envs = env.num_envs

        # Cache for body indices
        self._left_foot_idx = None
        self._right_foot_idx = None
        self._initialized = False

    def _initialize(self):
        """Initialize body indices."""
        if self._initialized:
            return

        robot = self.env.scene["robot"]
        body_names = robot.data.body_names

        for i, name in enumerate(body_names):
            name_lower = name.lower()
            if "left" in name_lower and ("foot" in name_lower or "ankle" in name_lower):
                self._left_foot_idx = i
            elif "right" in name_lower and ("foot" in name_lower or "ankle" in name_lower):
                self._right_foot_idx = i

        # Fallback
        if self._left_foot_idx is None:
            self._left_foot_idx = 0
        if self._right_foot_idx is None:
            self._right_foot_idx = 1

        self._initialized = True

    def compute_all(self) -> dict:
        """
        Compute all CoM-related rewards.

        Returns:
            rewards: Dictionary of reward components
        """
        self._initialize()

        robot = self.env.scene["robot"]

        # Get states
        base_pos = robot.data.root_pos_w
        base_quat = robot.data.root_quat_w
        base_lin_vel = robot.data.root_lin_vel_w
        body_pos = robot.data.body_pos_w

        # Foot positions
        left_foot_xy = body_pos[:, self._left_foot_idx, :2]
        right_foot_xy = body_pos[:, self._right_foot_idx, :2]

        # Use base as CoM proxy
        com_xy = base_pos[:, :2]
        com_height = base_pos[:, 2]
        com_vel_xy = base_lin_vel[:, :2]

        rewards = {}

        # Position reward
        support_center = (left_foot_xy + right_foot_xy) / 2.0
        rewards["com_position"] = compute_com_position_reward(com_xy, support_center)

        # Velocity reward
        rewards["com_velocity"] = compute_com_velocity_reward(com_vel_xy)

        # Height reward
        rewards["com_height"] = compute_com_height_reward(com_height, self.target_height)

        # Margin reward
        rewards["com_margin"] = compute_com_margin_reward(
            com_xy, left_foot_xy, right_foot_xy
        )

        # Combined stability
        rewards["com_stability"] = (
                0.25 * rewards["com_position"] +
                0.25 * rewards["com_velocity"] +
                0.25 * rewards["com_height"] +
                0.25 * rewards["com_margin"]
        )

        return rewards