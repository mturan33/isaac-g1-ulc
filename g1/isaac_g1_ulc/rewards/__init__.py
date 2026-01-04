"""ULC Reward Functions."""

from .com_rewards import (
    compute_com_position_reward,
    compute_com_velocity_reward,
    compute_com_height_reward,
    compute_com_margin_reward,
    compute_stability_reward,
    CoMRewardComputer,
)

from .tracking_rewards import (
    compute_velocity_tracking_reward,
    compute_arm_tracking_reward,
    compute_torso_tracking_reward,
    compute_height_tracking_reward,
    compute_orientation_reward,
    compute_joint_acceleration_penalty,
    compute_action_rate_penalty,
    compute_energy_penalty,
    compute_arm_smoothness_penalty,
    TrackingRewardComputer,
)

__all__ = [
    # CoM rewards
    "compute_com_position_reward",
    "compute_com_velocity_reward",
    "compute_com_height_reward",
    "compute_com_margin_reward",
    "compute_stability_reward",
    "CoMRewardComputer",
    # Tracking rewards
    "compute_velocity_tracking_reward",
    "compute_arm_tracking_reward",
    "compute_torso_tracking_reward",
    "compute_height_tracking_reward",
    "compute_orientation_reward",
    "compute_joint_acceleration_penalty",
    "compute_action_rate_penalty",
    "compute_energy_penalty",
    "compute_arm_smoothness_penalty",
    "TrackingRewardComputer",
]