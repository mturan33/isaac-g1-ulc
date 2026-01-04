"""
Sequential Curriculum for ULC Training
=======================================

ULC'nin en önemli tekniklerinden biri: Sıralı beceri edinimi.
Standing → Locomotion → Torso → Arms sırası ile öğretim.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Dict, List
from collections import deque


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    name: str
    duration_steps: int
    reward_threshold: float
    active_commands: List[str]
    num_actions: int
    num_observations: int
    reward_weights: Dict[str, float]


class SequentialCurriculum:
    """
    Sequential Skill Acquisition Manager.

    Stages:
    1. Standing: Height tracking, balance
    2. Locomotion: Velocity tracking (vx, vy, yaw)
    3. Torso: Torso orientation control
    4. Arms: Dual-arm position tracking
    5. Full: All commands + domain randomization
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.current_stage = 1
        self.steps_in_stage = 0
        self.total_steps = 0

        # Reward tracking for stage advancement
        self.reward_history = deque(maxlen=100)

        # Stage configurations
        self.stages = {
            1: StageConfig(
                name="standing",
                duration_steps=500_000,
                reward_threshold=0.7,
                active_commands=["height"],
                num_actions=12,  # Legs only
                num_observations=46,
                reward_weights={
                    "height_tracking": 5.0,
                    "orientation": 3.0,
                    "com_stability": 4.0,
                    "joint_acceleration": -0.0005,
                    "action_rate": -0.01,
                }
            ),
            2: StageConfig(
                name="locomotion",
                duration_steps=1_000_000,
                reward_threshold=0.65,
                active_commands=["height", "velocity"],
                num_actions=12,
                num_observations=49,  # +3 velocity commands
                reward_weights={
                    "height_tracking": 3.0,
                    "velocity_tracking": 5.0,
                    "orientation": 2.0,
                    "com_stability": 3.0,
                    "gait_frequency": 1.0,
                    "joint_acceleration": -0.0005,
                    "action_rate": -0.01,
                }
            ),
            3: StageConfig(
                name="torso",
                duration_steps=1_000_000,
                reward_threshold=0.6,
                active_commands=["height", "velocity", "torso"],
                num_actions=15,  # Legs + Waist
                num_observations=58,  # +3 waist pos, +3 waist vel, +3 torso cmd
                reward_weights={
                    "height_tracking": 2.0,
                    "velocity_tracking": 4.0,
                    "torso_tracking": 4.0,
                    "orientation": 1.5,
                    "com_stability": 3.0,
                    "joint_acceleration": -0.0005,
                    "action_rate": -0.01,
                }
            ),
            4: StageConfig(
                name="arms",
                duration_steps=2_000_000,
                reward_threshold=0.55,
                active_commands=["height", "velocity", "torso", "arms"],
                num_actions=29,  # All joints
                num_observations=93,
                reward_weights={
                    "height_tracking": 1.5,
                    "velocity_tracking": 3.0,
                    "torso_tracking": 3.0,
                    "arm_tracking": 5.0,
                    "orientation": 1.0,
                    "com_stability": 4.0,
                    "joint_acceleration": -0.0005,
                    "action_rate": -0.01,
                    "arm_smoothness": -0.02,
                }
            ),
            5: StageConfig(
                name="full",
                duration_steps=2_000_000,
                reward_threshold=0.5,  # No advancement needed
                active_commands=["height", "velocity", "torso", "arms"],
                num_actions=29,
                num_observations=93,
                reward_weights={
                    "height_tracking": 1.5,
                    "velocity_tracking": 3.0,
                    "torso_tracking": 3.0,
                    "arm_tracking": 5.0,
                    "orientation": 1.0,
                    "com_stability": 4.0,
                    "load_robustness": 2.0,
                    "external_disturbance": 1.0,
                    "joint_acceleration": -0.0005,
                    "action_rate": -0.01,
                    "arm_smoothness": -0.02,
                }
            ),
        }

    def get_current_stage_config(self) -> StageConfig:
        """Get current stage configuration."""
        return self.stages[self.current_stage]

    def update(self, mean_reward: float, num_steps: int = 1) -> bool:
        """
        Update curriculum state and check for stage advancement.

        Args:
            mean_reward: Mean episode reward
            num_steps: Number of steps taken

        Returns:
            advanced: True if advanced to next stage
        """
        self.steps_in_stage += num_steps
        self.total_steps += num_steps
        self.reward_history.append(mean_reward)

        # Check advancement conditions
        stage_cfg = self.stages[self.current_stage]

        # Condition 1: Minimum steps in stage
        min_steps_met = self.steps_in_stage >= stage_cfg.duration_steps * 0.5

        # Condition 2: Duration exceeded
        duration_exceeded = self.steps_in_stage >= stage_cfg.duration_steps

        # Condition 3: Reward threshold
        if len(self.reward_history) >= 50:
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            reward_met = avg_reward >= stage_cfg.reward_threshold
        else:
            reward_met = False

        # Advance if conditions met
        if (min_steps_met and reward_met) or duration_exceeded:
            if self.current_stage < 5:
                return self.advance_stage()

        return False

    def advance_stage(self) -> bool:
        """Advance to next stage."""
        if self.current_stage >= 5:
            return False

        old_stage = self.current_stage
        self.current_stage += 1
        self.steps_in_stage = 0
        self.reward_history.clear()

        new_cfg = self.stages[self.current_stage]
        print(f"\n{'=' * 60}")
        print(f"CURRICULUM ADVANCEMENT: Stage {old_stage} → Stage {self.current_stage}")
        print(f"New Stage: {new_cfg.name.upper()}")
        print(f"Active Commands: {new_cfg.active_commands}")
        print(f"Num Actions: {new_cfg.num_actions}")
        print(f"{'=' * 60}\n")

        return True

    def get_reward_weights(self) -> Dict[str, float]:
        """Get current stage reward weights."""
        return self.stages[self.current_stage].reward_weights

    def get_active_commands(self) -> List[str]:
        """Get commands active in current stage."""
        return self.stages[self.current_stage].active_commands

    def get_action_mask(self, total_actions: int = 29) -> torch.Tensor:
        """
        Get mask for active actions in current stage.

        Args:
            total_actions: Total number of possible actions

        Returns:
            mask: Boolean tensor [total_actions] indicating active actions
        """
        stage_cfg = self.stages[self.current_stage]
        num_active = stage_cfg.num_actions

        mask = torch.zeros(total_actions, dtype=torch.bool, device=self.device)
        mask[:num_active] = True

        return mask

    def get_status_string(self) -> str:
        """Get formatted status string."""
        stage_cfg = self.stages[self.current_stage]
        progress = self.steps_in_stage / stage_cfg.duration_steps * 100

        return (
            f"Stage {self.current_stage}/5 ({stage_cfg.name}) | "
            f"Progress: {progress:.1f}% | "
            f"Steps: {self.steps_in_stage:,}/{stage_cfg.duration_steps:,}"
        )

    def save_state(self) -> dict:
        """Save curriculum state for checkpointing."""
        return {
            "current_stage": self.current_stage,
            "steps_in_stage": self.steps_in_stage,
            "total_steps": self.total_steps,
            "reward_history": list(self.reward_history),
        }

    def load_state(self, state: dict):
        """Load curriculum state from checkpoint."""
        self.current_stage = state["current_stage"]
        self.steps_in_stage = state["steps_in_stage"]
        self.total_steps = state["total_steps"]
        self.reward_history = deque(state["reward_history"], maxlen=100)


class AdaptiveCurriculum(SequentialCurriculum):
    """
    Adaptive curriculum that adjusts based on learning progress.
    More sophisticated than basic sequential curriculum.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)

        # Additional tracking for adaptive behavior
        self.performance_window = deque(maxlen=500)
        self.difficulty_level = 1.0  # 0.0 to 1.0

    def update(self, mean_reward: float, num_steps: int = 1) -> bool:
        """Update with adaptive difficulty adjustment."""
        self.performance_window.append(mean_reward)

        # Adjust difficulty based on recent performance
        if len(self.performance_window) >= 100:
            recent_avg = sum(list(self.performance_window)[-100:]) / 100
            older_avg = sum(list(self.performance_window)[:100]) / 100 if len(
                self.performance_window) > 100 else recent_avg

            # If improving, increase difficulty
            if recent_avg > older_avg * 1.1:
                self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
            # If struggling, decrease difficulty
            elif recent_avg < older_avg * 0.9:
                self.difficulty_level = max(0.0, self.difficulty_level - 0.05)

        return super().update(mean_reward, num_steps)

    def get_command_range_scale(self) -> float:
        """Get scaling factor for command ranges based on difficulty."""
        base_scale = 0.3 + 0.7 * self.difficulty_level
        return base_scale

    def get_noise_scale(self) -> float:
        """Get scaling factor for domain randomization noise."""
        return self.difficulty_level