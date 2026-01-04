"""
Load Randomizer
===============

Manipulation sırasında robot'un taşıdığı yükü simüle eder.
Training'de random yükler eklenerek policy yük değişimlerine
dayanıklı hale getirilir.

Bu, ULC'nin "pick and place" görevlerinde düşmemesini sağlar!
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple


class LoadRandomizer:
    """
    Randomizes external loads on robot's hands/grippers.

    Simulates:
    - Picking up objects of varying weights
    - Asymmetric loads (one hand full, other empty)
    - Dynamic load changes during episodes
    """

    def __init__(
            self,
            num_envs: int,
            max_load_kg: float = 2.0,
            min_load_kg: float = 0.0,
            gravity: float = 9.81,
            device: str = "cuda"
    ):
        """
        Args:
            num_envs: Number of environments
            max_load_kg: Maximum load per hand (kg)
            min_load_kg: Minimum load per hand (kg)
            gravity: Gravitational acceleration (m/s²)
            device: Torch device
        """
        self.num_envs = num_envs
        self.max_load = max_load_kg
        self.min_load = min_load_kg
        self.gravity = gravity
        self.device = device

        # Current loads [num_envs, 2] for left and right hand
        self.loads = torch.zeros(num_envs, 2, device=device)

        # Hand body indices (will be set by environment)
        self.left_hand_idx = None
        self.right_hand_idx = None

        # Load change probability per step
        self.change_prob = 0.001  # ~1 change per 1000 steps

        # Whether loads are active
        self.active = False

    def set_hand_indices(self, left_idx: int, right_idx: int):
        """Set body indices for hands."""
        self.left_hand_idx = left_idx
        self.right_hand_idx = right_idx
        self.active = True

    def randomize(self, env_ids: Optional[torch.Tensor] = None):
        """
        Randomize loads for given environments.

        Args:
            env_ids: Environment indices to randomize (default: all)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Random loads for both hands
        new_loads = torch.empty(len(env_ids), 2, device=self.device).uniform_(
            self.min_load, self.max_load
        )

        self.loads[env_ids] = new_loads

    def maybe_change_loads(self) -> bool:
        """
        Randomly change loads during episode.

        Returns:
            changed: True if any loads were changed
        """
        # Random mask for which envs get load change
        change_mask = torch.rand(self.num_envs, device=self.device) < self.change_prob

        if change_mask.any():
            change_ids = torch.where(change_mask)[0]
            self.randomize(change_ids)
            return True

        return False

    def get_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get gravitational forces for current loads.

        Returns:
            left_force: [num_envs, 3] force on left hand
            right_force: [num_envs, 3] force on right hand
        """
        # Force = mass * gravity, pointing down (-Z)
        left_force = torch.zeros(self.num_envs, 3, device=self.device)
        right_force = torch.zeros(self.num_envs, 3, device=self.device)

        left_force[:, 2] = -self.loads[:, 0] * self.gravity
        right_force[:, 2] = -self.loads[:, 1] * self.gravity

        return left_force, right_force

    def apply_forces(self, robot):
        """
        Apply external forces to robot hands.

        Args:
            robot: Isaac Lab Articulation object
        """
        if not self.active:
            return

        if self.left_hand_idx is None or self.right_hand_idx is None:
            return

        left_force, right_force = self.get_forces()

        # Stack forces for both hands
        forces = torch.stack([left_force, right_force], dim=1)  # [num_envs, 2, 3]
        torques = torch.zeros_like(forces)

        # Apply external forces
        try:
            robot.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                body_ids=torch.tensor([self.left_hand_idx, self.right_hand_idx], device=self.device)
            )
        except Exception as e:
            # Some Isaac Lab versions may have different API
            pass

    def get_observation(self) -> torch.Tensor:
        """
        Get load as observation (for policy to be aware of).

        Returns:
            obs: [num_envs, 2] load masses (kg)
        """
        return self.loads

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """
        Reset loads for given environments.

        Args:
            env_ids: Environment indices to reset
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Randomize on reset
        self.randomize(env_ids)


class ProgressiveLoadRandomizer(LoadRandomizer):
    """
    Load randomizer that increases difficulty over training.

    Starts with small loads and gradually increases as
    policy improves (curriculum learning for loads).
    """

    def __init__(
            self,
            num_envs: int,
            final_max_load_kg: float = 2.0,
            initial_max_load_kg: float = 0.5,
            gravity: float = 9.81,
            device: str = "cuda"
    ):
        super().__init__(
            num_envs=num_envs,
            max_load_kg=initial_max_load_kg,
            min_load_kg=0.0,
            gravity=gravity,
            device=device
        )

        self.final_max_load = final_max_load_kg
        self.initial_max_load = initial_max_load_kg
        self.progress = 0.0  # 0 to 1

    def set_progress(self, progress: float):
        """
        Set curriculum progress.

        Args:
            progress: Training progress from 0 to 1
        """
        self.progress = min(max(progress, 0.0), 1.0)
        self.max_load = (
                self.initial_max_load +
                self.progress * (self.final_max_load - self.initial_max_load)
        )


class ExternalPushRandomizer:
    """
    Applies random external pushes to test balance.

    Simulates:
    - Accidental bumps
    - Wind gusts
    - Human interaction
    """

    def __init__(
            self,
            num_envs: int,
            min_push_force: float = 50.0,
            max_push_force: float = 150.0,
            push_duration: float = 0.1,
            min_interval: float = 5.0,
            max_interval: float = 15.0,
            device: str = "cuda"
    ):
        """
        Args:
            num_envs: Number of environments
            min_push_force: Minimum push force (N)
            max_push_force: Maximum push force (N)
            push_duration: Duration of push (seconds)
            min_interval: Minimum time between pushes (seconds)
            max_interval: Maximum time between pushes (seconds)
            device: Torch device
        """
        self.num_envs = num_envs
        self.min_force = min_push_force
        self.max_force = max_push_force
        self.push_duration = push_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.device = device

        # Time until next push for each env
        self.time_to_push = self._sample_intervals()

        # Current push state
        self.is_pushing = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.push_time_remaining = torch.zeros(num_envs, device=device)
        self.push_force = torch.zeros(num_envs, 3, device=device)

        # Pelvis/torso body index (will be set)
        self.push_body_idx = 0
        self.active = False

    def _sample_intervals(self) -> torch.Tensor:
        """Sample random intervals until next push."""
        return torch.empty(self.num_envs, device=self.device).uniform_(
            self.min_interval, self.max_interval
        )

    def set_push_body(self, body_idx: int):
        """Set body index to apply pushes to."""
        self.push_body_idx = body_idx
        self.active = True

    def step(self, dt: float) -> torch.Tensor:
        """
        Step push simulation.

        Args:
            dt: Time step (seconds)

        Returns:
            force: [num_envs, 3] push force to apply
        """
        if not self.active:
            return torch.zeros(self.num_envs, 3, device=self.device)

        # Update timers
        self.time_to_push -= dt

        # Start new pushes
        start_push = (self.time_to_push <= 0) & (~self.is_pushing)
        if start_push.any():
            # Random force magnitude
            magnitudes = torch.empty(start_push.sum(), device=self.device).uniform_(
                self.min_force, self.max_force
            )

            # Random horizontal direction
            angles = torch.empty(start_push.sum(), device=self.device).uniform_(0, 2 * 3.14159)

            self.push_force[start_push, 0] = magnitudes * torch.cos(angles)
            self.push_force[start_push, 1] = magnitudes * torch.sin(angles)
            self.push_force[start_push, 2] = 0  # Horizontal push only

            self.is_pushing[start_push] = True
            self.push_time_remaining[start_push] = self.push_duration
            self.time_to_push[start_push] = self._sample_intervals()[start_push]

        # Update ongoing pushes
        self.push_time_remaining -= dt
        end_push = self.is_pushing & (self.push_time_remaining <= 0)
        self.is_pushing[end_push] = False
        self.push_force[end_push] = 0

        # Return current forces
        force = self.push_force.clone()
        force[~self.is_pushing] = 0

        return force

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset push state for given environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.time_to_push[env_ids] = self._sample_intervals()[env_ids]
        self.is_pushing[env_ids] = False
        self.push_time_remaining[env_ids] = 0
        self.push_force[env_ids] = 0