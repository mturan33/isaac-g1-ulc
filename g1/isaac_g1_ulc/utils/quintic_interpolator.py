"""
Quintic Polynomial Interpolator
===============================

Smooth command transitions için 5. derece polinom interpolasyonu.
ULC'nin "jerky motion" problemini çözer.

Matematiksel temel:
p(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵

Boundary conditions:
- p(0) = start, p(T) = end
- p'(0) = 0, p'(T) = 0  (sıfır hız)
- p''(0) = 0, p''(T) = 0 (sıfır ivme)
"""

from __future__ import annotations

import torch
from typing import Optional


class QuinticInterpolator:
    """
    Quintic polynomial interpolator for smooth transitions.

    Her command değişiminde ani geçişler yerine smooth
    polinom ile geçiş yapar.
    """

    def __init__(
            self,
            num_envs: int,
            cmd_dim: int,
            interpolation_time: float = 0.5,
            dt: float = 0.02,
            device: str = "cuda"
    ):
        """
        Args:
            num_envs: Number of environments
            cmd_dim: Command dimension
            interpolation_time: Duration of interpolation (seconds)
            dt: Simulation time step
            device: Torch device
        """
        self.num_envs = num_envs
        self.cmd_dim = cmd_dim
        self.T = interpolation_time
        self.dt = dt
        self.device = device

        # State buffers
        self.cmd_start = torch.zeros(num_envs, cmd_dim, device=device)
        self.cmd_end = torch.zeros(num_envs, cmd_dim, device=device)
        self.t_elapsed = torch.zeros(num_envs, device=device)
        self.is_interpolating = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def set_target(self, new_cmd: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        Set new target command and start interpolation.

        Args:
            new_cmd: New target command [num_envs, cmd_dim] or [len(env_ids), cmd_dim]
            env_ids: Optional environment indices to update
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Start from current interpolated value
        current = self.get_current_cmd()
        self.cmd_start[env_ids] = current[env_ids]
        self.cmd_end[env_ids] = new_cmd
        self.t_elapsed[env_ids] = 0.0
        self.is_interpolating[env_ids] = True

    def get_current_cmd(self) -> torch.Tensor:
        """
        Get current interpolated command.

        Returns:
            cmd: [num_envs, cmd_dim] interpolated command
        """
        # Normalized time [0, 1]
        s = torch.clamp(self.t_elapsed / self.T, 0.0, 1.0)

        # Quintic blend factor: 10s³ - 15s⁴ + 6s⁵
        blend = 10 * s.pow(3) - 15 * s.pow(4) + 6 * s.pow(5)
        blend = blend.unsqueeze(-1)  # [num_envs, 1]

        # Interpolate
        cmd = self.cmd_start + blend * (self.cmd_end - self.cmd_start)

        return cmd

    def get_current_velocity(self) -> torch.Tensor:
        """
        Get current command velocity (derivative of interpolation).

        Returns:
            vel: [num_envs, cmd_dim] command velocity
        """
        s = torch.clamp(self.t_elapsed / self.T, 0.0, 1.0)

        # Derivative of quintic: 30s² - 60s³ + 30s⁴
        ds_dt = 1.0 / self.T
        blend_dot = (30 * s.pow(2) - 60 * s.pow(3) + 30 * s.pow(4)) * ds_dt
        blend_dot = blend_dot.unsqueeze(-1)

        vel = blend_dot * (self.cmd_end - self.cmd_start)

        return vel

    def step(self, dt: Optional[float] = None):
        """
        Step interpolation forward in time.

        Args:
            dt: Time step (uses default if None)
        """
        if dt is None:
            dt = self.dt

        self.t_elapsed = self.t_elapsed + dt

        # Mark finished interpolations
        finished = self.t_elapsed >= self.T
        self.is_interpolating[finished] = False

    def reset(self, env_ids: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None):
        """
        Reset interpolator state.

        Args:
            env_ids: Environment indices to reset
            value: Initial value to set (default: zeros)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if value is None:
            value = torch.zeros(len(env_ids), self.cmd_dim, device=self.device)

        self.cmd_start[env_ids] = value
        self.cmd_end[env_ids] = value
        self.t_elapsed[env_ids] = self.T  # Already at target
        self.is_interpolating[env_ids] = False


class CommandInterpolatorBank:
    """
    Bank of interpolators for different command types.

    Different commands may need different interpolation times:
    - Velocity: Fast (0.3s)
    - Height: Medium (0.5s)
    - Arms: Slow (0.8s) for safety
    """

    def __init__(self, num_envs: int, device: str = "cuda"):
        self.num_envs = num_envs
        self.device = device

        # Create interpolators for each command type
        self.interpolators = {
            "velocity": QuinticInterpolator(num_envs, 3, interpolation_time=0.3, device=device),
            "height": QuinticInterpolator(num_envs, 1, interpolation_time=0.5, device=device),
            "torso": QuinticInterpolator(num_envs, 3, interpolation_time=0.4, device=device),
            "arms": QuinticInterpolator(num_envs, 14, interpolation_time=0.8, device=device),
        }

    def set_target(self, cmd_type: str, new_cmd: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """Set target for specific command type."""
        self.interpolators[cmd_type].set_target(new_cmd, env_ids)

    def get_current_cmd(self, cmd_type: str) -> torch.Tensor:
        """Get current interpolated command for specific type."""
        return self.interpolators[cmd_type].get_current_cmd()

    def step(self, dt: float = 0.02):
        """Step all interpolators."""
        for interp in self.interpolators.values():
            interp.step(dt)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset all interpolators for given environments."""
        for interp in self.interpolators.values():
            interp.reset(env_ids)