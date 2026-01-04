"""
Random Delay Buffer
===================

Sim-to-real transfer için kritik: Training sırasında rastgele
gecikme ekleyerek policy'yi real robot gecikmelerine dayanıklı yapar.

Real robot'ta tipik gecikmeler:
- Communication: 10-30ms
- Computation: 5-15ms
- Motor response: 10-20ms
Toplam: 25-65ms

Training'de 20-100ms arası random delay simüle edilir.
"""

from __future__ import annotations

import torch
from typing import Optional


class DelayBuffer:
    """
    Circular buffer that adds random delay to commands.

    Simulates real-world latency between command generation
    and motor execution.
    """

    def __init__(
            self,
            num_envs: int,
            cmd_dim: int,
            max_delay_steps: int = 5,
            min_delay_steps: int = 1,
            device: str = "cuda"
    ):
        """
        Args:
            num_envs: Number of environments
            cmd_dim: Command dimension
            max_delay_steps: Maximum delay in simulation steps
            min_delay_steps: Minimum delay in simulation steps
            device: Torch device

        Note: At 50Hz control (20ms per step), 5 steps = 100ms max delay
        """
        self.num_envs = num_envs
        self.cmd_dim = cmd_dim
        self.max_delay = max_delay_steps
        self.min_delay = min_delay_steps
        self.device = device

        # Circular buffer: [num_envs, buffer_size, cmd_dim]
        buffer_size = max_delay_steps + 1
        self.buffer = torch.zeros(num_envs, buffer_size, cmd_dim, device=device)

        # Write index for each environment
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Random delay for each environment (steps)
        self.delays = self._sample_delays()

        # Counter for when to re-sample delays
        self.resample_interval = 500  # steps
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _sample_delays(self) -> torch.Tensor:
        """Sample random delays for each environment."""
        return torch.randint(
            self.min_delay,
            self.max_delay + 1,
            (self.num_envs,),
            device=self.device
        )

    def push(self, new_cmd: torch.Tensor):
        """
        Push new command into buffer.

        Args:
            new_cmd: [num_envs, cmd_dim] new command
        """
        # Write to current position
        for i in range(self.num_envs):
            self.buffer[i, self.write_idx[i]] = new_cmd[i]

        # Advance write index
        self.write_idx = (self.write_idx + 1) % (self.max_delay + 1)

        # Update step counter and resample delays if needed
        self.step_counter += 1
        resample_mask = self.step_counter >= self.resample_interval
        if resample_mask.any():
            self.delays[resample_mask] = torch.randint(
                self.min_delay,
                self.max_delay + 1,
                (resample_mask.sum(),),
                device=self.device
            )
            self.step_counter[resample_mask] = 0

    def get_delayed_cmd(self) -> torch.Tensor:
        """
        Get delayed command from buffer.

        Returns:
            cmd: [num_envs, cmd_dim] delayed command
        """
        delayed_cmds = []
        for i in range(self.num_envs):
            # Read from delayed position
            read_idx = (self.write_idx[i] - 1 - self.delays[i]) % (self.max_delay + 1)
            delayed_cmds.append(self.buffer[i, read_idx])

        return torch.stack(delayed_cmds)

    def reset(self, env_ids: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None):
        """
        Reset buffer for given environments.

        Args:
            env_ids: Environment indices to reset
            value: Initial value to fill buffer (default: zeros)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if value is None:
            value = torch.zeros(len(env_ids), self.cmd_dim, device=self.device)

        # Fill buffer with initial value
        for i, env_id in enumerate(env_ids):
            self.buffer[env_id] = value[i].unsqueeze(0).expand(self.max_delay + 1, -1)

        self.write_idx[env_ids] = 0
        self.step_counter[env_ids] = 0

        # Resample delays
        self.delays[env_ids] = torch.randint(
            self.min_delay,
            self.max_delay + 1,
            (len(env_ids),),
            device=self.device
        )


class ActionDelayBuffer(DelayBuffer):
    """
    Specialized delay buffer for actions.

    Simulates motor/communication delay between policy output
    and actual motor execution.
    """

    def __init__(
            self,
            num_envs: int,
            num_actions: int,
            max_delay_ms: float = 100.0,
            min_delay_ms: float = 20.0,
            control_dt: float = 0.02,
            device: str = "cuda"
    ):
        """
        Args:
            num_envs: Number of environments
            num_actions: Number of actions
            max_delay_ms: Maximum delay in milliseconds
            min_delay_ms: Minimum delay in milliseconds
            control_dt: Control time step (seconds)
            device: Torch device
        """
        # Convert ms to steps
        max_steps = int(max_delay_ms / (control_dt * 1000))
        min_steps = int(min_delay_ms / (control_dt * 1000))

        super().__init__(
            num_envs=num_envs,
            cmd_dim=num_actions,
            max_delay_steps=max(max_steps, 1),
            min_delay_steps=max(min_steps, 1),
            device=device
        )

        self.control_dt = control_dt


class ObservationDelayBuffer(DelayBuffer):
    """
    Delay buffer for observations.

    Simulates sensor/perception delay in observation.
    Typically smaller than action delay.
    """

    def __init__(
            self,
            num_envs: int,
            num_obs: int,
            max_delay_ms: float = 50.0,
            min_delay_ms: float = 10.0,
            control_dt: float = 0.02,
            device: str = "cuda"
    ):
        max_steps = max(int(max_delay_ms / (control_dt * 1000)), 1)
        min_steps = max(int(min_delay_ms / (control_dt * 1000)), 1)

        super().__init__(
            num_envs=num_envs,
            cmd_dim=num_obs,
            max_delay_steps=max_steps,
            min_delay_steps=min_steps,
            device=device
        )