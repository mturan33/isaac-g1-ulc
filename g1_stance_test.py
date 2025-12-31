# Copyright (c) 2025, VLM-RL G1 Project
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Stance Test - Robot Ayakta Durma + DiffIK Upper Body Test

Bu script robotun ayakta durmasını sağlar ve upper body IK'yı test eder.
Locomotion policy yüklemeden, sadece default stance ile çalışır.

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\g1_stance_test.py --num_envs 1
"""

from __future__ import annotations

import argparse
import torch
import math
import gymnasium as gym

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Argparse
parser = argparse.ArgumentParser(description="G1 Stance Test with DiffIK Upper Body")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")

# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
    G1PickPlaceLocomanipulationDiffIKEnvCfg,
)


class G1StanceController:
    """Simple stance controller - keeps robot standing with arms reaching."""

    def __init__(self, env, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.step_count = 0

        # Action dimensions from environment
        # Upper body IK: 28D (left_ee: 7, right_ee: 7, hands: 14)
        # Lower body: 4D (locomotion command placeholder)
        self.upper_body_dim = 28
        self.lower_body_dim = 4
        self.total_action_dim = self.upper_body_dim + self.lower_body_dim

        # Default EE positions (relative to base, approximate)
        # These are the default "home" positions for the arms
        self.default_left_ee_pos = torch.tensor([0.0, 0.25, 0.3], device=device)  # Left side
        self.default_right_ee_pos = torch.tensor([0.0, -0.25, 0.3], device=device)  # Right side

        # Default quaternions (pointing forward)
        # w, x, y, z format
        self.default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        # Wave parameters
        self.wave_freq = 2.0  # Hz
        self.wave_amp = 0.15  # meters

        print(f"\n[Stance Controller] Initialized:")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Action dim: {self.total_action_dim} (upper: {self.upper_body_dim}, lower: {self.lower_body_dim})")
        print(f"  - Wave frequency: {self.wave_freq} Hz")
        print(f"  - Wave amplitude: {self.wave_amp} m")

    def get_action(self, obs: dict, sim_time: float) -> torch.Tensor:
        """Generate action for stance + arm waving."""
        actions = torch.zeros(self.num_envs, self.total_action_dim, device=self.device)

        # ============ UPPER BODY (DiffIK) ============
        # Action format: [left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), hands(14)]

        # Left arm - stays at default position
        actions[:, 0:3] = self.default_left_ee_pos
        actions[:, 3:7] = self.default_quat

        # Right arm - wave motion (sinusoidal in Z)
        wave_offset = self.wave_amp * math.sin(2 * math.pi * self.wave_freq * sim_time)
        right_pos = self.default_right_ee_pos.clone()
        right_pos[2] += wave_offset  # Add wave to Z

        actions[:, 7:10] = right_pos
        actions[:, 10:14] = self.default_quat

        # Hands - relaxed position (middle of range)
        actions[:, 14:28] = 0.0  # -1 to 1 range, 0 = middle

        # ============ LOWER BODY (Stance) ============
        # For DiffIK env, lower body action is velocity command: [vx, vy, wz, height_offset]
        # Zero velocity = stand still
        actions[:, 28:32] = 0.0

        self.step_count += 1
        return actions


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("  G1 STANCE TEST - DiffIK Upper Body + Standing")
    print("=" * 70)

    # Create environment
    try:
        env = gym.make(
            "Isaac-PickPlace-Locomanipulation-G1-DiffIK-v0",
            num_envs=args_cli.num_envs,
            device=args_cli.device,
        )
        print(f"\n[✓] Environment created: Isaac-PickPlace-Locomanipulation-G1-DiffIK-v0")
    except Exception as e:
        print(f"\n[✗] Failed to create DiffIK environment: {e}")
        print("\n[!] Trying alternative approach with direct scene setup...")
        raise e

    # Create stance controller
    controller = G1StanceController(env.unwrapped, device=args_cli.device)

    # Reset environment
    obs, info = env.reset()
    print(f"\n[✓] Environment reset")
    print(f"    Observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

    # Run simulation
    sim_time = 0.0
    dt = 0.02  # Environment step time

    print("\n" + "-" * 70)
    print("  Starting simulation... Press Ctrl+C to stop")
    print("-" * 70)
    print("\nExpected behavior:")
    print("  - Robot stands still (lower body)")
    print("  - Right arm waves up and down")
    print("  - Left arm stays at default position")

    try:
        while simulation_app.is_running():
            # Get action from controller
            action = controller.get_action(obs, sim_time)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Update time
            sim_time += dt

            # Log every 2 seconds
            if controller.step_count % 100 == 0:
                print(f"[Step {controller.step_count}] Time: {sim_time:.2f}s | Reward: {reward.mean().item():.4f}")

            # Reset if terminated
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {controller.step_count}")
                obs, info = env.reset()
                sim_time = 0.0

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
    finally:
        env.close()
        print("\n[✓] Simulation ended")


if __name__ == "__main__":
    main()