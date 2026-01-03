# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V10
# Based on the WORKING test_locomanipulation.py

"""
G1 Pick-and-Place Demo V10
- Uses ManagerBasedRLEnv with LocomanipulationG1DiffIKEnvCfg (WORKS!)
- Adds state machine for pick-and-place task
- Lower body: Agile policy (automatic via env)
- Upper body: DiffIK (automatic via env)

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v10.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V10")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V10")
print("  Using ManagerBasedRLEnv (WORKING!)")
print("=" * 70 + "\n")


# ============================================================================
# STATE MACHINE
# ============================================================================

class PickPlaceStateMachine:
    """State machine for pick-and-place task."""

    def __init__(self, device: str, init_right_offset: torch.Tensor, init_right_quat: torch.Tensor):
        self.device = device
        self.init_right_offset = init_right_offset  # Default EE offset from base
        self.init_right_quat = init_right_quat  # Default EE orientation

        # State definitions with EE offsets from robot base
        # Robot at origin, table at (0, 0.55)
        # Cube approximately at (0, 0.45, 0.70) on table

        # All positions are OFFSETS from robot base (not world positions)
        # Right hand default offset is approximately (0.15, -0.25, 0.15)

        self.states = [
            {"name": "HOME", "ee_offset": [0.15, 0.25, 0.20], "dur": 2.0, "gripper": "open"},
            {"name": "APPROACH", "ee_offset": [0.0, 0.35, 0.15], "dur": 2.0, "gripper": "open"},
            {"name": "REACH", "ee_offset": [0.0, 0.42, 0.02], "dur": 1.5, "gripper": "open"},
            {"name": "GRASP", "ee_offset": [0.0, 0.42, 0.02], "dur": 1.0, "gripper": "close"},
            {"name": "LIFT", "ee_offset": [0.0, 0.35, 0.20], "dur": 1.5, "gripper": "close"},
            {"name": "MOVE", "ee_offset": [0.15, 0.35, 0.20], "dur": 2.0, "gripper": "close"},
            {"name": "LOWER", "ee_offset": [0.15, 0.42, 0.05], "dur": 1.5, "gripper": "close"},
            {"name": "RELEASE", "ee_offset": [0.15, 0.42, 0.05], "dur": 1.0, "gripper": "open"},
            {"name": "RETRACT", "ee_offset": [0.15, 0.25, 0.20], "dur": 2.0, "gripper": "open"},
            {"name": "DONE", "ee_offset": [0.15, 0.25, 0.20], "dur": 999.0, "gripper": "open"},
        ]

        self.current_state_idx = 0
        self.state_timer = 0.0
        self.cycle_count = 0

    def reset(self):
        self.current_state_idx = 0
        self.state_timer = 0.0
        print(f"\n[State] → {self.states[0]['name']}")

    def step(self, dt: float):
        self.state_timer += dt
        current_state = self.states[self.current_state_idx]

        if self.state_timer >= current_state["dur"]:
            if self.current_state_idx < len(self.states) - 1:
                self.current_state_idx += 1
                self.state_timer = 0.0
                print(f"\n[State] → {self.states[self.current_state_idx]['name']}")

    def get_current_state(self):
        return self.states[self.current_state_idx]

    def get_right_ee_offset(self) -> torch.Tensor:
        """Get right end-effector offset from base."""
        state = self.states[self.current_state_idx]
        return torch.tensor([state["ee_offset"]], device=self.device)

    def is_done(self) -> bool:
        return self.states[self.current_state_idx]["name"] == "DONE"


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_diffik_env_cfg import (
            LocomanipulationG1DiffIKEnvCfg
        )

        print("[INFO] Creating environment...")

        env_cfg = LocomanipulationG1DiffIKEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)

        print(f"[SUCCESS] ✓ Environment created!")

        obs_dict, _ = env.reset()

        action_dim = env.action_manager.total_action_dim
        num_envs = args_cli.num_envs
        device = env.device

        robot = env.scene["robot"]

        print(f"\n[INFO] Action dimension: {action_dim}")
        print("[INFO] Action format:")
        print("  [0:3]   - Left EE position")
        print("  [3:7]   - Left EE quaternion")
        print("  [7:10]  - Right EE position")
        print("  [10:14] - Right EE quaternion")
        print("  [14:28] - Hand joints")
        print("  [28:32] - Lower body command (vx, vy, wz, hip_height)")

        # Get initial EE positions (for left arm - keep static)
        left_ee_idx = 28
        right_ee_idx = 29

        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos
        init_right_offset = init_right_pos - init_base_pos

        print(f"\n[INFO] Initial right EE offset from base: {init_right_offset[0].tolist()}")

        # Create state machine
        state_machine = PickPlaceStateMachine(device, init_right_offset, init_right_quat)
        state_machine.reset()

        print("\n[INFO] Starting simulation...")
        print("[INFO] Lower body: Agile Policy (standing still)")
        print("[INFO] Upper body: DiffIK (state machine control)\n")

        step_count = 0
        max_steps = 3000
        max_cycles = 2

        dt = env_cfg.sim.dt * env_cfg.decimation  # Control dt

        while simulation_app.is_running() and step_count < max_steps:
            # Get current base position
            current_base_pos = robot.data.root_pos_w[:, :3]

            # Create action tensor
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== LEFT ARM - Keep at initial position =====
            target_left_pos = current_base_pos + init_left_offset
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

            # ===== RIGHT ARM - State machine control =====
            right_ee_offset = state_machine.get_right_ee_offset()
            target_right_pos = current_base_pos + right_ee_offset
            actions[:, 7:10] = target_right_pos
            actions[:, 10:14] = init_right_quat  # Keep same orientation

            # ===== HANDS - Neutral =====
            actions[:, 14:28] = 0.0

            # ===== LOWER BODY - Stand still =====
            actions[:, 28] = 0.0  # vx
            actions[:, 29] = 0.0  # vy
            actions[:, 30] = 0.0  # wz
            actions[:, 31] = 0.0  # hip_height

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Update state machine
            state_machine.step(dt)

            step_count += 1

            # Log every 50 steps
            if step_count % 50 == 0:
                root_height = robot.data.root_pos_w[:, 2].mean().item()
                right_ee_pos = robot.data.body_pos_w[:, right_ee_idx]
                ee_error = torch.norm(right_ee_pos - target_right_pos, dim=-1).mean().item()

                current_state = state_machine.get_current_state()
                status = "✓ STABLE" if root_height > 0.5 else "✗ FALLEN"

                print(f"[{step_count:4d}] {current_state['name']:10s} | "
                      f"EE Error: {ee_error:.3f}m | "
                      f"Base Z: {root_height:.3f}m {status}")

            # Check for episode reset
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}")
                print("    Resetting...")
                obs_dict, _ = env.reset()
                init_base_pos = robot.data.root_pos_w[:, :3].clone()
                state_machine.reset()

            # Check for cycle completion
            if state_machine.is_done() and state_machine.state_timer > 3.0:
                state_machine.cycle_count += 1
                if state_machine.cycle_count >= max_cycles:
                    print(f"\n[INFO] Completed {max_cycles} cycles!")
                    break
                print(f"\n[INFO] Starting cycle {state_machine.cycle_count + 1}...")
                state_machine.reset()

        print("\n" + "=" * 70)
        print(f"  ✓ Pick-and-Place Demo V10 Complete!")
        print(f"  Cycles completed: {state_machine.cycle_count}")
        print("=" * 70)

        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()