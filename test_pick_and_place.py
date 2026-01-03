# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V12
# Fixed workspace targets - robot reaches forward

"""
G1 Pick-and-Place Demo V12
- Uses RELATIVE offsets from initial EE position (proven to work in V10)
- Moves within robot's workspace
- State machine with smooth transitions

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v12.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V12")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V12")
print("  Fixed Workspace Targets")
print("=" * 70 + "\n")


# ============================================================================
# STATE MACHINE - WORKSPACE-AWARE
# ============================================================================

class PickPlaceStateMachine:
    """State machine with workspace-aware targets."""

    def __init__(self, device: str, init_right_offset: torch.Tensor):
        self.device = device
        self.init_right_offset = init_right_offset.clone()

        # Get initial offset values
        init_x = init_right_offset[0, 0].item()
        init_y = init_right_offset[0, 1].item()
        init_z = init_right_offset[0, 2].item()

        print(f"[INFO] Initial right EE offset: x={init_x:.3f}, y={init_y:.3f}, z={init_z:.3f}")

        # Define states with OFFSETS from robot base
        # All targets are in front of robot (positive Y) and within arm reach
        # Right arm can reach: X: -0.2 to 0.4, Y: 0.2 to 0.6, Z: -0.2 to 0.4 (from base)

        self.states = [
            # HOME - default position
            {"name": "HOME", "offset": [init_x, init_y, init_z + 0.05], "dur": 2.0, "grip": 0.0},

            # APPROACH - move forward and slightly right, above table
            {"name": "APPROACH", "offset": [0.05, 0.35, 0.10], "dur": 2.5, "grip": 0.0},

            # REACH - go down towards table surface
            {"name": "REACH", "offset": [0.05, 0.40, -0.05], "dur": 2.0, "grip": 0.0},

            # GRASP - close gripper
            {"name": "GRASP", "offset": [0.05, 0.40, -0.05], "dur": 1.5, "grip": 0.5},

            # LIFT - raise up
            {"name": "LIFT", "offset": [0.05, 0.35, 0.15], "dur": 2.0, "grip": 0.5},

            # MOVE - move to left side (towards crate)
            {"name": "MOVE", "offset": [-0.15, 0.35, 0.15], "dur": 2.5, "grip": 0.5},

            # LOWER - go down into crate area
            {"name": "LOWER", "offset": [-0.15, 0.40, 0.0], "dur": 2.0, "grip": 0.5},

            # RELEASE - open gripper
            {"name": "RELEASE", "offset": [-0.15, 0.40, 0.0], "dur": 1.0, "grip": 0.0},

            # RETRACT - back to home
            {"name": "RETRACT", "offset": [init_x, init_y, init_z + 0.10], "dur": 2.0, "grip": 0.0},

            # DONE
            {"name": "DONE", "offset": [init_x, init_y, init_z], "dur": 999.0, "grip": 0.0},
        ]

        self.current_state_idx = 0
        self.state_timer = 0.0
        self.cycle_count = 0

        # For smooth interpolation
        self.prev_offset = torch.tensor([self.states[0]["offset"]], device=device)
        self.target_offset = torch.tensor([self.states[0]["offset"]], device=device)
        self.interp_progress = 1.0

    def reset(self):
        self.current_state_idx = 0
        self.state_timer = 0.0
        self.interp_progress = 1.0
        state = self.states[0]
        self.prev_offset = torch.tensor([state["offset"]], device=self.device)
        self.target_offset = torch.tensor([state["offset"]], device=self.device)
        print(f"\n[State] → {state['name']}")

    def step(self, dt: float):
        self.state_timer += dt
        current_state = self.states[self.current_state_idx]

        # Update interpolation
        if self.interp_progress < 1.0:
            self.interp_progress = min(1.0, self.interp_progress + dt / 1.0)  # 1 second transition

        if self.state_timer >= current_state["dur"]:
            if self.current_state_idx < len(self.states) - 1:
                # Save current as prev for interpolation
                self.prev_offset = self.target_offset.clone()

                self.current_state_idx += 1
                self.state_timer = 0.0
                self.interp_progress = 0.0

                next_state = self.states[self.current_state_idx]
                self.target_offset = torch.tensor([next_state["offset"]], device=self.device)
                print(f"\n[State] → {next_state['name']}")

    def get_current_state(self) -> dict:
        return self.states[self.current_state_idx]

    def get_right_ee_offset(self) -> torch.Tensor:
        """Get interpolated end-effector offset from base."""
        # Smooth interpolation between states
        t = self.interp_progress
        # Smooth step function
        t = t * t * (3 - 2 * t)

        return self.prev_offset * (1 - t) + self.target_offset * t

    def get_gripper_value(self) -> float:
        """Get gripper command (0=open, 0.5=closed)."""
        return self.states[self.current_state_idx]["grip"]

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

        # Get body indices
        left_ee_idx = robot.body_names.index("left_wrist_yaw_link")
        right_ee_idx = robot.body_names.index("right_wrist_yaw_link")

        # Get initial positions
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_pos = robot.data.body_pos_w[:, right_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos
        init_right_offset = init_right_pos - init_base_pos

        print(f"[INFO] Robot base: {init_base_pos[0].tolist()}")
        print(f"[INFO] Right EE world pos: {init_right_pos[0].tolist()}")

        # Create state machine
        state_machine = PickPlaceStateMachine(device, init_right_offset)
        state_machine.reset()

        print("\n[INFO] Starting simulation...")
        print("[INFO] Lower body: Agile Policy (standing)")
        print("[INFO] Upper body: DiffIK (smooth transitions)\n")

        step_count = 0
        max_steps = 4000
        max_cycles = 2

        dt = env_cfg.sim.dt * env_cfg.decimation

        while simulation_app.is_running() and step_count < max_steps:
            # Get current base position
            current_base_pos = robot.data.root_pos_w[:, :3]

            # Get target offset from state machine
            right_ee_offset = state_machine.get_right_ee_offset()
            gripper_val = state_machine.get_gripper_value()

            # Calculate world position
            target_right_pos = current_base_pos + right_ee_offset
            target_left_pos = current_base_pos + init_left_offset

            # Create action tensor
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== LEFT ARM - Keep at initial offset =====
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

            # ===== RIGHT ARM - State machine control =====
            actions[:, 7:10] = target_right_pos
            actions[:, 10:14] = init_right_quat

            # ===== HANDS =====
            actions[:, 14:28] = gripper_val

            # ===== LOWER BODY - Stand still =====
            actions[:, 28:32] = 0.0

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

                # Also show actual vs target offset
                actual_offset = right_ee_pos - current_base_pos
                target_off = right_ee_offset[0]

                print(f"[{step_count:4d}] {current_state['name']:10s} | "
                      f"EE Err: {ee_error:.3f}m | "
                      f"Target: [{target_off[0].item():.2f}, {target_off[1].item():.2f}, {target_off[2].item():.2f}] | "
                      f"Base Z: {root_height:.3f}m {status}")

            # Check for episode reset
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}")
                obs_dict, _ = env.reset()
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
        print(f"  ✓ Pick-and-Place Demo V12 Complete!")
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