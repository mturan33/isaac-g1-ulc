# Copyright (c) 2025, VLM-RL G1 Project
# G1 Pick-and-Place Demo - V11
# Real pick-and-place with object position tracking

"""
G1 Pick-and-Place Demo V11
- Reads actual object position from scene
- Moves EE to object location
- Picks up and places in crate

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_place_v11.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick-and-Place V11")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo - V11")
print("  Real Object Tracking + Crate Placement")
print("=" * 70 + "\n")


# ============================================================================
# STATE MACHINE WITH DYNAMIC OBJECT TRACKING
# ============================================================================

class PickPlaceStateMachine:
    """State machine for pick-and-place with real object tracking."""

    def __init__(self, device: str):
        self.device = device

        # State durations
        self.state_durations = {
            "HOME": 2.0,
            "APPROACH": 2.0,
            "REACH": 2.0,
            "GRASP": 1.5,
            "LIFT": 2.0,
            "MOVE_TO_CRATE": 2.5,
            "LOWER": 2.0,
            "RELEASE": 1.0,
            "RETRACT": 2.0,
            "DONE": 999.0,
        }

        self.state_order = [
            "HOME", "APPROACH", "REACH", "GRASP", "LIFT",
            "MOVE_TO_CRATE", "LOWER", "RELEASE", "RETRACT", "DONE"
        ]

        self.current_state_idx = 0
        self.state_timer = 0.0
        self.cycle_count = 0

        # Will be set dynamically
        self.object_pos = None
        self.crate_pos = None
        self.base_pos = None

    def reset(self):
        self.current_state_idx = 0
        self.state_timer = 0.0
        print(f"\n[State] → {self.state_order[0]}")

    def step(self, dt: float):
        self.state_timer += dt
        current_state = self.state_order[self.current_state_idx]
        duration = self.state_durations[current_state]

        if self.state_timer >= duration:
            if self.current_state_idx < len(self.state_order) - 1:
                self.current_state_idx += 1
                self.state_timer = 0.0
                print(f"\n[State] → {self.state_order[self.current_state_idx]}")

    def get_current_state(self) -> str:
        return self.state_order[self.current_state_idx]

    def update_positions(self, object_pos: torch.Tensor, crate_pos: torch.Tensor, base_pos: torch.Tensor):
        """Update object and crate positions."""
        self.object_pos = object_pos.clone()
        self.crate_pos = crate_pos.clone()
        self.base_pos = base_pos.clone()

    def get_ee_target(self) -> torch.Tensor:
        """Get end-effector target position in world frame."""
        state = self.get_current_state()

        if self.object_pos is None or self.base_pos is None:
            # Default position if not initialized
            return self.base_pos + torch.tensor([[0.15, 0.25, 0.20]], device=self.device)

        # Heights relative to base
        approach_height = 0.25  # Above object
        grasp_height = 0.05  # At object level
        lift_height = 0.35  # Lifted

        if state == "HOME":
            # Home position - in front of robot
            return self.base_pos + torch.tensor([[0.15, 0.30, 0.20]], device=self.device)

        elif state == "APPROACH":
            # Above the object
            target = self.object_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + approach_height
            return target

        elif state == "REACH":
            # At object height
            target = self.object_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + grasp_height
            return target

        elif state == "GRASP":
            # Same as reach - holding position
            target = self.object_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + grasp_height
            return target

        elif state == "LIFT":
            # Lift up
            target = self.object_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + lift_height
            return target

        elif state == "MOVE_TO_CRATE":
            # Move towards crate, keep lifted
            target = self.crate_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + lift_height
            return target

        elif state == "LOWER":
            # Lower into crate
            target = self.crate_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + 0.15  # Into crate
            return target

        elif state == "RELEASE":
            # Same position for release
            target = self.crate_pos.clone()
            target[:, 2] = self.base_pos[:, 2] + 0.15
            return target

        elif state == "RETRACT":
            # Back to home
            return self.base_pos + torch.tensor([[0.15, 0.30, 0.25]], device=self.device)

        else:  # DONE
            return self.base_pos + torch.tensor([[0.15, 0.30, 0.20]], device=self.device)

    def is_done(self) -> bool:
        return self.get_current_state() == "DONE"


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
        obj = env.scene["object"]

        print(f"\n[INFO] Action dimension: {action_dim}")

        # Get body indices
        left_ee_idx = robot.body_names.index("left_wrist_yaw_link")
        right_ee_idx = robot.body_names.index("right_wrist_yaw_link")

        print(f"[INFO] Left EE body idx: {left_ee_idx}")
        print(f"[INFO] Right EE body idx: {right_ee_idx}")

        # Get initial positions
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()
        init_right_quat = robot.data.body_quat_w[:, right_ee_idx].clone()

        init_base_pos = robot.data.root_pos_w[:, :3].clone()
        init_left_offset = init_left_pos - init_base_pos

        # Get object and crate positions
        object_pos = obj.data.root_pos_w[:, :3].clone()

        # Crate position (from scene config - approximately)
        # The packing table crate is at the left side of the table
        crate_pos = init_base_pos + torch.tensor([[-0.35, 0.45, 0.0]], device=device)

        print(f"\n[INFO] Object position: {object_pos[0].tolist()}")
        print(f"[INFO] Crate position (approx): {crate_pos[0].tolist()}")
        print(f"[INFO] Robot base position: {init_base_pos[0].tolist()}")

        # Create state machine
        state_machine = PickPlaceStateMachine(device)
        state_machine.update_positions(object_pos, crate_pos, init_base_pos)
        state_machine.reset()

        print("\n[INFO] Starting simulation...")
        print("[INFO] Lower body: Agile Policy (standing still)")
        print("[INFO] Upper body: DiffIK (object tracking)\n")

        step_count = 0
        max_steps = 4000
        max_cycles = 2

        dt = env_cfg.sim.dt * env_cfg.decimation

        while simulation_app.is_running() and step_count < max_steps:
            # Update positions
            current_base_pos = robot.data.root_pos_w[:, :3]
            current_object_pos = obj.data.root_pos_w[:, :3]

            # Update state machine with current positions
            state_machine.update_positions(current_object_pos, crate_pos, current_base_pos)

            # Get EE target from state machine
            right_ee_target = state_machine.get_ee_target()

            # Create action tensor
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== LEFT ARM - Keep at initial offset =====
            target_left_pos = current_base_pos + init_left_offset
            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

            # ===== RIGHT ARM - Track target =====
            actions[:, 7:10] = right_ee_target
            actions[:, 10:14] = init_right_quat

            # ===== HANDS =====
            current_state = state_machine.get_current_state()
            if current_state in ["GRASP", "LIFT", "MOVE_TO_CRATE", "LOWER"]:
                # Close hand (positive values for gripper close)
                actions[:, 14:28] = 0.5  # Adjust based on hand joint config
            else:
                # Open hand
                actions[:, 14:28] = 0.0

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
                ee_error = torch.norm(right_ee_pos - right_ee_target, dim=-1).mean().item()
                obj_height = current_object_pos[:, 2].mean().item()

                status = "✓ STABLE" if root_height > 0.5 else "✗ FALLEN"

                print(f"[{step_count:4d}] {current_state:15s} | "
                      f"EE Err: {ee_error:.3f}m | "
                      f"Obj Z: {obj_height:.3f}m | "
                      f"Base Z: {root_height:.3f}m {status}")

            # Check for episode reset
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}")
                obs_dict, _ = env.reset()
                init_base_pos = robot.data.root_pos_w[:, :3].clone()
                object_pos = obj.data.root_pos_w[:, :3].clone()
                state_machine.update_positions(object_pos, crate_pos, init_base_pos)
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
        print(f"  ✓ Pick-and-Place Demo V11 Complete!")
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