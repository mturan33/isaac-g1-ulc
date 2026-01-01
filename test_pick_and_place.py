# Copyright (c) 2025, VLM-RL G1 Project
# Pick and Place Test Script

"""
G1 Pick-and-Place Demo with DiffIK
State machine: Approach → Grasp → Lift → Transport → Release

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\test_pick_and_place.py

[0] HOME      (1.0s)  → Başlangıç pozisyonu
[1] APPROACH  (2.5s)  → Eli objenin 10cm üstüne götür
[2] REACH     (1.5s)  → Eli objeye indir
[3] GRASP     (1.0s)  → Eli kapat (tut)
[4] LIFT      (1.5s)  → Objeyi kaldır
[5] TRANSPORT (2.5s)  → Kutuya götür
[6] LOWER     (1.5s)  → Kutuya indir
[7] RELEASE   (1.0s)  → Eli aç (bırak)
[8] RETRACT   (1.5s)  → Eli geri çek
[9] DONE      (∞)     → Tamamlandı
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Pick and Place Demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--walk", action="store_true", help="Enable walking during task")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "=" * 70)
print("  G1 Pick-and-Place Demo")
print("  Task: Grab steering wheel → Move to box → Release")
print("=" * 70 + "\n")


# ============================================================================
# STATE MACHINE DEFINITION
# ============================================================================

class PickAndPlaceStateMachine:
    """Simple state machine for pick-and-place task."""

    def __init__(self, device):
        self.device = device
        self.current_state = 0
        self.state_timer = 0.0
        self.dt = 0.02  # 50 Hz

        # Gripper values: -1 = fully open, +1 = fully closed
        GRIPPER_OPEN = -1.0
        GRIPPER_CLOSED = 1.0

        # ============================================================
        # TASK CONFIGURATION - MODIFY THESE VALUES AS NEEDED
        # ============================================================

        # Object position (steering wheel) - from scene
        # The object spawns at [-0.35, 0.45, 0.6996] in env config
        self.OBJECT_POS = [-0.35, 0.45, 0.72]  # Slightly above to approach

        # Box/target position (on packing table)
        # Table is at [0.0, 0.55, -0.3], surface is ~0.7m high
        self.BOX_POS = [0.0, 0.55, 0.85]

        # Lift height
        self.LIFT_HEIGHT = 0.95

        # Home position (rest pose offset from body)
        self.HOME_OFFSET = [0.15, 0.20, 0.10]  # x, y, z from pelvis

        # ============================================================
        # STATE DEFINITIONS
        # ============================================================

        self.states = [
            {
                "name": "HOME",
                "description": "Starting position - arm at rest",
                "target_offset": self.HOME_OFFSET,  # Relative to body
                "use_absolute": False,
                "gripper": GRIPPER_OPEN,
                "duration": 1.0,
            },
            {
                "name": "APPROACH",
                "description": "Move hand above object",
                "target_pos": [self.OBJECT_POS[0], self.OBJECT_POS[1], self.OBJECT_POS[2] + 0.10],
                "use_absolute": True,
                "gripper": GRIPPER_OPEN,
                "duration": 2.5,
            },
            {
                "name": "REACH",
                "description": "Lower hand to object",
                "target_pos": self.OBJECT_POS,
                "use_absolute": True,
                "gripper": GRIPPER_OPEN,
                "duration": 1.5,
            },
            {
                "name": "GRASP",
                "description": "Close gripper on object",
                "target_pos": self.OBJECT_POS,
                "use_absolute": True,
                "gripper": GRIPPER_CLOSED,
                "duration": 1.0,
            },
            {
                "name": "LIFT",
                "description": "Lift object up",
                "target_pos": [self.OBJECT_POS[0], self.OBJECT_POS[1], self.LIFT_HEIGHT],
                "use_absolute": True,
                "gripper": GRIPPER_CLOSED,
                "duration": 1.5,
            },
            {
                "name": "TRANSPORT",
                "description": "Move to box location",
                "target_pos": [self.BOX_POS[0], self.BOX_POS[1], self.LIFT_HEIGHT],
                "use_absolute": True,
                "gripper": GRIPPER_CLOSED,
                "duration": 2.5,
            },
            {
                "name": "LOWER",
                "description": "Lower object into box",
                "target_pos": self.BOX_POS,
                "use_absolute": True,
                "gripper": GRIPPER_CLOSED,
                "duration": 1.5,
            },
            {
                "name": "RELEASE",
                "description": "Open gripper to release",
                "target_pos": self.BOX_POS,
                "use_absolute": True,
                "gripper": GRIPPER_OPEN,
                "duration": 1.0,
            },
            {
                "name": "RETRACT",
                "description": "Move hand up and away",
                "target_pos": [self.BOX_POS[0], self.BOX_POS[1], self.LIFT_HEIGHT],
                "use_absolute": True,
                "gripper": GRIPPER_OPEN,
                "duration": 1.5,
            },
            {
                "name": "DONE",
                "description": "Task complete!",
                "target_offset": self.HOME_OFFSET,
                "use_absolute": False,
                "gripper": GRIPPER_OPEN,
                "duration": 999.0,  # Stay here
            },
        ]

        print(f"[StateMachine] Initialized with {len(self.states)} states:")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s['name']}: {s['description']} ({s['duration']}s)")

    def get_current_state(self):
        return self.states[self.current_state]

    def update(self, dt):
        """Update state machine, return True if state changed."""
        self.state_timer += dt

        current = self.get_current_state()
        if self.state_timer >= current["duration"] and self.current_state < len(self.states) - 1:
            self.current_state += 1
            self.state_timer = 0.0
            new_state = self.get_current_state()
            print(f"\n[State Change] → {new_state['name']}: {new_state['description']}")
            return True
        return False

    def get_target_position(self, base_pos):
        """Get target end-effector position (world frame)."""
        current = self.get_current_state()

        if current.get("use_absolute", False):
            # Absolute world position
            return torch.tensor(current["target_pos"], device=self.device)
        else:
            # Relative to robot base
            offset = torch.tensor(current["target_offset"], device=self.device)
            return base_pos + offset

    def get_gripper_value(self):
        """Get gripper target value (-1 to +1)."""
        return self.get_current_state()["gripper"]

    def reset(self):
        """Reset state machine to beginning."""
        self.current_state = 0
        self.state_timer = 0.0
        print(f"\n[StateMachine] Reset to state 0: {self.states[0]['name']}")


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

        # Get EE body indices
        left_ee_idx = 28
        right_ee_idx = 29

        # Initial EE states for left arm (keep it stable)
        init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
        init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()

        # Default quaternion for right hand (palm down)
        # w, x, y, z format - pointing forward and down
        right_ee_quat = torch.tensor([[0.707, 0.0, 0.707, 0.0]], device=device)

        # Create state machine
        state_machine = PickAndPlaceStateMachine(device)

        # Print object position from scene
        obj_pos = obj.data.root_pos_w[0].cpu().numpy()
        print(f"\n[INFO] Object (steering wheel) position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")

        print(f"\n[INFO] Starting pick-and-place task...")
        print(f"[INFO] Press Ctrl+C to stop.\n")

        print(
            f"[State] → {state_machine.get_current_state()['name']}: {state_machine.get_current_state()['description']}")

        step_count = 0
        max_steps = 5000  # ~100 seconds

        while simulation_app.is_running() and step_count < max_steps:

            # Get current robot state
            base_pos = robot.data.root_pos_w[:, :3]

            # Update state machine
            state_machine.update(state_machine.dt)

            # Get targets from state machine
            target_right_pos = state_machine.get_target_position(base_pos[0])
            gripper_value = state_machine.get_gripper_value()

            # Build actions
            actions = torch.zeros(num_envs, action_dim, device=device)

            # ===== LEFT ARM - Keep at initial position relative to body =====
            init_base_pos = robot.data.root_pos_w[:, :3]
            init_left_offset = init_left_pos - init_base_pos
            target_left_pos = base_pos + init_left_offset

            actions[:, 0:3] = target_left_pos
            actions[:, 3:7] = init_left_quat

            # ===== RIGHT ARM - DiffIK target from state machine =====
            actions[:, 7:10] = target_right_pos.unsqueeze(0) if target_right_pos.dim() == 1 else target_right_pos
            actions[:, 10:14] = right_ee_quat

            # ===== GRIPPER - All hand joints =====
            actions[:, 14:28] = gripper_value

            # ===== LOWER BODY =====
            if args_cli.walk:
                actions[:, 28] = -0.2  # Slow walk
            else:
                actions[:, 28:32] = 0.0  # Stand still

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            step_count += 1

            # Log every 50 steps
            if step_count % 50 == 0:
                current_state = state_machine.get_current_state()
                right_ee_pos = robot.data.body_pos_w[0, right_ee_idx].cpu().numpy()
                target_pos = target_right_pos.cpu().numpy()

                error = ((right_ee_pos[0] - target_pos[0]) ** 2 +
                         (right_ee_pos[1] - target_pos[1]) ** 2 +
                         (right_ee_pos[2] - target_pos[2]) ** 2) ** 0.5

                obj_pos = obj.data.root_pos_w[0].cpu().numpy()

                print(f"[Step {step_count:4d}] State: {current_state['name']:10s} | "
                      f"EE: [{right_ee_pos[0]:.2f}, {right_ee_pos[1]:.2f}, {right_ee_pos[2]:.2f}] | "
                      f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] | "
                      f"Error: {error:.3f}m | "
                      f"Obj: [{obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f}]")

            # Check for episode end
            if terminated.any() or truncated.any():
                print(f"\n[!] Episode ended at step {step_count}")
                print("    Resetting...")
                obs_dict, _ = env.reset()
                state_machine.reset()
                init_left_pos = robot.data.body_pos_w[:, left_ee_idx].clone()
                init_left_quat = robot.data.body_quat_w[:, left_ee_idx].clone()

        print("\n" + "=" * 70)
        print("  ✓ Pick-and-Place Demo Complete!")
        print("=" * 70)
        env.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    simulation_app.close()