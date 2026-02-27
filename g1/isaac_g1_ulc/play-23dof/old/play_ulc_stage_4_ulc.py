#!/usr/bin/env python3
"""
ULC G1 Stage 4 - Enhanced Play Script with Random Arm Motion
=============================================================
Robot walks while arms randomly move to new targets within workspace.
Targets change every few seconds with smooth interpolation.
"""

import argparse
import math
import random
import time

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="ULC G1 Stage 4 - Random Arm Motion Demo")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--vx", type=float, default=0.3, help="Forward velocity command")
parser.add_argument("--pitch", type=float, default=0.0, help="Torso pitch command (rad)")
parser.add_argument("--roll", type=float, default=0.0, help="Torso roll command (rad)")

# Random motion parameters
parser.add_argument("--target_interval", type=float, default=3.0, help="Seconds between target changes")
parser.add_argument("--interpolation_speed", type=float, default=0.02, help="Interpolation speed (0-1)")
parser.add_argument("--sync_arms", action="store_true", help="Sync both arms to same position")
parser.add_argument("--mirror_arms", action="store_true", help="Mirror arm positions (symmetric)")

# Manual override (disables random motion)
parser.add_argument("--left_shoulder_pitch", type=float, default=None, help="Fixed left shoulder pitch")
parser.add_argument("--left_elbow", type=float, default=None, help="Fixed left elbow")
parser.add_argument("--right_shoulder_pitch", type=float, default=None, help="Fixed right shoulder pitch")
parser.add_argument("--right_elbow", type=float, default=None, help="Fixed right elbow")

# Presets
parser.add_argument("--preset", type=str, default=None,
                    choices=["arms_up", "arms_front", "arms_down", "wave", "t_pose", "zombie"],
                    help="Use preset arm configuration")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import torch
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_env_stage4 import ULCG1Stage4Env, ULCG1Stage4EnvCfg

# ============================================================
# ARM WORKSPACE DEFINITION (based on training ranges)
# ============================================================
# G1 Joint Convention (discovered through testing):
#   shoulder_pitch: negative = forward, positive = backward
#   elbow: negative = extend, positive = flex (bend)

ARM_WORKSPACE = {
    # Training range was ±0.8 rad, stay within for stability
    "shoulder_pitch": {"min": -0.7, "max": 0.7},   # Forward/backward
    "elbow": {"min": -0.5, "max": 0.3},            # Extend/flex
}

# ============================================================
# PRESET CONFIGURATIONS
# ============================================================
PRESETS = {
    "arms_up": {
        "left": {"shoulder_pitch": 0.6, "elbow": 0.0},     # Arms raised back/up
        "right": {"shoulder_pitch": 0.6, "elbow": 0.0},
    },
    "arms_front": {
        "left": {"shoulder_pitch": -0.6, "elbow": -0.3},   # Arms forward, slightly extended
        "right": {"shoulder_pitch": -0.6, "elbow": -0.3},
    },
    "arms_down": {
        "left": {"shoulder_pitch": 0.0, "elbow": 0.0},     # Natural hanging
        "right": {"shoulder_pitch": 0.0, "elbow": 0.0},
    },
    "t_pose": {
        "left": {"shoulder_pitch": 0.0, "elbow": -0.4},    # Arms out to sides (approximation)
        "right": {"shoulder_pitch": 0.0, "elbow": -0.4},
    },
    "zombie": {
        "left": {"shoulder_pitch": -0.5, "elbow": -0.2},   # Classic zombie pose
        "right": {"shoulder_pitch": -0.5, "elbow": -0.2},
    },
    "wave": {
        "left": {"shoulder_pitch": 0.0, "elbow": 0.0},     # Left down
        "right": {"shoulder_pitch": 0.6, "elbow": 0.3},    # Right up waving
    },
}


def random_arm_target():
    """Generate random arm target within workspace."""
    return {
        "shoulder_pitch": random.uniform(
            ARM_WORKSPACE["shoulder_pitch"]["min"],
            ARM_WORKSPACE["shoulder_pitch"]["max"]
        ),
        "elbow": random.uniform(
            ARM_WORKSPACE["elbow"]["min"],
            ARM_WORKSPACE["elbow"]["max"]
        ),
    }


def interpolate(current, target, speed):
    """Smoothly interpolate between current and target."""
    return current + (target - current) * speed


class ArmController:
    """Manages smooth arm motion with random targets."""

    def __init__(self, target_interval=3.0, interp_speed=0.02, sync=False, mirror=False):
        self.target_interval = target_interval
        self.interp_speed = interp_speed
        self.sync = sync
        self.mirror = mirror

        # Current arm positions
        self.left_arm = {"shoulder_pitch": 0.0, "elbow": 0.0}
        self.right_arm = {"shoulder_pitch": 0.0, "elbow": 0.0}

        # Target positions
        self.left_target = random_arm_target()
        self.right_target = random_arm_target()

        if self.sync:
            self.right_target = self.left_target.copy()
        elif self.mirror:
            self.right_target = {
                "shoulder_pitch": self.left_target["shoulder_pitch"],
                "elbow": self.left_target["elbow"],
            }

        self.last_target_time = time.time()

    def update(self):
        """Update arm positions, generate new targets if needed."""
        current_time = time.time()

        # Check if it's time for new targets
        if current_time - self.last_target_time >= self.target_interval:
            self.left_target = random_arm_target()

            if self.sync:
                self.right_target = self.left_target.copy()
            elif self.mirror:
                self.right_target = {
                    "shoulder_pitch": self.left_target["shoulder_pitch"],
                    "elbow": self.left_target["elbow"],
                }
            else:
                self.right_target = random_arm_target()

            self.last_target_time = current_time
            print(f"\n[New Target] L: sp={self.left_target['shoulder_pitch']:.2f}, "
                  f"elbow={self.left_target['elbow']:.2f} | "
                  f"R: sp={self.right_target['shoulder_pitch']:.2f}, "
                  f"elbow={self.right_target['elbow']:.2f}")

        # Interpolate towards targets
        self.left_arm["shoulder_pitch"] = interpolate(
            self.left_arm["shoulder_pitch"],
            self.left_target["shoulder_pitch"],
            self.interp_speed
        )
        self.left_arm["elbow"] = interpolate(
            self.left_arm["elbow"],
            self.left_target["elbow"],
            self.interp_speed
        )
        self.right_arm["shoulder_pitch"] = interpolate(
            self.right_arm["shoulder_pitch"],
            self.right_target["shoulder_pitch"],
            self.interp_speed
        )
        self.right_arm["elbow"] = interpolate(
            self.right_arm["elbow"],
            self.right_target["elbow"],
            self.interp_speed
        )

        return self.left_arm, self.right_arm


def main():
    print("=" * 60)
    print("ULC G1 STAGE 4 - RANDOM ARM MOTION DEMO")
    print("=" * 60)

    # Determine arm control mode
    use_random = True
    fixed_left = {"shoulder_pitch": 0.0, "elbow": 0.0}
    fixed_right = {"shoulder_pitch": 0.0, "elbow": 0.0}

    # Check for preset
    if args.preset:
        preset = PRESETS[args.preset]
        fixed_left = preset["left"]
        fixed_right = preset["right"]
        use_random = False
        print(f"Using preset: {args.preset}")

    # Check for manual override
    if args.left_shoulder_pitch is not None:
        fixed_left["shoulder_pitch"] = args.left_shoulder_pitch
        use_random = False
    if args.left_elbow is not None:
        fixed_left["elbow"] = args.left_elbow
        use_random = False
    if args.right_shoulder_pitch is not None:
        fixed_right["shoulder_pitch"] = args.right_shoulder_pitch
        use_random = False
    if args.right_elbow is not None:
        fixed_right["elbow"] = args.right_elbow
        use_random = False

    if use_random:
        print(f"Mode: RANDOM ARM MOTION")
        print(f"Target interval: {args.target_interval}s")
        print(f"Sync arms: {args.sync_arms}, Mirror arms: {args.mirror_arms}")
    else:
        print(f"Mode: FIXED POSITION")
        print(f"Left arm: sp={fixed_left['shoulder_pitch']:.2f}, elbow={fixed_left['elbow']:.2f}")
        print(f"Right arm: sp={fixed_right['shoulder_pitch']:.2f}, elbow={fixed_right['elbow']:.2f}")

    print(f"\nWalking: vx={args.vx}, pitch={args.pitch}, roll={args.roll}")
    print("=" * 60)

    # Load checkpoint
    print(f"\n[INFO] Loading: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    print(f"[INFO] Best reward: {checkpoint.get('best_reward', 'N/A')}")
    print(f"[INFO] Iteration: {checkpoint.get('iteration', 'N/A')}")
    print(f"[INFO] Curriculum level: {checkpoint.get('curriculum_level', 'N/A')}")

    # Create environment
    env_cfg = ULCG1Stage4EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ULCG1Stage4Env(cfg=env_cfg)

    # Load policy
    from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.agents.rsl_rl_ppo_cfg_stage4 import agent_cfg

    policy_cfg = agent_cfg.policy
    policy = policy_cfg.class_name(
        num_actor_obs=env.observation_space.shape[1],
        num_critic_obs=env.observation_space.shape[1],
        num_actions=env.action_space.shape[1],
        **policy_cfg.to_dict()
    ).to("cuda:0")

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    print("[INFO] Model loaded successfully!")

    # Initialize arm controller
    arm_controller = ArmController(
        target_interval=args.target_interval,
        interp_speed=args.interpolation_speed,
        sync=args.sync_arms,
        mirror=args.mirror_arms
    )

    # Set initial commands
    if use_random:
        left_arm, right_arm = arm_controller.update()
    else:
        left_arm = fixed_left
        right_arm = fixed_right

    env.set_commands(
        vx=args.vx,
        pitch=args.pitch,
        roll=args.roll,
        left_shoulder_pitch=left_arm["shoulder_pitch"],
        left_elbow=left_arm["elbow"],
        right_shoulder_pitch=right_arm["shoulder_pitch"],
        right_elbow=right_arm["elbow"],
    )

    print(f"\n[Play] Starting simulation...")
    print(f"       Press Ctrl+C to stop")
    print("-" * 60)

    # Run simulation
    obs, _ = env.reset()
    step = 0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy.act_inference(obs)

            obs, rewards, dones, truncated, info = env.step(actions)
            step += 1

            # Update arm positions (if random mode)
            if use_random:
                left_arm, right_arm = arm_controller.update()
                env.set_commands(
                    vx=args.vx,
                    pitch=args.pitch,
                    roll=args.roll,
                    left_shoulder_pitch=left_arm["shoulder_pitch"],
                    left_elbow=left_arm["elbow"],
                    right_shoulder_pitch=right_arm["shoulder_pitch"],
                    right_elbow=right_arm["elbow"],
                )

            # Print status
            if step % 100 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vel_x = env.robot.data.root_lin_vel_b[:, 0].mean().item()

                # Get actual joint positions for arms
                actual_left_sp = env.left_arm_pos[:, 0].mean().item()
                actual_right_sp = env.right_arm_pos[:, 0].mean().item()

                print(f"Step {step:5d} | H={height:.3f}m | Vx={vel_x:.2f}m/s | "
                      f"L_sp={actual_left_sp:.2f} | R_sp={actual_right_sp:.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()