"""
G1 Arm Reach - Play/Test Script (Stage 4)
==========================================

Eğitilmiş arm policy'yi test et - smooth motion metrikleri ile.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_4_arm.py --num_envs 4 --checkpoint logs/ulc/ulc_g1_stage4_arm_XXXX/model_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description="G1 Arm Reach Play - Stage 4")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--duration", type=float, default=60.0, help="Play duration in seconds")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# LAUNCH APP
# =============================================================================

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# IMPORTS
# =============================================================================

import torch
import numpy as np

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
from rsl_rl.modules import ActorCritic


# =============================================================================
# SMOOTHNESS METRICS
# =============================================================================

class SmoothnessMetrics:
    """Track smoothness of motion during playback."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.prev_actions = None
        self.prev_prev_actions = None
        self.prev_joint_vel = None

        self.action_rates = []
        self.action_accels = []
        self.joint_vels = []
        self.joint_accels = []

    def update(self, actions, joint_vel):
        if self.prev_actions is not None:
            # Action rate
            action_rate = torch.norm(actions - self.prev_actions, dim=-1).mean().item()
            self.action_rates.append(action_rate)

            # Action acceleration
            if self.prev_prev_actions is not None:
                action_accel = torch.norm(
                    (actions - self.prev_actions) -
                    (self.prev_actions - self.prev_prev_actions),
                    dim=-1
                ).mean().item()
                self.action_accels.append(action_accel)

        if self.prev_joint_vel is not None:
            # Joint acceleration
            joint_accel = torch.norm(joint_vel - self.prev_joint_vel, dim=-1).mean().item()
            self.joint_accels.append(joint_accel)

        # Joint velocity magnitude
        joint_vel_mag = torch.norm(joint_vel, dim=-1).mean().item()
        self.joint_vels.append(joint_vel_mag)

        # Update history
        self.prev_prev_actions = self.prev_actions.clone() if self.prev_actions is not None else None
        self.prev_actions = actions.clone()
        self.prev_joint_vel = joint_vel.clone()

    def get_summary(self):
        return {
            "action_rate_mean": np.mean(self.action_rates) if self.action_rates else 0,
            "action_rate_max": np.max(self.action_rates) if self.action_rates else 0,
            "action_accel_mean": np.mean(self.action_accels) if self.action_accels else 0,
            "action_accel_max": np.max(self.action_accels) if self.action_accels else 0,
            "joint_vel_mean": np.mean(self.joint_vels) if self.joint_vels else 0,
            "joint_vel_max": np.max(self.joint_vels) if self.joint_vels else 0,
            "joint_accel_mean": np.mean(self.joint_accels) if self.joint_accels else 0,
            "joint_accel_max": np.max(self.joint_accels) if self.joint_accels else 0,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration

    # Set max target radius for testing
    env_cfg.initial_target_radius = env_cfg.max_target_radius

    env = G1ArmReachEnv(cfg=env_cfg)

    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")

    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]

    policy = ActorCritic(
        num_actor_obs=env_cfg.num_observations,
        num_critic_obs=env_cfg.num_observations,
        num_actions=env_cfg.num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation="elu",
    ).to("cuda:0")

    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    print("[INFO] Policy loaded successfully!")
    print(f"[INFO] Running for {args.duration} seconds...")
    print("[INFO] Legend:")
    print("  🟢 Green sphere = Target position")
    print("  🟢 Green cone = Target orientation")
    print("  🟠 Orange sphere = End effector (2cm in front of palm)")
    print("  🟣 Purple sphere = Palm center")
    print("[INFO] Press Ctrl+C to exit\n")

    # Initialize metrics
    metrics = SmoothnessMetrics(device="cuda:0")

    obs, _ = env.reset()
    step_count = 0
    total_reward = 0.0
    reaches = 0
    pos_threshold = env_cfg.pos_threshold

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy.act(obs["policy"])

            # Track smoothness
            arm_joint_vel = env.robot.data.joint_vel[:, env.arm_joint_indices]
            metrics.update(actions, arm_joint_vel)

            obs, rewards, terminated, truncated, info = env.step(actions)

            total_reward += rewards.mean().item()

            # Check if reached (reward > 15 indicates reaching bonus)
            if rewards.mean().item() > 15:
                reaches += 1

            step_count += 1

            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                summary = metrics.get_summary()
                print(f"[Step {step_count:5d}] Reward: {avg_reward:.3f} | "
                      f"Reaches: {reaches:3d} | "
                      f"JointVel: {summary['joint_vel_mean']:.3f} rad/s")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Final summary
    summary = metrics.get_summary()

    print("\n" + "=" * 70)
    print("PLAY SESSION COMPLETE")
    print("=" * 70)
    print(f"  Total steps:       {step_count}")
    print(f"  Average reward:    {total_reward / max(step_count, 1):.4f}")
    print(f"  Total reaches:     {reaches}")
    print("-" * 70)
    print("  SMOOTHNESS METRICS:")
    print(f"    Action Rate (mean/max):  {summary['action_rate_mean']:.4f} / {summary['action_rate_max']:.4f}")
    print(f"    Action Accel (mean/max): {summary['action_accel_mean']:.4f} / {summary['action_accel_max']:.4f}")
    print(f"    Joint Vel (mean/max):    {summary['joint_vel_mean']:.3f} / {summary['joint_vel_max']:.3f} rad/s")
    print(f"    Joint Accel (mean/max):  {summary['joint_accel_mean']:.4f} / {summary['joint_accel_max']:.4f}")
    print("-" * 70)

    # Quality assessment
    if summary['joint_vel_max'] < 2.0 and summary['action_accel_max'] < 0.5:
        print("  ✅ SMOOTH MOTION ACHIEVED!")
    elif summary['joint_vel_max'] < 3.0:
        print("  ⚠️ Motion is acceptable but could be smoother")
    else:
        print("  ❌ Motion is too jerky - consider more training")

    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()