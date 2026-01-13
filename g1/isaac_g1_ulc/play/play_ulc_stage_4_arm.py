"""
G1 Arm Reach - Play/Test Script (Stage 4)
==========================================

Eğitilmiş arm policy'yi test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_4_arm.py --num_envs 4 --checkpoint logs/ulc/ulc_g1_stage4_arm_XXXX/model_5000.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# ARGUMENT PARSING (BEFORE AppLauncher)
# =============================================================================

parser = argparse.ArgumentParser(description="G1 Arm Reach Play - Stage 4")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--duration", type=float, default=60.0, help="Play duration in seconds")

# Add AppLauncher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# LAUNCH APP (BEFORE any isaaclab imports)
# =============================================================================

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# NOW WE CAN IMPORT ISAACLAB MODULES
# =============================================================================

import torch

# Add env path to import our environment
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
from rsl_rl.modules import ActorCritic


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create environment
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration

    env = G1ArmReachEnv(cfg=env_cfg)

    # Load policy
    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")

    # Network architecture (must match training)
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

    # Load weights
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

    # Run simulation
    obs, _ = env.reset()
    step_count = 0
    total_reward = 0.0
    reaches = 0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy.act(obs["policy"])

            obs, rewards, terminated, truncated, info = env.step(actions)

            total_reward += rewards.mean().item()

            # Count reaches (reward > 10 indicates reaching bonus)
            if rewards.mean().item() > 10:
                reaches += 1

            step_count += 1

            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"[Step {step_count}] Avg reward: {avg_reward:.4f} | Reaches: {reaches}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Summary
    print("\n" + "=" * 60)
    print("PLAY SESSION COMPLETE")
    print("=" * 60)
    print(f"  Total steps:    {step_count}")
    print(f"  Average reward: {total_reward / max(step_count, 1):.4f}")
    print(f"  Total reaches:  {reaches}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()