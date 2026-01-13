"""
G1 Arm Reach - Play/Test Script
================================

Eğitilmiş arm policy'yi test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train_g1_arm_reach.py --num_envs 4 --play --checkpoint logs/rsl_rl/g1_arm_reach/XXXX/model_5000.pt
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--duration", type=float, default=60.0, help="Play duration in seconds")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import after app launch
from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
from rsl_rl.modules import ActorCritic

# Create environment
env_cfg = G1ArmReachEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.episode_length_s = args.duration

env = G1ArmReachEnv(cfg=env_cfg)

# Load policy
print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")

# Determine network architecture from config
actor_hidden_dims = [128, 128, 64]
critic_hidden_dims = [128, 128, 64]

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
print("[INFO] Green sphere = Target, Orange sphere = Palm")
print("[INFO] Press Ctrl+C to exit\n")

# Run simulation
obs, _ = env.reset()
step_count = 0
total_reward = 0.0

try:
    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy.act(obs["policy"])

        obs, rewards, terminated, truncated, info = env.step(actions)

        total_reward += rewards.mean().item()
        step_count += 1

        if step_count % 100 == 0:
            avg_reward = total_reward / step_count
            print(f"[Step {step_count}] Avg reward: {avg_reward:.4f}")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

# Summary
print("\n" + "=" * 60)
print("PLAY SESSION COMPLETE")
print("=" * 60)
print(f"  Total steps: {step_count}")
print(f"  Average reward: {total_reward / max(step_count, 1):.4f}")
print("=" * 60)

env.close()
simulation_app.close()