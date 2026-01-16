"""
G1 Arm Reach - Play/Test Script (Stage 4)
==========================================

Eğitilmiş arm policy'yi test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_4_arm.py --num_envs 4 --checkpoint logs/ulc/ulc_g1_stage4_arm_2026-01-15_23-24-32/model_15999.pt
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Reach Play - Stage 4")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--duration", type=float, default=60.0, help="Play duration in seconds")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg


def main():
    # Environment oluştur
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration
    env_cfg.initial_target_radius = env_cfg.max_target_radius

    env = G1ArmReachEnv(cfg=env_cfg)

    # Model'i doğrudan yükle (RSL-RL runner kullanmadan)
    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")

    # Actor network'ü oluştur
    from rsl_rl.modules import ActorCritic

    # Model boyutlarını checkpoint'tan al
    actor_input_dim = checkpoint['model_state_dict']['actor.0.weight'].shape[1]
    actor_output_dim = checkpoint['model_state_dict']['actor.6.weight'].shape[0]

    print(f"[INFO] Model expects {actor_input_dim} observations, outputs {actor_output_dim} actions")

    actor_critic = ActorCritic(
        num_actor_obs=actor_input_dim,
        num_critic_obs=actor_input_dim,
        num_actions=actor_output_dim,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=0.5,
    ).to("cuda:0")

    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()

    print("[INFO] Policy loaded successfully!")
    print(f"[INFO] Running for {args.duration} seconds...")
    print("[INFO] Press Ctrl+C to exit\n")

    # İlk observation
    obs_dict = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Observation boyutunu kontrol et ve gerekirse pad et
    env_obs_dim = obs.shape[-1]
    if env_obs_dim != actor_input_dim:
        print(f"[WARNING] Env produces {env_obs_dim} obs, model expects {actor_input_dim}")
        print(f"[INFO] Will pad observations with zeros")

    step_count = 0
    total_reward = 0.0
    reach_count = 0

    try:
        while simulation_app.is_running():
            # Observation'ı model boyutuna getir
            if obs.shape[-1] < actor_input_dim:
                pad_size = actor_input_dim - obs.shape[-1]
                obs = torch.cat([obs, torch.zeros(obs.shape[0], pad_size, device=obs.device)], dim=-1)

            with torch.no_grad():
                actions = actor_critic.act_inference(obs)

            obs_dict = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

            # Reward hesapla (env'den al)
            rewards = env.reward_buf if hasattr(env, 'reward_buf') else torch.zeros(args.num_envs)

            total_reward += rewards.mean().item()
            step_count += 1

            if rewards.max().item() > 15:
                reach_count += 1

            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"[Step {step_count:5d}] Avg Reward: {avg_reward:+.3f} | Reaches: {reach_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print("\n" + "=" * 70)
    print("PLAY SESSION COMPLETE")
    print("=" * 70)
    print(f"  Total steps:       {step_count}")
    print(f"  Average reward:    {total_reward / max(step_count, 1):+.4f}")
    print(f"  Total reaches:     {reach_count}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()