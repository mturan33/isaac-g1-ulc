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

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import configclass


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "g1_arm_reach"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration
    env_cfg.initial_target_radius = env_cfg.max_target_radius

    env = G1ArmReachEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")

    runner_cfg = G1ArmReachPPORunnerCfg()
    checkpoint_dir = os.path.dirname(args.checkpoint)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=checkpoint_dir, device="cuda:0")
    runner.load(args.checkpoint)

    print("[INFO] Policy loaded successfully!")
    print(f"[INFO] Running for {args.duration} seconds...")
    print("[INFO] Legend:")
    print("  🟢 Green sphere = Target position")
    print("  🟠 Orange sphere = End effector (palm + 2cm)")
    print("[INFO] Press Ctrl+C to exit\n")

    policy = runner.get_inference_policy(device="cuda:0")

    obs_dict = env.get_observations()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    step_count = 0
    total_reward = 0.0
    reach_count = 0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(obs)

            obs_dict, rewards, dones, infos = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

            total_reward += rewards.mean().item()
            step_count += 1

            if rewards.max().item() > 40:
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