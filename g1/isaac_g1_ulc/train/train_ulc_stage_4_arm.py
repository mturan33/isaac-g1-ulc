"""
G1 Arm Reach - Training Script (Stage 4) with Curriculum Learning
===================================================================

Fixed-base arm reaching with ULC-style smooth motion and curriculum.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_4_arm.py --num_envs 4096 --max_iterations 5000 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# ARGUMENT PARSING (BEFORE AppLauncher)
# =============================================================================

parser = argparse.ArgumentParser(description="G1 Arm Reach Training - Stage 4")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# LAUNCH APP
# =============================================================================

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# IMPORTS (AFTER AppLauncher)
# =============================================================================

import torch
import time
from datetime import datetime

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
# RSL-RL TRAINING CONFIG
# =============================================================================

@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for arm reaching with smooth motion."""

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
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = G1ArmReachEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = G1ArmReachPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"ulc_g1_stage4_arm_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Create runner with config dict
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    print("\n" + "=" * 70)
    print("         G1 ARM REACH TRAINING - STAGE 4 (ULC Smooth Motion)")
    print("=" * 70)
    print(f"  Environments:     {args.num_envs}")
    print(f"  Max iterations:   {args.max_iterations}")
    print(f"  Log directory:    {log_dir}")
    print("-" * 70)
    print("  ULC SMOOTH MOTION FEATURES:")
    print(f"    Action smoothing α: {env_cfg.action_smoothing_alpha}")
    print(f"    Action scale: {env_cfg.action_scale} rad")
    print(f"    Max joint velocity: {env_cfg.max_joint_vel} rad/s")
    print("-" * 70)
    print("  CURRICULUM:")
    print(f"    Initial radius: {env_cfg.initial_target_radius}m")
    print(f"    Final radius: {env_cfg.max_target_radius}m")
    print(f"    Curriculum steps: {env_cfg.curriculum_steps}")
    print("-" * 70)
    print("  REWARD STRUCTURE:")
    print(f"    Position distance: {env_cfg.reward_pos_distance}")
    print(f"    Orientation distance: {env_cfg.reward_ori_distance}")
    print(f"    Reaching bonus: {env_cfg.reward_reaching}")
    print(f"    Action rate penalty: {env_cfg.reward_action_rate}")
    print(f"    Smooth approach bonus: {env_cfg.reward_smooth_approach}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("=" * 70 + "\n")

    # Custom training loop with curriculum
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env._env

    # Reset environment
    obs_dict = env.get_observations()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    runner.alg.actor_critic.train()

    best_reward = float('-inf')
    start_time = time.time()

    for iteration in range(args.max_iterations):
        # =============================================
        # CURRICULUM UPDATE
        # =============================================
        if hasattr(unwrapped_env, 'update_curriculum'):
            unwrapped_env.update_curriculum(iteration)
            if iteration % 500 == 0 and iteration > 0:
                print(f"[Curriculum] Iteration {iteration}: target_radius = {unwrapped_env.current_target_radius:.3f}m")

        # Collect rollouts
        with torch.inference_mode():
            for _ in range(runner_cfg.num_steps_per_env):
                actions = runner.alg.act(obs, obs)
                obs_dict, rewards, dones, infos = env.step(actions)
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                runner.alg.process_env_step(rewards, dones, infos)

        # Update policy
        runner.alg.compute_returns(obs)
        mean_value_loss, mean_surrogate_loss = runner.alg.update()

        # Logging
        mean_reward = runner.alg.storage.rewards.mean().item()

        if iteration % 100 == 0:
            elapsed = time.time() - start_time
            fps = int((iteration + 1) * runner_cfg.num_steps_per_env * args.num_envs / elapsed)
            print(f"[Iter {iteration:5d}] Reward: {mean_reward:7.3f} | FPS: {fps:5d} | "
                  f"VLoss: {mean_value_loss:.4f} | SLoss: {mean_surrogate_loss:.4f}")

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            runner.save(os.path.join(log_dir, "model_best.pt"))

        # Periodic save
        if iteration % runner_cfg.save_interval == 0 and iteration > 0:
            runner.save(os.path.join(log_dir, f"model_{iteration}.pt"))

    # Final save
    runner.save(os.path.join(log_dir, f"model_{args.max_iterations}.pt"))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Logs saved to: {log_dir}")
    print(f"  Best model: {log_dir}/model_best.pt")
    print(f"  Best reward: {best_reward:.3f}")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()