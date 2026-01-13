"""
G1 Arm Reach - Training Script (Stage 4)
=========================================

Fixed-base arm reaching with orientation.

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
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")

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
from datetime import datetime

# Add env path to import our environment
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner
from isaaclab.utils import configclass


# =============================================================================
# RSL-RL TRAINING CONFIG
# =============================================================================

@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for arm reaching."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "g1_arm_reach"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],  # Bigger for orientation
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create environment config
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create environment
    env = G1ArmReachEnv(cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner config
    runner_cfg = G1ArmReachPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations

    # Setup logging directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"ulc_g1_stage4_arm_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Create runner
    runner = RslRlOnPolicyRunner(env, runner_cfg, log_dir=log_dir, device="cuda:0")

    # Resume if specified
    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    # Print info
    print("\n" + "=" * 70)
    print("         G1 ARM REACH TRAINING - STAGE 4")
    print("=" * 70)
    print(f"  Environments:     {args.num_envs}")
    print(f"  Max iterations:   {args.max_iterations}")
    print(f"  Log directory:    {log_dir}")
    print(f"  Observations:     {env_cfg.num_observations} (with orientation)")
    print(f"  Actions:          {env_cfg.num_actions}")
    print(f"  Episode length:   {env_cfg.episode_length_s}s")
    print(f"  Position thresh:  {env_cfg.pos_threshold}m")
    print(f"  Orient thresh:    {env_cfg.ori_threshold} rad (~15°)")
    if args.resume:
        print(f"  Resume from:      {args.resume}")
    print("=" * 70 + "\n")

    # Train
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Logs saved to: {log_dir}")
    print(f"  Best model: {log_dir}/model_best.pt")
    print("=" * 70 + "\n")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()