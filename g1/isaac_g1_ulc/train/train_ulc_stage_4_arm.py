"""
G1 Arm Reach - Environment Registration & Training Config
==========================================================

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p scripts/rsl_rl/train.py --task Isaac-G1-Arm-Reach-v0 --num_envs 1024 --headless

PLAY:
./isaaclab.bat -p scripts/rsl_rl/play.py --task Isaac-G1-Arm-Reach-v0 --num_envs 4
"""

import gymnasium as gym

from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

# Import the environment class
from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg

# =============================================================================
# ENVIRONMENT REGISTRATION
# =============================================================================

gym.register(
    id="Isaac-G1-Arm-Reach-v0",
    entry_point="isaaclab.envs:DirectRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1ArmReachEnvCfg,
        "rl_games_cfg_entry_point": None,
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": None,
        "sb3_cfg_entry_point": None,
    },
)

# =============================================================================
# RSL-RL TRAINING CONFIG
# =============================================================================

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg
from isaaclab.utils import configclass


@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for arm reaching."""

    num_steps_per_env = 24
    max_iterations = 5000  # ~10-15 minutes
    save_interval = 500
    experiment_name = "g1_arm_reach"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 64],  # Smaller network for simple task
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Lower entropy for more exploitation
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
# STANDALONE TRAINING SCRIPT
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    import torch

    from isaaclab.app import AppLauncher

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import after app launch
    from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner
    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

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

    # Setup logging directory
    log_root = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name)
    os.makedirs(log_root, exist_ok=True)

    # Create runner
    runner = RslRlOnPolicyRunner(env, runner_cfg, log_dir=log_root, device="cuda:0")

    # Resume if specified
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    # Train
    print("\n" + "=" * 60)
    print("STARTING G1 ARM REACH TRAINING")
    print("=" * 60)
    print(f"  Environments: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Log directory: {log_root}")
    print("=" * 60 + "\n")

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # Cleanup
    env.close()
    simulation_app.close()