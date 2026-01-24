"""
G1 + Dex1 Stage 6 Training Script
==================================

Trains the G1 humanoid with Dex1 gripper for loco-manipulation tasks.

Usage:
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\g1_dex1_stage6\train_stage6.py --num_envs 4096

Author: Turan (VLM-RL Project)
Date: January 2026
"""

import argparse
import os
import sys
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaacsim import SimulationApp

# Parse arguments before SimulationApp
parser = argparse.ArgumentParser(description="Train G1 Dex1 Stage 6")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--max_iterations", type=int, default=10000, help="Max training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
parser.add_argument("--curriculum_level", type=int, default=0, help="Starting curriculum level")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--log_dir", type=str, default="logs/g1_dex1_stage6", help="Log directory")
args, unknown = parser.parse_known_args()

# Launch simulation
simulation_app = SimulationApp({"headless": args.headless})

import torch
import numpy as np
from g1_dex1_stage6_env import G1Dex1Stage6Env, G1Dex1Stage6EnvCfg

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


class PPORunnerConfig:
    """PPO Runner configuration."""
    seed: int = 42
    device: str = "cuda"
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 500
    experiment_name: str = "g1_dex1_stage6"
    run_name: str = ""
    log_dir: str = "logs"
    resume: bool = False
    load_run: str = ""
    checkpoint: int = -1

    # Learning
    learning_rate: float = 3e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4

    # PPO
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True

    # Network
    policy_class_name: str = "ActorCritic"
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    activation: str = "elu"

    def __init__(self):
        self.actor_hidden_dims = [512, 256, 128]
        self.critic_hidden_dims = [512, 256, 128]


def train():
    """Main training function."""

    # Create log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, f"stage6_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"G1 + Dex1 Stage 6 Training")
    print(f"{'='*60}")
    print(f"Log directory: {log_dir}")
    print(f"Environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Curriculum level: {args.curriculum_level}")
    print(f"{'='*60}\n")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment config
    env_cfg = G1Dex1Stage6EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.curriculum_level = args.curriculum_level

    # Create environment
    env = G1Dex1Stage6Env(env_cfg)

    # Runner config
    runner_cfg = PPORunnerConfig()
    runner_cfg.seed = args.seed
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.log_dir = log_dir
    runner_cfg.experiment_name = f"g1_dex1_stage6_level{args.curriculum_level}"
    runner_cfg.run_name = timestamp

    # Create actor-critic network
    actor_critic = ActorCritic(
        num_obs=env_cfg.num_observations,
        num_actions=env_cfg.num_actions,
        actor_hidden_dims=runner_cfg.actor_hidden_dims,
        critic_hidden_dims=runner_cfg.critic_hidden_dims,
        activation=runner_cfg.activation,
        init_noise_std=1.0,
    ).to("cuda")

    # Create PPO algorithm
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=runner_cfg.num_learning_epochs,
        num_mini_batches=runner_cfg.num_mini_batches,
        clip_param=runner_cfg.clip_param,
        gamma=runner_cfg.gamma,
        lam=runner_cfg.lam,
        value_loss_coef=runner_cfg.value_loss_coef,
        entropy_coef=runner_cfg.entropy_coef,
        learning_rate=runner_cfg.learning_rate,
        max_grad_norm=runner_cfg.max_grad_norm,
        use_clipped_value_loss=runner_cfg.use_clipped_value_loss,
        device="cuda",
    )

    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Create runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=runner_cfg,
        log_dir=log_dir,
        device="cuda",
    )

    # Training loop
    print("\nStarting training...")

    try:
        obs, _ = env.reset()
        obs = obs["policy"]

        for iteration in range(args.max_iterations):
            # Collect rollouts
            for step in range(runner_cfg.num_steps_per_env):
                # Get actions
                with torch.no_grad():
                    actions = actor_critic.act(obs)

                # Step environment
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                next_obs = next_obs["policy"]
                dones = terminated | truncated

                # Store transition
                runner.storage.add_transitions(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    values=actor_critic.evaluate(obs),
                    log_probs=actor_critic.get_actions_log_prob(actions),
                )

                obs = next_obs

                # Reset done environments
                if dones.any():
                    obs[dones] = env.reset()[0]["policy"][dones]

            # Compute returns
            with torch.no_grad():
                last_values = actor_critic.evaluate(obs)
            runner.storage.compute_returns(last_values, runner_cfg.gamma, runner_cfg.lam)

            # Update policy
            mean_loss = ppo.update(runner.storage)
            runner.storage.clear()

            # Log progress
            if iteration % 10 == 0:
                mean_reward = rewards.mean().item()
                print(f"Iter {iteration}/{args.max_iterations} | "
                      f"Reward: {mean_reward:.3f} | "
                      f"Loss: {mean_loss:.4f}")

            # Save checkpoint
            if iteration % runner_cfg.save_interval == 0 and iteration > 0:
                save_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": actor_critic.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                    "curriculum_level": env._curriculum_level,
                }, save_path)
                print(f"Saved checkpoint: {save_path}")

            # Check curriculum advancement
            if env.advance_curriculum():
                # Save curriculum advancement checkpoint
                save_path = os.path.join(log_dir, f"model_curriculum_{env._curriculum_level}.pt")
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": actor_critic.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                    "curriculum_level": env._curriculum_level,
                }, save_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Save final model
        final_path = os.path.join(log_dir, "model_final.pt")
        torch.save({
            "iteration": iteration if 'iteration' in dir() else 0,
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "curriculum_level": env._curriculum_level,
        }, final_path)
        print(f"Saved final model: {final_path}")

        env.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    train()
    simulation_app.close()