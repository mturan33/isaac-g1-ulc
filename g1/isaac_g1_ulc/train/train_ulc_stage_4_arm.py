"""
G1 Arm Reach - Training Script (Stage 4)
=========================================

Fixed-base arm reaching - Position Only.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_4_arm.py --num_envs 2048 --max_iterations 5000 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description="G1 Arm Reach Training - Stage 4")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
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
# IMPORTS
# =============================================================================

import torch
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
    """PPO runner config for arm reaching."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "g1_arm_reach"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,           # Daha düşük başla
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,           # ÇOK DÜŞÜK - noise artışını engelle
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,           # Daha düşük LR
        schedule="fixed",             # FIXED schedule - adaptive değil!
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =============================================================================
# CURRICULUM WRAPPER
# =============================================================================

class CurriculumEnvWrapper(RslRlVecEnvWrapper):
    """Wrapper that updates curriculum and logs progress."""

    def __init__(self, env):
        super().__init__(env)
        self._iteration = 0
        self._step_count = 0
        self._unwrapped = env
        self._total_reaches = 0

        # TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        self._writer = None

    def set_writer(self, log_dir):
        """Set TensorBoard writer."""
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def step(self, actions):
        """Step with curriculum update and logging."""
        self._step_count += 1

        if self._step_count % 24 == 0:
            self._iteration += 1
            if hasattr(self._unwrapped, 'update_curriculum'):
                self._unwrapped.update_curriculum(self._iteration)

                # Log to TensorBoard
                if self._writer is not None:
                    self._writer.add_scalar(
                        'Curriculum/target_radius',
                        self._unwrapped.current_target_radius,
                        self._iteration
                    )
                    self._writer.add_scalar(
                        'Curriculum/progress',
                        self._unwrapped.curriculum_progress,
                        self._iteration
                    )

                    # Log reach count
                    avg_reaches = self._unwrapped.reach_count.mean().item()
                    self._writer.add_scalar(
                        'Curriculum/avg_reaches_per_env',
                        avg_reaches,
                        self._iteration
                    )

                if self._iteration % 200 == 0:
                    reach_rate = self._unwrapped.reach_count.mean().item()
                    print(f"[Curriculum] Iter {self._iteration}: "
                          f"radius={self._unwrapped.current_target_radius:.3f}m, "
                          f"progress={self._unwrapped.curriculum_progress:.1%}, "
                          f"avg_reaches={reach_rate:.1f}")

        return super().step(actions)


# =============================================================================
# MAIN
# =============================================================================

def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = G1ArmReachEnv(cfg=env_cfg)
    env = CurriculumEnvWrapper(env)

    runner_cfg = G1ArmReachPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"ulc_g1_stage4_arm_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Set TensorBoard writer for curriculum logging
    env.set_writer(log_dir)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    print("\n" + "=" * 70)
    print("    G1 ARM REACH TRAINING - STAGE 4 (Must Move to Reach!)")
    print("=" * 70)
    print(f"  Environments:     {args.num_envs}")
    print(f"  Max iterations:   {args.max_iterations}")
    print(f"  Log directory:    {log_dir}")
    print("-" * 70)
    print("  TASK: Targets spawn BEYOND threshold!")
    print(f"    Position threshold: {env_cfg.pos_threshold}m (5cm)")
    print(f"    Min spawn distance: 7cm (threshold + 2cm)")
    print(f"    Reaching bonus: +{env_cfg.reward_reaching} (sparse)")
    print("-" * 70)
    print("  CURRICULUM:")
    print(f"    Initial radius: {env_cfg.initial_target_radius}m → targets 7-10cm")
    print(f"    Final radius: {env_cfg.max_target_radius}m")
    print(f"    Steps: {env_cfg.curriculum_steps}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("=" * 70 + "\n")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Logs saved to: {log_dir}")
    print(f"  Final model: {log_dir}/model_{args.max_iterations}.pt")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()