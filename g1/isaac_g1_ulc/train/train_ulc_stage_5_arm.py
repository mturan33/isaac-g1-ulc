"""
G1 Arm Reach with Orientation - Stage 5 Training
=================================================

Sadece SAĞ KOL - Position + Orientation (Palm Down) reaching.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_5_arm.py --num_envs 2048 --max_iterations 8000 --headless

STAGE 4 CHECKPOINT İLE BAŞLA:
./isaaclab.bat -p source/isaaclab_tasks/.../train/train_ulc_stage_5_arm.py --num_envs 2048 --max_iterations 8000 --headless --stage4_checkpoint logs/ulc/ulc_g1_stage4_arm_XXXX/model_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description="G1 Arm Orient Training - Stage 5")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=8000, help="Max training iterations")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--stage4_checkpoint", type=str, default=None, help="Stage 4 checkpoint to initialize from")

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
import torch.nn as nn
from datetime import datetime

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_orient_env import G1ArmOrientEnv, G1ArmOrientEnvCfg
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
class G1ArmOrientPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for arm reaching with orientation."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 500
    experiment_name = "g1_arm_orient"
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
        entropy_coef=0.005,           # Biraz daha exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,           # Stage 4'ten biraz yüksek
        schedule="adaptive",
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
                        'Curriculum/workspace_radius',
                        self._unwrapped.current_workspace_radius,
                        self._iteration
                    )
                    self._writer.add_scalar(
                        'Curriculum/stage',
                        self._unwrapped.curriculum_stage,
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
                    stage = self._unwrapped.curriculum_stage + 1
                    radius = self._unwrapped.current_workspace_radius
                    print(f"[Curriculum] Iter {self._iteration} | "
                          f"Stage {stage}/8 | "
                          f"Radius={radius:.2f}m | "
                          f"Reaches={reach_rate:.1f}")

        return super().step(actions)


# =============================================================================
# CHECKPOINT LOADING UTILITIES
# =============================================================================

def load_stage4_weights(runner, checkpoint_path: str):
    """
    Load Stage 4 weights into Stage 5 network.

    Stage 4: obs=19 (position only), act=5
    Stage 5: obs=28 (position + orientation), act=5

    We can transfer the actor/critic hidden layers since action dim is same.
    Input layer will be reinitialized due to different obs dim.
    """
    print(f"\n[INFO] Loading Stage 4 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

    if 'model_state_dict' in checkpoint:
        stage4_state = checkpoint['model_state_dict']
    else:
        stage4_state = checkpoint

    # Get current model
    current_state = runner.alg.actor_critic.state_dict()

    # Track what we transfer
    transferred = []
    skipped = []

    for key in stage4_state.keys():
        if key in current_state:
            if stage4_state[key].shape == current_state[key].shape:
                current_state[key] = stage4_state[key]
                transferred.append(key)
            else:
                skipped.append(f"{key} (shape mismatch: {stage4_state[key].shape} vs {current_state[key].shape})")
        else:
            skipped.append(f"{key} (not in current model)")

    # Load transferred weights
    runner.alg.actor_critic.load_state_dict(current_state)

    print(f"[INFO] Transferred {len(transferred)} layers:")
    for key in transferred[:5]:
        print(f"  ✓ {key}")
    if len(transferred) > 5:
        print(f"  ... and {len(transferred) - 5} more")

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} layers (will be randomly initialized):")
        for key in skipped[:3]:
            print(f"  ✗ {key}")
        if len(skipped) > 3:
            print(f"  ... and {len(skipped) - 3} more")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    env_cfg = G1ArmOrientEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = G1ArmOrientEnv(cfg=env_cfg)
    env = CurriculumEnvWrapper(env)

    runner_cfg = G1ArmOrientPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"ulc_g1_stage5_arm_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Set TensorBoard writer
    env.set_writer(log_dir)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Load Stage 4 checkpoint if provided
    if args.stage4_checkpoint:
        load_stage4_weights(runner, args.stage4_checkpoint)

    # Resume from Stage 5 checkpoint if provided
    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    print("\n" + "=" * 70)
    print("    G1 ARM ORIENT TRAINING - STAGE 5 (Global Workspace + Palm Down)")
    print("=" * 70)
    print(f"  Environments:     {args.num_envs}")
    print(f"  Max iterations:   {args.max_iterations}")
    print(f"  Log directory:    {log_dir}")
    print("-" * 70)
    print("  TASK: Reach ANY target in workspace with PALM DOWN!")
    print(f"    Position threshold: {env_cfg.pos_threshold}m")
    print(f"    Orientation threshold: {env_cfg.ori_threshold:.2f} rad (~15°)")
    print(f"    Reaching bonus: +{env_cfg.reward_reaching}")
    print("-" * 70)
    print("  WORKSPACE (Omuz Merkezi Etrafında Yarım Küre):")
    print(f"    Inner radius (exclusion): {env_cfg.min_target_radius}m")
    print(f"    Max radius: {env_cfg.max_target_radius}m")
    print("-" * 70)
    print("  CURRICULUM (8 Seviye):")
    print(f"    L1: 15cm → L8: 45cm")
    print(f"    Total steps: {env_cfg.curriculum_steps}")
    if args.stage4_checkpoint:
        print(f"  Stage 4 init: {args.stage4_checkpoint}")
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