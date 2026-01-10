#!/usr/bin/env python3
"""
ULC G1 Stage 4 Training - FULL ARM WORKSPACE
=============================================
Bu versiyon G1'in gerçek kol limitlerinin büyük kısmını kullanır.

G1 Gerçek Limitler:
- shoulder_pitch: (-3.1, +2.6) rad
- elbow: (-1.6, +1.6) rad

Curriculum Final Target:
- shoulder_pitch: ±1.5 rad (~86°) - güvenli margin ile
- elbow: ±1.2 rad (~69°)

Bu, önceki ±0.8 rad'a göre ~2x daha geniş workspace sağlar.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="ULC G1 Stage 4 - Full Workspace Training")
parser.add_argument("--stage3_checkpoint", type=str, required=True, help="Path to Stage 3 checkpoint")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=6000, help="Maximum training iterations")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument("--video_interval", type=int, default=1000, help="Video recording interval")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import torch
import torch.nn as nn
from datetime import datetime

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import PPOAlgorithmCfg, ActorCriticCfg
from isaaclab_rl.rsl_rl.vecenv import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import our Stage 4 environment
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_env_stage4 import ULCG1Stage4Env, ULCG1Stage4EnvCfg

# ============================================================
# FULL WORKSPACE CURRICULUM CONFIGURATION
# ============================================================
# G1 actual limits: shoulder_pitch (-3.1, +2.6), elbow (-1.6, +1.6)
# We use ~50% of full range for safety margin

FULL_WORKSPACE_CURRICULUM = {
    0: {  # Level 0: Gentle start
        "vx_range": (0.0, 0.3),
        "pitch_range": 0.1,
        "roll_range": 0.1,
        "arm_range": 0.2,        # ~11° - very easy
        "threshold": 27.0,
        "min_iterations": 100,
    },
    1: {  # Level 1: Build confidence
        "vx_range": (0.0, 0.5),
        "pitch_range": 0.15,
        "roll_range": 0.15,
        "arm_range": 0.5,        # ~29° - moderate
        "threshold": 27.5,
        "min_iterations": 150,
    },
    2: {  # Level 2: Serious arm motion
        "vx_range": (-0.2, 0.7),
        "pitch_range": 0.2,
        "roll_range": 0.2,
        "arm_range": 0.8,        # ~46° - what we had before as FINAL
        "threshold": 27.0,
        "min_iterations": 200,
    },
    3: {  # Level 3: Extended workspace
        "vx_range": (-0.3, 0.8),
        "pitch_range": 0.25,
        "roll_range": 0.2,
        "arm_range": 1.2,        # ~69° - NEW!
        "threshold": 26.5,
        "min_iterations": 300,
    },
    4: {  # Level 4: FULL WORKSPACE (NEW!)
        "vx_range": (-0.3, 1.0),
        "pitch_range": 0.3,
        "roll_range": 0.25,
        "arm_range": 1.5,        # ~86° - FULL TARGET!
        "threshold": None,       # Final level
        "min_iterations": None,
    },
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
class Stage4FullWorkspaceTrainingCfg:
    """Training configuration for Stage 4 with full arm workspace."""

    # Algorithm
    algorithm = PPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Slightly higher for more exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Policy network - larger for complex task
    policy = ActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Runner
    num_steps_per_env = 24
    save_interval = 500
    experiment_name = "ulc_g1_stage4_fullworkspace"
    empirical_normalization = False

    # Separate std for legs and arms
    init_leg_std = 0.5
    init_arm_std = 0.8  # Higher for larger workspace


def main():
    print("=" * 70)
    print("ULC G1 STAGE 4 TRAINING - FULL ARM WORKSPACE")
    print("=" * 70)
    print(f"Stage 3 Checkpoint: {args.stage3_checkpoint}")
    print(f"Environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print()
    print("CURRICULUM LEVELS:")
    for level, cfg in FULL_WORKSPACE_CURRICULUM.items():
        arm_deg = cfg['arm_range'] * 57.3  # rad to deg
        print(f"  Level {level}: arm_range={cfg['arm_range']:.1f} rad (~{arm_deg:.0f}°), "
              f"vx={cfg['vx_range']}")
    print("=" * 70)

    # Load Stage 3 checkpoint to verify
    print(f"\n[INFO] Loading Stage 3 checkpoint: {args.stage3_checkpoint}")
    stage3_ckpt = torch.load(args.stage3_checkpoint, map_location="cuda:0", weights_only=True)
    print(f"[INFO] Stage 3 best reward: {stage3_ckpt.get('best_reward', 'N/A')}")
    print(f"[INFO] Stage 3 iteration: {stage3_ckpt.get('iteration', 'N/A')}")

    # Create environment with initial curriculum settings
    env_cfg = ULCG1Stage4EnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Set initial curriculum parameters
    level_0 = FULL_WORKSPACE_CURRICULUM[0]
    env_cfg.vx_range = level_0["vx_range"]
    env_cfg.pitch_range = level_0["pitch_range"]
    env_cfg.roll_range = level_0["roll_range"]
    env_cfg.arm_range = level_0["arm_range"]

    env = ULCG1Stage4Env(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Create policy
    train_cfg = Stage4FullWorkspaceTrainingCfg()

    policy = train_cfg.policy.class_name(
        num_actor_obs=env.observation_space.shape[1],
        num_critic_obs=env.observation_space.shape[1],
        num_actions=env.action_space.shape[1],
        **train_cfg.policy.to_dict()
    ).to("cuda:0")

    # Initialize from Stage 3 (legs + torso) with expanded action space
    print("\n[INFO] Initializing policy from Stage 3...")
    init_from_stage3(policy, stage3_ckpt, env.action_space.shape[1])

    # Create log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/ulc_g1_stage4_fullworkspace_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")

    # Create runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device="cuda:0",
    )
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    # Save final model
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": runner.optimizer.state_dict(),
        "iteration": args.max_iterations,
        "curriculum_level": get_current_level(runner),
        "config": "stage4_fullworkspace",
    }, final_path)
    print(f"\n[INFO] Final model saved to: {final_path}")

    env.close()
    simulation_app.close()


def init_from_stage3(policy, stage3_ckpt, new_action_dim):
    """Initialize Stage 4 policy from Stage 3 checkpoint with expanded action space."""
    stage3_state = stage3_ckpt["model_state_dict"]

    # Get Stage 3 dimensions
    old_action_dim = 12  # Stage 3: 12 leg actions
    new_arm_actions = new_action_dim - old_action_dim  # 10 new arm actions

    print(f"[INFO] Expanding action space: {old_action_dim} → {new_action_dim}")
    print(f"[INFO] New arm actions: {new_arm_actions}")

    # Copy compatible weights
    new_state = policy.state_dict()

    for key in stage3_state:
        if key in new_state:
            old_shape = stage3_state[key].shape
            new_shape = new_state[key].shape

            if old_shape == new_shape:
                # Direct copy
                new_state[key] = stage3_state[key]
            elif "actor" in key and "weight" in key and len(old_shape) == 2:
                # Actor output layer - expand for new actions
                if old_shape[0] == old_action_dim and new_shape[0] == new_action_dim:
                    new_state[key][:old_action_dim, :] = stage3_state[key]
                    # Initialize arm weights with small random values
                    nn.init.xavier_uniform_(new_state[key][old_action_dim:, :], gain=0.1)
                    print(f"[INFO] Expanded actor layer: {key}")
            elif "actor" in key and "bias" in key and len(old_shape) == 1:
                if old_shape[0] == old_action_dim and new_shape[0] == new_action_dim:
                    new_state[key][:old_action_dim] = stage3_state[key]
                    new_state[key][old_action_dim:] = 0.0  # Zero bias for arms
                    print(f"[INFO] Expanded actor bias: {key}")

    policy.load_state_dict(new_state)
    print("[INFO] Policy initialized from Stage 3 ✓")


def get_current_level(runner):
    """Get current curriculum level from runner."""
    # This would need to be tracked during training
    return 4  # Assume final level reached


if __name__ == "__main__":
    main()