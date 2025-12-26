# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modified by Turan for G1 Wave Demo - v4 FINAL
#
# Kullanım:
#   cd C:\IsaacLab
#   .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\play_wave.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 16 --load_run 2025-12-24_13-11-23 --wave_hand right

"""Play a trained policy with wave animation overlay."""

from __future__ import annotations

import argparse
import math
import torch
import os
import re

from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="Play trained G1 policy with wave animation")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--load_run", type=str, required=True, help="Run folder name (e.g., 2025-12-24_13-11-23)")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (default: latest)")
parser.add_argument("--wave_hand", type=str, default="right", choices=["left", "right", "both", "none"])
parser.add_argument("--wave_freq", type=float, default=2.0, help="Wave frequency in Hz")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.video = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def find_checkpoint(run_dir: str, checkpoint: str = None) -> str:
    """Find checkpoint file in run directory."""
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # List all model files
    model_files = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]

    if not model_files:
        raise FileNotFoundError(f"No model files found in: {run_dir}")

    if checkpoint:
        # Use specified checkpoint
        if checkpoint in model_files:
            return os.path.join(run_dir, checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found in {run_dir}")
    else:
        # Get latest checkpoint
        model_files.sort(key=lambda x: int(re.search(r'model_(\d+)', x).group(1)))
        latest = model_files[-1]
        return os.path.join(run_dir, latest)


def get_wave_override(time: float, wave_hand: str, wave_freq: float, amplitude: float = 0.5) -> dict:
    """Calculate wave animation joint overrides."""
    wave = amplitude * math.sin(2.0 * math.pi * wave_freq * time)

    overrides = {}

    # G1 arm indices (37 DoF action space)
    # 13-16: Left arm, 17-20: Right arm

    if wave_hand in ["right", "both"]:
        overrides[17] = -1.0  # right_shoulder_pitch - kaldır
        overrides[18] = -0.3 + wave * 0.5  # right_shoulder_roll - salla
        overrides[19] = -wave * 0.3  # right_shoulder_yaw
        overrides[20] = -0.8 - wave * 0.2  # right_elbow

    if wave_hand in ["left", "both"]:
        overrides[13] = -1.0  # left_shoulder_pitch
        overrides[14] = 0.3 + wave * 0.5  # left_shoulder_roll
        overrides[15] = wave * 0.3  # left_shoulder_yaw
        overrides[16] = -0.8 + wave * 0.2  # left_elbow

    return overrides


def main():
    """Main function."""

    print("\n" + "=" * 70)
    print("  G1 WALK + WAVE DEMO")
    print("  Policy-based locomotion with arm wave animation")
    print("=" * 70 + "\n")

    # Parse env config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Build checkpoint path MANUALLY (no duplicate)
    run_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
    resume_path = find_checkpoint(run_dir, args_cli.checkpoint)
    print(f"[INFO] Loading: {resume_path}")

    # Wrap for RSL-RL
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env)

    # Get agent config
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    agent_cfg = G1FlatPPORunnerCfg()

    # Create runner and load checkpoint
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(resume_path)

    # Get policy
    policy = runner.get_inference_policy(device=args_cli.device)

    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Wave hand: {args_cli.wave_hand}")
    print(f"[INFO] Wave freq: {args_cli.wave_freq} Hz")

    # Reset environment and get initial observation
    obs = env.get_observations()

    # Handle different return types from get_observations
    if isinstance(obs, tuple):
        obs = obs[0]  # First element is observations

    # Sim loop
    sim_time = 0.0
    step_count = 0
    dt = env.unwrapped.step_dt

    print("\n" + "-" * 50)
    print(" Running... Ctrl+C to exit")
    print(" Robots will WALK and WAVE!")
    print("-" * 50 + "\n")

    try:
        while simulation_app.is_running():
            # Policy inference
            with torch.no_grad():
                actions = policy(obs)

            # Apply wave override
            if args_cli.wave_hand != "none":
                overrides = get_wave_override(sim_time, args_cli.wave_hand, args_cli.wave_freq)
                for idx, val in overrides.items():
                    actions[:, idx] = val

            # Step environment
            step_result = env.step(actions)

            # Handle different return formats
            if isinstance(step_result, tuple):
                if len(step_result) == 4:
                    obs, _, _, _ = step_result
                elif len(step_result) == 5:
                    obs, _, _, _, _ = step_result
                else:
                    obs = step_result[0]
            else:
                obs = step_result

            # Handle obs if it's a tuple
            if isinstance(obs, tuple):
                obs = obs[0]

            sim_time += dt
            step_count += 1

            if step_count % 500 == 0:
                print(f"[Step {step_count}] Time: {sim_time:.2f}s")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()