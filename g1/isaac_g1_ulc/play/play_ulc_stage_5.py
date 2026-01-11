#!/usr/bin/env python3
"""
ULC G1 Stage 5 - Play Script
============================

Eğitilmiş Stage 5 modelini görsel olarak test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/play_ulc_stage_5.py ^
    --checkpoint logs/ulc/ulc_g1_stage5_.../model_best.pt ^
    --num_envs 4

PARAMETRELER:
--random_commands : Her 5 saniyede yeni random komutlar
--demo_mode      : Önceden tanımlı hareketler göster
--full_workspace : Maksimum açılar ile test et
"""

import argparse
import time

parser = argparse.ArgumentParser(description="ULC G1 Stage 5 Play")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")

# Command options
parser.add_argument("--random_commands", action="store_true", help="Randomize commands every 5 seconds")
parser.add_argument("--demo_mode", action="store_true", help="Show predefined demo motions")
parser.add_argument("--full_workspace", action="store_true", help="Test with full mechanical limits")

# Manual command overrides
parser.add_argument("--target_height", type=float, default=0.75, help="Target height (0.35-0.85)")
parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity")
parser.add_argument("--arm_range", type=float, default=1.5, help="Arm movement range")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym
import math

from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg


class ActorCritic(nn.Module):
    """Same architecture as training."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        actor_layers = []
        prev = obs_dim
        for h in hidden_dims:
            actor_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()])
            prev = h
        actor_layers.append(nn.Linear(prev, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        prev = obs_dim
        for h in hidden_dims:
            critic_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def act(self, obs, deterministic=True):
        mean = self.actor(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std).sample()


# ============================================================
# DEMO SEQUENCES
# ============================================================

DEMO_SEQUENCES = {
    "wave": [
        {"duration": 2.0, "left_arm": [0.0, 0.0], "right_arm": [1.5, 0.5], "height": 0.75},
        {"duration": 0.5, "left_arm": [0.0, 0.0], "right_arm": [1.5, -0.3], "height": 0.75},
        {"duration": 0.5, "left_arm": [0.0, 0.0], "right_arm": [1.5, 0.5], "height": 0.75},
        {"duration": 0.5, "left_arm": [0.0, 0.0], "right_arm": [1.5, -0.3], "height": 0.75},
        {"duration": 2.0, "left_arm": [0.0, 0.0], "right_arm": [0.0, 0.0], "height": 0.75},
    ],
    "squat": [
        {"duration": 2.0, "left_arm": [0.0, 0.0], "right_arm": [0.0, 0.0], "height": 0.75},
        {"duration": 2.0, "left_arm": [1.0, 0.0], "right_arm": [1.0, 0.0], "height": 0.45},
        {"duration": 2.0, "left_arm": [0.0, 0.0], "right_arm": [0.0, 0.0], "height": 0.75},
    ],
    "arms_up": [
        {"duration": 2.0, "left_arm": [0.0, 0.0], "right_arm": [0.0, 0.0], "height": 0.75},
        {"duration": 2.0, "left_arm": [2.0, 0.0], "right_arm": [2.0, 0.0], "height": 0.75},
        {"duration": 2.0, "left_arm": [-2.0, 0.0], "right_arm": [-2.0, 0.0], "height": 0.75},
        {"duration": 2.0, "left_arm": [0.0, 0.0], "right_arm": [0.0, 0.0], "height": 0.75},
    ],
    "full_workspace": [
        {"duration": 3.0, "left_arm": [2.6, 1.5], "right_arm": [-2.6, -1.5], "height": 0.85},
        {"duration": 3.0, "left_arm": [-2.6, -1.5], "right_arm": [2.6, 1.5], "height": 0.35},
        {"duration": 3.0, "left_arm": [2.6, 1.5], "right_arm": [2.6, 1.5], "height": 0.60},
        {"duration": 3.0, "left_arm": [0.0, 0.0], "right_arm": [0.0, 0.0], "height": 0.75},
    ],
}


def set_commands(env, height=0.75, vx=0.5, left_arm=(0, 0), right_arm=(0, 0), torso=(0, 0, 0)):
    """Set commands for all environments."""
    n = env.num_envs
    d = env.device

    # 1D tensors - ESKİ: env.height_command[:, 0] = height
    env.height_command[:] = height
    env.velocity_commands[:, 0] = vx
    env.velocity_commands[:, 1] = 0.0
    env.velocity_commands[:, 2] = 0.0

    env.torso_commands[:, 0] = torso[0]
    env.torso_commands[:, 1] = torso[1]
    env.torso_commands[:, 2] = torso[2]

    # Arms - 10 joints (5 per arm)
    env.arm_commands[:, 0] = left_arm[0]  # L shoulder_pitch
    env.arm_commands[:, 3] = left_arm[1]  # L elbow_pitch
    env.arm_commands[:, 5] = right_arm[0]  # R shoulder_pitch
    env.arm_commands[:, 8] = right_arm[1]  # R elbow_pitch


def random_commands(env, arm_range=2.6):
    """Generate random commands."""
    n = env.num_envs
    d = env.device

    env.height_command[:] = torch.empty(n, device=d).uniform_(0.35, 0.85)
    env.velocity_commands[:, 0] = torch.empty(n, device=d).uniform_(-0.8, 1.2)
    env.velocity_commands[:, 1] = torch.empty(n, device=d).uniform_(-0.4, 0.4)
    env.velocity_commands[:, 2] = torch.empty(n, device=d).uniform_(-0.8, 0.8)

    env.torso_commands[:, 0] = torch.empty(n, device=d).uniform_(-0.5, 0.5)
    env.torso_commands[:, 1] = torch.empty(n, device=d).uniform_(-0.5, 0.5)
    env.torso_commands[:, 2] = torch.empty(n, device=d).uniform_(-0.5, 0.5)

    # Arms - 10 joints
    env.arm_commands[:, 0] = torch.empty(n, device=d).uniform_(-arm_range, arm_range)
    env.arm_commands[:, 3] = torch.empty(n, device=d).uniform_(-arm_range * 0.6, arm_range * 0.6)
    env.arm_commands[:, 5] = torch.empty(n, device=d).uniform_(-arm_range, arm_range)
    env.arm_commands[:, 8] = torch.empty(n, device=d).uniform_(-arm_range * 0.6, arm_range * 0.6)


def main():
    print("=" * 60)
    print("🤖 ULC G1 STAGE 5 - PLAY MODE")
    print("=" * 60)

    # Load checkpoint
    print(f"\n[INFO] Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    print(f"[INFO] Best reward: {ckpt.get('best_reward', 'N/A')}")
    print(f"[INFO] Curriculum level: {ckpt.get('curriculum_level', 'N/A')}")
    print(f"[INFO] Workspace: {ckpt.get('workspace_pct', 'N/A')}%")

    # Get checkpoint dimensions
    state_dict = ckpt.get("actor_critic", ckpt)
    obs_dim = state_dict["actor.0.weight"].shape[1]
    act_dim = state_dict["log_std"].shape[0]
    print(f"[INFO] Model dims: obs={obs_dim}, act={act_dim}")

    # Create environment
    cfg = ULC_G1_Stage4_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.termination["base_height_min"] = 0.25
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(cfg.num_observations,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_actions,))

    env = ULC_G1_Env(cfg=cfg)
    env.current_stage = 4

    # Load policy
    policy = ActorCritic(obs_dim, act_dim).to("cuda:0")
    policy.load_state_dict(state_dict)
    policy.eval()
    print("[INFO] ✓ Model loaded!")

    # Initial commands
    if args.full_workspace:
        arm_range = 2.6
        print("[MODE] Full Workspace Test")
    else:
        arm_range = args.arm_range

    if args.random_commands:
        random_commands(env, arm_range)
        print("[MODE] Random Commands (every 5 sec)")
    elif args.demo_mode:
        print("[MODE] Demo Mode")
    else:
        set_commands(env, height=args.target_height, vx=args.vx)
        print(f"[MODE] Fixed Commands: height={args.target_height}, vx={args.vx}")

    # Run simulation
    print("\n[PLAY] Starting simulation... Press Ctrl+C to stop")
    print("-" * 60)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    step = 0
    last_random_time = time.time()
    demo_start_time = time.time()
    demo_idx = 0
    demo_seq = DEMO_SEQUENCES.get("full_workspace" if args.full_workspace else "wave", [])

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                action = policy.act(obs, deterministic=True)

            obs_dict, reward, term, trunc, _ = env.step(action)
            obs = obs_dict["policy"]
            step += 1

            # Handle resets
            done = term | trunc
            if done.any():
                if args.random_commands:
                    random_commands(env, arm_range)

            # Random command refresh
            if args.random_commands and time.time() - last_random_time > 5.0:
                random_commands(env, arm_range)
                last_random_time = time.time()
                print("[CMD] New random commands!")

            # Demo mode
            if args.demo_mode and len(demo_seq) > 0:
                elapsed = time.time() - demo_start_time

                # Find current demo step
                total_time = 0
                for i, step_cfg in enumerate(demo_seq):
                    total_time += step_cfg["duration"]
                    if elapsed < total_time:
                        if i != demo_idx:
                            demo_idx = i
                            print(f"[DEMO] Step {i + 1}/{len(demo_seq)}")
                        set_commands(
                            env,
                            height=step_cfg["height"],
                            vx=args.vx,
                            left_arm=step_cfg["left_arm"],
                            right_arm=step_cfg["right_arm"],
                        )
                        break
                else:
                    # Loop demo
                    demo_start_time = time.time()
                    demo_idx = 0

            # Status print
            if step % 100 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vel_x = env.robot.data.root_lin_vel_w[:, 0].mean().item()
                cmd_h = env.height_command[0].item()

                print(f"Step {step:5d} | H={height:.3f}m (cmd={cmd_h:.2f}) | "
                      f"Vx={vel_x:.2f}m/s | R={reward.mean():.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")

    env.close()
    simulation_app.close()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()