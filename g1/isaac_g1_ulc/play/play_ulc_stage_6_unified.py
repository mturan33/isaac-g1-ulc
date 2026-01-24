"""
G1 Reactive Balance - Play Script
==================================

Test the reactive balance system:
- Arm policy (FROZEN) reaches for targets
- Balance policy maintains stability

USAGE:
./isaaclab.bat -p .../play_reactive_balance.py \
    --balance_checkpoint logs/ulc/g1_reactive_balance_.../model_best.pt \
    --arm_checkpoint logs/ulc/g1_arm_reach_.../model_19998.pt \
    --num_envs 4
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description="G1 Reactive Balance Play")
parser.add_argument("--balance_checkpoint", type=str, required=True, help="Balance policy checkpoint")
parser.add_argument("--arm_checkpoint", type=str, required=True, help="Arm policy checkpoint (frozen)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs")
parser.add_argument("--steps", type=int, default=3000, help="Steps to run")
parser.add_argument("--mode", type=str, default="full",
                    choices=["standing", "walking", "squat", "full"])

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_reactive_balance_env import G1ReactiveBalanceEnv, G1ReactiveBalanceEnvCfg


# ============================================================================
# NETWORKS
# ============================================================================

class FrozenArmPolicy(nn.Module):
    def __init__(self, num_obs=29, num_act=5, hidden=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


class BalanceLocoPolicy(nn.Module):
    def __init__(self, num_obs=72, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        critic_layers = []
        prev = num_obs
        for h in hidden:
            critic_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)


def extract_arm_obs(full_obs: torch.Tensor) -> torch.Tensor:
    """Extract arm observations for frozen policy."""
    batch_size = full_obs.shape[0]
    device = full_obs.device

    base_state = full_obs[:, 0:9]
    arm_joint_pos = full_obs[:, 53:58]
    arm_joint_vel = full_obs[:, 58:63] * 10
    target_z = full_obs[:, 63:64]

    target_pos = torch.cat([
        torch.zeros(batch_size, 2, device=device),
        target_z
    ], dim=-1)

    ee_pos_approx = torch.zeros(batch_size, 3, device=device)
    pos_error = target_pos - ee_pos_approx
    pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5

    arm_obs = torch.cat([
        base_state, arm_joint_pos, arm_joint_vel,
        target_pos, ee_pos_approx, pos_error, pos_dist
    ], dim=-1)

    return arm_obs


MODE_CONFIGS = {
    "standing": {"vx_range": (0.0, 0.0), "pitch_range": (0.0, 0.0)},
    "walking": {"vx_range": (0.3, 0.5), "pitch_range": (0.0, 0.0)},
    "squat": {"vx_range": (0.0, 0.2), "pitch_range": (-0.3, -0.15)},
    "full": {"vx_range": (-0.2, 0.6), "pitch_range": (-0.35, 0.0)},
}


def main():
    device = "cuda:0"

    # Environment
    env_cfg = G1ReactiveBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = G1ReactiveBalanceEnv(cfg=env_cfg)

    # Load arm policy (frozen)
    print(f"\n[Load] Arm policy (FROZEN): {args.arm_checkpoint}")
    arm_policy = FrozenArmPolicy().to(device)
    arm_ckpt = torch.load(args.arm_checkpoint, map_location=device, weights_only=False)
    arm_state = arm_ckpt.get("model_state_dict", arm_ckpt)
    arm_actor = {k: v for k, v in arm_state.items() if k.startswith("actor.")}
    arm_policy.load_state_dict(arm_actor, strict=False)
    arm_policy.eval()

    # Load balance policy
    print(f"[Load] Balance policy: {args.balance_checkpoint}")
    balance_policy = BalanceLocoPolicy().to(device)
    balance_ckpt = torch.load(args.balance_checkpoint, map_location=device, weights_only=False)
    balance_policy.load_state_dict(balance_ckpt["model_state_dict"], strict=False)
    balance_policy.eval()

    mode_cfg = MODE_CONFIGS[args.mode]

    print(f"\n{'='*60}")
    print("G1 REACTIVE BALANCE PLAY")
    print(f"{'='*60}")
    print(f"  Mode: {args.mode}")
    if balance_ckpt.get("curriculum_level"):
        print(f"  Trained level: {balance_ckpt['curriculum_level'] + 1}")
    print(f"{'='*60}\n")

    def set_commands():
        n = args.num_envs
        vx = mode_cfg["vx_range"]
        env.vel_cmd[:, 0] = torch.rand(n, device=device) * (vx[1] - vx[0]) + vx[0]

        pitch = mode_cfg["pitch_range"]
        env.torso_cmd[:, 1] = torch.rand(n, device=device) * (pitch[1] - pitch[0]) + pitch[0]

    obs, _ = env.reset()
    set_commands()

    total_reward = 0.0
    total_reaches = 0

    with torch.no_grad():
        for step in range(args.steps):
            obs_tensor = obs["policy"]

            # Arm actions from frozen policy
            arm_obs = extract_arm_obs(obs_tensor)
            arm_actions = arm_policy(arm_obs)
            env.set_frozen_arm_actions(arm_actions)

            # Leg actions from balance policy
            leg_actions, _ = balance_policy.forward(obs_tensor)

            obs, reward, terminated, truncated, info = env.step(leg_actions)
            total_reward += reward.mean().item()

            new_reaches = env.total_reaches - total_reaches
            if new_reaches > 0:
                total_reaches = env.total_reaches
                print(f"[Step {step:4d}] 🎯 REACH #{total_reaches}!")

            if step > 0 and step % 500 == 0:
                set_commands()

            if step > 0 and step % 200 == 0:
                height = env.extras.get("M/height", 0)
                com_x = env.extras.get("M/com_x", 0)
                com_y = env.extras.get("M/com_y", 0)
                pitch = env.extras.get("M/pitch", 0)
                arm_dist = env.extras.get("M/arm_dist", 0)

                print(f"[Step {step:4d}] H={height:.3f}m | "
                      f"CoM=({com_x:.2f},{com_y:.2f}) | "
                      f"Pitch={pitch:.2f} | "
                      f"ArmDist={arm_dist:.3f}m | "
                      f"Reaches={total_reaches}")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"  Total reaches: {total_reaches}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()