#!/usr/bin/env python3
"""
ULC G1 Stage 6 - Play Script
============================

Stage 6 modelini test et - düzeltilmiş arm tracking ile.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_6.py ^
    --checkpoint logs/ulc/ulc_g1_stage6_.../model_best.pt ^
    --num_envs 4 --arms_forward
"""

import argparse
import time

parser = argparse.ArgumentParser(description="ULC G1 Stage 6 Play")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--arms_forward", action="store_true", help="Test arms forward")
parser.add_argument("--random_commands", action="store_true")
parser.add_argument("--target_height", type=float, default=0.72)
parser.add_argument("--smoothing", type=float, default=0.5, help="Action smoothing (0-0.9)")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym

from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_Stage4_EnvCfg


class ActorCritic(nn.Module):
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
        return mean if deterministic else torch.distributions.Normal(mean, torch.exp(self.log_std)).sample()


class Stage6PlayEnv(ULC_G1_Env):
    """Play environment with Stage 6 residual scales."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Stage 6 residual scales (larger for better arm tracking)
        self.residual_scales = torch.tensor(
            [1.5, 1.0, 1.0, 1.2, 0.8,
             1.5, 1.0, 1.0, 1.2, 0.8],
            device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        """Use Stage 6 arm target calculation."""
        self.actions = torch.clamp(actions, -1.0, 1.0)

        leg_actions = self.actions[:, :12]
        arm_actions = self.actions[:, 12:]

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_joint_indices] = self.default_leg + leg_actions * 0.4

        # Stage 6: Direct scaling (no double tanh)
        arm_cmd = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)
        arm_residual = arm_actions * self.residual_scales
        arm_target = torch.clamp(arm_cmd + arm_residual, -2.6, 2.6)

        target_pos[:, self.arm_joint_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)
        self.gait_phase = (self.gait_phase + self.gait_frequency * self.step_dt) % 1.0
        self._prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()


def set_commands(env, height=0.72, vx=0.0, left_arm=None, right_arm=None):
    """Set commands."""
    if left_arm is None:
        left_arm = [0.0] * 5
    if right_arm is None:
        right_arm = [0.0] * 5

    env.height_command[:] = height
    env.velocity_commands[:, 0] = vx
    env.velocity_commands[:, 1] = 0.0
    env.velocity_commands[:, 2] = 0.0
    env.torso_commands[:] = 0.0

    for i in range(5):
        env.left_arm_cmd[:, i] = left_arm[i]
        env.right_arm_cmd[:, i] = right_arm[i]

    env.arm_commands = torch.cat([env.left_arm_cmd, env.right_arm_cmd], dim=-1)


def main():
    print("=" * 60)
    print("🤖 ULC G1 STAGE 6 - PLAY MODE")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
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

    env = Stage6PlayEnv(cfg=cfg)
    env.current_stage = 4

    # Load policy
    policy = ActorCritic(obs_dim, act_dim).to("cuda:0")
    policy.load_state_dict(state_dict)
    policy.eval()
    print("[INFO] ✓ Model loaded!")

    # Set initial commands
    if args.arms_forward:
        print("[MODE] Arms Forward Test")
        set_commands(env, height=0.72, vx=0.0,
                     left_arm=[1.5, 0.0, 0.0, 0.0, 0.0],
                     right_arm=[1.5, 0.0, 0.0, 0.0, 0.0])
    else:
        set_commands(env, height=args.target_height, vx=0.0)

    print(f"[INFO] Smoothing: {args.smoothing}")
    print("\n[PLAY] Starting... Press Ctrl+C to stop\n")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    prev_action = torch.zeros(env.num_envs, act_dim, device="cuda:0")
    step = 0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                raw_action = policy.act(obs, deterministic=True)
                action = args.smoothing * prev_action + (1 - args.smoothing) * raw_action
                prev_action = action.clone()

            obs_dict, reward, term, trunc, _ = env.step(action)
            obs = obs_dict["policy"]
            step += 1

            done = term | trunc
            if done.any():
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                prev_action[reset_ids] = 0.0

            if step % 100 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vel_z = env.robot.data.root_lin_vel_w[:, 2].mean().item()
                left_arm = env.robot.data.joint_pos[:, env.left_arm_indices[0]].mean().item()
                right_arm = env.robot.data.joint_pos[:, env.right_arm_indices[0]].mean().item()
                cmd_left = env.left_arm_cmd[0, 0].item()

                print(f"Step {step:5d} | H={height:.3f}m | Vz={vel_z:.3f} | "
                      f"L_arm={left_arm:.2f} (cmd={cmd_left:.2f}) | R_arm={right_arm:.2f} | R={reward.mean():.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()