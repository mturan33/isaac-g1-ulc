#!/usr/bin/env python3
"""
ULC G1 Stage 6 - STABLE Play Script
====================================

Yüksek smoothing ve arm action limiting ile sallanmayı azalt.

KULLANIM:
./isaaclab.bat -p .../play_ulc_stage_6_stable.py ^
    --checkpoint logs/ulc/ulc_g1_stage6_.../model_best.pt ^
    --num_envs 4 --arms_forward
"""

import argparse

parser = argparse.ArgumentParser(description="ULC G1 Stage 6 Play - STABLE")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--arms_forward", action="store_true", help="Test arms forward (1.5 rad)")
parser.add_argument("--arms_neutral", action="store_true", help="Arms at neutral position")
parser.add_argument("--target_height", type=float, default=0.72)
parser.add_argument("--smoothing", type=float, default=0.8, help="Action smoothing (higher=more stable)")
parser.add_argument("--arm_action_scale", type=float, default=0.3, help="Scale arm actions (lower=more stable)")

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


class Stage6StablePlayEnv(ULC_G1_Env):
    """Play environment with stability improvements."""

    def __init__(self, cfg, arm_action_scale=0.3, **kwargs):
        super().__init__(cfg, **kwargs)

        self.arm_action_scale = arm_action_scale

        # Stage 6 residual scales - ama play'de daha düşük kullan
        self.base_residual_scales = torch.tensor(
            [1.5, 1.0, 1.0, 1.2, 0.8,
             1.5, 1.0, 1.0, 1.2, 0.8],
            device=self.device
        )

        # Effective scales = base * arm_action_scale
        self.residual_scales = self.base_residual_scales * self.arm_action_scale

        print(f"[STABLE] Arm action scale: {arm_action_scale}")
        print(f"[STABLE] Effective residual scales: {self.residual_scales.tolist()}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Use reduced arm actions for stability."""
        self.actions = torch.clamp(actions, -1.0, 1.0)

        leg_actions = self.actions[:, :12]
        arm_actions = self.actions[:, 12:]

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_joint_indices] = self.default_leg + leg_actions * 0.4

        # Reduced arm residuals for stability
        arm_cmd = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)
        arm_residual = arm_actions * self.residual_scales
        arm_target = torch.clamp(arm_cmd + arm_residual, -2.6, 2.6)

        target_pos[:, self.arm_joint_indices] = arm_target

        self.robot.set_joint_position_target(target_pos)
        self.gait_phase = (self.gait_phase + self.gait_frequency * self.step_dt) % 1.0
        self._prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()


def apply_arm_commands(env, left_arm, right_arm):
    """Set arm commands."""
    for i in range(5):
        env.left_arm_cmd[:, i] = left_arm[i]
        env.right_arm_cmd[:, i] = right_arm[i]
    env.arm_commands = torch.cat([env.left_arm_cmd, env.right_arm_cmd], dim=-1)


def main():
    print("=" * 60)
    print("🤖 ULC G1 STAGE 6 - STABLE PLAY MODE")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    state_dict = ckpt.get("actor_critic", ckpt)
    obs_dim = state_dict["actor.0.weight"].shape[1]
    act_dim = state_dict["log_std"].shape[0]
    print(f"[INFO] Model dims: obs={obs_dim}, act={act_dim}")

    # Create environment with arm action scaling
    cfg = ULC_G1_Stage4_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.termination["base_height_min"] = 0.25
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(cfg.num_observations,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_actions,))

    env = Stage6StablePlayEnv(cfg=cfg, arm_action_scale=args.arm_action_scale)
    env.current_stage = 4

    # Load policy
    policy = ActorCritic(obs_dim, act_dim).to("cuda:0")
    policy.load_state_dict(state_dict)
    policy.eval()
    print("[INFO] ✓ Model loaded!")

    # Determine arm commands
    if args.arms_forward:
        left_arm = [1.5, 0.0, 0.0, 0.0, 0.0]
        right_arm = [1.5, 0.0, 0.0, 0.0, 0.0]
        mode_name = "Arms Forward (1.5 rad)"
    elif args.arms_neutral:
        left_arm = [0.0, 0.0, 0.0, 0.0, 0.0]
        right_arm = [0.0, 0.0, 0.0, 0.0, 0.0]
        mode_name = "Arms Neutral"
    else:
        left_arm = [0.0, 0.0, 0.0, 0.0, 0.0]
        right_arm = [0.0, 0.0, 0.0, 0.0, 0.0]
        mode_name = "Arms Neutral (default)"

    print(f"[MODE] {mode_name}")
    print(f"[CMD] Left arm: {left_arm}")
    print(f"[CMD] Right arm: {right_arm}")
    print(f"[STABILITY] Smoothing: {args.smoothing}")
    print(f"[STABILITY] Arm action scale: {args.arm_action_scale}")

    # Reset and set commands
    obs_dict, _ = env.reset()
    env.height_command[:] = args.target_height
    env.velocity_commands[:, :] = 0.0
    env.torso_commands[:] = 0.0
    apply_arm_commands(env, left_arm, right_arm)
    obs = env._get_observations()["policy"]

    # Verify
    cmd_left = env.left_arm_cmd[0, 0].item()
    cmd_right = env.right_arm_cmd[0, 0].item()
    print(f"[VERIFY] Arm cmd: L={cmd_left:.2f}, R={cmd_right:.2f}")

    print("\n[PLAY] Starting... Press Ctrl+C to stop\n")

    prev_action = torch.zeros(env.num_envs, act_dim, device="cuda:0")
    step = 0

    # Statistics for tracking
    errors_left = []
    errors_right = []

    try:
        while simulation_app.is_running():
            apply_arm_commands(env, left_arm, right_arm)
            env.height_command[:] = args.target_height

            with torch.no_grad():
                raw_action = policy.act(obs, deterministic=True)
                # High smoothing for stability
                action = args.smoothing * prev_action + (1 - args.smoothing) * raw_action
                prev_action = action.clone()

            obs_dict, reward, term, trunc, _ = env.step(action)
            obs = obs_dict["policy"]
            step += 1

            done = term | trunc
            if done.any():
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                prev_action[reset_ids] = 0.0
                apply_arm_commands(env, left_arm, right_arm)

            if step % 100 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vel_z = env.robot.data.root_lin_vel_w[:, 2].mean().item()

                left_arm_pos = env.robot.data.joint_pos[:, env.left_arm_indices[0]].mean().item()
                right_arm_pos = env.robot.data.joint_pos[:, env.right_arm_indices[0]].mean().item()

                cmd_left = env.left_arm_cmd[0, 0].item()
                cmd_right = env.right_arm_cmd[0, 0].item()

                left_err = abs(left_arm_pos - cmd_left)
                right_err = abs(right_arm_pos - cmd_right)

                errors_left.append(left_err)
                errors_right.append(right_err)

                # Running average error
                avg_left = sum(errors_left[-10:]) / min(10, len(errors_left))
                avg_right = sum(errors_right[-10:]) / min(10, len(errors_right))

                print(f"Step {step:5d} | H={height:.3f}m | Vz={vel_z:+.3f} | "
                      f"L={left_arm_pos:+.2f} (err={left_err:.2f}, avg={avg_left:.2f}) | "
                      f"R={right_arm_pos:+.2f} (err={right_err:.2f}, avg={avg_right:.2f}) | "
                      f"R={reward.mean():.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        if errors_left:
            print(f"[STATS] Avg L_err: {sum(errors_left) / len(errors_left):.3f}")
            print(f"[STATS] Avg R_err: {sum(errors_right) / len(errors_right):.3f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()