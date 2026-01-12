#!/usr/bin/env python3
"""
ULC G1 Stage 4 Hierarchical - Play Script
==========================================

Test DiffIK arms + RL legs with HEIGHT variation.

MODLAR:
- stand:  Kollar sabit
- wave:   Yana sallama
- reach:  İleri uzanma
- down:   Aşağı uzanma (robot eğilmeli!)
- circle: Dairesel (YZ düzlemi)
- crouch: Aşağı hedef, robot çömelme testi

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/play_ulc_stage_4_hier.py ^
    --checkpoint logs/ulc/ulc_g1_stage4_hier_.../model_best.pt ^
    --num_envs 4 --mode crouch
"""

import argparse
import math

parser = argparse.ArgumentParser(description="ULC G1 Stage 4 Hierarchical Play")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--mode", type=str, default="wave",
                    choices=["stand", "wave", "reach", "down", "circle", "crouch"])
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--amplitude", type=float, default=0.15)
parser.add_argument("--frequency", type=float, default=0.3)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.ulc_g1_env import ULC_G1_Env
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.config.ulc_g1_env_cfg import ULC_G1_EnvCfg


# Actor network
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
        return mean


# Play environment
class HierarchicalPlayEnv(ULC_G1_Env):
    def __init__(self, cfg, **kwargs):
        cfg.num_actions = 12
        cfg.num_observations = 65
        super().__init__(cfg, **kwargs)

        self.diffik_damping = 0.05
        self.diffik_max_delta = 0.02

        self._find_ee_indices()

        self.left_arm_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_arm_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_arm_joints = torch.zeros(self.num_envs, 5, device=self.device)
        self.right_arm_joints = torch.zeros(self.num_envs, 5, device=self.device)

        self.init_left_offset = None
        self.init_right_offset = None
        self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)

    def _find_ee_indices(self):
        body_names = self.robot.data.body_names
        self.left_ee_idx = 28
        self.right_ee_idx = 29
        for i, name in enumerate(body_names):
            if "left_wrist_yaw_link" in name:
                self.left_ee_idx = i
            elif "right_wrist_yaw_link" in name:
                self.right_ee_idx = i

    def initialize_arm_state(self):
        base_pos = self.robot.data.root_pos_w[:, :3]
        left_ee = self.robot.data.body_pos_w[:, self.left_ee_idx]
        right_ee = self.robot.data.body_pos_w[:, self.right_ee_idx]

        self.init_left_offset = (left_ee - base_pos).clone()
        self.init_right_offset = (right_ee - base_pos).clone()

        self.left_arm_target = self.init_left_offset.clone()
        self.right_arm_target = self.init_right_offset.clone()

        self.left_arm_joints = self.robot.data.joint_pos[:, self.left_arm_indices].clone()
        self.right_arm_joints = self.robot.data.joint_pos[:, self.right_arm_indices].clone()

        print(f"[INIT] Left EE offset: {self.init_left_offset[0].tolist()}")
        print(f"[INIT] Right EE offset: {self.init_right_offset[0].tolist()}")

    def set_arm_offset(self, left_off, right_off):
        if self.init_left_offset is not None:
            self.left_arm_target = self.init_left_offset + left_off
            self.right_arm_target = self.init_right_offset + right_off

    def _get_observations(self):
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b

        leg_pos = self.robot.data.joint_pos[:, self.leg_joint_indices]
        leg_pos_norm = (leg_pos - self.default_leg) / 0.5
        leg_vel = self.robot.data.joint_vel[:, self.leg_joint_indices] * 0.1

        height_cmd = self.height_command.unsqueeze(-1)
        vel_cmd = self.velocity_commands

        gait_sin = torch.sin(2 * math.pi * self.gait_phase).unsqueeze(-1)
        gait_cos = torch.cos(2 * math.pi * self.gait_phase).unsqueeze(-1)

        base_pos = self.robot.data.root_pos_w[:, :3]
        left_ee_offset = self.robot.data.body_pos_w[:, self.left_ee_idx] - base_pos
        right_ee_offset = self.robot.data.body_pos_w[:, self.right_ee_idx] - base_pos

        left_target_z = self.left_arm_target[:, 2:3]
        right_target_z = self.right_arm_target[:, 2:3]

        obs = torch.cat([
            base_lin_vel, base_ang_vel, proj_gravity,
            leg_pos_norm, leg_vel,
            height_cmd, vel_cmd,
            gait_sin, gait_cos,
            self._prev_leg_actions,
            left_ee_offset, right_ee_offset,
            self.left_arm_target, self.right_arm_target,
            left_target_z, right_target_z,
        ], dim=-1)

        return {"policy": obs}

    def _apply_diffik(self):
        base_pos = self.robot.data.root_pos_w[:, :3]

        left_ee = self.robot.data.body_pos_w[:, self.left_ee_idx]
        left_error = (base_pos + self.left_arm_target) - left_ee

        right_ee = self.robot.data.body_pos_w[:, self.right_ee_idx]
        right_error = (base_pos + self.right_arm_target) - right_ee

        try:
            jacobians = self.robot.root_physx_view.get_jacobians()

            left_J = jacobians[:, self.left_ee_idx, :3, :][:, :, self.left_arm_indices]
            left_delta = self._damped_ls(left_J, left_error)

            right_J = jacobians[:, self.right_ee_idx, :3, :][:, :, self.right_arm_indices]
            right_delta = self._damped_ls(right_J, right_error)

            self.left_arm_joints = torch.clamp(self.left_arm_joints + left_delta, -2.6, 2.6)
            self.right_arm_joints = torch.clamp(self.right_arm_joints + right_delta, -2.6, 2.6)
        except:
            pass

    def _damped_ls(self, J, error):
        batch = J.shape[0]
        JJT = torch.bmm(J, J.transpose(1, 2))
        damping = (self.diffik_damping ** 2) * torch.eye(3, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        try:
            x = torch.linalg.solve(JJT + damping, error.unsqueeze(-1))
            delta = torch.bmm(J.transpose(1, 2), x).squeeze(-1)
            return torch.clamp(delta, -self.diffik_max_delta, self.diffik_max_delta)
        except:
            return torch.zeros(batch, J.shape[2], device=self.device)

    def _pre_physics_step(self, actions):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._apply_diffik()

        leg_targets = self.default_leg + self.actions * 0.4

        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_joint_indices] = leg_targets
        target_pos[:, self.left_arm_indices] = self.left_arm_joints
        target_pos[:, self.right_arm_indices] = self.right_arm_joints

        self.robot.set_joint_position_target(target_pos)
        self.gait_phase = (self.gait_phase + self.gait_frequency * self.step_dt) % 1.0
        self._prev_leg_actions = self.actions.clone()


MODE_DESC = {
    "stand": "Arms stationary, balance test",
    "wave": "Wave motion (Y axis)",
    "reach": "Reach forward (X axis)",
    "down": "Reach down (Z- axis) - robot should lower!",
    "circle": "Circular motion (YZ plane)",
    "crouch": "Crouch test - progressive down movement",
}


def main():
    print("\n" + "=" * 70)
    print("  ULC G1 STAGE 4 HIERARCHICAL - PLAY")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Description: {MODE_DESC[args.mode]}")
    print("=" * 70 + "\n")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    state_dict = ckpt.get("actor_critic", ckpt)

    obs_dim = state_dict["actor.0.weight"].shape[1]
    act_dim = state_dict["log_std"].shape[0]
    print(f"[INFO] Model: obs={obs_dim}, act={act_dim}")

    if "curriculum_level" in ckpt:
        print(f"[INFO] Curriculum level: {ckpt['curriculum_level']}")

    # Create environment
    cfg = ULC_G1_EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(65,))
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))

    env = HierarchicalPlayEnv(cfg=cfg)

    # Load policy
    policy = ActorCritic(obs_dim, act_dim).to("cuda:0")
    policy.load_state_dict(state_dict)
    policy.eval()
    print("[INFO] ✓ Policy loaded!")

    # Reset
    obs_dict, _ = env.reset()
    env.height_command[:] = 0.72
    env.velocity_commands[:, :] = 0.0
    env.initialize_arm_state()
    obs = env._get_observations()["policy"]

    print(f"\n[PLAY] Mode: {args.mode}")
    print(f"[PLAY] Amplitude: {args.amplitude}m")
    print(f"[PLAY] Max steps: {args.max_steps}")
    print("[PLAY] Press Ctrl+C to stop\n")

    step = 0
    dt = 0.02
    initial_height = env.robot.data.root_pos_w[0, 2].item()

    try:
        while simulation_app.is_running() and step < args.max_steps:
            t = step * dt

            left_off = torch.zeros(args.num_envs, 3, device="cuda:0")
            right_off = torch.zeros(args.num_envs, 3, device="cuda:0")

            amp = args.amplitude
            freq = args.frequency

            if args.mode == "stand":
                pass

            elif args.mode == "wave":
                wave = amp * math.sin(2 * math.pi * freq * t)
                right_off[:, 1] = wave

            elif args.mode == "reach":
                reach = amp * (1 - math.cos(2 * math.pi * freq * t)) / 2
                left_off[:, 0] = reach
                right_off[:, 0] = reach

            elif args.mode == "down":
                # Go DOWN - robot should crouch!
                down = -amp * (1 - math.cos(2 * math.pi * freq * t)) / 2
                left_off[:, 2] = down
                right_off[:, 2] = down

            elif args.mode == "circle":
                angle = 2 * math.pi * freq * t
                right_off[:, 1] = amp * math.sin(angle)
                right_off[:, 2] = amp * math.cos(angle)

            elif args.mode == "crouch":
                # Progressive crouch - slow descent
                progress = min(1.0, t / 5.0)  # 5 seconds to full crouch
                down = -amp * progress
                left_off[:, 0] = 0.05 * progress  # Slight forward reach
                right_off[:, 0] = 0.05 * progress
                left_off[:, 2] = down
                right_off[:, 2] = down

            env.set_arm_offset(left_off, right_off)

            with torch.no_grad():
                action = policy.act(obs, deterministic=True)

            obs_dict, reward, term, trunc, _ = env.step(action)
            obs = obs_dict["policy"]
            step += 1

            if (term | trunc).any():
                print(f"\n[!] Episode reset at step {step}")
                obs_dict, _ = env.reset()
                env.initialize_arm_state()
                obs = env._get_observations()["policy"]

            if step % 100 == 0:
                height = env.robot.data.root_pos_w[0, 2].item()
                height_change = height - initial_height
                vz = env.robot.data.root_lin_vel_w[:, 2].mean().item()

                base = env.robot.data.root_pos_w[0, :3]
                left_ee = env.robot.data.body_pos_w[0, env.left_ee_idx]
                right_ee = env.robot.data.body_pos_w[0, env.right_ee_idx]

                left_err = torch.norm(left_ee - (base + env.left_arm_target[0])).item()
                right_err = torch.norm(right_ee - (base + env.right_arm_target[0])).item()

                # Target height info
                target_z = env.left_arm_target[0, 2].item()

                status = "✅" if height > 0.55 else "⚠️"
                crouch_indicator = f"[CROUCHING {abs(height_change) * 100:.1f}cm]" if height_change < -0.03 else ""

                print(f"Step {step:4d} | H={height:.3f}m (Δ={height_change:+.3f}) | "
                      f"Target_Z={target_z:+.3f}m | "
                      f"Arm_err=({left_err:.3f}, {right_err:.3f}) | "
                      f"R={reward.mean():.2f} {status} {crouch_indicator}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")

    # Summary
    final_height = env.robot.data.root_pos_w[0, 2].item()
    total_crouch = initial_height - final_height

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Steps: {step}")
    print(f"  Initial height: {initial_height:.3f}m")
    print(f"  Final height: {final_height:.3f}m")
    print(f"  Total crouch: {total_crouch * 100:.1f}cm")

    if args.mode in ["down", "crouch"]:
        if total_crouch > 0.05:
            print(f"\n  🎉 SUCCESS - Robot learned to crouch!")
        else:
            print(f"\n  ⚠️ Robot didn't crouch much - may need more training")
    elif final_height > 0.55:
        print(f"\n  ✅ Robot stayed stable!")

    print("=" * 60 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()