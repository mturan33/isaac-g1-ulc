"""
G1 DUAL ARM PLAY - Using G1DualArmEnv
======================================

Sağ kol policy'sini sol kola mirror olarak uygular.
4 visual marker ile çalışır.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_dual_arm.py
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Dual Arm Play")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

# Add env path
env_path = "source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/envs"
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

# Import dual arm env
from g1_arm_dual_env import G1DualArmEnv, G1DualArmEnvCfg


class SimpleActor(nn.Module):
    """RSL-RL actor network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    print("\n" + "=" * 70)
    print("   G1 DUAL ARM PLAY")
    print("   Sağ kol policy'si → Sol kola mirror")
    print("=" * 70)

    # Find checkpoint
    log_dir = "logs/ulc"
    checkpoint_path = None

    if os.path.exists(log_dir):
        for folder in sorted(os.listdir(log_dir), reverse=True):
            if "stage4_arm" in folder:
                model_path = os.path.join(log_dir, folder, "model_15999.pt")
                if os.path.exists(model_path):
                    checkpoint_path = model_path
                    break

    if checkpoint_path is None:
        print("[ERROR] Checkpoint bulunamadı!")
        return

    print(f"\n[INFO] Checkpoint: {checkpoint_path}")

    # Create dual arm environment
    env_cfg = G1DualArmEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 300.0

    env = G1DualArmEnv(cfg=env_cfg)

    # Load policy
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint)

    actor_keys = [k for k in model_state.keys() if k.startswith('actor.') and 'weight' in k]
    actor_keys.sort()

    obs_dim = model_state[actor_keys[0]].shape[1]
    act_dim = model_state[actor_keys[-1]].shape[0]
    hidden_dims = [model_state[k].shape[0] for k in actor_keys[:-1]]

    print(f"[INFO] Policy: obs={obs_dim}, act={act_dim}, hidden={hidden_dims}")

    actor = SimpleActor(obs_dim, act_dim, hidden_dims).to("cuda:0")
    actor_state = {k.replace('actor.', 'net.'): v for k, v in model_state.items() if k.startswith('actor.')}
    actor.load_state_dict(actor_state)
    actor.eval()

    print("\n" + "-" * 70)
    print("VISUAL MARKERS:")
    print("  🟢 Yeşil   = Sağ kol target")
    print("  🔵 Mavi    = Sol kol target")
    print("  🟠 Turuncu = Sağ el (EE)")
    print("  🟣 Mor     = Sol el (EE)")
    print("-" * 70)
    print("[INFO] Ctrl+C ile çık\n")

    # Reset
    obs_dict, _ = env.reset()

    step = 0
    right_smoothed = torch.zeros(5, device="cuda:0")
    left_smoothed = torch.zeros(5, device="cuda:0")

    try:
        while simulation_app.is_running():
            step += 1

            root_pos = env.robot.data.root_pos_w[0]

            # ===== RIGHT ARM OBSERVATION =====
            right_joint_pos = env.robot.data.joint_pos[0, env.right_arm_indices]
            right_joint_vel = env.robot.data.joint_vel[0, env.right_arm_indices]
            right_ee_pos = env._compute_right_ee_pos()[0] - root_pos
            right_target = env.right_target_pos[0]
            right_error = right_target - right_ee_pos

            right_obs = torch.cat([
                right_target,
                right_ee_pos,
                right_joint_pos,
                right_joint_vel * 0.1,
                right_smoothed[:4]  # Prev actions
            ]).unsqueeze(0)

            # Pad to policy input size
            if right_obs.shape[-1] < obs_dim:
                pad = obs_dim - right_obs.shape[-1]
                right_obs = torch.cat([right_obs, torch.zeros(1, pad, device="cuda:0")], dim=-1)
            elif right_obs.shape[-1] > obs_dim:
                right_obs = right_obs[:, :obs_dim]

            with torch.no_grad():
                right_actions = actor(right_obs)[0]

            # ===== LEFT ARM OBSERVATION (MIRRORED) =====
            left_joint_pos = env.robot.data.joint_pos[0, env.left_arm_indices]
            left_joint_vel = env.robot.data.joint_vel[0, env.left_arm_indices]
            left_ee_pos = env._compute_left_ee_pos()[0] - root_pos
            left_target = env.left_target_pos[0]

            # Mirror Y coordinates for policy input
            left_target_mirrored = left_target.clone()
            left_target_mirrored[1] = -left_target_mirrored[1]

            left_ee_mirrored = left_ee_pos.clone()
            left_ee_mirrored[1] = -left_ee_mirrored[1]

            # Mirror joint positions (roll joints are opposite)
            left_joint_mirrored = left_joint_pos.clone()
            left_joint_mirrored[1] = -left_joint_mirrored[1]  # shoulder_roll
            left_joint_mirrored[2] = -left_joint_mirrored[2]  # shoulder_yaw
            left_joint_mirrored[4] = -left_joint_mirrored[4]  # elbow_roll

            left_joint_vel_mirrored = left_joint_vel.clone()
            left_joint_vel_mirrored[1] = -left_joint_vel_mirrored[1]
            left_joint_vel_mirrored[2] = -left_joint_vel_mirrored[2]
            left_joint_vel_mirrored[4] = -left_joint_vel_mirrored[4]

            left_obs = torch.cat([
                left_target_mirrored,
                left_ee_mirrored,
                left_joint_mirrored,
                left_joint_vel_mirrored * 0.1,
                left_smoothed[:4]
            ]).unsqueeze(0)

            if left_obs.shape[-1] < obs_dim:
                pad = obs_dim - left_obs.shape[-1]
                left_obs = torch.cat([left_obs, torch.zeros(1, pad, device="cuda:0")], dim=-1)
            elif left_obs.shape[-1] > obs_dim:
                left_obs = left_obs[:, :obs_dim]

            with torch.no_grad():
                left_actions_raw = actor(left_obs)[0]

            # Mirror actions back (roll joints opposite)
            left_actions = left_actions_raw.clone()
            left_actions[1] = -left_actions[1]  # shoulder_roll
            left_actions[2] = -left_actions[2]  # shoulder_yaw
            left_actions[4] = -left_actions[4]  # elbow_roll

            # ===== COMBINE ACTIONS =====
            combined_actions = torch.cat([right_actions, left_actions]).unsqueeze(0)

            # Smoothing for logging
            alpha = 0.5
            right_smoothed = alpha * right_actions + (1 - alpha) * right_smoothed
            left_smoothed = alpha * left_actions + (1 - alpha) * left_smoothed

            # Step
            obs_dict, rewards, terminated, truncated, info = env.step(combined_actions)

            # Compute distances
            right_dist = (right_ee_pos - right_target).norm().item()
            left_dist = (left_ee_pos - left_target).norm().item()

            # Log
            if step % 50 == 0:
                print(f"[Step {step:4d}]")
                print(f"  SAĞ KOL:  Dist={right_dist:.3f}m | Reaches={int(env.right_reach_count[0].item())}")
                print(f"  SOL KOL:  Dist={left_dist:.3f}m | Reaches={int(env.left_reach_count[0].item())}")
                total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
                print(f"  TOPLAM:   {total} reaches")
                print()

    except KeyboardInterrupt:
        print("\n\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print("DUAL ARM SONUÇLAR")
    print("=" * 70)
    print(f"  Toplam step:      {step}")
    print(f"  Sağ kol reaches:  {int(env.right_reach_count[0].item())}")
    print(f"  Sol kol reaches:  {int(env.left_reach_count[0].item())}")
    total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
    print(f"  TOPLAM reaches:   {total}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()