"""
G1 Arm - TRAINING-COMPATIBLE PLAY
==================================

Training parametreleriyle AYNI workspace kullan!

Training'de:
  - Curriculum radius: 0.1m → 0.3m
  - Target mesafesi: 0.07m - current_radius arası
  - Action scale: 0.08

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_arm_training_compatible.py
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Training-Compatible Play")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

# Environment import
env_path = "source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/envs"
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg


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
    print("   G1 ARM - TRAINING-COMPATIBLE PLAY")
    print("   Training'deki workspace ve parametreler kullanılıyor!")
    print("=" * 70)

    # Checkpoint bul
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

    # Environment - TRAINING PARAMETRELERİ
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 300.0

    env = G1ArmReachEnv(cfg=env_cfg)

    # Policy yükle
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

    print("[INFO] Policy yüklendi!")

    # Training parametreleri
    print("\n" + "-" * 70)
    print("TRAINING PARAMETRELERİ:")
    print(f"  Action scale:     {env_cfg.action_scale}")
    print(f"  Smoothing alpha:  {env_cfg.action_smoothing_alpha}")
    print(f"  Position thresh:  {env_cfg.pos_threshold}m")
    print(f"  Min spawn dist:   {env_cfg.min_spawn_dist}m")
    print("-" * 70)

    # Joint limitleri göster
    print("\nJOINT LİMİTLERİ:")
    for i, idx in enumerate(env.arm_joint_indices.tolist()):
        name = env.robot.data.joint_names[idx]
        lower = env.joint_lower[i].item()
        upper = env.joint_upper[i].item()
        print(f"  {name}: [{lower:.2f}, {upper:.2f}] rad = [{lower*57.3:.0f}°, {upper*57.3:.0f}°]")
    print("-" * 70)

    print("\n[INFO] Ctrl+C ile çık\n")

    # Reset - environment kendi target'ını spawn edecek
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    root_pos = env.robot.data.root_pos_w[0]

    step = 0
    reach_count = 0
    episode_steps = 0

    try:
        while simulation_app.is_running():
            step += 1
            episode_steps += 1

            # ===== POLICY'DEN ACTION AL =====
            if obs.shape[-1] < obs_dim:
                pad = obs_dim - obs.shape[-1]
                obs = torch.cat([obs, torch.zeros(obs.shape[0], pad, device=obs.device)], dim=-1)

            with torch.no_grad():
                actions = actor(obs)

            # ===== ENV STEP =====
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]

            # ===== SPHERE'LERİ GÜNCELLE =====
            # Target sphere
            target_world = root_pos + env.target_pos[0]
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
            target_pose = torch.cat([target_world, default_quat]).unsqueeze(0)
            env.target_obj.write_root_pose_to_sim(target_pose)

            # EE sphere
            ee_pos_world = env._compute_ee_pos()
            palm_quat = env.robot.data.body_quat_w[:, env.palm_idx]
            ee_marker_pose = torch.cat([ee_pos_world[0], palm_quat[0]]).unsqueeze(0)
            env.ee_marker.write_root_pose_to_sim(ee_marker_pose)

            # ===== MESAFE =====
            ee_pos_rel = ee_pos_world[0] - root_pos
            distance = (ee_pos_rel - env.target_pos[0]).norm().item()

            # Reach detection (sparse reward > 15)
            if rewards[0].item() > 15:
                reach_count += 1
                print(f"\n  🎯 REACHED #{reach_count}! (in {episode_steps} steps)")
                episode_steps = 0

            # Log (her 50 step)
            if step % 50 == 0:
                current_joints = env.robot.data.joint_pos[0, env.arm_joint_indices]

                print(f"[Step {step:4d}] Dist: {distance:.3f}m | Reward: {rewards[0].item():+.2f} | "
                      f"Reaches: {reach_count}")
                print(f"  Target: [{env.target_pos[0,0]:.3f}, {env.target_pos[0,1]:.3f}, {env.target_pos[0,2]:.3f}]")
                print(f"  EE:     [{ee_pos_rel[0]:.3f}, {ee_pos_rel[1]:.3f}, {ee_pos_rel[2]:.3f}]")
                print(f"  Joints: [{current_joints[0]:.2f}, {current_joints[1]:.2f}, {current_joints[2]:.2f}, "
                      f"{current_joints[3]:.2f}, {current_joints[4]:.2f}]")
                print(f"  Actions: [{actions[0,0]:.3f}, {actions[0,1]:.3f}, {actions[0,2]:.3f}, "
                      f"{actions[0,3]:.3f}, {actions[0,4]:.3f}]")
                print()

    except KeyboardInterrupt:
        print("\n\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print("SONUÇLAR")
    print("=" * 70)
    print(f"  Toplam step:     {step}")
    print(f"  Toplam reach:    {reach_count}")
    if reach_count > 0:
        print(f"  Reach rate:      {reach_count / (step/100):.1f} per 100 steps")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()