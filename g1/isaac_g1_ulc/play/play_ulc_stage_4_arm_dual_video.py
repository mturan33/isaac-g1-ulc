"""
G1 DUAL ARM PLAY - VIDEO RECORDING VERSION
============================================

Kamera robotun önünde, robota bakacak şekilde konumlandırılmış.
X paylaşımı için video çekimi uygun.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_4_arm_dual_video.py
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Dual Arm Play - Video")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

# Import camera view utility
import isaaclab.sim as sim_utils

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


def setup_camera_for_video():
    """
    Kamerayı robotun önüne konumlandır.
    Robot pozisyonu: (0, 0, 1.0)

    Koordinat sistemi:
    - X- = İleri (robotun baktığı yön)
    - Y+ = Sağ
    - Z+ = Yukarı
    """
    # === KAMERA AÇISI SEÇENEKLERİ ===

    # SEÇENEK 1: 3/4 diagonal view (ÖNERİLEN - kollar net görünür)
    eye = (-1.4, 0.8, 1.4)  # Önde-sağda, hafif yukarıda
    target = (0.0, 0.0, 1.0)

    # SEÇENEK 2: Düz önden bakış
    # eye = (-1.6, 0.0, 1.35)
    # target = (0.0, 0.0, 1.05)

    # SEÇENEK 3: Daha yakın, dramatik açı
    # eye = (-1.0, 0.5, 1.2)
    # target = (0.0, 0.0, 1.0)

    # SEÇENEK 4: Üstten bakış (kuş bakışı)
    # eye = (-0.8, 0.0, 2.0)
    # target = (0.0, 0.0, 1.0)

    sim_utils.set_camera_view(eye=eye, target=target)
    print(f"[CAMERA] Eye: {eye}, Target: {target}")


def main():
    print("\n" + "=" * 70)
    print("   G1 DUAL ARM PLAY - VIDEO RECORDING VERSION")
    print("   Kamera robotun önünde konumlandırılmış")
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

    # === KAMERAYI AYARLA ===
    setup_camera_for_video()

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
    print("VIDEO ÇEKIMI İÇİN HAZIR!")
    print("-" * 70)
    print("VISUAL MARKERS:")
    print("  🟢 Yeşil   = Sağ kol target")
    print("  🔵 Mavi    = Sol kol target")
    print("  🟠 Turuncu = Sağ el (EE)")
    print("  🟣 Mor     = Sol el (EE)")
    print("-" * 70)
    print("  📹 Screen recording başlat (Win+G veya OBS)")
    print("  ⏱️  20-30 saniye kaydet")
    print("  🛑 Ctrl+C ile çık")
    print("-" * 70 + "\n")

    # Reset
    obs_dict, _ = env.reset()

    # Reset sonrası kamerayı tekrar ayarla (bazı env'ler resetliyor)
    setup_camera_for_video()

    step = 0

    try:
        while simulation_app.is_running():
            step += 1

            root_pos = env.robot.data.root_pos_w[0]

            # ===== RIGHT ARM OBSERVATION (MUST MATCH TRAINING ORDER!) =====
            right_joint_pos = env.robot.data.joint_pos[0, env.right_arm_indices]
            right_joint_vel = env.robot.data.joint_vel[0, env.right_arm_indices]
            right_ee_pos = env._compute_right_ee_pos()[0] - root_pos
            right_target = env.right_target_pos[0]
            right_error = right_target - right_ee_pos
            right_error_norm = right_error / 0.31

            right_obs = torch.cat([
                right_joint_pos,
                right_joint_vel * 0.1,
                right_target,
                right_ee_pos,
                right_error_norm,
            ]).unsqueeze(0)

            with torch.no_grad():
                right_actions = actor(right_obs)[0]

            # ===== LEFT ARM OBSERVATION (MIRRORED) =====
            left_joint_pos = env.robot.data.joint_pos[0, env.left_arm_indices]
            left_joint_vel = env.robot.data.joint_vel[0, env.left_arm_indices]
            left_ee_pos = env._compute_left_ee_pos()[0] - root_pos
            left_target = env.left_target_pos[0]

            left_target_mirrored = left_target.clone()
            left_target_mirrored[1] = -left_target_mirrored[1]

            left_ee_mirrored = left_ee_pos.clone()
            left_ee_mirrored[1] = -left_ee_mirrored[1]

            left_joint_mirrored = left_joint_pos.clone()
            left_joint_mirrored[1] = -left_joint_mirrored[1]
            left_joint_mirrored[2] = -left_joint_mirrored[2]
            left_joint_mirrored[4] = -left_joint_mirrored[4]

            left_joint_vel_mirrored = left_joint_vel.clone()
            left_joint_vel_mirrored[1] = -left_joint_vel_mirrored[1]
            left_joint_vel_mirrored[2] = -left_joint_vel_mirrored[2]
            left_joint_vel_mirrored[4] = -left_joint_vel_mirrored[4]

            left_error = left_target_mirrored - left_ee_mirrored
            left_error_norm = left_error / 0.31

            left_obs = torch.cat([
                left_joint_mirrored,
                left_joint_vel_mirrored * 0.1,
                left_target_mirrored,
                left_ee_mirrored,
                left_error_norm,
            ]).unsqueeze(0)

            with torch.no_grad():
                left_actions_raw = actor(left_obs)[0]

            left_actions = left_actions_raw.clone()
            left_actions[1] = -left_actions[1]
            left_actions[2] = -left_actions[2]
            left_actions[4] = -left_actions[4]

            # ===== COMBINE ACTIONS =====
            combined_actions = torch.cat([right_actions, left_actions]).unsqueeze(0)

            # Step
            obs_dict, rewards, terminated, truncated, info = env.step(combined_actions)

            # Compute distances
            root_pos_new = env.robot.data.root_pos_w[0]
            right_ee_new = env._compute_right_ee_pos()[0] - root_pos_new
            left_ee_new = env._compute_left_ee_pos()[0] - root_pos_new
            right_dist = (right_ee_new - env.right_target_pos[0]).norm().item()
            left_dist = (left_ee_new - env.left_target_pos[0]).norm().item()

            # Log (daha az sıklıkta - video için temiz konsol)
            if step % 100 == 0:
                total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
                print(
                    f"[Step {step:4d}] Right: {int(env.right_reach_count[0].item())} | Left: {int(env.left_reach_count[0].item())} | Total: {total} reaches")

    except KeyboardInterrupt:
        print("\n\n[INFO] Video kaydı durduruldu")

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