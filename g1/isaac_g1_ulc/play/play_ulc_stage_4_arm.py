"""
G1 Arm - TRAINED POLICY PLAY
=============================

16 saat eğitilmiş policy'yi kullan!
Sadece ulaşılabilir workspace içinde hedefler.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_arm_trained_policy.py
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Trained Policy Play")
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


def spawn_reachable_target(env, root_pos):
    """
    Sağ kolun ULAŞABİLECEĞİ workspace içinde hedef spawn et.

    G1 Sağ Kol Workspace (root-relative):
    - X (ileri): 0.05 to 0.35m (çok ileri ulaşamaz)
    - Y (sağ):   0.10 to 0.35m (sol tarafa ulaşamaz!)
    - Z (yukarı): -0.15 to 0.25m

    Toplam mesafe: 0.15 - 0.35m arası
    """
    device = root_pos.device

    while True:
        # Random pozisyon
        x = torch.empty(1, device=device).uniform_(0.05, 0.35)
        y = torch.empty(1, device=device).uniform_(0.10, 0.35)  # SADECE SAĞ TARAF!
        z = torch.empty(1, device=device).uniform_(-0.15, 0.25)

        target_rel = torch.tensor([x.item(), y.item(), z.item()], device=device)

        # Mesafe kontrolü (çok yakın veya çok uzak olmasın)
        distance = target_rel.norm().item()
        if 0.15 < distance < 0.40:
            return target_rel


def main():
    print("\n" + "=" * 70)
    print("   G1 ARM - TRAINED POLICY PLAY")
    print("   16 saat eğitilmiş policy çalışıyor!")
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
        print("  Beklenen: logs/ulc/ulc_g1_stage4_arm_*/model_15999.pt")
        return

    print(f"\n[INFO] Checkpoint: {checkpoint_path}")

    # Environment
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 300.0

    env = G1ArmReachEnv(cfg=env_cfg)

    # Policy yükle
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint)

    # Actor boyutları
    actor_keys = [k for k in model_state.keys() if k.startswith('actor.') and 'weight' in k]
    actor_keys.sort()

    obs_dim = model_state[actor_keys[0]].shape[1]
    act_dim = model_state[actor_keys[-1]].shape[0]
    hidden_dims = [model_state[k].shape[0] for k in actor_keys[:-1]]

    print(f"[INFO] Policy: obs={obs_dim}, act={act_dim}, hidden={hidden_dims}")

    # Actor
    actor = SimpleActor(obs_dim, act_dim, hidden_dims).to("cuda:0")
    actor_state = {k.replace('actor.', 'net.'): v for k, v in model_state.items() if k.startswith('actor.')}
    actor.load_state_dict(actor_state)
    actor.eval()

    print("[INFO] Policy yüklendi!")
    print("\n" + "-" * 70)
    print("WORKSPACE (Sağ kol ulaşabilir):")
    print("  X (ileri):  0.05 - 0.35m")
    print("  Y (sağ):    0.10 - 0.35m  ← Sol tarafa ULAŞAMAZ!")
    print("  Z (dikey): -0.15 - 0.25m")
    print("-" * 70)
    print("\n[INFO] Ctrl+C ile çık\n")

    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    root_pos = env.robot.data.root_pos_w[0]

    # İlk hedef - ulaşılabilir workspace içinde
    target_rel = spawn_reachable_target(env, root_pos)
    env.target_pos[0] = target_rel

    step = 0
    reach_count = 0
    reach_threshold = 0.05  # 5cm
    steps_since_reach = 0

    try:
        while simulation_app.is_running():
            step += 1
            steps_since_reach += 1

            # ===== POLICY'DEN ACTION AL =====
            if obs.shape[-1] < obs_dim:
                pad = obs_dim - obs.shape[-1]
                obs = torch.cat([obs, torch.zeros(obs.shape[0], pad, device=obs.device)], dim=-1)

            with torch.no_grad():
                actions = actor(obs)

            # ===== ENV STEP - POLICY ACTION KULLAN =====
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

            # ===== MESAFE KONTROLÜ =====
            ee_pos_rel = ee_pos_world[0] - root_pos
            distance = (ee_pos_rel - env.target_pos[0]).norm().item()

            # Hedefe ulaştı mı?
            if distance < reach_threshold:
                reach_count += 1
                print(f"\n  🎯 REACHED #{reach_count}! (in {steps_since_reach} steps)")

                # Yeni hedef spawn
                target_rel = spawn_reachable_target(env, root_pos)
                env.target_pos[0] = target_rel
                steps_since_reach = 0

                print(f"     New target: [{target_rel[0]:.3f}, {target_rel[1]:.3f}, {target_rel[2]:.3f}]")

            # Çok uzun sürerse yeni hedef (timeout)
            if steps_since_reach > 500:
                print(f"\n  ⏱️ Timeout! Spawning easier target...")
                target_rel = spawn_reachable_target(env, root_pos)
                env.target_pos[0] = target_rel
                steps_since_reach = 0

            # Log
            if step % 100 == 0:
                print(f"[Step {step:4d}] Dist: {distance:.3f}m | Reward: {rewards[0].item():+.2f} | "
                      f"Reaches: {reach_count}")

    except KeyboardInterrupt:
        print("\n\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print("SONUÇLAR")
    print("=" * 70)
    print(f"  Toplam step:     {step}")
    print(f"  Toplam reach:    {reach_count}")
    if reach_count > 0:
        print(f"  Ortalama süre:   {step // reach_count} step/reach")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()