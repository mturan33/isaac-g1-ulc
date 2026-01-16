"""
G1 DUAL ARM PLAY
=================

Sağ ve sol kolu AYNI ANDA çalıştır!
Sağ kol policy'sini sol kola mirror olarak uygula.

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

# Environment import
env_path = "source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/envs"
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg

# Isaac Lab imports for visual markers
from isaaclab.sim.spawners.shapes import SphereCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
import isaaclab.sim as sim_utils


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
    print("   Sağ ve Sol kol AYNI ANDA çalışıyor!")
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

    # Environment
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

    print(f"[INFO] Policy: obs={obs_dim}, act={act_dim}")

    actor = SimpleActor(obs_dim, act_dim, hidden_dims).to("cuda:0")
    actor_state = {k.replace('actor.', 'net.'): v for k, v in model_state.items() if k.startswith('actor.')}
    actor.load_state_dict(actor_state)
    actor.eval()

    # ===== SOL KOL JOINT İNDEXLERİ =====
    # Sağ kol: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_elbow_roll
    # Sol kol: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch, left_elbow_roll

    joint_names = env.robot.data.joint_names
    print(f"\n[INFO] Tüm joint'ler: {joint_names}")

    # Sol kol joint'lerini bul
    left_arm_joint_names = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint"
    ]

    left_arm_indices = []
    for name in left_arm_joint_names:
        if name in joint_names:
            left_arm_indices.append(joint_names.index(name))

    left_arm_indices = torch.tensor(left_arm_indices, device="cuda:0")
    print(f"[INFO] Sağ kol indices: {env.arm_joint_indices.tolist()}")
    print(f"[INFO] Sol kol indices: {left_arm_indices.tolist()}")

    # Sol palm (el) body index bul
    body_names = env.robot.data.body_names
    left_palm_idx = body_names.index("left_palm_link") if "left_palm_link" in body_names else None
    print(f"[INFO] Sağ palm idx: {env.palm_idx}")
    print(f"[INFO] Sol palm idx: {left_palm_idx}")

    if left_palm_idx is None:
        print("[ERROR] Sol palm bulunamadı!")
        return

    # ===== SOL KOL İÇİN JOINT LİMİTLERİ =====
    all_lower = env.robot.data.soft_joint_pos_limits[0, :, 0]
    all_upper = env.robot.data.soft_joint_pos_limits[0, :, 1]
    left_joint_lower = all_lower[left_arm_indices]
    left_joint_upper = all_upper[left_arm_indices]

    # ===== İKİ TARGET POZİSYONU =====
    root_pos = env.robot.data.root_pos_w[0]

    # Sağ kol target'ı env'den geliyor
    # Sol kol için ayrı target
    left_target_pos = torch.tensor([0.0, -0.15, 0.10], device="cuda:0")  # Sol tarafta (y negatif)

    print("\n" + "-" * 70)
    print("DUAL ARM CONTROL:")
    print("  🟢 Yeşil küre  = Sağ kol target")
    print("  🔵 Mavi küre   = Sol kol target")
    print("  🟠 Turuncu     = Sağ el (EE)")
    print("  🟣 Mor         = Sol el (EE)")
    print("-" * 70)
    print("\n[INFO] Ctrl+C ile çık\n")

    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    step = 0
    right_reach_count = 0
    left_reach_count = 0
    right_episode_steps = 0
    left_episode_steps = 0

    # Sol kol için smoothed actions
    left_smoothed_actions = torch.zeros(5, device="cuda:0")

    try:
        while simulation_app.is_running():
            step += 1
            right_episode_steps += 1
            left_episode_steps += 1

            # ===== SAĞ KOL - ORİJİNAL POLİCY =====
            if obs.shape[-1] < obs_dim:
                pad = obs_dim - obs.shape[-1]
                obs = torch.cat([obs, torch.zeros(obs.shape[0], pad, device=obs.device)], dim=-1)

            with torch.no_grad():
                right_actions = actor(obs)

            # ===== SOL KOL - MİRROR OBSERVATİON =====
            # Sol kol için observation oluştur
            left_ee_pos = env.robot.data.body_pos_w[0, left_palm_idx] - root_pos
            left_joint_pos = env.robot.data.joint_pos[0, left_arm_indices]
            left_joint_vel = env.robot.data.joint_vel[0, left_arm_indices]

            # Mirror: Y koordinatını ve bazı joint'leri ters çevir
            left_target_mirrored = left_target_pos.clone()
            # Y'yi aynala (sol taraf için target'ın y'si pozitif olacak şekilde policy'ye ver)
            left_obs_target = torch.tensor([left_target_mirrored[0], -left_target_mirrored[1], left_target_mirrored[2]], device="cuda:0")
            left_obs_ee = torch.tensor([left_ee_pos[0], -left_ee_pos[1], left_ee_pos[2]], device="cuda:0")

            # Joint pos/vel mirror (roll joint'ler ters)
            left_obs_joint_pos = left_joint_pos.clone()
            left_obs_joint_pos[1] = -left_obs_joint_pos[1]  # shoulder_roll
            left_obs_joint_pos[2] = -left_obs_joint_pos[2]  # shoulder_yaw
            left_obs_joint_pos[4] = -left_obs_joint_pos[4]  # elbow_roll

            left_obs_joint_vel = left_joint_vel.clone()
            left_obs_joint_vel[1] = -left_obs_joint_vel[1]
            left_obs_joint_vel[2] = -left_obs_joint_vel[2]
            left_obs_joint_vel[4] = -left_obs_joint_vel[4]

            # Sol kol observation
            left_obs = torch.cat([
                left_obs_target,
                left_obs_ee,
                left_obs_joint_pos,
                left_obs_joint_vel,
                left_smoothed_actions
            ]).unsqueeze(0)

            # Padding
            if left_obs.shape[-1] < obs_dim:
                pad = obs_dim - left_obs.shape[-1]
                left_obs = torch.cat([left_obs, torch.zeros(1, pad, device="cuda:0")], dim=-1)

            with torch.no_grad():
                left_actions_raw = actor(left_obs)

            # Mirror actions (roll joint'ler ters)
            left_actions = left_actions_raw.clone()
            left_actions[0, 1] = -left_actions[0, 1]  # shoulder_roll
            left_actions[0, 2] = -left_actions[0, 2]  # shoulder_yaw
            left_actions[0, 4] = -left_actions[0, 4]  # elbow_roll

            # ===== SOL KOL ACTION UYGULA =====
            # Smoothing
            alpha = 0.5
            left_smoothed_actions = alpha * left_actions[0] + (1 - alpha) * left_smoothed_actions

            # Scale ve delta
            action_scale = 0.08
            left_delta = left_smoothed_actions * action_scale

            # Yeni joint pozisyonları
            left_new_pos = left_joint_pos + left_delta
            left_new_pos = torch.clamp(left_new_pos, left_joint_lower, left_joint_upper)

            # Sol kol joint target'larını set et
            all_joint_targets = env.robot.data.joint_pos.clone()
            all_joint_targets[0, left_arm_indices] = left_new_pos

            # ===== SAĞ KOL - ENV.STEP =====
            obs_dict, rewards, terminated, truncated, info = env.step(right_actions)
            obs = obs_dict["policy"]

            # Sol kol target'larını uygula (env.step sonrası)
            current_targets = env.robot.data.joint_pos.clone()
            current_targets[0, left_arm_indices] = left_new_pos
            env.robot.set_joint_position_target(current_targets)

            # ===== VISUAL MARKERS =====
            # Sağ target (yeşil - env'den)
            right_target_world = root_pos + env.target_pos[0]
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
            right_target_pose = torch.cat([right_target_world, default_quat]).unsqueeze(0)
            env.target_obj.write_root_pose_to_sim(right_target_pose)

            # Sağ EE (turuncu)
            right_ee_pos = env._compute_ee_pos()
            right_palm_quat = env.robot.data.body_quat_w[:, env.palm_idx]
            right_ee_pose = torch.cat([right_ee_pos[0], right_palm_quat[0]]).unsqueeze(0)
            env.ee_marker.write_root_pose_to_sim(right_ee_pose)

            # ===== MESAFE HESAPLARI =====
            # Sağ kol
            right_ee_rel = right_ee_pos[0] - root_pos
            right_distance = (right_ee_rel - env.target_pos[0]).norm().item()

            # Sol kol
            left_ee_world = env.robot.data.body_pos_w[0, left_palm_idx]
            left_ee_rel = left_ee_world - root_pos
            left_distance = (left_ee_rel - left_target_pos).norm().item()

            # ===== REACH DETECTION =====
            # Sağ kol
            if rewards[0].item() > 15:
                right_reach_count += 1
                right_episode_steps = 0

            # Sol kol
            if left_distance < 0.05:
                left_reach_count += 1
                # Yeni sol target spawn
                left_target_pos = torch.tensor([
                    torch.empty(1).uniform_(0.05, 0.30).item(),
                    torch.empty(1).uniform_(-0.35, -0.10).item(),  # Sol taraf (negatif Y)
                    torch.empty(1).uniform_(-0.15, 0.25).item()
                ], device="cuda:0")
                left_episode_steps = 0

            # Log
            if step % 50 == 0:
                print(f"[Step {step:4d}]")
                print(f"  SAĞ KOL:  Dist={right_distance:.3f}m | Reaches={right_reach_count}")
                print(f"  SOL KOL:  Dist={left_distance:.3f}m | Reaches={left_reach_count}")
                print(f"  TOPLAM:   {right_reach_count + left_reach_count} reaches")
                print()

    except KeyboardInterrupt:
        print("\n\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print("DUAL ARM SONUÇLAR")
    print("=" * 70)
    print(f"  Toplam step:      {step}")
    print(f"  Sağ kol reaches:  {right_reach_count}")
    print(f"  Sol kol reaches:  {left_reach_count}")
    print(f"  TOPLAM reaches:   {right_reach_count + left_reach_count}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()