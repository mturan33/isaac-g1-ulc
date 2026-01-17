"""
G1 DUAL ARM PLAY - 2 Bağımsız Policy
=====================================

Her kol için ayrı policy instance kullanır.
Sol kol için sağ kol policy'si mirror edilir.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_4_arm_dual.py
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

env_path = "source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/envs"
if os.path.exists(env_path):
    sys.path.insert(0, env_path)

from g1_arm_dual_env import G1DualArmEnv, G1DualArmEnvCfg


class SimpleActor(nn.Module):
    """RSL-RL actor network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ELU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_policy(checkpoint_path: str, device: str = "cuda:0"):
    """Load policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint)

    actor_keys = sorted([k for k in model_state.keys() if k.startswith('actor.') and 'weight' in k])

    obs_dim = model_state[actor_keys[0]].shape[1]
    act_dim = model_state[actor_keys[-1]].shape[0]
    hidden_dims = [model_state[k].shape[0] for k in actor_keys[:-1]]

    actor = SimpleActor(obs_dim, act_dim, hidden_dims).to(device)
    actor_state = {k.replace('actor.', 'net.'): v for k, v in model_state.items() if k.startswith('actor.')}
    actor.load_state_dict(actor_state)
    actor.eval()

    return actor, obs_dim, act_dim, hidden_dims


def build_right_obs(env, idx: int = 0) -> torch.Tensor:
    """Build observation for RIGHT arm (matches training format)."""
    root_pos = env.robot.data.root_pos_w[idx]

    joint_pos = env.robot.data.joint_pos[idx, env.right_arm_indices]
    joint_vel = env.robot.data.joint_vel[idx, env.right_arm_indices]
    ee_pos = env._compute_right_ee_pos()[idx] - root_pos
    target = env.right_target_pos[idx]
    error = target - ee_pos
    error_norm = error / 0.31

    obs = torch.cat([
        joint_pos,          # 5
        joint_vel * 0.1,    # 5
        target,             # 3
        ee_pos,             # 3
        error_norm,         # 3
    ])  # Total: 19

    return obs.unsqueeze(0)


def build_left_obs_mirrored(env, idx: int = 0) -> torch.Tensor:
    """Build MIRRORED observation for LEFT arm (so right policy can control it)."""
    root_pos = env.robot.data.root_pos_w[idx]

    joint_pos = env.robot.data.joint_pos[idx, env.left_arm_indices]
    joint_vel = env.robot.data.joint_vel[idx, env.left_arm_indices]
    ee_pos = env._compute_left_ee_pos()[idx] - root_pos
    target = env.left_target_pos[idx]

    # Mirror joints: roll ve yaw ters (index 1, 2, 4)
    joint_pos_m = joint_pos.clone()
    joint_pos_m[1] = -joint_pos_m[1]  # shoulder_roll
    joint_pos_m[2] = -joint_pos_m[2]  # shoulder_yaw
    joint_pos_m[4] = -joint_pos_m[4]  # elbow_roll

    joint_vel_m = joint_vel.clone()
    joint_vel_m[1] = -joint_vel_m[1]
    joint_vel_m[2] = -joint_vel_m[2]
    joint_vel_m[4] = -joint_vel_m[4]

    # Mirror positions: Y ters
    ee_pos_m = ee_pos.clone()
    ee_pos_m[1] = -ee_pos_m[1]

    target_m = target.clone()
    target_m[1] = -target_m[1]

    error = target_m - ee_pos_m
    error_norm = error / 0.31

    obs = torch.cat([
        joint_pos_m,        # 5
        joint_vel_m * 0.1,  # 5
        target_m,           # 3
        ee_pos_m,           # 3
        error_norm,         # 3
    ])  # Total: 19

    return obs.unsqueeze(0)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror actions from right to left (roll ve yaw ters)."""
    actions_m = actions.clone()
    actions_m[1] = -actions_m[1]  # shoulder_roll
    actions_m[2] = -actions_m[2]  # shoulder_yaw
    actions_m[4] = -actions_m[4]  # elbow_roll
    return actions_m


def main():
    print("\n" + "=" * 70)
    print("   G1 DUAL ARM PLAY - 2 Bağımsız Policy")
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

    print(f"[INFO] Checkpoint: {checkpoint_path}")

    # Load 2 policy instances (same weights, independent inference)
    right_policy, obs_dim, act_dim, hidden = load_policy(checkpoint_path)
    left_policy, _, _, _ = load_policy(checkpoint_path)

    print(f"[INFO] Policy: obs={obs_dim}, act={act_dim}, hidden={hidden}")
    print("[INFO] 2 bağımsız policy yüklendi (sağ ve sol kol için)")

    # Create environment
    env_cfg = G1DualArmEnvCfg()
    env_cfg.scene.num_envs = 1
    env = G1DualArmEnv(cfg=env_cfg)

    print("\n" + "-" * 70)
    print("MARKERS: 🟢Yeşil=Sağ target | 🔵Mavi=Sol target")
    print("         🟠Turuncu=Sağ EE   | 🟣Mor=Sol EE")
    print("-" * 70)
    print("[INFO] Ctrl+C ile çık\n")

    # Reset
    obs_dict, _ = env.reset()
    step = 0

    try:
        while simulation_app.is_running():
            step += 1

            # ===== RIGHT ARM =====
            right_obs = build_right_obs(env)
            with torch.no_grad():
                right_actions = right_policy(right_obs)[0]

            # ===== LEFT ARM (mirrored) =====
            left_obs = build_left_obs_mirrored(env)
            with torch.no_grad():
                left_actions_raw = left_policy(left_obs)[0]

            # Mirror actions back
            left_actions = mirror_actions(left_actions_raw)

            # ===== COMBINE & STEP =====
            combined = torch.cat([right_actions, left_actions]).unsqueeze(0)
            obs_dict, rewards, terminated, truncated, info = env.step(combined)

            # ===== DEBUG: Print actions every 100 steps =====
            if step % 100 == 0:
                root_pos = env.robot.data.root_pos_w[0]
                r_ee = env._compute_right_ee_pos()[0] - root_pos
                l_ee = env._compute_left_ee_pos()[0] - root_pos
                r_dist = (r_ee - env.right_target_pos[0]).norm().item()
                l_dist = (l_ee - env.left_target_pos[0]).norm().item()

                print(f"[Step {step:4d}]")
                print(f"  SAĞ:  Dist={r_dist:.3f}m | Reaches={int(env.right_reach_count[0].item())}")
                print(f"  SOL:  Dist={l_dist:.3f}m | Reaches={int(env.left_reach_count[0].item())}")
                print(f"  Actions R: [{right_actions[0]:.2f}, {right_actions[1]:.2f}, {right_actions[2]:.2f}, {right_actions[3]:.2f}, {right_actions[4]:.2f}]")
                print(f"  Actions L: [{left_actions[0]:.2f}, {left_actions[1]:.2f}, {left_actions[2]:.2f}, {left_actions[3]:.2f}, {left_actions[4]:.2f}]")
                total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
                print(f"  TOPLAM: {total} reaches\n")

    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu")

    print("\n" + "=" * 70)
    print("SONUÇLAR")
    print("=" * 70)
    print(f"  Toplam step:  {step}")
    print(f"  Sağ reaches:  {int(env.right_reach_count[0].item())}")
    print(f"  Sol reaches:  {int(env.left_reach_count[0].item())}")
    total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
    print(f"  TOPLAM:       {total}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()