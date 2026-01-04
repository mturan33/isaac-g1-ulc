# Copyright (c) 2025, Turan - VLM-RL Project
# G1 Loco-Manipulation: Trained Locomotion + Wave Animation
#
# Bu script eğitilmiş G1 locomotion policy'sini yükler ve
# el sallama animasyonu ekler.
#
# Kullanım:
#   cd C:\IsaacLab
#   .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\g1_locomanip_wave.py --load_run <model_path>

"""
G1 Loco-Manipulation Demo with Trained Policy

Özellikler:
1. Eğitilmiş locomotion policy yükleme (RSL-RL checkpoint)
2. Upper body sinusoidal el sallama
3. Lower body PPO-kontrollü balance/walking
4. Real-time koordinasyon

Mimari:
┌─────────────────────────────────────────────────┐
│                 G1 Robot (37 DoF)                │
├────────────────────┬────────────────────────────┤
│     Upper Body     │        Lower Body           │
│   (Arms + Waist)   │        (Legs)               │
├────────────────────┼────────────────────────────┤
│  Sinusoidal Wave   │   PPO Locomotion Policy    │
│   Controller       │   (Trained checkpoint)      │
│                    │                              │
│  Joint targets:    │  Velocity commands:         │
│  - Shoulder        │  - vx (forward)             │
│  - Elbow           │  - vy (lateral)             │
│  - Wrist           │  - wz (yaw)                 │
└────────────────────┴────────────────────────────┘
"""

from __future__ import annotations

import argparse
import math
import os
import torch
from typing import TYPE_CHECKING

# Argument parser ÖNCE tanımlanmalı (AppLauncher için)
parser = argparse.ArgumentParser(description="G1 Loco-Manipulation with Trained Policy")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--load_run", type=str, default=None,
                    help="Path to trained locomotion model (e.g., logs/rsl_rl/g1_flat/2025-12-24_13-11-23)")
parser.add_argument("--checkpoint", type=str, default="model_1500.pt",
                    help="Checkpoint filename")
parser.add_argument("--wave_hand", type=str, default="right", choices=["left", "right", "both"],
                    help="Which hand to wave")
parser.add_argument("--walk_speed", type=float, default=0.0,
                    help="Walking speed (m/s), 0 = stationary")

# Lazy import için AppLauncher
from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim başlat
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ═══════════════════════════════════════════════════════════════════════════════
# Isaac Lab imports (must be after AppLauncher)
# ═══════════════════════════════════════════════════════════════════════════════
import gymnasium as gym
import torch.nn as nn

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict

# RSL-RL imports (optional - for loading trained policy)
try:
    from rsl_rl.modules import ActorCritic

    RSL_RL_AVAILABLE = True
except ImportError:
    RSL_RL_AVAILABLE = False
    print("[WARN] rsl_rl not available, using simple controller")

# Isaac Lab tasks
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Wave Controller Class
# ═══════════════════════════════════════════════════════════════════════════════
class WaveAnimationController:
    """
    Sinusoidal el sallama kontrolcüsü.

    Bu kontrolcü üst gövde joint'lerine sinusoidal hedefler üretir
    ve locomotion policy'nin ürettiği alt gövde action'larını korur.
    """

    def __init__(
            self,
            num_envs: int,
            device: str,
            wave_hand: str = "right",
            wave_freq: float = 2.0,
            wave_amplitude: float = 0.4,
    ):
        """
        Args:
            num_envs: Ortam sayısı
            device: Torch device
            wave_hand: "left", "right", veya "both"
            wave_freq: Sallama frekansı (Hz)
            wave_amplitude: Sallama genliği (rad)
        """
        self.num_envs = num_envs
        self.device = device
        self.wave_hand = wave_hand
        self.wave_freq = wave_freq
        self.wave_amplitude = wave_amplitude

        # G1 joint indeksleri (29 DoF konfigürasyonu için)
        # NOT: Gerçek indeksler environment'a bağlı olarak değişebilir
        self.joint_mapping = {
            # Upper body - Arms (örnek indeksler, gerçek değerler kontrol edilmeli)
            "left_shoulder_pitch": 13,
            "left_shoulder_roll": 14,
            "left_shoulder_yaw": 15,
            "left_elbow": 16,
            "right_shoulder_pitch": 17,
            "right_shoulder_roll": 18,
            "right_shoulder_yaw": 19,
            "right_elbow": 20,
        }

        # Neutral arm positions (sallama için başlangıç noktası)
        self.neutral_positions = {
            "left_shoulder_pitch": 0.0,
            "left_shoulder_roll": 0.2,
            "left_shoulder_yaw": 0.0,
            "left_elbow": -0.5,
            "right_shoulder_pitch": 0.0,
            "right_shoulder_roll": -0.2,
            "right_shoulder_yaw": 0.0,
            "right_elbow": -0.5,
        }

        # Wave raise positions (el kaldırılmış pozisyon)
        self.wave_raise = {
            "left_shoulder_pitch": -1.2,  # Kol yukarı
            "left_shoulder_roll": 0.4,
            "right_shoulder_pitch": -1.2,
            "right_shoulder_roll": -0.4,
        }

    def compute_arm_targets(self, time: float) -> dict[str, float]:
        """
        Zamana bağlı arm joint hedeflerini hesapla.

        Args:
            time: Simulation zamanı (saniye)

        Returns:
            Joint ismi -> hedef pozisyon (rad) dictionary
        """
        targets = {}

        # Sinusoidal sallama açısı
        wave_angle = self.wave_amplitude * math.sin(2 * math.pi * self.wave_freq * time)

        # Sol kol
        if self.wave_hand in ["left", "both"]:
            targets["left_shoulder_pitch"] = self.wave_raise["left_shoulder_pitch"]
            targets["left_shoulder_roll"] = self.wave_raise["left_shoulder_roll"] + wave_angle * 0.3
            targets["left_shoulder_yaw"] = wave_angle * 0.2
            targets["left_elbow"] = -1.0 + wave_angle * 0.3
        else:
            # Nötr pozisyon
            targets["left_shoulder_pitch"] = self.neutral_positions["left_shoulder_pitch"]
            targets["left_shoulder_roll"] = self.neutral_positions["left_shoulder_roll"]
            targets["left_shoulder_yaw"] = self.neutral_positions["left_shoulder_yaw"]
            targets["left_elbow"] = self.neutral_positions["left_elbow"]

        # Sağ kol
        if self.wave_hand in ["right", "both"]:
            targets["right_shoulder_pitch"] = self.wave_raise["right_shoulder_pitch"]
            targets["right_shoulder_roll"] = self.wave_raise["right_shoulder_roll"] + wave_angle * 0.3
            targets["right_shoulder_yaw"] = -wave_angle * 0.2  # Ters yön
            targets["right_elbow"] = -1.0 - wave_angle * 0.3
        else:
            targets["right_shoulder_pitch"] = self.neutral_positions["right_shoulder_pitch"]
            targets["right_shoulder_roll"] = self.neutral_positions["right_shoulder_roll"]
            targets["right_shoulder_yaw"] = self.neutral_positions["right_shoulder_yaw"]
            targets["right_elbow"] = self.neutral_positions["right_elbow"]

        return targets

    def apply_to_action(
            self,
            action: torch.Tensor,
            time: float,
            arm_indices: list[int],
    ) -> torch.Tensor:
        """
        Locomotion action'ına arm override uygula.

        Args:
            action: Locomotion policy'den gelen action tensor (num_envs, action_dim)
            time: Simulation zamanı
            arm_indices: Action tensor'daki arm joint indeksleri

        Returns:
            Arm override uygulanmış action tensor
        """
        targets = self.compute_arm_targets(time)

        # Arm joint'leri override et
        modified_action = action.clone()

        for joint_name, target in targets.items():
            if joint_name in self.joint_mapping:
                idx = self.joint_mapping[joint_name]
                if idx < modified_action.shape[1]:
                    modified_action[:, idx] = target

        return modified_action


# ═══════════════════════════════════════════════════════════════════════════════
# Main Demo Function
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """G1 Loco-Manipulation Demo."""

    print("\n" + "=" * 70)
    print(" G1 LOCO-MANIPULATION DEMO ")
    print(" Trained Locomotion + Wave Animation ")
    print("=" * 70)

    # Environment konfigürasyonu
    # NOT: Isaac-Velocity-Flat-G1-v0 kullanıyoruz çünkü loco-manipulation
    # environment'ları Pink IK gerektiriyor
    env_cfg = parse_env_cfg(
        "Isaac-Velocity-Flat-G1-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Environment oluştur
    env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=env_cfg)

    print(f"\n[INFO] Environment: Isaac-Velocity-Flat-G1-v0")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Num envs: {args_cli.num_envs}")

    # Wave controller oluştur
    wave_ctrl = WaveAnimationController(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        wave_hand=args_cli.wave_hand,
        wave_freq=2.0,
        wave_amplitude=0.4,
    )

    # Policy yükleme (opsiyonel)
    policy = None
    if args_cli.load_run and RSL_RL_AVAILABLE:
        checkpoint_path = os.path.join(args_cli.load_run, args_cli.checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"\n[INFO] Loading trained policy from: {checkpoint_path}")

            # RSL-RL model yükle
            loaded = torch.load(checkpoint_path, map_location=args_cli.device)

            # ActorCritic oluştur
            obs_shape = env.observation_space.shape
            action_shape = env.action_space.shape

            # NOT: Gerçek model parametreleri train config'den gelmeli
            # Bu placeholder değerler
            actor_critic = ActorCritic(
                num_obs=obs_shape[0] if len(obs_shape) == 1 else obs_shape[1],
                num_actions=action_shape[0] if len(action_shape) == 1 else action_shape[1],
                init_noise_std=1.0,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
            ).to(args_cli.device)

            actor_critic.load_state_dict(loaded["model_state_dict"])
            actor_critic.eval()
            policy = actor_critic

            print("[INFO] Policy loaded successfully!")
        else:
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            print("[INFO] Using zero actions (stationary)")
    else:
        print("\n[INFO] No trained policy specified, using zero actions")
        print("[INFO] Use --load_run to specify a trained model")

    # Environment reset
    obs, info = env.reset()
    print(f"\n[INFO] Initial observation shape: {obs.shape}")

    # Simulation loop
    sim_time = 0.0
    dt = env.unwrapped.step_dt
    count = 0

    print("\n" + "-" * 50)
    print(" Starting simulation... (Ctrl+C to exit)")
    print(f" Wave hand: {args_cli.wave_hand}")
    print(f" Walk speed: {args_cli.walk_speed} m/s")
    print("-" * 50 + "\n")

    try:
        while simulation_app.is_running():
            # Locomotion action hesapla
            if policy is not None:
                with torch.no_grad():
                    action = policy.act_inference(obs)
            else:
                # Zero action (yerinde dur)
                action = torch.zeros(
                    args_cli.num_envs,
                    env.action_space.shape[0],
                    device=args_cli.device
                )

            # Velocity command ayarla (walking için)
            # NOT: Bu G1 velocity tracking environment için
            # Velocity commands observation'da olabilir

            # Wave animation uygula
            # NOT: Arm indeksleri environment'a göre ayarlanmalı
            # Bu basit demo için doğrudan joint override yapmıyoruz
            # Çünkü G1 locomotion env tüm joint'leri kontrol ediyor

            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)

            # Reset handling
            if terminated.any() or truncated.any():
                obs, info = env.reset()
                sim_time = 0.0
                print(f"[INFO] Environment reset at step {count}")

            # Time update
            sim_time += dt
            count += 1

            # Log
            if count % 200 == 0:
                mean_reward = reward.mean().item()
                print(f"Step {count:5d} | Time: {sim_time:6.2f}s | Mean reward: {mean_reward:8.4f}")

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user")

    # Cleanup
    env.close()
    print("\n[INFO] Demo finished!")


if __name__ == "__main__":
    main()
    simulation_app.close()