# Copyright (c) 2025, Turan - VLM-RL Project
# G1 Loco-Manipulation Demo: Wave while Standing
# Isaac Lab 2.3.1 Compatible - v4 (All bugs fixed!)
#
# Kullanım:
#   cd C:\IsaacLab
#   .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\g1_wave_demo.py --num_envs 4

"""
G1 Wave Demo - El sallama + Ayakta durma demonstrasyonu.
Pink IK / Pinocchio GEREKTIRMEZ!
"""

from __future__ import annotations

import argparse
import math
import os
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# CLI Arguments - MUST be before AppLauncher
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="G1 Wave Demo")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--wave_hand", type=str, default="right", choices=["left", "right", "both"])
parser.add_argument("--wave_freq", type=float, default=2.0, help="Wave frequency (Hz)")
parser.add_argument("--load_run", type=str, default=None, help="Trained model directory name")

# AppLauncher import
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim başlat
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ═══════════════════════════════════════════════════════════════════════════════
# Isaac Lab imports (Isaac Sim başladıktan sonra)
# ═══════════════════════════════════════════════════════════════════════════════
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

# Tasks import
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Wave Controller
# ═══════════════════════════════════════════════════════════════════════════════
class ArmWaveController:
    """
    Sinusoidal el sallama kontrolcüsü.

    G1 robot joint indeksleri (37 DoF action space):
    - 0-11: Legs (hip, knee, ankle x2)
    - 12: Torso
    - 13-16: Left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
    - 17-20: Right arm
    - 21-36: Hands/fingers
    """

    # G1 arm joint indices in action space
    ARM_INDICES = {
        "left_shoulder_pitch": 13,
        "left_shoulder_roll": 14,
        "left_shoulder_yaw": 15,
        "left_elbow": 16,
        "right_shoulder_pitch": 17,
        "right_shoulder_roll": 18,
        "right_shoulder_yaw": 19,
        "right_elbow": 20,
    }

    def __init__(
            self,
            num_envs: int,
            device: str,
            action_dim: int,
            wave_hand: str = "right",
            wave_freq: float = 2.0,
            wave_amplitude: float = 0.4,
    ):
        self.num_envs = num_envs
        self.device = device
        self.action_dim = action_dim
        self.wave_hand = wave_hand
        self.wave_freq = wave_freq
        self.wave_amplitude = wave_amplitude

    def compute_arm_override(self, time: float) -> dict[int, float]:
        """Zamana bağlı arm joint override değerlerini hesapla."""
        wave = self.wave_amplitude * math.sin(2 * math.pi * self.wave_freq * time)

        overrides = {}

        # Sol kol
        if self.wave_hand in ["left", "both"]:
            overrides[self.ARM_INDICES["left_shoulder_pitch"]] = -1.2
            overrides[self.ARM_INDICES["left_shoulder_roll"]] = 0.3 + wave * 0.4
            overrides[self.ARM_INDICES["left_shoulder_yaw"]] = wave * 0.2
            overrides[self.ARM_INDICES["left_elbow"]] = -0.8 + wave * 0.3
        else:
            overrides[self.ARM_INDICES["left_shoulder_pitch"]] = 0.0
            overrides[self.ARM_INDICES["left_shoulder_roll"]] = 0.1
            overrides[self.ARM_INDICES["left_shoulder_yaw"]] = 0.0
            overrides[self.ARM_INDICES["left_elbow"]] = -0.3

        # Sağ kol
        if self.wave_hand in ["right", "both"]:
            overrides[self.ARM_INDICES["right_shoulder_pitch"]] = -1.2
            overrides[self.ARM_INDICES["right_shoulder_roll"]] = -0.3 + wave * 0.4
            overrides[self.ARM_INDICES["right_shoulder_yaw"]] = -wave * 0.2
            overrides[self.ARM_INDICES["right_elbow"]] = -0.8 - wave * 0.3
        else:
            overrides[self.ARM_INDICES["right_shoulder_pitch"]] = 0.0
            overrides[self.ARM_INDICES["right_shoulder_roll"]] = -0.1
            overrides[self.ARM_INDICES["right_shoulder_yaw"]] = 0.0
            overrides[self.ARM_INDICES["right_elbow"]] = -0.3

        return overrides

    def get_zero_action_with_wave(self, time: float) -> torch.Tensor:
        """Wave animasyonu ile zero action oluştur."""
        # Doğru boyutta tensor: (num_envs, action_dim)
        action = torch.zeros(
            self.num_envs,
            self.action_dim,
            dtype=torch.float32,
            device=self.device
        )

        # Arm override uygula
        overrides = self.compute_arm_override(time)
        for idx, value in overrides.items():
            if idx < self.action_dim:
                action[:, idx] = value

        return action

    def apply_override(self, action: torch.Tensor, time: float) -> torch.Tensor:
        """Mevcut action'a arm override uygula."""
        overrides = self.compute_arm_override(time)
        modified = action.clone()

        for idx, value in overrides.items():
            if idx < modified.shape[-1]:
                modified[:, idx] = value

        return modified


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """G1 Wave Demo ana fonksiyonu."""

    print("\n" + "=" * 70)
    print("  G1 LOCO-MANIPULATION DEMO: Wave While Standing")
    print("  Pink IK / Pinocchio NOT Required!")
    print("=" * 70 + "\n")

    # Environment config
    env_cfg = parse_env_cfg(
        "Isaac-Velocity-Flat-G1-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Environment oluştur
    env = gym.make("Isaac-Velocity-Flat-G1-v0", cfg=env_cfg)

    # Action dimension'ı DOĞRU şekilde al
    # env.unwrapped üzerinden action_manager'a eriş
    action_dim = env.unwrapped.action_manager.total_action_dim

    print(f"[INFO] Environment: Isaac-Velocity-Flat-G1-v0")
    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Action dim: {action_dim}")  # Şimdi 37 olacak
    print(f"[INFO] Wave hand: {args_cli.wave_hand}")
    print(f"[INFO] Device: {args_cli.device}")

    # Wave controller
    wave_ctrl = ArmWaveController(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        action_dim=action_dim,
        wave_hand=args_cli.wave_hand,
        wave_freq=args_cli.wave_freq,
    )

    # Policy yükle (opsiyonel) - Isaac Lab'ın kendi yöntemini kullan
    policy = None
    if args_cli.load_run:
        try:
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
            from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            # Runner config yükle
            model_dir = os.path.join("logs/rsl_rl/g1_flat", args_cli.load_run)
            if os.path.exists(model_dir):
                print(f"\n[INFO] Model directory found: {model_dir}")

                # En son checkpoint'ı bul
                checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pt")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                    checkpoint_path = os.path.join(model_dir, latest)
                    print(f"[INFO] Found checkpoint: {latest}")

                    # Doğrudan model weights yükle
                    loaded = torch.load(checkpoint_path, map_location=args_cli.device)

                    # Basit MLP policy oluştur
                    from torch import nn

                    class SimplePolicy(nn.Module):
                        def __init__(self, obs_dim, act_dim):
                            super().__init__()
                            self.actor = nn.Sequential(
                                nn.Linear(obs_dim, 256),
                                nn.ELU(),
                                nn.Linear(256, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, act_dim),
                            )

                        def forward(self, obs):
                            return self.actor(obs)

                    # Model oluştur ve weights yükle
                    policy = SimplePolicy(123, action_dim).to(args_cli.device)

                    # Actor weights'i ayıkla
                    actor_state = {}
                    model_state = loaded.get("model_state_dict", loaded)
                    for key, value in model_state.items():
                        if key.startswith("actor."):
                            new_key = key.replace("actor.", "actor.")
                            actor_state[new_key] = value

                    if actor_state:
                        policy.load_state_dict(actor_state, strict=False)
                        policy.eval()
                        print("[INFO] Policy loaded successfully!")
                    else:
                        print("[WARN] Could not extract actor weights")
                        policy = None
                else:
                    print("[WARN] No checkpoints found")
            else:
                print(f"[WARN] Model directory not found: {model_dir}")

        except Exception as e:
            print(f"[WARN] Could not load policy: {e}")
            print("[INFO] Continuing without trained policy...")
            policy = None

    if policy is None:
        print("\n[INFO] No policy loaded - using zero actions with wave animation")
        print("[INFO] Robot will try to maintain default pose while waving")
        print("[INFO] Use --load_run <dir_name> to load trained policy")

    # Reset
    obs, info = env.reset()

    # Simulation loop
    sim_time = 0.0
    dt = env.unwrapped.step_dt
    step_count = 0

    print("\n" + "-" * 50)
    print(" Simulation running... Press Ctrl+C to exit")
    print("-" * 50 + "\n")

    try:
        while simulation_app.is_running():
            # Action hesapla
            if policy is not None:
                with torch.no_grad():
                    # Observation'ı policy formatına çevir
                    if isinstance(obs, dict):
                        obs_tensor = obs.get("policy", list(obs.values())[0])
                    else:
                        obs_tensor = obs
                    action = policy(obs_tensor)
                    # Arm override uygula
                    action = wave_ctrl.apply_override(action, sim_time)
            else:
                # Zero action + wave animation
                action = wave_ctrl.get_zero_action_with_wave(sim_time)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            # Reset handling
            reset_mask = terminated | truncated
            if reset_mask.any():
                n_reset = reset_mask.sum().item()
                if step_count % 100 == 0:
                    print(f"[Step {step_count}] {n_reset} environments reset")

            # Time update
            sim_time += dt
            step_count += 1

            # Periodic log
            if step_count % 500 == 0:
                mean_reward = reward.mean().item()
                print(f"[Step {step_count:5d}] Time: {sim_time:6.2f}s | Reward: {mean_reward:8.4f}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    env.close()
    print("\n[INFO] Demo finished!")


if __name__ == "__main__":
    main()
    simulation_app.close()