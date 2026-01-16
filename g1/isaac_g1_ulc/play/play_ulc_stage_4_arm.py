"""
G1 Arm Reach - Play/Debug Script (Stage 4)
==========================================

Test modları:
  --forward  : Kolu düz ileri uzat (manuel test)
  --wave     : Kolu salla (joint test)
  --circle   : Dairesel hareket (workspace test)
  (default)  : Eğitilmiş policy kullan

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p play_ulc_stage_4_arm_debug.py --forward --num_envs 1
./isaaclab.bat -p play_ulc_stage_4_arm_debug.py --wave --num_envs 1
./isaaclab.bat -p play_ulc_stage_4_arm_debug.py --checkpoint logs/ulc/ulc_g1_stage4_arm_2026-01-15_23-24-32/model_15999.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(description="G1 Arm Reach Debug - Stage 4")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--duration", type=float, default=60.0, help="Play duration in seconds")

# Test modları
parser.add_argument("--forward", action="store_true", help="Manuel: Kolu düz ileri uzat")
parser.add_argument("--wave", action="store_true", help="Manuel: Kolu salla (joint test)")
parser.add_argument("--circle", action="store_true", help="Manuel: Dairesel hareket")
parser.add_argument("--home", action="store_true", help="Manuel: Home pozisyonuna git")

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
    """Simple actor network matching RSL-RL architecture."""

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


class ManualController:
    """Manuel test kontrolleri."""

    def __init__(self, num_envs: int, device: str = "cuda:0"):
        self.num_envs = num_envs
        self.device = device
        self.step_count = 0

        # Joint isimleri (referans için)
        # 0: right_shoulder_pitch  (-2.97, 2.79)  - ileri/geri
        # 1: right_shoulder_roll   (-2.25, 1.59)  - yana açma
        # 2: right_shoulder_yaw    (-2.62, 2.62)  - döndürme
        # 3: right_elbow_pitch     (-0.23, 3.42)  - dirsek bükme
        # 4: right_elbow_roll      (-2.09, 2.09)  - bilek döndürme

        # Home pozisyon (default)
        self.home_pos = torch.tensor([
            -0.3,  # shoulder_pitch
            0.0,   # shoulder_roll
            0.0,   # shoulder_yaw
            0.5,   # elbow_pitch
            0.0,   # elbow_roll
        ], device=device)

        # İleri uzatma pozisyonu
        self.forward_pos = torch.tensor([
            0.8,   # shoulder_pitch - ileri
            0.0,   # shoulder_roll - düz
            0.0,   # shoulder_yaw - düz
            0.1,   # elbow_pitch - düz (bükülmemiş)
            0.0,   # elbow_roll
        ], device=device)

    def get_forward_action(self) -> torch.Tensor:
        """Kolu yavaşça ileri uzat."""
        self.step_count += 1

        # Yavaş geçiş (ilk 100 step'te home'dan forward'a)
        t = min(self.step_count / 100.0, 1.0)
        target = self.home_pos + t * (self.forward_pos - self.home_pos)

        # Action = delta, yani hedef - current farkı gibi davran
        # Ama biz direkt küçük adımlar atalım
        if t < 1.0:
            # İleri git
            action = torch.tensor([0.5, 0.0, 0.0, -0.3, 0.0], device=self.device)
        else:
            # Yerinde kal
            action = torch.zeros(5, device=self.device)

        return action.unsqueeze(0).expand(self.num_envs, -1)

    def get_wave_action(self) -> torch.Tensor:
        """Kolu salla - joint'leri test et."""
        self.step_count += 1

        # Sinüsoidal hareket
        freq = 0.05  # Yavaş sallanma
        t = self.step_count * freq

        action = torch.tensor([
            0.3 * math.sin(t),           # shoulder_pitch - ileri/geri
            0.2 * math.sin(t * 0.7),     # shoulder_roll - yana
            0.1 * math.sin(t * 1.3),     # shoulder_yaw - döndür
            0.2 * math.sin(t * 0.5),     # elbow_pitch - dirsek
            0.1 * math.sin(t * 0.9),     # elbow_roll
        ], device=self.device)

        return action.unsqueeze(0).expand(self.num_envs, -1)

    def get_circle_action(self) -> torch.Tensor:
        """Dairesel hareket - workspace test."""
        self.step_count += 1

        freq = 0.03
        t = self.step_count * freq

        # Omuz ile daire çiz
        action = torch.tensor([
            0.3 * math.sin(t),       # shoulder_pitch
            0.3 * math.cos(t),       # shoulder_roll
            0.0,                      # shoulder_yaw
            0.1 * math.sin(t * 2),   # elbow
            0.0,
        ], device=self.device)

        return action.unsqueeze(0).expand(self.num_envs, -1)

    def get_home_action(self) -> torch.Tensor:
        """Home pozisyonuna dön."""
        return torch.zeros(5, device=self.device).unsqueeze(0).expand(self.num_envs, -1)


def main():
    # Test modu belirleme
    manual_mode = None
    if args.forward:
        manual_mode = "forward"
    elif args.wave:
        manual_mode = "wave"
    elif args.circle:
        manual_mode = "circle"
    elif args.home:
        manual_mode = "home"

    if manual_mode is None and args.checkpoint is None:
        print("\n" + "=" * 70)
        print("HATA: Ya --checkpoint ya da test modu (--forward, --wave, --circle) belirt!")
        print("=" * 70)
        print("\nKullanım örnekleri:")
        print("  --forward   : Kolu düz ileri uzat")
        print("  --wave      : Kolu salla (joint test)")
        print("  --circle    : Dairesel hareket")
        print("  --checkpoint <path> : Eğitilmiş policy")
        print("=" * 70 + "\n")
        simulation_app.close()
        return

    # Environment oluştur
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.duration
    env_cfg.initial_target_radius = env_cfg.max_target_radius

    env = G1ArmReachEnv(cfg=env_cfg)

    # Controller seç
    actor = None
    manual_ctrl = None
    obs_dim = 19  # default

    if manual_mode:
        manual_ctrl = ManualController(args.num_envs, "cuda:0")
        print(f"\n[INFO] Manuel mod: {manual_mode.upper()}")
        print("[INFO] Robot kolunu manuel kontrol ediyorum...")

        if manual_mode == "forward":
            print("[INFO] Kol düz ileri uzatılacak (shoulder_pitch=0.8, elbow_pitch=0.1)")
        elif manual_mode == "wave":
            print("[INFO] Kol sinüsoidal olarak sallanacak")
        elif manual_mode == "circle":
            print("[INFO] Kol dairesel hareket yapacak")
    else:
        # Checkpoint yükle
        print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)

        model_state = checkpoint.get('model_state_dict', checkpoint)

        actor_keys = [k for k in model_state.keys() if k.startswith('actor.') and 'weight' in k]
        actor_keys.sort()

        if actor_keys:
            first_weight = model_state[actor_keys[0]]
            last_weight = model_state[actor_keys[-1]]
            obs_dim = first_weight.shape[1]
            act_dim = last_weight.shape[0]
            print(f"[INFO] Detected: obs_dim={obs_dim}, act_dim={act_dim}")
        else:
            obs_dim = 19
            act_dim = 5

        hidden_dims = []
        for key in actor_keys[:-1]:
            hidden_dims.append(model_state[key].shape[0])

        if not hidden_dims:
            hidden_dims = [256, 128, 64]

        actor = SimpleActor(obs_dim, act_dim, hidden_dims).to("cuda:0")

        actor_state = {}
        for key, value in model_state.items():
            if key.startswith('actor.'):
                new_key = key.replace('actor.', 'net.')
                actor_state[new_key] = value

        actor.load_state_dict(actor_state)
        actor.eval()
        print("[INFO] Policy loaded!")

    print(f"\n[INFO] Running for {args.duration} seconds...")
    print("[INFO] Press Ctrl+C to exit\n")

    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict[0]
    env_obs_dim = obs.shape[-1]

    step_count = 0
    total_reward = 0.0
    reach_count = 0

    try:
        while simulation_app.is_running():
            # Action seç
            if manual_ctrl:
                if manual_mode == "forward":
                    actions = manual_ctrl.get_forward_action()
                elif manual_mode == "wave":
                    actions = manual_ctrl.get_wave_action()
                elif manual_mode == "circle":
                    actions = manual_ctrl.get_circle_action()
                else:
                    actions = manual_ctrl.get_home_action()
            else:
                # Policy kullan
                if obs.shape[-1] < obs_dim:
                    pad_size = obs_dim - obs.shape[-1]
                    obs = torch.cat([obs, torch.zeros(obs.shape[0], pad_size, device=obs.device)], dim=-1)
                elif obs.shape[-1] > obs_dim:
                    obs = obs[:, :obs_dim]

                with torch.no_grad():
                    actions = actor(obs)

            # Step
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

            total_reward += rewards.mean().item()
            step_count += 1

            if rewards.max().item() > 15:
                reach_count += 1

            # Detaylı log (her 50 step)
            if step_count % 50 == 0:
                # Joint pozisyonlarını al
                arm_joint_pos = env.robot.data.joint_pos[:, env.arm_joint_indices]
                ee_pos = env._compute_ee_pos()
                target_pos = env.target_pos + env.robot.data.root_pos_w

                print(f"[Step {step_count:5d}]")
                print(f"  Joint pos: [{arm_joint_pos[0, 0]:.2f}, {arm_joint_pos[0, 1]:.2f}, "
                      f"{arm_joint_pos[0, 2]:.2f}, {arm_joint_pos[0, 3]:.2f}, {arm_joint_pos[0, 4]:.2f}]")
                print(f"  EE pos:    [{ee_pos[0, 0]:.3f}, {ee_pos[0, 1]:.3f}, {ee_pos[0, 2]:.3f}]")
                print(f"  Target:    [{target_pos[0, 0]:.3f}, {target_pos[0, 1]:.3f}, {target_pos[0, 2]:.3f}]")
                print(f"  Reward: {rewards[0].item():+.3f} | Reaches: {reach_count}")
                print()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"  Mode:              {manual_mode or 'policy'}")
    print(f"  Total steps:       {step_count}")
    print(f"  Average reward:    {total_reward / max(step_count, 1):+.4f}")
    print(f"  Total reaches:     {reach_count}")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()