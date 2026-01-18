"""
G1 DUAL ARM - GENERAL REACHING POLICY (VIDEO RECORDING)
=========================================================

"General Reaching Policy" - Aynı observation structure her iki kol için kullanılıyor,
böylece policy genel bir reaching davranışı öğrenmiş oluyor.

Isaac Sim'in viewport capture özelliği ile sessiz MP4 kaydı.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_dual_arm_video_record.py

Video kaydedilecek yer: C:\IsaacLab\recordings\
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

parser = argparse.ArgumentParser(description="G1 Dual Arm - Video Recording")
parser.add_argument("--record_duration", type=float, default=20.0, help="Video süresi (saniye)")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

# Import camera view and capture utilities
import isaaclab.sim as sim_utils
import omni.kit.app

# Add env path
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
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrameRecorder:
    """Viewport frame'lerini kaydeder, sonra ffmpeg ile MP4'e çevirir."""

    def __init__(self, output_dir: str, fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0

        # Viewport API
        from omni.kit.viewport.utility import get_active_viewport
        self.viewport = get_active_viewport()

    def capture_frame(self):
        """Mevcut frame'i PNG olarak kaydet."""
        from omni.kit.viewport.utility import capture_viewport_to_file
        frame_path = os.path.join(self.frame_dir, f"frame_{self.frame_count:06d}.png")
        capture_viewport_to_file(self.viewport, frame_path)
        self.frame_count += 1

    def finalize_video(self, output_name: str = "dual_arm_reaching.mp4"):
        """Frame'leri ffmpeg ile MP4'e çevir."""
        import subprocess

        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(self.frame_dir, "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",  # Yüksek kalite
            output_path
        ]

        print(f"\n[VIDEO] Converting {self.frame_count} frames to MP4...")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[VIDEO] ✓ Saved: {output_path}")

            # Frame'leri temizle
            import shutil
            shutil.rmtree(self.frame_dir)
            print(f"[VIDEO] ✓ Cleaned up frames")

            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[VIDEO] ✗ ffmpeg error: {e}")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            return None
        except FileNotFoundError:
            print(f"[VIDEO] ✗ ffmpeg not found! Install ffmpeg or use frames in: {self.frame_dir}")
            return None


def setup_camera_for_video():
    """Kamerayı robotun önüne konumlandır - 3/4 diagonal view."""
    # 3/4 diagonal view - kolların hareketi net görünür
    eye = (-1.4, 0.8, 1.4)  # Önde-sağda, hafif yukarıda
    target = (0.0, 0.0, 1.0)  # Robot gövdesi

    sim_utils.set_camera_view(eye=eye, target=target)
    print(f"[CAMERA] Eye: {eye}, Target: {target}")


def main():
    print("\n" + "=" * 70)
    print("   G1 DUAL ARM - GENERAL REACHING POLICY")
    print("   Video Recording Mode")
    print("=" * 70)
    print(f"   Kayıt süresi: {args.record_duration} saniye")
    print(f"   FPS: {args.fps}")
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
    env_cfg.episode_length_s = args.record_duration + 10.0

    env = G1DualArmEnv(cfg=env_cfg)

    # Setup camera
    setup_camera_for_video()

    # Setup video recorder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = os.path.join(os.getcwd(), "recordings", timestamp)
    os.makedirs(record_dir, exist_ok=True)

    recorder = FrameRecorder(record_dir, fps=args.fps)

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
    print("GENERAL REACHING POLICY:")
    print("  → Aynı observation structure her iki kol için")
    print("  → Policy kol-agnostik reaching öğrenmiş")
    print("  → Sol kol: koordinatlar mirror edilip aynı policy kullanılıyor")
    print("-" * 70)
    print(f"[RECORDING] {args.record_duration} saniye kayıt başlıyor...")
    print("-" * 70 + "\n")

    # Reset
    obs_dict, _ = env.reset()
    setup_camera_for_video()  # Reset sonrası tekrar ayarla

    step = 0
    physics_dt = 1.0 / 30.0  # ~30 Hz (decimation=4, dt=1/120)
    record_interval = int(1.0 / (physics_dt * args.fps))  # Her kaç step'te bir frame kaydet
    max_steps = int(args.record_duration / physics_dt)

    frame_step = 0

    try:
        while simulation_app.is_running() and step < max_steps:
            step += 1

            root_pos = env.robot.data.root_pos_w[0]

            # ===== RIGHT ARM OBSERVATION =====
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

            # ===== LEFT ARM OBSERVATION (MIRRORED - SAME POLICY!) =====
            left_joint_pos = env.robot.data.joint_pos[0, env.left_arm_indices]
            left_joint_vel = env.robot.data.joint_vel[0, env.left_arm_indices]
            left_ee_pos = env._compute_left_ee_pos()[0] - root_pos
            left_target = env.left_target_pos[0]

            # Mirror Y coordinates
            left_target_mirrored = left_target.clone()
            left_target_mirrored[1] = -left_target_mirrored[1]

            left_ee_mirrored = left_ee_pos.clone()
            left_ee_mirrored[1] = -left_ee_mirrored[1]

            # Mirror joint positions
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

            # AYNI POLICY - mirror edilmiş observation ile
            left_obs = torch.cat([
                left_joint_mirrored,
                left_joint_vel_mirrored * 0.1,
                left_target_mirrored,
                left_ee_mirrored,
                left_error_norm,
            ]).unsqueeze(0)

            with torch.no_grad():
                left_actions_raw = actor(left_obs)[0]

            # Mirror actions back
            left_actions = left_actions_raw.clone()
            left_actions[1] = -left_actions[1]
            left_actions[2] = -left_actions[2]
            left_actions[4] = -left_actions[4]

            # Combine and step
            combined_actions = torch.cat([right_actions, left_actions]).unsqueeze(0)
            obs_dict, rewards, terminated, truncated, info = env.step(combined_actions)

            # Record frame
            if step % max(1, record_interval) == 0:
                recorder.capture_frame()
                frame_step += 1

            # Progress log
            if step % 100 == 0:
                total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
                progress = (step / max_steps) * 100
                print(f"[{progress:5.1f}%] Step {step}/{max_steps} | Reaches: {total} | Frames: {frame_step}")

    except KeyboardInterrupt:
        print("\n[INFO] Kayıt durduruldu")

    # Finalize video
    print("\n" + "=" * 70)
    print("KAYIT TAMAMLANDI")
    print("=" * 70)
    print(f"  Toplam step: {step}")
    print(f"  Toplam frame: {frame_step}")
    total_reaches = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
    print(f"  Toplam reaches: {total_reaches}")
    print("=" * 70)

    # Convert to MP4
    video_path = recorder.finalize_video(f"g1_general_reaching_{timestamp}.mp4")

    if video_path:
        print(f"\n🎬 VIDEO HAZIR: {video_path}")
        print("\nX POST İÇİN:")
        print("-" * 50)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()