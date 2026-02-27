"""
G1 DUAL ARM PLAY - Using G1DualArmEnv (VIDEO RECORDING)
========================================================

Sağ kol policy'sini sol kola mirror olarak uygular.
4 visual marker ile çalışır.
OTOMATİK VİDEO KAYDI EKLENDİ.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/play/play_ulc_stage_4_arm_dual_record.py

Video: C:\IsaacLab\recordings\ klasörüne kaydedilir
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

parser = argparse.ArgumentParser(description="G1 Dual Arm Play")
parser.add_argument("--record_duration", type=float, default=25.0, help="Video süresi (saniye)")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

# Camera view için
from isaacsim.core.utils.viewports import set_camera_view

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


# ============== VIDEO RECORDING CLASS ==============
class FrameRecorder:
    """Viewport frame'lerini kaydeder, sonra ffmpeg ile MP4'e çevirir."""

    def __init__(self, output_dir: str, fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0

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
            "-crf", "18",
            output_path
        ]

        print(f"\n[VIDEO] Converting {self.frame_count} frames to MP4...")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[VIDEO] ✓ Saved: {output_path}")

            import shutil
            shutil.rmtree(self.frame_dir)
            print(f"[VIDEO] ✓ Cleaned up frames")

            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[VIDEO] ✗ ffmpeg error: {e}")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            return None
        except FileNotFoundError:
            print(f"[VIDEO] ✗ ffmpeg not found! Frames in: {self.frame_dir}")
            return None
# ===================================================


def main():
    print("\n" + "=" * 70)
    print("   G1 DUAL ARM PLAY - VIDEO RECORDING")
    print("   Sağ kol policy'si → Sol kola mirror")
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
    env_cfg.episode_length_s = 300.0

    env = G1DualArmEnv(cfg=env_cfg)

    # ============== CAMERA SETUP ==============
    # 3/4 diagonal view - her iki kol görünür
    eye = (-1.4, 0.8, 1.4)
    target = (0.0, 0.0, 1.0)
    set_camera_view(eye=eye, target=target)
    print(f"[CAMERA] Eye: {eye}, Target: {target}")
    # ==========================================

    # ============== RECORDER SETUP ==============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = os.path.join(os.getcwd(), "recordings", timestamp)
    os.makedirs(record_dir, exist_ok=True)
    recorder = FrameRecorder(record_dir, fps=args.fps)
    # ============================================

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
    print(f"[RECORDING] {args.record_duration} saniye kayıt başlıyor...")
    print("-" * 70 + "\n")

    # Reset
    obs_dict, _ = env.reset()

    # Reset sonrası kamerayı tekrar ayarla
    set_camera_view(eye=eye, target=target)

    step = 0

    # ============== RECORDING PARAMS ==============
    physics_dt = 1.0 / 30.0  # ~30 Hz
    record_interval = max(1, int(1.0 / (physics_dt * args.fps)))
    max_steps = int(args.record_duration / physics_dt)
    # ==============================================

    try:
        while simulation_app.is_running() and step < max_steps:
            step += 1

            root_pos = env.robot.data.root_pos_w[0]

            # ===== RIGHT ARM OBSERVATION (MUST MATCH TRAINING ORDER!) =====
            # Training order: joint_pos(5), joint_vel(5), target(3), ee_pos(3), error_norm(3) = 19
            right_joint_pos = env.robot.data.joint_pos[0, env.right_arm_indices]
            right_joint_vel = env.robot.data.joint_vel[0, env.right_arm_indices]
            right_ee_pos = env._compute_right_ee_pos()[0] - root_pos
            right_target = env.right_target_pos[0]
            right_error = right_target - right_ee_pos
            right_error_norm = right_error / 0.31  # max_target_radius + 0.01

            right_obs = torch.cat([
                right_joint_pos,        # 5
                right_joint_vel * 0.1,  # 5
                right_target,           # 3
                right_ee_pos,           # 3
                right_error_norm,       # 3
            ]).unsqueeze(0)  # Total: 19

            with torch.no_grad():
                right_actions = actor(right_obs)[0]

            # ===== LEFT ARM OBSERVATION (MIRRORED, SAME ORDER) =====
            left_joint_pos = env.robot.data.joint_pos[0, env.left_arm_indices]
            left_joint_vel = env.robot.data.joint_vel[0, env.left_arm_indices]
            left_ee_pos = env._compute_left_ee_pos()[0] - root_pos
            left_target = env.left_target_pos[0]

            # Mirror Y coordinates for policy input
            left_target_mirrored = left_target.clone()
            left_target_mirrored[1] = -left_target_mirrored[1]

            left_ee_mirrored = left_ee_pos.clone()
            left_ee_mirrored[1] = -left_ee_mirrored[1]

            # Mirror joint positions (roll/yaw joints are opposite)
            left_joint_mirrored = left_joint_pos.clone()
            left_joint_mirrored[1] = -left_joint_mirrored[1]  # shoulder_roll
            left_joint_mirrored[2] = -left_joint_mirrored[2]  # shoulder_yaw
            left_joint_mirrored[4] = -left_joint_mirrored[4]  # elbow_roll

            left_joint_vel_mirrored = left_joint_vel.clone()
            left_joint_vel_mirrored[1] = -left_joint_vel_mirrored[1]
            left_joint_vel_mirrored[2] = -left_joint_vel_mirrored[2]
            left_joint_vel_mirrored[4] = -left_joint_vel_mirrored[4]

            left_error = left_target_mirrored - left_ee_mirrored
            left_error_norm = left_error / 0.31

            left_obs = torch.cat([
                left_joint_mirrored,         # 5
                left_joint_vel_mirrored * 0.1,  # 5
                left_target_mirrored,        # 3
                left_ee_mirrored,            # 3
                left_error_norm,             # 3
            ]).unsqueeze(0)  # Total: 19

            with torch.no_grad():
                left_actions_raw = actor(left_obs)[0]

            # Mirror actions back (roll/yaw joints opposite)
            left_actions = left_actions_raw.clone()
            left_actions[1] = -left_actions[1]  # shoulder_roll
            left_actions[2] = -left_actions[2]  # shoulder_yaw
            left_actions[4] = -left_actions[4]  # elbow_roll

            # ===== COMBINE ACTIONS =====
            combined_actions = torch.cat([right_actions, left_actions]).unsqueeze(0)

            # Step
            obs_dict, rewards, terminated, truncated, info = env.step(combined_actions)

            # ============== CAPTURE FRAME ==============
            if step % record_interval == 0:
                recorder.capture_frame()
            # ===========================================

            # Compute distances (after step, use fresh data)
            root_pos_new = env.robot.data.root_pos_w[0]
            right_ee_new = env._compute_right_ee_pos()[0] - root_pos_new
            left_ee_new = env._compute_left_ee_pos()[0] - root_pos_new
            right_dist = (right_ee_new - env.right_target_pos[0]).norm().item()
            left_dist = (left_ee_new - env.left_target_pos[0]).norm().item()

            # Log (daha az sıklıkta - video için temiz)
            if step % 100 == 0:
                total = int(env.right_reach_count[0].item() + env.left_reach_count[0].item())
                progress = (step / max_steps) * 100
                print(f"[{progress:5.1f}%] Step {step}/{max_steps} | R:{int(env.right_reach_count[0].item())} L:{int(env.left_reach_count[0].item())} | Total: {total} | Frames: {recorder.frame_count}")

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
    print(f"  Frames captured:  {recorder.frame_count}")
    print("=" * 70)

    # ============== FINALIZE VIDEO ==============
    video_path = recorder.finalize_video(f"g1_dual_arm_{timestamp}.mp4")
    if video_path:
        print(f"\n🎬 VIDEO HAZIR: {video_path}")
    # ============================================

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()