"""
G1 Stage 6: Unified Loco-Manipulation - Play Script
====================================================

Eğitilmiş unified policy'yi test etmek için.

KULLANIM:
./isaaclab.bat -p .../play/play_ulc_stage_6_unified.py \
    --checkpoint logs/ulc/g1_unified_.../model_best.pt \
    --num_envs 4 \
    --steps 3000

KONTROL MODLARı:
--mode standing    : Durarak kol kontrolü
--mode walking     : Yürürken kol kontrolü
--mode squat       : Squat yaparak kol kontrolü
--mode full        : Tam loco-manipulation (default)

VELOCİTY OVERRIDE:
--vx 0.5           : İleri hız (m/s)
--pitch -0.2       : Öne eğilme (rad)
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="G1 Stage 6 Play")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--steps", type=int, default=3000, help="Steps to run")
parser.add_argument("--mode", type=str, default="full",
                    choices=["standing", "walking", "squat", "full"],
                    help="Control mode")
parser.add_argument("--vx", type=float, default=None, help="Override vx command")
parser.add_argument("--vy", type=float, default=None, help="Override vy command")
parser.add_argument("--vyaw", type=float, default=None, help="Override vyaw command")
parser.add_argument("--pitch", type=float, default=None, help="Override pitch command (rad)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force non-headless for visualization
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Add env path
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_unified_env import G1UnifiedEnv, G1UnifiedEnvCfg


# ============================================================================
# SIMPLE ACTOR (for loading checkpoint)
# ============================================================================

class SimpleActor(nn.Module):
    """Simple actor network matching training architecture."""

    def __init__(self, num_obs: int, num_act: int, hidden=[512, 256, 128]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


# ============================================================================
# CONTROL MODE CONFIGURATIONS
# ============================================================================

MODE_CONFIGS = {
    "standing": {
        "vx_range": (0.0, 0.0),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "pitch_range": (0.0, 0.0),
        "description": "Standing still with arm reaching",
    },
    "walking": {
        "vx_range": (0.3, 0.5),
        "vy_range": (-0.1, 0.1),
        "vyaw_range": (-0.2, 0.2),
        "pitch_range": (0.0, 0.0),
        "description": "Walking forward with arm reaching",
    },
    "squat": {
        "vx_range": (0.0, 0.2),
        "vy_range": (0.0, 0.0),
        "vyaw_range": (0.0, 0.0),
        "pitch_range": (-0.25, -0.15),
        "description": "Squatting/leaning forward with arm reaching",
    },
    "full": {
        "vx_range": (-0.2, 0.6),
        "vy_range": (-0.2, 0.2),
        "vyaw_range": (-0.4, 0.4),
        "pitch_range": (-0.3, 0.0),
        "description": "Full loco-manipulation (random commands)",
    },
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = "cuda:0"

    # Create environment
    env_cfg = G1UnifiedEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = G1UnifiedEnv(cfg=env_cfg)

    num_obs = env_cfg.num_observations
    num_act = env_cfg.num_actions

    # Load checkpoint
    print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get state dict
    if "actor_critic" in checkpoint:
        state_dict = checkpoint["actor_critic"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Create actor
    actor = SimpleActor(num_obs, num_act).to(device)

    # Load weights
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key] = value

    if actor_state:
        actor.load_state_dict(actor_state, strict=False)
        print(f"[INFO] Loaded {len(actor_state)} actor weights")
    else:
        print("[WARNING] No actor weights found!")

    actor.eval()

    # Get mode config
    mode_cfg = MODE_CONFIGS[args.mode]

    print(f"\n{'='*60}")
    print(f"G1 STAGE 6: UNIFIED LOCO-MANIPULATION PLAY")
    print(f"{'='*60}")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  Mode: {args.mode} - {mode_cfg['description']}")
    print(f"  Observation dim: {num_obs}")
    print(f"  Action dim: {num_act}")
    print(f"  Steps: {args.steps}")
    if checkpoint.get("curriculum_level"):
        print(f"  Trained to level: {checkpoint['curriculum_level'] + 1}")
    if checkpoint.get("best_reward"):
        print(f"  Best training reward: {checkpoint['best_reward']:.2f}")
    print(f"{'-'*60}")
    print(f"  Controls:")
    print(f"    - Robot walks and reaches for green targets")
    print(f"    - Orange sphere = end effector")
    print(f"    - Watch for squat behavior when pitch < 0")
    print(f"{'='*60}\n")

    # Override commands function
    def set_commands():
        n = args.num_envs

        # Velocity commands
        if args.vx is not None:
            env.vel_cmd[:, 0] = args.vx
        else:
            vx_range = mode_cfg["vx_range"]
            env.vel_cmd[:, 0] = torch.rand(n, device=device) * (vx_range[1] - vx_range[0]) + vx_range[0]

        if args.vy is not None:
            env.vel_cmd[:, 1] = args.vy
        else:
            vy_range = mode_cfg["vy_range"]
            env.vel_cmd[:, 1] = torch.rand(n, device=device) * (vy_range[1] - vy_range[0]) + vy_range[0]

        if args.vyaw is not None:
            env.vel_cmd[:, 2] = args.vyaw
        else:
            vyaw_range = mode_cfg["vyaw_range"]
            env.vel_cmd[:, 2] = torch.rand(n, device=device) * (vyaw_range[1] - vyaw_range[0]) + vyaw_range[0]

        # Torso commands (pitch for squat)
        if args.pitch is not None:
            env.torso_cmd[:, 1] = args.pitch
        else:
            pitch_range = mode_cfg["pitch_range"]
            env.torso_cmd[:, 1] = torch.rand(n, device=device) * (pitch_range[1] - pitch_range[0]) + pitch_range[0]

    # Reset and set initial commands
    obs, _ = env.reset()
    set_commands()

    # Run
    total_reward = 0.0
    total_reaches = 0
    min_arm_dist = float('inf')
    distance_sum = 0.0
    distance_count = 0

    with torch.no_grad():
        for step in range(args.steps):
            # Get action
            obs_tensor = obs["policy"]
            action = actor(obs_tensor)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward.mean().item()

            # Track arm distance
            arm_dist = env.extras.get("M/arm_dist", 0)
            if arm_dist > 0:
                distance_sum += arm_dist
                distance_count += 1
                if arm_dist < min_arm_dist:
                    min_arm_dist = arm_dist

            # Check reaches
            new_reaches = env.total_reaches - total_reaches
            if new_reaches > 0:
                total_reaches = env.total_reaches
                print(f"[Step {step:4d}] 🎯 REACH #{total_reaches}! Dist was: {arm_dist:.3f}m")

            # Resample commands periodically (for variety)
            if step > 0 and step % 500 == 0:
                set_commands()
                print(f"[Step {step:4d}] Commands resampled: vx={env.vel_cmd[0,0]:.2f}, pitch={env.torso_cmd[0,1]:.2f}")

            # Progress update
            if step > 0 and step % 200 == 0:
                avg_dist = distance_sum / max(distance_count, 1)
                height = env.extras.get("M/height", 0)
                vx = env.extras.get("M/vx", 0)
                pitch = env.extras.get("M/pitch", 0)

                print(f"[Step {step:4d}] "
                      f"Height: {height:.3f}m | "
                      f"Vx: {vx:.2f}m/s | "
                      f"Pitch: {pitch:.2f}rad | "
                      f"ArmDist: {arm_dist:.3f}m (avg: {avg_dist:.3f}m) | "
                      f"Reaches: {total_reaches}")

    # Summary
    avg_distance = distance_sum / max(distance_count, 1)

    print(f"\n{'='*60}")
    print("PLAY COMPLETE")
    print(f"{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  Total reaches: {total_reaches}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Avg arm distance: {avg_distance:.3f}m")
    print(f"  Min arm distance: {min_arm_dist:.3f}m")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()