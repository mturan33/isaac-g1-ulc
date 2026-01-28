#!/usr/bin/env python3
"""
Stage 5.5 Dual Policy Play Script
==================================

İki ayrı policy'yi birlikte çalıştırır:
- Stage 3 (Loco): 57 obs → 12 leg actions
- Stage 5 (Arm): 29 obs → 5 arm actions

USAGE:
./isaaclab.bat -p .../play/play_ulc_stage_5.5_both.py \
    --loco_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt \
    --arm_checkpoint logs/ulc/g1_arm_reach_2026-01-22_14-06-41/model_19998.pt \
    --num_envs 4 \
    --vx 0.0

Author: Turan
Date: January 2026
"""

from __future__ import annotations

import argparse
import os
import sys

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

parser = argparse.ArgumentParser(description="Stage 5.5 Dual Policy Play")
parser.add_argument("--loco_checkpoint", type=str, required=True,
                    help="Path to Stage 3 locomotion checkpoint")
parser.add_argument("--arm_checkpoint", type=str, required=True,
                    help="Path to Stage 5 arm checkpoint")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity")
parser.add_argument("--pitch", type=float, default=0.0, help="Torso pitch (rad)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

# Add envs dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
envs_dir = os.path.join(os.path.dirname(script_dir), "envs")
sys.path.insert(0, envs_dir)

from g1_locomanip_env import G1LocoManipEnv, G1LocoManipEnvCfg


# ==============================================================================
# NETWORK DEFINITIONS
# ==============================================================================

class LocoActorCritic(nn.Module):
    """Stage 3 network: 57 obs → 12 actions, hidden=[512, 256, 128]"""
    
    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)
        
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)
        
        self.log_std = nn.Parameter(torch.zeros(num_act))
    
    def act(self, x, deterministic=True):
        mean = self.actor(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


class ArmActor(nn.Module):
    """Stage 5 network: 29 obs → 5 actions, hidden=[256, 128, 64]"""
    
    def __init__(self, num_obs=29, num_act=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, num_act),
        )
    
    def forward(self, x):
        return self.net(x)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = "cuda:0"
    
    print("\n" + "=" * 70)
    print("STAGE 5.5 DUAL POLICY PLAY")
    print("=" * 70)
    print(f"Loco checkpoint: {args.loco_checkpoint}")
    print(f"Arm checkpoint: {args.arm_checkpoint}")
    print("=" * 70 + "\n")
    
    # =========================================================================
    # LOAD LOCO POLICY (Stage 3)
    # =========================================================================
    print("[1/4] Loading LOCO policy (Stage 3)...")
    
    loco_ckpt = torch.load(args.loco_checkpoint, map_location=device, weights_only=False)
    
    loco_net = LocoActorCritic(57, 12).to(device)
    
    if "actor_critic" in loco_ckpt:
        loco_net.load_state_dict(loco_ckpt["actor_critic"])
        print("      Loaded from 'actor_critic' key")
    else:
        loco_net.load_state_dict(loco_ckpt)
        print("      Loaded directly")
    
    loco_net.eval()
    
    if "best_reward" in loco_ckpt:
        print(f"      Best reward: {loco_ckpt['best_reward']:.2f}")
    if "iteration" in loco_ckpt:
        print(f"      Iteration: {loco_ckpt['iteration']}")
    
    print("      ✓ LOCO policy loaded!")
    
    # =========================================================================
    # LOAD ARM POLICY (Stage 5)
    # =========================================================================
    print("[2/4] Loading ARM policy (Stage 5)...")
    
    arm_ckpt = torch.load(args.arm_checkpoint, map_location=device, weights_only=False)
    
    arm_net = ArmActor(29, 5).to(device)
    
    if "model_state_dict" in arm_ckpt:
        # Extract actor weights
        state_dict = arm_ckpt["model_state_dict"]
        actor_state = {}
        for key, value in state_dict.items():
            if "actor" in key:
                new_key = key.replace("actor.", "")
                actor_state[new_key] = value
        
        if actor_state:
            arm_net.net.load_state_dict(actor_state)
            print(f"      Loaded {len(actor_state)} actor weights")
        else:
            print("      WARNING: No actor weights found!")
    else:
        arm_net.load_state_dict(arm_ckpt)
        print("      Loaded directly")
    
    arm_net.eval()
    print("      ✓ ARM policy loaded!")
    
    # =========================================================================
    # CREATE ENVIRONMENT
    # =========================================================================
    print("[3/4] Creating environment...")
    
    cfg = G1LocoManipEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    env = G1LocoManipEnv(cfg)
    
    # Set commands
    env.set_velocity_command(vx=args.vx)
    env.set_torso_command(pitch=args.pitch)
    
    print(f"      Commands: vx={args.vx}, pitch={np.rad2deg(args.pitch):.1f}°")
    print("      ✓ Environment created!")
    
    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    print("[4/4] Running simulation...")
    print("-" * 70)
    
    obs, _ = env.reset()
    
    total_reaches = 0
    prev_reaches = 0
    
    with torch.no_grad():
        for step in range(args.steps):
            # Get separate observations
            loco_obs = env.get_loco_obs()  # 57 dim
            arm_obs = env.get_arm_obs()    # 29 dim
            
            # Get actions from each policy
            leg_actions = loco_net.act(loco_obs, deterministic=True)  # 12
            arm_actions = arm_net(arm_obs)                              # 5
            
            # Combine actions
            combined_actions = torch.cat([leg_actions, arm_actions], dim=-1)  # 17
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(combined_actions)
            
            # Check reaches
            if env.total_reaches > prev_reaches:
                print(f"[Step {step:5d}] 🎯 REACH #{env.total_reaches}!")
                prev_reaches = env.total_reaches
            
            # Progress
            if step > 0 and step % 200 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()
                
                ee_pos = env._compute_ee_pos() - env.robot.data.root_pos_w
                dist = (ee_pos - env.target_pos).norm(dim=-1).mean().item()
                
                print(f"[Step {step:5d}] H={height:.3f}m | Vx={vx:.2f}m/s | "
                      f"EE dist={dist:.3f}m | Reaches={env.total_reaches}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DUAL POLICY PLAY COMPLETE")
    print("=" * 70)
    print(f"  Total reaches: {env.total_reaches}")
    print(f"  Steps: {args.steps}")
    print("=" * 70 + "\n")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()