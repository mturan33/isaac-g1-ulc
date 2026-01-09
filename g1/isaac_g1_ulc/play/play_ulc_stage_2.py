#!/usr/bin/env python3
"""
ULC G1 Stage 2 v2 Play Script - Fixed Network Architecture
Network matches train_ulc_stage_2_v2.py with LayerNorm
"""

import argparse
import torch
import torch.nn as nn

# Isaac Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ULC G1 Stage 2 v2 Play")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Post-launch imports
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.isaac_g1_ulc.g1.isaac_g1_ulc.envs.g1_ulc_env import G1ULCEnv, G1ULCEnvCfg

print("=" * 60)
print("ULC G1 STAGE 2 v2 - PLAY (Fixed Network)")
print("=" * 60)


class ActorCriticNetwork(nn.Module):
    """Network architecture matching train_ulc_stage_2_v2.py with LayerNorm"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256, 128]):
        super().__init__()

        # Actor network with LayerNorm (matches training)
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.LayerNorm(hidden_dim))
            actor_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network with LayerNorm (matches training)
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action std (not used in play, but needed for state_dict compatibility)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean of policy)"""
        return self.actor(obs)

    def forward(self, obs: torch.Tensor):
        return self.act(obs)


def main():
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    print(f"[INFO] Loaded: {args.checkpoint}")

    if "best_reward" in checkpoint:
        print(f"[INFO] Best reward: {checkpoint['best_reward']:.2f}")
    if "iteration" in checkpoint:
        print(f"[INFO] Iteration: {checkpoint['iteration']}")

    # Create environment config
    env_cfg = G1ULCEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    # Create environment
    env = G1ULCEnv(cfg=env_cfg)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"[INFO] Obs dim: {obs_dim}, Act dim: {act_dim}")

    # Create network with same architecture as training
    actor_critic = ActorCriticNetwork(obs_dim, act_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # Load weights
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()
    print("[INFO] Model loaded successfully!")

    # Initialize
    obs, _ = env.reset()

    # Velocity command for testing
    target_vx = 0.5  # m/s forward

    print(f"[Play] Running with vx={target_vx:.2f} m/s")
    print("[Play] Press Ctrl+C to stop")
    print("-" * 60)

    step = 0
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # Get action from policy
                actions = actor_critic.act(obs)

                # Clip actions
                actions = torch.clamp(actions, -1.0, 1.0)

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Log periodically
            if step % 100 == 0:
                mean_reward = rewards.mean().item()
                print(f"Step {step:5d} | Reward: {mean_reward:.3f}")

            step += 1

            # Reset if needed
            dones = terminated | truncated
            if dones.any():
                # Environment handles reset internally
                pass

    except KeyboardInterrupt:
        print("\n[Play] Stopped by user")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()