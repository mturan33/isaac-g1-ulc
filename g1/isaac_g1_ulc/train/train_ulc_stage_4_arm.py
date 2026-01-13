"""
G1 Arm Reach - Training Script (Stage 4) with Curriculum Learning
===================================================================

Fixed-base arm reaching with ULC-style smooth motion and curriculum.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_4_arm.py --num_envs 4096 --max_iterations 5000 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

# =============================================================================
# ARGUMENT PARSING (BEFORE AppLauncher)
# =============================================================================

parser = argparse.ArgumentParser(description="G1 Arm Reach Training - Stage 4")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# =============================================================================
# LAUNCH APP
# =============================================================================

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =============================================================================
# IMPORTS (AFTER AppLauncher)
# =============================================================================

import torch
from datetime import datetime

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(env_dir, "envs"))

from g1_arm_reach_env import G1ArmReachEnv, G1ArmReachEnvCfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlOnPolicyRunner,
    RslRlVecEnvWrapper,
)
from isaaclab.utils import configclass


# =============================================================================
# CUSTOM RUNNER WITH CURRICULUM
# =============================================================================

class CurriculumRunner(RslRlOnPolicyRunner):
    """PPO Runner with curriculum learning support."""

    def __init__(self, env, train_cfg, log_dir, device):
        super().__init__(env, train_cfg, log_dir, device)
        self.unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env._env

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True):
        """Override learn to add curriculum updates."""

        # Get the original learn method's setup
        if init_at_random_ep_len:
            self.env.reset()

        obs = self.env.get_observations()
        critic_obs = obs.get("critic", obs["policy"])
        obs, critic_obs = obs["policy"], critic_obs

        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = []
        lenbuffer = []

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        import time
        start_time = time.time()

        for it in range(self.current_learning_iteration, num_learning_iterations):
            # =============================================
            # CURRICULUM UPDATE
            # =============================================
            if hasattr(self.unwrapped_env, 'update_curriculum'):
                self.unwrapped_env.update_curriculum(it)
                if it % 500 == 0:
                    print(f"[Curriculum] Iteration {it}: target_radius = {self.unwrapped_env.current_target_radius:.3f}m")

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = obs.get("critic", obs)
                    obs, critic_obs = obs["policy"] if isinstance(obs, dict) else obs, critic_obs["policy"] if isinstance(critic_obs, dict) else critic_obs

                    self.alg.process_env_step(rewards, dones, infos)

                    if "episode" in infos:
                        ep_infos.append(infos["episode"])

            # Compute returns and update
            self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()

            # Logging
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.current_learning_iteration = it + 1

            stop_time = time.time()
            self.tot_time += stop_time - start_time
            start_time = stop_time

            if len(ep_infos) > 0:
                for key in ep_infos[0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in ep_infos:
                        if key in ep_info:
                            infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    if infotensor.numel() > 0:
                        self.writer.add_scalar(f"Episode/{key}", infotensor.mean(), it)
                ep_infos = []

            # Standard logging
            fps = int(self.num_steps_per_env * self.env.num_envs / (stop_time - start_time + 1e-9))

            if it % 100 == 0:
                mean_reward = self.alg.storage.rewards.mean()
                print(f"[Iter {it:5d}] Reward: {mean_reward:.3f} | FPS: {fps:5d} | Value Loss: {mean_value_loss:.4f}")

            self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
            self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
            self.writer.add_scalar("Perf/fps", fps, it)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Save best model based on mean reward
            mean_reward = self.alg.storage.rewards.mean().item()
            if not hasattr(self, 'best_reward') or mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save(os.path.join(self.log_dir, "model_best.pt"))

        # Final save
        self.save(os.path.join(self.log_dir, f"model_{num_learning_iterations}.pt"))


# =============================================================================
# RSL-RL TRAINING CONFIG
# =============================================================================

@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for arm reaching with smooth motion."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "g1_arm_reach"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Lower initial noise for smoother start
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,  # Lower entropy for less exploration noise
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,  # Lower LR for smoother learning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,  # Tighter KL for stability
        max_grad_norm=1.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = G1ArmReachEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = G1ArmReachPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"ulc_g1_stage4_arm_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Use curriculum runner
    runner = CurriculumRunner(env, runner_cfg, log_dir=log_dir, device="cuda:0")

    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    print("\n" + "=" * 70)
    print("         G1 ARM REACH TRAINING - STAGE 4 (ULC Smooth Motion)")
    print("=" * 70)
    print(f"  Environments:     {args.num_envs}")
    print(f"  Max iterations:   {args.max_iterations}")
    print(f"  Log directory:    {log_dir}")
    print("-" * 70)
    print("  ULC SMOOTH MOTION FEATURES:")
    print(f"    Action smoothing α: {env_cfg.action_smoothing_alpha}")
    print(f"    Action scale: {env_cfg.action_scale} rad")
    print(f"    Max joint velocity: {env_cfg.max_joint_vel} rad/s")
    print("-" * 70)
    print("  CURRICULUM:")
    print(f"    Initial radius: {env_cfg.initial_target_radius}m")
    print(f"    Final radius: {env_cfg.max_target_radius}m")
    print(f"    Curriculum steps: {env_cfg.curriculum_steps}")
    print("-" * 70)
    print("  REWARD STRUCTURE:")
    print(f"    Position distance: {env_cfg.reward_pos_distance}")
    print(f"    Orientation distance: {env_cfg.reward_ori_distance}")
    print(f"    Reaching bonus: {env_cfg.reward_reaching}")
    print(f"    Action rate penalty: {env_cfg.reward_action_rate}")
    print(f"    Action accel penalty: {env_cfg.reward_action_accel}")
    print(f"    Joint velocity penalty: {env_cfg.reward_joint_vel}")
    print(f"    Joint accel penalty: {env_cfg.reward_joint_accel}")
    print(f"    Smooth approach bonus: {env_cfg.reward_smooth_approach}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("=" * 70 + "\n")

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Logs saved to: {log_dir}")
    print(f"  Best model: {log_dir}/model_best.pt")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()