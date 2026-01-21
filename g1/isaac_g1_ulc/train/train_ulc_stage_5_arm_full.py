"""
G1 Arm Reach - Stage 5 V3.1 Training (SPAWN FIX)
=================================================

V3 -> V3.1 DEĞİŞİKLİKLER:
1. SPAWN GEOMETRY FIX - hedefler artık doğru bölgede
2. Daha detaylı logging
3. Distance tracking eklendi

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/train/train_ulc_stage_5_arm_full.py --num_envs 4096 --max_iterations 3000 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="G1 Arm Reach Training - V3.1 (SPAWN FIX)")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=3000, help="Max training iterations")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from datetime import datetime

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

# V3.1 Environment
from g1_arm_dual_orient_env import G1ArmReachEnv, G1ArmReachEnvCfg

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import configclass


@configclass
class G1ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """V3.1 PPO config with fixed learning rate."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "g1_arm_reach_v3_1"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="fixed",  # FIXED - adaptive değil!
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


class CurriculumEnvWrapper(RslRlVecEnvWrapper):
    """Wrapper with success rate tracking and distance logging."""

    def __init__(self, env):
        super().__init__(env)
        self._iteration = 0
        self._step_count = 0
        self._unwrapped = env
        self._writer = None

        # Distance tracking
        self._distance_sum = 0.0
        self._distance_count = 0

    def set_writer(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def step(self, actions):
        self._step_count += 1
        result = super().step(actions)

        # Track distance to target
        if hasattr(self._unwrapped, 'target_pos') and hasattr(self._unwrapped, '_compute_ee_pos'):
            try:
                ee_pos = self._unwrapped._compute_ee_pos() - self._unwrapped.robot.data.root_pos_w
                target_pos = self._unwrapped.target_pos
                distances = (ee_pos - target_pos).norm(dim=-1)
                self._distance_sum += distances.mean().item()
                self._distance_count += 1
            except:
                pass

        if self._step_count % 24 == 0:
            self._iteration += 1

            if hasattr(self._unwrapped, 'update_curriculum'):
                success_rate = self._unwrapped.update_curriculum(self._iteration)

                if self._writer is not None:
                    # Curriculum metrics
                    self._writer.add_scalar('Curriculum/stage', self._unwrapped.curriculum_stage + 1, self._iteration)
                    self._writer.add_scalar('Curriculum/spawn_radius', self._unwrapped.current_spawn_radius, self._iteration)
                    self._writer.add_scalar('Curriculum/orientation_enabled', float(self._unwrapped.orientation_enabled), self._iteration)

                    # Success metrics
                    self._writer.add_scalar('Success/rate', success_rate, self._iteration)
                    self._writer.add_scalar('Success/total_reaches', self._unwrapped.total_reaches, self._iteration)
                    self._writer.add_scalar('Success/total_attempts', self._unwrapped.total_attempts, self._iteration)
                    self._writer.add_scalar('Success/stage_reaches', self._unwrapped.stage_reaches, self._iteration)
                    self._writer.add_scalar('Success/avg_reaches_per_env', self._unwrapped.reach_count.mean().item(), self._iteration)

                    # Distance metrics
                    if self._distance_count > 0:
                        avg_distance = self._distance_sum / self._distance_count
                        self._writer.add_scalar('Distance/avg_to_target', avg_distance, self._iteration)
                        self._distance_sum = 0.0
                        self._distance_count = 0

                if self._iteration % 50 == 0:
                    stage = self._unwrapped.curriculum_stage + 1
                    radius = self._unwrapped.current_spawn_radius
                    total_r = self._unwrapped.total_reaches
                    total_a = self._unwrapped.total_attempts
                    stage_r = self._unwrapped.stage_reaches
                    stage_a = self._unwrapped.stage_attempts

                    stage_sr = stage_r / max(stage_a, 1) * 100
                    global_sr = total_r / max(total_a, 1) * 100

                    print(f"[Curriculum] Iter {self._iteration:5d} | "
                          f"Stage {stage}/10 | "
                          f"Radius={radius*100:.0f}cm | "
                          f"Stage SR: {stage_sr:.1f}% ({stage_r}/{stage_a}) | "
                          f"Global SR: {global_sr:.1f}% | "
                          f"Total Reaches: {total_r}")

        return result


def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = G1ArmReachEnv(cfg=env_cfg)
    env = CurriculumEnvWrapper(env)

    runner_cfg = G1ArmReachPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"ulc_g1_arm_reach_v3_1_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    env.set_writer(log_dir)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    if args.resume:
        print(f"\n[INFO] Resuming from: {args.resume}")
        runner.load(args.resume)

    print("\n" + "=" * 70)
    print("    G1 ARM REACH TRAINING - V3.1 (SPAWN FIX)")
    print("=" * 70)
    print(f"  Environments:     {args.num_envs}")
    print(f"  Max iterations:   {args.max_iterations}")
    print(f"  Log directory:    {log_dir}")
    print("-" * 70)
    print("  🔧 V3.1 SPAWN GEOMETRY FIX:")
    print(f"    ✓ workspace_inner_radius: 8cm")
    print(f"    ✓ initial_spawn_radius:   18cm")
    print(f"    ✓ Hedefler 8-18cm arasında spawn olacak")
    print(f"    ✓ pos_threshold: 10cm (daha kolay)")
    print(f"    ✓ action_scale: 0.15 (daha büyük)")
    print("-" * 70)
    print("  BEKLENEN DAVRANIŞLAR:")
    print("    - İlk 100-200 iter içinde reach'ler başlamalı")
    print("    - Success rate %60'a ulaşınca stage ilerler")
    print("    - noise_std ~0.5-1.0 civarında kalmalı")
    print("=" * 70 + "\n")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    final_sr = env._unwrapped.get_success_rate() * 100
    final_stage = env._unwrapped.curriculum_stage + 1
    total_reaches = env._unwrapped.total_reaches

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Final stage: {final_stage}/10")
    print(f"  Final success rate: {final_sr:.1f}%")
    print(f"  Total reaches: {total_reaches}")
    print(f"  Logs: {log_dir}")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()