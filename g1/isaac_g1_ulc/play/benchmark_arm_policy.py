"""
G1 Arm Reach - Policy Benchmark & Evaluation
=============================================

Policy'nin yeteneklerini kapsamlı test eder:
1. Farklı zorluk seviyelerinde (spawn radius) test
2. Success rate, reach time, smoothness metrikleri
3. Detaylı rapor çıktısı

KULLANIM:
./isaaclab.bat -p .../play/benchmark_arm_policy.py --checkpoint logs/ulc/.../model_XXXX.pt --num_envs 64

HEADLESS (hızlı test):
./isaaclab.bat -p .../play/benchmark_arm_policy.py --checkpoint logs/ulc/.../model_XXXX.pt --num_envs 256 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description="G1 Arm Reach Policy Benchmark")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--episodes_per_test", type=int, default=100, help="Episodes per difficulty level")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from datetime import datetime

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
envs_dir = os.path.join(env_dir, "envs")
sys.path.insert(0, envs_dir)

from g1_arm_dual_orient_env import G1ArmReachEnv, G1ArmReachEnvCfg


class PolicyBenchmark:
    """Comprehensive policy evaluation."""

    def __init__(self, env, actor, device="cuda:0"):
        self.env = env
        self.actor = actor
        self.device = device

        # Test configurations (spawn_radius, threshold, name)
        # G1 minimum reach ~18cm based on arm kinematics
        self.test_levels = [
            (0.20, 0.15, "Level 1 - Easy (20cm spawn)"),
            (0.25, 0.14, "Level 2 - Medium-Easy (25cm)"),
            (0.30, 0.13, "Level 3 - Medium (30cm)"),
            (0.35, 0.12, "Level 4 - Medium-Hard (35cm)"),
            (0.40, 0.11, "Level 5 - Hard (40cm)"),
            (0.45, 0.10, "Level 6 - Max Range (45cm)"),
        ]

    def run_episode_batch(self, spawn_radius: float, threshold: float, num_episodes: int):
        """Run multiple episodes and collect metrics."""

        # Configure environment
        self.env.current_spawn_radius = spawn_radius
        self.env.current_pos_threshold = threshold

        # Metrics
        total_reaches = 0
        total_attempts = 0
        reach_times = []
        min_distances = []
        action_smoothness = []
        velocity_smoothness = []

        episodes_completed = 0
        step_count = 0
        episode_steps = torch.zeros(self.env.num_envs, device=self.device)

        # Reset tracking
        self.env.total_reaches = 0
        self.env.total_attempts = 0

        obs, _ = self.env.reset()

        prev_actions = torch.zeros((self.env.num_envs, 5), device=self.device)

        max_steps = num_episodes * 350 // self.env.num_envs + 500

        with torch.no_grad():
            while episodes_completed < num_episodes and step_count < max_steps:
                # Get action
                obs_tensor = obs["policy"]
                action = self.actor(obs_tensor)

                # Calculate smoothness (action rate)
                action_diff = (action - prev_actions).norm(dim=-1)
                action_smoothness.extend(action_diff.cpu().numpy().tolist())
                prev_actions = action.clone()

                # Step
                obs, reward, terminated, truncated, info = self.env.step(action)
                step_count += 1
                episode_steps += 1

                # Track velocity smoothness
                if hasattr(self.env, 'robot'):
                    joint_vel = self.env.robot.data.joint_vel[:, self.env.arm_indices]
                    vel_norm = joint_vel.norm(dim=-1)
                    velocity_smoothness.extend(vel_norm.cpu().numpy().tolist())

                # Track min distances
                if hasattr(self.env, 'prev_distance'):
                    min_distances.extend(self.env.prev_distance.cpu().numpy().tolist())

                # Check for episode ends
                done = terminated | truncated
                done_ids = torch.where(done)[0]

                if len(done_ids) > 0:
                    for idx in done_ids:
                        episodes_completed += 1

                        # Record reach time for successful episodes
                        if self.env.episode_reach_count[idx] > 0:
                            reach_times.append(episode_steps[idx].item())

                    episode_steps[done_ids] = 0

        # Collect final stats
        total_reaches = self.env.total_reaches
        total_attempts = self.env.total_attempts

        success_rate = total_reaches / max(total_attempts, 1)
        avg_reach_time = np.mean(reach_times) if reach_times else float('inf')
        avg_min_distance = np.mean(min_distances) if min_distances else float('inf')
        avg_action_smoothness = np.mean(action_smoothness) if action_smoothness else 0
        avg_velocity = np.mean(velocity_smoothness) if velocity_smoothness else 0

        return {
            "success_rate": success_rate,
            "total_reaches": total_reaches,
            "total_attempts": total_attempts,
            "avg_reach_time": avg_reach_time,
            "avg_min_distance": avg_min_distance,
            "avg_action_rate": avg_action_smoothness,
            "avg_velocity": avg_velocity,
            "episodes_completed": episodes_completed,
        }

    def run_full_benchmark(self, episodes_per_level: int = 100):
        """Run benchmark across all difficulty levels."""

        print("\n" + "=" * 70)
        print("🎯 G1 ARM REACH POLICY BENCHMARK")
        print("=" * 70)
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Environments: {self.env.num_envs}")
        print(f"  Episodes per level: {episodes_per_level}")
        print(f"  Test levels: {len(self.test_levels)}")
        print("=" * 70 + "\n")

        results = []

        for i, (spawn_radius, threshold, name) in enumerate(self.test_levels):
            print(f"\n{'─' * 60}")
            print(f"Testing: {name}")
            print(f"  Spawn radius: {spawn_radius*100:.0f}cm, Threshold: {threshold*100:.0f}cm")
            print(f"{'─' * 60}")

            start_time = time.time()
            metrics = self.run_episode_batch(spawn_radius, threshold, episodes_per_level)
            elapsed = time.time() - start_time

            metrics["level"] = i + 1
            metrics["name"] = name
            metrics["spawn_radius"] = spawn_radius
            metrics["threshold"] = threshold
            metrics["elapsed_time"] = elapsed

            results.append(metrics)

            # Print level results
            sr = metrics["success_rate"] * 100
            sr_emoji = "🟢" if sr >= 50 else "🟡" if sr >= 25 else "🔴"

            print(f"\n  {sr_emoji} Success Rate: {sr:.1f}%")
            print(f"  📊 Reaches: {metrics['total_reaches']} / {metrics['total_attempts']}")
            print(f"  ⏱️  Avg Reach Time: {metrics['avg_reach_time']:.1f} steps")
            print(f"  📏 Avg Min Distance: {metrics['avg_min_distance']*100:.1f}cm")
            print(f"  🔄 Avg Action Rate: {metrics['avg_action_rate']:.4f}")
            print(f"  💨 Avg Joint Velocity: {metrics['avg_velocity']:.3f}")
            print(f"  ⏰ Test Time: {elapsed:.1f}s")

        return results

    def print_summary_report(self, results: list):
        """Print comprehensive summary report."""

        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "📋 BENCHMARK SUMMARY REPORT" + " " * 26 + "║")
        print("╠" + "═" * 78 + "╣")

        # Header
        print("║ {:^6} │ {:^20} │ {:^10} │ {:^10} │ {:^10} │ {:^10} ║".format(
            "Level", "Difficulty", "Success%", "Reaches", "AvgTime", "Smoothness"
        ))
        print("╠" + "═" * 78 + "╣")

        total_reaches = 0
        total_attempts = 0

        for r in results:
            sr = r["success_rate"] * 100
            sr_str = f"{sr:.1f}%"

            if sr >= 50:
                status = "🟢"
            elif sr >= 25:
                status = "🟡"
            else:
                status = "🔴"

            reach_str = f"{r['total_reaches']}/{r['total_attempts']}"
            time_str = f"{r['avg_reach_time']:.0f}" if r['avg_reach_time'] < 1000 else "N/A"
            smooth_str = f"{r['avg_action_rate']:.3f}"

            print("║ {:^6} │ {:^20} │ {:^10} │ {:^10} │ {:^10} │ {:^10} ║".format(
                f"{status} {r['level']}",
                f"{r['spawn_radius']*100:.0f}cm/{r['threshold']*100:.0f}cm",
                sr_str,
                reach_str,
                time_str,
                smooth_str
            ))

            total_reaches += r["total_reaches"]
            total_attempts += r["total_attempts"]

        print("╠" + "═" * 78 + "╣")

        # Overall stats
        overall_sr = total_reaches / max(total_attempts, 1) * 100
        avg_smoothness = np.mean([r["avg_action_rate"] for r in results])
        avg_velocity = np.mean([r["avg_velocity"] for r in results])

        print("║" + " " * 78 + "║")
        print("║  📊 OVERALL STATISTICS:" + " " * 53 + "║")
        print("║" + " " * 78 + "║")
        print("║    Total Reaches:     {:>10} / {:<10}".format(total_reaches, total_attempts) + " " * 34 + "║")
        print("║    Overall Success:   {:>10.1f}%".format(overall_sr) + " " * 45 + "║")
        print("║    Avg Smoothness:    {:>10.4f} (lower = smoother)".format(avg_smoothness) + " " * 22 + "║")
        print("║    Avg Joint Vel:     {:>10.3f}".format(avg_velocity) + " " * 45 + "║")
        print("║" + " " * 78 + "║")

        # Grade
        if overall_sr >= 40:
            grade = "A - Excellent! 🏆"
        elif overall_sr >= 30:
            grade = "B - Good 👍"
        elif overall_sr >= 20:
            grade = "C - Fair 📈"
        elif overall_sr >= 10:
            grade = "D - Needs Work 🔧"
        else:
            grade = "F - Poor ❌"

        print("║    Policy Grade:      {:>30}".format(grade) + " " * 25 + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")

        # Recommendations
        print("\n📝 RECOMMENDATIONS:")

        # Find failing levels
        failing_levels = [r for r in results if r["success_rate"] < 0.25]
        passing_levels = [r for r in results if r["success_rate"] >= 0.50]

        if len(passing_levels) >= 4:
            print("  ✅ Policy performs well on most difficulty levels!")

        if failing_levels:
            max_passing = max([r["spawn_radius"] for r in results if r["success_rate"] >= 0.25], default=0)
            print(f"  ⚠️  Policy struggles at radius > {max_passing*100:.0f}cm")
            print(f"      Consider more training with curriculum starting at {max_passing*100:.0f}cm")

        if avg_smoothness > 0.15:
            print(f"  ⚠️  Action smoothness is high ({avg_smoothness:.3f})")
            print("      Robot movements may be jerky. Consider increasing smoothing penalties.")
        elif avg_smoothness < 0.08:
            print(f"  ✅ Good smoothness ({avg_smoothness:.3f}) - movements are fluid")

        if avg_velocity > 2.0:
            print(f"  ⚠️  High joint velocities ({avg_velocity:.2f}) - may cause oscillation")

        return {
            "overall_success_rate": overall_sr,
            "total_reaches": total_reaches,
            "total_attempts": total_attempts,
            "avg_smoothness": avg_smoothness,
            "avg_velocity": avg_velocity,
            "grade": grade,
        }


def build_actor(obs_dim: int, act_dim: int, checkpoint_path: str, device: str = "cuda:0"):
    """Build actor network and load weights."""

    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, act_dim),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    actor_state = {}
    for key, value in state_dict.items():
        if "actor" in key:
            new_key = key.replace("actor.", "")
            actor_state[new_key] = value

    if actor_state:
        actor.load_state_dict(actor_state)
        print(f"✓ Loaded {len(actor_state)} actor weights from checkpoint")
    else:
        print("⚠️ WARNING: No actor weights found!")

    actor.eval()
    return actor


def main():
    # Environment setup
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # G1 minimum reach ~18cm based on arm kinematics
    env_cfg.workspace_inner_radius = 0.18

    env = G1ArmReachEnv(cfg=env_cfg)

    # Build actor
    actor = build_actor(
        obs_dim=env_cfg.num_observations,
        act_dim=env_cfg.num_actions,
        checkpoint_path=args.checkpoint,
    )

    # Run benchmark
    benchmark = PolicyBenchmark(env, actor)
    results = benchmark.run_full_benchmark(episodes_per_level=args.episodes_per_test)

    # Print summary
    summary = benchmark.print_summary_report(results)

    # Save report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"benchmark_report_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("G1 ARM REACH POLICY BENCHMARK REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Environments: {args.num_envs}\n")
        f.write(f"Episodes per level: {args.episodes_per_test}\n\n")

        f.write("RESULTS BY LEVEL:\n")
        f.write("-" * 50 + "\n")

        for r in results:
            f.write(f"\nLevel {r['level']}: {r['name']}\n")
            f.write(f"  Success Rate: {r['success_rate']*100:.1f}%\n")
            f.write(f"  Reaches: {r['total_reaches']} / {r['total_attempts']}\n")
            f.write(f"  Avg Reach Time: {r['avg_reach_time']:.1f} steps\n")
            f.write(f"  Avg Min Distance: {r['avg_min_distance']*100:.1f}cm\n")
            f.write(f"  Avg Action Rate: {r['avg_action_rate']:.4f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("SUMMARY:\n")
        f.write(f"  Overall Success Rate: {summary['overall_success_rate']:.1f}%\n")
        f.write(f"  Total Reaches: {summary['total_reaches']}\n")
        f.write(f"  Avg Smoothness: {summary['avg_smoothness']:.4f}\n")
        f.write(f"  Grade: {summary['grade']}\n")

    print(f"\n📄 Report saved to: {report_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()