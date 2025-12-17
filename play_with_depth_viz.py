"""
Student policy test with live depth camera visualization.
Fixed version for Isaac Lab Manager-Based environments.
"""

import argparse
import torch
import numpy as np

# Parse arguments BEFORE importing Isaac modules
parser = argparse.ArgumentParser(description="Play trained policy with depth visualization")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-DepthDistill-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_length", type=int, default=1000, help="Number of steps to run")

# Add Isaac Sim launcher args
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# CRITICAL: Enable cameras for depth
args.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules (after app launch)
import isaaclab.envs as envs
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab_tasks  # This registers all tasks
import gymnasium as gym
import cv2
import os


def colorize_depth(depth_tensor, min_depth=0.1, max_depth=5.0):
    """
    Convert depth tensor to colorized image for visualization.

    Args:
        depth_tensor: Depth values tensor (N, H, W, 1) or (H, W, 1)
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization

    Returns:
        Colorized depth image as numpy array (H, W, 3) BGR format
    """
    # Handle batch dimension
    if depth_tensor.dim() == 4:
        depth_np = depth_tensor[0].cpu().numpy().squeeze()  # First env
    else:
        depth_np = depth_tensor.cpu().numpy().squeeze()

    # Handle inf values (beyond max range)
    depth_np = np.nan_to_num(depth_np, nan=max_depth, posinf=max_depth, neginf=min_depth)

    # Normalize to 0-255
    depth_normalized = np.clip((depth_np - min_depth) / (max_depth - min_depth), 0, 1)
    depth_uint8 = (255 * (1 - depth_normalized)).astype(np.uint8)  # Invert: close=bright

    # Apply TURBO colormap (good for depth)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    # Upscale for better visibility
    depth_colored = cv2.resize(depth_colored, (256, 256), interpolation=cv2.INTER_NEAREST)

    return depth_colored


def main():
    # Get environment config from registry
    env_cfg = envs.ManagerBasedRLEnvCfg.from_registry(args.task)
    env_cfg.scene.num_envs = args.num_envs

    # Disable episode termination for continuous viewing
    env_cfg.terminations.time_out = None

    # Create environment (Isaac Lab way, not gym.make)
    env = envs.ManagerBasedRLEnv(cfg=env_cfg)

    print(f"[INFO] Environment created: {args.task}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"[INFO] Checkpoint loaded from: {args.checkpoint}")
    print(f"[INFO] Checkpoint keys: {checkpoint.keys()}")

    # Extract policy network
    # RSL-RL saves model differently than custom implementations
    if "model_state_dict" in checkpoint:
        # RSL-RL format
        from rsl_rl.modules import ActorCritic

        # Get observation dimensions
        obs_dim = env.observation_space["policy"].shape[0]
        action_dim = env.action_space.shape[0]

        # Create network (adjust hidden dims based on your config)
        policy = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
        ).to(device)

        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()

        # Get observation normalizer if available
        obs_normalizer = None
        if "obs_normalizer" in checkpoint:
            from rsl_rl.utils import EmpiricalNormalization
            obs_normalizer = EmpiricalNormalization(obs_dim, device=device)
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
    else:
        print("[WARNING] Unknown checkpoint format, attempting direct load...")
        policy = checkpoint

    # Create OpenCV window
    cv2.namedWindow("Depth Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Camera View", 512, 512)

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    print("\n" + "=" * 60)
    print("DEPTH CAMERA VISUALIZATION")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset environment")
    print("=" * 60 + "\n")

    step = 0
    total_reward = 0
    saved_frames = 0

    # Create video writer for recording
    video_dir = os.path.dirname(args.checkpoint)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        os.path.join(video_dir, "depth_visualization.mp4"),
        fourcc, 30.0, (256, 256)
    )

    while simulation_app.is_running() and step < args.video_length:
        # Normalize observations if normalizer available
        if obs_normalizer is not None:
            obs_normalized = obs_normalizer.normalize(obs)
        else:
            obs_normalized = obs

        # Get action from policy (deterministic for evaluation)
        with torch.no_grad():
            if hasattr(policy, 'act'):
                actions = policy.act(obs_normalized)
            elif hasattr(policy, 'forward'):
                actions = policy(obs_normalized)[0]  # Actor output
            else:
                # Fallback: random actions
                actions = torch.zeros(args.num_envs, env.action_space.shape[0], device=device)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"]
        total_reward += rewards[0].item()

        # Get depth image from camera sensor
        depth_viz = None

        # Try to access depth camera from scene sensors
        if hasattr(env, 'scene') and hasattr(env.scene, 'sensors'):
            sensors = env.scene.sensors

            # Find depth camera (name might vary)
            for sensor_name, sensor in sensors.items():
                if 'depth' in sensor_name.lower() or 'camera' in sensor_name.lower():
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                        depth_data = sensor.data.output.get("distance_to_camera")
                        if depth_data is not None:
                            depth_viz = colorize_depth(depth_data)
                            break

        # If no depth sensor found, try observation directly
        if depth_viz is None:
            # Student observations contain flattened depth (4096 = 64x64)
            # Extract and reshape
            prop_dim = 48  # Proprioception dimensions
            depth_start = prop_dim
            depth_end = prop_dim + 64 * 64

            if obs.shape[1] >= depth_end:
                depth_flat = obs[0, depth_start:depth_end]
                depth_img = depth_flat.reshape(64, 64, 1)
                depth_viz = colorize_depth(depth_img)

        # Create display with info overlay
        if depth_viz is not None:
            display = depth_viz.copy()

            # Add text overlay
            cv2.putText(display, f"Step: {step}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Reward: {rewards[0].item():.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Total: {total_reward:.2f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add colorbar legend
            cv2.putText(display, "Near", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(display, "Far", (220, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)

            cv2.imshow("Depth Camera View", display)
            video_writer.write(display)
        else:
            # Show placeholder
            placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No depth data", (50, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Depth Camera View", placeholder)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quit requested")
            break
        elif key == ord('s'):
            filename = f"depth_frame_{saved_frames:04d}.png"
            cv2.imwrite(filename, display if depth_viz is not None else placeholder)
            print(f"[INFO] Saved: {filename}")
            saved_frames += 1
        elif key == ord('r'):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]
            total_reward = 0
            print("[INFO] Environment reset")

        # Handle episode termination
        if terminated.any() or truncated.any():
            print(f"[INFO] Episode ended at step {step}, total reward: {total_reward:.2f}")
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]
            total_reward = 0

        step += 1

        # Progress update
        if step % 100 == 0:
            print(f"[INFO] Step {step}/{args.video_length}")

    # Cleanup
    video_writer.release()
    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()

    print("\n" + "=" * 60)
    print(f"Visualization complete!")
    print(f"Total steps: {step}")
    print(f"Video saved to: {os.path.join(video_dir, 'depth_visualization.mp4')}")
    print("=" * 60)


if __name__ == "__main__":
    main()