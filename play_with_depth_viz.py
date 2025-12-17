# play_with_depth_viz.py
"""
Student policy test with live depth camera visualization.
"""

import argparse
import torch
import numpy as np
import cv2
from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-DepthDistill-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)  # Single env for visualization
parser.add_argument("--checkpoint", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # Required for depth

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import isaaclab.envs as envs
import gymnasium as gym


def colorize_depth(depth_image, min_depth=0.1, max_depth=5.0):
    """Convert depth to colormap for visualization."""
    depth_np = depth_image.cpu().numpy().squeeze()

    # Normalize to 0-255
    depth_normalized = np.clip((depth_np - min_depth) / (max_depth - min_depth), 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # Apply colormap (TURBO for depth)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    return depth_colored


def main():
    # Create environment
    env = gym.make(args.task, num_envs=args.num_envs)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cuda")

    # Get policy (adjust based on your checkpoint structure)
    policy = checkpoint.get("model_state_dict") or checkpoint.get("actor")
    # ... setup policy network ...

    # Create OpenCV window
    cv2.namedWindow("Depth Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Camera", 512, 512)

    obs, _ = env.reset()

    print("Press 'q' to quit, 's' to save screenshot")

    step = 0
    while simulation_app.is_running():
        # Get action from policy
        with torch.no_grad():
            # Extract student obs (depth included)
            student_obs = obs["policy"] if isinstance(obs, dict) else obs
            # ... compute action ...
            action = torch.zeros(args.num_envs, 12, device="cuda")  # placeholder

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get depth image from camera sensor
        # Access via env.scene.sensors["depth_camera"] or similar
        if hasattr(env.unwrapped, "scene"):
            scene = env.unwrapped.scene
            if "depth_camera" in scene.sensors:
                depth_data = scene.sensors["depth_camera"].data.output["distance_to_camera"]

                # Visualize first environment's depth
                depth_viz = colorize_depth(depth_data[0])

                # Add info overlay
                cv2.putText(depth_viz, f"Step: {step}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_viz, f"Reward: {reward[0].item():.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Depth Camera", depth_viz)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"depth_frame_{step}.png", depth_viz)
            print(f"Saved depth_frame_{step}.png")

        step += 1

    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()