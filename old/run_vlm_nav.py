"""
VLM-Guided Quadruped Navigation
Full Pipeline: Language Command → VLM → Target → RL Policy → Robot Action

Usage:
    # Standalone test (without Isaac Lab)
    python run_vlm_nav.py --test-vlm --image test_scene.png
    python run_vlm_nav.py --test-scene

    # Isaac Lab simulation (from C:\IsaacLab directory)
    cd C:\IsaacLab
    .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\go2_vlm_rl\run_vlm_nav.py --num_envs 4 --enable_cameras
"""

import argparse
import torch
import numpy as np
from typing import Optional
import time
import os


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Navigation Demo")

    # Test modes
    parser.add_argument("--test-vlm", action="store_true", help="Test VLM only")
    parser.add_argument("--test-scene", action="store_true", help="Test scene generation")
    parser.add_argument("--image", type=str, default=None, help="Test image path")
    parser.add_argument("--command", type=str, default="mavi sandalyeye git", help="Navigation command")

    # Isaac Lab args
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--load_policy", type=str, default=None, help="Path to trained policy")
    parser.add_argument("--vlm_interval", type=int, default=20, help="VLM call interval (steps)")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera rendering")

    return parser.parse_args()


def test_vlm_standalone(image_path: str, command: str):
    """Test VLM wrapper standalone."""
    from vlm_wrapper import VLMWrapper, NavigationController
    from PIL import Image

    print("\n" + "=" * 60)
    print("VLM Standalone Test")
    print("=" * 60)

    # Load image
    print(f"\n[1] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    print(f"    Image shape: {image_np.shape}")

    # Initialize VLM
    print("\n[2] Initializing Phi-3-Vision...")
    start = time.time()
    vlm = VLMWrapper(low_memory=True)
    print(f"    Load time: {time.time()-start:.1f}s")

    # Warmup
    print("\n[3] Warming up model...")
    vlm.warmup()

    # Test commands
    test_commands = [
        command,
        "mavi sandalyeye git",
        "go to the red box",
        "yeşil topu bul",
        "kırmızı masaya git",
    ]

    nav = NavigationController()

    print("\n[4] Testing commands:")
    for cmd in test_commands:
        print(f"\n    Command: '{cmd}'")

        start = time.time()
        result = vlm.ground_object(image_np, cmd)
        elapsed = time.time() - start

        print(f"    Time: {elapsed*1000:.1f}ms")
        print(f"    Found: {result['found']}")
        if result['found']:
            print(f"    Position: x={result['x']:.2f}, y={result['y']:.2f}")
            print(f"    Confidence: {result['confidence']:.2f}")

            cmd_vel = nav.target_to_velocity(result)
            print(f"    Velocity: vx={cmd_vel[0]:.2f}, vy={cmd_vel[1]:.2f}, vyaw={cmd_vel[2]:.2f}")

    print("\n" + "=" * 60)
    print("VLM Test Complete!")


def test_scene_generation():
    """Test scene generation."""
    from scene_generator import ObjectSpawner, VLMNavigationEnvHelper

    print("\n" + "=" * 60)
    print("Scene Generation Test")
    print("=" * 60)

    spawner = ObjectSpawner()

    # Generate scenes
    print("\n[1] Generating random scenes...")
    for i in range(3):
        objects = spawner.generate_random_scene(
            num_objects=6,
            arena_size=5.0,
        )
        print(f"\n    Scene {i+1}:")
        print("    " + spawner.get_object_info_for_vlm(objects).replace("\n", "\n    "))

    # Show code sample
    print("\n[2] Sample Isaac Lab config code:")
    print("-" * 40)
    objects = spawner.generate_random_scene(num_objects=4)
    code = spawner.get_spawn_config_code(objects)
    # Show first part
    print(code[:1500])
    print("...")

    # Test helper
    print("\n[3] Testing navigation helper...")
    helper = VLMNavigationEnvHelper(objects)

    for _ in range(3):
        cmd, color, obj = helper.get_random_command()
        target = helper.get_target_position(color, obj)
        print(f"    '{cmd}' → target at {target}")

    print("\n" + "=" * 60)
    print("Scene Generation Test Complete!")


class VLMNavigationPipeline:
    """
    Full VLM navigation pipeline for Isaac Lab.
    """

    def __init__(
        self,
        vlm_model_id: str = "microsoft/Phi-3-vision-128k-instruct",
        policy_path: Optional[str] = None,
        vlm_call_interval: int = 20,
        device: str = "cuda",
    ):
        print("[Pipeline] Initializing VLM Navigation Pipeline...")

        # Import here to avoid loading when not needed
        from vlm_wrapper import VLMWrapper, NavigationController
        from scene_generator import ObjectSpawner, VLMNavigationEnvHelper

        # VLM
        print("[Pipeline] Loading VLM...")
        self.vlm = VLMWrapper(model_id=vlm_model_id, device=device, low_memory=True)
        self.nav_controller = NavigationController()

        # Scene
        self.spawner = ObjectSpawner()
        self.scene_objects = None
        self.env_helper = None

        # RL Policy (placeholder - load actual trained policy)
        self.policy = None
        if policy_path and os.path.exists(policy_path):
            print(f"[Pipeline] Loading policy from {policy_path}...")
            # TODO: Load actual RSL-RL policy
            # self.policy = torch.load(policy_path)

        # State
        self.vlm_call_interval = vlm_call_interval
        self.step_counter = 0
        self.current_target = None
        self.current_command = None

        print("[Pipeline] Initialization complete!")

    def setup_scene(self, num_objects: int = 6, arena_size: float = 5.0):
        """Generate new scene with random objects."""
        from scene_generator import VLMNavigationEnvHelper

        print(f"[Pipeline] Setting up scene with {num_objects} objects...")

        self.scene_objects = self.spawner.generate_random_scene(
            num_objects=num_objects,
            arena_size=arena_size,
        )
        self.env_helper = VLMNavigationEnvHelper(self.scene_objects)

        print("[Pipeline] Scene objects:")
        for obj in self.scene_objects:
            print(f"    - {obj['color']} {obj['type']} at ({obj['position'][0]:.1f}, {obj['position'][1]:.1f})")

        return self.scene_objects

    def set_command(self, command: str):
        """Set navigation command."""
        self.current_command = command
        self.current_target = None  # Force VLM call
        self.step_counter = 0
        print(f"[Pipeline] New command: '{command}'")

    def get_random_command(self) -> str:
        """Get random command from scene objects."""
        if self.env_helper is None:
            raise RuntimeError("Scene not set up. Call setup_scene() first.")
        cmd, _, _ = self.env_helper.get_random_command()
        return cmd

    def step(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        proprioception: np.ndarray,
    ) -> tuple:
        """
        Execute one step of VLM navigation.

        Args:
            rgb_image: RGB camera image (H, W, 3)
            depth_image: Depth camera image (H, W) or (H, W, 1)
            proprioception: Robot proprioceptive state (48 dims)

        Returns:
            (action, target_info, cmd_vel)
        """
        self.step_counter += 1

        # Call VLM periodically
        if (self.step_counter % self.vlm_call_interval == 0
            or self.current_target is None):

            if self.current_command:
                self.current_target = self.vlm.ground_object(
                    rgb_image,
                    self.current_command
                )
                print(f"[VLM] Step {self.step_counter}: found={self.current_target['found']}, "
                      f"x={self.current_target['x']:.2f}, y={self.current_target['y']:.2f}")

        # Convert target to velocity command
        if self.current_target:
            cmd_vel = self.nav_controller.target_to_velocity(self.current_target)
        else:
            cmd_vel = np.array([0.0, 0.0, 0.0])

        # Build observation for RL policy
        if depth_image.ndim == 3:
            depth_flat = depth_image.reshape(-1)
        else:
            depth_flat = depth_image.flatten()

        obs = np.concatenate([proprioception, cmd_vel, depth_flat])

        # Get action from policy
        if self.policy is not None:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to("cuda")
                action = self.policy(obs_tensor).cpu().numpy().squeeze()
        else:
            # Placeholder action
            action = np.zeros(12)  # 12 DoF for Go2

        return action, self.current_target, cmd_vel

    def is_goal_reached(self) -> bool:
        """Check if current navigation goal is reached."""
        if self.current_target is None:
            return False
        return self.nav_controller.is_goal_reached(self.current_target)


def run_isaac_lab_demo(args):
    """Run VLM navigation demo (framework demonstration)."""
    print("\n" + "=" * 60)
    print("Isaac Lab VLM Navigation Demo")
    print("=" * 60)

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  VLM Navigation Framework Demo                            ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  This is a framework demonstration.                       ║
    ║                                                           ║
    ║  To test individual components:                           ║
    ║    python run_vlm_nav.py --test-vlm --image scene.png     ║
    ║    python run_vlm_nav.py --test-scene                     ║
    ║                                                           ║
    ║  For full Isaac Lab integration:                          ║
    ║    1. cd C:\\IsaacLab                                      ║
    ║    2. .\\isaaclab.bat -p <path_to>/run_vlm_nav.py          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Demo: Initialize pipeline
    print("\n[Demo] Initializing pipeline...")

    try:
        pipeline = VLMNavigationPipeline(
            vlm_call_interval=args.vlm_interval,
        )

        # Setup scene
        pipeline.setup_scene(num_objects=6)

        # Get random command
        cmd = pipeline.get_random_command()
        pipeline.set_command(cmd)

        print(f"\n[Demo] Ready to navigate to: '{cmd}'")
        print("[Demo] In full implementation, this would control the robot in Isaac Lab.")

        # Simulate a few steps with dummy data
        print("\n[Demo] Simulating navigation steps...")
        dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_depth = np.random.rand(64, 64).astype(np.float32)
        dummy_proprio = np.zeros(48)

        for i in range(3):
            action, target, cmd_vel = pipeline.step(dummy_rgb, dummy_depth, dummy_proprio)
            print(f"  Step {i+1}: cmd_vel=[{cmd_vel[0]:.2f}, {cmd_vel[1]:.2f}, {cmd_vel[2]:.2f}]")

    except Exception as e:
        print(f"\n[Demo] Error: {e}")
        print("[Demo] This is expected if VLM model is not downloaded yet.")
        print("[Demo] Use --test-vlm or --test-scene for standalone tests.")


def main():
    args = parse_args()

    if args.test_vlm:
        if args.image is None:
            print("Error: --image required for VLM test")
            print("Usage: python run_vlm_nav.py --test-vlm --image test_scene.png")
            return
        test_vlm_standalone(args.image, args.command)

    elif args.test_scene:
        test_scene_generation()

    else:
        run_isaac_lab_demo(args)


if __name__ == "__main__":
    main()