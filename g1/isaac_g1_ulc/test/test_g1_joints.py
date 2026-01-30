"""
Test script to inspect G1 robot joints in Isaac Lab
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg

# Use existing G1 config
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

class TestSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 2.0], [0.0, 0.0, 0.75])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]

    print("\n" + "="*80)
    print("G1 ROBOT JOINT ANALYSIS")
    print("="*80)

    print(f"\nTotal joints: {robot.num_joints}")
    print(f"Total bodies: {robot.num_bodies}")

    print("\n--- ALL JOINT NAMES ---")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    print("\n--- ALL BODY NAMES ---")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {name}")

    # Find gripper/hand related
    print("\n--- GRIPPER/HAND RELATED JOINTS ---")
    for i, name in enumerate(robot.joint_names):
        if any(kw in name.lower() for kw in ['grip', 'hand', 'finger', 'palm', 'wrist']):
            print(f"  Joint [{i:2d}] {name}")

    print("\n--- GRIPPER/HAND RELATED BODIES ---")
    for i, name in enumerate(robot.body_names):
        if any(kw in name.lower() for kw in ['grip', 'hand', 'finger', 'palm', 'wrist']):
            print(f"  Body [{i:2d}] {name}")

    # Find right arm
    print("\n--- RIGHT ARM JOINTS ---")
    for i, name in enumerate(robot.joint_names):
        if 'right' in name.lower() and any(kw in name.lower() for kw in ['shoulder', 'elbow', 'wrist']):
            print(f"  [{i:2d}] {name}")

    print("\n--- RIGHT ARM BODIES ---")
    for i, name in enumerate(robot.body_names):
        if 'right' in name.lower():
            print(f"  [{i:2d}] {name}")

    # Find left arm
    print("\n--- LEFT ARM JOINTS ---")
    for i, name in enumerate(robot.joint_names):
        if 'left' in name.lower() and any(kw in name.lower() for kw in ['shoulder', 'elbow', 'wrist']):
            print(f"  [{i:2d}] {name}")

    print("\n--- LEFT ARM BODIES ---")
    for i, name in enumerate(robot.body_names):
        if 'left' in name.lower():
            print(f"  [{i:2d}] {name}")

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)

    # Run a few steps
    for _ in range(100):
        sim.step()
        scene.update(sim_cfg.dt)

    simulation_app.close()

if __name__ == "__main__":
    main()