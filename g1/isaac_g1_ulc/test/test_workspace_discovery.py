#!/usr/bin/env python3
"""
G1 Arm Workspace Discovery & Visualization
===========================================

Bu script G1 robotunun kol workspace'ini keşfeder ve saydam küreler ile görselleştirir.

ÖZELLİKLER:
1. Simülasyonda joint limitlerini kullanarak workspace keşfi
2. Her kol için ayrı saydam küre ile workspace görselleştirme
3. Pratik reach zone gösterimi (safety margin ile)

KÜRELER:
🔵 MAVİ saydam küre: SOL kol workspace (merkez: sol omuz)
🟢 YEŞİL saydam küre: SAĞ kol workspace (merkez: sağ omuz)
🟡 SARI küçük küreler: Keşfedilen palm pozisyonları (sampling)
🟠 TURUNCU küre: Sol palm pozisyonu
🟣 MOR küre: Sağ palm pozisyonu

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/test/test_workspace_discovery.py --num_envs 1
"""

import argparse
import torch
import numpy as np
import math

# Argument parser
parser = argparse.ArgumentParser(description="G1 Workspace Discovery")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=120.0, help="Test duration in seconds")
parser.add_argument("--samples", type=int, default=500, help="Number of random joint samples")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac imports (after AppLauncher)
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Nucleus path for G1
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# ==================== G1 ARM SPECIFICATIONS ====================

# Joint limits (radians) - from our previous tests
G1_ARM_JOINT_LIMITS = {
    "shoulder_pitch": (-2.97, 2.79),  # [-170°, 160°]
    "shoulder_roll": (-2.25, 1.59),  # [-129°, 91°] - asymmetric!
    "shoulder_yaw": (-2.62, 2.62),  # [-150°, 150°]
    "elbow_pitch": (-0.23, 3.42),  # [-13°, 196°] - mostly positive
    "elbow_roll": (-2.09, 2.09),  # [-120°, 120°]
}

# Joint names
G1_ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ],
}

# Arm dimensions (approximate from testing)
ARM_LENGTH_UPPER = 0.25  # shoulder to elbow
ARM_LENGTH_LOWER = 0.20  # elbow to palm
ARM_LENGTH_TOTAL = ARM_LENGTH_UPPER + ARM_LENGTH_LOWER  # ~0.45m

# Shoulder offsets from torso (approximate)
LEFT_SHOULDER_OFFSET = torch.tensor([0.0, 0.15, 0.35])  # relative to root
RIGHT_SHOULDER_OFFSET = torch.tensor([0.0, -0.15, 0.35])


@configclass
class WorkspaceDiscoverySceneCfg(InteractiveSceneCfg):
    """Scene configuration for workspace discovery."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # G1 Robot - GRAVITY DISABLED
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=5.0,
                angular_damping=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=200.0,  # Higher for faster settling
                damping=20.0,
            ),
        },
    )

    # ========== WORKSPACE VISUALIZATION SPHERES ==========

    # LEFT ARM WORKSPACE - Blue transparent sphere
    left_workspace: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftWorkspace",
        spawn=sim_utils.SphereCfg(
            radius=ARM_LENGTH_TOTAL * 0.9,  # Slightly smaller than max reach
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.3, 0.8),  # Blue
                opacity=0.2,  # Transparent
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 1.15)),
    )

    # RIGHT ARM WORKSPACE - Green transparent sphere
    right_workspace: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightWorkspace",
        spawn=sim_utils.SphereCfg(
            radius=ARM_LENGTH_TOTAL * 0.9,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 0.3),  # Green
                opacity=0.2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.3, 1.15)),
    )

    # PRACTICAL WORKSPACE - Yellow transparent (smaller, safe zone)
    practical_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PracticalLeft",
        spawn=sim_utils.SphereCfg(
            radius=ARM_LENGTH_TOTAL * 0.6,  # 60% of max reach = safe zone
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.0),  # Yellow
                opacity=0.15,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 1.15)),
    )

    practical_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PracticalRight",
        spawn=sim_utils.SphereCfg(
            radius=ARM_LENGTH_TOTAL * 0.6,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.0),
                opacity=0.15,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.3, 1.15)),
    )

    # Hand tracking spheres
    left_hand_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftHandMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # Orange
                emissive_color=(0.5, 0.25, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.2, 1.0)),
    )

    right_hand_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightHandMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.0, 0.8),  # Purple
                emissive_color=(0.4, 0.0, 0.4),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


def find_body_and_joint_indices(robot) -> dict:
    """Find body and joint indices."""
    body_names = robot.data.body_names
    joint_names = robot.data.joint_names

    indices = {
        "left_shoulder": None,
        "right_shoulder": None,
        "left_palm": None,
        "right_palm": None,
        "torso": None,
        "arm_joints": {"left": [], "right": []},
    }

    # Find body indices
    for i, name in enumerate(body_names):
        name_lower = name.lower()
        if "left" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            indices["left_shoulder"] = i
        elif "right" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            indices["right_shoulder"] = i
        elif "left" in name_lower and "palm" in name_lower:
            indices["left_palm"] = i
        elif "right" in name_lower and "palm" in name_lower:
            indices["right_palm"] = i
        elif "torso" in name_lower:
            indices["torso"] = i

    # Find joint indices
    for i, name in enumerate(joint_names):
        for side in ["left", "right"]:
            if name in G1_ARM_JOINTS[side]:
                indices["arm_joints"][side].append(i)

    return indices


def sample_random_arm_pose():
    """Generate random arm joint positions within limits."""
    joints = []
    for joint_name in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]:
        low, high = G1_ARM_JOINT_LIMITS[joint_name]
        joints.append(np.random.uniform(low, high))
    return np.array(joints)


def compute_workspace_bounds(positions: np.ndarray) -> dict:
    """Compute workspace bounding box from sampled positions."""
    if len(positions) == 0:
        return None

    pos_array = np.array(positions)

    return {
        "min_x": pos_array[:, 0].min(),
        "max_x": pos_array[:, 0].max(),
        "min_y": pos_array[:, 1].min(),
        "max_y": pos_array[:, 1].max(),
        "min_z": pos_array[:, 2].min(),
        "max_z": pos_array[:, 2].max(),
        "center": pos_array.mean(axis=0),
        "radius": np.linalg.norm(pos_array - pos_array.mean(axis=0), axis=1).max(),
    }


def main():
    """Main function."""

    print("\n" + "=" * 70)
    print("  G1 ARM WORKSPACE DISCOVERY & VISUALIZATION")
    print("=" * 70)
    print(f"""
    Bu test G1 kollarının workspace'ini keşfeder ve görselleştirir.

    WORKSPACE KÜRELERİ:
    🔵 MAVİ saydam   = Sol kol MAX workspace (~{ARM_LENGTH_TOTAL:.2f}m radius)
    🟢 YEŞİL saydam  = Sağ kol MAX workspace
    🟡 SARI saydam   = Pratik/güvenli workspace (~{ARM_LENGTH_TOTAL * 0.6:.2f}m radius)

    EL TAKİP:
    🟠 TURUNCU = Sol palm pozisyonu
    🟣 MOR     = Sağ palm pozisyonu

    KEŞIF:
    • {args.samples} rastgele joint konfigürasyonu test edilecek
    • Her kol için workspace sınırları hesaplanacak
    """)
    print("=" * 70 + "\n")

    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 2.0), target=(0.0, 0.0, 1.0))

    # Create scene
    scene_cfg = WorkspaceDiscoverySceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    # Reset and play
    sim.reset()
    device = sim.device

    # Find indices
    robot = scene["robot"]
    indices = find_body_and_joint_indices(robot)

    print("-" * 70)
    print("DETECTED INDICES:")
    print("-" * 70)
    print(f"  Left shoulder:  {indices['left_shoulder']}")
    print(f"  Right shoulder: {indices['right_shoulder']}")
    print(f"  Left palm:      {indices['left_palm']}")
    print(f"  Right palm:     {indices['right_palm']}")
    print(f"  Torso:          {indices['torso']}")
    print(f"  Left arm joints:  {indices['arm_joints']['left']}")
    print(f"  Right arm joints: {indices['arm_joints']['right']}")
    print("-" * 70 + "\n")

    left_palm_idx = indices["left_palm"]
    right_palm_idx = indices["right_palm"]
    left_shoulder_idx = indices["left_shoulder"]
    right_shoulder_idx = indices["right_shoulder"]

    # Get initial positions
    robot.update(sim.get_physics_dt())
    root_pos = robot.data.root_pos_w[0].cpu().numpy()

    # Position workspace spheres at shoulder locations
    if left_shoulder_idx is not None:
        left_shoulder_pos = robot.data.body_pos_w[0, left_shoulder_idx].cpu().numpy()
    else:
        left_shoulder_pos = root_pos + np.array([0.0, 0.15, 0.35])

    if right_shoulder_idx is not None:
        right_shoulder_pos = robot.data.body_pos_w[0, right_shoulder_idx].cpu().numpy()
    else:
        right_shoulder_pos = root_pos + np.array([0.0, -0.15, 0.35])

    print("-" * 70)
    print("SHOULDER POSITIONS (Workspace Centers):")
    print("-" * 70)
    print(f"  Left shoulder:  [{left_shoulder_pos[0]:.3f}, {left_shoulder_pos[1]:.3f}, {left_shoulder_pos[2]:.3f}]")
    print(f"  Right shoulder: [{right_shoulder_pos[0]:.3f}, {right_shoulder_pos[1]:.3f}, {right_shoulder_pos[2]:.3f}]")
    print("-" * 70 + "\n")

    # Update workspace sphere positions
    default_quat = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)

    scene["left_workspace"].write_root_pose_to_sim(
        torch.cat([torch.tensor([left_shoulder_pos], device=device, dtype=torch.float32), default_quat], dim=-1)
    )
    scene["right_workspace"].write_root_pose_to_sim(
        torch.cat([torch.tensor([right_shoulder_pos], device=device, dtype=torch.float32), default_quat], dim=-1)
    )
    scene["practical_left"].write_root_pose_to_sim(
        torch.cat([torch.tensor([left_shoulder_pos], device=device, dtype=torch.float32), default_quat], dim=-1)
    )
    scene["practical_right"].write_root_pose_to_sim(
        torch.cat([torch.tensor([right_shoulder_pos], device=device, dtype=torch.float32), default_quat], dim=-1)
    )

    # Workspace discovery - sample random joint configurations
    print("=" * 70)
    print("WORKSPACE DISCOVERY - Sampling random joint configurations...")
    print("=" * 70 + "\n")

    left_positions = []
    right_positions = []

    left_joint_ids = indices["arm_joints"]["left"]
    right_joint_ids = indices["arm_joints"]["right"]

    # Store original joint positions
    original_joint_pos = robot.data.joint_pos.clone()

    sample_count = 0
    settle_steps = 10  # Steps to let physics settle

    for sample_idx in range(args.samples):
        # Generate random joint poses
        left_joints = sample_random_arm_pose()
        right_joints = sample_random_arm_pose()

        # Apply to robot
        new_joint_pos = original_joint_pos.clone()

        for i, joint_idx in enumerate(left_joint_ids):
            if i < len(left_joints):
                new_joint_pos[0, joint_idx] = left_joints[i]

        for i, joint_idx in enumerate(right_joint_ids):
            if i < len(right_joints):
                new_joint_pos[0, joint_idx] = right_joints[i]

        # Write to sim
        robot.write_joint_state_to_sim(new_joint_pos, robot.data.joint_vel)

        # Step simulation to let it settle
        for _ in range(settle_steps):
            sim.step()
            scene.update(sim_cfg.dt)
            robot.update(sim_cfg.dt)

        # Read palm positions
        if left_palm_idx is not None:
            left_pos = robot.data.body_pos_w[0, left_palm_idx].cpu().numpy()
            left_positions.append(left_pos.copy())

        if right_palm_idx is not None:
            right_pos = robot.data.body_pos_w[0, right_palm_idx].cpu().numpy()
            right_positions.append(right_pos.copy())

        # Update hand markers
        if left_palm_idx is not None:
            scene["left_hand_marker"].write_root_pose_to_sim(
                torch.cat([robot.data.body_pos_w[:, left_palm_idx], default_quat.expand(1, -1)], dim=-1)
            )
        if right_palm_idx is not None:
            scene["right_hand_marker"].write_root_pose_to_sim(
                torch.cat([robot.data.body_pos_w[:, right_palm_idx], default_quat.expand(1, -1)], dim=-1)
            )

        sample_count += 1

        # Progress update
        if (sample_idx + 1) % 50 == 0:
            print(f"  Sampled {sample_idx + 1}/{args.samples} configurations...")

    print(f"\n  ✅ Workspace discovery complete! {sample_count} samples collected.\n")

    # Compute workspace bounds
    left_bounds = compute_workspace_bounds(left_positions)
    right_bounds = compute_workspace_bounds(right_positions)

    print("=" * 70)
    print("WORKSPACE ANALYSIS RESULTS")
    print("=" * 70)

    if left_bounds:
        print("\n📍 LEFT ARM WORKSPACE (relative to world):")
        print(f"   X range: [{left_bounds['min_x']:.3f}, {left_bounds['max_x']:.3f}] m")
        print(f"   Y range: [{left_bounds['min_y']:.3f}, {left_bounds['max_y']:.3f}] m")
        print(f"   Z range: [{left_bounds['min_z']:.3f}, {left_bounds['max_z']:.3f}] m")
        print(
            f"   Center:  [{left_bounds['center'][0]:.3f}, {left_bounds['center'][1]:.3f}, {left_bounds['center'][2]:.3f}]")
        print(f"   Radius:  {left_bounds['radius']:.3f} m")

    if right_bounds:
        print("\n📍 RIGHT ARM WORKSPACE (relative to world):")
        print(f"   X range: [{right_bounds['min_x']:.3f}, {right_bounds['max_x']:.3f}] m")
        print(f"   Y range: [{right_bounds['min_y']:.3f}, {right_bounds['max_y']:.3f}] m")
        print(f"   Z range: [{right_bounds['min_z']:.3f}, {right_bounds['max_z']:.3f}] m")
        print(
            f"   Center:  [{right_bounds['center'][0]:.3f}, {right_bounds['center'][1]:.3f}, {right_bounds['center'][2]:.3f}]")
        print(f"   Radius:  {right_bounds['radius']:.3f} m")

    # Convert to root-relative coordinates
    print("\n" + "=" * 70)
    print("WORKSPACE RELATIVE TO ROOT (for RL training)")
    print("=" * 70)

    if left_bounds:
        left_rel_center = left_bounds['center'] - root_pos
        print("\n📍 LEFT ARM (root-relative):")
        print(f"   X range: [{left_bounds['min_x'] - root_pos[0]:.3f}, {left_bounds['max_x'] - root_pos[0]:.3f}] m")
        print(f"   Y range: [{left_bounds['min_y'] - root_pos[1]:.3f}, {left_bounds['max_y'] - root_pos[1]:.3f}] m")
        print(f"   Z range: [{left_bounds['min_z'] - root_pos[2]:.3f}, {left_bounds['max_z'] - root_pos[2]:.3f}] m")
        print(f"   Center:  [{left_rel_center[0]:.3f}, {left_rel_center[1]:.3f}, {left_rel_center[2]:.3f}]")

    if right_bounds:
        right_rel_center = right_bounds['center'] - root_pos
        print("\n📍 RIGHT ARM (root-relative):")
        print(f"   X range: [{right_bounds['min_x'] - root_pos[0]:.3f}, {right_bounds['max_x'] - root_pos[0]:.3f}] m")
        print(f"   Y range: [{right_bounds['min_y'] - root_pos[1]:.3f}, {right_bounds['max_y'] - root_pos[1]:.3f}] m")
        print(f"   Z range: [{right_bounds['min_z'] - root_pos[2]:.3f}, {right_bounds['max_z'] - root_pos[2]:.3f}] m")
        print(f"   Center:  [{right_rel_center[0]:.3f}, {right_rel_center[1]:.3f}, {right_rel_center[2]:.3f}]")

    # Practical workspace recommendation
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDED PRACTICAL WORKSPACE FOR RL TRAINING")
    print("=" * 70)

    if left_bounds and right_bounds:
        # Use 60% of radius for safety
        safe_radius = min(left_bounds['radius'], right_bounds['radius']) * 0.6

        print(f"""
    Güvenli workspace parametreleri (root-relative):

    SOL KOL:
    ├── Merkez: [{left_rel_center[0]:.2f}, {left_rel_center[1]:.2f}, {left_rel_center[2]:.2f}]
    └── Yarıçap: {safe_radius:.2f} m

    SAĞ KOL:
    ├── Merkez: [{right_rel_center[0]:.2f}, {right_rel_center[1]:.2f}, {right_rel_center[2]:.2f}]
    └── Yarıçap: {safe_radius:.2f} m

    RL EĞİTİMİ İÇİN TARGET SAMPLING:
    ```python
    # Sol kol için hedef
    left_target = left_center + random_point_in_sphere(radius={safe_radius:.2f})

    # Sağ kol için hedef  
    right_target = right_center + random_point_in_sphere(radius={safe_radius:.2f})
    ```
        """)

    # Reset robot to initial pose
    print("\nResetting robot to initial pose for visualization...")
    robot.write_joint_state_to_sim(original_joint_pos, torch.zeros_like(robot.data.joint_vel))

    # Visualization loop
    print(f"\nRunning visualization for {args.duration} seconds...")
    print("Workspace küreleri görünür olmalı:")
    print("  • MAVİ/YEŞİL büyük küreler = Max reach")
    print("  • SARI küçük küreler = Güvenli/pratik reach")
    print("\nPress Ctrl+C to exit.\n")

    step_count = 0
    max_steps = int(args.duration / sim_cfg.dt)

    try:
        while simulation_app.is_running() and step_count < max_steps:
            sim.step()
            scene.update(sim_cfg.dt)
            robot.update(sim_cfg.dt)

            # Update hand markers
            if left_palm_idx is not None:
                scene["left_hand_marker"].write_root_pose_to_sim(
                    torch.cat([robot.data.body_pos_w[:, left_palm_idx], default_quat.expand(1, -1)], dim=-1)
                )
            if right_palm_idx is not None:
                scene["right_hand_marker"].write_root_pose_to_sim(
                    torch.cat([robot.data.body_pos_w[:, right_palm_idx], default_quat.expand(1, -1)], dim=-1)
                )

            step_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Final summary
    print("\n" + "=" * 70)
    print("WORKSPACE DISCOVERY COMPLETE")
    print("=" * 70)

    sim.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()