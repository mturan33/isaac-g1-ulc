#!/usr/bin/env python3
"""
G1 Arm Workspace Discovery v3 - FULL RESET PER SAMPLE
======================================================

Her sample sonrası robot tamamen resetleniyor - drift birikmiyor.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/test/test_workspace_discovery.py --num_envs 1 --samples 200
"""

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description="G1 Workspace Discovery v3")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=60.0)
parser.add_argument("--samples", type=int, default=200)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# G1 Arm joint limits (radians) - from URDF
G1_ARM_JOINT_LIMITS = {
    "shoulder_pitch": (-2.97, 2.79),
    "shoulder_roll": (-2.25, 1.59),
    "shoulder_yaw": (-2.62, 2.62),
    "elbow_pitch": (-0.23, 3.42),
    "elbow_roll": (-2.09, 2.09),
}

G1_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
]

G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

ARM_LENGTH_TOTAL = 0.45


@configclass
class WorkspaceSceneCfg(InteractiveSceneCfg):
    """Scene config."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # G1 Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=1000.0,
                angular_damping=1000.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )

    # Hand markers
    left_hand: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LeftHand",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0), emissive_color=(0.5, 0.25, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.2, 1.0)),
    )

    right_hand: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightHand",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8), emissive_color=(0.4, 0.0, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


def find_indices(robot) -> dict:
    """Find body and joint indices."""
    body_names = robot.data.body_names
    joint_names = robot.data.joint_names

    indices = {
        "left_palm": None,
        "right_palm": None,
        "left_shoulder": None,
        "right_shoulder": None,
        "torso": None,
        "left_arm_joints": [],
        "right_arm_joints": [],
    }

    for i, name in enumerate(body_names):
        name_lower = name.lower()
        if "left" in name_lower and "palm" in name_lower:
            indices["left_palm"] = i
        elif "right" in name_lower and "palm" in name_lower:
            indices["right_palm"] = i
        elif "left" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            indices["left_shoulder"] = i
        elif "right" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            indices["right_shoulder"] = i
        elif "torso" in name_lower:
            indices["torso"] = i

    for i, name in enumerate(joint_names):
        if name in G1_LEFT_ARM_JOINTS:
            indices["left_arm_joints"].append(i)
        elif name in G1_RIGHT_ARM_JOINTS:
            indices["right_arm_joints"].append(i)

    return indices


def sample_arm_joints():
    """Sample random arm joint values within limits."""
    joints = []
    for joint_name in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]:
        low, high = G1_ARM_JOINT_LIMITS[joint_name]
        joints.append(np.random.uniform(low, high))
    return joints


def main():
    print("\n" + "=" * 70)
    print("  G1 WORKSPACE DISCOVERY v3 - FULL RESET PER SAMPLE")
    print("=" * 70)
    print(f"""
    Her sample sonrası robot tamamen resetleniyor.
    Root drift ölçülüp palm pozisyonları düzeltiliyor.

    {args.samples} rastgele konfigürasyon test edilecek.
    """)
    print("=" * 70 + "\n")

    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.0, 2.0, 2.0), target=(0.0, 0.0, 1.0))

    # Create scene
    scene_cfg = WorkspaceSceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    device = sim.device

    robot = scene["robot"]
    indices = find_indices(robot)

    print("-" * 70)
    print("DETECTED INDICES:")
    print(f"  Left palm:  {indices['left_palm']}")
    print(f"  Right palm: {indices['right_palm']}")
    print(f"  Left shoulder:  {indices['left_shoulder']}")
    print(f"  Right shoulder: {indices['right_shoulder']}")
    print(f"  Torso: {indices['torso']}")
    print(f"  Left arm joints:  {indices['left_arm_joints']}")
    print(f"  Right arm joints: {indices['right_arm_joints']}")
    print("-" * 70 + "\n")

    # Fixed reference values
    FIXED_ROOT_POS = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    FIXED_ROOT_QUAT = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    FIXED_ROOT_POSE = torch.cat([FIXED_ROOT_POS, FIXED_ROOT_QUAT], dim=-1)
    ZERO_VEL = torch.zeros((1, 6), device=device, dtype=torch.float32)
    ZERO_JOINT_VEL = torch.zeros((1, robot.data.joint_vel.shape[1]), device=device, dtype=torch.float32)

    # Default joint positions (all zeros)
    DEFAULT_JOINT_POS = torch.zeros((1, robot.data.joint_pos.shape[1]), device=device, dtype=torch.float32)

    default_quat = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)

    # Get initial shoulder positions (with robot at origin)
    robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
    robot.write_root_velocity_to_sim(ZERO_VEL)
    robot.write_joint_state_to_sim(DEFAULT_JOINT_POS, ZERO_JOINT_VEL)

    for _ in range(10):
        sim.step()
        robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
        robot.write_root_velocity_to_sim(ZERO_VEL)

    scene.update(sim_cfg.dt)
    robot.update(sim_cfg.dt)

    # Get reference positions
    ref_root_pos = robot.data.root_pos_w[0].cpu().numpy()

    left_shoulder_idx = indices["left_shoulder"]
    right_shoulder_idx = indices["right_shoulder"]
    left_palm_idx = indices["left_palm"]
    right_palm_idx = indices["right_palm"]

    if left_shoulder_idx is not None:
        ref_left_shoulder = robot.data.body_pos_w[0, left_shoulder_idx].cpu().numpy()
    else:
        ref_left_shoulder = ref_root_pos + np.array([0.0, 0.15, 0.35])

    if right_shoulder_idx is not None:
        ref_right_shoulder = robot.data.body_pos_w[0, right_shoulder_idx].cpu().numpy()
    else:
        ref_right_shoulder = ref_root_pos + np.array([0.0, -0.15, 0.35])

    print("-" * 70)
    print("REFERENCE POSITIONS:")
    print(f"  Root:           [{ref_root_pos[0]:.3f}, {ref_root_pos[1]:.3f}, {ref_root_pos[2]:.3f}]")
    print(f"  Left shoulder:  [{ref_left_shoulder[0]:.3f}, {ref_left_shoulder[1]:.3f}, {ref_left_shoulder[2]:.3f}]")
    print(f"  Right shoulder: [{ref_right_shoulder[0]:.3f}, {ref_right_shoulder[1]:.3f}, {ref_right_shoulder[2]:.3f}]")
    print("-" * 70 + "\n")

    # Workspace discovery
    print("=" * 70)
    print("WORKSPACE DISCOVERY - Full reset per sample...")
    print("=" * 70 + "\n")

    left_positions = []  # shoulder-relative
    right_positions = []  # shoulder-relative

    left_joint_ids = indices["left_arm_joints"]
    right_joint_ids = indices["right_arm_joints"]

    valid_samples = 0
    max_drift_seen = 0.0

    for sample_idx in range(args.samples):
        # ====== STEP 1: Full reset ======
        robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
        robot.write_root_velocity_to_sim(ZERO_VEL)
        robot.write_joint_state_to_sim(DEFAULT_JOINT_POS, ZERO_JOINT_VEL)

        # Let it settle
        for _ in range(3):
            sim.step()
            robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
            robot.write_root_velocity_to_sim(ZERO_VEL)

        # ====== STEP 2: Set arm joints ======
        left_joints = sample_arm_joints()
        right_joints = sample_arm_joints()

        joint_pos = DEFAULT_JOINT_POS.clone()

        for i, joint_idx in enumerate(left_joint_ids):
            if i < len(left_joints):
                joint_pos[0, joint_idx] = left_joints[i]

        for i, joint_idx in enumerate(right_joint_ids):
            if i < len(right_joints):
                joint_pos[0, joint_idx] = right_joints[i]

        # Write joint positions
        robot.write_joint_state_to_sim(joint_pos, ZERO_JOINT_VEL)

        # Force root to stay fixed
        robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
        robot.write_root_velocity_to_sim(ZERO_VEL)

        # ====== STEP 3: Single sim step ======
        sim.step()

        # Force root again (counteract physics)
        robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
        robot.write_root_velocity_to_sim(ZERO_VEL)

        # Update robot data
        robot.update(sim_cfg.dt)

        # ====== STEP 4: Measure positions ======
        current_root = robot.data.root_pos_w[0].cpu().numpy()
        root_drift = np.linalg.norm(current_root - ref_root_pos)
        max_drift_seen = max(max_drift_seen, root_drift)

        # Only accept samples with minimal drift
        if root_drift < 0.05:  # Less than 5cm drift
            if left_palm_idx is not None:
                left_palm_world = robot.data.body_pos_w[0, left_palm_idx].cpu().numpy()
                # Convert to shoulder-relative
                left_palm_rel = left_palm_world - ref_left_shoulder
                left_positions.append(left_palm_rel.copy())

            if right_palm_idx is not None:
                right_palm_world = robot.data.body_pos_w[0, right_palm_idx].cpu().numpy()
                right_palm_rel = right_palm_world - ref_right_shoulder
                right_positions.append(right_palm_rel.copy())

            valid_samples += 1

            # Update hand markers
            if left_palm_idx is not None:
                scene["left_hand"].write_root_pose_to_sim(
                    torch.cat([robot.data.body_pos_w[:, left_palm_idx], default_quat], dim=-1)
                )
            if right_palm_idx is not None:
                scene["right_hand"].write_root_pose_to_sim(
                    torch.cat([robot.data.body_pos_w[:, right_palm_idx], default_quat], dim=-1)
                )

        if (sample_idx + 1) % 50 == 0:
            print(
                f"  Sample {sample_idx + 1}/{args.samples} - Valid: {valid_samples} - Max drift: {max_drift_seen * 100:.1f}cm")

    print(f"\n  ✅ Discovery complete! {valid_samples}/{args.samples} valid samples.\n")
    print(f"  Max root drift observed: {max_drift_seen * 100:.1f}cm\n")

    if valid_samples < 10:
        print("  ⚠️ Too few valid samples! Robot is still moving too much.")
        print("  Showing raw results anyway...\n")

    # Compute workspace bounds (shoulder-relative)
    left_arr = np.array(left_positions) if left_positions else np.zeros((1, 3))
    right_arr = np.array(right_positions) if right_positions else np.zeros((1, 3))

    print("=" * 70)
    print("WORKSPACE RESULTS (Shoulder-relative)")
    print("=" * 70)

    if len(left_positions) > 0:
        print("\n📍 LEFT ARM (relative to left shoulder):")
        print(f"   X range: [{left_arr[:, 0].min():.3f}, {left_arr[:, 0].max():.3f}] m")
        print(f"   Y range: [{left_arr[:, 1].min():.3f}, {left_arr[:, 1].max():.3f}] m")
        print(f"   Z range: [{left_arr[:, 2].min():.3f}, {left_arr[:, 2].max():.3f}] m")
        left_radius = np.linalg.norm(left_arr, axis=1).max()
        left_mean_radius = np.linalg.norm(left_arr, axis=1).mean()
        print(f"   Max reach:  {left_radius:.3f} m")
        print(f"   Mean reach: {left_mean_radius:.3f} m")

    if len(right_positions) > 0:
        print("\n📍 RIGHT ARM (relative to right shoulder):")
        print(f"   X range: [{right_arr[:, 0].min():.3f}, {right_arr[:, 0].max():.3f}] m")
        print(f"   Y range: [{right_arr[:, 1].min():.3f}, {right_arr[:, 1].max():.3f}] m")
        print(f"   Z range: [{right_arr[:, 2].min():.3f}, {right_arr[:, 2].max():.3f}] m")
        right_radius = np.linalg.norm(right_arr, axis=1).max()
        right_mean_radius = np.linalg.norm(right_arr, axis=1).mean()
        print(f"   Max reach:  {right_radius:.3f} m")
        print(f"   Mean reach: {right_mean_radius:.3f} m")

    # Convert to root-relative for RL
    # Shoulder offset from root
    left_shoulder_offset = ref_left_shoulder - ref_root_pos
    right_shoulder_offset = ref_right_shoulder - ref_root_pos

    print("\n" + "=" * 70)
    print("WORKSPACE RESULTS (Root-relative, for RL)")
    print("=" * 70)

    print(f"\n📍 Shoulder offsets from root:")
    print(
        f"   Left shoulder:  [{left_shoulder_offset[0]:.3f}, {left_shoulder_offset[1]:.3f}, {left_shoulder_offset[2]:.3f}]")
    print(
        f"   Right shoulder: [{right_shoulder_offset[0]:.3f}, {right_shoulder_offset[1]:.3f}, {right_shoulder_offset[2]:.3f}]")

    if len(left_positions) > 0 and len(right_positions) > 0:
        # Root-relative = shoulder_offset + palm_relative_to_shoulder
        left_root_rel = left_arr + left_shoulder_offset
        right_root_rel = right_arr + right_shoulder_offset

        print("\n📍 LEFT ARM (relative to root):")
        print(f"   X range: [{left_root_rel[:, 0].min():.3f}, {left_root_rel[:, 0].max():.3f}] m")
        print(f"   Y range: [{left_root_rel[:, 1].min():.3f}, {left_root_rel[:, 1].max():.3f}] m")
        print(f"   Z range: [{left_root_rel[:, 2].min():.3f}, {left_root_rel[:, 2].max():.3f}] m")
        left_center = left_root_rel.mean(axis=0)
        print(f"   Center:  [{left_center[0]:.3f}, {left_center[1]:.3f}, {left_center[2]:.3f}]")

        print("\n📍 RIGHT ARM (relative to root):")
        print(f"   X range: [{right_root_rel[:, 0].min():.3f}, {right_root_rel[:, 0].max():.3f}] m")
        print(f"   Y range: [{right_root_rel[:, 1].min():.3f}, {right_root_rel[:, 1].max():.3f}] m")
        print(f"   Z range: [{right_root_rel[:, 2].min():.3f}, {right_root_rel[:, 2].max():.3f}] m")
        right_center = right_root_rel.mean(axis=0)
        print(f"   Center:  [{right_center[0]:.3f}, {right_center[1]:.3f}, {right_center[2]:.3f}]")

        # Practical recommendations
        safe_radius = min(left_radius, right_radius) * 0.7

        print("\n" + "=" * 70)
        print("🎯 RL EĞİTİMİ İÇİN ÖNERİLEN PARAMETRELER")
        print("=" * 70)
        print(f"""
    # Python kodu için kopyala-yapıştır:

    # Shoulder offsets (root-relative)
    LEFT_SHOULDER_OFFSET = [{left_shoulder_offset[0]:.3f}, {left_shoulder_offset[1]:.3f}, {left_shoulder_offset[2]:.3f}]
    RIGHT_SHOULDER_OFFSET = [{right_shoulder_offset[0]:.3f}, {right_shoulder_offset[1]:.3f}, {right_shoulder_offset[2]:.3f}]

    # Workspace centers (root-relative)
    LEFT_ARM_CENTER = [{left_center[0]:.3f}, {left_center[1]:.3f}, {left_center[2]:.3f}]
    RIGHT_ARM_CENTER = [{right_center[0]:.3f}, {right_center[1]:.3f}, {right_center[2]:.3f}]

    # Reach limits
    MAX_REACH = {min(left_radius, right_radius):.3f}  # metre
    SAFE_RADIUS = {safe_radius:.3f}  # 70% of max, for RL training

    # RL target sampling (root-relative):
    def sample_target(arm="right"):
        center = RIGHT_ARM_CENTER if arm == "right" else LEFT_ARM_CENTER
        direction = normalize(random_vector())
        distance = random() * SAFE_RADIUS
        return center + direction * distance
        """)

    # Reset robot and show visualization
    print("\nResetting robot for visualization...")
    robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
    robot.write_root_velocity_to_sim(ZERO_VEL)
    robot.write_joint_state_to_sim(DEFAULT_JOINT_POS, ZERO_JOINT_VEL)

    print(f"\nVisualization running for {args.duration}s...")
    print("Robot should be STATIONARY now.")
    print("Press Ctrl+C to exit.\n")

    step_count = 0
    max_steps = int(args.duration / sim_cfg.dt)

    try:
        while simulation_app.is_running() and step_count < max_steps:
            # Keep root fixed during visualization
            robot.write_root_pose_to_sim(FIXED_ROOT_POSE)
            robot.write_root_velocity_to_sim(ZERO_VEL)

            sim.step()
            scene.update(sim_cfg.dt)
            robot.update(sim_cfg.dt)

            # Update hand markers
            if left_palm_idx is not None:
                scene["left_hand"].write_root_pose_to_sim(
                    torch.cat([robot.data.body_pos_w[:, left_palm_idx], default_quat], dim=-1)
                )
            if right_palm_idx is not None:
                scene["right_hand"].write_root_pose_to_sim(
                    torch.cat([robot.data.body_pos_w[:, right_palm_idx], default_quat], dim=-1)
                )

            step_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    print("\n" + "=" * 70)
    print("WORKSPACE DISCOVERY COMPLETE")
    print("=" * 70)

    sim.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()