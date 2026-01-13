#!/usr/bin/env python3
"""
G1 Head & Hand Coordinate System Test v3
=========================================

Bu script G1 robotunun kafası ve elleri etrafındaki koordinat sistemini test eder.

KAFA ETRAFINDAKİ KÜRELER (50cm mesafe):
- KIRMIZI: Kafanın 50cm ÖNÜnde (+X, forward)
- MAVİ: Kafanın 50cm ARKASINda (-X, backward)
- YEŞİL: Kafanın 50cm SOLunda (+Y, left)
- SARI: Kafanın 50cm SAĞında (-Y, right)

EL POZİSYON KÜRELERİ (küçük, el takibi):
- TURUNCU: Sol el pozisyonu
- MOR: Sağ el pozisyonu

Robot yerçekimi kapalı - düşmeyecek.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/test/test_head_coordinate_system.py --num_envs 1
"""

import argparse
import torch
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="G1 Head & Hand Coordinate System Test")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac imports (after AppLauncher)
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Nucleus path for G1
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"

# Sphere sizes
HEAD_SPHERE_RADIUS = 0.03  # 3cm - head reference spheres
HAND_SPHERE_RADIUS = 0.025  # 2.5cm - hand tracking spheres


@configclass
class CoordinateTestSceneCfg(InteractiveSceneCfg):
    """Scene configuration for coordinate system test."""

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

    # G1 Robot - SIMPLE CONFIG (no articulation_props to avoid errors)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # NO GRAVITY - robot floats
                linear_damping=5.0,
                angular_damping=5.0,
            ),
            # NO articulation_props - let it use defaults
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    # ========== HEAD REFERENCE SPHERES (50cm from head) ==========

    # RED sphere - FRONT (+X direction)
    red_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedSphere_Front",
        spawn=sim_utils.SphereCfg(
            radius=HEAD_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # RED
                emissive_color=(0.3, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.2)),
    )

    # BLUE sphere - BACK (-X direction)
    blue_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueSphere_Back",
        spawn=sim_utils.SphereCfg(
            radius=HEAD_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # BLUE
                emissive_color=(0.0, 0.0, 0.3),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 1.2)),
    )

    # GREEN sphere - LEFT (+Y direction)
    green_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GreenSphere_Left",
        spawn=sim_utils.SphereCfg(
            radius=HEAD_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # GREEN
                emissive_color=(0.0, 0.3, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 1.2)),
    )

    # YELLOW sphere - RIGHT (-Y direction)
    yellow_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/YellowSphere_Right",
        spawn=sim_utils.SphereCfg(
            radius=HEAD_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),  # YELLOW
                emissive_color=(0.3, 0.3, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 1.2)),
    )

    # ========== HAND TRACKING SPHERES ==========

    # ORANGE sphere - LEFT HAND
    orange_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OrangeSphere_LeftHand",
        spawn=sim_utils.SphereCfg(
            radius=HAND_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # ORANGE
                emissive_color=(0.5, 0.25, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.2, 1.0)),
    )

    # PURPLE/MAGENTA sphere - RIGHT HAND
    purple_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PurpleSphere_RightHand",
        spawn=sim_utils.SphereCfg(
            radius=HAND_SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.0, 0.8),  # PURPLE/MAGENTA
                emissive_color=(0.4, 0.0, 0.4),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.2, 1.0)),
    )


def find_body_indices(robot) -> dict:
    """Find important body indices in robot."""
    body_names = robot.data.body_names

    # Print all body names for debugging
    print("\n" + "=" * 70)
    print("G1 BODY NAMES (Total: {})".format(len(body_names)))
    print("=" * 70)
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")
    print("=" * 70 + "\n")

    indices = {
        "head": None,
        "pelvis": None,
        "torso": None,
        "left_hand": None,
        "right_hand": None,
    }

    # Search for body indices
    for i, name in enumerate(body_names):
        name_lower = name.lower()

        # Head
        if "head" in name_lower and indices["head"] is None:
            indices["head"] = i
            print(f"[FOUND] Head: index {i} = '{name}'")

        # Pelvis
        if "pelvis" in name_lower and indices["pelvis"] is None:
            indices["pelvis"] = i
            print(f"[FOUND] Pelvis: index {i} = '{name}'")

        # Torso
        if "torso" in name_lower and indices["torso"] is None:
            indices["torso"] = i
            print(f"[FOUND] Torso: index {i} = '{name}'")

        # Left hand/palm - prefer palm_link
        if "left" in name_lower:
            if "palm" in name_lower:
                indices["left_hand"] = i
                print(f"[FOUND] Left Palm: index {i} = '{name}'")
            elif indices["left_hand"] is None and ("hand" in name_lower or "wrist" in name_lower):
                indices["left_hand"] = i
                print(f"[FOUND] Left Hand/Wrist: index {i} = '{name}'")

        # Right hand/palm - prefer palm_link
        if "right" in name_lower:
            if "palm" in name_lower:
                indices["right_hand"] = i
                print(f"[FOUND] Right Palm: index {i} = '{name}'")
            elif indices["right_hand"] is None and ("hand" in name_lower or "wrist" in name_lower):
                indices["right_hand"] = i
                print(f"[FOUND] Right Hand/Wrist: index {i} = '{name}'")

    # Fallbacks
    if indices["head"] is None:
        # Use torso as head reference
        if indices["torso"] is not None:
            indices["head"] = indices["torso"]
            print(f"[FALLBACK] Using torso (index {indices['head']}) as head reference")
        else:
            indices["head"] = 0
            print(f"[FALLBACK] Using index 0 as head reference")

    # For hands, search for elbow if palm/wrist not found
    if indices["left_hand"] is None:
        for i, name in enumerate(body_names):
            if "left" in name.lower() and "elbow" in name.lower():
                indices["left_hand"] = i
                print(f"[FALLBACK] Using '{name}' (index {i}) as left hand")
                break
        if indices["left_hand"] is None:
            indices["left_hand"] = 0
            print(f"[FALLBACK] Using index 0 as left hand")

    if indices["right_hand"] is None:
        for i, name in enumerate(body_names):
            if "right" in name.lower() and "elbow" in name.lower():
                indices["right_hand"] = i
                print(f"[FALLBACK] Using '{name}' (index {i}) as right hand")
                break
        if indices["right_hand"] is None:
            indices["right_hand"] = 0
            print(f"[FALLBACK] Using index 0 as right hand")

    print()
    return indices


def update_head_spheres(scene, head_pos: torch.Tensor, device: str) -> dict:
    """
    Update sphere positions relative to head position.

    Coordinate system (robot's perspective):
    - +X: FORWARD (önde)
    - -X: BACKWARD (arkada)
    - +Y: LEFT (sol)
    - -Y: RIGHT (sağ)
    - +Z: UP (yukarı)
    """
    num_envs = head_pos.shape[0]
    offset = 0.5  # 50cm
    default_quat = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32).expand(num_envs, -1)

    positions = {}

    # RED: Front (+X)
    red_pos = head_pos.clone()
    red_pos[:, 0] += offset
    scene["red_sphere"].write_root_pose_to_sim(torch.cat([red_pos, default_quat], dim=-1))
    positions["red_front"] = red_pos[0].cpu().numpy()

    # BLUE: Back (-X)
    blue_pos = head_pos.clone()
    blue_pos[:, 0] -= offset
    scene["blue_sphere"].write_root_pose_to_sim(torch.cat([blue_pos, default_quat], dim=-1))
    positions["blue_back"] = blue_pos[0].cpu().numpy()

    # GREEN: Left (+Y)
    green_pos = head_pos.clone()
    green_pos[:, 1] += offset
    scene["green_sphere"].write_root_pose_to_sim(torch.cat([green_pos, default_quat], dim=-1))
    positions["green_left"] = green_pos[0].cpu().numpy()

    # YELLOW: Right (-Y)
    yellow_pos = head_pos.clone()
    yellow_pos[:, 1] -= offset
    scene["yellow_sphere"].write_root_pose_to_sim(torch.cat([yellow_pos, default_quat], dim=-1))
    positions["yellow_right"] = yellow_pos[0].cpu().numpy()

    return positions


def update_hand_spheres(scene, left_hand_pos: torch.Tensor, right_hand_pos: torch.Tensor, device: str) -> dict:
    """Update hand tracking spheres to follow hand positions."""
    num_envs = left_hand_pos.shape[0]
    default_quat = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32).expand(num_envs, -1)

    # ORANGE: Left Hand
    scene["orange_sphere"].write_root_pose_to_sim(torch.cat([left_hand_pos, default_quat], dim=-1))

    # PURPLE: Right Hand
    scene["purple_sphere"].write_root_pose_to_sim(torch.cat([right_hand_pos, default_quat], dim=-1))

    return {
        "left_hand": left_hand_pos[0].cpu().numpy(),
        "right_hand": right_hand_pos[0].cpu().numpy(),
    }


def main():
    """Main function."""

    print("\n" + "=" * 70)
    print("  G1 HEAD & HAND COORDINATE SYSTEM TEST v3")
    print("=" * 70)
    print("""
    KAFA ETRAFINDAKİ KÜRELER (50cm mesafe):
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │                    🟢 YEŞİL (SOL, +Y)                       │
    │                          │                                  │
    │                          │                                  │
    │       🔵 MAVİ ───────── 👤 ─────────🔴 KIRMIZI             │
    │       (ARKA,-X)          │           (ÖN, +X)               │
    │                          │                                  │
    │                          │                                  │
    │                    🟡 SARI (SAĞ, -Y)                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    EL TAKİP KÜRELERİ:
    🟠 TURUNCU = Sol el pozisyonu (ellerin üzerinde takip eder)
    🟣 MOR     = Sağ el pozisyonu (ellerin üzerinde takip eder)

    Robot YERÇEKİMSİZ - düşmeyecek.
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
    scene_cfg = CoordinateTestSceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    # Reset and play
    sim.reset()

    # Get device
    device = sim.device

    # Find body indices
    robot = scene["robot"]
    body_indices = find_body_indices(robot)

    head_idx = body_indices["head"]
    left_hand_idx = body_indices["left_hand"]
    right_hand_idx = body_indices["right_hand"]

    # Print body indices summary
    print("-" * 70)
    print("BODY INDICES SUMMARY:")
    print("-" * 70)
    for name, idx in body_indices.items():
        if idx is not None and idx < len(robot.data.body_names):
            body_name = robot.data.body_names[idx]
            print(f"  {name:12s}: index {idx:2d} = '{body_name}'")
        else:
            print(f"  {name:12s}: NOT FOUND")
    print("-" * 70 + "\n")

    # Get initial positions
    robot.update(sim.get_physics_dt())

    head_pos = robot.data.body_pos_w[:, head_idx, :]
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx, :]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx, :]
    root_pos = robot.data.root_pos_w

    print("-" * 70)
    print("INITIAL POSITIONS (World Frame):")
    print("-" * 70)
    print(f"  Robot root:  [{root_pos[0, 0]:.3f}, {root_pos[0, 1]:.3f}, {root_pos[0, 2]:.3f}]")
    print(f"  Head/Torso:  [{head_pos[0, 0]:.3f}, {head_pos[0, 1]:.3f}, {head_pos[0, 2]:.3f}]")
    print(f"  Left hand:   [{left_hand_pos[0, 0]:.3f}, {left_hand_pos[0, 1]:.3f}, {left_hand_pos[0, 2]:.3f}]")
    print(f"  Right hand:  [{right_hand_pos[0, 0]:.3f}, {right_hand_pos[0, 1]:.3f}, {right_hand_pos[0, 2]:.3f}]")
    print("-" * 70 + "\n")

    # Calculate hand offsets from root
    left_offset_from_root = left_hand_pos[0] - root_pos[0]
    right_offset_from_root = right_hand_pos[0] - root_pos[0]

    # Calculate hand offsets from head
    left_offset_from_head = left_hand_pos[0] - head_pos[0]
    right_offset_from_head = right_hand_pos[0] - head_pos[0]

    print("-" * 70)
    print("HAND OFFSETS (Initial Pose):")
    print("-" * 70)
    print("From ROOT:")
    print(
        f"  Left hand:   [{left_offset_from_root[0]:.3f}, {left_offset_from_root[1]:.3f}, {left_offset_from_root[2]:.3f}]")
    print(
        f"  Right hand:  [{right_offset_from_root[0]:.3f}, {right_offset_from_root[1]:.3f}, {right_offset_from_root[2]:.3f}]")
    print("From HEAD/TORSO:")
    print(
        f"  Left hand:   [{left_offset_from_head[0]:.3f}, {left_offset_from_head[1]:.3f}, {left_offset_from_head[2]:.3f}]")
    print(
        f"  Right hand:  [{right_offset_from_head[0]:.3f}, {right_offset_from_head[1]:.3f}, {right_offset_from_head[2]:.3f}]")
    print("-" * 70 + "\n")

    # Update spheres
    head_sphere_positions = update_head_spheres(scene, head_pos, device)
    hand_sphere_positions = update_hand_spheres(scene, left_hand_pos, right_hand_pos, device)

    print("-" * 70)
    print("HEAD REFERENCE SPHERE POSITIONS (50cm from head):")
    print("-" * 70)
    for name, pos in head_sphere_positions.items():
        print(f"  {name:15s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")
    print("-" * 70 + "\n")

    # Print coordinate system reference
    print("=" * 70)
    print("COORDINATE SYSTEM REFERENCE:")
    print("=" * 70)
    print("  +X = FORWARD  (robotun bakış yönü, KIRMIZI küre)")
    print("  -X = BACKWARD (robotun arkası, MAVİ küre)")
    print("  +Y = LEFT     (robotun solu, YEŞİL küre)")
    print("  -Y = RIGHT    (robotun sağı, SARI küre)")
    print("  +Z = UP       (yukarı)")
    print("")
    print("  Sol el  → +Y tarafında olmalı (YEŞİL küre tarafı)")
    print("  Sağ el  → -Y tarafında olmalı (SARI küre tarafı)")
    print("=" * 70 + "\n")

    # Simulation loop
    print(f"Running simulation for {args.duration} seconds...")
    print("\nKONTROL LİSTESİ:")
    print("  ✓ KIRMIZI küre robotun ÖNÜNDE mi?")
    print("  ✓ MAVİ küre robotun ARKASINDA mı?")
    print("  ✓ YEŞİL küre robotun SOLUNDA mı?")
    print("  ✓ SARI küre robotun SAĞINDA mı?")
    print("  ✓ TURUNCU küre SOL elde mi?")
    print("  ✓ MOR küre SAĞ elde mi?")
    print("\nPress Ctrl+C to exit early.\n")

    step_count = 0
    max_steps = int(args.duration / sim_cfg.dt)

    try:
        while simulation_app.is_running() and step_count < max_steps:
            # Step simulation
            sim.step()

            # Update scene
            scene.update(sim_cfg.dt)

            # Update robot data
            robot.update(sim_cfg.dt)

            # Get current positions
            head_pos = robot.data.body_pos_w[:, head_idx, :]
            left_hand_pos = robot.data.body_pos_w[:, left_hand_idx, :]
            right_hand_pos = robot.data.body_pos_w[:, right_hand_idx, :]

            # Update spheres to follow robot
            update_head_spheres(scene, head_pos, device)
            update_hand_spheres(scene, left_hand_pos, right_hand_pos, device)

            # Print status every 3 seconds
            if step_count % int(3.0 / sim_cfg.dt) == 0:
                elapsed = step_count * sim_cfg.dt
                lh = left_hand_pos[0]
                rh = right_hand_pos[0]
                print(f"[{elapsed:5.1f}s] L.Hand: [{lh[0]:.3f}, {lh[1]:.3f}, {lh[2]:.3f}] | "
                      f"R.Hand: [{rh[0]:.3f}, {rh[1]:.3f}, {rh[2]:.3f}]")

            step_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Final summary
    print("\n" + "=" * 70)
    print("TEST TAMAMLANDI")
    print("=" * 70)
    print("""
    KONTROL LİSTESİ:

    KAFA REFERANS KÜRELERİ:
    ┌────────────┬──────────────────┬──────────────────────────────┐
    │ Küre       │ Beklenen Pozisyon│ Kontrol                      │
    ├────────────┼──────────────────┼──────────────────────────────┤
    │ 🔴 Kırmızı │ Robotun ÖNÜnde   │ +X yönünde mi?               │
    │ 🔵 Mavi    │ Robotun ARKASINda│ -X yönünde mi?               │
    │ 🟢 Yeşil   │ Robotun SOLunda  │ +Y yönünde mi?               │
    │ 🟡 Sarı    │ Robotun SAĞında  │ -Y yönünde mi?               │
    └────────────┴──────────────────┴──────────────────────────────┘

    EL TAKİP KÜRELERİ:
    ┌────────────┬──────────────────┬──────────────────────────────┐
    │ Küre       │ Beklenen Pozisyon│ Kontrol                      │
    ├────────────┼──────────────────┼──────────────────────────────┤
    │ 🟠 Turuncu │ SOL el           │ +Y tarafında mı?             │
    │ 🟣 Mor     │ SAĞ el           │ -Y tarafında mı?             │
    └────────────┴──────────────────┴──────────────────────────────┘

    Eğer hepsi doğruysa, koordinat sistemi DOĞRU çalışıyor!

    SONRAKİ ADIM: Arm reaching test (kolları hedefe uzatma)
    """)
    print("=" * 70 + "\n")

    # Cleanup
    sim.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()