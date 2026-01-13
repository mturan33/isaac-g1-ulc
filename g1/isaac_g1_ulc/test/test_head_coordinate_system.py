#!/usr/bin/env python3
"""
G1 Head Coordinate System Test
==============================

Bu script G1 robotunun kafası etrafındaki koordinat sistemini test eder.
4 renkli küre oluşturur:
- KIRMIZI: Kafanın 50cm ÖNÜnde (forward, +X)
- MAVİ: Kafanın 50cm ARKASINda (backward, -X)
- YEŞİL: Kafanın 50cm SOLunda (left, +Y)
- SARI: Kafanın 50cm SAĞında (right, -Y)

Robot tamamen SABİT - sadece koordinat sistemini doğruluyoruz.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/test/test_head_coordinate_system.py --num_envs 1
"""

import argparse
import torch
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="G1 Head Coordinate System Test")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")

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

# Sphere radius (small for debug)
SPHERE_RADIUS = 0.03  # 3cm


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

    # G1 Robot - FIXED ROOT (won't fall)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,  # ROBOT FIXED - WON'T FALL!
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),  # Start at 0.8m height
            joint_pos={
                ".*": 0.0,  # All joints at zero
            },
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

    # ========== DEBUG SPHERES ==========

    # RED sphere - FRONT (+X direction)
    red_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedSphere_Front",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Won't be affected by physics
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # RED
                emissive_color=(0.5, 0.0, 0.0),  # Slight glow
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 1.2),  # Will be updated dynamically
        ),
    )

    # BLUE sphere - BACK (-X direction)
    blue_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueSphere_Back",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0),  # BLUE
                emissive_color=(0.0, 0.0, 0.5),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 1.2),
        ),
    )

    # GREEN sphere - LEFT (+Y direction)
    green_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GreenSphere_Left",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),  # GREEN
                emissive_color=(0.0, 0.5, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.5, 1.2),
        ),
    )

    # YELLOW sphere - RIGHT (-Y direction)
    yellow_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/YellowSphere_Right",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),  # YELLOW
                emissive_color=(0.5, 0.5, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.5, 1.2),
        ),
    )


def find_head_body_index(robot) -> int:
    """Find the head/pelvis body index in robot."""
    body_names = robot.data.body_names

    # Print all body names for debugging
    print("\n" + "=" * 60)
    print("G1 BODY NAMES:")
    print("=" * 60)
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")
    print("=" * 60 + "\n")

    # Try to find head link
    head_idx = None
    for i, name in enumerate(body_names):
        name_lower = name.lower()
        if "head" in name_lower:
            head_idx = i
            print(f"[INFO] Found head at index {i}: {name}")
            break

    # Fallback: use pelvis or torso
    if head_idx is None:
        for i, name in enumerate(body_names):
            name_lower = name.lower()
            if "pelvis" in name_lower or "torso" in name_lower or "base" in name_lower:
                head_idx = i
                print(f"[INFO] Using {name} as reference (index {i})")
                break

    # Last fallback: use index 0 (usually base)
    if head_idx is None:
        head_idx = 0
        print(f"[WARN] No head found, using body 0: {body_names[0]}")

    return head_idx


def update_sphere_positions(scene, head_pos: torch.Tensor, device: str):
    """
    Update sphere positions relative to head position.

    Coordinate system (robot's perspective):
    - +X: FORWARD (önde)
    - -X: BACKWARD (arkada)
    - +Y: LEFT (sol)
    - -Y: RIGHT (sağ)
    - +Z: UP (yukarı)

    Args:
        scene: InteractiveScene
        head_pos: [num_envs, 3] head position in world frame
        device: torch device
    """
    num_envs = head_pos.shape[0]
    offset = 0.5  # 50cm

    # RED: Front (+X)
    red_pos = head_pos.clone()
    red_pos[:, 0] += offset  # +X = forward
    scene["red_sphere"].write_root_pose_to_sim(
        torch.cat([red_pos, torch.tensor([[0, 0, 0, 1]], device=device).expand(num_envs, -1)], dim=-1)
    )

    # BLUE: Back (-X)
    blue_pos = head_pos.clone()
    blue_pos[:, 0] -= offset  # -X = backward
    scene["blue_sphere"].write_root_pose_to_sim(
        torch.cat([blue_pos, torch.tensor([[0, 0, 0, 1]], device=device).expand(num_envs, -1)], dim=-1)
    )

    # GREEN: Left (+Y)
    green_pos = head_pos.clone()
    green_pos[:, 1] += offset  # +Y = left
    scene["green_sphere"].write_root_pose_to_sim(
        torch.cat([green_pos, torch.tensor([[0, 0, 0, 1]], device=device).expand(num_envs, -1)], dim=-1)
    )

    # YELLOW: Right (-Y)
    yellow_pos = head_pos.clone()
    yellow_pos[:, 1] -= offset  # -Y = right
    scene["yellow_sphere"].write_root_pose_to_sim(
        torch.cat([yellow_pos, torch.tensor([[0, 0, 0, 1]], device=device).expand(num_envs, -1)], dim=-1)
    )

    return {
        "red_front": red_pos[0].cpu().numpy(),
        "blue_back": blue_pos[0].cpu().numpy(),
        "green_left": green_pos[0].cpu().numpy(),
        "yellow_right": yellow_pos[0].cpu().numpy(),
    }


def main():
    """Main function."""

    print("\n" + "=" * 70)
    print("  G1 HEAD COORDINATE SYSTEM TEST")
    print("=" * 70)
    print("""
    Bu test 4 renkli küre ile koordinat sistemini doğrular:

    🔴 KIRMIZI = ÖN    (+X, forward,  kafanın 50cm önünde)
    🔵 MAVİ    = ARKA  (-X, backward, kafanın 50cm arkasında)
    🟢 YEŞİL   = SOL   (+Y, left,     kafanın 50cm solunda)
    🟡 SARI    = SAĞ   (-Y, right,    kafanın 50cm sağında)

    Robot SABİT (fix_root_link=True) - düşmeyecek.
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

    # Find head body index
    robot = scene["robot"]
    head_idx = find_head_body_index(robot)

    # Print initial info
    print("\n" + "-" * 60)
    print("INITIAL STATE:")
    print("-" * 60)

    # Get initial positions
    robot.update(sim.get_physics_dt())
    head_pos = robot.data.body_pos_w[:, head_idx, :]
    root_pos = robot.data.root_pos_w

    print(f"  Robot root position: {root_pos[0].cpu().numpy()}")
    print(f"  Head position: {head_pos[0].cpu().numpy()}")
    print(f"  Head body index: {head_idx}")
    print(f"  Head body name: {robot.data.body_names[head_idx]}")
    print("-" * 60 + "\n")

    # Update sphere positions
    sphere_positions = update_sphere_positions(scene, head_pos, device)

    print("SPHERE POSITIONS (relative to head):")
    print("-" * 60)
    for name, pos in sphere_positions.items():
        print(f"  {name:15s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}]")
    print("-" * 60 + "\n")

    # Print coordinate system reference
    print("COORDINATE SYSTEM REFERENCE:")
    print("-" * 60)
    print("  Robotun bakış yönü: +X (forward)")
    print("  Robotun solu:       +Y (left)")
    print("  Robotun sağı:       -Y (right)")
    print("  Robotun arkası:     -X (backward)")
    print("  Yukarı:             +Z (up)")
    print("-" * 60 + "\n")

    # Simulation loop
    print(f"Running simulation for {args.duration} seconds...")
    print("Küreleri kontrol et:")
    print("  - KIRMIZI robotun ÖNÜNDE mi?")
    print("  - MAVİ robotun ARKASINDA mı?")
    print("  - YEŞİL robotun SOLUNDA mı?")
    print("  - SARI robotun SAĞINDA mı?")
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

            # Get current head position
            head_pos = robot.data.body_pos_w[:, head_idx, :]

            # Update spheres to follow head
            update_sphere_positions(scene, head_pos, device)

            # Print status every 2 seconds
            if step_count % int(2.0 / sim_cfg.dt) == 0:
                elapsed = step_count * sim_cfg.dt
                print(f"[{elapsed:5.1f}s] Head pos: [{head_pos[0, 0]:.3f}, {head_pos[0, 1]:.3f}, {head_pos[0, 2]:.3f}]")

            step_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Final summary
    print("\n" + "=" * 70)
    print("TEST TAMAMLANDI")
    print("=" * 70)
    print("""
    KONTROL LİSTESİ:
    ✅ Kırmızı küre robotun ÖNÜNDE (forward direction)?
    ✅ Mavi küre robotun ARKASINDA?
    ✅ Yeşil küre robotun SOLUNDA?
    ✅ Sarı küre robotun SAĞINDA?

    Eğer hepsi doğruysa, koordinat sistemi DOĞRU çalışıyor!

    Sonraki adım: Hand position test (ellerin koordinatları)
    """)
    print("=" * 70 + "\n")

    # Cleanup
    sim.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()