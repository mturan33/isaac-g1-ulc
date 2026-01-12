#!/usr/bin/env python3
"""
G1 Pose-Based Demo: Reach and Wave
==================================

Kolları ileri uzat, sonra el salla!

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_wave_demo.py --num_envs 1
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 Wave Demo")
parser.add_argument("--num_envs", type=int, default=1)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

print("\n" + "=" * 70)
print("  G1 POSE-BASED DEMO: REACH AND WAVE")
print("  Kolları ileri uzat, sonra el salla!")
print("=" * 70 + "\n")


class G1PoseArmController:
    """Pose-based arm controller"""

    def __init__(self, device="cuda:0"):
        self.device = device

        # === TEMEL POZLAR ===

        # REST: Kollar aşağıda
        self.rest_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)

        # FORWARD: Kollar ileri uzanmış
        self.forward_pose = torch.tensor([-1.57, 0.0, 0.0, 1.57, 0.0], device=device)

        # WAVE UP: El sallama yukarı pozisyonu
        self.wave_up_pose = torch.tensor([-1.2, -0.3, 0.0, 2.0, 0.0], device=device)

        # WAVE DOWN: El sallama aşağı pozisyonu
        self.wave_down_pose = torch.tensor([-1.2, 0.3, 0.0, 1.2, 0.0], device=device)

        # GREETING: Selamlama (tek el yukarı)
        self.greeting_pose = torch.tensor([-2.0, -0.5, 0.0, 1.5, 0.0], device=device)

        # Joint limits
        self.limits_low = torch.tensor([-2.97, -2.25, -2.62, -0.23, -2.09], device=device)
        self.limits_high = torch.tensor([2.79, 1.59, 2.62, 3.42, 2.09], device=device)

    def get_reach_joints(self, reach_factor, lateral_offset=0.0, arm="right"):
        """Forward reach with lateral adjustment"""
        reach_factor = torch.clamp(torch.tensor(reach_factor, device=self.device), 0, 1)
        lateral_offset = torch.clamp(torch.tensor(lateral_offset, device=self.device), -1, 1)

        joints = self.rest_pose + reach_factor * (self.forward_pose - self.rest_pose)
        joints[1] = joints[1] + lateral_offset * (-0.5)
        joints = torch.clamp(joints, self.limits_low, self.limits_high)

        if arm == "left":
            joints[1] = -joints[1]
            joints[2] = -joints[2]
            joints[4] = -joints[4]

        return joints

    def get_wave_joints(self, wave_phase, arm="right"):
        """
        El sallama hareketi.
        wave_phase: 0 to 2π (sinüsoidal hareket)
        """
        # Sinüsoidal interpolation between wave_up and wave_down
        t = (math.sin(wave_phase) + 1) / 2  # 0 to 1

        joints = self.wave_down_pose + t * (self.wave_up_pose - self.wave_down_pose)
        joints = torch.clamp(joints, self.limits_low, self.limits_high)

        if arm == "left":
            joints[1] = -joints[1]
            joints[2] = -joints[2]
            joints[4] = -joints[4]

        return joints

    def get_greeting_joints(self, arm="right"):
        """Selamlama pozu"""
        joints = self.greeting_pose.clone()
        joints = torch.clamp(joints, self.limits_low, self.limits_high)

        if arm == "left":
            joints[1] = -joints[1]
            joints[2] = -joints[2]
            joints[4] = -joints[4]

        return joints

    def blend_poses(self, pose_a, pose_b, alpha):
        """İki poz arasında yumuşak geçiş"""
        alpha = torch.clamp(torch.tensor(alpha, device=self.device), 0, 1)
        return pose_a * (1 - alpha) + pose_b * alpha


def smooth_alpha(t):
    """Cosine interpolation for smooth motion"""
    return 0.5 * (1 - math.cos(math.pi * t))


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 1.3], [0.0, 0.0, 1.1])

    # Load robot
    try:
        from isaaclab_assets import G1_MINIMAL_CFG
        robot_cfg = G1_MINIMAL_CFG.copy()
    except:
        from isaaclab_assets import G1_CFG
        robot_cfg = G1_CFG.copy()

    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)

    try:
        robot_cfg.spawn.articulation_props.fix_root_link = True
    except:
        pass

    # Scene
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    robot.update(sim.cfg.dt)

    # Find indices
    joint_names = list(robot.data.joint_names)

    right_joint_order = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]

    right_indices = []
    for order_name in right_joint_order:
        for i, name in enumerate(joint_names):
            if "right" in name.lower() and order_name in name.lower():
                right_indices.append(i)
                break

    left_indices = []
    for order_name in right_joint_order:
        for i, name in enumerate(joint_names):
            if "left" in name.lower() and order_name in name.lower():
                left_indices.append(i)
                break

    right_indices = torch.tensor(right_indices, device="cuda:0")
    left_indices = torch.tensor(left_indices, device="cuda:0")

    print(f"[INFO] Right arm indices: {right_indices.tolist()}")
    print(f"[INFO] Left arm indices: {left_indices.tolist()}")

    # Create controller
    controller = G1PoseArmController(device="cuda:0")

    # Animation state
    current_right = controller.rest_pose.clone()
    current_left = controller.rest_pose.clone()

    # === ANIMATION SEQUENCE ===

    print("\n" + "=" * 60)
    print("  PHASE 1: FORWARD REACH (3 seconds)")
    print("=" * 60)
    print("[INFO] Kollar ileri uzanıyor...")

    # Phase 1: Reach forward (3 seconds = 300 steps)
    target_right = controller.get_reach_joints(1.0, 0.0, "right")
    target_left = controller.get_reach_joints(1.0, 0.0, "left")

    for step in range(300):
        alpha = smooth_alpha(min(1.0, step / 200.0))

        current_right = controller.blend_poses(controller.rest_pose, target_right, alpha)
        current_left = controller.blend_poses(
            controller.rest_pose.clone() * torch.tensor([1, -1, -1, 1, -1], device="cuda:0"),
            target_left,
            alpha
        )

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] = current_right
        joint_targets[0, left_indices] = current_left

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        if step % 100 == 0:
            print(f"  [{step / 100:.0f}s] Reach: {alpha * 100:.0f}%")

    print("\n" + "=" * 60)
    print("  PHASE 2: HOLD POSITION (2 seconds)")
    print("=" * 60)
    print("[INFO] Pozisyon tutuluyor...")

    # Phase 2: Hold (2 seconds = 200 steps)
    for step in range(200):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        if step % 100 == 0:
            print(f"  [{step / 100:.0f}s] Holding...")

    print("\n" + "=" * 60)
    print("  PHASE 3: WAVE! (5 seconds)")
    print("=" * 60)
    print("[INFO] 🖐️ El sallıyor!")

    # Phase 3: Wave animation (5 seconds = 500 steps)
    # First transition to wave pose
    start_right = current_right.clone()
    wave_base_right = controller.get_wave_joints(0, "right")

    # Transition to wave pose (1 second)
    for step in range(100):
        alpha = smooth_alpha(step / 100.0)
        current_right = controller.blend_poses(start_right, wave_base_right, alpha)

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] = current_right
        # Left arm stays in forward position

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # Wave continuously (4 seconds)
    wave_count = 0
    wave_speed = 4.0  # Waves per second

    for step in range(400):
        # Calculate wave phase
        time = step / 100.0  # seconds
        wave_phase = time * wave_speed * 2 * math.pi

        # Get wave pose
        current_right = controller.get_wave_joints(wave_phase, "right")

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] = current_right

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        # Count waves
        if step % 25 == 0:  # ~4 waves per second
            wave_count += 1
            if step % 100 == 0:
                print(f"  [{step / 100:.0f}s] 🖐️ Wave #{wave_count // 4 + 1}")

    print("\n" + "=" * 60)
    print("  PHASE 4: RETURN TO REST (2 seconds)")
    print("=" * 60)
    print("[INFO] Kollar dinlenme pozisyonuna dönüyor...")

    # Phase 4: Return to rest
    start_right = current_right.clone()
    start_left = current_left.clone()

    rest_left = controller.rest_pose.clone()
    rest_left[1] = -rest_left[1]
    rest_left[2] = -rest_left[2]
    rest_left[4] = -rest_left[4]

    for step in range(200):
        alpha = smooth_alpha(min(1.0, step / 150.0))

        current_right = controller.blend_poses(start_right, controller.rest_pose, alpha)
        current_left = controller.blend_poses(start_left, rest_left, alpha)

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] = current_right
        joint_targets[0, left_indices] = current_left

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        if step % 100 == 0:
            print(f"  [{step / 100:.0f}s] Returning...")

    print("\n" + "=" * 60)
    print("  BONUS: BOTH ARMS WAVE! (3 seconds)")
    print("=" * 60)
    print("[INFO] 🙌 İki elle selamlama!")

    # Bonus: Both arms wave
    # Transition to greeting pose
    for step in range(100):
        alpha = smooth_alpha(step / 100.0)

        target_right = controller.get_greeting_joints("right")
        target_left = controller.get_greeting_joints("left")

        current_right = controller.blend_poses(controller.rest_pose, target_right, alpha)
        current_left = controller.blend_poses(rest_left, target_left, alpha)

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] = current_right
        joint_targets[0, left_indices] = current_left

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # Both arms wave
    for step in range(200):
        time = step / 100.0
        wave_phase = time * wave_speed * 2 * math.pi

        current_right = controller.get_wave_joints(wave_phase, "right")
        current_left = controller.get_wave_joints(wave_phase + math.pi, "left")  # Opposite phase

        joint_targets = robot.data.joint_pos.clone()
        joint_targets[0, right_indices] = current_right
        joint_targets[0, left_indices] = current_left

        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        if step % 100 == 0:
            print(f"  [{step / 100:.0f}s] 🙌 Both waving!")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)
    print("""
  ✅ Pose-Based Controller Demo:

     1. Forward Reach  - Kollar ileri uzandı
     2. Hold           - Pozisyon tutuldu
     3. Wave           - Sağ el salladı
     4. Return         - Kollar geri döndü
     5. Both Wave      - İki el birlikte salladı

  🎯 Bu demo Pose-Based yaklaşımın avantajlarını gösteriyor:
     - IK hesaplama YOK
     - Kararsızlık YOK  
     - Smooth, doğal hareket
     - Her şey önceden tanımlı pozlarla
""")

    # Final hold
    print("[INFO] Final pose (3 seconds)...")
    for _ in range(300):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()