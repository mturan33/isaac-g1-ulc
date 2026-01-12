#!/usr/bin/env python3
"""
G1 Position Controller - Move Hand to Exact XYZ
================================================

Simülasyon-tabanlı IK: Jacobian tahmini yerine gerçek simülasyonu
kullanarak eli istenen konuma götürür.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_position_controller.py --num_envs 1

ÖRNEK:
- "Sağ eli (0.4, -0.2, 1.2) konumuna götür"
- "Sol eli robotun 1m önüne götür"
"""

import argparse
import math
import time

parser = argparse.ArgumentParser(description="G1 Position Controller")
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
print("  G1 POSITION CONTROLLER")
print("  Move hand to exact XYZ coordinates")
print("=" * 70 + "\n")


class G1PositionController:
    """
    Simülasyon-tabanlı pozisyon kontrolcüsü.

    Jacobian hesaplamak yerine simülasyonda deneyerek
    eli istenen konuma götürür.
    """

    def __init__(self, robot, sim, device="cuda:0"):
        self.robot = robot
        self.sim = sim
        self.device = device
        self.dt = sim.cfg.dt

        # Joint isimleri ve indeksleri
        joint_names = list(robot.data.joint_names)
        body_names = list(robot.data.body_names)

        # Kol joint sırası
        arm_joint_order = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                           "elbow_pitch", "elbow_roll"]

        # Sağ kol indeksleri
        self.right_indices = []
        for order_name in arm_joint_order:
            for i, name in enumerate(joint_names):
                if "right" in name.lower() and order_name in name.lower():
                    self.right_indices.append(i)
                    break

        # Sol kol indeksleri
        self.left_indices = []
        for order_name in arm_joint_order:
            for i, name in enumerate(joint_names):
                if "left" in name.lower() and order_name in name.lower():
                    self.left_indices.append(i)
                    break

        self.right_indices = torch.tensor(self.right_indices, device=device)
        self.left_indices = torch.tensor(self.left_indices, device=device)

        # EE body indeksleri
        self.right_ee_idx = 29
        self.left_ee_idx = 28
        for i, name in enumerate(body_names):
            if "right_palm" in name.lower() or "right_hand" in name.lower():
                self.right_ee_idx = i
            if "left_palm" in name.lower() or "left_hand" in name.lower():
                self.left_ee_idx = i

        # Joint limitleri
        self.limits_low = torch.tensor([-2.97, -2.25, -2.62, -0.23, -2.09], device=device)
        self.limits_high = torch.tensor([2.79, 1.59, 2.62, 3.42, 2.09], device=device)

        # Kontrol parametreleri
        self.max_iterations = 200
        self.tolerance = 0.02  # 2cm hata toleransı
        self.step_size = 0.1  # Joint değişim step size (rad)

        print(f"[PositionController] Right arm indices: {self.right_indices.tolist()}")
        print(f"[PositionController] Left arm indices: {self.left_indices.tolist()}")
        print(f"[PositionController] Right EE body: {self.right_ee_idx}")
        print(f"[PositionController] Left EE body: {self.left_ee_idx}")

    def get_ee_position(self, arm="right"):
        """Mevcut EE pozisyonunu al"""
        ee_idx = self.right_ee_idx if arm == "right" else self.left_ee_idx
        return self.robot.data.body_pos_w[0, ee_idx].clone()

    def get_arm_joints(self, arm="right"):
        """Mevcut kol joint açılarını al"""
        indices = self.right_indices if arm == "right" else self.left_indices
        return self.robot.data.joint_pos[0, indices].clone()

    def set_arm_joints(self, joints, arm="right"):
        """Kol joint açılarını ayarla"""
        indices = self.right_indices if arm == "right" else self.left_indices
        joint_targets = self.robot.data.joint_pos.clone()
        joint_targets[0, indices] = joints
        self.robot.set_joint_position_target(joint_targets)
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)

    def move_to_position(self, target_pos, arm="right", verbose=True):
        """
        Eli hedef pozisyona götür.

        Args:
            target_pos: Hedef pozisyon [X, Y, Z] (world frame)
            arm: "right" veya "left"
            verbose: Detaylı çıktı

        Returns:
            success: Hedefe ulaşıldı mı
            final_error: Son hata (metre)
            iterations: Kaç iterasyon sürdü
        """
        target = torch.tensor(target_pos, device=self.device, dtype=torch.float32)
        indices = self.right_indices if arm == "right" else self.left_indices

        if verbose:
            print(f"\n[MOVE] Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

        # Mevcut joint açıları
        current_joints = self.get_arm_joints(arm)
        best_joints = current_joints.clone()
        best_error = float('inf')

        for iteration in range(self.max_iterations):
            # Mevcut EE pozisyonu
            current_ee = self.get_ee_position(arm)
            error_vec = target - current_ee
            error = torch.norm(error_vec).item()

            # En iyi sonucu kaydet
            if error < best_error:
                best_error = error
                best_joints = current_joints.clone()

            # Hedefe ulaştık mı?
            if error < self.tolerance:
                if verbose:
                    print(f"[MOVE] ✅ Target reached in {iteration} iterations! Error: {error * 100:.1f}cm")
                return True, error, iteration

            # Her joint için gradient hesapla (finite difference)
            gradients = torch.zeros(5, device=self.device)

            for j in range(5):
                # Pozitif pertürbasyon
                test_joints = current_joints.clone()
                test_joints[j] += self.step_size
                test_joints = torch.clamp(test_joints, self.limits_low, self.limits_high)
                self.set_arm_joints(test_joints, arm)
                ee_plus = self.get_ee_position(arm)
                error_plus = torch.norm(target - ee_plus).item()

                # Negatif pertürbasyon
                test_joints = current_joints.clone()
                test_joints[j] -= self.step_size
                test_joints = torch.clamp(test_joints, self.limits_low, self.limits_high)
                self.set_arm_joints(test_joints, arm)
                ee_minus = self.get_ee_position(arm)
                error_minus = torch.norm(target - ee_minus).item()

                # Gradient: hangi yönde error azalıyor?
                gradients[j] = (error_plus - error_minus) / (2 * self.step_size)

            # Joint'leri gradient yönünde güncelle
            # Negatif gradient = error azalma yönü
            grad_norm = torch.norm(gradients)
            if grad_norm > 0.001:
                # Adaptive step size
                step = min(0.2, error * 2) / grad_norm
                current_joints = current_joints - step * gradients
                current_joints = torch.clamp(current_joints, self.limits_low, self.limits_high)

            # Yeni joint'leri uygula
            self.set_arm_joints(current_joints, arm)

            # Progress
            if verbose and iteration % 20 == 0:
                print(f"[MOVE] Iter {iteration}: Error = {error * 100:.1f}cm")

        # Max iterasyona ulaştık, en iyi sonuca dön
        self.set_arm_joints(best_joints, arm)

        # Birkaç adım stabilize et
        for _ in range(10):
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.dt)

        final_ee = self.get_ee_position(arm)
        final_error = torch.norm(target - final_ee).item()

        if verbose:
            if final_error < self.tolerance * 2:
                print(f"[MOVE] ⚠️ Close to target. Error: {final_error * 100:.1f}cm")
            else:
                print(f"[MOVE] ❌ Could not reach target. Error: {final_error * 100:.1f}cm")

        return final_error < self.tolerance * 2, final_error, self.max_iterations

    def move_relative(self, delta_pos, arm="right", verbose=True):
        """
        Eli mevcut konumdan göreceli olarak hareket ettir.

        Args:
            delta_pos: [dX, dY, dZ] metre cinsinden
            arm: "right" veya "left"
        """
        current_ee = self.get_ee_position(arm)
        target = current_ee + torch.tensor(delta_pos, device=self.device)
        return self.move_to_position(target.tolist(), arm, verbose)

    def move_in_front_of_head(self, distance=0.5, arm="right", verbose=True):
        """
        Eli robotun kafasının önüne götür.

        Args:
            distance: Kafadan uzaklık (metre)
            arm: "right" veya "left"
        """
        # Robot pozisyonu (kafa yaklaşık 1.7m yükseklikte)
        root_pos = self.robot.data.root_pos_w[0]

        # Hedef: Kafanın önünde
        target = [
            root_pos[0].item() + distance,  # X: ileri
            root_pos[1].item() + (0.15 if arm == "left" else -0.15),  # Y: kol tarafı
            root_pos[2].item() + 0.5  # Z: göğüs/kafa seviyesi
        ]

        if verbose:
            print(f"\n[HEAD FRONT] Moving {arm} hand {distance}m in front of head")

        return self.move_to_position(target, arm, verbose)

    def move_to_floor(self, forward=0.3, arm="right", verbose=True):
        """
        Eli yere doğru götür (eğilme simulasyonu için).

        Args:
            forward: İleri mesafe
            arm: "right" veya "left"
        """
        root_pos = self.robot.data.root_pos_w[0]

        target = [
            root_pos[0].item() + forward,
            root_pos[1].item() + (0.1 if arm == "left" else -0.1),
            root_pos[2].item() - 0.3  # Bel seviyesinin altı
        ]

        if verbose:
            print(f"\n[FLOOR] Moving {arm} hand toward floor")

        return self.move_to_position(target, arm, verbose)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 1.0])

    # Robot yükle
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

    # Sahne
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    robot.update(sim.cfg.dt)

    # Controller oluştur
    controller = G1PositionController(robot, sim, device="cuda:0")

    # Başlangıç pozisyonları
    init_right_ee = controller.get_ee_position("right")
    init_left_ee = controller.get_ee_position("left")
    root_pos = robot.data.root_pos_w[0]

    print(f"\n[INFO] Robot root: ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
    print(f"[INFO] Initial Right EE: ({init_right_ee[0]:.3f}, {init_right_ee[1]:.3f}, {init_right_ee[2]:.3f})")
    print(f"[INFO] Initial Left EE: ({init_left_ee[0]:.3f}, {init_left_ee[1]:.3f}, {init_left_ee[2]:.3f})")

    # ========== TEST 1: Absolute Position ==========
    print("\n" + "=" * 60)
    print("  TEST 1: ABSOLUTE POSITION")
    print("  Sağ eli (0.4, -0.2, 1.2) konumuna götür")
    print("=" * 60)

    target1 = [0.4, -0.2, 1.2]
    success, error, iters = controller.move_to_position(target1, "right")

    final_ee = controller.get_ee_position("right")
    print(f"\n[RESULT] Target: ({target1[0]:.3f}, {target1[1]:.3f}, {target1[2]:.3f})")
    print(f"[RESULT] Actual: ({final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f})")
    print(f"[RESULT] Error: {error * 100:.1f}cm, Iterations: {iters}")

    # Bekle
    for _ in range(100):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # ========== TEST 2: Head Front ==========
    print("\n" + "=" * 60)
    print("  TEST 2: HEAD FRONT")
    print("  Sağ eli kafanın 50cm önüne götür")
    print("=" * 60)

    success, error, iters = controller.move_in_front_of_head(0.5, "right")

    final_ee = controller.get_ee_position("right")
    print(f"\n[RESULT] Final position: ({final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f})")
    print(f"[RESULT] Error: {error * 100:.1f}cm")

    # Bekle
    for _ in range(100):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # ========== TEST 3: Both Arms ==========
    print("\n" + "=" * 60)
    print("  TEST 3: BOTH ARMS")
    print("  Her iki eli de kafanın 1m önüne götür")
    print("=" * 60)

    # Önce sağ
    print("\n--- Right Arm ---")
    success_r, error_r, _ = controller.move_in_front_of_head(1.0, "right")

    # Sonra sol
    print("\n--- Left Arm ---")
    success_l, error_l, _ = controller.move_in_front_of_head(1.0, "left")

    right_ee = controller.get_ee_position("right")
    left_ee = controller.get_ee_position("left")

    print(
        f"\n[RESULT] Right EE: ({right_ee[0]:.3f}, {right_ee[1]:.3f}, {right_ee[2]:.3f}), Error: {error_r * 100:.1f}cm")
    print(f"[RESULT] Left EE:  ({left_ee[0]:.3f}, {left_ee[1]:.3f}, {left_ee[2]:.3f}), Error: {error_l * 100:.1f}cm")

    # Bekle
    for _ in range(100):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # ========== TEST 4: Floor Reach ==========
    print("\n" + "=" * 60)
    print("  TEST 4: FLOOR REACH")
    print("  Sağ eli yere doğru götür")
    print("=" * 60)

    success, error, iters = controller.move_to_floor(0.3, "right")

    final_ee = controller.get_ee_position("right")
    print(f"\n[RESULT] Final position: ({final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f})")
    print(f"[RESULT] Error: {error * 100:.1f}cm")

    # ========== TEST 5: Relative Movement ==========
    print("\n" + "=" * 60)
    print("  TEST 5: RELATIVE MOVEMENT")
    print("  Sağ eli 20cm sola, 10cm yukarı")
    print("=" * 60)

    before_ee = controller.get_ee_position("right")
    success, error, iters = controller.move_relative([0.0, 0.2, 0.1], "right")
    after_ee = controller.get_ee_position("right")

    actual_delta = after_ee - before_ee
    print(f"\n[RESULT] Requested delta: (0.0, 0.2, 0.1)")
    print(f"[RESULT] Actual delta:    ({actual_delta[0]:.3f}, {actual_delta[1]:.3f}, {actual_delta[2]:.3f})")

    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
  ✅ Position Controller Features:

     move_to_position(target, arm)    - Go to exact XYZ
     move_relative(delta, arm)        - Move by delta XYZ
     move_in_front_of_head(dist, arm) - Go in front of head
     move_to_floor(forward, arm)      - Reach toward floor

  🎯 How it works:
     1. Iterative optimization (no Jacobian)
     2. Finite difference gradient
     3. Adaptive step size
     4. Joint limit enforcement

  📊 Typical accuracy: 2-5cm
""")

    # Final hold
    print("[INFO] Holding final pose (5 seconds)...")
    for _ in range(500):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()