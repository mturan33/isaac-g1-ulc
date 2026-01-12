#!/usr/bin/env python3
"""
G1 Workspace Mapper v2 - With Palm Orientation
===============================================

Özellikler:
1. Palm pozisyonu (XYZ) kontrolü
2. Palm orientasyonu - HER ZAMAN AŞAĞI BAKSIN (obje almak için)
3. Workspace mapping ve lookup IK
4. Proje klasörüne kaydetme

Avuç içi zemine bakması için elbow_roll kullanılır.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_workspace_mapper_v2.py --num_envs 1 --resolution 8
"""

import argparse
import math
import pickle
import os

parser = argparse.ArgumentParser(description="G1 Workspace Mapper v2")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--resolution", type=int, default=8, help="Samples per joint (8 recommended)")
parser.add_argument("--save_dir", type=str, default="", help="Save directory")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

print("\n" + "=" * 70)
print("  G1 WORKSPACE MAPPER v2")
print("  Palm position + orientation (always facing down)")
print("=" * 70 + "\n")


class G1WorkspaceMapperV2:
    """
    Joint space → Cartesian space + Orientation mapping.
    Palm her zaman zemine bakar (grasping için ideal).
    """

    def __init__(self, robot, sim, device="cuda:0"):
        self.robot = robot
        self.sim = sim
        self.device = device
        self.dt = sim.cfg.dt

        # Joint isimleri
        joint_names = list(robot.data.joint_names)
        body_names = list(robot.data.body_names)

        # Tüm kol joint'leri
        self.arm_joint_names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                "elbow_pitch", "elbow_roll"]

        # Sağ kol indeksleri
        self.right_indices = []
        for order_name in self.arm_joint_names:
            for i, name in enumerate(joint_names):
                if "right" in name.lower() and order_name in name.lower():
                    self.right_indices.append(i)
                    break

        # Sol kol indeksleri
        self.left_indices = []
        for order_name in self.arm_joint_names:
            for i, name in enumerate(joint_names):
                if "left" in name.lower() and order_name in name.lower():
                    self.left_indices.append(i)
                    break

        self.right_indices = torch.tensor(self.right_indices, device=device)
        self.left_indices = torch.tensor(self.left_indices, device=device)

        # EE body indices
        self.right_ee_idx = None
        self.left_ee_idx = None
        for i, name in enumerate(body_names):
            if "right_palm" in name.lower() or "right_hand" in name.lower():
                self.right_ee_idx = i
            if "left_palm" in name.lower() or "left_hand" in name.lower():
                self.left_ee_idx = i

        # Fallback
        if self.right_ee_idx is None:
            self.right_ee_idx = 29
        if self.left_ee_idx is None:
            self.left_ee_idx = 28

        # Joint limitleri - 4 ana joint (elbow_roll ayrı hesaplanacak)
        # Daha dar aralık = daha iyi kapsama
        self.joint_limits = {
            'shoulder_pitch': (-2.2, 0.3),  # Yukarı-aşağı
            'shoulder_roll': (-1.2, 0.8),  # İçe-dışa
            'shoulder_yaw': (-1.0, 1.0),  # Dönüş
            'elbow_pitch': (0.2, 2.5),  # Dirsek bükümü
        }

        # Mapping data
        self.joint_samples = []
        self.ee_positions = []
        self.ee_orientations = []  # Palm orientation (quaternion)

        print(f"[Mapper] Right arm indices: {self.right_indices.tolist()}")
        print(f"[Mapper] Left arm indices: {self.left_indices.tolist()}")
        print(f"[Mapper] Right palm body: {self.right_ee_idx}")
        print(f"[Mapper] Left palm body: {self.left_ee_idx}")

    def _calculate_elbow_roll_for_palm_down(self, shoulder_pitch, shoulder_roll, elbow_pitch):
        """
        elbow_roll'u hesapla ki palm zemine baksın.

        Palm'ın zemine bakması için elbow_roll, diğer joint'lere bağlı.
        Basit yaklaşım: shoulder_roll'un tersini kullan + kompanzasyon.
        """
        # Temel kompanzasyon: shoulder_roll'u dengele
        elbow_roll = -shoulder_roll * 0.8

        # Shoulder pitch'e göre ayarla
        # Kol yukarı kalkınca (negatif pitch), palm dönmeye başlar
        if shoulder_pitch < -0.5:
            elbow_roll += (shoulder_pitch + 0.5) * 0.3

        # Limitle
        elbow_roll = np.clip(elbow_roll, -2.0, 2.0)

        return elbow_roll

    def _set_joints_and_measure(self, joint_values, arm="right"):
        """
        Set joint values and measure EE position + orientation.

        joint_values: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch]
        elbow_roll otomatik hesaplanır (palm down için)
        """
        sp, sr, sy, ep = joint_values

        # elbow_roll'u hesapla (palm zemine baksın)
        er = self._calculate_elbow_roll_for_palm_down(sp, sr, ep)

        full_joints = torch.tensor([sp, sr, sy, ep, er], device=self.device, dtype=torch.float32)

        # Sol kol için mirror
        if arm == "left":
            full_joints[1] = -full_joints[1]  # shoulder_roll
            full_joints[2] = -full_joints[2]  # shoulder_yaw
            full_joints[4] = -full_joints[4]  # elbow_roll

        # Apply
        indices = self.right_indices if arm == "right" else self.left_indices
        ee_idx = self.right_ee_idx if arm == "right" else self.left_ee_idx

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_pos[0, indices] = full_joints

        # Write and step
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        self.robot.write_data_to_sim()

        for _ in range(5):
            self.sim.step()
            self.robot.update(self.dt)

        # Measure
        ee_pos = self.robot.data.body_pos_w[0, ee_idx].clone()
        ee_quat = self.robot.data.body_quat_w[0, ee_idx].clone()  # [w, x, y, z]

        return ee_pos, ee_quat, full_joints

    def _check_palm_facing_down(self, quat):
        """
        Palm'ın zemine bakıp bakmadığını kontrol et.
        Z-ekseni aşağı bakmalı (negative Z direction).

        Returns: angle from down direction (0 = perfect, 90 = horizontal, 180 = up)
        """
        # Quaternion to rotation matrix (z-axis)
        w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()

        # Z-axis of rotated frame (local Z in world coordinates)
        zx = 2 * (x * z + w * y)
        zy = 2 * (y * z - w * x)
        zz = 1 - 2 * (x * x + y * y)

        # Angle from down direction [0, 0, -1]
        # dot product with [0, 0, -1] = -zz
        cos_angle = -zz
        angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))

        return angle_deg

    def map_workspace(self, resolution=8, arm="right"):
        """
        Map workspace with automatic palm-down orientation.
        """
        print(f"\n[Mapper] Mapping {arm} arm workspace...")
        print(f"[Mapper] Resolution: {resolution} per joint")
        print(f"[Mapper] Total samples: {resolution ** 4}")

        self.joint_samples = []
        self.ee_positions = []
        self.ee_orientations = []
        self.full_joint_samples = []  # 5 joint dahil

        # Generate samples
        sp_range = np.linspace(*self.joint_limits['shoulder_pitch'], resolution)
        sr_range = np.linspace(*self.joint_limits['shoulder_roll'], resolution)
        sy_range = np.linspace(*self.joint_limits['shoulder_yaw'], resolution)
        ep_range = np.linspace(*self.joint_limits['elbow_pitch'], resolution)

        total = resolution ** 4
        count = 0
        good_samples = 0

        for sp in sp_range:
            for sr in sr_range:
                for sy in sy_range:
                    for ep in ep_range:
                        joint_values = [sp, sr, sy, ep]
                        ee_pos, ee_quat, full_joints = self._set_joints_and_measure(joint_values, arm)

                        # Palm açısını kontrol et
                        palm_angle = self._check_palm_facing_down(ee_quat)

                        # Sadece palm kabul edilebilir açıda ise kaydet
                        # (60 derece tolerans - tam aşağı bakmak zor)
                        if palm_angle < 90:  # Kabul edilebilir
                            self.joint_samples.append(joint_values)
                            self.ee_positions.append(ee_pos.cpu().numpy())
                            self.ee_orientations.append(ee_quat.cpu().numpy())
                            self.full_joint_samples.append(full_joints.cpu().numpy())
                            good_samples += 1

                        count += 1
                        if count % 200 == 0:
                            print(f"[Mapper] Progress: {count}/{total} ({100 * count / total:.1f}%), "
                                  f"Good samples: {good_samples}")

        self.joint_samples = np.array(self.joint_samples)
        self.ee_positions = np.array(self.ee_positions)
        self.ee_orientations = np.array(self.ee_orientations)
        self.full_joint_samples = np.array(self.full_joint_samples)

        print(f"\n[Mapper] Mapping complete!")
        print(f"[Mapper] Total samples: {count}")
        print(f"[Mapper] Good samples (palm down): {good_samples} ({100 * good_samples / count:.1f}%)")
        print(f"[Mapper] Workspace bounds:")
        print(f"         X: [{self.ee_positions[:, 0].min():.3f}, {self.ee_positions[:, 0].max():.3f}]")
        print(f"         Y: [{self.ee_positions[:, 1].min():.3f}, {self.ee_positions[:, 1].max():.3f}]")
        print(f"         Z: [{self.ee_positions[:, 2].min():.3f}, {self.ee_positions[:, 2].max():.3f}]")

        return self.joint_samples, self.ee_positions

    def find_joints_for_position(self, target_pos, k_neighbors=5):
        """
        Find joint values to reach target position.
        Uses K-nearest neighbors interpolation.

        Returns:
            full_joints: [5] joint values (elbow_roll dahil)
            estimated_error: tahmini hata
        """
        target = np.array(target_pos)

        # Find distances
        distances = np.linalg.norm(self.ee_positions - target, axis=1)

        # K nearest neighbors
        k = min(k_neighbors, len(self.ee_positions))
        indices = np.argsort(distances)[:k]

        # Inverse distance weighting
        weights = 1.0 / (distances[indices] + 1e-6)
        weights = weights / weights.sum()

        # Weighted average of full joints
        full_joints = np.sum(weights[:, np.newaxis] * self.full_joint_samples[indices], axis=0)

        # Estimated error (weighted average distance)
        estimated_error = np.sum(weights * distances[indices])

        return full_joints, estimated_error

    def save_mapping(self, filepath):
        """Save mapping to file"""
        data = {
            'joint_samples': self.joint_samples,
            'ee_positions': self.ee_positions,
            'ee_orientations': self.ee_orientations,
            'full_joint_samples': self.full_joint_samples,
            'joint_limits': self.joint_limits,
            'arm_joint_names': self.arm_joint_names,
        }

        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Mapper] Saved mapping to {filepath}")
        print(f"[Mapper] File size: {os.path.getsize(filepath) / 1024:.1f} KB")

    def load_mapping(self, filepath):
        """Load mapping from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.joint_samples = data['joint_samples']
        self.ee_positions = data['ee_positions']
        self.ee_orientations = data['ee_orientations']
        self.full_joint_samples = data['full_joint_samples']
        self.joint_limits = data['joint_limits']

        print(f"[Mapper] Loaded mapping from {filepath}")
        print(f"[Mapper] {len(self.joint_samples)} samples loaded")


class G1ArmController:
    """
    High-level arm controller using workspace mapping.
    """

    def __init__(self, mapper, robot, sim):
        self.mapper = mapper
        self.robot = robot
        self.sim = sim
        self.device = robot.data.joint_pos.device

    def move_to_position(self, target_pos, arm="right", smooth=True, duration=1.0):
        """
        Eli hedef pozisyona götür (palm aşağı bakacak şekilde).

        Args:
            target_pos: [X, Y, Z] hedef pozisyon
            arm: "right" veya "left"
            smooth: Yumuşak hareket
            duration: Hareket süresi (saniye)

        Returns:
            success: Hedefe ulaşıldı mı
            final_error: Son hata
        """
        # IK solve
        full_joints, estimated_error = self.mapper.find_joints_for_position(target_pos)

        # Sol kol için mirror
        if arm == "left":
            full_joints = full_joints.copy()
            full_joints[1] = -full_joints[1]  # shoulder_roll
            full_joints[2] = -full_joints[2]  # shoulder_yaw
            full_joints[4] = -full_joints[4]  # elbow_roll

        indices = self.mapper.right_indices if arm == "right" else self.mapper.left_indices
        ee_idx = self.mapper.right_ee_idx if arm == "right" else self.mapper.left_ee_idx

        target_joints = torch.tensor(full_joints, device=self.device, dtype=torch.float32)

        if smooth:
            # Smooth interpolation
            steps = int(duration * 100)  # 100 Hz
            start_joints = self.robot.data.joint_pos[0, indices].clone()

            for step in range(steps):
                alpha = step / steps
                # Cosine interpolation
                alpha = 0.5 * (1 - np.cos(np.pi * alpha))

                current_target = start_joints + alpha * (target_joints - start_joints)

                joint_pos = self.robot.data.joint_pos.clone()
                joint_pos[0, indices] = current_target

                self.robot.set_joint_position_target(joint_pos)
                self.robot.write_data_to_sim()
                self.sim.step()
                self.robot.update(self.sim.cfg.dt)
        else:
            # Direct set
            joint_pos = self.robot.data.joint_pos.clone()
            joint_pos[0, indices] = target_joints

            self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
            for _ in range(20):
                self.robot.write_data_to_sim()
                self.sim.step()
                self.robot.update(self.sim.cfg.dt)

        # Measure final position
        final_ee = self.robot.data.body_pos_w[0, ee_idx]
        final_error = torch.norm(final_ee - torch.tensor(target_pos, device=self.device)).item()

        return final_error < 0.10, final_error  # 10cm tolerance


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.5, 1.3], [0.0, 0.0, 1.0])

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

    # Mapper oluştur
    mapper = G1WorkspaceMapperV2(robot, sim, device="cuda:0")

    # Başlangıç bilgileri
    init_ee = robot.data.body_pos_w[0, mapper.right_ee_idx].clone()
    root_pos = robot.data.root_pos_w[0]

    print(f"\n[INFO] Robot root: ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
    print(f"[INFO] Initial right palm: ({init_ee[0]:.3f}, {init_ee[1]:.3f}, {init_ee[2]:.3f})")

    # === PHASE 1: Workspace Mapping ===
    print("\n" + "=" * 60)
    print("  PHASE 1: WORKSPACE MAPPING (Palm Always Down)")
    print("=" * 60)

    resolution = args.resolution
    mapper.map_workspace(resolution=resolution, arm="right")

    # Save path - proje klasörüne
    if args.save_dir:
        save_dir = args.save_dir
    else:
        # Default: test klasörüne
        save_dir = "source/isaaclab_tasks/isaaclab_tasks/direct/isaac_g1_ulc/g1/isaac_g1_ulc/data"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"g1_workspace_map_res{resolution}.pkl")
    mapper.save_mapping(save_path)

    # === PHASE 2: Test Arm Controller ===
    print("\n" + "=" * 60)
    print("  PHASE 2: TEST ARM CONTROLLER")
    print("=" * 60)

    controller = G1ArmController(mapper, robot, sim)

    # Test hedefleri (workspace içinde)
    x_min, x_max = mapper.ee_positions[:, 0].min(), mapper.ee_positions[:, 0].max()
    y_min, y_max = mapper.ee_positions[:, 1].min(), mapper.ee_positions[:, 1].max()
    z_min, z_max = mapper.ee_positions[:, 2].min(), mapper.ee_positions[:, 2].max()

    # Güvenli hedefler
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    test_targets = [
        ("Önde-Orta", [x_mid + 0.1, y_mid, z_mid]),
        ("Önde-Aşağı", [x_mid + 0.1, y_mid, z_mid - 0.15]),
        ("Önde-Yukarı", [x_mid + 0.1, y_mid, z_mid + 0.1]),
        ("Yanda", [x_mid, y_mid - 0.1, z_mid]),
        ("Uzak-İleri", [x_max * 0.9, y_mid, z_mid]),
    ]

    print(f"\n[INFO] Workspace bounds:")
    print(f"       X: [{x_min:.3f}, {x_max:.3f}]")
    print(f"       Y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"       Z: [{z_min:.3f}, {z_max:.3f}]")

    results = []

    for name, target in test_targets:
        print(f"\n--- {name} ---")
        print(f"Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

        success, error = controller.move_to_position(target, "right", smooth=True, duration=1.5)

        # Measure
        actual_ee = robot.data.body_pos_w[0, mapper.right_ee_idx]
        palm_quat = robot.data.body_quat_w[0, mapper.right_ee_idx]
        palm_angle = mapper._check_palm_facing_down(palm_quat)

        print(f"Actual:  ({actual_ee[0]:.3f}, {actual_ee[1]:.3f}, {actual_ee[2]:.3f})")
        print(f"Error:   {error * 100:.1f}cm")
        print(f"Palm angle from down: {palm_angle:.1f}°")

        results.append({
            'name': name,
            'target': target,
            'actual': actual_ee.cpu().numpy(),
            'error': error,
            'palm_angle': palm_angle,
            'success': success,
        })

        # Hold
        for _ in range(100):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    avg_error = np.mean([r['error'] for r in results])
    avg_palm_angle = np.mean([r['palm_angle'] for r in results])
    success_count = sum([r['success'] for r in results])

    print(f"""
  Workspace Mapping v2 Results:

     Resolution: {resolution} per joint
     Total samples: {resolution ** 4}
     Good samples: {len(mapper.ee_positions)} (palm down)

     Workspace:
       X: [{x_min:.3f}, {x_max:.3f}] ({x_max - x_min:.2f}m)
       Y: [{y_min:.3f}, {y_max:.3f}] ({y_max - y_min:.2f}m)
       Z: [{z_min:.3f}, {z_max:.3f}] ({z_max - z_min:.2f}m)

  Position Accuracy:
     Average error: {avg_error * 100:.1f}cm
     Success rate: {success_count}/{len(results)}

  Palm Orientation:
     Average palm angle: {avg_palm_angle:.1f}° from down
     (0° = perfect down, 90° = horizontal)

  Saved to: {save_path}
""")

    # Test detayları
    print("  Test Results:")
    for r in results:
        pos_status = "✅" if r['error'] < 0.05 else "⚠️" if r['error'] < 0.10 else "❌"
        palm_status = "✅" if r['palm_angle'] < 45 else "⚠️" if r['palm_angle'] < 70 else "❌"
        print(
            f"     {pos_status}{palm_status} {r['name']:12}: pos={r['error'] * 100:.1f}cm, palm={r['palm_angle']:.0f}°")

    # === DEMO: Continuous Movement ===
    print("\n" + "=" * 60)
    print("  DEMO: CONTINUOUS MOVEMENT")
    print("=" * 60)

    print("\n[INFO] Moving through a path...")

    # Path of points
    path = [
        [x_mid, y_mid, z_mid],
        [x_mid + 0.1, y_mid, z_mid - 0.1],
        [x_mid + 0.15, y_mid - 0.05, z_mid],
        [x_mid + 0.1, y_mid + 0.05, z_mid + 0.05],
        [x_mid, y_mid, z_mid],
    ]

    for i, target in enumerate(path):
        print(f"  Point {i + 1}/{len(path)}: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        controller.move_to_position(target, "right", smooth=True, duration=0.8)

    print("\n[INFO] Demo complete!")

    # Final hold
    print("[INFO] Holding final pose (3 seconds)...")
    for _ in range(300):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()