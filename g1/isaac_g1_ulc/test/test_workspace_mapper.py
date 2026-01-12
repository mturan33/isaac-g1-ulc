#!/usr/bin/env python3
"""
G1 Workspace Mapper & Lookup-Based IK
=====================================

1. Tüm joint kombinasyonlarını dene
2. Her kombinasyon için EE pozisyonunu ölç
3. Lookup table oluştur
4. IK = En yakın pozisyonu bul

Bu yaklaşım:
- Simülasyondan GERÇEK ölçüm
- Bir kere çalış, sonra hızlı lookup
- RL eğitiminde kullanılabilir

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../test/test_workspace_mapper.py --num_envs 1
"""

import argparse
import math
import pickle
import os

parser = argparse.ArgumentParser(description="G1 Workspace Mapper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--resolution", type=int, default=5, help="Samples per joint")

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
print("  G1 WORKSPACE MAPPER")
print("  Map joint space to cartesian space")
print("=" * 70 + "\n")


class G1WorkspaceMapper:
    """
    Joint space → Cartesian space mapping.
    """

    def __init__(self, robot, sim, device="cuda:0"):
        self.robot = robot
        self.sim = sim
        self.device = device
        self.dt = sim.cfg.dt

        # Joint isimleri
        joint_names = list(robot.data.joint_names)
        body_names = list(robot.data.body_names)

        arm_joint_order = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                           "elbow_pitch", "elbow_roll"]

        # Sağ kol indeksleri
        self.right_indices = []
        for order_name in arm_joint_order:
            for i, name in enumerate(joint_names):
                if "right" in name.lower() and order_name in name.lower():
                    self.right_indices.append(i)
                    break

        self.right_indices = torch.tensor(self.right_indices, device=device)

        # EE body index
        self.right_ee_idx = 29
        for i, name in enumerate(body_names):
            if "right_palm" in name.lower():
                self.right_ee_idx = i
                break

        # Joint limitleri (sadece önemli 4 joint için - elbow_roll çok az etkili)
        # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch
        self.joint_limits = [
            (-2.5, 0.5),  # shoulder_pitch: -2.5 (yukarı) to 0.5 (aşağı)
            (-1.5, 1.5),  # shoulder_roll: -1.5 (dışa) to 1.5 (içe)
            (-1.5, 1.5),  # shoulder_yaw
            (0.0, 2.8),  # elbow_pitch: 0 (düz) to 2.8 (bükük)
        ]

        # Mapping data
        self.joint_samples = []
        self.ee_positions = []

        print(f"[Mapper] Right arm indices: {self.right_indices.tolist()}")
        print(f"[Mapper] Right EE body: {self.right_ee_idx}")

    def _set_joints_and_measure(self, joint_values):
        """Set joint values and measure EE position"""
        # Full 5-joint values (elbow_roll = 0)
        full_joints = torch.tensor([
            joint_values[0],  # shoulder_pitch
            joint_values[1],  # shoulder_roll
            joint_values[2],  # shoulder_yaw
            joint_values[3],  # elbow_pitch
            0.0,  # elbow_roll
        ], device=self.device)

        # Apply
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_pos[0, self.right_indices] = full_joints

        # Write and step
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        self.robot.write_data_to_sim()

        # Multiple steps to settle
        for _ in range(5):
            self.sim.step()
            self.robot.update(self.dt)

        # Measure EE
        ee_pos = self.robot.data.body_pos_w[0, self.right_ee_idx].clone()
        return ee_pos

    def map_workspace(self, resolution=5):
        """
        Map entire workspace by sampling joint space.

        Args:
            resolution: Number of samples per joint
        """
        print(f"\n[Mapper] Mapping workspace with resolution {resolution}...")
        print(f"[Mapper] Total samples: {resolution ** 4}")

        self.joint_samples = []
        self.ee_positions = []

        # Generate samples for each joint
        samples = []
        for low, high in self.joint_limits:
            samples.append(np.linspace(low, high, resolution))

        total = resolution ** 4
        count = 0

        # Iterate through all combinations
        for sp in samples[0]:  # shoulder_pitch
            for sr in samples[1]:  # shoulder_roll
                for sy in samples[2]:  # shoulder_yaw
                    for ep in samples[3]:  # elbow_pitch
                        joint_values = [sp, sr, sy, ep]
                        ee_pos = self._set_joints_and_measure(joint_values)

                        self.joint_samples.append(joint_values)
                        self.ee_positions.append(ee_pos.cpu().numpy())

                        count += 1
                        if count % 100 == 0:
                            print(f"[Mapper] Progress: {count}/{total} ({100 * count / total:.1f}%)")

        self.joint_samples = np.array(self.joint_samples)
        self.ee_positions = np.array(self.ee_positions)

        print(f"\n[Mapper] Mapping complete!")
        print(f"[Mapper] Workspace bounds:")
        print(f"         X: [{self.ee_positions[:, 0].min():.3f}, {self.ee_positions[:, 0].max():.3f}]")
        print(f"         Y: [{self.ee_positions[:, 1].min():.3f}, {self.ee_positions[:, 1].max():.3f}]")
        print(f"         Z: [{self.ee_positions[:, 2].min():.3f}, {self.ee_positions[:, 2].max():.3f}]")

        return self.joint_samples, self.ee_positions

    def find_nearest_joints(self, target_pos):
        """
        Find joint values for nearest EE position to target.

        Args:
            target_pos: [X, Y, Z] target position

        Returns:
            joints: [4] joint values
            ee_pos: [3] actual EE position
            error: distance to target
        """
        target = np.array(target_pos)

        # Find nearest
        distances = np.linalg.norm(self.ee_positions - target, axis=1)
        nearest_idx = np.argmin(distances)

        return (
            self.joint_samples[nearest_idx],
            self.ee_positions[nearest_idx],
            distances[nearest_idx]
        )

    def save_mapping(self, filepath):
        """Save mapping to file"""
        data = {
            'joint_samples': self.joint_samples,
            'ee_positions': self.ee_positions,
            'joint_limits': self.joint_limits,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Mapper] Saved mapping to {filepath}")

    def load_mapping(self, filepath):
        """Load mapping from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.joint_samples = data['joint_samples']
        self.ee_positions = data['ee_positions']
        self.joint_limits = data['joint_limits']
        print(f"[Mapper] Loaded mapping from {filepath}")
        print(f"[Mapper] {len(self.joint_samples)} samples loaded")


class G1LookupIK:
    """
    Lookup-based IK using pre-computed workspace mapping.
    """

    def __init__(self, mapper):
        self.mapper = mapper
        self.joint_samples = torch.tensor(mapper.joint_samples, dtype=torch.float32)
        self.ee_positions = torch.tensor(mapper.ee_positions, dtype=torch.float32)

    def solve(self, target_pos, refine=True):
        """
        Solve IK using lookup + optional refinement.

        Args:
            target_pos: [3] target position
            refine: Use interpolation between nearest samples

        Returns:
            joints: [5] joint values (including elbow_roll=0)
            error: estimated error
        """
        target = torch.tensor(target_pos, dtype=torch.float32)

        # Find nearest sample
        distances = torch.norm(self.ee_positions - target, dim=1)
        nearest_idx = torch.argmin(distances)

        joints_4 = self.joint_samples[nearest_idx]
        error = distances[nearest_idx].item()

        if refine and error > 0.01:
            # Find K nearest neighbors
            k = min(8, len(self.joint_samples))
            _, indices = torch.topk(distances, k, largest=False)

            # Weighted average (inverse distance weighting)
            weights = 1.0 / (distances[indices] + 1e-6)
            weights = weights / weights.sum()

            joints_4 = (weights.unsqueeze(1) * self.joint_samples[indices]).sum(dim=0)

        # Add elbow_roll = 0
        joints_5 = torch.cat([joints_4, torch.tensor([0.0])])

        return joints_5.numpy(), error


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

    # Mapper oluştur
    mapper = G1WorkspaceMapper(robot, sim, device="cuda:0")

    # İlk EE pozisyonu
    init_ee = robot.data.body_pos_w[0, mapper.right_ee_idx].clone()
    root_pos = robot.data.root_pos_w[0]

    print(f"\n[INFO] Robot root: ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
    print(f"[INFO] Initial EE: ({init_ee[0]:.3f}, {init_ee[1]:.3f}, {init_ee[2]:.3f})")

    # === PHASE 1: Workspace Mapping ===
    print("\n" + "=" * 60)
    print("  PHASE 1: WORKSPACE MAPPING")
    print("=" * 60)

    resolution = args.resolution
    joint_samples, ee_positions = mapper.map_workspace(resolution=resolution)

    # Save mapping
    save_path = "g1_workspace_map.pkl"
    mapper.save_mapping(save_path)

    # === PHASE 2: Test Lookup IK ===
    print("\n" + "=" * 60)
    print("  PHASE 2: TEST LOOKUP IK")
    print("=" * 60)

    ik = G1LookupIK(mapper)

    # Test hedefleri (workspace içinde olmalı)
    x_min, x_max = ee_positions[:, 0].min(), ee_positions[:, 0].max()
    y_min, y_max = ee_positions[:, 1].min(), ee_positions[:, 1].max()
    z_min, z_max = ee_positions[:, 2].min(), ee_positions[:, 2].max()

    test_targets = [
        ("Center", [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]),
        ("Forward", [x_max * 0.8, (y_min + y_max) / 2, (z_min + z_max) / 2]),
        ("Up", [(x_min + x_max) / 2, (y_min + y_max) / 2, z_max * 0.9]),
        ("Down", [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min * 1.1 + 0.5]),
        ("Side", [(x_min + x_max) / 2, y_min * 0.8, (z_min + z_max) / 2]),
    ]

    print(f"\n[INFO] Workspace bounds:")
    print(f"       X: [{x_min:.3f}, {x_max:.3f}]")
    print(f"       Y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"       Z: [{z_min:.3f}, {z_max:.3f}]")

    results = []

    for name, target in test_targets:
        print(f"\n--- {name} ---")
        print(f"Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

        # IK solve
        joints, estimated_error = ik.solve(target, refine=True)

        # Apply joints
        full_joints = torch.tensor(joints, device="cuda:0")
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[0, mapper.right_indices] = full_joints

        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        for _ in range(20):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        # Measure actual
        actual_ee = robot.data.body_pos_w[0, mapper.right_ee_idx]
        actual_error = torch.norm(actual_ee - torch.tensor(target, device="cuda:0")).item()

        print(f"Actual:  ({actual_ee[0]:.3f}, {actual_ee[1]:.3f}, {actual_ee[2]:.3f})")
        print(f"Error:   {actual_error * 100:.1f}cm (estimated: {estimated_error * 100:.1f}cm)")

        results.append({
            'name': name,
            'target': target,
            'actual': actual_ee.cpu().numpy(),
            'error': actual_error,
        })

        # Hold
        for _ in range(50):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    avg_error = np.mean([r['error'] for r in results])
    max_error = np.max([r['error'] for r in results])

    print(f"""
  Workspace Mapping Results:

     Resolution: {resolution} samples per joint
     Total samples: {len(joint_samples)}

     X range: [{x_min:.3f}, {x_max:.3f}] ({x_max - x_min:.2f}m)
     Y range: [{y_min:.3f}, {y_max:.3f}] ({y_max - y_min:.2f}m)
     Z range: [{z_min:.3f}, {z_max:.3f}] ({z_max - z_min:.2f}m)

  Lookup IK Results:
     Average error: {avg_error * 100:.1f}cm
     Max error: {max_error * 100:.1f}cm

  Saved to: {save_path}

  📊 Higher resolution = lower error:
     resolution=5  → ~625 samples  → ~5-10cm error
     resolution=10 → ~10000 samples → ~2-3cm error
     resolution=15 → ~50000 samples → ~1cm error
""")

    # Test detayları
    print("  Test Results:")
    for r in results:
        status = "✅" if r['error'] < 0.05 else "⚠️" if r['error'] < 0.10 else "❌"
        print(f"     {status} {r['name']:10}: {r['error'] * 100:.1f}cm")

    # Final hold
    print("\n[INFO] Holding final pose (3 seconds)...")
    for _ in range(300):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("=" * 60)


if __name__ == "__main__":
    main()
    simulation_app.close()