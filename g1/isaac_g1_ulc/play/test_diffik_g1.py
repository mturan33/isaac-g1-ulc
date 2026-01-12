#!/usr/bin/env python3
"""
G1 DiffIK Test Script
======================

EĞİTİMDEN ÖNCE DiffIK'ın çalıştığını doğrula!

TEST SENARYOLARI:
1. Sağ kolu Y+ yönüne (yana) götür
2. Sağ kolu X+ yönüne (ileri) götür
3. Sağ kolu Z- yönüne (aşağı) götür
4. Dairesel hareket

HER SENARYO İÇİN:
- Hedef pozisyon
- Jacobian hesaplama
- IK çözümü
- Hata ölçümü

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p source/isaaclab_tasks/.../play/test_diffik_g1.py --num_envs 1
"""

import argparse
import math
import time

parser = argparse.ArgumentParser(description="G1 DiffIK Test")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--test", type=str, default="all",
                    choices=["side", "forward", "down", "circle", "all"])
parser.add_argument("--duration", type=float, default=3.0, help="Test duration per scenario (s)")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

print("\n" + "=" * 70)
print("  G1 DiffIK TEST")
print("  Verifying Jacobian-based IK before training")
print("=" * 70 + "\n")


class DiffIKTester:
    """DiffIK tester for G1 robot."""

    def __init__(self, robot, device="cuda:0"):
        self.robot = robot
        self.device = device

        # IK parameters
        self.damping = 0.05
        self.max_delta = 0.05  # rad per step (larger for testing)

        # Find joint indices
        self._find_joint_indices()

        # Find EE body index
        self._find_ee_index()

        print(f"\n[DiffIK] Initialized")
        print(f"  Damping: {self.damping}")
        print(f"  Max delta: {self.max_delta} rad/step")

    def _find_joint_indices(self):
        """Find right arm joint indices."""
        joint_names = self.robot.data.joint_names

        # G1 right arm joints
        arm_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ]

        self.arm_indices = []
        print(f"\n[DiffIK] Finding right arm joints...")

        for name in arm_joint_names:
            found = False
            for i, jname in enumerate(joint_names):
                if name in jname or name.replace("_joint", "") in jname:
                    self.arm_indices.append(i)
                    print(f"  ✅ {name} -> index {i}")
                    found = True
                    break
            if not found:
                # Try partial match
                for i, jname in enumerate(joint_names):
                    if "right" in jname and any(x in jname for x in ["shoulder", "elbow"]):
                        if i not in self.arm_indices:
                            self.arm_indices.append(i)
                            print(f"  ⚠️ {name} -> approx match: {jname} (index {i})")
                            break

        print(f"  Total arm joints found: {len(self.arm_indices)}")

        if len(self.arm_indices) < 4:
            print("\n[ERROR] Not enough arm joints! Available joints:")
            for i, name in enumerate(joint_names):
                print(f"  [{i:2d}] {name}")
            raise RuntimeError("Cannot find arm joints")

        self.arm_indices = torch.tensor(self.arm_indices, device=self.device)

    def _find_ee_index(self):
        """Find end-effector body index."""
        body_names = self.robot.data.body_names

        candidates = [
            "right_wrist_yaw_link",
            "right_palm_link",
            "right_hand_link",
            "right_wrist_link",
        ]

        self.ee_idx = None
        print(f"\n[DiffIK] Finding end-effector body...")

        for candidate in candidates:
            for i, name in enumerate(body_names):
                if candidate in name:
                    self.ee_idx = i
                    print(f"  ✅ Found: {name} (index {i})")
                    break
            if self.ee_idx is not None:
                break

        if self.ee_idx is None:
            # Fallback - try to find anything with "right" and "wrist" or "hand"
            for i, name in enumerate(body_names):
                if "right" in name.lower() and ("wrist" in name.lower() or "hand" in name.lower()):
                    self.ee_idx = i
                    print(f"  ⚠️ Using fallback: {name} (index {i})")
                    break

        if self.ee_idx is None:
            print("\n[ERROR] Cannot find end-effector! Available bodies:")
            for i, name in enumerate(body_names):
                print(f"  [{i:2d}] {name}")
            raise RuntimeError("Cannot find EE body")

    def get_ee_pos(self):
        """Get current EE position."""
        return self.robot.data.body_pos_w[:, self.ee_idx].clone()

    def get_jacobian(self):
        """Get Jacobian for right arm."""
        try:
            jacobians = self.robot.root_physx_view.get_jacobians()
            # Shape: (num_envs, num_bodies, 6, num_dofs)
            # We want position rows (0:3) for arm joints
            J = jacobians[:, self.ee_idx, :3, :][:, :, self.arm_indices]
            return J  # (num_envs, 3, num_arm_joints)
        except Exception as e:
            print(f"[ERROR] Failed to get Jacobian: {e}")
            return None

    def compute_ik(self, target_pos):
        """
        Compute IK using Damped Least Squares.

        delta_q = J^T (J J^T + λ²I)^{-1} error
        """
        # Current EE position
        current_pos = self.get_ee_pos()

        # Error
        error = target_pos - current_pos  # (num_envs, 3)

        # Get Jacobian
        J = self.get_jacobian()
        if J is None:
            return None, None

        batch_size = J.shape[0]
        num_joints = J.shape[2]

        # J J^T
        JJT = torch.bmm(J, J.transpose(1, 2))  # (B, 3, 3)

        # Add damping: (J J^T + λ²I)
        damping_eye = (self.damping ** 2) * torch.eye(3, device=self.device)
        damping_eye = damping_eye.unsqueeze(0).expand(batch_size, -1, -1)
        JJT_damped = JJT + damping_eye

        # Solve (J J^T + λ²I) x = error
        try:
            x = torch.linalg.solve(JJT_damped, error.unsqueeze(-1))  # (B, 3, 1)
        except Exception as e:
            print(f"[WARN] linalg.solve failed: {e}")
            return None, error.norm(dim=-1)

        # delta_q = J^T x
        delta_q = torch.bmm(J.transpose(1, 2), x).squeeze(-1)  # (B, num_joints)

        # Clamp
        delta_q = torch.clamp(delta_q, -self.max_delta, self.max_delta)

        return delta_q, error.norm(dim=-1)

    def apply_ik(self, delta_q):
        """Apply IK solution to robot."""
        if delta_q is None:
            return

        # Get current joint positions
        current_joints = self.robot.data.joint_pos.clone()

        # Update arm joints
        current_joints[:, self.arm_indices] += delta_q

        # Clamp to limits
        current_joints[:, self.arm_indices] = torch.clamp(
            current_joints[:, self.arm_indices], -2.6, 2.6
        )

        # Apply
        self.robot.set_joint_position_target(current_joints)


def run_test(tester, sim, target_offset, test_name, duration=3.0):
    """Run a single IK test."""
    print(f"\n{'=' * 50}")
    print(f"  TEST: {test_name}")
    print(f"  Target offset: {target_offset.tolist()}")
    print(f"{'=' * 50}")

    # Get initial position
    tester.robot.update(sim.cfg.dt)
    init_ee_pos = tester.get_ee_pos()
    target_pos = init_ee_pos + target_offset.unsqueeze(0).to(tester.device)

    print(f"  Initial EE: {init_ee_pos[0].tolist()}")
    print(f"  Target EE:  {target_pos[0].tolist()}")

    # Run IK loop
    steps = int(duration / sim.cfg.dt)
    best_error = float('inf')

    for step in range(steps):
        # Compute IK
        delta_q, error = tester.compute_ik(target_pos)

        if error is not None:
            err_val = error[0].item()
            if err_val < best_error:
                best_error = err_val

        # Apply
        tester.apply_ik(delta_q)
        tester.robot.write_data_to_sim()

        # Step sim
        sim.step()
        tester.robot.update(sim.cfg.dt)

        # Log every 50 steps
        if step % 50 == 0 and error is not None:
            current_pos = tester.get_ee_pos()[0]
            print(f"  [{step:3d}] Error: {err_val:.4f}m | "
                  f"EE: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")

    # Final result
    final_pos = tester.get_ee_pos()[0]
    final_error = (final_pos - target_pos[0]).norm().item()

    print(f"\n  RESULT:")
    print(f"    Final EE:    ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"    Target:      ({target_pos[0, 0]:.4f}, {target_pos[0, 1]:.4f}, {target_pos[0, 2]:.4f})")
    print(f"    Final error: {final_error:.4f}m")
    print(f"    Best error:  {best_error:.4f}m")

    if final_error < 0.05:
        print(f"    ✅ PASS - Error < 5cm")
        return True
    elif final_error < 0.1:
        print(f"    ⚠️ ACCEPTABLE - Error < 10cm")
        return True
    else:
        print(f"    ❌ FAIL - Error >= 10cm")
        return False


def main():
    # ============================================================
    # Setup Simulation
    # ============================================================
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])

    # ============================================================
    # Load G1 Robot
    # ============================================================
    print("[INFO] Loading G1 robot...")

    # Try different import paths
    robot_cfg = None

    try:
        from isaaclab_assets import G1_MINIMAL_CFG
        robot_cfg = G1_MINIMAL_CFG.copy()
        print("[INFO] Using G1_MINIMAL_CFG")
    except ImportError:
        pass

    if robot_cfg is None:
        try:
            from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
            robot_cfg = G1_MINIMAL_CFG.copy()
            print("[INFO] Using isaaclab_assets.robots.unitree.G1_MINIMAL_CFG")
        except ImportError:
            pass

    if robot_cfg is None:
        try:
            from isaaclab_assets import G1_CFG
            robot_cfg = G1_CFG.copy()
            print("[INFO] Using G1_CFG")
        except ImportError:
            pass

    if robot_cfg is None:
        print("[ERROR] Cannot find G1 config!")
        return

    # Configure robot
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)

    # Create scene
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    # Create robot
    robot = Articulation(cfg=robot_cfg)

    # Reset simulation
    sim.reset()
    robot.update(sim.cfg.dt)

    print(f"\n[INFO] Robot created")
    print(f"  Joints: {robot.num_joints}")
    print(f"  Bodies: {robot.num_bodies}")

    # ============================================================
    # Create Tester
    # ============================================================
    tester = DiffIKTester(robot, device=sim.device)

    # ============================================================
    # Run Tests
    # ============================================================
    results = {}

    tests = {
        "side": torch.tensor([0.0, 0.15, 0.0]),  # Y+ 15cm
        "forward": torch.tensor([0.15, 0.0, 0.0]),  # X+ 15cm
        "down": torch.tensor([0.0, 0.0, -0.15]),  # Z- 15cm
    }

    if args.test == "all":
        for name, offset in tests.items():
            results[name] = run_test(tester, sim, offset, name.upper(), args.duration)
            # Reset robot between tests
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos,
                robot.data.default_joint_vel
            )
            sim.step()
            robot.update(sim.cfg.dt)

    elif args.test == "circle":
        # Circle test
        print(f"\n{'=' * 50}")
        print(f"  TEST: CIRCLE")
        print(f"{'=' * 50}")

        robot.update(sim.cfg.dt)
        init_ee_pos = tester.get_ee_pos()

        radius = 0.1
        freq = 0.3
        steps = int(args.duration * 3 / sim.cfg.dt)  # 3x duration for circle

        errors = []

        for step in range(steps):
            t = step * sim.cfg.dt
            angle = 2 * math.pi * freq * t

            offset = torch.tensor([
                0.0,
                radius * math.sin(angle),
                radius * math.cos(angle)
            ], device=sim.device)

            target_pos = init_ee_pos + offset.unsqueeze(0)

            delta_q, error = tester.compute_ik(target_pos)
            if error is not None:
                errors.append(error[0].item())

            tester.apply_ik(delta_q)
            tester.robot.write_data_to_sim()
            sim.step()
            tester.robot.update(sim.cfg.dt)

            if step % 100 == 0:
                current = tester.get_ee_pos()[0]
                print(f"  [{step:4d}] t={t:.2f}s | Error: {errors[-1]:.4f}m | "
                      f"EE: ({current[1]:.3f}, {current[2]:.3f})")

        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        print(f"\n  RESULT:")
        print(f"    Avg error: {avg_error:.4f}m")
        print(f"    Max error: {max_error:.4f}m")
        results["circle"] = avg_error < 0.1

    else:
        if args.test in tests:
            results[args.test] = run_test(tester, sim, tests[args.test],
                                          args.test.upper(), args.duration)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  DIFFIK TEST SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name.upper():10s}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🎉 ALL TESTS PASSED!")
        print("  DiffIK is working correctly.")
        print("  You can proceed with Stage 4 training.")
    else:
        print("  ⚠️ SOME TESTS FAILED")
        print("  Check joint indices and EE body index.")
        print("  May need to adjust IK parameters.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()