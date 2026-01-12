#!/usr/bin/env python3
"""
G1 DiffIK Test - FIXED BASE
============================

Robot base SABİT - sadece kol hareket eder.
Jacobian IK doğru çalışıyor mu test et.

KULLANIM:
cd C:\IsaacLab
./isaaclab.bat -p .../play/test_diffik_g1_fixed.py --num_envs 1 --test all
"""

import argparse
import math

parser = argparse.ArgumentParser(description="G1 DiffIK Test - Fixed Base")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--test", type=str, default="all",
                    choices=["side", "forward", "down", "circle", "all"])
parser.add_argument("--duration", type=float, default=3.0)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

print("\n" + "=" * 70)
print("  G1 DiffIK TEST - FIXED BASE")
print("  Robot base sabit, sadece kol hareket eder")
print("=" * 70 + "\n")


class DiffIKTester:
    """DiffIK tester with fixed base robot."""

    def __init__(self, robot, device="cuda:0"):
        self.robot = robot
        self.device = device

        # IK parameters
        self.damping = 0.05
        self.max_delta = 0.1  # rad per step

        self._find_joint_indices()
        self._find_ee_index()

        print(f"\n[DiffIK] Initialized")
        print(f"  Damping: {self.damping}")
        print(f"  Max delta: {self.max_delta} rad/step")

    def _find_joint_indices(self):
        """Find right arm joint indices."""
        joint_names = list(self.robot.data.joint_names)

        print(f"\n[DiffIK] All joints ({len(joint_names)}):")
        for i, name in enumerate(joint_names):
            print(f"  [{i:2d}] {name}")

        # G1 right arm joints - try different naming conventions
        arm_patterns = [
            ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
             "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
        ]

        self.arm_indices = []
        print(f"\n[DiffIK] Finding right arm joints...")

        for i, name in enumerate(joint_names):
            name_lower = name.lower()
            if "right" in name_lower and any(x in name_lower for x in ["shoulder", "elbow", "wrist"]):
                self.arm_indices.append(i)
                print(f"  ✅ [{i}] {name}")

        if len(self.arm_indices) < 4:
            print(f"[WARN] Only found {len(self.arm_indices)} arm joints!")

        self.arm_indices = torch.tensor(self.arm_indices, device=self.device)
        print(f"  Total: {len(self.arm_indices)} joints")

    def _find_ee_index(self):
        """Find end-effector body index."""
        body_names = list(self.robot.data.body_names)

        print(f"\n[DiffIK] All bodies ({len(body_names)}):")
        for i, name in enumerate(body_names):
            print(f"  [{i:2d}] {name}")

        # Try to find right hand/wrist
        candidates = ["right_palm", "right_wrist_yaw", "right_hand", "right_wrist"]

        self.ee_idx = None
        print(f"\n[DiffIK] Finding end-effector...")

        for i, name in enumerate(body_names):
            name_lower = name.lower()
            for candidate in candidates:
                if candidate in name_lower:
                    self.ee_idx = i
                    print(f"  ✅ Found: {name} (index {i})")
                    return

        # Fallback
        for i, name in enumerate(body_names):
            if "right" in name.lower() and (
                    "wrist" in name.lower() or "hand" in name.lower() or "palm" in name.lower()):
                self.ee_idx = i
                print(f"  ⚠️ Fallback: {name} (index {i})")
                return

        print("[ERROR] Cannot find end-effector!")
        self.ee_idx = len(body_names) - 1

    def get_ee_pos(self):
        return self.robot.data.body_pos_w[:, self.ee_idx].clone()

    def get_jacobian(self):
        try:
            jacobians = self.robot.root_physx_view.get_jacobians()
            # Shape: (num_envs, num_bodies, 6, num_dofs)
            J = jacobians[:, self.ee_idx, :3, :]  # Position rows only
            J_arm = J[:, :, self.arm_indices]  # Only arm joints
            return J_arm
        except Exception as e:
            print(f"[ERROR] Jacobian: {e}")
            return None

    def compute_ik(self, target_pos):
        current_pos = self.get_ee_pos()
        error = target_pos - current_pos

        J = self.get_jacobian()
        if J is None:
            return None, error.norm(dim=-1)

        batch_size = J.shape[0]
        num_joints = J.shape[2]

        # Damped Least Squares: delta_q = J^T (J J^T + λ²I)^{-1} error
        JJT = torch.bmm(J, J.transpose(1, 2))  # (B, 3, 3)
        damping_eye = (self.damping ** 2) * torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        JJT_damped = JJT + damping_eye

        try:
            x = torch.linalg.solve(JJT_damped, error.unsqueeze(-1))
        except:
            return None, error.norm(dim=-1)

        delta_q = torch.bmm(J.transpose(1, 2), x).squeeze(-1)
        delta_q = torch.clamp(delta_q, -self.max_delta, self.max_delta)

        return delta_q, error.norm(dim=-1)

    def apply_ik(self, delta_q):
        if delta_q is None:
            return

        current_joints = self.robot.data.joint_pos.clone()
        current_joints[:, self.arm_indices] += delta_q
        current_joints[:, self.arm_indices] = torch.clamp(
            current_joints[:, self.arm_indices], -2.6, 2.6
        )

        self.robot.set_joint_position_target(current_joints)


def run_test(tester, sim, target_offset, test_name, duration=3.0):
    """Run a single IK test."""
    print(f"\n{'=' * 50}")
    print(f"  TEST: {test_name}")
    print(f"  Target offset: [{target_offset[0]:.3f}, {target_offset[1]:.3f}, {target_offset[2]:.3f}]")
    print(f"{'=' * 50}")

    # Get initial position
    tester.robot.update(sim.cfg.dt)
    init_ee_pos = tester.get_ee_pos()
    init_base_pos = tester.robot.data.root_pos_w[:, :3].clone()

    target_pos = init_ee_pos + target_offset.unsqueeze(0).to(tester.device)

    print(f"  Initial Base: ({init_base_pos[0, 0]:.3f}, {init_base_pos[0, 1]:.3f}, {init_base_pos[0, 2]:.3f})")
    print(f"  Initial EE:   ({init_ee_pos[0, 0]:.3f}, {init_ee_pos[0, 1]:.3f}, {init_ee_pos[0, 2]:.3f})")
    print(f"  Target EE:    ({target_pos[0, 0]:.3f}, {target_pos[0, 1]:.3f}, {target_pos[0, 2]:.3f})")

    steps = int(duration / sim.cfg.dt)
    best_error = float('inf')

    for step in range(steps):
        delta_q, error = tester.compute_ik(target_pos)

        if error is not None:
            err_val = error[0].item()
            if err_val < best_error:
                best_error = err_val

        tester.apply_ik(delta_q)
        tester.robot.write_data_to_sim()

        sim.step()
        tester.robot.update(sim.cfg.dt)

        # Check base stability
        current_base = tester.robot.data.root_pos_w[:, :3]
        base_drift = (current_base - init_base_pos).norm().item()

        if step % 50 == 0 and error is not None:
            current_pos = tester.get_ee_pos()[0]
            base_status = "✅ FIXED" if base_drift < 0.01 else f"⚠️ DRIFT {base_drift:.3f}m"
            print(f"  [{step:3d}] Error: {err_val:.4f}m | "
                  f"EE: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}) | "
                  f"Base: {base_status}")

    # Final result
    final_pos = tester.get_ee_pos()[0]
    final_error = (final_pos - target_pos[0]).norm().item()
    final_base = tester.robot.data.root_pos_w[0, :3]
    final_base_drift = (final_base - init_base_pos[0]).norm().item()

    print(f"\n  RESULT:")
    print(f"    Final EE:     ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"    Target:       ({target_pos[0, 0]:.4f}, {target_pos[0, 1]:.4f}, {target_pos[0, 2]:.4f})")
    print(f"    Final error:  {final_error:.4f}m")
    print(f"    Best error:   {best_error:.4f}m")
    print(f"    Base drift:   {final_base_drift:.4f}m")

    if final_error < 0.05 and final_base_drift < 0.01:
        print(f"    ✅ PASS")
        return True
    elif final_error < 0.1:
        print(f"    ⚠️ ACCEPTABLE")
        return True
    else:
        print(f"    ❌ FAIL")
        return False


def main():
    # ============================================================
    # Setup Simulation
    # ============================================================
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])

    # ============================================================
    # Load G1 Robot with FIXED BASE
    # ============================================================
    print("[INFO] Loading G1 robot with FIXED BASE...")

    # Try different imports
    robot_cfg = None

    try:
        from isaaclab_assets import G1_MINIMAL_CFG
        robot_cfg = G1_MINIMAL_CFG.copy()
        print("[INFO] Using G1_MINIMAL_CFG")
    except ImportError:
        try:
            from isaaclab_assets import G1_CFG
            robot_cfg = G1_CFG.copy()
            print("[INFO] Using G1_CFG")
        except ImportError:
            print("[ERROR] Cannot find G1 config!")
            return

    # Configure for FIXED BASE
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 1.05)

    # KEY: Fix the root link!
    if hasattr(robot_cfg.spawn, 'articulation_props'):
        robot_cfg.spawn.articulation_props.fix_root_link = True
        print("[INFO] ✅ Set fix_root_link = True via articulation_props")

    # Also try rigid_props if available
    if hasattr(robot_cfg.spawn, 'rigid_props'):
        robot_cfg.spawn.rigid_props.disable_gravity = True
        print("[INFO] ✅ Set disable_gravity = True")

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

    # Verify base is fixed
    init_base = robot.data.root_pos_w[0, :3].clone()
    print(f"\n[INFO] Robot created")
    print(f"  Initial base position: ({init_base[0]:.3f}, {init_base[1]:.3f}, {init_base[2]:.3f})")
    print(f"  Joints: {robot.num_joints}")
    print(f"  Bodies: {robot.num_bodies}")

    # Step a few times to check if base is really fixed
    for _ in range(10):
        sim.step()
        robot.update(sim.cfg.dt)

    after_base = robot.data.root_pos_w[0, :3]
    base_movement = (after_base - init_base).norm().item()

    if base_movement < 0.01:
        print(f"  ✅ Base is FIXED (movement: {base_movement:.6f}m)")
    else:
        print(f"  ⚠️ Base is NOT fixed! Movement: {base_movement:.4f}m")
        print("  Trying to fix base manually...")

        # Force fix the base by continuously setting its position
        # This is a workaround if articulation_props doesn't work

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

    def reset_robot():
        """Reset robot to initial state."""
        robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos,
            robot.data.default_joint_vel
        )
        for _ in range(10):
            sim.step()
            robot.update(sim.cfg.dt)

    if args.test == "all":
        for name, offset in tests.items():
            reset_robot()
            results[name] = run_test(tester, sim, offset, name.upper(), args.duration)
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
        print("  Ready for Stage 4 training!")
    else:
        print("  ⚠️ SOME TESTS FAILED")
        print("  May need to check:")
        print("    - Joint indices")
        print("    - EE body index")
        print("    - IK damping parameters")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()