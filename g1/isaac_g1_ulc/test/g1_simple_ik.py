"""
G1 Simple IK Controller - Windows Compatible
=============================================

Pink IK Windows'ta çalışmıyor (Pinocchio bağımlılığı).
Bu controller test sonuçlarımızdan öğrenilen kinematikleri kullanıyor.

Özellikler:
- Pure PyTorch (GPU accelerated)
- Windows 11 compatible
- G1 arm kinematikleri için optimize
- Jacobian-based differential IK

Test Sonuçlarından Öğrenilenler:
- shoulder_pitch NEG = öne (+X) ve yukarı (+Z)
- elbow_pitch POS = dirsek büker, öne uzatır
- shoulder_yaw = sadece yana (Y) hareket
- shoulder_roll = sadece yana (Y) hareket

KULLANIM:
    from g1_simple_ik import G1SimpleIK

    ik = G1SimpleIK(device="cuda:0")
    target_pos = torch.tensor([[0.4, -0.2, 1.2]])  # [batch, 3]
    joint_deltas = ik.compute(current_joints, target_pos, ee_pos)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class G1ArmConfig:
    """G1 arm configuration from our tests"""
    # Joint indices (right arm)
    shoulder_pitch_idx: int = 6
    shoulder_roll_idx: int = 10
    shoulder_yaw_idx: int = 14
    elbow_pitch_idx: int = 18
    elbow_roll_idx: int = 22

    # Joint limits (radians)
    shoulder_pitch_limits: Tuple[float, float] = (-2.97, 2.79)
    shoulder_roll_limits: Tuple[float, float] = (-2.25, 1.59)
    shoulder_yaw_limits: Tuple[float, float] = (-2.62, 2.62)
    elbow_pitch_limits: Tuple[float, float] = (-0.23, 3.42)
    elbow_roll_limits: Tuple[float, float] = (-2.09, 2.09)

    # Jacobian coefficients from tests (approximate, meters per radian)
    # Format: [dX/dq, dY/dq, dZ/dq] for each joint
    # These are empirical values from test_joint_explorer.py
    shoulder_pitch_jacobian: Tuple[float, float, float] = (0.31, 0.09, 0.31)  # NEG = forward
    shoulder_roll_jacobian: Tuple[float, float, float] = (0.00, 0.30, 0.01)  # lateral
    shoulder_yaw_jacobian: Tuple[float, float, float] = (0.01, 0.35, 0.00)  # lateral
    elbow_pitch_jacobian: Tuple[float, float, float] = (0.10, 0.00, 0.21)  # NEG = forward
    elbow_roll_jacobian: Tuple[float, float, float] = (0.00, 0.00, 0.00)  # minimal effect


class G1SimpleIK:
    """
    Simple Jacobian-based IK for G1 arms.
    Works on Windows without Pinocchio dependency.
    """

    def __init__(
            self,
            device: str = "cuda:0",
            num_envs: int = 1,
            damping: float = 0.05,
            max_iterations: int = 10,
            position_tolerance: float = 0.01,
    ):
        self.device = device
        self.num_envs = num_envs
        self.damping = damping
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance

        # Arm configs
        self.right_arm = G1ArmConfig()
        self.left_arm = G1ArmConfig(
            shoulder_pitch_idx=5,
            shoulder_roll_idx=9,
            shoulder_yaw_idx=13,
            elbow_pitch_idx=17,
            elbow_roll_idx=21,
        )

        # Build empirical Jacobian matrices
        self._build_jacobians()

    def _build_jacobians(self):
        """Build Jacobian matrices from empirical test data"""
        # Right arm Jacobian (5 joints x 3 DOF)
        # Sign convention: what happens when joint value INCREASES
        # shoulder_pitch: POS = backward, so NEG coefficient for forward
        # elbow_pitch: POS = backward, so NEG coefficient for forward

        self.right_jacobian = torch.tensor([
            # dX,    dY,    dZ    (effect of positive joint movement)
            [-0.31, -0.09, -0.31],  # shoulder_pitch (NEG moves forward)
            [0.00, 0.30, 0.01],  # shoulder_roll
            [0.01, 0.35, 0.00],  # shoulder_yaw
            [-0.10, 0.00, -0.21],  # elbow_pitch (NEG moves forward)
            [0.00, 0.00, 0.00],  # elbow_roll (minimal effect)
        ], device=self.device, dtype=torch.float32)

        # Left arm Jacobian (mirrored Y)
        self.left_jacobian = torch.tensor([
            [-0.31, 0.09, -0.31],  # shoulder_pitch
            [0.00, -0.30, 0.01],  # shoulder_roll (Y mirrored)
            [0.01, -0.35, 0.00],  # shoulder_yaw (Y mirrored)
            [-0.10, 0.00, -0.21],  # elbow_pitch
            [0.00, 0.00, 0.00],  # elbow_roll
        ], device=self.device, dtype=torch.float32)

        # Joint limits
        self.right_limits_low = torch.tensor([
            self.right_arm.shoulder_pitch_limits[0],
            self.right_arm.shoulder_roll_limits[0],
            self.right_arm.shoulder_yaw_limits[0],
            self.right_arm.elbow_pitch_limits[0],
            self.right_arm.elbow_roll_limits[0],
        ], device=self.device, dtype=torch.float32)

        self.right_limits_high = torch.tensor([
            self.right_arm.shoulder_pitch_limits[1],
            self.right_arm.shoulder_roll_limits[1],
            self.right_arm.shoulder_yaw_limits[1],
            self.right_arm.elbow_pitch_limits[1],
            self.right_arm.elbow_roll_limits[1],
        ], device=self.device, dtype=torch.float32)

        self.left_limits_low = self.right_limits_low.clone()
        self.left_limits_high = self.right_limits_high.clone()

    def compute_damped_pseudoinverse(
            self,
            jacobian: torch.Tensor,
            damping: float = None,
    ) -> torch.Tensor:
        """
        Compute damped pseudo-inverse (DLS method)
        J_dls = J^T (J J^T + λ²I)^(-1)
        """
        if damping is None:
            damping = self.damping

        # J: [5, 3] or [batch, 5, 3]
        if jacobian.dim() == 2:
            jacobian = jacobian.unsqueeze(0)  # [1, 5, 3]

        batch_size = jacobian.shape[0]

        # J J^T: [batch, 5, 5]
        JJT = torch.bmm(jacobian, jacobian.transpose(-2, -1))

        # Add damping: J J^T + λ²I
        damping_matrix = (damping ** 2) * torch.eye(
            jacobian.shape[1], device=self.device
        ).unsqueeze(0).expand(batch_size, -1, -1)

        JJT_damped = JJT + damping_matrix

        # Inverse: (J J^T + λ²I)^(-1)
        JJT_inv = torch.linalg.inv(JJT_damped)

        # J^T (J J^T + λ²I)^(-1): [batch, 3, 5]
        J_pinv = torch.bmm(jacobian.transpose(-2, -1), JJT_inv)

        return J_pinv.squeeze(0) if batch_size == 1 else J_pinv

    def compute(
            self,
            current_joint_pos: torch.Tensor,
            target_ee_pos: torch.Tensor,
            current_ee_pos: torch.Tensor,
            arm: str = "right",
            dt: float = 0.02,
    ) -> torch.Tensor:
        """
        Compute joint position deltas to reach target EE position.

        Args:
            current_joint_pos: Current joint positions [batch, 5] or [5]
            target_ee_pos: Target end-effector position [batch, 3] or [3]
            current_ee_pos: Current end-effector position [batch, 3] or [3]
            arm: "right" or "left"
            dt: Time step for velocity integration

        Returns:
            joint_deltas: Joint position changes [batch, 5] or [5]
        """
        # Ensure batch dimension
        squeeze_output = False
        if current_joint_pos.dim() == 1:
            current_joint_pos = current_joint_pos.unsqueeze(0)
            target_ee_pos = target_ee_pos.unsqueeze(0)
            current_ee_pos = current_ee_pos.unsqueeze(0)
            squeeze_output = True

        batch_size = current_joint_pos.shape[0]

        # Select Jacobian and limits
        if arm == "right":
            jacobian = self.right_jacobian
            limits_low = self.right_limits_low
            limits_high = self.right_limits_high
        else:
            jacobian = self.left_jacobian
            limits_low = self.left_limits_low
            limits_high = self.left_limits_high

        # Expand Jacobian for batch
        jacobian = jacobian.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 5, 3]

        # Position error
        pos_error = target_ee_pos - current_ee_pos  # [batch, 3]

        # Clip error magnitude for stability
        error_norm = torch.norm(pos_error, dim=-1, keepdim=True)
        max_error = 0.5  # meters
        pos_error = torch.where(
            error_norm > max_error,
            pos_error * max_error / (error_norm + 1e-6),
            pos_error
        )

        # Compute pseudo-inverse
        J_pinv = self.compute_damped_pseudoinverse(jacobian)  # [batch, 3, 5]

        # Compute joint velocity: dq = J_pinv @ dx
        # J_pinv: [batch, 3, 5], pos_error: [batch, 3]
        joint_vel = torch.bmm(
            J_pinv.transpose(-2, -1),  # [batch, 5, 3]
            pos_error.unsqueeze(-1)  # [batch, 3, 1]
        ).squeeze(-1)  # [batch, 5]

        # Scale by gain (proportional control)
        gain = 5.0
        joint_deltas = gain * joint_vel * dt

        # Clip joint deltas for safety
        max_delta = 0.5  # radians per step
        joint_deltas = torch.clamp(joint_deltas, -max_delta, max_delta)

        # Apply joint limits
        new_joints = current_joint_pos + joint_deltas
        new_joints = torch.clamp(new_joints, limits_low, limits_high)
        joint_deltas = new_joints - current_joint_pos

        if squeeze_output:
            joint_deltas = joint_deltas.squeeze(0)

        return joint_deltas

    def compute_target_joints(
            self,
            current_joint_pos: torch.Tensor,
            target_ee_pos: torch.Tensor,
            current_ee_pos: torch.Tensor,
            arm: str = "right",
            dt: float = 0.02,
    ) -> torch.Tensor:
        """
        Compute target joint positions (current + delta).

        Returns:
            target_joints: Target joint positions [batch, 5] or [5]
        """
        joint_deltas = self.compute(
            current_joint_pos, target_ee_pos, current_ee_pos, arm, dt
        )
        return current_joint_pos + joint_deltas

    def forward_kinematics_approx(
            self,
            joint_pos: torch.Tensor,
            base_ee_pos: torch.Tensor,
            arm: str = "right",
    ) -> torch.Tensor:
        """
        Approximate forward kinematics using linear Jacobian model.

        Note: This is a first-order approximation, good for small movements.
        For accurate FK, use the simulator.

        Args:
            joint_pos: Joint positions [batch, 5] or [5]
            base_ee_pos: EE position at zero joint angles [batch, 3] or [3]
            arm: "right" or "left"

        Returns:
            ee_pos: Approximate EE position [batch, 3] or [3]
        """
        squeeze_output = False
        if joint_pos.dim() == 1:
            joint_pos = joint_pos.unsqueeze(0)
            base_ee_pos = base_ee_pos.unsqueeze(0)
            squeeze_output = True

        if arm == "right":
            jacobian = self.right_jacobian
        else:
            jacobian = self.left_jacobian

        # Linear approximation: ee_pos ≈ base_ee_pos + J @ q
        ee_delta = torch.mm(joint_pos, jacobian)  # [batch, 3]
        ee_pos = base_ee_pos + ee_delta

        if squeeze_output:
            ee_pos = ee_pos.squeeze(0)

        return ee_pos


class G1DualArmIK:
    """
    Dual arm IK controller for G1.
    Controls both arms simultaneously.
    """

    def __init__(
            self,
            device: str = "cuda:0",
            num_envs: int = 1,
            damping: float = 0.05,
    ):
        self.device = device
        self.num_envs = num_envs
        self.ik = G1SimpleIK(device, num_envs, damping)

        # Full robot joint indices
        self.right_arm_indices = torch.tensor([6, 10, 14, 18, 22], device=device)
        self.left_arm_indices = torch.tensor([5, 9, 13, 17, 21], device=device)

    def compute(
            self,
            current_joints: torch.Tensor,
            right_target: Optional[torch.Tensor],
            left_target: Optional[torch.Tensor],
            right_ee_pos: torch.Tensor,
            left_ee_pos: torch.Tensor,
            dt: float = 0.02,
    ) -> torch.Tensor:
        """
        Compute joint targets for both arms.

        Args:
            current_joints: All robot joints [batch, num_joints]
            right_target: Right arm target position [batch, 3] or None
            left_target: Left arm target position [batch, 3] or None
            right_ee_pos: Current right EE position [batch, 3]
            left_ee_pos: Current left EE position [batch, 3]
            dt: Time step

        Returns:
            target_joints: Full robot joint targets [batch, num_joints]
        """
        target_joints = current_joints.clone()

        # Right arm
        if right_target is not None:
            right_joints = current_joints[:, self.right_arm_indices]
            right_deltas = self.ik.compute(
                right_joints, right_target, right_ee_pos, "right", dt
            )
            target_joints[:, self.right_arm_indices] += right_deltas

        # Left arm
        if left_target is not None:
            left_joints = current_joints[:, self.left_arm_indices]
            left_deltas = self.ik.compute(
                left_joints, left_target, left_ee_pos, "left", dt
            )
            target_joints[:, self.left_arm_indices] += left_deltas

        return target_joints


# ============================================================
# Convenience functions for direct use
# ============================================================

def create_g1_ik(device: str = "cuda:0") -> G1SimpleIK:
    """Create a G1 IK controller"""
    return G1SimpleIK(device=device)


def create_g1_dual_arm_ik(device: str = "cuda:0") -> G1DualArmIK:
    """Create a G1 dual arm IK controller"""
    return G1DualArmIK(device=device)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  G1 Simple IK Controller Test")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")

    # Create controller
    ik = G1SimpleIK(device=device)

    # Test data
    current_joints = torch.zeros(5, device=device)
    current_ee = torch.tensor([0.214, -0.162, 1.135], device=device)
    target_ee = torch.tensor([0.35, -0.15, 1.15], device=device)  # Forward target

    print(f"\n[TEST] Current EE: {current_ee.tolist()}")
    print(f"[TEST] Target EE:  {target_ee.tolist()}")

    # Compute IK
    joint_deltas = ik.compute(
        current_joints, target_ee, current_ee, arm="right"
    )

    print(f"\n[RESULT] Joint deltas:")
    print(f"  shoulder_pitch: {joint_deltas[0].item():+.4f} rad")
    print(f"  shoulder_roll:  {joint_deltas[1].item():+.4f} rad")
    print(f"  shoulder_yaw:   {joint_deltas[2].item():+.4f} rad")
    print(f"  elbow_pitch:    {joint_deltas[3].item():+.4f} rad")
    print(f"  elbow_roll:     {joint_deltas[4].item():+.4f} rad")

    # Verify: shoulder_pitch should be NEGATIVE for forward motion
    if joint_deltas[0] < 0:
        print("\n✅ shoulder_pitch is NEGATIVE (correct for forward motion)")
    else:
        print("\n⚠️ shoulder_pitch is positive (unexpected)")

    # Test batch processing
    print("\n[TEST] Batch processing (4 environments)...")
    batch_joints = torch.zeros(4, 5, device=device)
    batch_current_ee = current_ee.unsqueeze(0).expand(4, -1)
    batch_target_ee = torch.tensor([
        [0.35, -0.15, 1.15],
        [0.30, -0.20, 1.20],
        [0.40, -0.10, 1.10],
        [0.25, -0.25, 1.25],
    ], device=device)

    batch_deltas = ik.compute(
        batch_joints, batch_target_ee, batch_current_ee, arm="right"
    )

    print(f"  Batch shape: {batch_deltas.shape}")
    print(f"  All shoulder_pitch negative: {(batch_deltas[:, 0] < 0).all().item()}")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)