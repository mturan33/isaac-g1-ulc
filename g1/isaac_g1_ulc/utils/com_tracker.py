"""
Center of Mass (CoM) Tracker
============================

ULC için kritik bileşen - Robot'un ağırlık merkezini hesaplar ve
support polygon'a göre stabilite değerlendirir.

Bu, senin "drift" problemini çözen ana mekanizma!
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


class CoMTracker:
    """
    Center of Mass tracker for humanoid balance.

    Hesaplar:
    - CoM pozisyonu (3D)
    - CoM hızı (3D)
    - Support polygon mesafesi
    - Stabilite margin'i
    """

    def __init__(self, env: DirectRLEnv, device: str = "cuda"):
        """
        Args:
            env: Isaac Lab DirectRLEnv instance
            device: Torch device
        """
        self.env = env
        self.device = device
        self.num_envs = env.num_envs

        # Robot reference
        self.robot = env.scene["robot"]

        # Body indices (G1 specific - will be set after robot spawns)
        self._body_indices_initialized = False
        self.left_foot_idx = None
        self.right_foot_idx = None
        self.pelvis_idx = None

        # Link masses (will be populated)
        self.link_masses = None
        self.total_mass = None

        # Target CoM height (standing)
        self.target_com_height = 0.75  # meters

        # Support polygon parameters
        self.foot_half_length = 0.10  # G1 foot ~20cm
        self.foot_half_width = 0.05  # G1 foot ~10cm

        # Safety margins
        self.max_safe_distance = 0.12  # meters from support center

    def initialize(self):
        """Initialize after robot is spawned."""
        if self._body_indices_initialized:
            return

        try:
            # Get body names
            body_names = self.robot.data.body_names

            # Find foot indices
            for i, name in enumerate(body_names):
                if "left" in name.lower() and ("foot" in name.lower() or "ankle" in name.lower()):
                    self.left_foot_idx = i
                elif "right" in name.lower() and ("foot" in name.lower() or "ankle" in name.lower()):
                    self.right_foot_idx = i
                elif "pelvis" in name.lower() or "base" in name.lower():
                    self.pelvis_idx = i

            # Fallback if not found
            if self.left_foot_idx is None:
                self.left_foot_idx = 0
            if self.right_foot_idx is None:
                self.right_foot_idx = 1
            if self.pelvis_idx is None:
                self.pelvis_idx = 0

            # Get link masses
            self._compute_link_masses()

            self._body_indices_initialized = True
            print(f"[CoMTracker] Initialized - Left foot: {self.left_foot_idx}, Right foot: {self.right_foot_idx}")

        except Exception as e:
            print(f"[CoMTracker] Warning: Could not initialize body indices: {e}")
            # Use defaults
            self.left_foot_idx = 0
            self.right_foot_idx = 1
            self.pelvis_idx = 0
            self._body_indices_initialized = True

    def _compute_link_masses(self):
        """Compute link masses from physics simulation."""
        try:
            # Get masses from PhysX view
            masses = self.robot.root_physx_view.get_masses()
            self.link_masses = masses[0]  # First env, all bodies
            self.total_mass = self.link_masses.sum().item()
            print(f"[CoMTracker] Total robot mass: {self.total_mass:.2f} kg")
        except Exception as e:
            print(f"[CoMTracker] Could not get masses, using estimate: {e}")
            # G1 is approximately 50kg
            self.total_mass = 50.0
            num_bodies = self.robot.data.body_pos_w.shape[1]
            self.link_masses = torch.ones(num_bodies, device=self.device) * (self.total_mass / num_bodies)

    def compute_com_position(self) -> torch.Tensor:
        """
        Compute Center of Mass position for all environments.

        Returns:
            com_pos: [num_envs, 3] CoM position in world frame
        """
        self.initialize()

        # Get all body positions: [num_envs, num_bodies, 3]
        body_pos = self.robot.data.body_pos_w

        # Weighted sum
        com_pos = torch.zeros(self.num_envs, 3, device=self.device)

        if self.link_masses is not None and len(self.link_masses) == body_pos.shape[1]:
            for i in range(body_pos.shape[1]):
                com_pos += self.link_masses[i] * body_pos[:, i, :]
            com_pos /= self.total_mass
        else:
            # Simple average if masses not available
            com_pos = body_pos.mean(dim=1)

        return com_pos

    def compute_com_velocity(self) -> torch.Tensor:
        """
        Compute Center of Mass velocity.

        Returns:
            com_vel: [num_envs, 3] CoM velocity in world frame
        """
        self.initialize()

        # Get all body velocities: [num_envs, num_bodies, 3]
        body_vel = self.robot.data.body_lin_vel_w

        com_vel = torch.zeros(self.num_envs, 3, device=self.device)

        if self.link_masses is not None and len(self.link_masses) == body_vel.shape[1]:
            for i in range(body_vel.shape[1]):
                com_vel += self.link_masses[i] * body_vel[:, i, :]
            com_vel /= self.total_mass
        else:
            com_vel = body_vel.mean(dim=1)

        return com_vel

    def get_support_polygon_center(self) -> torch.Tensor:
        """
        Get the center of support polygon (midpoint between feet).

        Returns:
            center: [num_envs, 2] XY position of support center
        """
        self.initialize()

        body_pos = self.robot.data.body_pos_w

        left_foot_pos = body_pos[:, self.left_foot_idx, :2]
        right_foot_pos = body_pos[:, self.right_foot_idx, :2]

        center = (left_foot_pos + right_foot_pos) / 2.0
        return center

    def compute_com_margin(self) -> torch.Tensor:
        """
        Compute how far CoM is from edge of support polygon.
        Positive = stable, Negative = unstable

        Returns:
            margin: [num_envs] stability margin in meters
        """
        com_pos = self.compute_com_position()
        com_xy = com_pos[:, :2]

        support_center = self.get_support_polygon_center()

        # Distance from CoM to support center
        distance = torch.norm(com_xy - support_center, dim=-1)

        # Margin = how much room before falling
        margin = self.max_safe_distance - distance

        return margin

    def compute_rewards(self) -> dict:
        """
        Compute all CoM-related rewards.

        Returns:
            rewards: Dictionary of reward components
        """
        com_pos = self.compute_com_position()
        com_xy = com_pos[:, :2]
        com_height = com_pos[:, 2]

        com_vel = self.compute_com_velocity()
        com_vel_xy = com_vel[:, :2]

        support_center = self.get_support_polygon_center()
        margin = self.compute_com_margin()

        rewards = {}

        # 1. CoM Position Reward (stay centered over feet)
        com_error = torch.norm(com_xy - support_center, dim=-1)
        rewards["com_position"] = torch.exp(-5.0 * com_error ** 2)

        # 2. CoM Velocity Reward (don't move CoM too fast)
        com_speed = torch.norm(com_vel_xy, dim=-1)
        rewards["com_velocity"] = torch.exp(-2.0 * com_speed ** 2)

        # 3. CoM Margin Reward (stay safely within support)
        normalized_margin = torch.clamp(margin / self.max_safe_distance, 0.0, 1.0)
        rewards["com_margin"] = normalized_margin

        # 4. CoM Height Reward (maintain standing height)
        height_error = torch.abs(com_height - self.target_com_height)
        rewards["com_height"] = torch.exp(-10.0 * height_error ** 2)

        # 5. Combined stability score
        rewards["com_stability"] = (
                0.3 * rewards["com_position"] +
                0.2 * rewards["com_velocity"] +
                0.3 * rewards["com_margin"] +
                0.2 * rewards["com_height"]
        )

        return rewards

    def get_observation(self) -> torch.Tensor:
        """
        Get CoM state as observation for policy.

        Returns:
            obs: [num_envs, 6] - CoM position (3) + CoM velocity (3) in local frame
        """
        com_pos = self.compute_com_position()
        com_vel = self.compute_com_velocity()

        # Get robot base position for relative coordinates
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w

        # Relative CoM position
        rel_com_pos = com_pos - base_pos

        # Could transform to local frame here if needed
        # For now, return world-frame relative position

        obs = torch.cat([rel_com_pos, com_vel], dim=-1)
        return obs

    def is_unstable(self) -> torch.Tensor:
        """
        Check if robot is in unstable state (for termination).

        Returns:
            unstable: [num_envs] boolean tensor
        """
        margin = self.compute_com_margin()
        com_height = self.compute_com_position()[:, 2]

        # Unstable conditions
        margin_unstable = margin < -0.05  # CoM significantly outside support
        height_unstable = com_height < 0.3  # Too low (falling)

        return margin_unstable | height_unstable


class SimplifiedCoMTracker:
    """
    Simplified CoM tracker that uses base position as CoM proxy.
    Faster computation, good for initial training.
    """

    def __init__(self, env: DirectRLEnv, device: str = "cuda"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]

        self.target_height = 0.75
        self.max_safe_radius = 0.15

    def compute_rewards(self) -> dict:
        """Simplified CoM rewards using base position."""
        base_pos = self.robot.data.root_pos_w
        base_vel = self.robot.data.root_lin_vel_w

        rewards = {}

        # Height tracking
        height_error = torch.abs(base_pos[:, 2] - self.target_height)
        rewards["com_height"] = torch.exp(-10.0 * height_error ** 2)

        # XY velocity (should be low for standing)
        xy_speed = torch.norm(base_vel[:, :2], dim=-1)
        rewards["com_velocity"] = torch.exp(-2.0 * xy_speed ** 2)

        # Combined
        rewards["com_stability"] = 0.5 * rewards["com_height"] + 0.5 * rewards["com_velocity"]

        return rewards