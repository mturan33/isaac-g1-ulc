"""
Height Map Utilities - Depth Camera to Height Map Conversion
=============================================================

Bu modül depth kamera görüntüsünü robot-centric height map'e dönüştürür.
Isaac Lab'daki Replicator depth annotator çıktısını alır.

Performans Hedefi:
- Conversion time: < 1 ms per frame
- Output resolution: 32x32 veya 64x64 (configurable)

Kullanım:
    extractor = HeightMapExtractor(
        output_size=(64, 64),
        x_range=(-2.0, 2.0),  # meters
        y_range=(-2.0, 2.0),
    )

    height_map = extractor.depth_to_heightmap(depth_image, camera_intrinsics)
    local_map = extractor.get_local_heightmap(height_map, robot_pose)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float = 320.0  # Focal length x
    fy: float = 320.0  # Focal length y
    cx: float = 160.0  # Principal point x
    cy: float = 120.0  # Principal point y
    width: int = 320
    height: int = 240

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CameraIntrinsics":
        return cls(
            fx=d.get("fx", 320.0),
            fy=d.get("fy", 320.0),
            cx=d.get("cx", 160.0),
            cy=d.get("cy", 120.0),
            width=int(d.get("width", 320)),
            height=int(d.get("height", 240)),
        )

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_deg: float) -> "CameraIntrinsics":
        """FOV'dan intrinsics hesapla"""
        fov_rad = np.deg2rad(horizontal_fov_deg)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assuming square pixels
        return cls(
            fx=fx, fy=fy,
            cx=width / 2, cy=height / 2,
            width=width, height=height
        )


class HeightMapExtractor:
    """
    Depth görüntüsünden height map extraction.

    Isaac Lab'da depth camera genellikle Z-buffer style depth verir:
    - Her pixel'de o noktanın kameraya uzaklığı (meters)
    - Invalid pixels: 0 veya çok büyük değerler (>100m gibi)

    Height map:
    - Robot etrafındaki 2D grid
    - Her cell'de o noktadaki yükseklik (ground'dan)
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (64, 64),
            x_range: Tuple[float, float] = (-2.0, 2.0),  # Front-back (meters)
            y_range: Tuple[float, float] = (-2.0, 2.0),  # Left-right (meters)
            max_height: float = 1.0,  # Max height to consider (meters)
            min_depth: float = 0.1,  # Min valid depth (meters)
            max_depth: float = 10.0,  # Max valid depth (meters)
            device: str = "cuda",
    ):
        self.output_size = output_size
        self.x_range = x_range
        self.y_range = y_range
        self.max_height = max_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = device

        # Precompute grid
        self.grid_resolution = (
            (x_range[1] - x_range[0]) / output_size[0],
            (y_range[1] - y_range[0]) / output_size[1],
        )

        # Camera height offset (robot base to camera)
        self.camera_height = 0.15  # Default, can be updated

    def set_camera_height(self, height: float):
        """Kamera yüksekliğini ayarla (robot base'den)"""
        self.camera_height = height

    def depth_to_pointcloud(
            self,
            depth_image: np.ndarray,
            intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """
        Depth image'ı 3D point cloud'a çevir (camera frame).

        Args:
            depth_image: H x W depth in meters
            intrinsics: Camera intrinsic parameters

        Returns:
            points: N x 3 point cloud [X, Y, Z] in camera frame
                   X: right, Y: down, Z: forward (OpenCV convention)
        """
        h, w = depth_image.shape[:2]

        # Create pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        # Filter valid depth
        valid = (depth_image > self.min_depth) & (depth_image < self.max_depth)

        # Unproject to 3D (camera frame: X right, Y down, Z forward)
        z = depth_image[valid]
        x = (u[valid] - intrinsics.cx) * z / intrinsics.fx
        y = (v[valid] - intrinsics.cy) * z / intrinsics.fy

        points = np.stack([x, y, z], axis=1)  # N x 3
        return points

    def pointcloud_to_heightmap(
            self,
            points: np.ndarray,
            robot_yaw: float = 0.0,
    ) -> np.ndarray:
        """
        Point cloud'u height map'e çevir.

        Isaac Lab convention:
        - X: forward (robot front)
        - Y: left
        - Z: up

        Camera convention (after rotation):
        - Camera Z -> Robot X (forward)
        - Camera -X -> Robot Y (left)
        - Camera -Y -> Robot Z (up)

        Args:
            points: N x 3 in camera frame [X_cam, Y_cam, Z_cam]
            robot_yaw: Robot heading in world frame (radians)

        Returns:
            height_map: output_size height map
        """
        if len(points) == 0:
            return np.zeros(self.output_size, dtype=np.float32)

        # Camera to robot frame transformation
        # Assuming camera mounted: looking forward, slightly down
        # Camera: X-right, Y-down, Z-forward
        # Robot: X-forward, Y-left, Z-up

        x_robot = points[:, 2]  # Z_cam -> X_robot (forward)
        y_robot = -points[:, 0]  # -X_cam -> Y_robot (left)
        z_robot = -points[:, 1] + self.camera_height  # -Y_cam + offset -> Z_robot (up, ground-relative)

        # Create height map
        height_map = np.zeros(self.output_size, dtype=np.float32)
        count_map = np.zeros(self.output_size, dtype=np.int32)

        # Convert to grid indices
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        # Filter points within range
        valid = (
                (x_robot >= x_min) & (x_robot < x_max) &
                (y_robot >= y_min) & (y_robot < y_max) &
                (z_robot >= -0.5) & (z_robot <= self.max_height)
        )

        x_robot = x_robot[valid]
        y_robot = y_robot[valid]
        z_robot = z_robot[valid]

        if len(x_robot) == 0:
            return height_map

        # Grid indices
        # X: front of robot is center of map (index output_size[0]//2)
        # Y: left of robot is center of map (index output_size[1]//2)
        grid_x = ((x_robot - x_min) / (x_max - x_min) * self.output_size[0]).astype(np.int32)
        grid_y = ((y_robot - y_min) / (y_max - y_min) * self.output_size[1]).astype(np.int32)

        # Clamp indices
        grid_x = np.clip(grid_x, 0, self.output_size[0] - 1)
        grid_y = np.clip(grid_y, 0, self.output_size[1] - 1)

        # Accumulate heights (take max for each cell)
        for i in range(len(x_robot)):
            gx, gy = grid_x[i], grid_y[i]
            if z_robot[i] > height_map[gx, gy]:
                height_map[gx, gy] = z_robot[i]
            count_map[gx, gy] += 1

        return height_map

    def depth_to_heightmap(
            self,
            depth_image: np.ndarray,
            intrinsics: CameraIntrinsics,
            robot_yaw: float = 0.0,
    ) -> np.ndarray:
        """
        Single call: depth image → height map

        Args:
            depth_image: H x W depth in meters
            intrinsics: Camera intrinsic parameters
            robot_yaw: Robot heading (radians)

        Returns:
            height_map: output_size[0] x output_size[1] height map
        """
        points = self.depth_to_pointcloud(depth_image, intrinsics)
        height_map = self.pointcloud_to_heightmap(points, robot_yaw)
        return height_map

    def depth_to_heightmap_torch(
            self,
            depth_tensor: "torch.Tensor",
            intrinsics: CameraIntrinsics,
    ) -> "torch.Tensor":
        """
        GPU-accelerated version (for batched processing).

        Args:
            depth_tensor: B x H x W or H x W depth tensor
            intrinsics: Camera intrinsics

        Returns:
            height_map: B x H_out x W_out or H_out x W_out
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for GPU-accelerated height map extraction")
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, H, W = depth_tensor.shape
        device = depth_tensor.device

        # Create pixel coordinate grids
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        u = u.float()
        v = v.float()

        # Valid depth mask
        valid = (depth_tensor > self.min_depth) & (depth_tensor < self.max_depth)

        # Unproject
        z = depth_tensor  # B x H x W
        x = (u.unsqueeze(0) - intrinsics.cx) * z / intrinsics.fx
        y = (v.unsqueeze(0) - intrinsics.cy) * z / intrinsics.fy

        # Camera to robot frame
        x_robot = z
        y_robot = -x
        z_robot = -y + self.camera_height

        # Initialize output
        output = torch.zeros(B, *self.output_size, device=device)

        # Grid indices
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        grid_x = ((x_robot - x_min) / (x_max - x_min) * self.output_size[0]).long()
        grid_y = ((y_robot - y_min) / (y_max - y_min) * self.output_size[1]).long()

        # Valid range mask
        range_valid = (
                valid &
                (grid_x >= 0) & (grid_x < self.output_size[0]) &
                (grid_y >= 0) & (grid_y < self.output_size[1]) &
                (z_robot >= -0.5) & (z_robot <= self.max_height)
        )

        # Scatter max (simplified - for full batched version use scatter_reduce)
        for b in range(B):
            mask = range_valid[b]
            if mask.sum() == 0:
                continue

            gx = grid_x[b][mask]
            gy = grid_y[b][mask]
            heights = z_robot[b][mask]

            # Linear index
            idx = gx * self.output_size[1] + gy

            # Use scatter_reduce for max aggregation (PyTorch 1.12+)
            flat_output = output[b].view(-1)
            flat_output.scatter_reduce_(
                0, idx, heights, reduce='amax', include_self=True
            )

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def get_local_heightmap(
            self,
            global_heightmap: np.ndarray,
            robot_pose: np.ndarray,
            local_size: Tuple[int, int] = (32, 32),
    ) -> np.ndarray:
        """
        Global height map'ten robot-centered crop al.

        Args:
            global_heightmap: Full height map
            robot_pose: [x, y, yaw]
            local_size: Output size

        Returns:
            local_heightmap: Robot-centered, yaw-aligned height map
        """
        # For robot-mounted camera, the output is already robot-centered
        # This is mainly for world-frame height maps

        # Simple crop from center
        h, w = global_heightmap.shape
        ch, cw = h // 2, w // 2
        lh, lw = local_size

        y1 = max(0, ch - lh // 2)
        y2 = min(h, ch + lh // 2)
        x1 = max(0, cw - lw // 2)
        x2 = min(w, cw + lw // 2)

        local = global_heightmap[y1:y2, x1:x2]

        # Pad if necessary
        if local.shape != local_size:
            padded = np.zeros(local_size, dtype=np.float32)
            ph = min(local.shape[0], local_size[0])
            pw = min(local.shape[1], local_size[1])
            padded[:ph, :pw] = local[:ph, :pw]
            local = padded

        return local

    def visualize_heightmap(
            self,
            height_map: np.ndarray,
            colormap: str = "viridis",
    ) -> np.ndarray:
        """
        Height map'i görselleştir (debug için).

        Returns:
            RGB image (H x W x 3)
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Normalize
        h_min, h_max = height_map.min(), height_map.max()
        if h_max - h_min > 0.001:
            normalized = (height_map - h_min) / (h_max - h_min)
        else:
            normalized = np.zeros_like(height_map)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(normalized)[:, :, :3]  # RGB only

        return (colored * 255).astype(np.uint8)


class ObjectHeightTracker:
    """
    Height map'ten obje pozisyonlarını track et.

    SemanticMap ile birlikte kullanılır:
    1. SemanticMap objelerin beklenen pozisyonlarını tutar
    2. Bu class height map'te anomalileri bulur
    3. Obje pozisyonlarını günceller
    """

    def __init__(
            self,
            expected_object_height: float = 0.15,  # Typical object height
            height_tolerance: float = 0.1,
            search_radius_cells: int = 5,
    ):
        self.expected_object_height = expected_object_height
        self.height_tolerance = height_tolerance
        self.search_radius = search_radius_cells

    def find_object_in_heightmap(
            self,
            height_map: np.ndarray,
            expected_position: Tuple[int, int],
            expected_size: float = 0.2,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Height map'te beklenen pozisyon etrafında obje ara.

        Args:
            height_map: H x W height map
            expected_position: (grid_x, grid_y) beklenen pozisyon
            expected_size: Obje boyutu (meters)

        Returns:
            (grid_x, grid_y, confidence) or None
        """
        h, w = height_map.shape
        ex, ey = expected_position
        r = self.search_radius

        # Search window
        x1 = max(0, ex - r)
        x2 = min(h, ex + r + 1)
        y1 = max(0, ey - r)
        y2 = min(w, ey + r + 1)

        roi = height_map[x1:x2, y1:y2]

        if roi.size == 0:
            return None

        # Find cells with expected height
        height_match = np.abs(roi - self.expected_object_height) < self.height_tolerance

        if not np.any(height_match):
            return None

        # Find centroid of matching cells
        y_idx, x_idx = np.where(height_match)
        if len(x_idx) == 0:
            return None

        cx = int(np.mean(x_idx)) + x1
        cy = int(np.mean(y_idx)) + y1

        # Confidence based on number of matching cells and distance from expected
        num_cells = len(x_idx)
        dist = np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
        confidence = min(1.0, num_cells / 10) * max(0, 1 - dist / r)

        return (cx, cy, confidence)

    def track_objects(
            self,
            height_map: np.ndarray,
            objects: Dict[int, Tuple[int, int, float]],  # id -> (grid_x, grid_y, expected_height)
    ) -> Dict[int, Tuple[int, int, float]]:
        """
        Birden fazla objeyi track et.

        Args:
            height_map: Current height map
            objects: Dictionary of object id -> expected position

        Returns:
            Updated positions with confidence
        """
        results = {}

        for obj_id, (gx, gy, exp_height) in objects.items():
            self.expected_object_height = exp_height
            result = self.find_object_in_heightmap(height_map, (gx, gy))
            if result is not None:
                results[obj_id] = result

        return results


# ========== Test / Demo ==========
if __name__ == "__main__":
    print("HeightMapExtractor Test")
    print("=" * 50)

    # Create extractor
    extractor = HeightMapExtractor(
        output_size=(64, 64),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
    )

    # Create fake depth image
    h, w = 240, 320
    intrinsics = CameraIntrinsics.from_fov(w, h, horizontal_fov_deg=70)
    print(f"Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    # Simulate depth image with a box in front
    depth_image = np.full((h, w), 5.0, dtype=np.float32)  # 5m background

    # Add a box at 2m distance, center of image
    box_y1, box_y2 = 100, 140
    box_x1, box_x2 = 140, 180
    depth_image[box_y1:box_y2, box_x1:box_x2] = 2.0  # 2m distance

    # Convert to height map
    import time

    N = 100
    t0 = time.time()
    for _ in range(N):
        height_map = extractor.depth_to_heightmap(depth_image, intrinsics)
    dt = (time.time() - t0) * 1000 / N

    print(f"\nHeight map shape: {height_map.shape}")
    print(f"Height range: [{height_map.min():.3f}, {height_map.max():.3f}]")
    print(f"Non-zero cells: {(height_map > 0).sum()}")
    print(f"Conversion time: {dt:.3f} ms (target: < 1ms)")

    # Test PyTorch version
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\n--- GPU Test ---")
        depth_tensor = torch.tensor(depth_image, device="cuda")

        N = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            height_map_gpu = extractor.depth_to_heightmap_torch(depth_tensor, intrinsics)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000 / N

        print(f"GPU conversion time: {dt:.3f} ms")

    # Visualize
    try:
        vis = extractor.visualize_heightmap(height_map)
        import cv2

        cv2.imwrite("/tmp/heightmap_test.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("\nVisualization saved to /tmp/heightmap_test.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")