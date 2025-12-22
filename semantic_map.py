"""
Semantic Map - VLM-based object tracking with real-time updates
================================================================

Bu modül VLM'i her frame çağırmak yerine:
1. Başlangıçta VLM ile scene'i analiz et
2. Semantic map oluştur (object id, label, color, world pos)
3. Runtime'da sadece depth/height map ile pozisyonları güncelle

Performans Hedefi:
- Initialization: < 1 saniye
- Per-frame update: < 2 ms
- Object tracking accuracy: > 90%

Kullanım:
    semantic_map = SemanticMap()

    # Başlangıçta VLM ile initialize (bir kere)
    semantic_map.initialize_from_vlm(vlm_output, depth_image, camera_intrinsics)

    # Her frame'de güncelle (VLM'siz, ~1-2ms)
    semantic_map.update_from_depth(height_map, robot_pose)

    # RL query
    rel_pos = semantic_map.get_relative_position(object_id, robot_pose)
    obj_id = semantic_map.get_object_by_description("mavi kutu")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ObjectType(Enum):
    """Desteklenen obje tipleri"""
    BOX = "box"
    BALL = "ball"
    CONE = "cone"
    CYLINDER = "cylinder"
    UNKNOWN = "unknown"


class ObjectColor(Enum):
    """Desteklenen renkler (Türkçe ve İngilizce)"""
    BLUE = "blue"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


# Türkçe-İngilizce mapping
COLOR_MAP_TR = {
    "mavi": ObjectColor.BLUE,
    "kırmızı": ObjectColor.RED,
    "yeşil": ObjectColor.GREEN,
    "sarı": ObjectColor.YELLOW,
    "turuncu": ObjectColor.ORANGE,
    "beyaz": ObjectColor.WHITE,
    "siyah": ObjectColor.BLACK,
}

OBJECT_MAP_TR = {
    "kutu": ObjectType.BOX,
    "top": ObjectType.BALL,
    "koni": ObjectType.CONE,
    "silindir": ObjectType.CYLINDER,
}


@dataclass
class TrackedObject:
    """Takip edilen obje bilgisi"""
    id: int
    label: ObjectType
    color: ObjectColor

    # World frame pozisyon (x, y, z)
    world_position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Bounding box (pixel coordinates from last VLM detection)
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2

    # Object properties
    estimated_size: float = 0.2  # meters (diameter/width)

    # Tracking state
    confidence: float = 1.0
    last_seen_frame: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_dynamic: bool = False

    # Position history for velocity estimation
    position_history: List[np.ndarray] = field(default_factory=list)
    history_max_len: int = 10

    def update_position(self, new_pos: np.ndarray, frame: int):
        """Pozisyonu güncelle ve velocity tahmin et"""
        old_pos = self.world_position.copy()
        self.world_position = new_pos
        self.last_seen_frame = frame

        # History'ye ekle
        self.position_history.append(new_pos.copy())
        if len(self.position_history) > self.history_max_len:
            self.position_history.pop(0)

        # Velocity estimation (simple moving average)
        if len(self.position_history) >= 2:
            velocities = []
            for i in range(1, len(self.position_history)):
                v = self.position_history[i] - self.position_history[i - 1]
                velocities.append(v)
            self.velocity = np.mean(velocities, axis=0)

            # Check if dynamic (moving faster than threshold)
            speed = np.linalg.norm(self.velocity)
            self.is_dynamic = speed > 0.05  # 5 cm/frame threshold

    def predict_position(self, frames_ahead: int = 1) -> np.ndarray:
        """Gelecek pozisyonu tahmin et (dynamic objects için)"""
        if not self.is_dynamic:
            return self.world_position
        return self.world_position + self.velocity * frames_ahead


class SemanticMap:
    """
    Semantic map for real-time object tracking.

    VLM ile başlangıçta initialize edilir, sonra depth/height map ile
    güncellenir. Bu sayede VLM'i her frame çağırmaya gerek kalmaz.
    """

    def __init__(
            self,
            device: str = "cuda",
            max_objects: int = 20,
            tracking_radius: float = 0.5,  # meters - search radius for object tracking
            confidence_decay: float = 0.95,  # per-frame confidence decay
            min_confidence: float = 0.1,
    ):
        self.device = device
        self.max_objects = max_objects
        self.tracking_radius = tracking_radius
        self.confidence_decay = confidence_decay
        self.min_confidence = min_confidence

        # Object storage
        self.objects: Dict[int, TrackedObject] = {}
        self.next_id = 0

        # Frame counter
        self.current_frame = 0

        # Initialization state
        self.initialized = False
        self.initialization_time_ms = 0.0

        # Performance tracking
        self.last_update_time_ms = 0.0

    def initialize_from_vlm(
            self,
            vlm_detections: List[Dict[str, Any]],
            depth_image: np.ndarray,
            camera_intrinsics: Dict[str, float],
            robot_pose: np.ndarray,  # [x, y, z, qw, qx, qy, qz] or [x, y, yaw]
    ) -> bool:
        """
        VLM detection sonuçlarından semantic map oluştur.

        Args:
            vlm_detections: List of {
                "label": str,      # "box", "ball", etc.
                "color": str,      # "blue", "red", etc.
                "bbox": [x1, y1, x2, y2],  # pixel coordinates
                "confidence": float  # optional
            }
            depth_image: H x W depth image (meters)
            camera_intrinsics: {"fx", "fy", "cx", "cy"}
            robot_pose: Robot world pose

        Returns:
            success: bool
        """
        t0 = time.time()

        self.objects.clear()
        self.next_id = 0

        for detection in vlm_detections:
            if self.next_id >= self.max_objects:
                print(f"[SemanticMap] Warning: Max objects ({self.max_objects}) reached")
                break

            # Parse detection
            label_str = detection.get("label", "unknown").lower()
            color_str = detection.get("color", "unknown").lower()
            bbox = detection.get("bbox", None)
            conf = detection.get("confidence", 1.0)

            if bbox is None:
                continue

            # Convert strings to enums
            label = self._parse_label(label_str)
            color = self._parse_color(color_str)

            # Calculate world position from bbox + depth
            world_pos = self._bbox_to_world_position(
                bbox, depth_image, camera_intrinsics, robot_pose
            )

            if world_pos is None:
                continue

            # Create tracked object
            obj = TrackedObject(
                id=self.next_id,
                label=label,
                color=color,
                world_position=world_pos,
                bbox=tuple(bbox),
                confidence=conf,
                last_seen_frame=self.current_frame,
            )

            self.objects[self.next_id] = obj
            self.next_id += 1

        self.initialized = len(self.objects) > 0
        self.initialization_time_ms = (time.time() - t0) * 1000

        print(f"[SemanticMap] Initialized with {len(self.objects)} objects in {self.initialization_time_ms:.1f}ms")
        for obj_id, obj in self.objects.items():
            print(f"  - [{obj_id}] {obj.color.value} {obj.label.value} at {obj.world_position}")

        return self.initialized

    def initialize_from_ground_truth(
            self,
            objects_info: List[Dict[str, Any]],
    ) -> bool:
        """
        Ground truth bilgisinden initialize et (test/debug için).

        Args:
            objects_info: List of {
                "label": str,
                "color": str,
                "position": [x, y, z],
                "size": float (optional)
            }
        """
        t0 = time.time()

        self.objects.clear()
        self.next_id = 0

        for info in objects_info:
            if self.next_id >= self.max_objects:
                break

            label = self._parse_label(info.get("label", "unknown"))
            color = self._parse_color(info.get("color", "unknown"))
            pos = np.array(info.get("position", [0, 0, 0]), dtype=np.float32)
            size = info.get("size", 0.2)

            obj = TrackedObject(
                id=self.next_id,
                label=label,
                color=color,
                world_position=pos,
                estimated_size=size,
                confidence=1.0,
                last_seen_frame=self.current_frame,
            )

            self.objects[self.next_id] = obj
            self.next_id += 1

        self.initialized = len(self.objects) > 0
        self.initialization_time_ms = (time.time() - t0) * 1000

        print(f"[SemanticMap] Ground truth init: {len(self.objects)} objects")
        return self.initialized

    def update_from_depth(
            self,
            height_map: np.ndarray,
            robot_pose: np.ndarray,
            camera_intrinsics: Optional[Dict[str, float]] = None,
    ):
        """
        Her frame'de çağrılır. Height map'ten obje pozisyonlarını günceller.

        Bu fonksiyon VLM ÇAĞIRMAZ - sadece geometric tracking yapar.
        Hedef: < 2ms per call

        Args:
            height_map: H x W height map (meters, robot-centric)
            robot_pose: [x, y, yaw] or [x, y, z, qw, qx, qy, qz]
        """
        t0 = time.time()
        self.current_frame += 1

        # Update each tracked object
        for obj_id, obj in list(self.objects.items()):
            # Skip if confidence too low
            if obj.confidence < self.min_confidence:
                del self.objects[obj_id]
                continue

            # Simple tracking: assume objects don't move much
            # For dynamic objects, use velocity prediction
            if obj.is_dynamic:
                predicted_pos = obj.predict_position(frames_ahead=1)
            else:
                predicted_pos = obj.world_position

            # Decay confidence
            obj.confidence *= self.confidence_decay

            # TODO: Advanced tracking using height map
            # For now, we trust the initialized positions
            # In a full implementation, we would:
            # 1. Convert predicted_pos to height map coordinates
            # 2. Search for height anomalies matching expected object size
            # 3. Update position if found

        self.last_update_time_ms = (time.time() - t0) * 1000

    def update_object_position(
            self,
            object_id: int,
            new_world_position: np.ndarray,
            confidence: float = 1.0,
    ):
        """
        Belirli bir objenin pozisyonunu güncelle (VLM re-detection sonrası).
        """
        if object_id in self.objects:
            self.objects[object_id].update_position(new_world_position, self.current_frame)
            self.objects[object_id].confidence = confidence

    def get_relative_position(
            self,
            object_id: int,
            robot_pose: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Robot frame'e göre obje pozisyonunu döndür.

        Args:
            object_id: Target object ID
            robot_pose: [x, y, yaw] or [x, y, z, qw, qx, qy, qz]

        Returns:
            [rel_x, rel_y, rel_z] robot frame'de (None if not found)
        """
        if object_id not in self.objects:
            return None

        obj = self.objects[object_id]
        world_pos = obj.world_position

        # Parse robot pose
        if len(robot_pose) == 3:
            rx, ry, yaw = robot_pose
            rz = 0.0
        else:
            rx, ry, rz = robot_pose[:3]
            # Quaternion to yaw (simplified)
            qw, qx, qy, qz = robot_pose[3:7]
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

        # World to robot frame transformation
        dx = world_pos[0] - rx
        dy = world_pos[1] - ry
        dz = world_pos[2] - rz if len(world_pos) > 2 else 0.0

        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        rel_x = dx * cos_yaw - dy * sin_yaw
        rel_y = dx * sin_yaw + dy * cos_yaw
        rel_z = dz

        return np.array([rel_x, rel_y, rel_z], dtype=np.float32)

    def get_object_by_description(
            self,
            description: str,
    ) -> Optional[int]:
        """
        Türkçe veya İngilizce description'dan obje ID'si bul.

        Args:
            description: "mavi kutu", "blue box", "kırmızı top", etc.

        Returns:
            object_id (None if not found)
        """
        desc_lower = description.lower()

        # Parse color and type from description
        target_color = None
        target_type = None

        # Check Turkish
        for tr_word, color_enum in COLOR_MAP_TR.items():
            if tr_word in desc_lower:
                target_color = color_enum
                break

        for tr_word, type_enum in OBJECT_MAP_TR.items():
            if tr_word in desc_lower:
                target_type = type_enum
                break

        # Check English
        if target_color is None:
            for color in ObjectColor:
                if color.value in desc_lower:
                    target_color = color
                    break

        if target_type is None:
            for obj_type in ObjectType:
                if obj_type.value in desc_lower:
                    target_type = obj_type
                    break

        # Find matching object
        best_match = None
        best_confidence = 0.0

        for obj_id, obj in self.objects.items():
            match_score = 0.0

            if target_color is not None and obj.color == target_color:
                match_score += 0.5
            if target_type is not None and obj.label == target_type:
                match_score += 0.5

            # Weight by confidence
            match_score *= obj.confidence

            if match_score > best_confidence:
                best_confidence = match_score
                best_match = obj_id

        return best_match if best_confidence > 0 else None

    def get_dynamic_obstacles(
            self,
            robot_pose: np.ndarray,
            max_distance: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """
        Hareketli engelleri döndür (RL observation için).

        Returns:
            List of {
                "id": int,
                "relative_position": [x, y, z],
                "velocity": [vx, vy, vz],
                "is_dynamic": bool
            }
        """
        obstacles = []

        for obj_id, obj in self.objects.items():
            rel_pos = self.get_relative_position(obj_id, robot_pose)
            if rel_pos is None:
                continue

            distance = np.linalg.norm(rel_pos[:2])
            if distance > max_distance:
                continue

            obstacles.append({
                "id": obj_id,
                "relative_position": rel_pos,
                "velocity": obj.velocity.copy(),
                "is_dynamic": obj.is_dynamic,
            })

        return obstacles

    def get_all_objects_info(self) -> List[Dict[str, Any]]:
        """Tüm objelerin bilgisini döndür (debug için)"""
        return [
            {
                "id": obj.id,
                "label": obj.label.value,
                "color": obj.color.value,
                "world_position": obj.world_position.tolist(),
                "confidence": obj.confidence,
                "is_dynamic": obj.is_dynamic,
                "velocity": obj.velocity.tolist(),
            }
            for obj in self.objects.values()
        ]

    # ========== Private Methods ==========

    def _parse_label(self, label_str: str) -> ObjectType:
        """String'den ObjectType'a çevir"""
        label_lower = label_str.lower()

        # Check Turkish first
        for tr_word, obj_type in OBJECT_MAP_TR.items():
            if tr_word in label_lower:
                return obj_type

        # Check English
        for obj_type in ObjectType:
            if obj_type.value in label_lower:
                return obj_type

        return ObjectType.UNKNOWN

    def _parse_color(self, color_str: str) -> ObjectColor:
        """String'den ObjectColor'a çevir"""
        color_lower = color_str.lower()

        # Check Turkish first
        for tr_word, color in COLOR_MAP_TR.items():
            if tr_word in color_lower:
                return color

        # Check English
        for color in ObjectColor:
            if color.value in color_lower:
                return color

        return ObjectColor.UNKNOWN

    def _bbox_to_world_position(
            self,
            bbox: List[float],
            depth_image: np.ndarray,
            camera_intrinsics: Dict[str, float],
            robot_pose: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Bounding box + depth'ten world position hesapla.

        Args:
            bbox: [x1, y1, x2, y2] pixel coordinates
            depth_image: H x W depth in meters
            camera_intrinsics: {fx, fy, cx, cy}
            robot_pose: Robot world pose

        Returns:
            [x, y, z] world position
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Get bbox center
            cx_pixel = (x1 + x2) // 2
            cy_pixel = (y1 + y2) // 2

            # Clamp to image bounds
            h, w = depth_image.shape[:2]
            cx_pixel = max(0, min(w - 1, cx_pixel))
            cy_pixel = max(0, min(h - 1, cy_pixel))

            # Get depth at center (or median in bbox region for robustness)
            roi = depth_image[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            # Use median depth (more robust to noise)
            valid_depths = roi[roi > 0.1]  # Filter out invalid depths
            if len(valid_depths) == 0:
                return None

            depth = np.median(valid_depths)

            # Camera intrinsics
            fx = camera_intrinsics.get("fx", 320.0)
            fy = camera_intrinsics.get("fy", 320.0)
            cx = camera_intrinsics.get("cx", w / 2)
            cy = camera_intrinsics.get("cy", h / 2)

            # Pixel to camera frame (X forward, Y left, Z up convention)
            x_cam = depth  # Forward
            y_cam = -(cx_pixel - cx) * depth / fx  # Left
            z_cam = -(cy_pixel - cy) * depth / fy  # Up

            # Camera to robot frame (assuming camera at robot base)
            # Simple case: camera aligned with robot
            x_robot = x_cam
            y_robot = y_cam
            z_robot = z_cam

            # Robot to world frame
            if len(robot_pose) == 3:
                rx, ry, yaw = robot_pose
                rz = 0.0
            else:
                rx, ry, rz = robot_pose[:3]
                qw, qx, qy, qz = robot_pose[3:7]
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            x_world = rx + x_robot * cos_yaw - y_robot * sin_yaw
            y_world = ry + x_robot * sin_yaw + y_robot * cos_yaw
            z_world = rz + z_robot

            return np.array([x_world, y_world, z_world], dtype=np.float32)

        except Exception as e:
            print(f"[SemanticMap] bbox_to_world error: {e}")
            return None


# ========== Test / Demo ==========
if __name__ == "__main__":
    print("SemanticMap Test")
    print("=" * 50)

    # Create semantic map
    sm = SemanticMap()

    # Initialize from ground truth (simulation test)
    objects_info = [
        {"label": "box", "color": "blue", "position": [2.0, 1.0, 0.2]},
        {"label": "ball", "color": "red", "position": [3.0, -1.5, 0.15]},
        {"label": "cone", "color": "yellow", "position": [1.5, 2.0, 0.3]},
    ]

    success = sm.initialize_from_ground_truth(objects_info)
    print(f"\nInitialization: {'SUCCESS' if success else 'FAILED'}")

    # Test queries
    robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, yaw

    print("\n--- Query Tests ---")

    # Turkish query
    obj_id = sm.get_object_by_description("mavi kutu")
    print(f"'mavi kutu' -> ID: {obj_id}")
    if obj_id is not None:
        rel_pos = sm.get_relative_position(obj_id, robot_pose)
        print(f"  Relative position: {rel_pos}")

    # English query
    obj_id = sm.get_object_by_description("red ball")
    print(f"'red ball' -> ID: {obj_id}")
    if obj_id is not None:
        rel_pos = sm.get_relative_position(obj_id, robot_pose)
        print(f"  Relative position: {rel_pos}")

    # Performance test
    print("\n--- Performance Test ---")
    import time

    # Simulate height map
    height_map = np.zeros((64, 64), dtype=np.float32)

    N = 1000
    t0 = time.time()
    for _ in range(N):
        sm.update_from_depth(height_map, robot_pose)
    dt = (time.time() - t0) * 1000 / N
    print(f"Average update time: {dt:.3f} ms (target: < 2ms)")

    print("\n--- All Objects ---")
    for info in sm.get_all_objects_info():
        print(f"  {info}")