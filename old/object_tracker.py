"""
Object Tracker - VLM-free Object Tracking using Height Maps
============================================================

SemanticMap'in VLM ile initialize ettiği objeleri,
sadece height map kullanarak frame-by-frame track eder.

Key Insight:
- VLM: "Bu sahne'de neler var?" sorusuna cevap (yavaş, ~200ms)
- Object Tracker: "Bu obje nereye gitti?" sorusuna cevap (hızlı, <1ms)

Kullanım:
    tracker = ObjectTracker()

    # Her frame'de
    updated_positions = tracker.track_objects(
        height_map=height_map,
        previous_positions=semantic_map.get_object_positions(),
        object_sizes=semantic_map.get_object_sizes(),
    )

    semantic_map.batch_update_positions(updated_positions)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class TrackingResult:
    """Tracking sonucu"""
    object_id: int
    found: bool
    grid_position: Tuple[int, int]  # (x, y) in height map
    world_position: np.ndarray  # (x, y, z) in world frame
    confidence: float
    velocity: np.ndarray  # (vx, vy, vz)
    is_occluded: bool


class ObjectTracker:
    """
    Height map tabanlı object tracker.

    Çalışma prensibi:
    1. Beklenen pozisyon etrafında search window
    2. Height anomaly detection (zemin seviyesinden farklı)
    3. Centroid calculation
    4. Velocity estimation

    Avantajlar:
    - VLM'den ~100x hızlı
    - GPU-friendly (batched operations)
    - Occlusion handling
    """

    def __init__(
            self,
            grid_resolution: float = 0.0625,  # 4m / 64 cells
            search_radius: int = 5,  # Grid cells
            min_object_height: float = 0.05,  # Min height to detect
            max_object_height: float = 0.8,  # Max height (filter out walls)
            velocity_smoothing: float = 0.3,  # EMA alpha for velocity
            occlusion_threshold: int = 5,  # Frames before marked as occluded
    ):
        self.grid_resolution = grid_resolution
        self.search_radius = search_radius
        self.min_object_height = min_object_height
        self.max_object_height = max_object_height
        self.velocity_smoothing = velocity_smoothing
        self.occlusion_threshold = occlusion_threshold

        # Per-object tracking state
        self.object_states: Dict[int, Dict] = {}

        # Performance stats
        self.last_track_time_ms = 0.0

    def track_object(
            self,
            height_map: np.ndarray,
            object_id: int,
            expected_grid_pos: Tuple[int, int],
            expected_height: float = 0.2,
            expected_size: float = 0.2,  # meters
    ) -> TrackingResult:
        """
        Tek bir objeyi track et.

        Args:
            height_map: H x W height map (meters)
            object_id: Tracking ID
            expected_grid_pos: (gx, gy) beklenen grid pozisyonu
            expected_height: Beklenen obje yüksekliği
            expected_size: Beklenen obje boyutu (diameter)

        Returns:
            TrackingResult
        """
        h, w = height_map.shape
        ex, ey = expected_grid_pos
        r = self.search_radius

        # Initialize state if new object
        if object_id not in self.object_states:
            self.object_states[object_id] = {
                "velocity": np.zeros(3),
                "last_position": None,
                "frames_since_seen": 0,
            }

        state = self.object_states[object_id]

        # Define search window
        x1 = max(0, ex - r)
        x2 = min(h, ex + r + 1)
        y1 = max(0, ey - r)
        y2 = min(w, ey + r + 1)

        roi = height_map[x1:x2, y1:y2]

        if roi.size == 0:
            state["frames_since_seen"] += 1
            return TrackingResult(
                object_id=object_id,
                found=False,
                grid_position=expected_grid_pos,
                world_position=self._grid_to_world(expected_grid_pos, expected_height),
                confidence=0.0,
                velocity=state["velocity"],
                is_occluded=state["frames_since_seen"] > self.occlusion_threshold,
            )

        # Height anomaly detection
        # Look for cells with height in expected range
        height_mask = (
                (roi > self.min_object_height) &
                (roi < self.max_object_height) &
                (np.abs(roi - expected_height) < expected_height * 0.5)  # Within 50% of expected
        )

        if not np.any(height_mask):
            # Object not found - might be occluded or moved away
            state["frames_since_seen"] += 1

            # Use velocity prediction
            if state["last_position"] is not None:
                predicted_pos = state["last_position"] + state["velocity"]
                predicted_grid = self._world_to_grid(predicted_pos)
            else:
                predicted_grid = expected_grid_pos
                predicted_pos = self._grid_to_world(expected_grid_pos, expected_height)

            return TrackingResult(
                object_id=object_id,
                found=False,
                grid_position=predicted_grid,
                world_position=predicted_pos,
                confidence=max(0, 1 - state["frames_since_seen"] * 0.1),
                velocity=state["velocity"],
                is_occluded=state["frames_since_seen"] > self.occlusion_threshold,
            )

        # Find centroid of detected region
        local_y, local_x = np.where(height_mask)

        # Weighted centroid by height (higher = more likely object top)
        weights = roi[height_mask]
        if weights.sum() > 0:
            cx_local = np.average(local_x, weights=weights)
            cy_local = np.average(local_y, weights=weights)
        else:
            cx_local = np.mean(local_x)
            cy_local = np.mean(local_y)

        # Convert to global grid coordinates
        cx = int(cx_local + x1)
        cy = int(cy_local + y1)

        # Estimate object height from max in detection region
        detected_height = np.max(roi[height_mask])

        # Convert to world position
        world_pos = self._grid_to_world((cx, cy), detected_height)

        # Velocity estimation
        if state["last_position"] is not None:
            instant_velocity = world_pos - state["last_position"]
            # EMA smoothing
            state["velocity"] = (
                    self.velocity_smoothing * instant_velocity +
                    (1 - self.velocity_smoothing) * state["velocity"]
            )

        # Update state
        state["last_position"] = world_pos.copy()
        state["frames_since_seen"] = 0

        # Confidence based on detection quality
        num_cells = np.sum(height_mask)
        expected_cells = (expected_size / self.grid_resolution) ** 2
        cell_ratio = min(1, num_cells / max(1, expected_cells))
        distance_penalty = min(r, np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)) / r
        confidence = cell_ratio * (1 - distance_penalty * 0.5)

        return TrackingResult(
            object_id=object_id,
            found=True,
            grid_position=(cx, cy),
            world_position=world_pos,
            confidence=confidence,
            velocity=state["velocity"],
            is_occluded=False,
        )

    def track_objects(
            self,
            height_map: np.ndarray,
            objects: Dict[int, Dict],  # id -> {"grid_pos": (x,y), "height": h, "size": s}
    ) -> Dict[int, TrackingResult]:
        """
        Birden fazla objeyi batch halinde track et.

        Args:
            height_map: Height map
            objects: Object bilgileri

        Returns:
            Dictionary of tracking results
        """
        t0 = time.time()
        results = {}

        for obj_id, obj_info in objects.items():
            result = self.track_object(
                height_map=height_map,
                object_id=obj_id,
                expected_grid_pos=obj_info.get("grid_pos", (32, 32)),
                expected_height=obj_info.get("height", 0.2),
                expected_size=obj_info.get("size", 0.2),
            )
            results[obj_id] = result

        self.last_track_time_ms = (time.time() - t0) * 1000
        return results

    def estimate_velocity(
            self,
            position_history: List[np.ndarray],
            method: str = "linear",
    ) -> np.ndarray:
        """
        Pozisyon history'den velocity tahmin et.

        Args:
            position_history: List of positions (newest last)
            method: "simple" (son 2 frame), "linear" (linear fit), "ema" (exponential)

        Returns:
            velocity: [vx, vy, vz]
        """
        if len(position_history) < 2:
            return np.zeros(3)

        if method == "simple":
            return position_history[-1] - position_history[-2]

        elif method == "ema":
            velocities = []
            for i in range(1, len(position_history)):
                v = position_history[i] - position_history[i - 1]
                velocities.append(v)

            # Exponential weighted average (newer = higher weight)
            alpha = 0.5
            result = np.zeros(3)
            weight_sum = 0
            for i, v in enumerate(velocities):
                weight = alpha ** (len(velocities) - 1 - i)
                result += weight * v
                weight_sum += weight
            return result / weight_sum if weight_sum > 0 else np.zeros(3)

        elif method == "linear":
            # Linear regression fit
            positions = np.array(position_history)
            t = np.arange(len(positions))

            # Fit line to each dimension
            velocity = np.zeros(3)
            for dim in range(3):
                if np.std(positions[:, dim]) > 1e-6:
                    coeffs = np.polyfit(t, positions[:, dim], 1)
                    velocity[dim] = coeffs[0]  # Slope
            return velocity

        return np.zeros(3)

    def is_dynamic(
            self,
            velocity: np.ndarray,
            speed_threshold: float = 0.05,  # m/frame
    ) -> bool:
        """Obje hareket ediyor mu?"""
        speed = np.linalg.norm(velocity[:2])  # XY speed only
        return speed > speed_threshold

    def predict_position(
            self,
            current_position: np.ndarray,
            velocity: np.ndarray,
            frames_ahead: int = 1,
    ) -> np.ndarray:
        """Gelecek pozisyonu tahmin et"""
        return current_position + velocity * frames_ahead

    def _grid_to_world(
            self,
            grid_pos: Tuple[int, int],
            height: float,
            x_range: Tuple[float, float] = (-2.0, 2.0),
            y_range: Tuple[float, float] = (-2.0, 2.0),
            grid_size: Tuple[int, int] = (64, 64),
    ) -> np.ndarray:
        """Grid koordinatlarını world frame'e çevir"""
        gx, gy = grid_pos

        # Grid [0, size] -> World [min, max]
        x = x_range[0] + (gx / grid_size[0]) * (x_range[1] - x_range[0])
        y = y_range[0] + (gy / grid_size[1]) * (y_range[1] - y_range[0])
        z = height

        return np.array([x, y, z], dtype=np.float32)

    def _world_to_grid(
            self,
            world_pos: np.ndarray,
            x_range: Tuple[float, float] = (-2.0, 2.0),
            y_range: Tuple[float, float] = (-2.0, 2.0),
            grid_size: Tuple[int, int] = (64, 64),
    ) -> Tuple[int, int]:
        """World koordinatlarını grid'e çevir"""
        x, y = world_pos[:2]

        gx = int((x - x_range[0]) / (x_range[1] - x_range[0]) * grid_size[0])
        gy = int((y - y_range[0]) / (y_range[1] - y_range[0]) * grid_size[1])

        gx = max(0, min(grid_size[0] - 1, gx))
        gy = max(0, min(grid_size[1] - 1, gy))

        return (gx, gy)

    def reset_object(self, object_id: int):
        """Obje tracking state'ini sıfırla"""
        if object_id in self.object_states:
            del self.object_states[object_id]

    def reset_all(self):
        """Tüm tracking state'lerini sıfırla"""
        self.object_states.clear()


class DynamicObstacleTracker:
    """
    Hareketli engelleri track et.

    Static objeler için SemanticMap yeterli,
    ama dynamic objeler için collision avoidance gerekli.
    """

    def __init__(
            self,
            max_obstacles: int = 10,
            danger_radius: float = 0.5,  # meters
            prediction_horizon: int = 10,  # frames
    ):
        self.max_obstacles = max_obstacles
        self.danger_radius = danger_radius
        self.prediction_horizon = prediction_horizon

        self.obstacles: Dict[int, Dict] = {}

    def update_obstacles(
            self,
            tracking_results: Dict[int, TrackingResult],
            min_speed: float = 0.03,  # m/frame threshold for "dynamic"
    ):
        """Obstacle listesini güncelle"""
        for obj_id, result in tracking_results.items():
            speed = np.linalg.norm(result.velocity[:2])

            if speed > min_speed:
                self.obstacles[obj_id] = {
                    "position": result.world_position,
                    "velocity": result.velocity,
                    "confidence": result.confidence,
                }
            elif obj_id in self.obstacles:
                # Stopped moving - remove from dynamic list
                del self.obstacles[obj_id]

        # Limit size
        if len(self.obstacles) > self.max_obstacles:
            # Keep highest confidence
            sorted_obs = sorted(
                self.obstacles.items(),
                key=lambda x: x[1]["confidence"],
                reverse=True
            )
            self.obstacles = dict(sorted_obs[:self.max_obstacles])

    def get_danger_zones(
            self,
            robot_position: np.ndarray,
    ) -> List[Dict]:
        """
        Robot'a yaklaşan tehlikeli bölgeleri döndür.

        Returns:
            List of {
                "position": predicted collision point,
                "time_to_collision": frames,
                "danger_level": 0-1
            }
        """
        dangers = []

        for obs_id, obs in self.obstacles.items():
            pos = obs["position"]
            vel = obs["velocity"]

            # Simple linear prediction
            for t in range(1, self.prediction_horizon + 1):
                future_pos = pos + vel * t
                dist = np.linalg.norm(future_pos[:2] - robot_position[:2])

                if dist < self.danger_radius:
                    danger_level = 1 - (dist / self.danger_radius)
                    dangers.append({
                        "obstacle_id": obs_id,
                        "position": future_pos,
                        "time_to_collision": t,
                        "danger_level": danger_level,
                    })
                    break  # Only report first collision

        return dangers

    def get_avoidance_vector(
            self,
            robot_position: np.ndarray,
            goal_direction: np.ndarray,
    ) -> np.ndarray:
        """
        Engelden kaçınma vektörü hesapla.

        Args:
            robot_position: Current robot position
            goal_direction: Desired movement direction (normalized)

        Returns:
            Adjusted direction vector
        """
        dangers = self.get_danger_zones(robot_position)

        if not dangers:
            return goal_direction

        # Compute repulsive force from each danger
        repulsion = np.zeros(2)

        for danger in dangers:
            danger_pos = danger["position"][:2]
            diff = robot_position[:2] - danger_pos
            dist = np.linalg.norm(diff)

            if dist > 0.01:
                direction = diff / dist
                magnitude = danger["danger_level"] / max(0.1, dist)
                repulsion += direction * magnitude

        # Blend goal direction with repulsion
        if np.linalg.norm(repulsion) > 0.01:
            repulsion = repulsion / np.linalg.norm(repulsion)
            # Weighted blend: more weight on avoidance when danger is high
            max_danger = max(d["danger_level"] for d in dangers)
            blend = max_danger * 0.8  # Up to 80% avoidance
            result = (1 - blend) * goal_direction[:2] + blend * repulsion
            return np.array([result[0], result[1], goal_direction[2] if len(goal_direction) > 2 else 0])

        return goal_direction


# ========== Test ==========
if __name__ == "__main__":
    print("ObjectTracker Test")
    print("=" * 50)

    tracker = ObjectTracker()

    # Create test height map with objects
    height_map = np.zeros((64, 64), dtype=np.float32)

    # Object 1: Box at (20, 30)
    height_map[18:23, 28:33] = 0.2

    # Object 2: Ball at (40, 45)
    height_map[38:43, 43:48] = 0.15

    # Test single object tracking
    result = tracker.track_object(
        height_map=height_map,
        object_id=0,
        expected_grid_pos=(20, 30),
        expected_height=0.2,
    )

    print(f"\nObject 0:")
    print(f"  Found: {result.found}")
    print(f"  Grid pos: {result.grid_position}")
    print(f"  Confidence: {result.confidence:.2f}")

    # Test batch tracking
    objects = {
        0: {"grid_pos": (20, 30), "height": 0.2, "size": 0.2},
        1: {"grid_pos": (40, 45), "height": 0.15, "size": 0.15},
        2: {"grid_pos": (10, 10), "height": 0.3, "size": 0.2},  # Doesn't exist
    }

    results = tracker.track_objects(height_map, objects)

    print(f"\nBatch tracking ({tracker.last_track_time_ms:.3f} ms):")
    for obj_id, res in results.items():
        print(f"  [{obj_id}] found={res.found}, pos={res.grid_position}, conf={res.confidence:.2f}")

    # Performance test
    print("\n--- Performance Test ---")
    import time

    N = 1000
    t0 = time.time()
    for _ in range(N):
        tracker.track_objects(height_map, objects)
    dt = (time.time() - t0) * 1000 / N
    print(f"Average tracking time: {dt:.3f} ms (target: < 1ms)")