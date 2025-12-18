"""
Isaac Lab Scene Generator
Creates colored objects (chairs, tables, etc.) for VLM navigation
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import random

# ============== Color Definitions ==============
COLORS = {
    "blue": (0.1, 0.3, 0.9),
    "red": (0.9, 0.1, 0.1),
    "green": (0.1, 0.8, 0.2),
    "yellow": (0.95, 0.9, 0.1),
    "orange": (1.0, 0.5, 0.1),
    "purple": (0.6, 0.1, 0.8),
    "white": (0.95, 0.95, 0.95),
    "pink": (1.0, 0.4, 0.7),
    "cyan": (0.1, 0.9, 0.9),
    "brown": (0.5, 0.3, 0.1),
}

# Object definitions: (name, size, height_offset)
OBJECTS = {
    "chair": {
        "type": "complex",  # Multi-part object
        "parts": [
            {"name": "seat", "size": (0.4, 0.4, 0.05), "offset": (0, 0, 0.45)},
            {"name": "back", "size": (0.4, 0.05, 0.5), "offset": (0, -0.175, 0.75)},
            {"name": "leg1", "size": (0.05, 0.05, 0.45), "offset": (0.15, 0.15, 0.225)},
            {"name": "leg2", "size": (0.05, 0.05, 0.45), "offset": (-0.15, 0.15, 0.225)},
            {"name": "leg3", "size": (0.05, 0.05, 0.45), "offset": (0.15, -0.15, 0.225)},
            {"name": "leg4", "size": (0.05, 0.05, 0.45), "offset": (-0.15, -0.15, 0.225)},
        ]
    },
    "table": {
        "type": "complex",
        "parts": [
            {"name": "top", "size": (0.8, 0.8, 0.05), "offset": (0, 0, 0.7)},
            {"name": "leg1", "size": (0.06, 0.06, 0.7), "offset": (0.35, 0.35, 0.35)},
            {"name": "leg2", "size": (0.06, 0.06, 0.7), "offset": (-0.35, 0.35, 0.35)},
            {"name": "leg3", "size": (0.06, 0.06, 0.7), "offset": (0.35, -0.35, 0.35)},
            {"name": "leg4", "size": (0.06, 0.06, 0.7), "offset": (-0.35, -0.35, 0.35)},
        ]
    },
    "box": {
        "type": "simple",
        "size": (0.3, 0.3, 0.3),
        "height_offset": 0.15,
    },
    "cylinder": {
        "type": "cylinder",
        "radius": 0.15,
        "height": 0.4,
        "height_offset": 0.2,
    },
    "ball": {
        "type": "sphere",
        "radius": 0.15,
        "height_offset": 0.15,
    },
    "cone": {
        "type": "cone",
        "radius": 0.2,
        "height": 0.4,
        "height_offset": 0.2,
    },
    "cabinet": {
        "type": "simple",
        "size": (0.6, 0.4, 1.2),
        "height_offset": 0.6,
    },
    "sofa": {
        "type": "complex",
        "parts": [
            {"name": "seat", "size": (0.8, 0.6, 0.3), "offset": (0, 0, 0.25)},
            {"name": "back", "size": (0.8, 0.15, 0.5), "offset": (0, -0.225, 0.65)},
            {"name": "arm_l", "size": (0.15, 0.6, 0.35), "offset": (0.325, 0, 0.375)},
            {"name": "arm_r", "size": (0.15, 0.6, 0.35), "offset": (-0.325, 0, 0.375)},
        ]
    },
}


class ObjectSpawner:
    """
    Spawns colored objects in Isaac Lab scene.
    Works with Isaac Lab's USD spawning system.
    """

    def __init__(self, stage=None):
        """
        Initialize spawner.

        Args:
            stage: USD stage (will be obtained if None)
        """
        self.stage = stage
        self.spawned_objects = []

    def generate_random_scene(
            self,
            num_objects: int = 6,
            arena_size: float = 4.0,
            min_distance: float = 1.0,
            robot_safe_radius: float = 1.5,
    ) -> List[Dict]:
        """
        Generate random object placements.

        Args:
            num_objects: Number of objects to place
            arena_size: Size of the arena (square)
            min_distance: Minimum distance between objects
            robot_safe_radius: Safe zone around robot (center)

        Returns:
            List of object specifications
        """
        available_colors = list(COLORS.keys())
        available_objects = list(OBJECTS.keys())

        objects = []
        positions = []

        for i in range(num_objects):
            # Random object and color
            obj_type = random.choice(available_objects)
            color = random.choice(available_colors)

            # Find valid position
            for _ in range(100):  # Max attempts
                x = random.uniform(-arena_size / 2, arena_size / 2)
                y = random.uniform(-arena_size / 2, arena_size / 2)

                # Check robot safe zone
                if np.sqrt(x ** 2 + y ** 2) < robot_safe_radius:
                    continue

                # Check distance from other objects
                valid = True
                for px, py in positions:
                    if np.sqrt((x - px) ** 2 + (y - py) ** 2) < min_distance:
                        valid = False
                        break

                if valid:
                    positions.append((x, y))
                    break
            else:
                continue  # Couldn't place this object

            # Random rotation
            yaw = random.uniform(0, 2 * np.pi)

            objects.append({
                "type": obj_type,
                "color": color,
                "position": (x, y, 0.0),
                "rotation": yaw,
                "id": i,
            })

        return objects

    def get_spawn_config_code(self, objects: List[Dict]) -> str:
        """
        Generate Isaac Lab Python code for spawning objects.

        Args:
            objects: List of object specifications

        Returns:
            Python code string
        """
        code_lines = [
            "# Auto-generated scene configuration",
            "# Copy this into your Isaac Lab environment config",
            "",
            "import isaaclab.sim as sim_utils",
            "from isaaclab.assets import RigidObjectCfg",
            "",
            "# Color definitions (RGB)",
            "COLORS = {",
        ]

        for name, rgb in COLORS.items():
            code_lines.append(f'    "{name}": {rgb},')
        code_lines.append("}")
        code_lines.append("")

        # Generate object configs
        code_lines.append("# Object configurations")
        code_lines.append("SCENE_OBJECTS = [")

        for obj in objects:
            obj_def = OBJECTS[obj["type"]]
            color_rgb = COLORS[obj["color"]]

            if obj_def["type"] == "simple":
                code_lines.append(f"""    RigidObjectCfg(
        prim_path="{{ENV_REGEX_NS}}/Objects/{obj['type']}_{obj['id']}",
        spawn=sim_utils.CuboidCfg(
            size={obj_def['size']},
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color={color_rgb},
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos={obj['position']},
            rot=euler_to_quat(0, 0, {obj['rotation']:.3f}),
        ),
    ),""")

            elif obj_def["type"] == "sphere":
                code_lines.append(f"""    RigidObjectCfg(
        prim_path="{{ENV_REGEX_NS}}/Objects/{obj['type']}_{obj['id']}",
        spawn=sim_utils.SphereCfg(
            radius={obj_def['radius']},
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color={color_rgb},
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=({obj['position'][0]}, {obj['position'][1]}, {obj_def['height_offset']}),
        ),
    ),""")

            elif obj_def["type"] == "cylinder":
                code_lines.append(f"""    RigidObjectCfg(
        prim_path="{{ENV_REGEX_NS}}/Objects/{obj['type']}_{obj['id']}",
        spawn=sim_utils.CylinderCfg(
            radius={obj_def['radius']},
            height={obj_def['height']},
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color={color_rgb},
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=({obj['position'][0]}, {obj['position'][1]}, {obj_def['height_offset']}),
        ),
    ),""")

        code_lines.append("]")
        code_lines.append("")
        code_lines.append("# Helper function")
        code_lines.append("""def euler_to_quat(roll, pitch, yaw):
    \"\"\"Convert Euler angles to quaternion (w, x, y, z).\"\"\"
    import math
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )""")

        return "\n".join(code_lines)

    def get_object_info_for_vlm(self, objects: List[Dict]) -> str:
        """
        Generate object list for VLM testing.

        Args:
            objects: List of object specifications

        Returns:
            Human-readable object list
        """
        lines = ["Objects in scene:"]
        for obj in objects:
            lines.append(f"  - {obj['color']} {obj['type']} at ({obj['position'][0]:.1f}, {obj['position'][1]:.1f})")
        return "\n".join(lines)


class VLMNavigationEnvHelper:
    """
    Helper class for VLM navigation environment.
    Integrates with Isaac Lab environment.
    """

    def __init__(self, objects: List[Dict]):
        """
        Initialize helper.

        Args:
            objects: List of spawned objects
        """
        self.objects = objects
        self.object_positions = {
            f"{obj['color']}_{obj['type']}": np.array(obj['position'][:2])
            for obj in objects
        }

    def get_target_position(self, color: str, obj_type: str) -> Optional[np.ndarray]:
        """Get world position of target object."""
        key = f"{color}_{obj_type}"
        return self.object_positions.get(key)

    def compute_distance_reward(
            self,
            robot_pos: np.ndarray,
            target_color: str,
            target_type: str,
    ) -> float:
        """
        Compute distance-based reward.

        Args:
            robot_pos: Robot's (x, y) position
            target_color: Target object color
            target_type: Target object type

        Returns:
            Reward (higher = closer to target)
        """
        target_pos = self.get_target_position(target_color, target_type)
        if target_pos is None:
            return 0.0

        distance = np.linalg.norm(robot_pos[:2] - target_pos)

        # Exponential reward: max 1.0 at target, decays with distance
        reward = np.exp(-distance / 2.0)

        return float(reward)

    def check_goal_reached(
            self,
            robot_pos: np.ndarray,
            target_color: str,
            target_type: str,
            threshold: float = 0.5,
    ) -> bool:
        """Check if robot reached the target."""
        target_pos = self.get_target_position(target_color, target_type)
        if target_pos is None:
            return False

        distance = np.linalg.norm(robot_pos[:2] - target_pos)
        return distance < threshold

    def get_random_command(self) -> Tuple[str, str, str]:
        """
        Get random navigation command.

        Returns:
            (command_str, color, object_type)
        """
        obj = random.choice(self.objects)

        # Random language style
        templates = [
            "{color} {type}'ye git",
            "{color} {type}'a git",
            "go to the {color} {type}",
            "{color} renkli {type}'yi bul",
            "navigate to {color} {type}",
        ]

        template = random.choice(templates)
        command = template.format(color=obj['color'], type=obj['type'])

        return command, obj['color'], obj['type']


# ============== Standalone Test ==============
if __name__ == "__main__":
    print("=" * 60)
    print("Scene Generator Test")
    print("=" * 60)

    # Create spawner
    spawner = ObjectSpawner()

    # Generate random scene
    print("\n[Test] Generating random scene...")
    objects = spawner.generate_random_scene(
        num_objects=8,
        arena_size=5.0,
        min_distance=1.2,
        robot_safe_radius=1.5,
    )

    # Print object info
    print("\n" + spawner.get_object_info_for_vlm(objects))

    # Generate code
    print("\n[Test] Generated Isaac Lab config code:")
    print("-" * 40)
    code = spawner.get_spawn_config_code(objects)
    print(code[:2000] + "\n..." if len(code) > 2000 else code)

    # Test helper
    print("\n[Test] Testing navigation helper...")
    helper = VLMNavigationEnvHelper(objects)

    # Get random command
    cmd, color, obj_type = helper.get_random_command()
    print(f"  Random command: '{cmd}'")

    # Check target position
    target = helper.get_target_position(color, obj_type)
    print(f"  Target position: {target}")

    # Test reward
    robot_pos = np.array([0.0, 0.0])
    reward = helper.compute_distance_reward(robot_pos, color, obj_type)
    print(f"  Distance reward from origin: {reward:.3f}")

    # Test goal check
    reached = helper.check_goal_reached(robot_pos, color, obj_type, threshold=0.5)
    print(f"  Goal reached: {reached}")

    print("\n" + "=" * 60)
    print("Test completed!")