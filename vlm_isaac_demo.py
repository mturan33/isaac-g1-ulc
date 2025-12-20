"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v4 - Full VLM)
==========================================================

Features:
- Front-mounted RGB camera on robot
- Camera view displayed in separate viewport
- Florence-2 VLM for object detection
- Autonomous navigation to detected objects

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_vlm_rl/vlm_isaac_demo_v4.py ^
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 ^
        --checkpoint "logs/rsl_rl/unitree_go2_flat/2025-12-20_18-58-21/model_999.pt"

Controls:
    SPACE - Change target object
    R     - Reset robot
    ESC   - Quit
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import importlib.util

# ============================================================
# Flash Attention Bypass (MUST BE FIRST)
# ============================================================
def setup_flash_attn_bypass():
    """Flash attention bypass for Windows."""
    fake_flash_attn = types.ModuleType('flash_attn')
    fake_flash_attn.__file__ = __file__
    fake_flash_attn.__path__ = []
    fake_flash_attn.__package__ = 'flash_attn'
    fake_spec = importlib.util.spec_from_loader('flash_attn', loader=None)
    fake_flash_attn.__spec__ = fake_spec
    fake_flash_attn.flash_attn_func = None

    fake_bert_padding = types.ModuleType('flash_attn.bert_padding')
    fake_bert_padding.__file__ = __file__
    fake_bert_padding.__package__ = 'flash_attn.bert_padding'
    fake_bert_padding.__spec__ = importlib.util.spec_from_loader('flash_attn.bert_padding', loader=None)
    fake_bert_padding.index_first_axis = lambda *a, **k: None
    fake_bert_padding.pad_input = lambda *a, **k: None
    fake_bert_padding.unpad_input = lambda *a, **k: None

    sys.modules['flash_attn'] = fake_flash_attn
    sys.modules['flash_attn.bert_padding'] = fake_bert_padding

    try:
        from transformers.utils import import_utils
        import_utils.is_flash_attn_2_available = lambda: False
    except:
        pass

    print("[PATCH] Flash attention bypass installed")

setup_flash_attn_bypass()
# ============================================================

from isaaclab.app import AppLauncher

# Argument parser
parser = argparse.ArgumentParser(description="VLM Navigation Demo for Go2")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0",
                   help="Isaac Lab task name")
parser.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_vlm", action="store_true", help="Disable VLM, use manual control")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric for debugging")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports (after AppLauncher)
import carb
import omni.appwindow
import omni.usd
import omni.ui as ui
from pxr import UsdGeom, Gf, UsdShade, Sdf, Usd
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


# ============================================================
# VLM Navigator (Florence-2)
# ============================================================
class VLMNavigator:
    """Florence-2 based object grounding."""

    COLOR_MAP = {
        "mavi": "blue", "kırmızı": "red", "yeşil": "green",
        "sarı": "yellow", "turuncu": "orange", "mor": "purple",
        "beyaz": "white", "siyah": "black", "pembe": "pink",
        "blue": "blue", "red": "red", "green": "green",
        "yellow": "yellow", "orange": "orange",
    }

    OBJECT_MAP = {
        "kutu": "box", "top": "ball", "sandalye": "chair",
        "masa": "table", "koni": "cone", "koltuk": "sofa",
        "box": "box", "ball": "ball", "cone": "cone",
    }

    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
        from PIL import Image

        self.Image = Image
        self.device = device

        model_id = "microsoft/Florence-2-base"
        print(f"[VLM] Loading {model_id}...")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # Warmup
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.find_object(dummy, "red box")

        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[VLM] GPU Memory: {mem:.2f} GB - Ready!")

    def parse_command(self, command: str):
        cmd = command.lower()
        color, obj = "", "object"
        for tr, en in self.COLOR_MAP.items():
            if tr in cmd:
                color = en
                break
        for tr, en in self.OBJECT_MAP.items():
            if tr in cmd:
                obj = en
                break
        return color, obj

    def find_object(self, image: np.ndarray, command: str):
        """Find object in image using VLM grounding."""
        import time
        t0 = time.time()

        color, obj = self.parse_command(command)
        target = f"{color} {obj}".strip()

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

        pil = self.Image.fromarray(image)
        w, h = pil.size

        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(text=task + target, images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        text = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

        dt = time.time() - t0

        result = {
            "found": False,
            "target": target,
            "x": 0.0,
            "y": 0.0,
            "distance": 1.0,
            "bbox": None,
            "time_ms": dt * 1000,
        }

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            result["x"] = (cx / w) * 2 - 1  # -1 (left) to 1 (right)
            result["y"] = (cy / h) * 2 - 1  # -1 (top) to 1 (bottom)
            area = (x2-x1) * (y2-y1) / (w*h)
            result["distance"] = max(0.1, 1.0 - area * 5)
            result["found"] = True
            result["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

        return result


# ============================================================
# Robot Camera with Viewport Display
# ============================================================
class RobotCamera:
    """Front-mounted camera on robot with viewport display."""

    def __init__(self, resolution=(320, 240)):
        self.resolution = resolution
        self.camera_prim = None
        self.render_product = None
        self.rgb_annotator = None
        self._initialized = False
        self._last_image = None
        self._viewport_window = None
        self._image_provider = None

    def create_camera(self, robot_base_path: str):
        """Create a front-mounted camera on the robot."""
        try:
            import omni.replicator.core as rep

            stage = omni.usd.get_context().get_stage()

            # Camera path - attached to robot base
            camera_path = f"{robot_base_path}/front_camera"

            # Delete if exists
            existing = stage.GetPrimAtPath(camera_path)
            if existing:
                stage.RemovePrim(camera_path)

            # Create camera
            camera = UsdGeom.Camera.Define(stage, camera_path)

            # Camera properties - wide angle for navigation
            camera.GetFocalLengthAttr().Set(15.0)  # Wide angle
            camera.GetHorizontalApertureAttr().Set(20.955)
            camera.GetVerticalApertureAttr().Set(15.2908)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            # Position camera at front of robot, looking forward
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()

            # Translate: forward (x=0.35), up (z=0.25) from robot base
            translate = xform.AddTranslateOp()
            translate.Set(Gf.Vec3d(0.35, 0.0, 0.25))

            # Rotate: look forward (no rotation needed, camera looks along +X by default)
            # But USD cameras look along -Z, so we need to rotate
            rotate = xform.AddRotateYXZOp()
            rotate.Set(Gf.Vec3f(0.0, 90.0, 0.0))  # Rotate to look along +X

            self.camera_prim = camera.GetPrim()
            self.camera_path = camera_path
            print(f"[CAMERA] Created front camera at {camera_path}")

            # Create render product
            self.render_product = rep.create.render_product(
                camera_path,
                self.resolution
            )

            # Create RGB annotator
            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])

            self._initialized = True
            print(f"[CAMERA] Render product ready: {self.resolution}")

            # Create viewport window to show camera view
            self._create_camera_viewport(camera_path)

            return True

        except Exception as e:
            print(f"[CAMERA] Failed to create camera: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_camera_viewport(self, camera_path: str):
        """Create a viewport window showing the robot's camera view."""
        try:
            from omni.kit.viewport.utility import get_active_viewport_and_window
            import omni.kit.viewport.utility as viewport_utils

            # Get viewport API
            viewport_api = viewport_utils.get_active_viewport()

            # Create a new viewport window for robot camera
            from omni.kit.widget.viewport import ViewportWidget

            # Create UI window
            self._viewport_window = ui.Window(
                "Robot Camera View",
                width=self.resolution[0] + 20,
                height=self.resolution[1] + 40,
                flags=ui.WINDOW_FLAGS_NO_SCROLLBAR
            )

            with self._viewport_window.frame:
                with ui.VStack():
                    ui.Label("Robot Front Camera", height=20, alignment=ui.Alignment.CENTER)
                    # We'll update this with captured images
                    self._image_widget = ui.ImageWithProvider(
                        width=self.resolution[0],
                        height=self.resolution[1]
                    )

            print(f"[CAMERA] Created viewport window")

        except Exception as e:
            print(f"[CAMERA] Viewport window creation failed: {e}")
            # Continue without viewport - capture will still work

    def capture(self) -> np.ndarray:
        """Capture RGB image from camera."""
        if not self._initialized:
            return None

        try:
            import omni.replicator.core as rep

            # Step replicator to render
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

            # Get data from annotator
            data = self.rgb_annotator.get_data()

            if data is not None and len(data) > 0:
                # Convert to numpy array
                img = np.array(data)

                # Handle different formats
                if img.ndim == 3:
                    if img.shape[2] == 4:  # RGBA
                        img = img[:, :, :3]  # RGB only
                    self._last_image = img

                    # Update viewport if available
                    self._update_viewport_image(img)

                    return img

        except Exception as e:
            if self._last_image is None:
                print(f"[CAMERA] Capture error: {e}")

        return self._last_image

    def _update_viewport_image(self, img: np.ndarray):
        """Update the viewport window with new image."""
        if self._viewport_window is None or self._image_widget is None:
            return

        try:
            # Convert numpy to bytes for UI
            from PIL import Image
            import io

            pil_img = Image.fromarray(img.astype(np.uint8))

            # Create byte provider
            byte_provider = ui.ByteImageProvider()
            byte_provider.set_bytes_data(
                img.flatten().tobytes(),
                [img.shape[1], img.shape[0]]  # width, height
            )

            self._image_widget.image_provider = byte_provider

        except Exception as e:
            pass  # Silent fail for viewport update


# ============================================================
# Policy Network
# ============================================================
class ActorNetwork(nn.Module):
    """Actor network for locomotion policy."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list = [512, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_actions))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class EmpiricalNormalization(nn.Module):
    """Running observation normalization."""

    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.epsilon = epsilon

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)


# ============================================================
# VLM Controller
# ============================================================
class VLMController:
    """Converts VLM output to velocity commands."""

    def __init__(self, device):
        self.device = device
        self._command = torch.tensor([[0.3, 0.0, 0.3]], device=device)  # Default: forward + turn
        self._search_direction = 1.0

        # Navigation targets
        self.targets = [
            "mavi kutu",      # blue box
            "kırmızı top",    # red ball
            "yeşil kutu",     # green box
            "sarı koni",      # yellow cone
            "turuncu kutu",   # orange box
        ]
        self.target_idx = 0
        self.current_target = self.targets[0]

        # State
        self.vlm_result = None
        self.target_reached = False
        self.search_steps = 0

    def next_target(self):
        self.target_idx = (self.target_idx + 1) % len(self.targets)
        self.current_target = self.targets[self.target_idx]
        self.vlm_result = None
        self.target_reached = False
        self.search_steps = 0
        print(f"\n[TARGET] New target: {self.current_target}")
        return self.current_target

    def update_from_vlm(self, vlm_result):
        """Update velocity command from VLM result."""
        self.vlm_result = vlm_result

        if not vlm_result["found"]:
            # Spin to search for object
            self.search_steps += 1
            if self.search_steps > 150:  # Change direction after ~3 seconds
                self._search_direction *= -1
                self.search_steps = 0

            # Search: slow forward + rotate
            self._command = torch.tensor(
                [[0.1, 0.0, 0.4 * self._search_direction]],
                device=self.device
            )
            self.target_reached = False
        else:
            self.search_steps = 0
            x = vlm_result["x"]      # -1 (left) to 1 (right)
            dist = vlm_result["distance"]  # 0 (close) to 1 (far)

            # Check if target reached
            if dist < 0.3 and abs(x) < 0.25:
                self._command = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
                if not self.target_reached:
                    print(f"\n[VLM] ★★★ TARGET REACHED: {vlm_result['target']} ★★★")
                    self.target_reached = True
            else:
                self.target_reached = False
                # Navigate towards target
                angular = -x * 0.6  # Turn towards target

                # Slow down when turning sharply
                turn_factor = max(0.4, 1.0 - abs(x) * 0.6)
                linear = (0.3 + dist * 0.3) * turn_factor

                self._command = torch.tensor(
                    [[linear, 0.0, angular]],
                    device=self.device
                )

    def get_command(self) -> torch.Tensor:
        return self._command

    def get_status(self) -> str:
        if self.vlm_result is None:
            return f"[VLM] Waiting... Target: {self.current_target}"

        r = self.vlm_result
        if r["found"]:
            status = "REACHED" if self.target_reached else "TRACKING"
            return f"[VLM] {status} '{r['target']}' | x={r['x']:.2f} dist={r['distance']:.2f} | {r['time_ms']:.0f}ms"
        else:
            return f"[VLM] SEARCHING '{r['target']}' | {r['time_ms']:.0f}ms"


# ============================================================
# Keyboard Handler
# ============================================================
class KeyboardHandler:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

        self.keys_pressed = set()
        self.keys_just_pressed = set()

    def _on_key(self, event, *args, **kwargs):
        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key not in self.keys_pressed:
                self.keys_just_pressed.add(key)
            self.keys_pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys_pressed.discard(key)
        return True

    def is_pressed(self, key: str) -> bool:
        return key in self.keys_pressed

    def just_pressed(self, key: str) -> bool:
        if key in self.keys_just_pressed:
            self.keys_just_pressed.discard(key)
            return True
        return False


# ============================================================
# Object Spawning
# ============================================================
def spawn_target_objects():
    """Spawn colored objects in the scene."""
    stage = omni.usd.get_context().get_stage()

    targets_path = "/World/Targets"
    if not stage.GetPrimAtPath(targets_path):
        UsdGeom.Xform.Define(stage, targets_path)

    objects = [
        {"name": "blue_box", "type": "cube", "pos": (3.0, 2.0, 0.3), "scale": 0.3, "color": (0.1, 0.3, 0.9)},
        {"name": "red_ball", "type": "sphere", "pos": (-2.0, 3.0, 0.25), "scale": 0.25, "color": (0.9, 0.1, 0.1)},
        {"name": "green_box", "type": "cube", "pos": (2.0, -2.5, 0.3), "scale": 0.3, "color": (0.1, 0.8, 0.2)},
        {"name": "yellow_cone", "type": "cone", "pos": (-3.0, -2.0, 0.4), "scale": 0.4, "color": (0.9, 0.9, 0.1)},
        {"name": "orange_box", "type": "cube", "pos": (0.0, 4.0, 0.35), "scale": 0.35, "color": (1.0, 0.5, 0.0)},
    ]

    for obj in objects:
        prim_path = f"{targets_path}/{obj['name']}"
        if stage.GetPrimAtPath(prim_path):
            continue

        if obj["type"] == "cube":
            geom = UsdGeom.Cube.Define(stage, prim_path)
            geom.GetSizeAttr().Set(obj["scale"] * 2)
        elif obj["type"] == "sphere":
            geom = UsdGeom.Sphere.Define(stage, prim_path)
            geom.GetRadiusAttr().Set(obj["scale"])
        elif obj["type"] == "cone":
            geom = UsdGeom.Cone.Define(stage, prim_path)
            geom.GetRadiusAttr().Set(obj["scale"])
            geom.GetHeightAttr().Set(obj["scale"] * 2)

        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*obj["pos"]))

        mat_path = f"{prim_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*obj["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)

        print(f"[SPAWN] Created {obj['name']} at {obj['pos']}")

    print(f"[SPAWN] Total {len(objects)} objects spawned!")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("     VLM Navigation Demo - Go2 + Florence-2 (v4)")
    print("="*60)

    # Create environment
    print(f"\n[ENV] Creating: {args_cli.task}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    num_obs = unwrapped.observation_space["policy"].shape[1]
    num_actions = unwrapped.action_space.shape[1]
    device = unwrapped.device

    print(f"[ENV] Obs: {num_obs}, Act: {num_actions}, Device: {device}")

    # Load policy
    print(f"\n[POLICY] Loading: {args_cli.checkpoint}")
    try:
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        hidden_dims = []
        for i in range(10):
            key = f"actor.{i*2}.weight"
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])
        hidden_dims = hidden_dims[:-1] if hidden_dims else [512, 256, 128]

        actor = ActorNetwork(num_obs, num_actions, hidden_dims).to(device)
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()

        obs_normalizer = None
        if "obs_normalizer" in checkpoint:
            obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

        print(f"[POLICY] Loaded! Hidden: {hidden_dims}")

    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}")
        env.close()
        simulation_app.close()
        return

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn objects
    print("\n[SPAWN] Creating target objects...")
    spawn_target_objects()

    # Create robot camera
    print("\n[CAMERA] Setting up front camera...")
    camera = RobotCamera(resolution=(320, 240))
    robot_base_path = "/World/envs/env_0/Robot/base"
    camera_ready = camera.create_camera(robot_base_path)

    # Initialize VLM
    vlm = None
    if not args_cli.disable_vlm:
        print("\n[VLM] Initializing Florence-2...")
        try:
            vlm = VLMNavigator(device=str(device))
        except Exception as e:
            print(f"[VLM] Failed: {e}")
            vlm = None

    # Controllers
    vlm_ctrl = VLMController(device)
    keyboard = KeyboardHandler()

    # Get command term
    cmd_term = None
    if hasattr(unwrapped, "command_manager"):
        cmd_term = unwrapped.command_manager.get_term("base_velocity")

    # Print controls
    print("\n" + "="*60)
    print("  SPACE - Change target | R - Reset | ESC - Quit")
    print("="*60)
    print(f"\n[START] Target: {vlm_ctrl.current_target}\n")

    step = 0
    vlm_interval = 10  # Run VLM every N steps

    # Main loop
    while simulation_app.is_running() and not keyboard.is_pressed("ESCAPE"):

        # Handle keyboard
        if keyboard.just_pressed("SPACE"):
            vlm_ctrl.next_target()

        if keyboard.just_pressed("R"):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[RESET] Robot reset")

        # Run VLM
        if vlm is not None and camera_ready and step % vlm_interval == 0:
            img = camera.capture()
            if img is not None:
                vlm_result = vlm.find_object(img, vlm_ctrl.current_target)
                vlm_ctrl.update_from_vlm(vlm_result)

                # Print status
                print(f"\r{vlm_ctrl.get_status()}     ", end="", flush=True)

        # Get velocity command
        cmd = vlm_ctrl.get_command()

        # Set command in environment
        if cmd_term is not None and hasattr(cmd_term, 'vel_command_b'):
            cmd_term.vel_command_b[:] = cmd
            if hasattr(cmd_term, 'command_counter'):
                cmd_term.command_counter[:] = 0

        # Get action from policy
        with torch.no_grad():
            obs_input = obs_normalizer.normalize(obs) if obs_normalizer else obs
            actions = actor(obs_input)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # Handle episode end
        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[ENV] Episode reset")

        step += 1

    print("\n[EXIT] Closing...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()