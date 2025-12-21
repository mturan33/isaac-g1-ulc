"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v6 - Heavy Debug)
=============================================================
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import importlib.util
import traceback

# Flash Attention Bypass
def setup_flash_attn_bypass():
    fake_flash_attn = types.ModuleType('flash_attn')
    fake_flash_attn.__file__ = __file__
    fake_flash_attn.__path__ = []
    fake_flash_attn.__package__ = 'flash_attn'
    fake_flash_attn.__spec__ = importlib.util.spec_from_loader('flash_attn', loader=None)
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--no_vlm", action="store_true", help="Disable VLM completely")
parser.add_argument("--no_camera", action="store_true", help="Disable camera")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import omni.usd
import omni.ui as ui
from pxr import UsdGeom, Gf, UsdShade, Sdf
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


# ============================================================
# VLM Navigator
# ============================================================
class VLMNavigator:
    COLOR_MAP = {"mavi": "blue", "kırmızı": "red", "yeşil": "green", "sarı": "yellow", "turuncu": "orange"}
    OBJECT_MAP = {"kutu": "box", "top": "ball", "koni": "cone"}

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
            model_id, config=config, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        self.model.eval()

        # Warmup
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.find_object(dummy, "red box")
        print(f"[VLM] Ready! GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def find_object(self, image: np.ndarray, command: str):
        import time
        t0 = time.time()

        # Parse command
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
                max_new_tokens=1024, num_beams=3,
            )

        text = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

        result = {"found": False, "target": target, "x": 0.0, "distance": 1.0, "time_ms": (time.time()-t0)*1000}

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox
            cx = (x1+x2)/2
            result["x"] = (cx / w) * 2 - 1
            area = (x2-x1) * (y2-y1) / (w*h)
            result["distance"] = max(0.1, 1.0 - area * 5)
            result["found"] = True

        return result


# ============================================================
# Policy Network
# ============================================================
class ActorNetwork(nn.Module):
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


# ============================================================
# Keyboard Handler
# ============================================================
class KeyboardHandler:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self.keys_just_pressed = set()

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keys_just_pressed.add(event.input.name)
        return True

    def just_pressed(self, key: str) -> bool:
        if key in self.keys_just_pressed:
            self.keys_just_pressed.discard(key)
            return True
        return False


# ============================================================
# Object Spawning
# ============================================================
def spawn_target_objects():
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
        xform.AddTranslateOp().Set(Gf.Vec3d(*obj["pos"]))

        mat_path = f"{prim_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*obj["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)

    print(f"[SPAWN] Objects created!")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("     VLM Navigation Demo - Go2 (v6 Debug)")
    print("="*60)

    # Create environment
    print(f"\n[ENV] Creating: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1,
                            use_fabric=not args_cli.disable_fabric)
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
        print(f"[POLICY] Loaded! Hidden: {hidden_dims}")
    except Exception as e:
        print(f"[ERROR] Policy: {e}")
        return

    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn objects
    spawn_target_objects()

    # Setup camera (optional)
    camera_img = None
    render_product = None
    rgb_annotator = None

    if not args_cli.no_camera:
        print("\n[CAMERA] Setting up...")
        try:
            import omni.replicator.core as rep
            stage = omni.usd.get_context().get_stage()

            # Create camera at fixed position (not attached to robot)
            camera_path = "/World/VLMCamera"
            if stage.GetPrimAtPath(camera_path):
                stage.RemovePrim(camera_path)

            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.GetFocalLengthAttr().Set(18.0)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -3.0, 2.0))  # Behind and above
            xform.AddRotateXYZOp().Set(Gf.Vec3f(30.0, 0.0, 0.0))  # Look down

            render_product = rep.create.render_product(camera_path, (320, 240))
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annotator.attach([render_product])

            print("[CAMERA] Ready!")
        except Exception as e:
            print(f"[CAMERA] Failed: {e}")
            traceback.print_exc()

    # Setup VLM (optional)
    vlm = None
    if not args_cli.no_vlm:
        print("\n[VLM] Initializing...")
        try:
            vlm = VLMNavigator(device=str(device))
        except Exception as e:
            print(f"[VLM] Failed: {e}")
            traceback.print_exc()

    # Get command term
    cmd_term = None
    if hasattr(unwrapped, "command_manager"):
        cmd_term = unwrapped.command_manager.get_term("base_velocity")
        print(f"[CMD] Ready: {cmd_term is not None}")

    keyboard = KeyboardHandler()

    # Navigation state
    targets = ["mavi kutu", "kırmızı top", "yeşil kutu", "sarı koni", "turuncu kutu"]
    target_idx = 0

    # Default command - robot should move forward and turn
    command = torch.tensor([[0.4, 0.0, 0.3]], device=device, dtype=torch.float32)
    search_dir = 1.0

    print("\n" + "="*60)
    print("  SPACE - Next target | R - Reset | ESC - Quit")
    print("="*60)
    print(f"\n[START] Target: {targets[target_idx]}")
    print(f"[START] Default command: {command.cpu().numpy()}")
    print(f"[START] VLM enabled: {vlm is not None}")
    print(f"[START] Camera enabled: {rgb_annotator is not None}\n")

    step = 0
    vlm_interval = 20

    # Main loop
    while simulation_app.is_running():

        # Check ESC
        if keyboard.just_pressed("ESCAPE"):
            print("\n[EXIT] ESC pressed")
            break

        # SPACE - next target
        if keyboard.just_pressed("SPACE"):
            target_idx = (target_idx + 1) % len(targets)
            print(f"\n[TARGET] {targets[target_idx]}")

        # R - reset
        if keyboard.just_pressed("R"):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[RESET]")

        # Debug print every 100 steps
        if step % 100 == 0:
            print(f"[STEP {step:5d}] cmd=[{command[0,0]:.2f}, {command[0,1]:.2f}, {command[0,2]:.2f}]")

        # Camera capture (every 10 steps)
        if rgb_annotator is not None and step % 10 == 0:
            try:
                import omni.replicator.core as rep
                rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
                data = rgb_annotator.get_data()
                if data is not None and len(data) > 0:
                    camera_img = np.array(data)
                    if camera_img.ndim == 3 and camera_img.shape[2] == 4:
                        camera_img = camera_img[:, :, :3]
                    if step % 100 == 0:
                        print(f"[CAMERA] Got image: {camera_img.shape}")
            except Exception as e:
                if step % 100 == 0:
                    print(f"[CAMERA] Error: {e}")

        # VLM inference (every vlm_interval steps)
        if vlm is not None and camera_img is not None and step % vlm_interval == 0:
            try:
                result = vlm.find_object(camera_img, targets[target_idx])

                if result["found"]:
                    x = result["x"]
                    dist = result["distance"]
                    print(f"[VLM] FOUND '{result['target']}' x={x:.2f} d={dist:.2f} | {result['time_ms']:.0f}ms")

                    # Navigate
                    if dist < 0.3 and abs(x) < 0.25:
                        command = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
                        print(f"[VLM] ★ TARGET REACHED ★")
                    else:
                        angular = -x * 0.5
                        linear = 0.3 + dist * 0.2
                        command = torch.tensor([[linear, 0.0, angular]], device=device, dtype=torch.float32)
                else:
                    print(f"[VLM] SEARCHING '{result['target']}' | {result['time_ms']:.0f}ms")
                    # Search pattern
                    command = torch.tensor([[0.15, 0.0, 0.4 * search_dir]], device=device, dtype=torch.float32)

            except Exception as e:
                print(f"[VLM] Error: {e}")
                traceback.print_exc()

        # Apply velocity command
        if cmd_term is not None:
            try:
                cmd_term.vel_command_b[:] = command
                if hasattr(cmd_term, 'command_counter'):
                    cmd_term.command_counter[:] = 0
                if hasattr(cmd_term, 'time_left'):
                    cmd_term.time_left[:] = 9999.0
            except Exception as e:
                if step % 100 == 0:
                    print(f"[CMD] Error: {e}")

        # Policy inference
        with torch.no_grad():
            actions = actor(obs)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[ENV] Episode reset")

        step += 1

    print("\n[EXIT] Done")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()