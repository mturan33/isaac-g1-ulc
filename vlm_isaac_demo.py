"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v3 - Fixed)
=======================================================

Fixes:
- Default forward motion when VLM/camera not available
- Debug output to see what's happening
- Robust command override for Manager-Based envs

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_vlm_rl/vlm_isaac_demo_v3.py ^
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 ^
        --checkpoint "logs/rsl_rl/unitree_go2_flat/2025-12-20_18-58-21/model_999.pt"
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
parser.add_argument("--disable_vlm", action="store_true", help="Disable VLM, use default motion")
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
from pxr import UsdGeom, Gf, UsdShade, Sdf
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


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
# Keyboard Handler
# ============================================================
class KeyboardHandler:
    """Simple keyboard input handler."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

        self.space_pressed = False
        self.reset_pressed = False
        self.quit = False

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "SPACE":
                self.space_pressed = True
            elif event.input.name == "R":
                self.reset_pressed = True
            elif event.input.name == "ESCAPE":
                self.quit = True
        return True

    def consume_space(self):
        if self.space_pressed:
            self.space_pressed = False
            return True
        return False

    def consume_reset(self):
        if self.reset_pressed:
            self.reset_pressed = False
            return True
        return False


# ============================================================
# Object Spawning
# ============================================================
def spawn_target_objects():
    """Spawn colored objects in the scene for VLM navigation."""
    stage = omni.usd.get_context().get_stage()

    # Create parent prim
    targets_path = "/World/Targets"
    if not stage.GetPrimAtPath(targets_path):
        UsdGeom.Xform.Define(stage, targets_path)

    # Define objects with positions and colors
    objects = [
        {"name": "blue_box", "type": "cube", "pos": (3.0, 2.0, 0.3), "scale": 0.3, "color": (0.1, 0.3, 0.9)},
        {"name": "red_ball", "type": "sphere", "pos": (-2.0, 3.0, 0.25), "scale": 0.25, "color": (0.9, 0.1, 0.1)},
        {"name": "green_box", "type": "cube", "pos": (2.0, -2.5, 0.3), "scale": 0.3, "color": (0.1, 0.8, 0.2)},
        {"name": "yellow_cone", "type": "cone", "pos": (-3.0, -2.0, 0.4), "scale": 0.4, "color": (0.9, 0.9, 0.1)},
        {"name": "orange_box", "type": "cube", "pos": (0.0, 4.0, 0.35), "scale": 0.35, "color": (1.0, 0.5, 0.0)},
    ]

    for obj in objects:
        prim_path = f"{targets_path}/{obj['name']}"

        # Skip if already exists
        if stage.GetPrimAtPath(prim_path):
            continue

        # Create geometry
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

        # Set position
        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*obj["pos"]))

        # Create and apply material with color
        mat_path = f"{prim_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*obj["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material to geometry
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)

        print(f"[SPAWN] Created {obj['name']} at {obj['pos']}")

    print(f"[SPAWN] Total {len(objects)} objects spawned!")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("       VLM Navigation Demo - Isaac Lab + Go2 (v3)")
    print("="*60)

    # Create environment
    print(f"[ENV] Creating: {args_cli.task}")

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

    print(f"[ENV] Observation dim: {num_obs}")
    print(f"[ENV] Action dim: {num_actions}")
    print(f"[ENV] Device: {device}")

    # Load policy
    print(f"\n[POLICY] Loading: {args_cli.checkpoint}")
    actor = None
    obs_normalizer = None

    try:
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)

        # Detect architecture
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Check observation dimension compatibility
        first_layer_key = "actor.0.weight"
        if first_layer_key in state_dict:
            checkpoint_obs_dim = state_dict[first_layer_key].shape[1]
            if checkpoint_obs_dim != num_obs:
                print(f"[WARNING] Observation dim mismatch: checkpoint={checkpoint_obs_dim}, env={num_obs}")

        # Find hidden dims
        hidden_dims = []
        for i in range(10):
            key = f"actor.{i*2}.weight"
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])
        if hidden_dims:
            hidden_dims = hidden_dims[:-1]
        if not hidden_dims:
            hidden_dims = [512, 256, 128]

        print(f"[POLICY] Hidden dims: {hidden_dims}")

        actor = ActorNetwork(num_obs, num_actions, hidden_dims).to(device)
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()

        # Load normalizer if available
        if "obs_normalizer" in checkpoint:
            obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
            print("[POLICY] Observation normalizer loaded")

        print("[POLICY] Loaded successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}")
        print("[ERROR] Cannot run without policy!")
        env.close()
        simulation_app.close()
        return

    # Controllers
    keyboard = KeyboardHandler()

    # Motion patterns for testing
    motion_patterns = [
        {"name": "Forward", "cmd": [0.5, 0.0, 0.0]},
        {"name": "Turn Left", "cmd": [0.3, 0.0, 0.5]},
        {"name": "Turn Right", "cmd": [0.3, 0.0, -0.5]},
        {"name": "Backward", "cmd": [-0.3, 0.0, 0.0]},
        {"name": "Strafe Left", "cmd": [0.0, 0.3, 0.0]},
        {"name": "Stop", "cmd": [0.0, 0.0, 0.0]},
    ]
    current_pattern = 0

    # Print controls
    print("\n" + "="*60)
    print("                    CONTROLS")
    print("="*60)
    print("  SPACE     - Change motion pattern")
    print("  R         - Reset robot")
    print("  ESC       - Quit")
    print("="*60)
    print(f"\n[START] Motion: {motion_patterns[current_pattern]['name']}")
    print(f"[START] Command: {motion_patterns[current_pattern]['cmd']}\n")

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn target objects
    print("\n[SPAWN] Creating target objects...")
    spawn_target_objects()

    step = 0

    # Get command term reference once
    cmd_term = None
    if hasattr(unwrapped, "command_manager"):
        cmd_term = unwrapped.command_manager.get_term("base_velocity")
        print(f"[DEBUG] Command term found: {cmd_term}")
        if cmd_term is not None:
            print(f"[DEBUG] Command term type: {type(cmd_term)}")
            print(f"[DEBUG] Has vel_command_b: {hasattr(cmd_term, 'vel_command_b')}")

    # Main loop
    while simulation_app.is_running() and not keyboard.quit:

        # Handle keyboard
        if keyboard.consume_space():
            current_pattern = (current_pattern + 1) % len(motion_patterns)
            pattern = motion_patterns[current_pattern]
            print(f"\n[MOTION] Changed to: {pattern['name']} -> {pattern['cmd']}")

        if keyboard.consume_reset():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[RESET] Robot reset")

        # Get current motion command
        pattern = motion_patterns[current_pattern]
        cmd = torch.tensor([pattern["cmd"]], device=device, dtype=torch.float32)

        # Set velocity command in environment
        if cmd_term is not None and hasattr(cmd_term, 'vel_command_b'):
            cmd_term.vel_command_b[:] = cmd

            # Prevent command resampling
            if hasattr(cmd_term, 'command_counter'):
                cmd_term.command_counter[:] = 0
            if hasattr(cmd_term, 'time_left'):
                cmd_term.time_left[:] = 9999.0

        # Debug output every 100 steps
        if step % 100 == 0:
            print(f"\r[Step {step:5d}] Pattern: {pattern['name']:12s} | "
                  f"Cmd: [{cmd[0,0]:.2f}, {cmd[0,1]:.2f}, {cmd[0,2]:.2f}]", end="", flush=True)

            # Show actual observation command values
            if obs.shape[1] >= 12:  # velocity_commands is at indices 9-11 (after base_lin_vel, base_ang_vel, projected_gravity)
                obs_cmd = obs[0, 9:12].cpu().numpy()
                print(f" | Obs cmd: [{obs_cmd[0]:.2f}, {obs_cmd[1]:.2f}, {obs_cmd[2]:.2f}]", end="", flush=True)

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
            print("\n[ENV] Episode ended, resetting...")

        step += 1

    # Cleanup
    print("\n[EXIT] Closing...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()