"""
VLM-RL Navigation Demo for Go2 Robot
=====================================

Bu demo:
- 1 Go2 robotu spawn eder
- Student policy (distillation) yükler
- RGB kamera ve Depth kamera görüntülerini gösterir
- VLM ile hedef nesneyi bulur
- Robot hedefe giderken izleyebilirsiniz

Kullanım:
    cd C:\IsaacLab
    python -m isaaclab.python source\isaaclab_tasks\isaaclab_tasks\direct\go2_vlm_rl\vlm_demo.py

Press 'q' to quit
Press SPACE to change target
"""

import torch
import numpy as np
import cv2
import time
import sys
import os
from datetime import datetime

# VLM Wrapper import (flash_attn bypass dahil)
# Bu dosyayı import etmeden önce flash_attn patch'i yapılmalı
import types
import importlib.util


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

    # Patch availability check
    try:
        from transformers.utils import import_utils
        import_utils.is_flash_attn_2_available = lambda: False
    except:
        pass

    print("[PATCH] Flash attention bypass installed")


# Apply patch before any transformers import
setup_flash_attn_bypass()


class VLMNavigator:
    """VLM-based navigation controller."""

    COLOR_MAP = {
        "mavi": "blue", "blue": "blue",
        "kırmızı": "red", "red": "red",
        "yeşil": "green", "green": "green",
        "sarı": "yellow", "yellow": "yellow",
        "turuncu": "orange", "orange": "orange",
        "mor": "purple", "purple": "purple",
    }

    OBJECT_MAP = {
        "sandalye": "chair", "chair": "chair",
        "kutu": "box", "box": "box",
        "top": "ball", "ball": "ball",
        "masa": "table", "table": "table",
        "koni": "cone", "cone": "cone",
    }

    def __init__(self):
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
        from PIL import Image

        self.Image = Image

        model_id = "microsoft/Florence-2-base"
        print(f"[VLM] Loading {model_id}...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[VLM] GPU Memory: {mem:.2f} GB")
        print("[VLM] Ready!")

        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup model."""
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.find_object(dummy, "red box")
        print("[VLM] Warmup done!")

    def parse_command(self, command: str):
        """Parse Turkish/English command."""
        cmd = command.lower()
        for s in ["'e", "'a", "ye", "ya", "git", "bul"]:
            cmd = cmd.replace(s, "")

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
        """Find object using VLM grounding."""
        t0 = time.time()

        color, obj = self.parse_command(command)
        target = f"{color} {obj}".strip()

        # Convert to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        pil = self.Image.fromarray(image)
        w, h = pil.size

        # Grounding
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

        # Parse result
        result = {
            "found": False,
            "target": target,
            "x": 0.0,  # -1 to 1 (left to right)
            "distance": 1.0,  # estimated distance
            "bbox": None,
            "time_ms": dt * 1000,
        }

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox

            # Normalize x position (-1 to 1)
            cx = (x1 + x2) / 2
            result["x"] = (cx / w) * 2 - 1

            # Estimate distance from bbox size
            area = (x2 - x1) * (y2 - y1) / (w * h)
            result["distance"] = max(0.1, 1.0 - area * 5)

            result["found"] = True
            result["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

        return result

    def get_velocity(self, result):
        """Convert VLM result to velocity command."""
        if not result["found"]:
            # Spin to search
            return np.array([0.0, 0.0, 0.3])

        x = result["x"]
        dist = result["distance"]

        # Check if reached target
        if dist < 0.15 and abs(x) < 0.2:
            return np.array([0.0, 0.0, 0.0])

        # Angular velocity (turn towards target)
        angular = -x * 1.0

        # Linear velocity (move forward if aligned)
        linear = min(0.3 + dist * 0.7, 1.0)
        if abs(x) > 0.5:
            linear *= 0.5  # Slow down when turning

        return np.array([linear, 0.0, angular])


class VLMDemo:
    """Main demo class with visualization."""

    COMMANDS = [
        "mavi kutuya git",
        "kırmızı topa git",
        "yeşil sandalyeye git",
        "sarı koniye git",
        "turuncu masaya git",
    ]

    def __init__(self, policy_path: str = None):
        self.policy_path = policy_path
        self.current_command_idx = 0

        # Window settings
        self.window_width = 1280
        self.window_height = 720
        self.cam_preview_size = (240, 180)

        # Recording
        self.recording = False
        self.video_writer = None

    def run_standalone(self):
        """
        Standalone demo - Isaac Lab olmadan VLM testini çalıştır.
        Simüle edilmiş görüntülerle VLM'i test eder.
        """
        print("=" * 60)
        print("VLM Navigation Demo (Standalone Mode)")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press SPACE to change target")
        print("Press 'r' to toggle recording")
        print("=" * 60)

        # Load VLM
        print("\n[INFO] Loading VLM...")
        vlm = VLMNavigator()

        # Create simulated scene image
        scene = self.create_test_scene()

        # Display window
        cv2.namedWindow('VLM Demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VLM Demo', self.window_width, self.window_height)

        # Simulated robot state
        robot_x, robot_y = 0.0, 0.0
        robot_yaw = 0.0

        step = 0
        start_time = time.time()

        while True:
            # Get current command
            command = self.COMMANDS[self.current_command_idx % len(self.COMMANDS)]

            # Run VLM (every 10 steps to save compute)
            if step % 10 == 0:
                vlm_result = vlm.find_object(scene, command)
                velocity = vlm.get_velocity(vlm_result)

                print(f"\n[VLM] Step {step}")
                print(f"  Command: '{command}'")
                print(f"  Target: '{vlm_result['target']}'")
                print(f"  Found: {vlm_result['found']}")
                if vlm_result['found']:
                    print(f"  Position X: {vlm_result['x']:.2f}")
                    print(f"  Distance: {vlm_result['distance']:.2f}")
                    print(f"  BBox: {vlm_result['bbox']}")
                print(f"  Time: {vlm_result['time_ms']:.0f}ms")
                print(f"  Velocity: vx={velocity[0]:.2f}, vyaw={velocity[2]:.2f}")

            # Update simulated robot position
            dt = 0.05
            robot_yaw += velocity[2] * dt
            robot_x += velocity[0] * np.cos(robot_yaw) * dt
            robot_y += velocity[0] * np.sin(robot_yaw) * dt

            # Create display
            display = self.create_display(
                scene,
                command,
                vlm_result,
                velocity,
                robot_x, robot_y, robot_yaw,
                step
            )

            # Show
            cv2.imshow('VLM Demo', display)

            # Record
            if self.recording and self.video_writer:
                self.video_writer.write(display)

            # Handle keys
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                print("\n[QUIT] User requested quit")
                break
            elif key == ord(' '):
                self.current_command_idx += 1
                print(f"\n[CMD] New target: {self.COMMANDS[self.current_command_idx % len(self.COMMANDS)]}")
            elif key == ord('r'):
                self.toggle_recording()
            elif key == ord('s'):
                self.save_screenshot(display)

            step += 1

        # Cleanup
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

        print("\n[DONE] Demo complete!")

    def create_test_scene(self):
        """Create a test scene with colored objects."""
        scene = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Gray background

        # Floor
        scene[300:512, :] = [180, 180, 160]

        # Blue box (right side)
        cv2.rectangle(scene, (350, 200), (450, 350), (200, 100, 50), -1)
        cv2.rectangle(scene, (350, 200), (450, 350), (150, 80, 30), 2)

        # Red ball (left side)
        cv2.circle(scene, (100, 280), 50, (50, 50, 220), -1)
        cv2.circle(scene, (100, 280), 50, (30, 30, 180), 2)

        # Green chair (center-left)
        cv2.rectangle(scene, (180, 220), (260, 340), (50, 180, 50), -1)
        cv2.rectangle(scene, (180, 180), (260, 220), (40, 150, 40), -1)

        # Yellow cone (center)
        pts = np.array([[280, 340], [320, 340], [300, 200]], np.int32)
        cv2.fillPoly(scene, [pts], (50, 220, 230))

        # Orange table (right)
        cv2.rectangle(scene, (380, 280), (490, 320), (50, 140, 240), -1)
        cv2.rectangle(scene, (390, 320), (410, 380), (40, 120, 200), -1)
        cv2.rectangle(scene, (460, 320), (480, 380), (40, 120, 200), -1)

        return scene

    def create_depth_simulation(self, scene):
        """Create simulated depth image."""
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        depth = 255 - gray  # Invert (darker = closer)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        return depth

    def create_display(self, scene, command, vlm_result, velocity,
                       robot_x, robot_y, robot_yaw, step):
        """Create visualization display."""

        # Main canvas
        display = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        display[:] = (30, 30, 30)  # Dark background

        # Title bar
        cv2.rectangle(display, (0, 0), (self.window_width, 60), (40, 40, 40), -1)
        cv2.putText(display, "VLM-RL Navigation Demo - Go2 Robot",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Main view (scene with bbox overlay)
        main_view = scene.copy()
        main_h, main_w = 400, 600

        # Draw bbox if found
        if vlm_result["found"] and vlm_result["bbox"]:
            x1, y1, x2, y2 = vlm_result["bbox"]
            # Scale bbox to main view size
            scale_x = main_w / scene.shape[1]
            scale_y = main_h / scene.shape[0]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        main_view = cv2.resize(main_view, (main_w, main_h))

        # Draw bbox on resized view
        if vlm_result["found"] and vlm_result["bbox"]:
            cv2.rectangle(main_view, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(main_view, vlm_result["target"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Place main view
        display[80:80 + main_h, 20:20 + main_w] = main_view

        # RGB camera preview (top-right)
        cam_w, cam_h = self.cam_preview_size
        rgb_preview = cv2.resize(scene, (cam_w, cam_h))
        cv2.putText(rgb_preview, "RGB Camera", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        display[80:80 + cam_h, self.window_width - cam_w - 20:self.window_width - 20] = rgb_preview

        # Depth camera preview (below RGB)
        depth = self.create_depth_simulation(scene)
        depth_preview = cv2.resize(depth, (cam_w, cam_h))
        cv2.putText(depth_preview, "Depth Camera", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        display[
            80 + cam_h + 20:80 + cam_h * 2 + 20, self.window_width - cam_w - 20:self.window_width - 20] = depth_preview

        # Info panel (right side)
        info_x = 650
        info_y = 500

        cv2.putText(display, "Current Command:", (info_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f'"{command}"', (info_x, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

        # VLM result
        status_color = (100, 255, 100) if vlm_result["found"] else (100, 100, 255)
        status_text = "TARGET FOUND" if vlm_result["found"] else "SEARCHING..."
        cv2.putText(display, status_text, (info_x, info_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Velocity display
        cv2.putText(display, f"Velocity:", (info_x, info_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Forward: {velocity[0]:.2f}", (info_x + 20, info_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Turn: {velocity[2]:.2f}", (info_x + 20, info_y + 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Robot position
        cv2.putText(display, f"Robot Position:", (info_x, info_y + 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"X: {robot_x:.2f}  Y: {robot_y:.2f}  Yaw: {np.degrees(robot_yaw):.0f}°",
                    (info_x + 20, info_y + 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mini-map (bottom-left)
        map_size = 150
        map_x, map_y = 20, self.window_height - map_size - 20
        minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        minimap[:] = (60, 60, 60)

        # Draw robot on minimap
        rx = int(map_size / 2 + robot_x * 20)
        ry = int(map_size / 2 - robot_y * 20)
        rx = np.clip(rx, 10, map_size - 10)
        ry = np.clip(ry, 10, map_size - 10)

        # Robot triangle
        angle = -robot_yaw + np.pi / 2
        pts = np.array([
            [rx + 10 * np.cos(angle), ry - 10 * np.sin(angle)],
            [rx + 6 * np.cos(angle + 2.5), ry - 6 * np.sin(angle + 2.5)],
            [rx + 6 * np.cos(angle - 2.5), ry - 6 * np.sin(angle - 2.5)],
        ], np.int32)
        cv2.fillPoly(minimap, [pts], (100, 255, 100))

        cv2.putText(minimap, "Top View", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        display[map_y:map_y + map_size, map_x:map_x + map_size] = minimap

        # Step counter
        cv2.putText(display, f"Step: {step}", (20, self.window_height - 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Recording indicator
        if self.recording:
            cv2.circle(display, (self.window_width - 40, 40), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (self.window_width - 100, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Instructions
        cv2.putText(display, "SPACE: Change Target | Q: Quit | R: Record | S: Screenshot",
                    (self.window_width // 2 - 250, self.window_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return display

    def toggle_recording(self):
        """Toggle video recording."""
        if not self.recording:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = f'vlm_demo_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, 20.0,
                (self.window_width, self.window_height)
            )
            self.recording = True
            print(f"[RECORD] Started: {video_path}")
        else:
            if self.video_writer:
                self.video_writer.release()
            self.recording = False
            print("[RECORD] Stopped")

    def save_screenshot(self, display):
        """Save screenshot."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'vlm_screenshot_{timestamp}.png'
        cv2.imwrite(path, display)
        print(f"[SCREENSHOT] Saved: {path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='VLM-RL Navigation Demo')
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to student policy checkpoint')
    parser.add_argument('--standalone', action='store_true',
                        help='Run standalone demo without Isaac Lab')
    args = parser.parse_args()

    demo = VLMDemo(policy_path=args.policy)

    # For now, run standalone mode (Isaac Lab integration will be added)
    demo.run_standalone()


if __name__ == '__main__':
    main()