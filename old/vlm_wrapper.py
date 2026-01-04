"""
Florence-2 VLM Wrapper with Grounding
Complete fake flash_attn module with __spec__ attribute
"""

import torch
import sys
import os
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import importlib.util
import types

# ============================================================
# PATCH: Complete fake flash_attn module
# Must include __spec__ for importlib.util.find_spec() check
# ============================================================

def create_fake_flash_attn():
    """Create a complete fake flash_attn module with all required attributes."""

    # Create main module
    fake_module = types.ModuleType('flash_attn')
    fake_module.__file__ = __file__
    fake_module.__path__ = []
    fake_module.__package__ = 'flash_attn'

    # Create a proper ModuleSpec
    fake_spec = importlib.util.spec_from_loader('flash_attn', loader=None)
    fake_module.__spec__ = fake_spec

    # Add fake functions
    fake_module.flash_attn_func = None
    fake_module.flash_attn_varlen_func = None

    # Create bert_padding submodule
    fake_bert_padding = types.ModuleType('flash_attn.bert_padding')
    fake_bert_padding.__file__ = __file__
    fake_bert_padding.__package__ = 'flash_attn.bert_padding'
    fake_bert_padding_spec = importlib.util.spec_from_loader('flash_attn.bert_padding', loader=None)
    fake_bert_padding.__spec__ = fake_bert_padding_spec

    # Add placeholder functions to bert_padding
    def index_first_axis(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")
    def pad_input(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")
    def unpad_input(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")

    fake_bert_padding.index_first_axis = index_first_axis
    fake_bert_padding.pad_input = pad_input
    fake_bert_padding.unpad_input = unpad_input

    # Install modules
    sys.modules['flash_attn'] = fake_module
    sys.modules['flash_attn.bert_padding'] = fake_bert_padding

    print("[PATCH] Installed complete fake flash_attn module")

# Install fake module BEFORE any imports
create_fake_flash_attn()

# Also patch the availability check function
def patch_flash_attn_check():
    """Patch transformers to think flash_attn is NOT available."""
    try:
        from transformers.utils import import_utils
        # Override the check function to always return False
        import_utils.is_flash_attn_2_available = lambda: False
        print("[PATCH] Patched is_flash_attn_2_available to return False")
    except Exception as e:
        print(f"[PATCH] Could not patch import_utils: {e}")

patch_flash_attn_check()
# ============================================================


class VLMWrapper:
    """Florence-2 with real grounding (bounding boxes)."""

    COLOR_MAP = {
        "mavi": "blue", "blue": "blue",
        "kırmızı": "red", "red": "red",
        "yeşil": "green", "green": "green",
        "sarı": "yellow", "yellow": "yellow",
        "turuncu": "orange", "orange": "orange",
        "mor": "purple", "purple": "purple",
        "beyaz": "white", "white": "white",
        "siyah": "black", "black": "black",
        "pembe": "pink", "pink": "pink",
    }

    OBJECT_MAP = {
        "sandalye": "chair", "chair": "chair",
        "masa": "table", "table": "table",
        "kutu": "box", "box": "box",
        "top": "ball", "ball": "ball",
        "koltuk": "sofa", "sofa": "sofa",
    }

    def __init__(self, device: str = "cuda"):
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

        model_id = "microsoft/Florence-2-base"
        print(f"[VLM] Loading {model_id}...")

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[VLM] VRAM: {vram:.1f} GB")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Load config and force eager attention
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"
        if hasattr(config, 'text_config'):
            config.text_config._attn_implementation = "eager"
        if hasattr(config, 'vision_config'):
            config.vision_config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

        self.device = device
        self.model.eval()

        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[VLM] GPU Memory: {mem:.2f} GB")
        print("[VLM] Ready!")

    def parse_command(self, command: str) -> Tuple[str, str]:
        cmd = command.lower()
        for s in ["'e", "'a", "ye", "ya", "e git", "a git"]:
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

    def ground_object(self, image: np.ndarray, command: str) -> Dict:
        """Find object with bounding box."""
        import time
        t0 = time.time()

        color, obj = self.parse_command(command)
        target = f"{color} {obj}".strip()
        print(f"[VLM] Looking for: '{target}'")

        # To PIL
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
            pil = Image.fromarray(image)
        else:
            pil = image

        w, h = pil.size

        # Grounding task
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
        print(f"[VLM] Time: {dt*1000:.0f}ms")
        print(f"[VLM] Result: {parsed}")

        # Parse result
        result = self._parse(parsed, w, h, color, obj)

        if not result["found"]:
            print("[VLM] Trying object detection fallback...")
            result = self._detect(pil, color, obj)

        return result

    def _parse(self, parsed, w, h, color, obj):
        try:
            key = "<CAPTION_TO_PHRASE_GROUNDING>"
            if key in parsed and parsed[key].get("bboxes"):
                bbox = parsed[key]["bboxes"][0]
                x1, y1, x2, y2 = bbox

                cx = (x1 + x2) / 2
                x_norm = (cx / w) * 2 - 1

                area = (x2-x1) * (y2-y1) / (w*h)
                dist = max(0.1, 1.0 - area * 5)

                return {"found": True, "x": x_norm, "y": dist, "confidence": 0.9,
                        "color": color, "object": obj, "bbox": list(bbox)}
        except Exception as e:
            print(f"[VLM] Parse error: {e}")

        return {"found": False, "x": 0, "y": 0.5, "confidence": 0, "color": color, "object": obj}

    def _detect(self, pil, color, obj):
        """Object detection fallback."""
        w, h = pil.size
        task = "<OD>"

        inputs = self.processor(text=task, images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=inputs["input_ids"],
                                       pixel_values=inputs["pixel_values"],
                                       max_new_tokens=1024, num_beams=3)

        text = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

        print(f"[VLM] Detection: {parsed}")

        if "<OD>" in parsed:
            data = parsed["<OD>"]
            for i, label in enumerate(data.get("labels", [])):
                if obj in label.lower():
                    bbox = data["bboxes"][i]
                    cx = (bbox[0] + bbox[2]) / 2
                    x_norm = (cx / w) * 2 - 1
                    area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) / (w*h)
                    dist = max(0.1, 1.0 - area * 5)
                    return {"found": True, "x": x_norm, "y": dist, "confidence": 0.7,
                            "color": color, "object": obj, "bbox": list(bbox), "label": label}

        return {"found": False, "x": 0, "y": 0.5, "confidence": 0, "color": color, "object": obj}

    def warmup(self):
        print("[VLM] Warmup...")
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.ground_object(dummy, "red box")
        print("[VLM] Warmup done!")


class NavigationController:
    def __init__(self, max_v=1.0, max_w=1.0, thresh=0.15):
        self.max_v, self.max_w, self.thresh = max_v, max_w, thresh

    def target_to_velocity(self, t: Dict) -> np.ndarray:
        if not t.get("found"):
            return np.array([0, 0, 0.3])
        x, y = t["x"], t["y"]
        if y < self.thresh and abs(x) < 0.2:
            return np.array([0, 0, 0])
        w = -x * self.max_w
        v = min(0.3 + y*0.7, 1) * self.max_v * (0.5 if abs(x) > 0.5 else 1)
        return np.array([v, 0, w])


if __name__ == "__main__":
    import time

    print("="*60)
    print("Florence-2 Grounding Test")
    print("="*60)

    args = sys.argv[1:]

    if len(args) >= 1:
        path = args[0]
        cmd = args[1] if len(args) > 1 else "blue chair"
        img = np.array(Image.open(path).convert("RGB"))
        print(f"\n[Test] Image: {path}, shape: {img.shape}")
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[200:350, 150:300] = [50, 100, 200]
        cmd = "blue box"
        print("\n[Test] Using dummy image")

    print("\n[Test] Loading model...")
    vlm = VLMWrapper()
    vlm.warmup()

    print(f"\n[Test] Command: '{cmd}'")
    result = vlm.ground_object(img, cmd)

    print(f"\n{'='*60}")
    print("RESULT")
    print("="*60)
    print(f"  Found: {result['found']}")
    if result['found']:
        print(f"  Position: x={result['x']:.2f}, y={result['y']:.2f}")
        print(f"  BBox: {result.get('bbox')}")

    nav = NavigationController()
    vel = nav.target_to_velocity(result)
    print(f"  Velocity: vx={vel[0]:.2f}, vyaw={vel[2]:.2f}")

    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)