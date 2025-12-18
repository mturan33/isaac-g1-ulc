"""
VLM Wrapper for Phi-3-Vision
Language-Conditioned Object Grounding
"""

import torch
import json
import re
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict


class VLMWrapper:
    """
    Phi-3-Vision wrapper for object grounding.

    Usage:
        vlm = VLMWrapper()
        target = vlm.ground_object(rgb_image, "mavi sandalye")
        # target = {"found": True, "x": 0.3, "y": 0.7, "confidence": 0.92}
    """

    GROUNDING_PROMPT = """<|user|>
<|image_1|>
You are a robot navigation assistant. The robot is at the bottom center of the image looking forward.

Task: Find the {color} {object} in the image.

Respond ONLY with this exact JSON format, nothing else:
{{"found": true, "x": 0.0, "y": 0.5, "confidence": 0.9}}

Where:
- found: true if object is visible, false otherwise
- x: horizontal position (-1.0=far left, 0.0=center, 1.0=far right)
- y: depth/distance (0.0=very close, 0.5=medium, 1.0=far away)
- confidence: 0.0 to 1.0

If object not found: {{"found": false, "x": 0.0, "y": 0.0, "confidence": 0.0}}
<|end|>
<|assistant|>"""

    # Türkçe → İngilizce çeviri
    COLOR_MAP = {
        "mavi": "blue", "blue": "blue",
        "kırmızı": "red", "red": "red",
        "yeşil": "green", "green": "green",
        "sarı": "yellow", "yellow": "yellow",
        "turuncu": "orange", "orange": "orange",
        "mor": "purple", "purple": "purple",
        "beyaz": "white", "white": "white",
        "siyah": "black", "black": "black",
        "kahverengi": "brown", "brown": "brown",
        "pembe": "pink", "pink": "pink",
        "gri": "gray", "gray": "gray", "grey": "gray",
    }

    OBJECT_MAP = {
        "sandalye": "chair", "chair": "chair",
        "masa": "table", "table": "table",
        "dolap": "cabinet", "cabinet": "cabinet",
        "koltuk": "sofa", "sofa": "sofa", "couch": "sofa",
        "kutu": "box", "box": "box", "cube": "box",
        "top": "ball", "ball": "ball",
        "silindir": "cylinder", "cylinder": "cylinder",
        "koni": "cone", "cone": "cone",
    }

    def __init__(
            self,
            model_id: str = "microsoft/Phi-3-vision-128k-instruct",
            device: str = "cuda",
            dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize VLM wrapper.

        Args:
            model_id: HuggingFace model ID
            device: "cuda" or "cpu"
            dtype: torch.float16 for efficiency
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"[VLM] Loading {model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )

        self.device = device
        print(f"[VLM] Model loaded on {device}")

    def parse_command(self, command: str) -> Tuple[str, str]:
        """
        Parse natural language command to extract color and object.

        Args:
            command: "mavi sandalyeye git" or "go to the blue chair"

        Returns:
            (color, object): ("blue", "chair")
        """
        command_lower = command.lower()

        # Türkçe ekleri temizle
        command_clean = command_lower
        for suffix in ["'e", "'a", "ye", "ya", "'ye", "'ya", "e git", "a git"]:
            command_clean = command_clean.replace(suffix, "")

        color = "unknown"
        obj = "object"

        # Renk bul
        for tr, en in self.COLOR_MAP.items():
            if tr in command_clean:
                color = en
                break

        # Obje bul
        for tr, en in self.OBJECT_MAP.items():
            if tr in command_clean:
                obj = en
                break

        return color, obj

    def ground_object(
            self,
            image: np.ndarray,
            command: str,
            max_new_tokens: int = 100,
    ) -> Dict:
        """
        Find object in image based on language command.

        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image
            command: Natural language command
            max_new_tokens: Max tokens to generate

        Returns:
            {
                "found": bool,
                "x": float (-1 to 1),
                "y": float (0 to 1),
                "confidence": float (0 to 1),
                "color": str,
                "object": str
            }
        """
        # Parse command
        color, obj = self.parse_command(command)

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Build prompt
        prompt = self.GROUNDING_PROMPT.format(color=color, object=obj)

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=[pil_image],
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON
        result = self._extract_json(response)
        result["color"] = color
        result["object"] = obj

        return result

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from VLM response."""
        # Find JSON block
        json_patterns = [
            r'\{[^{}]*"found"[^{}]*\}',
            r'\{[^{}]*found[^{}]*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Clean up common issues
                    json_str = match.group()
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)

                    result = json.loads(json_str)
                    return {
                        "found": bool(result.get("found", False)),
                        "x": float(result.get("x", 0.0)),
                        "y": float(result.get("y", 0.5)),
                        "confidence": float(result.get("confidence", 0.0)),
                    }
                except (json.JSONDecodeError, ValueError):
                    continue

        # Fallback
        return {"found": False, "x": 0.0, "y": 0.5, "confidence": 0.0}


class NavigationController:
    """
    Convert VLM targets to robot velocity commands.
    """

    def __init__(
            self,
            max_linear_vel: float = 1.0,
            max_angular_vel: float = 1.0,
            goal_threshold: float = 0.15,
    ):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.goal_threshold = goal_threshold

    def target_to_velocity(self, target: Dict) -> np.ndarray:
        """
        Convert VLM target to velocity command.

        Args:
            target: VLM output with x, y, found

        Returns:
            cmd_vel: [vx, vy, vyaw] velocity command
        """
        if not target.get("found", False):
            # Object not found - stop or search
            return np.array([0.0, 0.0, 0.3])  # Slow rotation to search

        x_offset = target["x"]  # -1 (left) to 1 (right)
        y_distance = target["y"]  # 0 (close) to 1 (far)

        # Check if reached goal
        if y_distance < self.goal_threshold and abs(x_offset) < 0.2:
            return np.array([0.0, 0.0, 0.0])  # Stop - reached goal

        # Angular velocity: Turn towards object
        vyaw = -x_offset * self.max_angular_vel

        # Linear velocity: Move forward based on distance
        # Slow down when close
        if y_distance < 0.3:
            vx = y_distance * self.max_linear_vel
        else:
            vx = min(0.3 + y_distance * 0.7, 1.0) * self.max_linear_vel

        # Reduce forward speed while turning sharply
        if abs(x_offset) > 0.5:
            vx *= 0.5

        # No lateral movement for now
        vy = 0.0

        return np.array([vx, vy, vyaw])

    def is_goal_reached(self, target: Dict) -> bool:
        """Check if robot reached the target object."""
        if not target.get("found", False):
            return False
        return target["y"] < self.goal_threshold and abs(target["x"]) < 0.2


# ============== Standalone Test ==============
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("VLM Wrapper Test")
    print("=" * 50)

    # Check if image path provided
    if len(sys.argv) < 2:
        print("Usage: python vlm_wrapper.py <image_path> [command]")
        print("Example: python vlm_wrapper.py scene.png 'mavi sandalyeye git'")
        sys.exit(1)

    image_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else "mavi sandalyeye git"

    # Load image
    print(f"\n[Test] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    print(f"[Test] Image shape: {image_np.shape}")

    # Initialize VLM
    print("\n[Test] Initializing VLM...")
    vlm = VLMWrapper()

    # Ground object
    print(f"\n[Test] Command: '{command}'")
    result = vlm.ground_object(image_np, command)

    print("\n[Test] Result:")
    print(f"  Found: {result['found']}")
    print(f"  Position: x={result['x']:.2f}, y={result['y']:.2f}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Parsed: {result['color']} {result['object']}")

    # Convert to velocity
    nav = NavigationController()
    cmd_vel = nav.target_to_velocity(result)
    print(f"\n[Test] Velocity command: vx={cmd_vel[0]:.2f}, vy={cmd_vel[1]:.2f}, vyaw={cmd_vel[2]:.2f}")

    print("\n" + "=" * 50)
    print("Test completed!")