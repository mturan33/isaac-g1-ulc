"""
ULC G1 Training Scripts
=======================
Stage-based training for Unified Loco-Manipulation Controller.
"""

# Environment envs klasöründen import
# Not: train klasöründen envs'e relative import
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import gymnasium as gym

# Register environment
gym.register(
    id="Isaac-G1-Dex1-Stage6-v0",
    entry_point="envs.g1_locomanip_env:G1Dex1Stage6Env",
    kwargs={"cfg": G1Dex1Stage6EnvCfg()},
    disable_env_checker=True,
)

__all__ = [
    "G1Dex1Stage6Env",
    "G1Dex1Stage6EnvCfg",
    "CURRICULUM_LEVELS",
]