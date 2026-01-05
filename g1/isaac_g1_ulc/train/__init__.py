"""
ULC G1 Training Scripts
=======================

Stage-based training for Unified Loco-Manipulation Controller.

Stages:
    - Stage 1: Standing (balance, height tracking)
    - Stage 2: Locomotion (walking, rough terrain, perturbation)
    - Stage 3: Torso Control (upper body movement during locomotion)
    - Stage 4: Arm Manipulation (reaching, grasping with whole-body control)

Usage:
    # Stage 1: Standing
    ./isaaclab.bat -p .../train/train_ulc_stage_1.py --num_envs 4096 --headless

    # Stage 2: Locomotion (from Stage 1 checkpoint)
    ./isaaclab.bat -p .../train/train_ulc_stage_2.py --num_envs 4096 --headless --stage1_checkpoint <path>
"""

# Training scripts are standalone executables, not importable modules
# This __init__.py exists for package structure

__all__ = []