"""
ULC G1 Play/Test Scripts
========================

Visualization and testing scripts for trained ULC policies.

Scripts:
    - play_ulc_stage_1.py: Test standing policy
    - play_ulc_stage_2.py: Test locomotion policy with velocity commands

Usage:
    # Play Stage 1
    ./isaaclab.bat -p .../play/play_ulc_stage_1.py --checkpoint <model_best.pt> --num_envs 4

    # Play Stage 2
    ./isaaclab.bat -p .../play/play_ulc_stage_2.py --checkpoint <model_best.pt> --num_envs 4
"""

# Play scripts are standalone executables, not importable modules
# This __init__.py exists for package structure

__all__ = []