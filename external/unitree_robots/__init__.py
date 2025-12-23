"""
external/unitree_robots/__init__.py
G1/H1 robot configurations
Source: https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/main/robots
"""

import os
import sys

# Klasordeki tum .py dosyalarini otomatik import et
_current_dir = os.path.dirname(os.path.abspath(__file__))
_py_files = [f[:-3] for f in os.listdir(_current_dir) 
             if f.endswith('.py') and f != '__init__.py']

# Dynamic import
for _module in _py_files:
    try:
        exec(f"from .{_module} import *")
    except ImportError as e:
        print(f"Warning: Could not import {_module}: {e}")

__all__ = _py_files
