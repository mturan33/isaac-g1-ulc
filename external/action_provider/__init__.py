"""
external/action_provider/__init__.py
Action providers for robot control
Source: https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/main/action_provider
"""

import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_py_files = [f[:-3] for f in os.listdir(_current_dir) 
             if f.endswith('.py') and f != '__init__.py']

for _module in _py_files:
    try:
        exec(f"from .{_module} import *")
    except ImportError as e:
        pass  # Silent fail for optional deps

__all__ = _py_files
