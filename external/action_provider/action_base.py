# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# Author: wei.li
# License: Apache License, Version 2.0

from abc import ABC, abstractmethod
from typing import Optional
import torch
import time
import threading

class ActionProvider(ABC):
    """Abstract base class for action providers"""
    
    def __init__(self, name: str):
        print(f"ActionProvider init")
        self.name = name
        self.is_running = False
        self._thread = None
    
    @abstractmethod
    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action
        
        Args:
            env: environment instance
            
        Returns:
            torch.Tensor: action tensor, return None if no action is available
        """
        pass
    
    def start(self):
        """Start action provider"""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            print(f"[{self.name}] ActionProvider started")
    
    def stop(self):
        """Stop action provider"""
        print(f"[{self.name}] ActionProvider stop")
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print(f"[{self.name}] ActionProvider stopped")
    
    def _run_loop(self):
        """Run loop (subclass can override)"""
        while self.is_running:
            time.sleep(0.01)
    
    def cleanup(self):
        """Clean up resources (subclass can override)"""
        pass