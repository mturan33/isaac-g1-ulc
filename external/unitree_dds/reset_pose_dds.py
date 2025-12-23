# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Reset pose DDS communication class
Specialized in receiving the reset pose command
"""

import threading
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


class ResetPoseCmdDDS(DDSObject):
    """Reset pose command DDS node (singleton pattern)"""
    
    
    def __init__(self,node_name:str="reset_pose_dds"):
        """Initialize the reset pose DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        
        self._initialized = True
        self.node_name = node_name
        # setup the shared memory
        self.setup_shared_memory(
            output_shm_name="isaac_reset_pose_cmd", 
            output_size=512, 
            outputshm_flag=True,
            inputshm_flag=False,
        )
        print(f"[{self.node_name}] Reset pose DDS node initialized")

    
    def setup_publisher(self) -> bool:
        """Setup the reset pose command publisher (this node is mainly used for subscribe, the publisher is optional)"""
        pass
    
    def setup_subscriber(self) -> bool:
        """Setup the reset pose command subscriber"""
        try:
            self.subscriber = ChannelSubscriber("rt/reset_pose/cmd", String_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 1)
            
            print(f"[{self.node_name}] Reset pose command subscriber initialized")
            return True
        except Exception as e:
            print(f"reset_pose_dds [{self.node_name}] Failed to initialize the reset pose command subscriber: {e}")
            return False
    
    
    def dds_publisher(self) -> Any:
        """Process the publish data (this node is mainly used for subscribe, the publish function is optional)"""
        pass
    
    def dds_subscriber(self, msg: String_,datatype:str=None) -> Dict[str, Any]:
        """Process the subscribe data"""
        try:
            cmd_data = {
                "reset_category": msg.data
            }
            self.output_shm.write_data(cmd_data)
        except Exception as e:
            print(f"reset_pose_dds [{self.node_name}] Failed to process the subscribe data: {e}")
            return {}
    
    def get_reset_pose_command(self) -> Optional[Dict[str, Any]]:
        """Get the reset pose command
        
        Returns:
            Dict: the reset pose command, if no command return None
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_reset_pose_command(self, flag_category):
        """Write the reset pose command to the shared memory
        
        Args:
            positions: the reset pose command, if no command return None
        """
        try:
            # prepare the reset pose data
            cmd_data = {
                "reset_category":flag_category
            }
            
            # write the reset pose data to the shared memory
            if self.output_shm:
                self.output_shm.write_data(cmd_data)
                
        except Exception as e:
            print(f"reset_pose_dds [{self.node_name}] Failed to write the reset pose command: {e}")