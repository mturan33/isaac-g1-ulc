# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Sim state DDS communication class
Specialized in publishing and receiving sim state data
"""

import threading
import time
import torch
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

import json

class RewardsDDS(DDSObject):
    """Sim state DDS node (singleton pattern)"""
    
    def __init__(self, env, task_name,node_name:str="rewards_dds"):
        """Initialize the sim state DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__()
        self.node_name = node_name
        self.env = env
        self.task_name = task_name
        self._initialized = True
        self.rewards = std_msgs_msg_dds__String_()
        self.last_published_timestamp = None  # 记录最后发布的时间戳

        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_rewards",  # read rewards data for publishing
            input_size=256,
            outputshm_flag=False
        )

        print(f"[{self.node_name}] Rewards DDS node initialized")

    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the rewards"""
        try:
            self.publisher = ChannelPublisher("rt/rewards_state", String_)
            self.publisher.Init()
            
            print(f"[{self.node_name}] Rewards publisher initialized")
            return True
        except Exception as e:
            print(f"rewards_dds [{self.node_name}] Rewards publisher initialization failed: {e}")
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the rewards"""
        try:
            self.subscriber = ChannelSubscriber("rt/rewards_state_cmd", String_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 1)
            
            print(f"[{self.node_name}] Rewards subscriber initialized")
            return True
        except Exception as e:
            print(f"rewards_dds [{self.node_name}] Rewards subscriber initialization failed: {e}")
            return False
    
    
    def dds_publisher(self) -> Any:
        """Process the publish data"""
        try:
            data = self.input_shm.read_data()
            
            if data is None:
                return
            
            # 检查时间戳，避免重复发布
            current_timestamp = data.get("timestamp", None)
            if current_timestamp is not None:
                if self.last_published_timestamp == current_timestamp:
                    # 相同时间戳的数据已经发布过，跳过
                    return
                # 更新最后发布的时间戳
                self.last_published_timestamp = current_timestamp
            
            # get rewards from environment
            rewards = json.dumps(data)
            self.rewards.data = rewards
            self.publisher.Write(self.rewards)
        except Exception as e:
            print(f"rewards_dds [{self.node_name}] Error processing publish data: {e}")
            return None
    
    def dds_subscriber(self, msg: String_,datatype:str=None) -> Dict[str, Any]:
        """Process the subscribe data"""
        try:
            # Parse received rewards command
            data = json.loads(msg.data)
            
            # Process the command (implement according to your needs)
            # For example, you might want to apply the received state to the environment
            return data
        except Exception as e:
            print(f"rewards_dds [{self.node_name}] Error processing subscribe data: {e}")
            return None

    def tensors_to_list(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensors_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensors_to_list(i) for i in obj]
        return obj

    def rewards_to_json(self,data):
        data_serializable = self.tensors_to_list(data)
        json_str = json.dumps(data_serializable)
        return json_str

    def write_rewards_data(self, rewards_data=None):
        """Write rewards data to shared memory to trigger publishing
        
        Args:
            rewards_data: Optional rewards data. If None, will get current state from environment
        """
        try:
            if rewards_data is None:
                print(f"rewards_dds [{self.node_name}] Warning: rewards_data is None")
                return
            
            # Convert tensor to appropriate format and wrap in dictionary
            if isinstance(rewards_data, torch.Tensor):
                rewards_list = rewards_data.cpu().numpy().tolist()
                rewards_dict = {
                    "rewards": rewards_list,
                    "timestamp": time.time()
                }
            else:
                # If it's already a scalar or list, wrap it in a dictionary
                rewards_dict = {
                    "rewards": rewards_data if isinstance(rewards_data, list) else [rewards_data],
                    "timestamp": time.time()
                }
            self.input_shm.write_data(rewards_dict)
                
        except Exception as e:
            print(f"rewards_dds [{self.node_name}] Error writing rewards data: {e}")

    def get_rewards_command(self) -> Optional[Dict[str, Any]]:
        """Get the rewards control command
        
        Returns:
            Dict: the rewards command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None