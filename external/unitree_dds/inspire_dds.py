# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Gripper DDS communication class
Handle the state publishing and command receiving of the gripper
"""

import threading
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_go_msg_dds__MotorState_
import numpy as np

class InspireDDS(DDSObject):
    """Gripper DDS communication class - singleton pattern
    
    Features:
    - Publish the state of the gripper to DDS (rt/unitree_actuator/state)
    - Receive the control command of the gripper (rt/unitree_actuator/cmd)
    """
    
    def __init__(self,node_name:str="inspire"):
        """Initialize the gripper DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
            
        super().__init__()
        self.node_name = node_name
        
        # initialize the gripper state message (2 grippers)
        self.inspire_hand_state = MotorStates_()
        # initialize 2 motor states
        self.inspire_hand_state.states = []
        for _ in range(12):
            motor_state = unitree_go_msg_dds__MotorState_()
            self.inspire_hand_state.states.append(motor_state)
        
        self._initialized = True
        
        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_inspire_state",  # read the state of the gripper from Isaac Lab
            input_size=1024,
            output_shm_name="isaac_inspire_cmd",  # output the command to Isaac Lab
            output_size=1024,  # output the command to Isaac Lab
        )
        
        print(f"[{self.node_name}] Inspire Hand DDS node initialized")
    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the gripper"""
        try:
            self.publisher = ChannelPublisher("rt/inspire/state", MotorStates_)
            self.publisher.Init()
            
            print(f"[{self.node_name}] Inspire Hand state publisher initialized")
            return True
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Gripper state publisher initialization failed: {e}")
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the gripper"""
        try:
            self.subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 32)
            
            print(f"[{self.node_name}] Inspire Hand command subscriber initialized")
            return True
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Gripper command subscriber initialization failed: {e}")
            return False
    def normalize(self,val, min_val, max_val):
        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)
    def dds_publisher(self) -> Any:
        """Process the publish data: convert the Isaac Lab state to the DDS message
        
        Expected data format:
        {
            "positions": [2 gripper joint positions] (Isaac Lab joint angle range [-0.02, 0.03])
            "velocities": [2 gripper joint velocities],
            "torques": [2 gripper joint torques]
        }
        """
        try:
            data = self.input_shm.read_data() 
            if data is None:
                return
            if all(key in data for key in ["positions", "velocities", "torques"]):
                positions = data["positions"]
                velocities = data["velocities"]
                torques = data["torques"]
                for i in range(min(12, len(positions))):
                    if i < len(self.inspire_hand_state.states):
                        # convert the Isaac Lab joint angle to the gripper control value    
                        if i in [0,1,2,3,6,7,8,9]:
                            inspire_q_value = self.normalize(float(positions[i]),0.0,1.7)
                        elif i in [4,10]:
                            inspire_q_value = self.normalize(float(positions[i]),0.0,0.5)
                        elif i in [5,11]:
                            inspire_q_value = self.normalize(float(positions[i]),-0.1,1.3)
                        self.inspire_hand_state.states[i].q = inspire_q_value
                        if i < len(velocities):
                            self.inspire_hand_state.states[i].dq = float(velocities[i])
                        if i < len(torques):
                            self.inspire_hand_state.states[i].tau_est = float(torques[i])
            
                self.publisher.Write(self.inspire_hand_state)
            
        except Exception as e:
            print(f"inspire_dds [{self.node_name}] Error processing publish data: {e}")    
            return None
    def denormalize(self,norm_val, min_val, max_val):
        return (1.0 - np.clip(norm_val, 0.0, 1.0)) * (max_val - min_val) + min_val
    def dds_subscriber(self, msg: MotorCmds_,datatype:str=None) -> Dict[str, Any]:
        """Process the subscribe data: convert the DDS command to the Isaac Lab format
        
        Returns:
            Dict: the gripper command, format as follows:
            {
                "positions": [2 gripper joint position target values] (Isaac Lab joint angle)
                "velocities": [2 gripper joint velocity target values],
                "torques": [2 gripper joint torque target values],
                "kp": [2 gripper position gains],
                "kd": [2 gripper speed gains]
            }
        """
        try:
            cmd_data = {
                "positions": [],
                "velocities": [],
                "torques": [],
                "kp": [],
                "kd": []
            }
            # process the gripper command (at most 2 grippers)
            for i in range(min(12, len(msg.cmds))):
                # convert the gripper control value to the Isaac Lab joint angle
                if i in [0,1,2,3,6,7,8,9]:
                    joint_angle = self.denormalize(float(msg.cmds[i].q),0.0,1.7)
                elif i in [4,10]:
                    joint_angle = self.denormalize(float(msg.cmds[i].q),0.0,0.5)
                elif i in [5,11]:
                    joint_angle = self.denormalize(float(msg.cmds[i].q),-0.1,1.3)
                cmd_data["positions"].append(joint_angle)
                cmd_data["velocities"].append(float(msg.cmds[i].dq))
                cmd_data["torques"].append(float(msg.cmds[i].tau))
                cmd_data["kp"].append(float(msg.cmds[i].kp))
                cmd_data["kd"].append(float(msg.cmds[i].kd))
            self.output_shm.write_data(cmd_data)
            
        except Exception as e:
            print(f"inspire_dds [{self.node_name}] Error processing subscribe data: {e}")
            return None
    
    def get_inspire_hand_command(self) -> Optional[Dict[str, Any]]:
        """Get the gripper control command
        
        Returns:
            Dict: the gripper command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_inspire_state(self, positions, velocities, torques):
        """Write the gripper state to the shared memory
        
        Args:
            positions: the gripper joint position list or torch.Tensor (Isaac Lab joint angle)
            velocities: the gripper joint velocity list or torch.Tensor  
            torques: the gripper joint torque list or torch.Tensor
        """
        try:
            # prepare the gripper data
            inspire_hand_data = {
                "positions": positions.tolist() if hasattr(positions, 'tolist') else positions,
                "velocities": velocities.tolist() if hasattr(velocities, 'tolist') else velocities,
                "torques": torques.tolist() if hasattr(torques, 'tolist') else torques
            }
            
            # write the input shared memory for publishing
            if self.input_shm:
                self.input_shm.write_data(inspire_hand_data)
                
        except Exception as e:
            print(f"gripper_dds [{self.node_name}] Error writing inspire hand state: {e}")