# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Dex3 DDS communication class
Handle the state publishing and command receiving of the hand (left and right)
"""

import threading
from typing import Any, Dict, Optional, Tuple
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, HandCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_, unitree_hg_msg_dds__HandCmd_


class Dex3DDS(DDSObject):
    """Hand DDS communication class - singleton pattern
    
    Features:
    - Publish the state of the hand to DDS (rt/dex3/left/state, rt/dex3/right/state)
    - Receive the control command of the hand (rt/dex3/left/cmd, rt/dex3/right/cmd)
    """
    
    def __init__(self,node_name:str="dex3"):
        """Initialize the hand DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
            
        super().__init__()
        self.node_name = node_name
        
        # initialize the state message of the hand
        self.left_hand_state = unitree_hg_msg_dds__HandState_()
        self.right_hand_state = unitree_hg_msg_dds__HandState_()
        
        # initialize the publisher and subscriber
        self.left_state_publisher = None
        self.right_state_publisher = None
        self.left_cmd_subscriber = None
        self.right_cmd_subscriber = None
        
        self._initialized = True
        self.existing_data = {"left_hand_cmd": {}, "right_hand_cmd": {}}
        # setup shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_dex3_state",  # read the state of the hand from Isaac Lab
            input_size=1180,
            output_shm_name="isaac_dex3_cmd",  # output the command to Isaac Lab
            output_size=1180,  # output the command to Isaac Lab
        )
        
        print(f"[{self.node_name}] Hand DDS node initialized")
    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the hand"""
        try:
            # left hand state publisher
            self.left_state_publisher = ChannelPublisher("rt/dex3/left/state", HandState_)
            self.left_state_publisher.Init()
            
            # right hand state publisher
            self.right_state_publisher = ChannelPublisher("rt/dex3/right/state", HandState_)
            self.right_state_publisher.Init()
            
            print(f"[{self.node_name}] Hand state publisher initialized")
            return True
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Hand state publisher initialization failed: {e}")
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the hand"""
        try:
            # left hand command subscriber
            self.left_cmd_subscriber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
            self.left_cmd_subscriber.Init(
                lambda msg: self.dds_subscriber(msg, "left"), 32
            )
            
            # right hand command subscriber
            self.right_cmd_subscriber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
            self.right_cmd_subscriber.Init(
                lambda msg: self.dds_subscriber(msg, "right"), 32
            )
            
            print(f"[{self.node_name}] Hand command subscriber initialized")
            return True
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Hand command subscriber initialization failed: {e}")
            return False
    
    def dds_subscriber(self, msg: HandCmd_, datatype:str=None):
        """Handle the command of the hand"""
        try:
            # process the command of the hand and write to the shared memory
            cmd_data = self.process_hand_command(msg, datatype)
            if cmd_data and self.output_shm:
                # write to shared memory
                self.existing_data[f"{datatype}_hand_cmd"] = cmd_data
                self.output_shm.write_data(self.existing_data)
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error handling {datatype} hand command: {e}")
    
    def process_hand_command(self, msg: HandCmd_, datatype:str=None) -> Dict[str, Any]:
        """Process the command of the hand"""
        try:
            cmd_data = {
                "positions": [float(msg.motor_cmd[i].q) for i in range(len(msg.motor_cmd))],
                "velocities": [float(msg.motor_cmd[i].dq) for i in range(len(msg.motor_cmd))],
                "torques": [float(msg.motor_cmd[i].tau) for i in range(len(msg.motor_cmd))],
                "kp": [float(msg.motor_cmd[i].kp) for i in range(len(msg.motor_cmd))],
                "kd": [float(msg.motor_cmd[i].kd) for i in range(len(msg.motor_cmd))]
            }
            return cmd_data
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error processing {datatype} hand command data: {e}")
            return {}
    
    def dds_publisher(self) -> Any:
        """Process the publish data: convert the hand state of Isaac Lab to DDS message
        
        Expected data format:
        {
            "left_hand": {
                "positions": [7 left hand joint positions],
                "velocities": [7 left hand joint velocities],
                "torques": [7 left hand joint torques]
            },
            "right_hand": {
                "positions": [7 right hand joint positions],
                "velocities": [7 right hand joint velocities],
                "torques": [7 right hand joint torques]
            }
        }
        """
        try:
            data = self.input_shm.read_data() or {}
            
            # process the left hand data
            if "left_hand" in data:
                left_data = data["left_hand"]
                self._update_hand_state(self.left_hand_state, left_data)
                if self.left_state_publisher:
                    self.left_state_publisher.Write(self.left_hand_state)
            
            # process the right hand data
            if "right_hand" in data:
                right_data = data["right_hand"]
                self._update_hand_state(self.right_hand_state, right_data)
                if self.right_state_publisher:
                    self.right_state_publisher.Write(self.right_hand_state)
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error processing publish data: {e}")
            return None
    
    def _update_hand_state(self, hand_state, hand_data: Dict[str, Any]):
        """Update the hand state"""
        try:
            if all(key in hand_data for key in ["positions", "velocities", "torques"]):
                positions = hand_data["positions"]
                velocities = hand_data["velocities"]
                torques = hand_data["torques"]
                
                for i in range(min(7, len(positions))):  # at most 7 fingers
                    if i < len(positions):
                        hand_state.motor_state[i].q = float(positions[i])
                    if i < len(velocities):
                        hand_state.motor_state[i].dq = float(velocities[i])
                    if i < len(torques):
                        hand_state.motor_state[i].tau_est = float(torques[i])
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error updating hand state: {e}")
    

    
    def get_hand_commands(self) -> Optional[Dict[str, Any]]:
        """Get the hand control commands
        
        Returns:
            Dict: the dictionary containing the commands of the left and right hands, the format is as follows:
            {
                "left_hand_cmd": {left hand command},
                "right_hand_cmd": {right hand command}
            }
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def get_left_hand_command(self) -> Optional[Dict[str, Any]]:
        """Get the left hand command"""
        commands = self.get_hand_commands()
        if commands and "left_hand_cmd" in commands:
            return commands["left_hand_cmd"]
        return None
    
    def get_right_hand_command(self) -> Optional[Dict[str, Any]]:
        """Get the right hand command"""
        commands = self.get_hand_commands()
        if commands and "right_hand_cmd" in commands:
            return commands["right_hand_cmd"]
        return None
    
    def publish_hand_states(self, left_hand_data: Dict[str, Any], right_hand_data: Dict[str, Any]):
        """Publish the left and right hand states
        
        Args:
            left_hand_data: the data of the left hand
            right_hand_data: the data of the right hand
        """
        try:
            combined_data = {
                "left_hand": left_hand_data,
                "right_hand": right_hand_data
            }
            
            # write to the input shared memory for publishing
            if self.input_shm:
                self.input_shm.write_data(combined_data)
                
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error publishing hand states: {e}")
    
    def write_hand_states(self, left_positions, left_velocities, left_torques, 
                         right_positions, right_velocities, right_torques):
        """Write the hand states to the shared memory directly
        
        Args:
            left_positions: the list or torch.Tensor of the left hand joint positions
            left_velocities: the list or torch.Tensor of the left hand joint velocities
            left_torques: the list or torch.Tensor of the left hand joint torques
            right_positions: the list or torch.Tensor of the right hand joint positions
            right_velocities: the list or torch.Tensor of the right hand joint velocities
            right_torques: the list or torch.Tensor of the right hand joint torques
        """
        try:
            # prepare the left hand data
            left_hand_data = {
                "positions": left_positions.tolist() if hasattr(left_positions, 'tolist') else left_positions,
                "velocities": left_velocities.tolist() if hasattr(left_velocities, 'tolist') else left_velocities,
                "torques": left_torques.tolist() if hasattr(left_torques, 'tolist') else left_torques
            }
            
            # prepare the right hand data
            right_hand_data = {
                "positions": right_positions.tolist() if hasattr(right_positions, 'tolist') else right_positions,
                "velocities": right_velocities.tolist() if hasattr(right_velocities, 'tolist') else right_velocities,
                "torques": right_torques.tolist() if hasattr(right_torques, 'tolist') else right_torques
            }
            
            # publish the states
            self.publish_hand_states(left_hand_data, right_hand_data)
            
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error writing hand states: {e}")
    
    def write_single_hand_state(self, hand_side: str, positions, velocities, torques):
        """Write the single hand state
        
        Args:
            hand_side: the side of the hand ("left" or "right")
            positions: the list or torch.Tensor of the hand joint positions
            velocities: the list or torch.Tensor of the hand joint velocities
            torques: the list or torch.Tensor of the hand joint torques
        """
        try:
            hand_data = {
                "positions": positions.tolist() if hasattr(positions, 'tolist') else positions,
                "velocities": velocities.tolist() if hasattr(velocities, 'tolist') else velocities,
                "torques": torques.tolist() if hasattr(torques, 'tolist') else torques
            }
            
            # decide how to publish based on the hand side
            if hand_side == "left":
                # get the existing right hand data or use the default value
                existing_data = self.input_shm.read_data() if self.input_shm else {}
                right_data = existing_data.get("right_hand", {"positions": [0], "velocities": [0], "torques": [0]})
                self.publish_hand_states(hand_data, right_data)
            elif hand_side == "right":
                # get the existing left hand data or use the default value
                existing_data = self.input_shm.read_data() if self.input_shm else {}
                left_data = existing_data.get("left_hand", {"positions": [0], "velocities": [0], "torques": [0]})
                self.publish_hand_states(left_data, hand_data)
            else:
                print(f"dex3_dds [{self.node_name}] Invalid hand side: {hand_side}")
                
        except Exception as e:
            print(f"dex3_dds [{self.node_name}] Error writing {hand_side} hand state: {e}")
    
