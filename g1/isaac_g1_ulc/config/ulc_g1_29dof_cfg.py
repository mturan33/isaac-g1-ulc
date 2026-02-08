"""
G1-29DoF + DEX3 Configuration
==============================
Joint definitions, actuator parameters, and default poses for the
Unitree G1-29DoF robot with DEX3 3-finger dexterous hands.

Based on: G129_CFG_WITH_DEX3_WHOLEBODY from unitree.py
USD: unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd

Total: 43 joints = 12 legs + 3 waist + 14 arms (7×2) + 14 fingers (7×2)
"""

import os

# ============================================================================
# USD PATH
# ============================================================================

# Relative path from this file to the USD asset
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
G1_29DOF_USD = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "..", "..",
    "unitree_sim_isaaclab", "assets", "robots",
    "g1-29dof_wholebody_dex3", "g1_29dof_with_dex3_rev_1_0.usd"
))

# ============================================================================
# JOINT NAMES - ordered by actuator group
# ============================================================================

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]  # 12 joints

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]  # 3 joints

# Loco controls legs + waist = 15 joints
LOCO_JOINT_NAMES = LEG_JOINT_NAMES + WAIST_JOINT_NAMES

ARM_JOINT_NAMES_LEFT = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]  # 7 joints

ARM_JOINT_NAMES_RIGHT = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # 7 joints

ARM_JOINT_NAMES = ARM_JOINT_NAMES_LEFT + ARM_JOINT_NAMES_RIGHT  # 14 joints

HAND_JOINT_NAMES_LEFT = [
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
]  # 7 joints

HAND_JOINT_NAMES_RIGHT = [
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]  # 7 joints

HAND_JOINT_NAMES = HAND_JOINT_NAMES_LEFT + HAND_JOINT_NAMES_RIGHT  # 14 joints

ALL_JOINT_NAMES = LOCO_JOINT_NAMES + ARM_JOINT_NAMES + HAND_JOINT_NAMES  # 43 joints

# ============================================================================
# JOINT COUNTS
# ============================================================================

NUM_LEG_JOINTS = len(LEG_JOINT_NAMES)      # 12
NUM_WAIST_JOINTS = len(WAIST_JOINT_NAMES)  # 3
NUM_LOCO_JOINTS = NUM_LEG_JOINTS + NUM_WAIST_JOINTS  # 15
NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)      # 14 (7 per arm)
NUM_HAND_JOINTS = len(HAND_JOINT_NAMES)    # 14 (7 per hand)
NUM_ALL_JOINTS = NUM_LOCO_JOINTS + NUM_ARM_JOINTS + NUM_HAND_JOINTS  # 43

# ============================================================================
# DEFAULT POSES (from G129_CFG_WITH_DEX3_WHOLEBODY init_state)
# ============================================================================

# Leg defaults - slight knee bend for stability
DEFAULT_LEG_POSES = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}

# Waist defaults - upright
DEFAULT_WAIST_POSES = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}

# Arm defaults - natural resting pose
DEFAULT_ARM_POSES = {
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Hand defaults - open
DEFAULT_HAND_POSES = {name: 0.0 for name in HAND_JOINT_NAMES}

# Combined default poses
DEFAULT_LOCO_POSES = {**DEFAULT_LEG_POSES, **DEFAULT_WAIST_POSES}
DEFAULT_ALL_POSES = {**DEFAULT_LOCO_POSES, **DEFAULT_ARM_POSES, **DEFAULT_HAND_POSES}

# Default pose as ordered list (matching LOCO_JOINT_NAMES order)
DEFAULT_LOCO_LIST = [DEFAULT_LOCO_POSES[j] for j in LOCO_JOINT_NAMES]
DEFAULT_ARM_LIST = [DEFAULT_ARM_POSES[j] for j in ARM_JOINT_NAMES]
DEFAULT_HAND_LIST = [DEFAULT_HAND_POSES[j] for j in HAND_JOINT_NAMES]

# ============================================================================
# ACTION SCALES
# ============================================================================

# Leg action scales (per-joint)
LEG_ACTION_SCALE = 0.4  # radians
WAIST_ACTION_SCALE = 0.2  # smaller for stability
ARM_ACTION_SCALE = 0.5
HAND_ACTION_SCALE = 0.3

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================

HEIGHT_DEFAULT = 0.80  # Standing height for 29DoF model (was 0.72 for 23DoF)
HEIGHT_MIN = 0.40  # Squat height
HEIGHT_MAX = 0.85  # Maximum standing height
GAIT_FREQUENCY = 1.5  # Hz

# Shoulder offset from base (for EE computation) — may need calibration
SHOULDER_OFFSET_LEFT = [0.0, 0.174, 0.259]
SHOULDER_OFFSET_RIGHT = [0.0, -0.174, 0.259]

# Palm forward offset (distance from wrist joint to fingertip center)
PALM_FORWARD_OFFSET = 0.10  # Slightly larger than 23DoF (0.08) due to DEX3 hand

# ============================================================================
# ACTUATOR PARAMETERS (from G129_CFG_WITH_DEX3_WHOLEBODY)
# ============================================================================

ACTUATOR_PARAMS = {
    "legs": {
        "joint_expr": [
            ".*_hip_yaw_joint", ".*_hip_roll_joint",
            ".*_hip_pitch_joint", ".*_knee_joint",
            ".*waist.*",
        ],
        "effort_limit": {
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
            ".*waist_yaw_joint": 88.0,
            ".*waist_roll_joint": 35.0,
            ".*waist_pitch_joint": 35.0,
        },
        "velocity_limit": {
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
            ".*waist_yaw_joint": 32.0,
            ".*waist_roll_joint": 30.0,
            ".*waist_pitch_joint": 30.0,
        },
        "stiffness": {
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
            ".*waist.*": 200.0,
        },
        "damping": {
            ".*_hip_yaw_joint": 5.0,
            ".*_hip_roll_joint": 5.0,
            ".*_hip_pitch_joint": 5.0,
            ".*_knee_joint": 5.0,
            ".*waist.*": 5.0,
        },
        "armature": 0.01,
    },
    "feet": {
        "joint_expr": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        "effort_limit": {
            ".*_ankle_pitch_joint": 35.0,
            ".*_ankle_roll_joint": 35.0,
        },
        "velocity_limit": {
            ".*_ankle_pitch_joint": 30.0,
            ".*_ankle_roll_joint": 30.0,
        },
        "stiffness": 20.0,
        "damping": 2.0,
        "armature": 0.01,
    },
    "shoulders": {
        "joint_expr": [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
        "effort_limit": {
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
        },
        "velocity_limit": {
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
        },
        "stiffness": 100.0,
        "damping": 2.0,
        "armature": 0.01,
    },
    "arms": {
        "joint_expr": [".*_shoulder_yaw_joint", ".*_elbow_joint"],
        "effort_limit": {
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
        },
        "velocity_limit": {
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
        },
        "stiffness": 50.0,
        "damping": 2.0,
        "armature": 0.01,
    },
    "wrist": {
        "joint_expr": [".*_wrist_.*"],
        "effort_limit": {
            ".*_wrist_yaw_joint": 5.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
        },
        "velocity_limit": {
            ".*_wrist_yaw_joint": 22.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
        },
        "stiffness": 40.0,
        "damping": 2.0,
        "armature": 0.01,
    },
    "hands": {
        "joint_expr": [
            ".*_hand_index_.*_joint",
            ".*_hand_middle_.*_joint",
            ".*_hand_thumb_.*_joint",
        ],
        "effort_limit": 300,
        "velocity_limit": 100.0,
        "stiffness": 100.0,
        "damping": 10.0,
        "armature": 0.1,
    },
}

# ============================================================================
# OBSERVATION DIMENSIONS
# ============================================================================

# Loco observation: 69 dim
LOCO_OBS_DIM = (
    3 +   # lin_vel_b
    3 +   # ang_vel_b
    3 +   # proj_gravity
    12 +  # joint_pos_leg
    12 +  # joint_vel_leg
    3 +   # joint_pos_waist
    3 +   # joint_vel_waist
    1 +   # height_cmd
    3 +   # vel_cmd (vx, vy, vyaw)
    2 +   # gait_phase (sin, cos)
    15 +  # prev_loco_actions
    3 +   # torso_cmd (roll, pitch, yaw)
    3 +   # torso_euler
    1     # payload_estimate
)  # = 67 (will be finalized during implementation)

# Arm observation: 75 dim (dual arm)
ARM_OBS_DIM = (
    7 +   # arm_pos_right
    7 +   # arm_vel_right
    7 +   # arm_pos_left
    7 +   # arm_vel_left
    3 +   # ee_right_body
    3 +   # ee_left_body
    3 +   # ee_vel_right
    3 +   # ee_vel_left
    4 +   # palm_quat_right
    4 +   # palm_quat_left
    3 +   # target_body
    3 +   # pos_error_right
    3 +   # pos_error_left
    1 +   # pos_dist
    1 +   # orient_err
    1 +   # target_reached
    3 +   # target_orient
    1 +   # height_cmd
    2 +   # lin_vel_xy
    1 +   # ang_vel_z
    1 +   # steps_norm (anti-gaming)
    1 +   # ee_displacement (anti-gaming)
    1     # initial_dist (anti-gaming)
)  # = 70 (will be finalized during implementation)

# Hand observation: 46 dim
HAND_OBS_DIM = (
    7 +   # finger_pos_right
    7 +   # finger_vel_right
    7 +   # finger_pos_left
    7 +   # finger_vel_left
    3 +   # contact_right (per finger)
    3 +   # contact_left (per finger)
    1 +   # grasp_cmd
    1 +   # object_in_hand
    3 +   # ee_right_body
    3 +   # ee_left_body
    4     # palm_quat_right
)  # = 46


if __name__ == "__main__":
    print("=" * 60)
    print("G1-29DoF + DEX3 Configuration")
    print("=" * 60)
    print(f"\nUSD path: {G1_29DOF_USD}")
    print(f"USD exists: {os.path.exists(G1_29DOF_USD)}")
    print(f"\nJoint counts:")
    print(f"  Legs:    {NUM_LEG_JOINTS}")
    print(f"  Waist:   {NUM_WAIST_JOINTS}")
    print(f"  Loco:    {NUM_LOCO_JOINTS}")
    print(f"  Arms:    {NUM_ARM_JOINTS}")
    print(f"  Hands:   {NUM_HAND_JOINTS}")
    print(f"  Total:   {NUM_ALL_JOINTS}")
    print(f"\nDefault poses:")
    for name, val in DEFAULT_ALL_POSES.items():
        if val != 0.0:
            print(f"  {name}: {val}")
    print(f"\nObs dims: Loco={LOCO_OBS_DIM}, Arm={ARM_OBS_DIM}, Hand={HAND_OBS_DIM}")
