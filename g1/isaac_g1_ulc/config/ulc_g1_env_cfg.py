"""
ULC G1 Environment Configuration
================================

Unified Loco-Manipulation Controller için G1 robot environment config.
Stage 1: Standing - Temel denge ve height tracking

Isaac Lab 2.3+ uyumlu Direct Workflow config.

G1_29DOF ACTUAL JOINTS:
- Legs (12): left/right_hip_pitch/roll/yaw_joint, left/right_knee_joint, left/right_ankle_pitch/roll_joint
- Arms (10): left/right_shoulder_pitch/roll/yaw_joint, left/right_elbow_pitch/roll_joint
- Torso (1): torso_joint
- Fingers (14): left/right_zero/one/two/three/four/five/six_joint

Total controllable: 23 joints (12 legs + 10 arms + 1 torso)
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg

# G1 Robot USD path
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/G1/g1.usd"


##
# Scene Configuration
##

@configclass
class ULC_G1_SceneCfg(InteractiveSceneCfg):
    """ULC G1 Scene configuration."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Distant light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # G1 Robot - CORRECTED JOINT NAMES
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=G1_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                # Legs - Standing pose (12 joints)
                ".*_hip_pitch_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
                # Arms - Relaxed pose (10 joints)
                ".*_shoulder_pitch_joint": 0.0,
                ".*_shoulder_roll_joint": 0.0,
                ".*_shoulder_yaw_joint": 0.0,
                ".*_elbow_pitch_joint": 0.0,  # FIXED: was elbow_joint
                ".*_elbow_roll_joint": 0.0,   # ADDED
                # Torso (1 joint)
                "torso_joint": 0.0,           # FIXED: was waist_*
                # Fingers - default
                ".*_zero_joint": 0.0,
                ".*_one_joint": 0.0,
                ".*_two_joint": 0.0,
                ".*_three_joint": 0.0,
                ".*_four_joint": 0.0,
                ".*_five_joint": 0.0,
                ".*_six_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Leg actuators - High stiffness for standing (12 joints)
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
                stiffness=150.0,
                damping=5.0,
            ),
            # Arm actuators - Medium stiffness (10 joints)
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",   # FIXED
                    ".*_elbow_roll_joint",    # ADDED
                ],
                stiffness=80.0,
                damping=4.0,
            ),
            # Torso actuator (1 joint)
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],  # FIXED: was waist_*
                stiffness=100.0,
                damping=5.0,
            ),
            # Finger actuators (14 joints) - low stiffness
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                    ".*_three_joint",
                    ".*_four_joint",
                    ".*_five_joint",
                    ".*_six_joint",
                ],
                stiffness=20.0,
                damping=2.0,
            ),
        },
    )


##
# ULC Environment Config
##

@configclass
class ULC_G1_EnvCfg(DirectRLEnvCfg):
    """ULC G1 Environment configuration."""

    # Environment settings
    episode_length_s = 20.0
    decimation = 4  # 50Hz control (200Hz sim / 4)
    num_actions = 23  # 12 legs + 10 arms + 1 torso (NO fingers for now)
    num_observations = 77  # Updated for actual joint count
    num_states = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Scene
    scene: ULC_G1_SceneCfg = ULC_G1_SceneCfg(num_envs=4096, env_spacing=2.5)

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # =========================================================================
    # REWARD WEIGHTS
    # =========================================================================
    reward_scales = {
        "height_tracking": 5.0,
        "orientation": 3.0,
        "com_stability": 4.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "energy": -0.001,
        "termination": -100.0,
    }

    # =========================================================================
    # TERMINATION CONDITIONS
    # =========================================================================
    termination = {
        "base_height_min": 0.3,
        "base_height_max": 1.2,
        "max_pitch": 0.8,
        "max_roll": 0.8,
    }

    # =========================================================================
    # DOMAIN RANDOMIZATION
    # =========================================================================
    randomization = {
        "init_pos_noise": 0.02,
        "init_rot_noise": 0.1,
        "init_joint_noise": 0.1,
        "friction_range": [0.5, 1.5],
        "mass_scale_range": [0.9, 1.1],
        "push_interval": [5.0, 10.0],
        "push_force": [50.0, 100.0],
    }

    # =========================================================================
    # COMMANDS CONFIG
    # =========================================================================
    commands = {
        "height_target": 0.75,
        "height_range": [0.5, 0.9],
        "velocity_range": {
            "vx": [-1.0, 1.0],
            "vy": [-0.5, 0.5],
            "yaw_rate": [-1.0, 1.0],
        },
        "torso_range": {
            "roll": [-0.3, 0.3],
            "pitch": [-0.3, 0.3],
            "yaw": [-0.5, 0.5],
        },
        "arm_range": [-2.6, 2.6],  # FULL WORKSPACE!
    }

    # =========================================================================
    # CURRICULUM STAGE CONFIG
    # =========================================================================
    curriculum = {
        "initial_stage": 1,
        "stage_thresholds": {
            1: 0.7,
            2: 0.65,
            3: 0.6,
            4: 0.55,
        },
        "stage_durations": {
            1: 500_000,
            2: 1_000_000,
            3: 1_000_000,
            4: 2_000_000,
        },
    }


##
# Stage-specific configs
##

@configclass
class ULC_G1_Stage1_EnvCfg(ULC_G1_EnvCfg):
    """Stage 1: Standing only."""
    num_observations = 48
    num_actions = 12  # Legs only

    reward_scales = {
        "height_tracking": 5.0,
        "orientation": 3.0,
        "com_stability": 4.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "termination": -100.0,
    }


@configclass
class ULC_G1_Stage2_EnvCfg(ULC_G1_EnvCfg):
    """Stage 2: Standing + Locomotion."""
    num_observations = 51
    num_actions = 12

    reward_scales = {
        "height_tracking": 3.0,
        "velocity_tracking": 5.0,
        "orientation": 2.0,
        "com_stability": 3.0,
        "gait_frequency": 1.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "termination": -100.0,
    }


@configclass
class ULC_G1_Stage3_EnvCfg(ULC_G1_EnvCfg):
    """Stage 3: Standing + Locomotion + Torso."""
    num_observations = 57
    num_actions = 13  # 12 legs + 1 torso

    reward_scales = {
        "height_tracking": 2.0,
        "velocity_tracking": 4.0,
        "torso_tracking": 4.0,
        "orientation": 1.5,
        "com_stability": 3.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "termination": -100.0,
    }


@configclass
class ULC_G1_Stage4_EnvCfg(ULC_G1_EnvCfg):
    """Stage 4: Full ULC with Arms."""
    num_observations = 77  # Full observation for 23 joints
    num_actions = 22  # 12 legs + 10 arms (no torso action in Stage 4)

    reward_scales = {
        "height_tracking": 1.5,
        "velocity_tracking": 3.0,
        "torso_tracking": 3.0,
        "arm_tracking": 5.0,
        "orientation": 1.0,
        "com_stability": 4.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "arm_smoothness": -0.02,
        "termination": -100.0,
    }

    # FULL WORKSPACE!
    commands = {
        "height_target": 0.75,
        "height_range": [0.35, 0.85],  # SQUAT!
        "velocity_range": {
            "vx": [-1.0, 1.5],
            "vy": [-0.5, 0.5],
            "yaw_rate": [-1.0, 1.0],
        },
        "torso_range": {
            "roll": [-0.52, 0.52],
            "pitch": [-0.52, 0.52],
            "yaw": [-0.5, 0.5],
        },
        "arm_range": [-2.6, 2.6],  # FULL!
    }

    termination = {
        "base_height_min": 0.25,  # SQUAT!
        "base_height_max": 1.2,
        "max_pitch": 0.8,
        "max_roll": 0.8,
    }