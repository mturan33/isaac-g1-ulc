"""
ULC G1 Environment Configuration
================================

Unified Loco-Manipulation Controller için G1 robot environment config.
Stage 1: Standing - Temel denge ve height tracking

Isaac Lab 2.3+ uyumlu Direct Workflow config.
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

# G1 Robot USD path - Isaac Lab'ın default lokasyonu
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

    # G1 Robot
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
            pos=(0.0, 0.0, 0.8),  # G1 height ~0.75m
            joint_pos={
                # Legs - Standing pose
                ".*_hip_pitch_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
                # Arms - Relaxed pose
                ".*_shoulder_pitch_joint": 0.0,
                ".*_shoulder_roll_joint": 0.0,
                ".*_shoulder_yaw_joint": 0.0,
                ".*_elbow_joint": 0.0,
                ".*_wrist_roll_joint": 0.0,
                ".*_wrist_pitch_joint": 0.0,
                ".*_wrist_yaw_joint": 0.0,
                # Waist
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Leg actuators - High stiffness for standing
            "legs": sim_utils.ImplicitActuatorCfg(
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
            # Arm actuators - Medium stiffness
            "arms": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
                stiffness=80.0,
                damping=4.0,
            ),
            # Waist actuators
            "waist": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                stiffness=100.0,
                damping=5.0,
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
    num_actions = 29  # 12 legs + 14 arms + 3 waist (Stage 5'te)
    num_observations = 93  # Detay aşağıda
    num_states = 0  # Asymmetric actor-critic kullanmıyoruz

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 200Hz simulation
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

    # Terrain (flat for now)
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
    # OBSERVATION SPACE (93 dim for full ULC)
    # =========================================================================
    # Stage'e göre dinamik olarak değişecek
    #
    # Base observations (always):
    #   - Base linear velocity (3)
    #   - Base angular velocity (3)
    #   - Projected gravity (3)
    #   - Joint positions (29)
    #   - Joint velocities (29)
    #   - Previous actions (29) - Stage'e göre
    #
    # Commands (stage dependent):
    #   - Velocity commands (3): vx, vy, yaw_rate
    #   - Height command (1)
    #   - Torso orientation (3): roll, pitch, yaw
    #   - Arm targets (14): 7 left + 7 right
    #
    # Total for Stage 1 (Standing): 3+3+3+29+29+29 = 96 base + 1 height = 97
    # Simplified: 48 (no previous actions initially)

    # =========================================================================
    # ACTION SPACE (29 joints for full G1)
    # =========================================================================
    # Left leg: 6 joints (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    # Right leg: 6 joints
    # Waist: 3 joints (yaw, roll, pitch)
    # Left arm: 7 joints (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
    # Right arm: 7 joints
    # Total: 12 + 3 + 14 = 29

    # =========================================================================
    # REWARD WEIGHTS
    # =========================================================================
    # Stage 1: Standing
    reward_scales = {
        # Primary objectives
        "height_tracking": 5.0,
        "orientation": 3.0,
        "com_stability": 4.0,

        # Regularization
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "energy": -0.001,

        # Termination penalties
        "termination": -100.0,
    }

    # =========================================================================
    # TERMINATION CONDITIONS
    # =========================================================================
    termination = {
        "base_height_min": 0.3,  # Düşme
        "base_height_max": 1.2,  # Çok yükseğe zıplama
        "max_pitch": 0.8,  # ~45 derece
        "max_roll": 0.8,
    }

    # =========================================================================
    # DOMAIN RANDOMIZATION
    # =========================================================================
    randomization = {
        # Initial state randomization
        "init_pos_noise": 0.02,  # ±2cm position noise
        "init_rot_noise": 0.1,  # ±0.1 rad rotation noise
        "init_joint_noise": 0.1,  # ±0.1 rad joint noise

        # Physics randomization (Stage 5'te aktif)
        "friction_range": [0.5, 1.5],
        "mass_scale_range": [0.9, 1.1],

        # External disturbances (Stage 5'te aktif)
        "push_interval": [5.0, 10.0],  # seconds
        "push_force": [50.0, 100.0],  # Newtons
    }

    # =========================================================================
    # COMMANDS CONFIG
    # =========================================================================
    commands = {
        # Stage 1: Only height
        "height_target": 0.75,  # meters (G1 standing height)
        "height_range": [0.5, 0.9],

        # Stage 2+: Velocity
        "velocity_range": {
            "vx": [-1.0, 1.0],
            "vy": [-0.5, 0.5],
            "yaw_rate": [-1.0, 1.0],
        },

        # Stage 3+: Torso
        "torso_range": {
            "roll": [-0.3, 0.3],
            "pitch": [-0.3, 0.3],
            "yaw": [-0.5, 0.5],
        },

        # Stage 4+: Arms
        "arm_range": [-1.5, 1.5],  # radians
    }

    # =========================================================================
    # CURRICULUM STAGE CONFIG
    # =========================================================================
    curriculum = {
        "initial_stage": 1,
        "stage_thresholds": {
            1: 0.7,  # Standing: %70 reward threshold
            2: 0.65,  # Locomotion
            3: 0.6,  # Torso
            4: 0.55,  # Arms
        },
        "stage_durations": {
            1: 500_000,  # 500K steps
            2: 1_000_000,  # 1M steps
            3: 1_000_000,  # 1M steps
            4: 2_000_000,  # 2M steps
        },
    }


##
# Stage-specific configs
##

@configclass
class ULC_G1_Stage1_EnvCfg(ULC_G1_EnvCfg):
    """Stage 1: Standing only."""

    # Simplified observation for Stage 1
    num_observations = 48  # Base state only, no arms
    num_actions = 12  # Legs only

    # Only height tracking rewards
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

    num_observations = 51  # +3 velocity commands
    num_actions = 12  # Still legs only

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

    num_observations = 57  # +3 waist positions, +3 torso commands
    num_actions = 15  # Legs + Waist

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

    num_observations = 93  # Full observation
    num_actions = 29  # All joints

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