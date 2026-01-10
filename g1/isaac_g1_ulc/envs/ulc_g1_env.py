"""
ULC G1 Environment Configuration - FULL MECHANICAL WORKSPACE
============================================================

G1 robotunun TÜM mekanik limitlerini kullanan config.
Unitree'nin belirlediği GÜVENLİ çalışma aralıkları!
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import G1_29DOF_CFG


# ============================================================
# G1 FULL MECHANICAL LIMITS (Unitree Official)
# ============================================================
# Bu değerler Unitree'nin belirlediği GÜVENLİ maksimum açılar!

G1_LIMITS = {
    # LEGS
    "hip_pitch": 1.57,      # ±90° - Squat için!
    "hip_roll": 0.5,        # ±29°
    "hip_yaw": 0.5,         # ±29°
    "knee": 2.0,            # 115° - Squat için!
    "ankle_pitch": 0.87,    # -50° to +30°
    "ankle_roll": 0.26,     # ±15°

    # WAIST
    "waist_yaw": 2.7,       # ±155°
    "waist_roll": 0.52,     # ±30°
    "waist_pitch": 0.52,    # ±30° - Eğilme için!

    # ARMS
    "shoulder_pitch": 2.6,  # ±149° (simetrik min)
    "shoulder_roll": 1.6,   # ±92° (simetrik min)
    "shoulder_yaw": 2.6,    # ±149°
    "elbow": 1.6,           # ±92°
    "wrist_roll": 1.6,      # ±92°
    "wrist_pitch": 1.0,     # ±57°
    "wrist_yaw": 1.0,       # ±57°
}


##
# Scene Configuration
##

@configclass
class ULC_G1_SceneCfg(InteractiveSceneCfg):
    """Scene configuration with G1 robot."""

    # Ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # G1 Robot
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                # Legs - default standing
                ".*_hip_pitch_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_yaw_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
                # Arms - relaxed
                ".*_shoulder_pitch_joint": 0.0,
                ".*_shoulder_roll_joint": 0.0,
                ".*_shoulder_yaw_joint": 0.0,
                ".*_elbow_pitch_joint": 0.0,
                ".*_elbow_roll_joint": 0.0,
                # Waist
                "waist_.*_joint": 0.0,
            },
        ),
        actuators={
            # Leg actuators
            "legs": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
                stiffness=100.0,
                damping=5.0,
            ),
            # Arm actuators
            "arms": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
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
# Base Environment Config
##

@configclass
class ULC_G1_EnvCfg(DirectRLEnvCfg):
    """ULC G1 Environment configuration - FULL WORKSPACE."""

    # Environment settings
    episode_length_s = 20.0
    decimation = 4  # 50Hz control
    num_actions = 29  # 12 legs + 14 arms + 3 waist
    num_observations = 93
    num_states = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
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
    )

    # =========================================================================
    # REWARD SCALES
    # =========================================================================
    reward_scales = {
        "height_tracking": 3.0,
        "velocity_tracking": 5.0,
        "torso_orientation": 2.0,
        "arm_tracking": 4.0,
        "orientation": 2.0,
        "com_stability": 2.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "termination": -100.0,
    }

    # =========================================================================
    # TERMINATION - SQUAT İÇİN GÜNCELLENDİ!
    # =========================================================================
    termination = {
        "base_height_min": 0.25,  # ← SQUAT İÇİN! (eskisi 0.5 idi)
        "base_height_max": 1.0,
        "max_roll": 0.8,          # ~46° - daha toleranslı
        "max_pitch": 0.8,         # ~46° - daha toleranslı
    }

    # =========================================================================
    # RANDOMIZATION
    # =========================================================================
    randomization = {
        "init_pos_noise": 0.02,
        "init_rot_noise": 0.05,
        "init_joint_noise": 0.1,
        "friction_range": [0.5, 1.5],
        "mass_scale_range": [0.9, 1.1],
        "push_interval": [5.0, 10.0],
        "push_force": [50.0, 150.0],
    }

    # =========================================================================
    # COMMANDS - FULL MECHANICAL LIMITS!
    # =========================================================================
    commands = {
        # Height - SQUAT DAHİL!
        "height_target": 0.75,
        "height_range": [0.35, 0.85],  # ← 0.35m = derin squat!

        # Velocity
        "velocity_range": {
            "vx": [-1.0, 1.5],
            "vy": [-0.5, 0.5],
            "yaw_rate": [-1.0, 1.0],
        },

        # Torso - FULL!
        "torso_range": {
            "roll": [-0.52, 0.52],   # ← FULL!
            "pitch": [-0.52, 0.52],  # ← FULL!
            "yaw": [-0.5, 0.5],
        },

        # Arms - FULL MECHANICAL LIMITS!
        "arm_range": [-2.6, 2.6],  # ← FULL! (eskisi -1.5, 1.5 idi)
    }

    # =========================================================================
    # CURRICULUM CONFIG
    # =========================================================================
    curriculum = {
        "initial_stage": 1,
        "stage_thresholds": {
            1: 0.7,
            2: 0.65,
            3: 0.6,
            4: 0.55,
            5: 0.5,
        },
        "stage_durations": {
            1: 500_000,
            2: 1_000_000,
            3: 1_000_000,
            4: 2_000_000,
            5: 2_000_000,
        },
    }


##
# Stage-specific Configs
##

@configclass
class ULC_G1_Stage1_EnvCfg(ULC_G1_EnvCfg):
    """Stage 1: Standing only."""
    num_observations = 48
    num_actions = 12

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
    """Stage 3: + Torso Control."""
    num_observations = 57
    num_actions = 12

    reward_scales = {
        "height_tracking": 2.0,
        "velocity_tracking": 4.0,
        "torso_orientation": 3.0,
        "orientation": 2.0,
        "com_stability": 2.0,
        "gait_frequency": 1.0,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "termination": -100.0,
    }

    # Torso FULL range
    commands = {
        **ULC_G1_EnvCfg.commands,
        "torso_range": {
            "roll": [-0.52, 0.52],   # FULL!
            "pitch": [-0.52, 0.52],  # FULL!
            "yaw": [-0.5, 0.5],
        },
    }


@configclass
class ULC_G1_Stage4_EnvCfg(ULC_G1_EnvCfg):
    """Stage 4: + Arm Control - FULL WORKSPACE!"""
    num_observations = 77
    num_actions = 22  # 12 legs + 10 arms

    reward_scales = {
        "height_tracking": 2.0,
        "velocity_tracking": 3.0,
        "torso_orientation": 2.0,
        "arm_tracking": 4.0,
        "orientation": 2.0,
        "com_stability": 2.0,
        "gait_frequency": 0.5,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "termination": -100.0,
    }

    # FULL WORKSPACE COMMANDS!
    commands = {
        "height_target": 0.75,
        "height_range": [0.35, 0.85],  # SQUAT!

        "velocity_range": {
            "vx": [-1.0, 1.5],
            "vy": [-0.5, 0.5],
            "yaw_rate": [-1.0, 1.0],
        },

        "torso_range": {
            "roll": [-0.52, 0.52],   # FULL!
            "pitch": [-0.52, 0.52],  # FULL!
            "yaw": [-0.5, 0.5],
        },

        # ARMS - FULL MECHANICAL LIMITS!
        "arm_range": [-2.6, 2.6],  # ← FULL!
    }

    # Termination - squat için
    termination = {
        "base_height_min": 0.25,  # SQUAT!
        "base_height_max": 1.0,
        "max_roll": 0.8,
        "max_pitch": 0.8,
    }


@configclass
class ULC_G1_Stage5_EnvCfg(ULC_G1_EnvCfg):
    """Stage 5: Full ULC - ALL CAPABILITIES AT FULL WORKSPACE!"""
    num_observations = 93
    num_actions = 29  # 12 legs + 14 arms + 3 waist

    reward_scales = {
        "height_tracking": 2.0,
        "velocity_tracking": 3.0,
        "torso_orientation": 2.0,
        "arm_tracking": 4.0,
        "waist_control": 2.0,
        "orientation": 2.0,
        "com_stability": 2.0,
        "gait_frequency": 0.5,
        "joint_acceleration": -0.0005,
        "action_rate": -0.01,
        "energy": -0.001,
        "termination": -100.0,
    }

    # ALL FULL WORKSPACE!
    commands = {
        "height_target": 0.75,
        "height_range": [0.35, 0.85],  # FULL SQUAT!

        "velocity_range": {
            "vx": [-1.0, 1.5],
            "vy": [-0.5, 0.5],
            "yaw_rate": [-1.0, 1.0],
        },

        "torso_range": {
            "roll": [-0.52, 0.52],   # FULL!
            "pitch": [-0.52, 0.52],  # FULL!
            "yaw": [-0.5, 0.5],
        },

        # ARMS - FULL!
        "arm_range": [-2.6, 2.6],

        # WAIST - FULL!
        "waist_range": {
            "yaw": [-2.7, 2.7],      # FULL - ±155°!
            "roll": [-0.52, 0.52],   # FULL!
            "pitch": [-0.52, 0.52],  # FULL!
        },
    }

    termination = {
        "base_height_min": 0.25,  # SQUAT!
        "base_height_max": 1.0,
        "max_roll": 0.8,
        "max_pitch": 0.8,
    }

    # Domain randomization aktif
    enable_domain_randomization = True