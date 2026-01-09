#!/usr/bin/env python3
"""
ULC G1 Stage 2 v2 Play Script - Standalone Version (Fixed Config)
Self-contained with all definitions, proper Isaac Lab 2.3+ config
"""

import argparse
import torch
import torch.nn as nn
import math

# Isaac Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ULC G1 Stage 2 v2 Play - Standalone")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--vx", type=float, default=0.5, help="Target forward velocity (m/s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Post-launch imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
import gymnasium as gym
from gymnasium import spaces
import numpy as np

print("=" * 60)
print("ULC G1 STAGE 2 v2 - PLAY (Standalone Fixed)")
print("=" * 60)

# Store vx for later use
TARGET_VX = args.vx


##############################################################################
# Network Definition (matches training)
##############################################################################

class ActorCriticNetwork(nn.Module):
    """Network architecture matching train_ulc_stage_2_v2.py with LayerNorm"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256, 128]):
        super().__init__()

        # Actor network with LayerNorm
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.LayerNorm(hidden_dim))
            actor_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network with LayerNorm
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def forward(self, obs: torch.Tensor):
        return self.act(obs)


##############################################################################
# Environment Definition (self-contained)
##############################################################################

# Joint names for G1 legs (12 DOF)
G1_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# Default joint positions (standing pose)
G1_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    # Arms at default
    ".*_shoulder_.*": 0.0,
    ".*_elbow_.*": 0.0,
    ".*_wrist_.*": 0.0,
    # Torso
    "torso_joint": 0.0,
}


@sim_utils.configclass
class G1PlaySceneCfg(InteractiveSceneCfg):
    """Scene config for G1 play"""

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

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
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
            joint_pos=G1_DEFAULT_JOINT_POS,
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=G1_LEG_JOINT_NAMES,
                stiffness=100.0,
                damping=5.0,
            ),
        },
    )


@sim_utils.configclass
class G1PlayEnvCfg(DirectRLEnvCfg):
    """Environment config for G1 play"""

    decimation = 4
    episode_length_s = 20.0

    action_scale = 0.5

    num_actions = 12
    num_observations = 45  # Same as training

    # Required by Isaac Lab 2.3+
    observation_space = 45
    action_space = 12

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: G1PlaySceneCfg = G1PlaySceneCfg(num_envs=4, env_spacing=2.5)


class G1PlayEnv(DirectRLEnv):
    """Simplified G1 environment for play/inference"""

    cfg: G1PlayEnvCfg

    def __init__(self, cfg: G1PlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for legs
        self._leg_joint_ids, _ = self._robot.find_joints(G1_LEG_JOINT_NAMES)

        # Velocity command
        self.target_vx = TARGET_VX

        # Phase for gait
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._gait_freq = 1.5  # Hz

        # Previous actions for smoothing
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        print(f"[Env] Leg joint IDs: {self._leg_joint_ids}")
        print(f"[Env] Target vx: {self.target_vx:.2f} m/s")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.scene.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.scene.terrain.env_spacing = self.scene.cfg.env_spacing
        self.scene.terrain = self.cfg.scene.terrain.class_type(self.cfg.scene.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._prev_actions = actions.clone()

    def _apply_action(self):
        # Scale actions and apply to leg joints
        targets = self._actions * self.cfg.action_scale

        # Get current default positions
        default_pos = self._robot.data.default_joint_pos[:, self._leg_joint_ids]

        # Add action offsets to default positions
        joint_targets = default_pos + targets

        # Set joint position targets
        self._robot.set_joint_position_target(joint_targets, joint_ids=self._leg_joint_ids)

    def _get_observations(self) -> dict:
        # Update phase
        self._phase += self._gait_freq * self.cfg.sim.dt * self.cfg.decimation
        self._phase = torch.fmod(self._phase, 1.0)

        # Get robot state
        root_quat = self._robot.data.root_quat_w
        root_lin_vel = self._robot.data.root_lin_vel_b
        root_ang_vel = self._robot.data.root_ang_vel_b

        joint_pos = self._robot.data.joint_pos[:, self._leg_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._leg_joint_ids]

        default_pos = self._robot.data.default_joint_pos[:, self._leg_joint_ids]

        # Gravity projection
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)

        # Velocity command
        cmd_vx = torch.full((self.num_envs, 1), self.target_vx, device=self.device)
        cmd_vy = torch.zeros(self.num_envs, 1, device=self.device)
        cmd_yaw = torch.zeros(self.num_envs, 1, device=self.device)

        # Phase signals
        phase_sin = torch.sin(2 * math.pi * self._phase).unsqueeze(1)
        phase_cos = torch.cos(2 * math.pi * self._phase).unsqueeze(1)

        # Build observation
        obs = torch.cat([
            root_lin_vel,  # 3
            root_ang_vel,  # 3
            gravity,  # 3
            cmd_vx, cmd_vy, cmd_yaw,  # 3
            joint_pos - default_pos,  # 12
            joint_vel * 0.1,  # 12
            self._prev_actions,  # 12
            phase_sin, phase_cos,  # 2
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Simple reward for play (not used for training)
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for falls
        root_pos = self._robot.data.root_pos_w
        fallen = root_pos[:, 2] < 0.3

        time_out = self.episode_length_buf >= self.max_episode_length

        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset robot
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset phase
        self._phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        self._prev_actions[env_ids] = 0.0


##############################################################################
# Main
##############################################################################

def main():
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)

    if "best_reward" in checkpoint:
        print(f"[INFO] Best reward: {checkpoint['best_reward']:.2f}")
    if "iteration" in checkpoint:
        print(f"[INFO] Iteration: {checkpoint['iteration']}")

    # Create environment
    env_cfg = G1PlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    env = G1PlayEnv(cfg=env_cfg)

    # Get dimensions from config
    obs_dim = env_cfg.num_observations
    act_dim = env_cfg.num_actions
    print(f"[INFO] Obs dim: {obs_dim}, Act dim: {act_dim}")

    # Create network
    actor_critic = ActorCriticNetwork(obs_dim, act_dim, hidden_dims=[256, 256, 128]).to("cuda:0")

    # Load weights
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()
    print("[INFO] Model loaded successfully!")

    # Initialize
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    print(f"\n[Play] Running with vx={TARGET_VX:.2f} m/s")
    print("[Play] Press Ctrl+C to stop")
    print("-" * 60)

    step = 0
    episode_rewards = torch.zeros(args.num_envs, device="cuda:0")

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # Get action from policy (deterministic)
                actions = actor_critic.act(obs)
                actions = torch.clamp(actions, -1.0, 1.0)

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]

            episode_rewards += rewards

            # Log periodically
            if step % 100 == 0:
                root_vel = env._robot.data.root_lin_vel_b[:, 0].mean().item()
                root_height = env._robot.data.root_pos_w[:, 2].mean().item()
                print(f"Step {step:5d} | Vel: {root_vel:.2f} m/s | Height: {root_height:.2f} m")

            step += 1

            # Handle resets
            dones = terminated | truncated
            if dones.any():
                episode_rewards[dones] = 0.0

    except KeyboardInterrupt:
        print("\n[Play] Stopped by user")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()