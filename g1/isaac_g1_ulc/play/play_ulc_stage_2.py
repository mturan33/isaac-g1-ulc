"""
ULC G1 Stage 2: Play/Test Script
================================
Visualize trained Stage 2 locomotion policy.

Usage:
    ./isaaclab.bat -p play_ulc_stage_2.py --checkpoint logs/ulc/ulc_g1_stage2_xxx/model_best.pt --num_envs 4
"""

import torch
import torch.nn as nn
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 2 Play")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    return parser.parse_args()

args_cli = parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 2.0
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
        super().__init__()

        actor_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            actor_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            critic_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.zeros(num_actions))
        self.log_std_min = -3.0
        self.log_std_max = 0.5

    def act(self, obs, deterministic=True):
        action_mean = self.actor(obs)
        return action_mean


def create_play_env(num_envs: int, device: str):

    @configclass
    class PlaySceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
        )

        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_USD_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=10.0, enable_gyroscopic_forces=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8),
                joint_pos={
                    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
                    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.25, "right_knee_joint": 0.25,
                    "left_ankle_pitch_joint": -0.15, "right_ankle_pitch_joint": -0.15,
                    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0, "right_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
                    "left_elbow_pitch_joint": 0.0, "right_elbow_pitch_joint": 0.0,
                    "left_elbow_roll_joint": 0.0, "right_elbow_roll_joint": 0.0,
                    "torso_joint": 0.0,
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"], stiffness=150.0, damping=10.0),
                "arms": ImplicitActuatorCfg(joint_names_expr=[".*shoulder.*", ".*elbow.*"], stiffness=50.0, damping=5.0),
                "torso": ImplicitActuatorCfg(joint_names_expr=["torso_joint"], stiffness=100.0, damping=10.0),
            },
        )

    @configclass
    class PlayEnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 30.0
        action_space = 12
        observation_space = 51
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=decimation)
        scene = PlaySceneCfg(num_envs=num_envs, env_spacing=2.5)

    class PlayEnv(DirectRLEnv):
        cfg: PlayEnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            joint_names = self.robot.joint_names
            leg_joint_names = [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
            ]

            self.leg_indices = [joint_names.index(n) for n in leg_joint_names if n in joint_names]
            self.leg_indices = torch.tensor(self.leg_indices, device=self.device)

            self.default_leg_positions = torch.tensor([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, -0.15, -0.15, 0.0, 0.0], device=self.device)
            self.target_heights = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
            self.velocity_commands[:, 0] = 0.5
            self.gait_phase = torch.zeros(self.num_envs, device=self.device)
            self.previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self.action_scale = 0.5
            print(f"[Play] Initialized: {self.num_envs} envs, vx={self.velocity_commands[0,0]:.2f} m/s")

        @property
        def robot(self):
            return self.scene["robot"]

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()
            targets = self.robot.data.default_joint_pos.clone()
            targets[:, self.leg_indices] = self.default_leg_positions.unsqueeze(0) + actions * self.action_scale
            self.robot.set_joint_position_target(targets)
            self.previous_actions = actions.clone()
            self.gait_phase = (self.gait_phase + GAIT_FREQUENCY * self.cfg.sim.dt * self.cfg.decimation) % 1.0

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            robot = self.robot
            from isaaclab.utils.math import quat_apply_inverse
            base_quat = robot.data.root_quat_w
            base_lin_vel_b = quat_apply_inverse(base_quat, robot.data.root_lin_vel_w)
            base_ang_vel_b = quat_apply_inverse(base_quat, robot.data.root_ang_vel_w)
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            leg_pos = robot.data.joint_pos[:, self.leg_indices]
            leg_vel = robot.data.joint_vel[:, self.leg_indices]
            gait_obs = torch.stack([torch.sin(2*np.pi*self.gait_phase), torch.cos(2*np.pi*self.gait_phase)], dim=-1)
            obs = torch.cat([base_lin_vel_b, base_ang_vel_b, proj_gravity, leg_pos, leg_vel,
                           self.target_heights.unsqueeze(-1), self.velocity_commands, gait_obs, self.previous_actions], dim=-1)
            return {"policy": torch.clamp(torch.nan_to_num(obs, nan=0.0), -100.0, 100.0)}

        def _get_rewards(self) -> torch.Tensor:
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            from isaaclab.utils.math import quat_apply_inverse
            height = self.robot.data.root_pos_w[:, 2]
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(self.robot.data.root_quat_w, gravity)
            terminated = (height < 0.3) | (height > 1.2) | (proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7)
            return terminated, self.episode_length_buf >= self.max_episode_length

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if len(env_ids) == 0: return
            pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(len(env_ids), -1).clone()
            quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(env_ids), -1)
            self.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
            default_pos = self.robot.data.default_joint_pos[env_ids]
            self.robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)
            self.gait_phase[env_ids] = 0.0
            self.previous_actions[env_ids] = 0.0

    cfg = PlayEnvCfg()
    cfg.scene.num_envs = num_envs
    return PlayEnv(cfg), cfg.observation_space, cfg.action_space


def main():
    device = "cuda:0"
    print("=" * 60)
    print("ULC G1 STAGE 2 - PLAY")
    print("=" * 60)

    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    print(f"[INFO] Loaded: {args_cli.checkpoint}")
    if "best_reward" in checkpoint: print(f"[INFO] Best reward: {checkpoint['best_reward']:.2f}")

    env, num_obs, num_actions = create_play_env(args_cli.num_envs, device)

    actor_critic = ActorCriticNetwork(num_obs, num_actions).to(device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

    print("\n[INFO] Running... Press Ctrl+C to exit")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    step = 0

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = actor_critic.act(obs, deterministic=True)
            obs_dict, _, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]
            step += 1
            if step % 100 == 0:
                h = env.robot.data.root_pos_w[:, 2].mean().item()
                vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()
                print(f"Step {step}: H={h:.3f}m, Vx={vx:.3f}m/s")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()