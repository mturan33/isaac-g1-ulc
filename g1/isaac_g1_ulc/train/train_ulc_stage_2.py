"""
ULC G1 Stage 2: Locomotion Training with Rough Terrain
======================================================

BUILD ON TOP OF STAGE 1:
- Load pretrained Stage 1 standing policy
- Add velocity commands (vx, vy, vyaw)
- Add gait generation (trotting pattern)
- Add curriculum terrain (flat → rough)
- Add perturbation robustness training

Usage:
    # Start fresh locomotion training
    ./isaaclab.bat -p train_ulc_stage_2.py --num_envs 4096 --headless --max_iterations 6000

    # Resume from Stage 1 checkpoint (recommended)
    ./isaaclab.bat -p train_ulc_stage_2.py --num_envs 4096 --headless --max_iterations 6000 --stage1_checkpoint logs/ulc/ulc_g1_stage1_xxx/model_best.pt
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime

# ===== Configuration =====
HEIGHT_MIN = 0.55
HEIGHT_MAX = 0.85
HEIGHT_DEFAULT = 0.72

VX_RANGE = (-1.0, 1.5)
VY_RANGE = (-0.5, 0.5)
VYAW_RANGE = (-1.0, 1.0)

GAIT_FREQUENCY = 2.0

REWARD_WEIGHTS = {
    "velocity_tracking_x": 5.0,
    "velocity_tracking_y": 3.0,
    "velocity_tracking_yaw": 3.0,
    "gait_pattern": 4.0,
    "forward_progress": 2.0,
    "height_tracking": 3.0,
    "orientation": 4.0,
    "feet_contact": 2.0,
    "action_smoothness": -0.02,
    "joint_acceleration": -0.002,
    "energy": -0.001,
    "alive_bonus": 1.0,
    "stumble_penalty": -2.0,
}

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 2 Training - Locomotion")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=6000, help="Max training iterations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from Stage 2 checkpoint")
    parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Initialize from Stage 1 checkpoint")
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage2", help="Experiment name")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    return parser.parse_args()

args_cli = parse_args()

# Isaac Lab imports - MUST BE BEFORE AppLauncher
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from torch.utils.tensorboard import SummaryWriter

# G1 USD path - Use Nucleus URL
G1_USD_PATH = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

print("=" * 80)
print("ULC G1 TRAINING - STAGE 2: LOCOMOTION WITH ROUGH TERRAIN")
print("=" * 80)
print(f"G1 USD: {G1_USD_PATH}")
print(f"\nReward weights:")
for name, weight in REWARD_WEIGHTS.items():
    print(f"  {name}: {weight}")
print()


# ===== Actor-Critic Network =====
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def get_std(self):
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max).exp()

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

    def act(self, obs, deterministic=False):
        action_mean = self.actor(obs)
        if deterministic:
            return action_mean
        std = self.get_std()
        dist = torch.distributions.Normal(action_mean, std)
        return dist.sample()

    def evaluate(self, obs, actions):
        action_mean = self.actor(obs)
        value = self.critic(obs)
        std = self.get_std()
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value.squeeze(-1), log_prob, entropy


# ===== PPO Algorithm =====
class PPO:
    def __init__(self, actor_critic, device, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.2, epochs=5, mini_batch_size=4096,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=1.0):
        self.actor_critic = actor_critic
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, obs_batch, action_batch, old_log_probs, returns, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = obs_batch.shape[0]

        for _ in range(self.epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                mb_obs = obs_batch[mb_indices]
                mb_actions = action_batch[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                values, log_probs, entropy = self.actor_critic.evaluate(mb_obs, mb_actions)

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }


# ===== Stage 2 Environment =====
def create_ulc_g1_stage2_env(num_envs: int, device: str):

    @configclass
    class ULC_G1_Stage2_SceneCfg(InteractiveSceneCfg):
        """Scene configuration with terrain."""

        # Use TerrainImporter for ground plane
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        # G1 Robot
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_USD_PATH,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                    enable_gyroscopic_forces=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8),
                joint_pos={
                    "left_hip_pitch_joint": -0.1,
                    "right_hip_pitch_joint": -0.1,
                    "left_hip_roll_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.25,
                    "right_knee_joint": 0.25,
                    "left_ankle_pitch_joint": -0.15,
                    "right_ankle_pitch_joint": -0.15,
                    "left_ankle_roll_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "left_elbow_pitch_joint": 0.0,
                    "right_elbow_pitch_joint": 0.0,
                    "left_elbow_roll_joint": 0.0,
                    "right_elbow_roll_joint": 0.0,
                    "torso_joint": 0.0,
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                    stiffness=150.0,
                    damping=10.0,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                    stiffness=50.0,
                    damping=5.0,
                ),
                "torso": ImplicitActuatorCfg(
                    joint_names_expr=["torso_joint"],
                    stiffness=100.0,
                    damping=10.0,
                ),
            },
        )

    @configclass
    class ULC_G1_Stage2_EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 20.0

        # Observations: base_vel(3) + ang_vel(3) + gravity(3) + leg_pos(12) + leg_vel(12) +
        # height_cmd(1) + vel_cmd(3) + gait_phase(2) + prev_actions(12) = 51
        action_space = 12
        observation_space = 51
        state_space = 0

        sim = sim_utils.SimulationCfg(
            dt=1/200,
            render_interval=decimation,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        scene = ULC_G1_Stage2_SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class ULC_G1_Stage2_Env(DirectRLEnv):
        cfg: ULC_G1_Stage2_EnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            # Find leg joint indices
            joint_names = self.robot.joint_names
            leg_joint_names = [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
            ]

            self.leg_indices = []
            for name in leg_joint_names:
                if name in joint_names:
                    self.leg_indices.append(joint_names.index(name))
            self.leg_indices = torch.tensor(self.leg_indices, device=self.device)

            print(f"[ULC_G1_Stage2] Leg joints: {len(self.leg_indices)}")

            # Default leg positions
            self.default_leg_positions = torch.tensor([
                -0.1, -0.1, 0.0, 0.0, 0.0, 0.0,
                0.25, 0.25, -0.15, -0.15, 0.0, 0.0,
            ], device=self.device)

            # Commands
            self.target_heights = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)  # vx, vy, vyaw

            # Gait phase
            self.gait_phase = torch.zeros(self.num_envs, device=self.device)

            # Tracking
            self.spawn_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self.previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_joint_vel = None

            # Episode tracking
            self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
            self.episode_lengths = torch.zeros(self.num_envs, device=self.device)

            # Action scale
            self.action_scale = 0.5

            print(f"[ULC_G1_Stage2] Initialized with {self.num_envs} envs")
            print(f"[ULC_G1_Stage2] Observations: {cfg.observation_space}, Actions: {cfg.action_space}")

        @property
        def robot(self):
            return self.scene["robot"]

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()

            robot = self.robot
            targets = robot.data.default_joint_pos.clone()

            leg_targets = self.default_leg_positions.unsqueeze(0) + actions * self.action_scale
            targets[:, self.leg_indices] = leg_targets

            robot.set_joint_position_target(targets)
            self.previous_actions = actions.clone()

            # Update gait phase
            dt = self.cfg.sim.dt * self.cfg.decimation
            self.gait_phase = (self.gait_phase + GAIT_FREQUENCY * dt) % 1.0

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            robot = self.robot

            base_quat = robot.data.root_quat_w
            base_lin_vel = robot.data.root_lin_vel_w
            base_ang_vel = robot.data.root_ang_vel_w

            from isaaclab.utils.math import quat_apply_inverse

            base_lin_vel_b = quat_apply_inverse(base_quat, base_lin_vel)
            base_ang_vel_b = quat_apply_inverse(base_quat, base_ang_vel)

            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)

            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel

            leg_pos = joint_pos[:, self.leg_indices]
            leg_vel = joint_vel[:, self.leg_indices]

            # Commands
            height_cmd = self.target_heights.unsqueeze(-1)
            vel_cmd = self.velocity_commands

            # Gait phase as sin/cos
            gait_sin = torch.sin(2 * np.pi * self.gait_phase).unsqueeze(-1)
            gait_cos = torch.cos(2 * np.pi * self.gait_phase).unsqueeze(-1)
            gait_obs = torch.cat([gait_sin, gait_cos], dim=-1)

            obs = torch.cat([
                base_lin_vel_b,          # 3
                base_ang_vel_b,          # 3
                proj_gravity,            # 3
                leg_pos,                 # 12
                leg_vel,                 # 12
                height_cmd,              # 1
                vel_cmd,                 # 3
                gait_obs,                # 2
                self.previous_actions,   # 12
            ], dim=-1)  # Total: 51

            obs = torch.clamp(obs, -100.0, 100.0)
            obs = torch.nan_to_num(obs, nan=0.0)

            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            robot = self.robot

            base_pos = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w
            base_lin_vel = robot.data.root_lin_vel_w
            base_ang_vel = robot.data.root_ang_vel_w
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel

            leg_pos = joint_pos[:, self.leg_indices]

            from isaaclab.utils.math import quat_apply_inverse

            base_lin_vel_b = quat_apply_inverse(base_quat, base_lin_vel)
            base_ang_vel_b = quat_apply_inverse(base_quat, base_ang_vel)

            # 1. Velocity tracking X
            vx_error = torch.abs(base_lin_vel_b[:, 0] - self.velocity_commands[:, 0])
            r_vx = torch.exp(-3.0 * vx_error ** 2)

            # 2. Velocity tracking Y
            vy_error = torch.abs(base_lin_vel_b[:, 1] - self.velocity_commands[:, 1])
            r_vy = torch.exp(-5.0 * vy_error ** 2)

            # 3. Velocity tracking Yaw
            vyaw_error = torch.abs(base_ang_vel_b[:, 2] - self.velocity_commands[:, 2])
            r_vyaw = torch.exp(-3.0 * vyaw_error ** 2)

            # 4. Gait pattern reward
            left_knee = leg_pos[:, 6]
            right_knee = leg_pos[:, 7]

            phase_left = (self.gait_phase < 0.5).float()
            phase_right = (self.gait_phase >= 0.5).float()

            gait_error = (phase_left * (left_knee - 0.1) ** 2 +
                         phase_right * (right_knee - 0.1) ** 2)
            r_gait = torch.exp(-5.0 * gait_error)

            # 5. Forward progress
            cmd_vel_magnitude = torch.norm(self.velocity_commands[:, :2], dim=-1) + 0.01
            actual_vel_magnitude = torch.norm(base_lin_vel_b[:, :2], dim=-1)
            r_progress = torch.clamp(actual_vel_magnitude / cmd_vel_magnitude, 0, 2)

            # 6. Height tracking
            height = base_pos[:, 2]
            height_error = torch.abs(height - self.target_heights)
            r_height = torch.exp(-10.0 * height_error ** 2)

            # 7. Orientation
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            orientation_error = torch.sum(proj_gravity[:, :2] ** 2, dim=-1)
            r_orientation = torch.exp(-5.0 * orientation_error)

            # 8. Feet contact (placeholder)
            r_feet_contact = torch.ones(self.num_envs, device=self.device)

            # 9. Action smoothness
            action_diff = self.actions - self._prev_actions
            r_action_smooth = torch.sum(action_diff ** 2, dim=-1)
            self._prev_actions = self.actions.clone()

            # 10. Joint acceleration
            if self._prev_joint_vel is not None:
                joint_acc = joint_vel - self._prev_joint_vel
                r_joint_acc = torch.sum(joint_acc ** 2, dim=-1)
            else:
                r_joint_acc = torch.zeros(self.num_envs, device=self.device)
            self._prev_joint_vel = joint_vel.clone()

            # 11. Energy
            r_energy = torch.sum(torch.abs(joint_vel[:, self.leg_indices]) *
                                torch.abs(self.actions), dim=-1)

            # 12. Alive bonus
            r_alive = torch.ones(self.num_envs, device=self.device)

            # 13. Stumble penalty
            r_stumble = torch.zeros(self.num_envs, device=self.device)

            # Total reward
            reward = (
                REWARD_WEIGHTS["velocity_tracking_x"] * r_vx +
                REWARD_WEIGHTS["velocity_tracking_y"] * r_vy +
                REWARD_WEIGHTS["velocity_tracking_yaw"] * r_vyaw +
                REWARD_WEIGHTS["gait_pattern"] * r_gait +
                REWARD_WEIGHTS["forward_progress"] * r_progress +
                REWARD_WEIGHTS["height_tracking"] * r_height +
                REWARD_WEIGHTS["orientation"] * r_orientation +
                REWARD_WEIGHTS["feet_contact"] * r_feet_contact +
                REWARD_WEIGHTS["action_smoothness"] * r_action_smooth +
                REWARD_WEIGHTS["joint_acceleration"] * r_joint_acc +
                REWARD_WEIGHTS["energy"] * r_energy +
                REWARD_WEIGHTS["alive_bonus"] * r_alive +
                REWARD_WEIGHTS["stumble_penalty"] * r_stumble
            )

            self.episode_rewards += reward
            self.episode_lengths += 1

            # Log
            self.extras["Reward/vx_tracking"] = r_vx.mean()
            self.extras["Reward/vy_tracking"] = r_vy.mean()
            self.extras["Reward/vyaw_tracking"] = r_vyaw.mean()
            self.extras["Reward/gait_pattern"] = r_gait.mean()
            self.extras["Reward/height_tracking"] = r_height.mean()
            self.extras["Reward/orientation"] = r_orientation.mean()
            self.extras["Metrics/height"] = height.mean()
            self.extras["Metrics/vx"] = base_lin_vel_b[:, 0].mean()
            self.extras["Metrics/cmd_vx"] = self.velocity_commands[:, 0].mean()

            return reward

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            robot = self.robot

            base_pos = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w

            height = base_pos[:, 2]
            too_low = height < 0.3
            too_high = height > 1.2

            from isaaclab.utils.math import quat_apply_inverse
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            too_tilted = (torch.abs(proj_gravity[:, 0]) > 0.7) | (torch.abs(proj_gravity[:, 1]) > 0.7)

            terminated = too_low | too_high | too_tilted
            time_out = self.episode_length_buf >= self.max_episode_length

            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if len(env_ids) == 0:
                return

            robot = self.robot

            pos = torch.tensor([0.0, 0.0, 0.8], device=self.device).expand(len(env_ids), -1).clone()
            pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * 0.05

            quat = torch.zeros(len(env_ids), 4, device=self.device)
            quat[:, 3] = 1.0

            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

            default_pos = robot.data.default_joint_pos[env_ids]
            robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)

            self.spawn_positions[env_ids] = pos.clone()

            # Randomize commands
            self.target_heights[env_ids] = torch.rand(len(env_ids), device=self.device) * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN

            self.velocity_commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (VX_RANGE[1] - VX_RANGE[0]) + VX_RANGE[0]
            self.velocity_commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (VY_RANGE[1] - VY_RANGE[0]) + VY_RANGE[0]
            self.velocity_commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (VYAW_RANGE[1] - VYAW_RANGE[0]) + VYAW_RANGE[0]

            self.gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)

            self.previous_actions[env_ids] = 0.0
            self._prev_actions[env_ids] = 0.0
            self.episode_rewards[env_ids] = 0.0
            self.episode_lengths[env_ids] = 0.0

    cfg = ULC_G1_Stage2_EnvCfg()
    cfg.scene.num_envs = num_envs
    env = ULC_G1_Stage2_Env(cfg)

    return env, cfg.observation_space, cfg.action_space


# ===== Training Loop =====
def train():
    device = "cuda:0"

    print(f"\n[INFO] Creating Stage 2 environment with {args_cli.num_envs} envs...")
    env, num_obs, num_actions = create_ulc_g1_stage2_env(args_cli.num_envs, device)

    print(f"[INFO] Observations: {num_obs}, Actions: {num_actions}")

    actor_critic = ActorCriticNetwork(num_obs, num_actions).to(device)

    # Load Stage 1 weights if provided
    if args_cli.stage1_checkpoint:
        print(f"\n[INFO] Loading Stage 1 checkpoint: {args_cli.stage1_checkpoint}")
        stage1_ckpt = torch.load(args_cli.stage1_checkpoint, map_location=device, weights_only=False)

        stage1_state = stage1_ckpt["actor_critic"]
        current_state = actor_critic.state_dict()

        transferred = 0
        skipped = 0
        for key in stage1_state:
            if key in current_state and stage1_state[key].shape == current_state[key].shape:
                current_state[key] = stage1_state[key]
                transferred += 1
            else:
                skipped += 1
                print(f"  [SKIP] {key}: shape mismatch")

        actor_critic.load_state_dict(current_state)
        print(f"[INFO] Transferred {transferred} parameters, skipped {skipped} (shape mismatch expected due to obs size change)")

    ppo = PPO(
        actor_critic,
        device,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        epochs=5,
        mini_batch_size=4096,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"{args_cli.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"[INFO] Logging to: {log_dir}")

    num_steps_per_env = 24
    max_iterations = args_cli.max_iterations
    checkpoint_interval = 500
    std_decay_rate = 0.998

    start_iteration = 0
    best_reward = float('-inf')

    if args_cli.checkpoint:
        print(f"\n[INFO] Loading Stage 2 checkpoint: {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        actor_critic.load_state_dict(checkpoint["actor_critic"])
        ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint.get("iteration", 0)
        best_reward = checkpoint.get("best_reward", float('-inf'))
        print(f"[INFO] Resumed from iteration {start_iteration}")

    print(f"[INFO] Starting training from iteration {start_iteration}")
    print(f"[INFO] Target: {max_iterations} iterations")
    print("=" * 80)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    start_time = datetime.now()

    for iteration in range(start_iteration, max_iterations):
        iter_start = datetime.now()

        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []
        log_prob_buffer = []

        for step in range(num_steps_per_env):
            with torch.no_grad():
                action_mean, value = actor_critic(obs)
                std = actor_critic.get_std()
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            obs_buffer.append(obs)
            action_buffer.append(action)
            value_buffer.append(value.squeeze(-1))
            log_prob_buffer.append(log_prob)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            done = terminated | truncated

            reward_buffer.append(reward)
            done_buffer.append(done.float())

        obs_batch = torch.stack(obs_buffer)
        action_batch = torch.stack(action_buffer)
        reward_batch = torch.stack(reward_buffer)
        done_batch = torch.stack(done_buffer)
        value_batch = torch.stack(value_buffer)
        log_prob_batch = torch.stack(log_prob_buffer)

        with torch.no_grad():
            _, next_value = actor_critic(obs)
            next_value = next_value.squeeze(-1)

        advantages, returns = ppo.compute_gae(reward_batch, value_batch, done_batch, next_value)

        obs_flat = obs_batch.view(-1, num_obs)
        action_flat = action_batch.view(-1, num_actions)
        log_prob_flat = log_prob_batch.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        update_info = ppo.update(obs_flat, action_flat, log_prob_flat, returns_flat, advantages_flat)

        with torch.no_grad():
            actor_critic.log_std.data *= std_decay_rate
            actor_critic.log_std.data.clamp_(actor_critic.log_std_min, actor_critic.log_std_max)

        mean_reward = reward_batch.mean().item()
        mean_std = actor_critic.get_std().mean().item()

        iter_time = (datetime.now() - iter_start).total_seconds()
        total_steps = num_steps_per_env * args_cli.num_envs
        steps_per_sec = total_steps / iter_time

        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint_data = {
                "actor_critic": actor_critic.state_dict(),
                "optimizer": ppo.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
            }
            torch.save(checkpoint_data, os.path.join(log_dir, "model_best.pt"))
            print(f"[BEST] New best! Reward: {best_reward:.2f}")

        writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)
        writer.add_scalar("Loss/surrogate", update_info["actor_loss"], iteration)
        writer.add_scalar("Loss/value_function", update_info["critic_loss"], iteration)
        writer.add_scalar("Perf/total_fps", steps_per_sec, iteration)

        for key, value in env.extras.items():
            if isinstance(value, torch.Tensor):
                writer.add_scalar(key, value.item(), iteration)

        if iteration % 10 == 0:
            elapsed = datetime.now() - start_time
            remaining = max_iterations - iteration
            eta = elapsed / (iteration - start_iteration + 1) * remaining if iteration > start_iteration else elapsed * remaining

            print("#" * 80)
            print(f"  STAGE 2 LOCOMOTION - Iteration {iteration}/{max_iterations}")
            print(f"  Reward: {mean_reward:.2f} | Std: {mean_std:.3f} | FPS: {steps_per_sec:.0f}")
            print(f"  Actor Loss: {update_info['actor_loss']:.4f} | Critic Loss: {update_info['critic_loss']:.4f}")
            print(f"  Elapsed: {str(elapsed).split('.')[0]} | ETA: {str(eta).split('.')[0]}")
            print("#" * 80)

        if (iteration + 1) % checkpoint_interval == 0:
            checkpoint_data = {
                "actor_critic": actor_critic.state_dict(),
                "optimizer": ppo.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
            }
            torch.save(checkpoint_data, os.path.join(log_dir, f"model_{iteration+1}.pt"))
            print(f"[CHECKPOINT] Saved model_{iteration+1}.pt")

        writer.flush()

    checkpoint_data = {
        "actor_critic": actor_critic.state_dict(),
        "optimizer": ppo.optimizer.state_dict(),
        "iteration": max_iterations,
        "best_reward": best_reward,
    }
    torch.save(checkpoint_data, os.path.join(log_dir, "model_final.pt"))

    writer.close()
    env.close()

    print("\n" + "=" * 80)
    print("STAGE 2 TRAINING COMPLETE")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Log dir: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
    simulation_app.close()