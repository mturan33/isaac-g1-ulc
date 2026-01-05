"""
ULC G1 Stage 1: Standing Training (v6 - Proper Standing Reward)
================================================================

PROBLEM SOLVED:
- Previous version learned "lunge stance" (one foot forward, one back)
- Robot was reward hacking by widening stance for stability
- High std caused trembling

NEW REWARD COMPONENTS:
1. r_height_tracking   - Track target height (0.65-0.85m)
2. r_orientation       - Stay upright (gravity projection)
3. r_base_xy_stability - Don't drift from spawn position
4. r_base_velocity     - Minimize linear/angular velocity
5. r_feet_proximity    - Keep feet close together (prevent wide stance)
6. r_joint_default     - Stay close to natural standing pose
7. r_symmetry          - Left/right leg symmetry
8. r_action_smooth     - Smooth actions (reduce jerk)
9. r_alive_bonus       - Small bonus for not falling

IMPROVEMENTS:
- More aggressive std decay (0.995)
- Lower std minimum (0.05)
- 4000 iterations (~2 hours)
- Initial position tracking to prevent drift
- Reduced action scale (0.5) for smoother motion

Usage:
    ./isaaclab.bat -p train_ulc_stage_1.py --num_envs 4096 --headless --max_iterations 4000
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime

# ===== Configuration =====
HEIGHT_MIN = 0.65
HEIGHT_MAX = 0.85
HEIGHT_DEFAULT = 0.75

# Reward weights - tuned for proper standing
REWARD_WEIGHTS = {
    "height_tracking": 4.0,      # Track target height
    "orientation": 3.0,          # Stay upright
    "base_xy_stability": 2.0,    # Don't drift from spawn
    "linear_velocity": 1.5,      # Stay still (linear)
    "angular_velocity": 1.0,     # Stay still (angular)
    "feet_proximity": 3.0,       # Keep feet together (NEW!)
    "joint_default": 2.5,        # Natural standing pose (NEW!)
    "symmetry": 2.0,             # Left/right symmetry (NEW!)
    "action_smoothness": -0.01,  # Penalize jerky actions
    "joint_acceleration": -0.001, # Penalize joint jerk
    "alive_bonus": 0.5,          # Bonus for staying alive
}

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 1 Training - Proper Standing")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=4000, help="Max training iterations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage1", help="Experiment name")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    return parser.parse_args()

args_cli = parse_args()

# Isaac Lab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from torch.utils.tensorboard import SummaryWriter

G1_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"

print("=" * 80)
print("ULC G1 TRAINING - STAGE 1: PROPER STANDING (v6)")
print("=" * 80)
print(f"\nReward weights:")
for name, weight in REWARD_WEIGHTS.items():
    print(f"  {name}: {weight}")
print()

# ===== Actor-Critic Network =====
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Actor network
        actor_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
            ])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
            ])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Learnable log_std with tighter bounds
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        self.log_std_min = -3.0  # std_min = 0.05 (tighter!)
        self.log_std_max = 0.5   # std_max = 1.65

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor output layer - small weights for initial exploration
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
        action = dist.sample()
        return action

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
                 value_coef=0.5, entropy_coef=0.005, max_grad_norm=1.0):
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

                # Policy loss
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
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


# ===== Environment =====
def create_ulc_g1_env(num_envs: int, device: str):

    @configclass
    class ULC_G1_Stage1_SceneCfg(InteractiveSceneCfg):
        # Ground plane
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
                    # Natural standing pose
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
                    # Arms and torso
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
    class ULC_G1_Stage1_EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 20.0

        # New API: use spaces instead of num_*
        action_space = 12
        observation_space = 46
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

        scene = ULC_G1_Stage1_SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class ULC_G1_Stage1_Env(DirectRLEnv):
        cfg: ULC_G1_Stage1_EnvCfg

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
            print(f"[ULC_G1_Stage1] Leg joints: {len(self.leg_indices)}")
            print(f"[ULC_G1_Stage1] Joint names: {leg_joint_names}")

            # Default leg positions tensor (natural standing pose)
            self.default_leg_positions = torch.tensor([
                -0.1, -0.1,   # hip pitch (L, R)
                0.0, 0.0,     # hip roll (L, R)
                0.0, 0.0,     # hip yaw (L, R)
                0.25, 0.25,   # knee (L, R)
                -0.15, -0.15, # ankle pitch (L, R)
                0.0, 0.0,     # ankle roll (L, R)
            ], device=self.device)

            # Left/Right leg indices for symmetry calculation
            self.left_leg_idx = torch.tensor([0, 2, 4, 6, 8, 10], device=self.device)
            self.right_leg_idx = torch.tensor([1, 3, 5, 7, 9, 11], device=self.device)

            # Per-environment target heights
            self.target_heights = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT

            # Initial spawn positions (to track drift)
            self.spawn_positions = torch.zeros(self.num_envs, 2, device=self.device)

            # Action tracking
            self.previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_joint_vel = None

            # Episode tracking
            self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
            self.episode_lengths = torch.zeros(self.num_envs, device=self.device)

            # Action scale - reduced for smoother motion
            self.action_scale = 0.5

            print(f"[ULC_G1_Stage1] Initialized with {self.num_envs} envs")
            print(f"[ULC_G1_Stage1] Height range: {HEIGHT_MIN}-{HEIGHT_MAX}m")
            print(f"[ULC_G1_Stage1] Action scale: {self.action_scale}")
            print(f"[ULC_G1_Stage1] Observations: {cfg.observation_space}, Actions: {cfg.action_space}")

        @property
        def robot(self):
            return self.scene["robot"]

        def _setup_scene(self):
            # InteractiveScene handles robot spawning automatically
            # Just clone environments and filter collisions
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()

            robot = self.robot
            targets = robot.data.default_joint_pos.clone()

            # Apply actions relative to default standing pose
            leg_targets = self.default_leg_positions.unsqueeze(0) + actions * self.action_scale
            targets[:, self.leg_indices] = leg_targets

            robot.set_joint_position_target(targets)
            self.previous_actions = actions.clone()

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

            height_cmd = self.target_heights.unsqueeze(-1)

            obs = torch.cat([
                base_lin_vel_b,          # 3
                base_ang_vel_b,          # 3
                proj_gravity,            # 3
                leg_pos,                 # 12
                leg_vel,                 # 12
                height_cmd,              # 1
                self.previous_actions,   # 12
            ], dim=-1)

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

            # ============================================
            # 1. HEIGHT TRACKING
            # ============================================
            height = base_pos[:, 2]
            height_error = torch.abs(height - self.target_heights)
            r_height = torch.exp(-10.0 * height_error ** 2)

            # ============================================
            # 2. ORIENTATION (stay upright)
            # ============================================
            from isaaclab.utils.math import quat_apply_inverse
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(base_quat, gravity)
            orientation_error = torch.sum(proj_gravity[:, :2] ** 2, dim=-1)
            r_orientation = torch.exp(-5.0 * orientation_error)

            # ============================================
            # 3. BASE XY STABILITY (don't drift from spawn)
            # ============================================
            xy_drift = torch.norm(base_pos[:, :2] - self.spawn_positions, dim=-1)
            r_base_stability = torch.exp(-5.0 * xy_drift ** 2)

            # ============================================
            # 4. LINEAR VELOCITY (stay still)
            # ============================================
            lin_vel_norm = torch.norm(base_lin_vel, dim=-1)
            r_lin_velocity = torch.exp(-2.0 * lin_vel_norm ** 2)

            # ============================================
            # 5. ANGULAR VELOCITY (don't rotate)
            # ============================================
            ang_vel_norm = torch.norm(base_ang_vel, dim=-1)
            r_ang_velocity = torch.exp(-1.0 * ang_vel_norm ** 2)

            # ============================================
            # 6. FEET PROXIMITY (keep feet together) - NEW!
            # ============================================
            # Use hip roll and yaw as proxy for feet separation
            left_hip_roll = leg_pos[:, 2]   # left_hip_roll_joint
            right_hip_roll = leg_pos[:, 3]  # right_hip_roll_joint
            left_hip_yaw = leg_pos[:, 4]    # left_hip_yaw_joint
            right_hip_yaw = leg_pos[:, 5]   # right_hip_yaw_joint

            # Penalize hip roll/yaw deviation (causes wide stance)
            feet_spread = (left_hip_roll ** 2 + right_hip_roll ** 2 +
                          left_hip_yaw ** 2 + right_hip_yaw ** 2)
            r_feet_proximity = torch.exp(-10.0 * feet_spread)

            # ============================================
            # 7. JOINT DEFAULT (stay near natural pose) - NEW!
            # ============================================
            joint_deviation = leg_pos - self.default_leg_positions.unsqueeze(0)
            joint_deviation_norm = torch.sum(joint_deviation ** 2, dim=-1)
            r_joint_default = torch.exp(-2.0 * joint_deviation_norm)

            # ============================================
            # 8. SYMMETRY (left/right balance) - NEW!
            # ============================================
            left_leg = leg_pos[:, self.left_leg_idx]
            right_leg = leg_pos[:, self.right_leg_idx]
            symmetry_error = torch.sum((left_leg - right_leg) ** 2, dim=-1)
            r_symmetry = torch.exp(-3.0 * symmetry_error)

            # ============================================
            # 9. ACTION SMOOTHNESS
            # ============================================
            action_diff = self.actions - self._prev_actions
            action_rate_penalty = torch.sum(action_diff ** 2, dim=-1)
            r_action_smooth = action_rate_penalty
            self._prev_actions = self.actions.clone()

            # ============================================
            # 10. JOINT ACCELERATION
            # ============================================
            if self._prev_joint_vel is not None:
                joint_acc = joint_vel - self._prev_joint_vel
                r_joint_acc = torch.sum(joint_acc ** 2, dim=-1)
            else:
                r_joint_acc = torch.zeros(self.num_envs, device=self.device)
            self._prev_joint_vel = joint_vel.clone()

            # ============================================
            # 11. ALIVE BONUS
            # ============================================
            r_alive = torch.ones(self.num_envs, device=self.device)

            # ============================================
            # TOTAL REWARD
            # ============================================
            reward = (
                REWARD_WEIGHTS["height_tracking"] * r_height +
                REWARD_WEIGHTS["orientation"] * r_orientation +
                REWARD_WEIGHTS["base_xy_stability"] * r_base_stability +
                REWARD_WEIGHTS["linear_velocity"] * r_lin_velocity +
                REWARD_WEIGHTS["angular_velocity"] * r_ang_velocity +
                REWARD_WEIGHTS["feet_proximity"] * r_feet_proximity +
                REWARD_WEIGHTS["joint_default"] * r_joint_default +
                REWARD_WEIGHTS["symmetry"] * r_symmetry +
                REWARD_WEIGHTS["action_smoothness"] * r_action_smooth +
                REWARD_WEIGHTS["joint_acceleration"] * r_joint_acc +
                REWARD_WEIGHTS["alive_bonus"] * r_alive
            )

            # Track episode rewards
            self.episode_rewards += reward
            self.episode_lengths += 1

            # Log components for TensorBoard
            self.extras["Reward/height_tracking"] = r_height.mean()
            self.extras["Reward/orientation"] = r_orientation.mean()
            self.extras["Reward/base_xy_stability"] = r_base_stability.mean()
            self.extras["Reward/linear_velocity"] = r_lin_velocity.mean()
            self.extras["Reward/angular_velocity"] = r_ang_velocity.mean()
            self.extras["Reward/feet_proximity"] = r_feet_proximity.mean()
            self.extras["Reward/joint_default"] = r_joint_default.mean()
            self.extras["Reward/symmetry"] = r_symmetry.mean()
            self.extras["Metrics/height"] = height.mean()
            self.extras["Metrics/xy_drift"] = xy_drift.mean()
            self.extras["Metrics/feet_spread"] = feet_spread.mean()

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

            # Also terminate if drifted too far
            xy_drift = torch.norm(base_pos[:, :2] - self.spawn_positions, dim=-1)
            too_far = xy_drift > 1.0

            terminated = too_low | too_high | too_tilted | too_far
            time_out = self.episode_length_buf >= self.max_episode_length

            return terminated, time_out

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if len(env_ids) == 0:
                return

            robot = self.robot

            # Reset position with small random variation
            pos = torch.tensor([0.0, 0.0, 0.8], device=self.device).expand(len(env_ids), -1).clone()
            pos = pos + torch.randn_like(pos) * 0.02
            pos[:, 2] = 0.8

            quat = torch.zeros(len(env_ids), 4, device=self.device)
            quat[:, 3] = 1.0

            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

            # Reset joints to default standing pose
            default_pos = robot.data.default_joint_pos[env_ids]
            robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos), None, env_ids)

            # Store spawn positions for drift tracking
            self.spawn_positions[env_ids] = pos[:, :2].clone()

            # Randomize target height
            self.target_heights[env_ids] = torch.rand(len(env_ids), device=self.device) * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN

            # Reset buffers
            self.previous_actions[env_ids] = 0.0
            self._prev_actions[env_ids] = 0.0
            self.episode_rewards[env_ids] = 0.0
            self.episode_lengths[env_ids] = 0.0

        def get_episode_stats(self, terminated, truncated):
            done = terminated | truncated
            done_indices = done.nonzero(as_tuple=False).squeeze(-1)

            if len(done_indices) > 0:
                ep_rewards = self.episode_rewards[done_indices].clone()
                ep_lengths = self.episode_lengths[done_indices].clone()
                return ep_rewards.cpu().numpy(), ep_lengths.cpu().numpy()
            return np.array([]), np.array([])

    cfg = ULC_G1_Stage1_EnvCfg()
    cfg.scene.num_envs = num_envs
    env = ULC_G1_Stage1_Env(cfg)

    return env, cfg.observation_space, cfg.action_space


# ===== Training Loop =====
def train():
    device = "cuda:0"

    print(f"\n[INFO] Creating environment with {args_cli.num_envs} envs...")
    env, num_obs, num_actions = create_ulc_g1_env(args_cli.num_envs, device)

    print(f"[INFO] Num Envs: {args_cli.num_envs}")
    print(f"[INFO] Observations: {num_obs}, Actions: {num_actions}")

    # Create actor-critic network
    actor_critic = ActorCriticNetwork(num_obs, num_actions).to(device)

    # PPO settings
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
        entropy_coef=0.005,
        max_grad_norm=1.0,
    )

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "ulc", f"{args_cli.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"[INFO] Logging to: {log_dir}")

    # Training settings
    num_steps_per_env = 24
    max_iterations = args_cli.max_iterations
    checkpoint_interval = 200

    # std decay - MORE AGGRESSIVE
    std_decay_rate = 0.995

    # Resume from checkpoint
    start_iteration = 0
    best_reward = float('-inf')

    if args_cli.checkpoint:
        print(f"\n[INFO] Loading checkpoint: {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        actor_critic.load_state_dict(checkpoint["actor_critic"])
        ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint.get("iteration", 0)
        best_reward = checkpoint.get("best_reward", float('-inf'))
        print(f"[INFO] Resumed from iteration {start_iteration}")

    print(f"[INFO] Starting training from iteration {start_iteration}")
    print(f"[INFO] Target: {max_iterations} iterations (~2 hours)")
    print("=" * 80)

    # Initialize environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    start_time = datetime.now()

    for iteration in range(start_iteration, max_iterations):
        iter_start = datetime.now()

        # Collect rollout
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

        # Stack buffers
        obs_batch = torch.stack(obs_buffer)
        action_batch = torch.stack(action_buffer)
        reward_batch = torch.stack(reward_buffer)
        done_batch = torch.stack(done_buffer)
        value_batch = torch.stack(value_buffer)
        log_prob_batch = torch.stack(log_prob_buffer)

        # Compute next value
        with torch.no_grad():
            _, next_value = actor_critic(obs)
            next_value = next_value.squeeze(-1)

        # Compute GAE
        advantages, returns = ppo.compute_gae(reward_batch, value_batch, done_batch, next_value)

        # Flatten batches
        obs_flat = obs_batch.view(-1, num_obs)
        action_flat = action_batch.view(-1, num_actions)
        log_prob_flat = log_prob_batch.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        # PPO update
        update_info = ppo.update(obs_flat, action_flat, log_prob_flat, returns_flat, advantages_flat)

        # Decay std
        with torch.no_grad():
            actor_critic.log_std.data *= std_decay_rate
            actor_critic.log_std.data.clamp_(actor_critic.log_std_min, actor_critic.log_std_max)

        # Compute metrics
        mean_reward = reward_batch.mean().item()
        mean_std = actor_critic.get_std().mean().item()

        iter_time = (datetime.now() - iter_start).total_seconds()
        total_steps = num_steps_per_env * args_cli.num_envs
        steps_per_sec = total_steps / iter_time

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint_data = {
                "actor_critic": actor_critic.state_dict(),
                "optimizer": ppo.optimizer.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
            }
            torch.save(checkpoint_data, os.path.join(log_dir, "model_best.pt"))
            print(f"[BEST] New best model saved! Reward: {best_reward:.2f}")

        # Logging
        writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)
        writer.add_scalar("Loss/surrogate", update_info["actor_loss"], iteration)
        writer.add_scalar("Loss/value_function", update_info["critic_loss"], iteration)
        writer.add_scalar("Loss/entropy", update_info["entropy"], iteration)
        writer.add_scalar("Perf/total_fps", steps_per_sec, iteration)

        # Log reward components
        for key, value in env.extras.items():
            if isinstance(value, torch.Tensor):
                writer.add_scalar(key, value.item(), iteration)

        # Print progress
        if iteration % 10 == 0:
            elapsed = datetime.now() - start_time
            remaining_iters = max_iterations - iteration
            eta = elapsed / (iteration - start_iteration + 1) * remaining_iters if iteration > start_iteration else elapsed * remaining_iters

            print("#" * 80)
            print(f"                          Learning iteration {iteration}/{max_iterations}")
            print(f"                           Computation: {steps_per_sec:.0f} steps/s")
            print(f"                       Mean reward: {mean_reward:.2f}")
            print(f"                          Mean std: {mean_std:.3f}")
            print(f"                        Actor loss: {update_info['actor_loss']:.4f}")
            print(f"                       Critic loss: {update_info['critic_loss']:.4f}")
            print("-" * 80)
            print(f"                   Total timesteps: {(iteration + 1) * total_steps}")
            print(f"                           Elapsed: {str(elapsed).split('.')[0]}")
            print(f"                               ETA: {str(eta).split('.')[0]}")
            print("#" * 80)
            print()

        # Checkpoint
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

    # Save final model
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
    print("                               TRAINING COMPLETE")
    print(f"                               Best reward: {best_reward:.2f}")
    print(f"                Log dir: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
    simulation_app.close()