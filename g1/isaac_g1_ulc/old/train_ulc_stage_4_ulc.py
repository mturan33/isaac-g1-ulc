"""
ULC G1 Stage 4: Arm Tracking with Residual Actions
===================================================
Stage 3'ten devam: Kol kontrolü + Full CoM tracking

Key Features:
- Residual actions: arm_cmd + scale * tanh(residual)
- Full CoM tracker using link masses
- 10 arm joints (5 per arm): shoulder_pitch/roll/yaw, elbow_pitch/roll
- Stage 3 checkpoint'ten transfer

Architecture:
- Observations: 77 dims (Stage 3: 57 + arm states: 20)
- Actions: 22 (12 legs + 10 arms)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Arm curriculum levels
ARM_CURRICULUM = [
    # Level 0: Minimal arm movement (warm-up)
    {
        "vx": (0.0, 0.3), "vy": (-0.1, 0.1), "vyaw": (-0.2, 0.2),
        "pitch": (-0.15, 0.1), "roll": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        "arm_range": 0.1,  # Small arm movements
        "threshold": 18.0
    },
    # Level 1: Light arm control
    {
        "vx": (0.0, 0.5), "vy": (-0.15, 0.15), "vyaw": (-0.3, 0.3),
        "pitch": (-0.2, 0.15), "roll": (-0.15, 0.15), "yaw": (-0.2, 0.2),
        "arm_range": 0.3,
        "threshold": 20.0
    },
    # Level 2: Medium arm control
    {
        "vx": (-0.2, 0.7), "vy": (-0.2, 0.2), "vyaw": (-0.4, 0.4),
        "pitch": (-0.3, 0.2), "roll": (-0.15, 0.15), "yaw": (-0.3, 0.3),
        "arm_range": 0.5,
        "threshold": 22.0
    },
    # Level 3: Full range
    {
        "vx": (-0.3, 1.0), "vy": (-0.3, 0.3), "vyaw": (-0.5, 0.5),
        "pitch": (-0.35, 0.25), "roll": (-0.2, 0.2), "yaw": (-0.4, 0.4),
        "arm_range": 0.8,  # Large arm movements
        "threshold": None
    },
]

# Default values
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5

# G1 Arm joint limits (approximate, in radians)
ARM_JOINT_LIMITS = {
    "shoulder_pitch": (-2.0, 2.0),   # Forward/backward
    "shoulder_roll": (-1.5, 1.5),    # In/out
    "shoulder_yaw": (-1.5, 1.5),     # Rotation
    "elbow_pitch": (-2.0, 0.0),      # Bend (negative = bent)
    "elbow_roll": (-1.5, 1.5),       # Forearm rotation
}

# Default arm pose (neutral hanging position)
# Note: elbow_pitch limits are [-0.227, 3.421], so use -0.2 instead of -0.3
DEFAULT_ARM_POSE = [0.0, 0.0, 0.0, -0.2, 0.0]  # per arm

# Residual action scales (how much the policy can deviate from command)
RESIDUAL_SCALES = [0.5, 0.3, 0.3, 0.4, 0.3]  # per joint type

# Reward weights
REWARD_WEIGHTS = {
    # Velocity tracking
    "vx": 2.5,
    "vy": 1.5,
    "vyaw": 1.5,

    # Gait quality
    "gait": 2.0,
    "symmetry": 1.0,

    # Posture
    "height": 2.0,
    "base_orientation": 1.5,

    # Torso tracking (from Stage 3)
    "torso_pitch": 3.0,
    "torso_roll": 2.0,
    "torso_yaw": 1.5,

    # NEW: Arm tracking
    "left_arm": 4.0,
    "right_arm": 4.0,

    # NEW: CoM stability (enhanced)
    "com_stability": 4.0,

    # Penalties
    "smooth_legs": -0.01,
    "smooth_arms": -0.005,  # Gentler for arms
    "torque": -0.0003,
    "alive": 0.5,
}

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 4: Arm Control")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--stage3_checkpoint", type=str, required=True,
                        help="Path to Stage 3 best model (required)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 4 checkpoint")
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage4")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()

args_cli = parse_args()

# ============================================================================
# ISAAC LAB IMPORTS
# ============================================================================

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_apply_inverse
from torch.utils.tensorboard import SummaryWriter


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

print("=" * 80)
print("ULC G1 STAGE 4 - ARM CONTROL")
print("=" * 80)
print(f"Stage 3 checkpoint: {args_cli.stage3_checkpoint}")
print(f"\nObservation space: 77 (Stage 3: 57 + arms: 20)")
print(f"Action space: 22 (12 legs + 10 arms)")
print(f"\nResidual scales: {RESIDUAL_SCALES}")
print("=" * 80)

# ============================================================================
# NEURAL NETWORK
# ============================================================================

class ActorCritic(nn.Module):
    """
    Expanded network for arm control.
    Stage 3: 57 obs, 12 act → Stage 4: 77 obs, 22 act
    """

    def __init__(self, num_obs=77, num_act=22, hidden=[512, 256, 128]):
        super().__init__()

        # Actor
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        # Critic
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

        # Separate log_std for legs and arms
        self.log_std = nn.Parameter(torch.zeros(num_act))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def act(self, x, deterministic=False):
        mean = self.actor(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, x, actions):
        mean, value = self.forward(x)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return value.squeeze(-1), dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# ============================================================================
# PPO TRAINER
# ============================================================================

class PPO:
    def __init__(self, net, device, lr=3e-4):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, args_cli.max_iterations, eta_min=1e-5
        )

    def gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        gamma, lam = 0.99, 0.95

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        return advantages, advantages + values

    def update(self, obs, actions, old_log_probs, returns, advantages, old_values):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                values, log_probs, entropy = self.net.evaluate(
                    obs[mb_idx], actions[mb_idx]
                )

                ratio = (log_probs - old_log_probs[mb_idx]).exp()
                surr1 = ratio * advantages[mb_idx]
                surr2 = ratio.clamp(0.8, 1.2) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                value_clipped = old_values[mb_idx] + (values - old_values[mb_idx]).clamp(-0.2, 0.2)
                critic_loss = 0.5 * torch.max(
                    (values - returns[mb_idx]) ** 2,
                    (value_clipped - returns[mb_idx]) ** 2
                ).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.sched.step()

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "lr": self.sched.get_last_lr()[0],
        }


# ============================================================================
# ENVIRONMENT
# ============================================================================

def create_env(num_envs, device):
    """Create Stage 4 environment with arm control"""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
        )

        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=G1_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=10.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8),
                joint_pos={
                    # Legs
                    "left_hip_pitch_joint": -0.2,
                    "right_hip_pitch_joint": -0.2,
                    "left_hip_roll_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.4,
                    "right_knee_joint": 0.4,
                    "left_ankle_pitch_joint": -0.2,
                    "right_ankle_pitch_joint": -0.2,
                    "left_ankle_roll_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    # Arms - default pose
                    "left_shoulder_pitch_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "left_elbow_pitch_joint": -0.2,
                    "right_elbow_pitch_joint": -0.2,
                    "left_elbow_roll_joint": 0.0,
                    "right_elbow_roll_joint": 0.0,
                    "torso_joint": 0.0,
                },
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
                    stiffness=150.0,
                    damping=15.0,
                ),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                    stiffness=80.0,  # Slightly higher for better tracking
                    damping=8.0,
                ),
                "torso": ImplicitActuatorCfg(
                    joint_names_expr=["torso_joint"],
                    stiffness=100.0,
                    damping=10.0,
                ),
            },
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 15.0
        action_space = 22  # 12 legs + 10 arms
        observation_space = 77  # Stage 3 (57) + arm states (20)
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class Stage4Env(DirectRLEnv):
        cfg: EnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            # Get joint indices
            joint_names = self.robot.joint_names

            # Leg joints (same as Stage 3)
            leg_names = [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
            ]
            self.leg_idx = torch.tensor(
                [joint_names.index(n) for n in leg_names if n in joint_names],
                device=self.device
            )

            # Arm joints (NEW)
            left_arm_names = [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
                "left_elbow_roll_joint",
            ]
            right_arm_names = [
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
                "right_elbow_roll_joint",
            ]

            self.left_arm_idx = torch.tensor(
                [joint_names.index(n) for n in left_arm_names if n in joint_names],
                device=self.device
            )
            self.right_arm_idx = torch.tensor(
                [joint_names.index(n) for n in right_arm_names if n in joint_names],
                device=self.device
            )
            self.arm_idx = torch.cat([self.left_arm_idx, self.right_arm_idx])

            print(f"[Env] Leg joint indices: {self.leg_idx.tolist()}")
            print(f"[Env] Left arm indices: {self.left_arm_idx.tolist()}")
            print(f"[Env] Right arm indices: {self.right_arm_idx.tolist()}")

            # Default positions
            self.default_leg = torch.tensor(
                [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
                device=self.device
            )
            self.default_arm = torch.tensor(
                DEFAULT_ARM_POSE * 2,  # Left + Right
                device=self.device
            )

            # Residual scales
            self.residual_scales = torch.tensor(
                RESIDUAL_SCALES * 2,  # Left + Right
                device=self.device
            )

            # Body indices for CoM calculation
            body_names = self.robot.body_names
            self.pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0

            # Find foot bodies
            self.left_foot_idx = None
            self.right_foot_idx = None
            for i, name in enumerate(body_names):
                if "left" in name.lower() and ("foot" in name.lower() or "ankle_roll" in name.lower()):
                    self.left_foot_idx = i
                if "right" in name.lower() and ("foot" in name.lower() or "ankle_roll" in name.lower()):
                    self.right_foot_idx = i

            # Find arm bodies for CoM (upper arm, forearm, hand)
            self.arm_body_indices = []
            arm_keywords = ["shoulder", "elbow", "palm", "arm"]
            for i, name in enumerate(body_names):
                if any(kw in name.lower() for kw in arm_keywords):
                    self.arm_body_indices.append(i)

            print(f"[Env] Pelvis idx: {self.pelvis_idx}")
            print(f"[Env] Feet idx: L={self.left_foot_idx}, R={self.right_foot_idx}")
            print(f"[Env] Arm body indices: {self.arm_body_indices}")

            # Curriculum state
            self.curr_level = 0
            self.curr_history = []

            # Commands
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

            # Arm commands (5 joints per arm)
            self.left_arm_cmd = torch.zeros(self.num_envs, 5, device=self.device)
            self.right_arm_cmd = torch.zeros(self.num_envs, 5, device=self.device)

            # Gait phase
            self.phase = torch.zeros(self.num_envs, device=self.device)

            # Action history
            self.prev_actions = torch.zeros(self.num_envs, 22, device=self.device)
            self._prev_actions = torch.zeros(self.num_envs, 22, device=self.device)

            print(f"[Env] Stage 4 initialized with {self.num_envs} envs")
            print(f"[Env] Observation space: {self.cfg.observation_space}")
            print(f"[Env] Action space: {self.cfg.action_space}")

        @property
        def robot(self):
            return self.scene["robot"]

        def get_torso_euler(self) -> torch.Tensor:
            quat = self.robot.data.root_quat_w
            return quat_to_euler_xyz(quat)

        def get_com_position(self) -> torch.Tensor:
            """
            Calculate approximate center of mass.
            Uses pelvis + arm contribution for better accuracy.
            """
            # Base CoM (pelvis)
            com = self.robot.data.root_pos_w.clone()

            # Add arm mass contribution (approximate)
            # Arms are ~5% of total mass each
            if len(self.arm_body_indices) > 0:
                arm_pos = self.robot.data.body_pos_w[:, self.arm_body_indices]
                arm_center = arm_pos.mean(dim=1)
                # Weighted average: 90% pelvis, 10% arms
                com = 0.90 * com + 0.10 * arm_center

            return com

        def get_feet_center(self) -> torch.Tensor:
            if self.left_foot_idx is not None and self.right_foot_idx is not None:
                left_pos = self.robot.data.body_pos_w[:, self.left_foot_idx]
                right_pos = self.robot.data.body_pos_w[:, self.right_foot_idx]
                return (left_pos + right_pos) / 2
            else:
                root_pos = self.robot.data.root_pos_w.clone()
                root_pos[:, 2] = 0
                return root_pos

        def update_curriculum(self, mean_reward):
            self.curr_history.append(mean_reward)

            if len(self.curr_history) >= 100:
                avg_reward = np.mean(self.curr_history[-100:])
                threshold = ARM_CURRICULUM[self.curr_level]["threshold"]

                if threshold is not None and avg_reward > threshold:
                    if self.curr_level < len(ARM_CURRICULUM) - 1:
                        self.curr_level += 1
                        print(f"\n{'='*60}")
                        print(f"LEVEL UP! Now at Level {self.curr_level}")
                        lv = ARM_CURRICULUM[self.curr_level]
                        print(f"  Velocity: vx={lv['vx']}")
                        print(f"  Arm range: {lv['arm_range']}")
                        print(f"{'='*60}\n")
                        self.curr_history = []

        def _sample_commands(self, env_ids):
            n = len(env_ids)
            lv = ARM_CURRICULUM[self.curr_level]

            # Velocity commands
            self.vel_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (lv["vx"][1] - lv["vx"][0]) + lv["vx"][0]
            self.vel_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (lv["vy"][1] - lv["vy"][0]) + lv["vy"][0]
            self.vel_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (lv["vyaw"][1] - lv["vyaw"][0]) + lv["vyaw"][0]

            # Torso commands
            self.torso_cmd[env_ids, 0] = torch.rand(n, device=self.device) * (lv["roll"][1] - lv["roll"][0]) + lv["roll"][0]
            self.torso_cmd[env_ids, 1] = torch.rand(n, device=self.device) * (lv["pitch"][1] - lv["pitch"][0]) + lv["pitch"][0]
            self.torso_cmd[env_ids, 2] = torch.rand(n, device=self.device) * (lv["yaw"][1] - lv["yaw"][0]) + lv["yaw"][0]

            # Arm commands (around default pose)
            arm_range = lv["arm_range"]
            default_arm = torch.tensor(DEFAULT_ARM_POSE, device=self.device)

            for i in range(5):
                # Left arm
                self.left_arm_cmd[env_ids, i] = default_arm[i] + (torch.rand(n, device=self.device) * 2 - 1) * arm_range * RESIDUAL_SCALES[i]
                # Right arm
                self.right_arm_cmd[env_ids, i] = default_arm[i] + (torch.rand(n, device=self.device) * 2 - 1) * arm_range * RESIDUAL_SCALES[i]

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()

            # Split actions: legs (0:12) and arms (12:22)
            leg_actions = actions[:, :12]
            arm_actions = actions[:, 12:]

            # Compute leg targets (same as Stage 3)
            target_pos = self.robot.data.default_joint_pos.clone()
            target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4

            # Compute arm targets with RESIDUAL ACTIONS
            # final_arm_pos = arm_cmd + scale * tanh(residual)
            arm_cmd = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)
            arm_residual = arm_actions * self.residual_scales  # Scale residuals
            arm_target = arm_cmd + torch.tanh(arm_residual) * self.residual_scales

            target_pos[:, self.arm_idx] = arm_target

            self.robot.set_joint_position_target(target_pos)

            # Update gait phase
            self.phase = (self.phase + GAIT_FREQUENCY * self.cfg.sim.dt * self.cfg.decimation) % 1.0

            # Store action history
            self._prev_actions = self.prev_actions.clone()
            self.prev_actions = actions.clone()

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            """
            Build observation vector.
            Stage 4: 77 dims = Stage 3 (57) + arm states (20)
            """
            robot = self.robot
            quat = robot.data.root_quat_w

            # Body-frame velocities
            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

            # Projected gravity
            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(quat, gravity_vec)

            # Leg joint states
            leg_pos = robot.data.joint_pos[:, self.leg_idx]
            leg_vel = robot.data.joint_vel[:, self.leg_idx]

            # Arm joint states (NEW)
            left_arm_pos = robot.data.joint_pos[:, self.left_arm_idx]
            right_arm_pos = robot.data.joint_pos[:, self.right_arm_idx]

            # Gait phase
            gait_phase = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)
            ], dim=-1)

            # Torso orientation
            torso_euler = self.get_torso_euler()

            # Build observation
            obs = torch.cat([
                # Stage 3 observations (57 dims)
                lin_vel_b,                      # 3
                ang_vel_b,                      # 3
                proj_gravity,                   # 3
                leg_pos,                        # 12
                leg_vel,                        # 12
                self.height_cmd.unsqueeze(-1),  # 1
                self.vel_cmd,                   # 3
                gait_phase,                     # 2
                self.prev_actions[:, :12],      # 12 (leg actions)
                self.torso_cmd,                 # 3
                torso_euler,                    # 3

                # NEW: Arm states (20 dims)
                left_arm_pos,                   # 5
                right_arm_pos,                  # 5
                self.left_arm_cmd,              # 5
                self.right_arm_cmd,             # 5
            ], dim=-1)

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self) -> torch.Tensor:
            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w

            # Body-frame velocities
            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

            # Joint states
            leg_pos = robot.data.joint_pos[:, self.leg_idx]
            leg_vel = robot.data.joint_vel[:, self.leg_idx]

            left_arm_pos = robot.data.joint_pos[:, self.left_arm_idx]
            right_arm_pos = robot.data.joint_pos[:, self.right_arm_idx]

            # Torso orientation
            torso_euler = self.get_torso_euler()

            # ==================== VELOCITY TRACKING ====================
            r_vx = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
            r_vy = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.vel_cmd[:, 1]) ** 2)
            r_vyaw = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.vel_cmd[:, 2]) ** 2)

            # ==================== GAIT QUALITY ====================
            left_knee, right_knee = leg_pos[:, 6], leg_pos[:, 7]
            phase = self.phase
            left_swing = (phase < 0.5).float()
            right_swing = (phase >= 0.5).float()

            knee_target_swing = 0.6
            knee_target_stance = 0.3
            knee_err = (
                (left_knee - (left_swing * knee_target_swing + (1 - left_swing) * knee_target_stance)) ** 2 +
                (right_knee - (right_swing * knee_target_swing + (1 - right_swing) * knee_target_stance)) ** 2
            )
            r_gait = torch.exp(-3.0 * knee_err)

            # Symmetry
            leg_actions = self.actions[:, :12]
            left_actions = leg_actions[:, 0::2]
            right_actions = leg_actions[:, 1::2]
            r_symmetry = torch.exp(-1.0 * (left_actions - right_actions).pow(2).mean(-1))

            # ==================== HEIGHT TRACKING ====================
            r_height = torch.exp(-10.0 * (pos[:, 2] - self.height_cmd) ** 2)

            # ==================== BASE ORIENTATION ====================
            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(quat, gravity_vec)
            base_tilt_error = proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2
            r_base_orientation = torch.exp(-3.0 * base_tilt_error)

            # ==================== TORSO TRACKING ====================
            roll_err = (torso_euler[:, 0] - self.torso_cmd[:, 0]) ** 2
            pitch_err = (torso_euler[:, 1] - self.torso_cmd[:, 1]) ** 2
            yaw_err = (torso_euler[:, 2] - self.torso_cmd[:, 2]) ** 2

            r_torso_roll = torch.exp(-5.0 * roll_err)
            r_torso_pitch = torch.exp(-5.0 * pitch_err)
            r_torso_yaw = torch.exp(-3.0 * yaw_err)

            # ==================== ARM TRACKING (NEW) ====================
            left_arm_err = (left_arm_pos - self.left_arm_cmd).pow(2).sum(-1)
            right_arm_err = (right_arm_pos - self.right_arm_cmd).pow(2).sum(-1)

            r_left_arm = torch.exp(-3.0 * left_arm_err)
            r_right_arm = torch.exp(-3.0 * right_arm_err)

            # ==================== CoM STABILITY (ENHANCED) ====================
            com_pos = self.get_com_position()
            feet_center = self.get_feet_center()

            com_offset = com_pos[:, :2] - feet_center[:, :2]
            com_dist = com_offset.norm(dim=-1)
            r_com_stability = torch.exp(-8.0 * com_dist)  # Stricter than Stage 3

            # ==================== PENALTIES ====================
            # Leg action smoothness
            leg_action_diff = self.actions[:, :12] - self._prev_actions[:, :12]
            p_smooth_legs = leg_action_diff.pow(2).sum(-1)

            # Arm action smoothness (gentler)
            arm_action_diff = self.actions[:, 12:] - self._prev_actions[:, 12:]
            p_smooth_arms = arm_action_diff.pow(2).sum(-1)

            # Torque penalty
            p_torque = (leg_vel.abs() * self.actions[:, :12].abs()).sum(-1)

            # ==================== TOTAL REWARD ====================
            reward = (
                REWARD_WEIGHTS["vx"] * r_vx +
                REWARD_WEIGHTS["vy"] * r_vy +
                REWARD_WEIGHTS["vyaw"] * r_vyaw +
                REWARD_WEIGHTS["gait"] * r_gait +
                REWARD_WEIGHTS["symmetry"] * r_symmetry +
                REWARD_WEIGHTS["height"] * r_height +
                REWARD_WEIGHTS["base_orientation"] * r_base_orientation +
                REWARD_WEIGHTS["torso_pitch"] * r_torso_pitch +
                REWARD_WEIGHTS["torso_roll"] * r_torso_roll +
                REWARD_WEIGHTS["torso_yaw"] * r_torso_yaw +
                REWARD_WEIGHTS["left_arm"] * r_left_arm +
                REWARD_WEIGHTS["right_arm"] * r_right_arm +
                REWARD_WEIGHTS["com_stability"] * r_com_stability +
                REWARD_WEIGHTS["smooth_legs"] * p_smooth_legs +
                REWARD_WEIGHTS["smooth_arms"] * p_smooth_arms +
                REWARD_WEIGHTS["torque"] * p_torque +
                REWARD_WEIGHTS["alive"]
            )

            # Store extras for logging
            self.extras = {
                "R/vx": r_vx.mean().item(),
                "R/gait": r_gait.mean().item(),
                "R/height": r_height.mean().item(),
                "R/torso_pitch": r_torso_pitch.mean().item(),
                "R/left_arm": r_left_arm.mean().item(),
                "R/right_arm": r_right_arm.mean().item(),
                "R/com_stability": r_com_stability.mean().item(),
                "M/height": pos[:, 2].mean().item(),
                "M/vx": lin_vel_b[:, 0].mean().item(),
                "M/pitch": torso_euler[:, 1].mean().item(),
                "M/com_dist": com_dist.mean().item(),
                "M/left_arm_err": left_arm_err.mean().item(),
                "M/right_arm_err": right_arm_err.mean().item(),
                "curriculum_level": self.curr_level,
            }

            return reward.clamp(-10, 35)

        def _get_dones(self) -> tuple:
            height = self.robot.data.root_pos_w[:, 2]

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(self.robot.data.root_quat_w, gravity_vec)

            fallen = (height < 0.3) | (height > 1.2)
            bad_orientation = proj_gravity[:, :2].abs().max(dim=-1)[0] > 0.7

            terminated = fallen | bad_orientation
            truncated = self.episode_length_buf >= self.max_episode_length

            return terminated, truncated

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)

            if len(env_ids) == 0:
                return

            # Reset pose
            default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(len(env_ids), -1).clone()
            default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(env_ids), -1)

            self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

            # Reset joint positions
            default_joint_pos = self.robot.data.default_joint_pos[env_ids]
            self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

            # Sample new commands
            self._sample_commands(env_ids)

            # Reset phase and actions
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            self.prev_actions[env_ids] = 0
            self._prev_actions[env_ids] = 0

    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    return Stage4Env(cfg), 77, 22


# ============================================================================
# WEIGHT TRANSFER FROM STAGE 3
# ============================================================================

def transfer_stage3_weights(net_stage4, stage3_checkpoint_path, device):
    """
    Transfer weights from Stage 3 to Stage 4.

    Stage 3: obs=57, act=12
    Stage 4: obs=77, act=22

    Strategy:
    - First layer: Expand input weights (57 → 77)
    - Last layer: Expand output weights (12 → 22)
    - Other layers: Direct copy
    """
    print(f"\n[Transfer] Loading Stage 3 checkpoint: {stage3_checkpoint_path}")

    checkpoint = torch.load(stage3_checkpoint_path, map_location=device, weights_only=False)
    stage3_state = checkpoint["actor_critic"]
    stage4_state = net_stage4.state_dict()

    transferred = 0
    expanded = 0

    for key in stage3_state:
        if key not in stage4_state:
            print(f"  [Skip] {key} not in Stage 4")
            continue

        s3_shape = stage3_state[key].shape
        s4_shape = stage4_state[key].shape

        if s3_shape == s4_shape:
            # Direct copy
            stage4_state[key] = stage3_state[key]
            transferred += 1
        elif key in ["actor.0.weight", "critic.0.weight"]:
            # First layer - expand inputs (57 → 77)
            print(f"  [Expand Input] {key}: {s3_shape} → {s4_shape}")
            stage4_state[key][:, :s3_shape[1]] = stage3_state[key]
            nn.init.orthogonal_(stage4_state[key][:, s3_shape[1]:], gain=0.1)
            expanded += 1
        elif key == "actor.9.weight":
            # Last actor layer - expand outputs (12 → 22)
            print(f"  [Expand Output] {key}: {s3_shape} → {s4_shape}")
            stage4_state[key][:s3_shape[0], :] = stage3_state[key]
            nn.init.orthogonal_(stage4_state[key][s3_shape[0]:, :], gain=0.01)
            expanded += 1
        elif key == "actor.9.bias":
            # Last actor bias - expand (12 → 22)
            print(f"  [Expand Bias] {key}: {s3_shape} → {s4_shape}")
            stage4_state[key][:s3_shape[0]] = stage3_state[key]
            stage4_state[key][s3_shape[0]:] = 0
            expanded += 1
        elif key == "log_std":
            # Log std - expand (12 → 22)
            print(f"  [Expand Std] {key}: {s3_shape} → {s4_shape}")
            stage4_state[key][:s3_shape[0]] = stage3_state[key]
            stage4_state[key][s3_shape[0]:] = np.log(0.5)  # Higher exploration for arms
            expanded += 1
        else:
            print(f"  [Skip] Shape mismatch: {key} {s3_shape} vs {s4_shape}")

    net_stage4.load_state_dict(stage4_state)

    print(f"\n[Transfer] Completed: {transferred} transferred, {expanded} expanded")

    if "best_reward" in checkpoint:
        print(f"[Transfer] Stage 3 best reward: {checkpoint['best_reward']:.2f}")
    if "curriculum_level" in checkpoint:
        print(f"[Transfer] Stage 3 curriculum level: {checkpoint['curriculum_level']}")

    return checkpoint


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    device = "cuda:0"

    print(f"\n[INFO] Creating environment with {args_cli.num_envs} envs...")
    env, num_obs, num_act = create_env(args_cli.num_envs, device)

    print(f"[INFO] Creating network: obs={num_obs}, act={num_act}")
    net = ActorCritic(num_obs, num_act).to(device)

    # Transfer from Stage 3
    stage3_info = transfer_stage3_weights(net, args_cli.stage3_checkpoint, device)

    # Resume from Stage 4 checkpoint if provided
    start_iter = 0
    if args_cli.checkpoint:
        print(f"\n[INFO] Resuming from: {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["actor_critic"])
        start_iter = ckpt.get("iteration", 0)
        env.curr_level = ckpt.get("curriculum_level", 0)
        print(f"[INFO] Resuming from iteration {start_iter}, level {env.curr_level}")

    # Create PPO trainer
    ppo = PPO(net, device)

    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"\n[INFO] Logging to: {log_dir}")

    # Training state
    best_reward = float('-inf')

    # Set initial exploration (higher for arms)
    net.log_std.data[:12].fill_(np.log(0.5))   # Legs
    net.log_std.data[12:].fill_(np.log(0.6))   # Arms - more exploration

    # Initial reset
    obs, _ = env.reset()
    obs = obs["policy"]

    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("STARTING STAGE 4 TRAINING")
    print("=" * 80 + "\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        iter_start = datetime.now()

        # Collect rollouts
        obs_buffer = []
        act_buffer = []
        rew_buffer = []
        done_buffer = []
        val_buffer = []
        logp_buffer = []

        rollout_steps = 24

        for _ in range(rollout_steps):
            with torch.no_grad():
                mean, value = net(obs)
                std = net.log_std.clamp(-2, 1).exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

            obs_buffer.append(obs)
            act_buffer.append(action)
            val_buffer.append(value.squeeze(-1))
            logp_buffer.append(log_prob)

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]

            rew_buffer.append(reward)
            done_buffer.append((terminated | truncated).float())

        # Stack buffers
        obs_buffer = torch.stack(obs_buffer)
        act_buffer = torch.stack(act_buffer)
        rew_buffer = torch.stack(rew_buffer)
        done_buffer = torch.stack(done_buffer)
        val_buffer = torch.stack(val_buffer)
        logp_buffer = torch.stack(logp_buffer)

        # Compute returns
        with torch.no_grad():
            _, next_value = net(obs)
            next_value = next_value.squeeze(-1)

        advantages, returns = ppo.gae(rew_buffer, val_buffer, done_buffer, next_value)

        # PPO update
        update_info = ppo.update(
            obs_buffer.view(-1, num_obs),
            act_buffer.view(-1, num_act),
            logp_buffer.view(-1),
            returns.view(-1),
            advantages.view(-1),
            val_buffer.view(-1),
        )

        # Anneal exploration
        progress = iteration / args_cli.max_iterations
        leg_std = 0.5 + (0.15 - 0.5) * progress
        arm_std = 0.6 + (0.2 - 0.6) * progress  # Arms need more exploration
        net.log_std.data[:12].fill_(np.log(leg_std))
        net.log_std.data[12:].fill_(np.log(arm_std))

        # Calculate stats
        mean_reward = rew_buffer.mean().item()
        env.update_curriculum(mean_reward)

        iter_time = (datetime.now() - iter_start).total_seconds()
        fps = rollout_steps * args_cli.num_envs / iter_time

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "actor_critic": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
            }, f"{log_dir}/model_best.pt")
            print(f"[BEST] New best reward: {best_reward:.2f}")

        # TensorBoard logging
        writer.add_scalar("Train/reward", mean_reward, iteration)
        writer.add_scalar("Train/leg_std", leg_std, iteration)
        writer.add_scalar("Train/arm_std", arm_std, iteration)
        writer.add_scalar("Train/best_reward", best_reward, iteration)
        writer.add_scalar("Loss/actor", update_info["actor_loss"], iteration)
        writer.add_scalar("Loss/critic", update_info["critic_loss"], iteration)
        writer.add_scalar("Curriculum/level", env.curr_level, iteration)

        for key, value in env.extras.items():
            if key != "curriculum_level":
                writer.add_scalar(f"Env/{key}", value, iteration)

        # Console logging
        if iteration % 10 == 0:
            elapsed = datetime.now() - start_time
            eta = elapsed / (iteration - start_iter + 1) * (args_cli.max_iterations - iteration)

            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Lv={env.curr_level} | "
                f"L_arm={env.extras.get('M/left_arm_err', 0):.3f} | "
                f"R_arm={env.extras.get('M/right_arm_err', 0):.3f} | "
                f"CoM={env.extras.get('M/com_dist', 0):.3f} | "
                f"FPS={fps:.0f} | "
                f"{str(elapsed).split('.')[0]} / {str(eta).split('.')[0]}"
            )

        # Periodic checkpoints
        if (iteration + 1) % 500 == 0:
            torch.save({
                "actor_critic": net.state_dict(),
                "optimizer": ppo.opt.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
            }, f"{log_dir}/model_{iteration + 1}.pt")

        writer.flush()

    # Final save
    torch.save({
        "actor_critic": net.state_dict(),
        "iteration": args_cli.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
    }, f"{log_dir}/model_final.pt")

    writer.close()
    env.close()

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE!")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Log Dir: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
    simulation_app.close()