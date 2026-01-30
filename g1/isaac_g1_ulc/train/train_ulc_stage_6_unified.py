"""
ULC G1 Stage 6: Unified Loco-Manipulation Training
===================================================
Loco policy + Arm policy AYNI ANDA eğitilir.

Strateji:
- Stage 3'ten loco ağırlıkları TRANSFER (frozen değil!)
- Arm policy sıfırdan
- Curriculum: sabit → yavaş yürüme → normal yürüme
- Loco, kol hareketliyken denge öğrenir
- Arm, hareket eden base ile reach öğrenir

KULLANIM:
./isaaclab.bat -p source/isaaclab_tasks/.../train/train_ulc_stage_6_unified.py \
    --stage3_checkpoint logs/ulc/ulc_g1_stage3_2026-01-09_14-28-58/model_best.pt \
    --num_envs 4096 --max_iterations 10000 --headless
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

# Joint names
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]

# Curriculum levels - sabitten hareketliye
CURRICULUM = [
    # Level 0: Sabit duruş, küçük workspace
    {
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.15, 0.22), "arm_height": (-0.05, 0.10),
        "pos_threshold": 0.05, "success_rate": 0.50, "min_reaches": 3000, "min_steps": 2000,
    },
    # Level 1: Sabit, orta workspace
    {
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.18, 0.28), "arm_height": (-0.08, 0.15),
        "pos_threshold": 0.05, "success_rate": 0.55, "min_reaches": 5000, "min_steps": 2500,
    },
    # Level 2: Çok yavaş yürüme başlangıcı
    {
        "vx": (0.0, 0.15), "vy": (-0.05, 0.05), "vyaw": (-0.1, 0.1),
        "arm_radius": (0.18, 0.30), "arm_height": (-0.10, 0.18),
        "pos_threshold": 0.05, "success_rate": 0.50, "min_reaches": 6000, "min_steps": 3000,
    },
    # Level 3: Yavaş yürüme
    {
        "vx": (0.0, 0.25), "vy": (-0.08, 0.08), "vyaw": (-0.15, 0.15),
        "arm_radius": (0.18, 0.32), "arm_height": (-0.10, 0.20),
        "pos_threshold": 0.05, "success_rate": 0.50, "min_reaches": 8000, "min_steps": 3500,
    },
    # Level 4: Normal yürüme
    {
        "vx": (0.0, 0.4), "vy": (-0.1, 0.1), "vyaw": (-0.2, 0.2),
        "arm_radius": (0.18, 0.35), "arm_height": (-0.12, 0.22),
        "pos_threshold": 0.05, "success_rate": 0.45, "min_reaches": 10000, "min_steps": 4000,
    },
    # Level 5: Hızlı yürüme
    {
        "vx": (-0.1, 0.6), "vy": (-0.15, 0.15), "vyaw": (-0.3, 0.3),
        "arm_radius": (0.18, 0.38), "arm_height": (-0.15, 0.25),
        "pos_threshold": 0.05, "success_rate": 0.40, "min_reaches": 12000, "min_steps": 5000,
    },
    # Level 6: Full range
    {
        "vx": (-0.2, 0.8), "vy": (-0.2, 0.2), "vyaw": (-0.4, 0.4),
        "arm_radius": (0.18, 0.40), "arm_height": (-0.15, 0.28),
        "pos_threshold": 0.05, "success_rate": None, "min_reaches": None, "min_steps": None,
    },
]

# Default values
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
REACH_THRESHOLD = 0.05  # 8cm → 5cm (daha zor)

# Shoulder offset (right shoulder relative to root)
SHOULDER_OFFSET = torch.tensor([0.0, -0.174, 0.259])

# Reward weights
REWARD_WEIGHTS = {
    # Locomotion rewards
    "loco_vx": 2.5,
    "loco_vy": 1.0,
    "loco_vyaw": 1.0,
    "loco_height": 2.0,
    "loco_orientation": 2.5,
    "loco_gait": 1.5,

    # Arm rewards
    "arm_distance": 3.0,
    "arm_reaching": 15.0,
    "arm_smooth": 1.0,

    # Joint coordination
    "balance": 2.0,  # CoM over feet

    # Penalties
    "loco_action_rate": -0.01,
    "arm_action_rate": -0.02,
    "energy": -0.0005,
    "alive": 0.5,
}


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 6: Unified Training")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--stage3_checkpoint", type=str, default=None,
                        help="Optional: Transfer loco weights from Stage 3")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 6 checkpoint")
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage6")
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
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from torch.utils.tensorboard import SummaryWriter

G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

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

print("=" * 80)
print("ULC G1 STAGE 6 - UNIFIED LOCO-MANIPULATION TRAINING")
print("=" * 80)
print(f"Stage 3 checkpoint: {args_cli.stage3_checkpoint}")
print("\nCurriculum Levels:")
for i, lv in enumerate(CURRICULUM):
    print(f"  Level {i}: vx={lv['vx']}, arm_radius={lv['arm_radius']}")
print("=" * 80)


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion policy: 57 obs → 12 leg actions (Stage 3 architecture)"""

    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x):
        return self.actor(x)

    def act(self, x, deterministic=False):
        mean = self.forward(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


class ArmActor(nn.Module):
    """Arm policy: 28 obs → 5 arm actions"""

    def __init__(self, num_obs=28, num_act=5, hidden=[256, 128, 64]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x):
        return self.actor(x)

    def act(self, x, deterministic=False):
        mean = self.forward(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


class UnifiedCritic(nn.Module):
    """Unified critic for both policies"""

    def __init__(self, num_obs=85, hidden=[512, 256, 128]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.critic(x).squeeze(-1)


class UnifiedActorCritic(nn.Module):
    """Combined network for PPO training"""

    def __init__(self, loco_obs=57, arm_obs=28, loco_act=12, arm_act=5):
        super().__init__()

        self.loco_actor = LocoActor(loco_obs, loco_act)
        self.arm_actor = ArmActor(arm_obs, arm_act)
        self.critic = UnifiedCritic(loco_obs + arm_obs)

        self.loco_obs_dim = loco_obs
        self.arm_obs_dim = arm_obs
        self.loco_act_dim = loco_act
        self.arm_act_dim = arm_act

    def forward(self, loco_obs, arm_obs):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)

        combined_obs = torch.cat([loco_obs, arm_obs], dim=-1)
        value = self.critic(combined_obs)

        return loco_mean, arm_mean, value

    def act(self, loco_obs, arm_obs, deterministic=False):
        loco_action = self.loco_actor.act(loco_obs, deterministic)
        arm_action = self.arm_actor.act(arm_obs, deterministic)
        return loco_action, arm_action

    def evaluate(self, loco_obs, arm_obs, loco_actions, arm_actions):
        loco_mean, arm_mean, value = self.forward(loco_obs, arm_obs)

        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()

        loco_dist = torch.distributions.Normal(loco_mean, loco_std)
        arm_dist = torch.distributions.Normal(arm_mean, arm_std)

        loco_logp = loco_dist.log_prob(loco_actions).sum(-1)
        arm_logp = arm_dist.log_prob(arm_actions).sum(-1)

        loco_entropy = loco_dist.entropy().sum(-1)
        arm_entropy = arm_dist.entropy().sum(-1)

        return value, loco_logp + arm_logp, loco_entropy + arm_entropy


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

        returns = advantages + values
        return advantages, returns

    def update(self, loco_obs, arm_obs, loco_actions, arm_actions,
               old_log_probs, returns, advantages, old_values):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = loco_obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                values, log_probs, entropy = self.net.evaluate(
                    loco_obs[mb_idx], arm_obs[mb_idx],
                    loco_actions[mb_idx], arm_actions[mb_idx]
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
                    "left_hip_pitch_joint": -0.2, "right_hip_pitch_joint": -0.2,
                    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
                    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0, "right_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
                    "left_elbow_pitch_joint": 0.0, "right_elbow_pitch_joint": 0.0,
                    "left_elbow_roll_joint": 0.0, "right_elbow_roll_joint": 0.0,
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
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 15.0
        action_space = 17  # 12 legs + 5 arm
        observation_space = 57  # Loco obs (arm obs computed separately)
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class Stage6Env(DirectRLEnv):
        cfg: EnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            # Joint indices
            joint_names = self.robot.joint_names

            self.leg_idx = torch.tensor(
                [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
                device=self.device
            )
            self.arm_idx = torch.tensor(
                [joint_names.index(n) for n in ARM_JOINT_NAMES if n in joint_names],
                device=self.device
            )

            # Default positions
            self.default_leg = torch.tensor(
                [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
                device=self.device
            )
            self.default_arm = self.robot.data.default_joint_pos[0, self.arm_idx].clone()

            # Find palm body for EE tracking
            body_names = self.robot.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right" in name.lower() and ("palm" in name.lower() or "hand" in name.lower() or "link6" in name.lower()):
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                for i, name in enumerate(body_names):
                    if "right" in name.lower() and "elbow" in name.lower():
                        self.palm_idx = i + 1
                        break
            if self.palm_idx is None:
                self.palm_idx = len(body_names) - 1

            print(f"[Stage6] Leg joints: {len(self.leg_idx)}")
            print(f"[Stage6] Arm joints: {len(self.arm_idx)}")
            print(f"[Stage6] Palm idx: {self.palm_idx}")

            # Commands
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)

            # Arm target
            self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

            # Gait phase
            self.phase = torch.zeros(self.num_envs, device=self.device)

            # Previous actions
            self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

            # Curriculum
            self.curr_level = 0
            self.reach_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.total_reaches = 0
            self.stage_reaches = 0
            self.stage_steps = 0

            # FIX: Prevent counting same reach multiple times
            self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            # Markers (lazy init)
            self._markers_initialized = False
            self.target_markers = None
            self.ee_markers = None
            self.outer_markers = None
            self.inner_markers = None
            self.shoulder_marker_vis = None
            self.num_wireframe_points = 24

        @property
        def robot(self):
            return self.scene["robot"]

        def _init_markers(self):
            """Create markers after simulation starts"""
            if self._markers_initialized:
                return

            # Target marker (yellow)
            self.target_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TargetMarkers",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 1.0, 0.0),
                            ),
                        ),
                    },
                )
            )

            # EE marker (green)
            self.ee_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/EEMarkers",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 0.0),
                            ),
                        ),
                    },
                )
            )

            # Outer workspace boundary (blue spheres)
            self.outer_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/OuterWorkspace",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.015,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.3, 1.0),
                                emissive_color=(0.0, 0.2, 0.5),
                            ),
                        ),
                    },
                )
            )

            # Inner exclusion zone (red spheres)
            self.inner_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/InnerExclusion",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.012,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0),
                                emissive_color=(0.8, 0.0, 0.0),
                            ),
                        ),
                    },
                )
            )

            # Shoulder center marker (white)
            self.shoulder_marker_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/ShoulderMarker",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.025,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 1.0, 1.0),
                                emissive_color=(0.5, 0.5, 0.5),
                            ),
                        ),
                    },
                )
            )

            # Body exclusion zone (orange) - shows the "cut" plane towards robot body
            self.body_exclusion_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/BodyExclusion",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.018,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.5, 0.0),  # Orange
                                emissive_color=(0.8, 0.3, 0.0),
                            ),
                        ),
                    },
                )
            )

            self._markers_initialized = True

        def _update_workspace_spheres(self):
            """Update workspace visualization in BODY FRAME

            Body frame convention:
            - +X = Robot FRONT (facing direction)
            - -Y = Robot RIGHT (right arm side)
            - +Z = Up

            Draw hemisphere centered on right shoulder, in front-right quadrant.
            All points defined in body frame, then transformed to world frame.
            """
            import math

            root_pos = self.robot.data.root_pos_w[0]  # [3]
            root_quat = self.robot.data.root_quat_w[0]  # [4]

            # Shoulder in body frame
            shoulder_body = self.shoulder_offset  # [3] - already in body frame

            # Transform shoulder to world for marker
            shoulder_world = root_pos + quat_apply(root_quat.unsqueeze(0), shoulder_body.unsqueeze(0)).squeeze(0)

            # Shoulder marker (white)
            identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
            self.shoulder_marker_vis.visualize(
                translations=shoulder_world.unsqueeze(0),
                orientations=identity_quat
            )

            n = 16  # points per arc

            def body_to_world(local_pts):
                """Transform body frame points to world frame"""
                # local_pts: [N, 3] in body frame relative to shoulder
                # Add shoulder offset, then rotate by root_quat, then add root_pos
                pts_from_root = local_pts + shoulder_body.unsqueeze(0)  # [N, 3]
                pts_world = root_pos.unsqueeze(0) + quat_apply(
                    root_quat.unsqueeze(0).expand(len(local_pts), -1),
                    pts_from_root
                )
                return pts_world

            # ===== OUTER workspace (blue) - 40cm radius =====
            outer_local = []
            R = 0.40

            # Arc 1: Horizontal half-circle in XY plane (front hemisphere, +X direction)
            # theta: 0 to pi, so cos goes from 1 to -1, sin goes 0 to 0 via 1
            # We want +X (front), so x = R*cos(theta) where theta from -pi/2 to pi/2
            for i in range(n + 1):
                theta = -math.pi/2 + math.pi * i / n  # -90° to +90°
                x = R * math.cos(theta)  # positive = front
                y = R * math.sin(theta)  # varies left to right
                z = 0.0
                outer_local.append([x, y, z])

            # Arc 2: Vertical half-circle in XZ plane (front view)
            for i in range(n + 1):
                theta = -math.pi/2 + math.pi * i / n
                x = R * math.cos(theta)  # positive = front
                y = 0.0
                z = R * math.sin(theta)  # up/down
                outer_local.append([x, y, z])

            # Arc 3: Vertical half-circle in YZ plane (side view, right side)
            # We want -Y (right side), so y = -R*cos(theta)
            for i in range(n + 1):
                theta = -math.pi/2 + math.pi * i / n
                x = 0.0
                y = -R * math.cos(theta)  # negative = right
                z = R * math.sin(theta)
                outer_local.append([x, y, z])

            outer_local_t = torch.tensor(outer_local, device=self.device, dtype=torch.float32)
            outer_world = body_to_world(outer_local_t)
            outer_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(outer_local), -1)
            self.outer_markers.visualize(translations=outer_world, orientations=outer_quat)

            # ===== INNER exclusion (red) - 18cm radius =====
            inner_local = []
            R = 0.18

            # Same pattern, smaller radius
            for i in range(n + 1):
                theta = -math.pi/2 + math.pi * i / n
                x = R * math.cos(theta)
                y = R * math.sin(theta)
                z = 0.0
                inner_local.append([x, y, z])

            for i in range(n + 1):
                theta = -math.pi/2 + math.pi * i / n
                x = R * math.cos(theta)
                y = 0.0
                z = R * math.sin(theta)
                inner_local.append([x, y, z])

            for i in range(n + 1):
                theta = -math.pi/2 + math.pi * i / n
                x = 0.0
                y = -R * math.cos(theta)
                z = R * math.sin(theta)
                inner_local.append([x, y, z])

            inner_local_t = torch.tensor(inner_local, device=self.device, dtype=torch.float32)
            inner_world = body_to_world(inner_local_t)
            inner_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(inner_local), -1)
            self.inner_markers.visualize(translations=inner_world, orientations=inner_quat)

            # ===== BODY EXCLUSION ZONE (orange) - Y > -0.10 cut plane =====
            # This shows the region that's too close to robot body (left/medial side)
            # Draw a vertical plane at Y = -0.10 (10cm to the right of shoulder)
            body_cut_local = []
            y_cut = -0.10  # 10cm sağda - bunun solundaki (Y > -0.10) hedefler yasak

            # Draw grid on the cut plane (X-Z plane at Y = y_cut)
            for i in range(n + 1):
                # Vertical line from -0.3 to +0.3 in Z
                x = 0.0
                z = -0.25 + 0.5 * i / n
                body_cut_local.append([x, y_cut, z])

            for i in range(n + 1):
                # Horizontal line from -0.2 to +0.4 in X
                x = -0.15 + 0.5 * i / n
                z = 0.0
                body_cut_local.append([x, y_cut, z])

            # Add some points showing the "forbidden" area (Y > y_cut)
            for i in range(5):
                for j in range(5):
                    x = -0.1 + 0.3 * i / 4
                    y = y_cut + 0.05 + 0.10 * j / 4  # Y > y_cut (towards robot)
                    z = -0.15 + 0.3 * j / 4
                    body_cut_local.append([x, y, z])

            body_cut_local_t = torch.tensor(body_cut_local, device=self.device, dtype=torch.float32)
            body_cut_world = body_to_world(body_cut_local_t)
            body_cut_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(len(body_cut_local), -1)
            self.body_exclusion_markers.visualize(translations=body_cut_world, orientations=body_cut_quat)

        def _compute_ee_pos(self) -> torch.Tensor:
            """Get end-effector world position"""
            return self.robot.data.body_pos_w[:, self.palm_idx]

        def _sample_commands(self, env_ids):
            """Sample velocity and arm target commands"""
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]

            # Reset already_reached flag for new targets
            self.already_reached[env_ids] = False

            # Velocity commands
            self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*lv["vx"])
            self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*lv["vy"])
            self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*lv["vyaw"])

            # Arm target - sample in front-right hemisphere
            azimuth = torch.empty(n, device=self.device).uniform_(-0.5, 1.0)
            radius = torch.empty(n, device=self.device).uniform_(*lv["arm_radius"])
            height = torch.empty(n, device=self.device).uniform_(*lv["arm_height"])

            x = radius * torch.cos(azimuth)
            y = -radius * torch.sin(azimuth)  # -Y is right side
            z = height

            # Body exclusion: Y must be < -0.10 (at least 10cm to the RIGHT of shoulder)
            # Targets with Y > -0.10 are too close to robot body
            y_cut = -0.10
            too_close_to_body = y > y_cut

            # Clamp Y to be at least y_cut (push away from body)
            y = torch.where(too_close_to_body, torch.full_like(y, y_cut), y)

            self.target_pos_body[env_ids, 0] = x + self.shoulder_offset[0]
            self.target_pos_body[env_ids, 1] = y + self.shoulder_offset[1]
            self.target_pos_body[env_ids, 2] = z + self.shoulder_offset[2]

        def get_loco_obs(self) -> torch.Tensor:
            """Locomotion observations (57 dim) - Stage 3 format"""
            robot = self.robot
            quat = robot.data.root_quat_w

            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(quat, gravity_vec)

            joint_pos = robot.data.joint_pos[:, self.leg_idx]
            joint_vel = robot.data.joint_vel[:, self.leg_idx]

            gait_phase = torch.stack([
                torch.sin(2 * np.pi * self.phase),
                torch.cos(2 * np.pi * self.phase)
            ], dim=-1)

            torso_euler = quat_to_euler_xyz(quat)

            obs = torch.cat([
                lin_vel_b,                       # 3
                ang_vel_b,                       # 3
                proj_gravity,                    # 3
                joint_pos,                       # 12
                joint_vel,                       # 12
                self.height_cmd.unsqueeze(-1),   # 1
                self.vel_cmd,                    # 3
                gait_phase,                      # 2
                self.prev_leg_actions,           # 12
                self.torso_cmd,                  # 3
                torso_euler,                     # 3
            ], dim=-1)  # Total: 57

            return obs.clamp(-10, 10).nan_to_num()

        def get_arm_obs(self) -> torch.Tensor:
            """Arm observations (28 dim) - body frame

            5 joint_pos + 5 joint_vel + 3 target + 3 ee + 3 pos_err + 1 dist + 2 lin_vel + 1 ang_vel + 5 prev_actions = 28
            """
            robot = self.robot
            root_pos = robot.data.root_pos_w
            root_quat = robot.data.root_quat_w

            # Joint states
            joint_pos = robot.data.joint_pos[:, self.arm_idx]
            joint_vel = robot.data.joint_vel[:, self.arm_idx]

            # EE in body frame
            ee_world = self._compute_ee_pos()
            ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

            # Target (already in body frame)
            target_body = self.target_pos_body

            # Position error
            pos_err = target_body - ee_body
            pos_dist = pos_err.norm(dim=-1, keepdim=True)

            # Body velocity (for adaptation)
            lin_vel_b = quat_apply_inverse(root_quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w)

            obs = torch.cat([
                joint_pos,                # 5
                joint_vel * 0.1,          # 5
                target_body,              # 3
                ee_body,                  # 3
                pos_err,                  # 3
                pos_dist / 0.5,           # 1
                lin_vel_b[:, :2],         # 2 (x, y velocity)
                ang_vel_b[:, 2:3],        # 1 (yaw rate)
                self.prev_arm_actions,    # 5
            ], dim=-1)  # Total: 26

            return obs.clamp(-10, 10).nan_to_num()

        def _pre_physics_step(self, actions):
            """Apply combined actions"""
            self.actions = actions.clone()

            leg_actions = actions[:, :12]
            arm_actions = actions[:, 12:17]

            target_pos = self.robot.data.default_joint_pos.clone()
            target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
            target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.3

            self.robot.set_joint_position_target(target_pos)

            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            # Check reaching - FIX: Only count NEW reaches
            ee_pos = self._compute_ee_pos()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            target_world = root_pos + quat_apply(root_quat, self.target_pos_body)

            dist = torch.norm(ee_pos - target_world, dim=-1)
            reached = dist < REACH_THRESHOLD

            # Only count if newly reached (wasn't reached before)
            new_reaches = reached & ~self.already_reached

            if new_reaches.any():
                reached_ids = torch.where(new_reaches)[0]
                self.reach_count[reached_ids] += 1
                self.total_reaches += len(reached_ids)
                self.stage_reaches += len(reached_ids)

                # Mark as reached and sample new target
                self.already_reached[reached_ids] = True
                self._sample_commands(reached_ids)

            # Update markers
            self._init_markers()
            default_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(self.num_envs, -1)
            self.target_markers.visualize(translations=target_world, orientations=default_quat)
            self.ee_markers.visualize(translations=ee_pos, orientations=default_quat)

            # Update workspace visualization
            self._update_workspace_spheres()

            # Store previous actions
            self._prev_leg_actions = self.prev_leg_actions.clone()
            self._prev_arm_actions = self.prev_arm_actions.clone()
            self.prev_leg_actions = leg_actions.clone()
            self.prev_arm_actions = arm_actions.clone()

            self.stage_steps += 1

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            return {"policy": self.get_loco_obs()}

        def _get_rewards(self) -> torch.Tensor:
            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w

            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

            # === LOCOMOTION REWARDS ===
            r_vx = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
            r_vy = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.vel_cmd[:, 1]) ** 2)
            r_vyaw = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.vel_cmd[:, 2]) ** 2)

            r_height = torch.exp(-10.0 * (pos[:, 2] - self.height_cmd) ** 2)

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(quat, gravity_vec)
            r_orientation = torch.exp(-3.0 * (proj_gravity[:, 0]**2 + proj_gravity[:, 1]**2))

            # Gait quality
            joint_pos = robot.data.joint_pos[:, self.leg_idx]
            left_knee, right_knee = joint_pos[:, 6], joint_pos[:, 7]
            left_swing = (self.phase < 0.5).float()
            right_swing = (self.phase >= 0.5).float()
            knee_err = (
                (left_knee - (left_swing * 0.6 + (1 - left_swing) * 0.3)) ** 2 +
                (right_knee - (right_swing * 0.6 + (1 - right_swing) * 0.3)) ** 2
            )
            r_gait = torch.exp(-3.0 * knee_err)

            # === ARM REWARDS ===
            ee_pos = self._compute_ee_pos()
            target_world = pos + quat_apply(quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            r_distance = torch.exp(-5.0 * dist)
            r_reaching = (dist < REACH_THRESHOLD).float()

            # Smoothness
            arm_diff = self.prev_arm_actions - self._prev_arm_actions
            r_arm_smooth = torch.exp(-0.5 * arm_diff.pow(2).sum(-1))

            # === BALANCE ===
            # Simple CoM proxy - penalize body tilt when arm is moving
            arm_activity = self.prev_arm_actions.abs().mean(-1)
            r_balance = torch.exp(-5.0 * (proj_gravity[:, 0]**2 + proj_gravity[:, 1]**2) * (1 + arm_activity))

            # === PENALTIES ===
            leg_diff = self.prev_leg_actions - self._prev_leg_actions
            p_leg_rate = leg_diff.pow(2).sum(-1)
            arm_diff = self.prev_arm_actions - self._prev_arm_actions
            p_arm_rate = arm_diff.pow(2).sum(-1)

            # Energy penalty - sadece aktif jointler
            leg_vel = robot.data.joint_vel[:, self.leg_idx]
            arm_vel = robot.data.joint_vel[:, self.arm_idx]
            p_energy = (leg_vel.abs() * self.prev_leg_actions.abs()).sum(-1) + \
                       (arm_vel.abs() * self.prev_arm_actions.abs()).sum(-1)

            # === TOTAL ===
            reward = (
                REWARD_WEIGHTS["loco_vx"] * r_vx +
                REWARD_WEIGHTS["loco_vy"] * r_vy +
                REWARD_WEIGHTS["loco_vyaw"] * r_vyaw +
                REWARD_WEIGHTS["loco_height"] * r_height +
                REWARD_WEIGHTS["loco_orientation"] * r_orientation +
                REWARD_WEIGHTS["loco_gait"] * r_gait +
                REWARD_WEIGHTS["arm_distance"] * r_distance +
                REWARD_WEIGHTS["arm_reaching"] * r_reaching +
                REWARD_WEIGHTS["arm_smooth"] * r_arm_smooth +
                REWARD_WEIGHTS["balance"] * r_balance +
                REWARD_WEIGHTS["loco_action_rate"] * p_leg_rate +
                REWARD_WEIGHTS["arm_action_rate"] * p_arm_rate +
                REWARD_WEIGHTS["energy"] * p_energy +
                REWARD_WEIGHTS["alive"]
            )

            self.extras = {
                "R/loco_vx": r_vx.mean().item(),
                "R/loco_height": r_height.mean().item(),
                "R/arm_distance": r_distance.mean().item(),
                "R/arm_reaching": r_reaching.mean().item(),
                "R/balance": r_balance.mean().item(),
                "M/height": pos[:, 2].mean().item(),
                "M/vx": lin_vel_b[:, 0].mean().item(),
                "M/ee_dist": dist.mean().item(),
                "M/reaches": self.total_reaches,
                "curriculum_level": self.curr_level,
            }

            return reward.clamp(-10, 50)

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

            n = len(env_ids)
            default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(n, -1).clone()
            default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(n, -1)

            self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

            default_joint_pos = self.robot.data.default_joint_pos[env_ids]
            self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

            self._sample_commands(env_ids)

            self.phase[env_ids] = torch.rand(n, device=self.device)
            self.prev_leg_actions[env_ids] = 0
            self.prev_arm_actions[env_ids] = 0
            self.reach_count[env_ids] = 0

        def update_curriculum(self, mean_reward):
            """Update curriculum based on reaches, success rate, AND minimum steps"""
            lv = CURRICULUM[self.curr_level]

            if lv["min_reaches"] is None:
                return  # Final level

            # Check ALL criteria: min_reaches, success_rate, AND min_steps
            min_steps = lv.get("min_steps", 0)

            if self.stage_steps >= min_steps and self.stage_reaches >= lv["min_reaches"]:
                success_rate = self.stage_reaches / max(self.stage_steps, 1)
                if success_rate >= lv["success_rate"]:
                    if self.curr_level < len(CURRICULUM) - 1:
                        self.curr_level += 1
                        print(f"\n{'='*60}")
                        print(f"🎯 LEVEL UP! Now at Level {self.curr_level}")
                        new_lv = CURRICULUM[self.curr_level]
                        print(f"   vx={new_lv['vx']}, arm_radius={new_lv['arm_radius']}")
                        print(f"   Reaches: {self.stage_reaches}, SR: {success_rate:.2%}, Steps: {self.stage_steps}")
                        print(f"{'='*60}\n")
                        self.stage_reaches = 0
                        self.stage_steps = 0

    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    return Stage6Env(cfg)


# ============================================================================
# WEIGHT TRANSFER
# ============================================================================

def transfer_stage3_weights(net, stage3_path, device):
    """Transfer loco weights from Stage 3"""
    if stage3_path is None:
        print("[Transfer] No Stage 3 checkpoint, starting fresh")
        return

    print(f"\n[Transfer] Loading Stage 3: {stage3_path}")

    ckpt = torch.load(stage3_path, map_location=device, weights_only=False)
    s3_state = ckpt["actor_critic"]

    # Map Stage 3 actor weights to loco_actor
    transferred = 0
    for key in s3_state:
        if key.startswith("actor."):
            new_key = "loco_actor." + key
            if new_key in net.state_dict():
                if s3_state[key].shape == net.state_dict()[new_key].shape:
                    net.state_dict()[new_key].copy_(s3_state[key])
                    transferred += 1
        elif key == "log_std":
            if "loco_actor.log_std" in net.state_dict():
                net.state_dict()["loco_actor.log_std"].copy_(s3_state[key])
                transferred += 1

    print(f"[Transfer] Transferred {transferred} loco parameters")

    if "best_reward" in ckpt:
        print(f"[Transfer] Stage 3 best reward: {ckpt['best_reward']:.2f}")


# ============================================================================
# MAIN
# ============================================================================

def train():
    device = "cuda:0"

    print(f"\n[INFO] Creating environment with {args_cli.num_envs} envs...")
    env = create_env(args_cli.num_envs, device)

    print(f"[INFO] Creating unified network...")
    net = UnifiedActorCritic(loco_obs=57, arm_obs=28, loco_act=12, arm_act=5).to(device)

    # Transfer Stage 3 weights
    transfer_stage3_weights(net, args_cli.stage3_checkpoint, device)

    # Resume if checkpoint provided
    start_iter = 0
    if args_cli.checkpoint:
        print(f"\n[INFO] Resuming from: {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0)
        env.curr_level = ckpt.get("curriculum_level", 0)

    ppo = PPO(net, device)

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"\n[INFO] Logging to: {log_dir}")

    best_reward = float('-inf')

    # Initial exploration
    net.loco_actor.log_std.data.fill_(np.log(0.5))
    net.arm_actor.log_std.data.fill_(np.log(0.6))

    obs, _ = env.reset()

    print("\n" + "=" * 80)
    print("STARTING STAGE 6 UNIFIED TRAINING")
    print("=" * 80 + "\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        # Collect rollouts
        loco_obs_buf = []
        arm_obs_buf = []
        loco_act_buf = []
        arm_act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        logp_buf = []

        rollout_steps = 24

        for _ in range(rollout_steps):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            with torch.no_grad():
                loco_mean, arm_mean, value = net(loco_obs, arm_obs)

                loco_std = net.loco_actor.log_std.clamp(-2, 1).exp()
                arm_std = net.arm_actor.log_std.clamp(-2, 1).exp()

                loco_dist = torch.distributions.Normal(loco_mean, loco_std)
                arm_dist = torch.distributions.Normal(arm_mean, arm_std)

                loco_action = loco_dist.sample()
                arm_action = arm_dist.sample()

                log_prob = loco_dist.log_prob(loco_action).sum(-1) + arm_dist.log_prob(arm_action).sum(-1)

            loco_obs_buf.append(loco_obs)
            arm_obs_buf.append(arm_obs)
            loco_act_buf.append(loco_action)
            arm_act_buf.append(arm_action)
            val_buf.append(value)
            logp_buf.append(log_prob)

            actions = torch.cat([loco_action, arm_action], dim=-1)
            obs_dict, reward, terminated, truncated, _ = env.step(actions)

            rew_buf.append(reward)
            done_buf.append((terminated | truncated).float())

        # Stack
        loco_obs_buf = torch.stack(loco_obs_buf)
        arm_obs_buf = torch.stack(arm_obs_buf)
        loco_act_buf = torch.stack(loco_act_buf)
        arm_act_buf = torch.stack(arm_act_buf)
        rew_buf = torch.stack(rew_buf)
        done_buf = torch.stack(done_buf)
        val_buf = torch.stack(val_buf)
        logp_buf = torch.stack(logp_buf)

        # GAE
        with torch.no_grad():
            final_loco_obs = env.get_loco_obs()
            final_arm_obs = env.get_arm_obs()
            _, _, next_value = net(final_loco_obs, final_arm_obs)

        advantages, returns = ppo.gae(rew_buf, val_buf, done_buf, next_value)

        # PPO update
        update_info = ppo.update(
            loco_obs_buf.view(-1, 57),
            arm_obs_buf.view(-1, 28),
            loco_act_buf.view(-1, 12),
            arm_act_buf.view(-1, 5),
            logp_buf.view(-1),
            returns.view(-1),
            advantages.view(-1),
            val_buf.view(-1),
        )

        # Anneal exploration
        progress = iteration / args_cli.max_iterations
        loco_std = 0.5 + (0.15 - 0.5) * progress
        arm_std = 0.6 + (0.2 - 0.6) * progress
        net.loco_actor.log_std.data.fill_(np.log(loco_std))
        net.arm_actor.log_std.data.fill_(np.log(arm_std))

        mean_reward = rew_buf.mean().item()
        env.update_curriculum(mean_reward)

        # Save best
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "total_reaches": env.total_reaches,
            }, f"{log_dir}/model_best.pt")

        # Logging
        writer.add_scalar("Train/reward", mean_reward, iteration)
        writer.add_scalar("Train/best_reward", best_reward, iteration)
        writer.add_scalar("Train/loco_std", loco_std, iteration)
        writer.add_scalar("Train/arm_std", arm_std, iteration)
        writer.add_scalar("Curriculum/level", env.curr_level, iteration)
        writer.add_scalar("Curriculum/total_reaches", env.total_reaches, iteration)

        for key, val in env.extras.items():
            writer.add_scalar(f"Env/{key}", val, iteration)

        if iteration % 10 == 0:
            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:6.2f} | "
                f"Best={best_reward:6.2f} | "
                f"Lv={env.curr_level} | "
                f"Reaches={env.total_reaches} | "
                f"EE_dist={env.extras.get('M/ee_dist', 0):.3f}"
            )

        if (iteration + 1) % 500 == 0:
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "total_reaches": env.total_reaches,
            }, f"{log_dir}/model_{iteration + 1}.pt")

        writer.flush()

    # Final save
    torch.save({
        "model": net.state_dict(),
        "iteration": args_cli.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
        "total_reaches": env.total_reaches,
    }, f"{log_dir}/model_final.pt")

    writer.close()
    env.close()

    print("\n" + "=" * 80)
    print("STAGE 6 TRAINING COMPLETE!")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Total Reaches: {env.total_reaches}")
    print(f"  Log Dir: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
    simulation_app.close()