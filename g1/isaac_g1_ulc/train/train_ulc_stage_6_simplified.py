"""
ULC G1 Stage 6: SIMPLIFIED 10-Level Curriculum for 3-Week Deadline
==================================================================
DUAL ACTOR-CRITIC ARCHITECTURE - GRIPPER DISABLED

KEY SIMPLIFICATIONS:
1. 40 levels → 10 levels (faster progression)
2. NO GRIPPER control (fingers stay open)
3. Phase 1 (0-4): Standing + Reaching
4. Phase 2 (5-9): Walking + Reaching + Orientation
5. Stage 3 AND Stage 5 checkpoint support with obs filler

ARCHITECTURE:
┌─────────────────────────────────────────────────────────┐
│     LOCO BRANCH         │        ARM BRANCH             │
├─────────────────────────┼───────────────────────────────┤
│  LocoActor (57→12)      │  ArmActor (52→5)  ← NO FINGER │
│  LocoCritic (57→1)      │  ArmCritic (52→1)             │
└─────────────────────────┴───────────────────────────────┘

CURRICULUM:
- Phase 1 (Level 0-4): Standing + Reaching ONLY
- Phase 2 (Level 5-9): Walking + Reaching + Orientation
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

FINGER_JOINT_NAMES = [
    "right_zero_joint", "right_one_joint", "right_two_joint",
    "right_three_joint", "right_four_joint", "right_five_joint",
    "right_six_joint",
]

PALM_FORWARD_OFFSET = 0.08
SHOULDER_OFFSET = torch.tensor([0.0, -0.174, 0.259])
HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5

# ============================================================================
# SIMPLIFIED REWARD WEIGHTS - NO GRIPPER
# ============================================================================

# LOCOMOTION REWARDS
LOCO_REWARD_WEIGHTS = {
    "vx": 3.0,
    "vy": 1.5,
    "vyaw": 1.5,
    "height": 3.0,
    "orientation": 4.0,
    "gait": 2.0,
    "com_stability": 2.5,
    "leg_posture": 2.5,
    "standing_still": 2.0,
    "foot_stability": 1.5,
    # Penalties
    "action_rate": -0.03,
    "jerk": -0.02,
    "energy": -0.0005,
    "alive": 0.5,
}

# ARM REWARDS - NO GRIPPER!
ARM_REWARD_WEIGHTS = {
    "distance": 4.0,
    "reaching": 12.0,        # Smooth sigmoid
    "final_push": 6.0,
    "smooth": 3.0,
    "palm_orient": 3.0,      # Only when enabled in Phase 2
    # NO GRIPPER REWARD!
    # Penalties
    "action_rate": -0.05,
    "jerk": -0.03,
    "workspace_violation": -2.0,
    "alive": 0.3,
}

# ============================================================================
# SIMPLIFIED 10-LEVEL CURRICULUM
# ============================================================================

def create_simplified_curriculum():
    """
    10-level curriculum for 3-week deadline

    Phase 1 (Level 0-4): Standing + Reaching
    - Robot stands still (vx=0)
    - Learns arm reaching
    - NO orientation, NO gripper

    Phase 2 (Level 5-9): Walking + Reaching + Orientation
    - Robot walks while reaching
    - Orientation added gradually
    - Still NO gripper
    """
    curriculum = []

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: STANDING + REACHING (Level 0-4)
    # Robot durarak kolunu hedeflere uzatmayı öğreniyor
    # ═══════════════════════════════════════════════════════════════════════

    # Level 0: Çok yakın hedefler, kolay başlangıç
    curriculum.append({
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.05, 0.10),
        "arm_height": (-0.05, 0.10),
        "pos_threshold": 0.12,
        "orient_threshold": None,
        "success_rate": 0.30,
        "min_reaches": 2000,
        "min_steps": 1000,
        "use_orientation": False,
        "use_gripper": False,
        "sampling_mode": "relative",
        "workspace_radius": (0.18, 0.28),
    })

    # Level 1: Biraz daha uzak hedefler
    curriculum.append({
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.08, 0.15),
        "arm_height": (-0.08, 0.14),
        "pos_threshold": 0.10,
        "orient_threshold": None,
        "success_rate": 0.32,
        "min_reaches": 3000,
        "min_steps": 1500,
        "use_orientation": False,
        "use_gripper": False,
        "sampling_mode": "mixed",
        "workspace_radius": (0.18, 0.30),
    })

    # Level 2: Orta mesafe hedefler
    curriculum.append({
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.10, 0.20),
        "arm_height": (-0.10, 0.18),
        "pos_threshold": 0.08,
        "orient_threshold": None,
        "success_rate": 0.34,
        "min_reaches": 4000,
        "min_steps": 2000,
        "use_orientation": False,
        "use_gripper": False,
        "sampling_mode": "mixed",
        "workspace_radius": (0.18, 0.32),
    })

    # Level 3: Geniş workspace
    curriculum.append({
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.15, 0.28),
        "arm_height": (-0.12, 0.22),
        "pos_threshold": 0.07,
        "orient_threshold": None,
        "success_rate": 0.36,
        "min_reaches": 5000,
        "min_steps": 2500,
        "use_orientation": False,
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.35),
    })

    # Level 4: Tam workspace, dar threshold
    curriculum.append({
        "vx": (0.0, 0.0), "vy": (0.0, 0.0), "vyaw": (0.0, 0.0),
        "arm_radius": (0.18, 0.35),
        "arm_height": (-0.15, 0.25),
        "pos_threshold": 0.06,
        "orient_threshold": None,
        "success_rate": 0.38,
        "min_reaches": 6000,
        "min_steps": 3000,
        "use_orientation": False,
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.38),
    })

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: WALKING + REACHING (Level 5-6) → + ORIENTATION (Level 7-9)
    # Önce yürürken reaching öğren, SONRA orientation ekle
    # ═══════════════════════════════════════════════════════════════════════

    # Level 5: Yavaş yürüyüş + reach, ORIENTATION KAPALI
    curriculum.append({
        "vx": (0.0, 0.2), "vy": (-0.05, 0.05), "vyaw": (-0.10, 0.10),
        "arm_radius": (0.18, 0.35),
        "arm_height": (-0.15, 0.25),
        "pos_threshold": 0.06,
        "orient_threshold": None,  # KAPALI
        "success_rate": 0.28,
        "min_reaches": 4000,  # Düşürüldü
        "min_steps": 2000,
        "use_orientation": False,  # ❌ KAPALI
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.38),
    })

    # Level 6: Orta hız + reach, ORIENTATION HALA KAPALI
    curriculum.append({
        "vx": (0.0, 0.3), "vy": (-0.07, 0.07), "vyaw": (-0.13, 0.13),
        "arm_radius": (0.18, 0.37),
        "arm_height": (-0.15, 0.25),
        "pos_threshold": 0.05,
        "orient_threshold": None,  # KAPALI
        "success_rate": 0.26,
        "min_reaches": 5000,
        "min_steps": 2500,
        "use_orientation": False,  # ❌ KAPALI
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.40),
    })

    # Level 7: Normal hız + reach, ORIENTATION BAŞLIYOR (çok gevşek: 1.5 rad = ~86°)
    curriculum.append({
        "vx": (0.0, 0.4), "vy": (-0.09, 0.09), "vyaw": (-0.16, 0.16),
        "arm_radius": (0.18, 0.38),
        "arm_height": (-0.15, 0.25),
        "pos_threshold": 0.05,
        "orient_threshold": 1.5,  # ~86 derece - neredeyse her şey geçer
        "success_rate": 0.24,
        "min_reaches": 5000,
        "min_steps": 2500,
        "use_orientation": True,  # ✅ AÇIK
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.40),
    })

    # Level 8: Hızlı yürüyüş + reach, orientation biraz sıkılaşıyor (1.2 rad = ~69°)
    curriculum.append({
        "vx": (0.0, 0.5), "vy": (-0.11, 0.11), "vyaw": (-0.19, 0.19),
        "arm_radius": (0.18, 0.40),
        "arm_height": (-0.18, 0.28),
        "pos_threshold": 0.04,
        "orient_threshold": 1.2,  # ~69 derece
        "success_rate": 0.22,
        "min_reaches": 6000,
        "min_steps": 3000,
        "use_orientation": True,  # ✅ AÇIK
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.40),
    })

    # Level 9: FINAL - En hızlı + orientation orta-sıkı (0.9 rad = ~52°)
    curriculum.append({
        "vx": (0.0, 0.6), "vy": (-0.13, 0.13), "vyaw": (-0.22, 0.22),
        "arm_radius": (0.18, 0.40),
        "arm_height": (-0.20, 0.30),
        "pos_threshold": 0.04,
        "orient_threshold": 0.9,  # ~52 derece - makul sıkılık
        "success_rate": None,  # Final level - no graduation
        "min_reaches": None,
        "min_steps": None,
        "use_orientation": True,  # ✅ AÇIK
        "use_gripper": False,
        "sampling_mode": "absolute",
        "workspace_radius": (0.18, 0.40),
    })

    return curriculum


CURRICULUM = create_simplified_curriculum()


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 6 SIMPLIFIED: 10-Level, No Gripper")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=15000)  # Reduced from 25000
    parser.add_argument("--stage3_checkpoint", type=str, default=None,
                        help="Stage 3 loco-only checkpoint (57 obs → 12 act)")
    parser.add_argument("--stage5_checkpoint", type=str, default=None,
                        help="Stage 5 arm reaching checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from V3 simplified checkpoint")
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage6_simplified")
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


def get_palm_forward(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fwd_x = 1 - 2*(y*y + z*z)
    fwd_y = 2*(x*y + w*z)
    fwd_z = 2*(x*z - w*y)
    return torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)


def compute_orientation_error(palm_quat: torch.Tensor) -> torch.Tensor:
    forward = get_palm_forward(palm_quat)
    target_dir = torch.zeros_like(forward)
    target_dir[:, 2] = -1.0
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.acos(dot)


print("=" * 80)
print("ULC G1 STAGE 6 SIMPLIFIED: 10-LEVEL CURRICULUM")
print("=" * 80)
print("SIMPLIFICATIONS:")
print("  1. Curriculum: 40 → 10 levels (faster training)")
print("  2. NO GRIPPER control (fingers stay open)")
print("  3. Phase 1 (0-4): Standing + Reaching")
print("  4. Phase 2 (5-9): Walking + Reaching + Orientation")
print("  5. Stage 3 AND Stage 5 checkpoint support")
print("=" * 80)
print(f"Stage 3 checkpoint: {args_cli.stage3_checkpoint}")
print(f"Stage 5 checkpoint: {args_cli.stage5_checkpoint}")
print(f"\nArchitecture:")
print("  LocoActor (57→12) + LocoCritic (57→1)")
print("  ArmActor (52→5)   + ArmCritic (52→1)  ← NO FINGER ACTIONS")
print(f"\nCurriculum: {len(CURRICULUM)} levels")
print("  Phase 1 (0-4): Standing + Reaching")
print("  Phase 2 (5-9): Walking + Reaching + Orientation")
print("=" * 80)


# ============================================================================
# DUAL ACTOR-CRITIC NETWORKS
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion policy: 57 obs → 12 leg actions"""
    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        return self.net(x)


class LocoCritic(nn.Module):
    """Locomotion value function: 57 obs → 1 value"""
    def __init__(self, num_obs=57, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ArmActor(nn.Module):
    """Arm policy: 52 obs → 5 actions (NO FINGERS!)"""
    def __init__(self, num_obs=52, num_act=5, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(num_act))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        return self.net(x)


class ArmCritic(nn.Module):
    """Arm value function: 52 obs → 1 value"""
    def __init__(self, num_obs=52, hidden=[256, 256, 128]):
        super().__init__()
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DualActorCritic(nn.Module):
    """
    Dual Actor-Critic - SIMPLIFIED (NO FINGER ACTIONS)

    Architecture:
    - LocoActor (57→12) + LocoCritic (57→1)
    - ArmActor (52→5)   + ArmCritic (52→1)
    """
    def __init__(self, loco_obs=57, arm_obs=52, loco_act=12, arm_act=5):
        super().__init__()
        self.loco_actor = LocoActor(loco_obs, loco_act)
        self.loco_critic = LocoCritic(loco_obs)
        self.arm_actor = ArmActor(arm_obs, arm_act)
        self.arm_critic = ArmCritic(arm_obs)

    def forward(self, loco_obs, arm_obs):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)
        loco_value = self.loco_critic(loco_obs)
        arm_value = self.arm_critic(arm_obs)
        return loco_mean, arm_mean, loco_value, arm_value

    def get_actions(self, loco_obs, arm_obs, deterministic=False):
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)

        if deterministic:
            return loco_mean, arm_mean

        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()

        loco_dist = torch.distributions.Normal(loco_mean, loco_std)
        arm_dist = torch.distributions.Normal(arm_mean, arm_std)

        loco_action = loco_dist.sample()
        arm_action = arm_dist.sample()

        loco_logp = loco_dist.log_prob(loco_action).sum(-1)
        arm_logp = arm_dist.log_prob(arm_action).sum(-1)

        return loco_action, arm_action, loco_logp, arm_logp

    def evaluate_loco(self, loco_obs, loco_actions):
        loco_mean = self.loco_actor(loco_obs)
        loco_value = self.loco_critic(loco_obs)
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        loco_dist = torch.distributions.Normal(loco_mean, loco_std)
        loco_logp = loco_dist.log_prob(loco_actions).sum(-1)
        loco_entropy = loco_dist.entropy().sum(-1)
        return loco_value, loco_logp, loco_entropy

    def evaluate_arm(self, arm_obs, arm_actions):
        arm_mean = self.arm_actor(arm_obs)
        arm_value = self.arm_critic(arm_obs)
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        arm_dist = torch.distributions.Normal(arm_mean, arm_std)
        arm_logp = arm_dist.log_prob(arm_actions).sum(-1)
        arm_entropy = arm_dist.entropy().sum(-1)
        return arm_value, arm_logp, arm_entropy


# ============================================================================
# DUAL PPO TRAINER
# ============================================================================

class DualPPO:
    def __init__(self, net, device, lr=3e-4):
        self.net = net
        self.device = device

        self.loco_opt = torch.optim.AdamW(
            list(net.loco_actor.parameters()) + list(net.loco_critic.parameters()),
            lr=lr, weight_decay=1e-5
        )
        self.arm_opt = torch.optim.AdamW(
            list(net.arm_actor.parameters()) + list(net.arm_critic.parameters()),
            lr=lr, weight_decay=1e-5
        )

        self.loco_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.loco_opt, args_cli.max_iterations, eta_min=1e-5
        )
        self.arm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.arm_opt, args_cli.max_iterations, eta_min=1e-5
        )

    def gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    def update_loco(self, loco_obs, loco_actions, old_log_probs, returns, advantages, old_values):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0
        num_updates = 0
        batch_size = loco_obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                values, log_probs, entropy = self.net.evaluate_loco(
                    loco_obs[mb_idx], loco_actions[mb_idx]
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
                self.loco_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.net.loco_actor.parameters()) + list(self.net.loco_critic.parameters()),
                    0.5
                )
                self.loco_opt.step()
                total_loss += loss.item()
                num_updates += 1

        self.loco_sched.step()
        return total_loss / num_updates

    def update_arm(self, arm_obs, arm_actions, old_log_probs, returns, advantages, old_values):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0
        num_updates = 0
        batch_size = arm_obs.shape[0]
        minibatch_size = 4096

        for _ in range(5):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                values, log_probs, entropy = self.net.evaluate_arm(
                    arm_obs[mb_idx], arm_actions[mb_idx]
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
                self.arm_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.net.arm_actor.parameters()) + list(self.net.arm_critic.parameters()),
                    0.5
                )
                self.arm_opt.step()
                total_loss += loss.item()
                num_updates += 1

        self.arm_sched.step()
        return total_loss / num_updates


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
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=[".*zero.*", ".*one.*", ".*two.*", ".*three.*", ".*four.*", ".*five.*", ".*six.*"],
                    stiffness=20.0,
                    damping=2.0,
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
        action_space = 17  # 12 leg + 5 arm (NO FINGERS!)
        observation_space = 57
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class Stage6SimplifiedEnv(DirectRLEnv):
        """Simplified environment - NO GRIPPER CONTROL"""
        cfg: EnvCfg

        def __init__(self, cfg, render_mode=None, **kwargs):
            super().__init__(cfg, render_mode, **kwargs)

            joint_names = self.robot.joint_names

            self.leg_idx = torch.tensor(
                [joint_names.index(n) for n in LEG_JOINT_NAMES if n in joint_names],
                device=self.device
            )
            self.arm_idx = torch.tensor(
                [joint_names.index(n) for n in ARM_JOINT_NAMES if n in joint_names],
                device=self.device
            )
            self.finger_idx = torch.tensor(
                [joint_names.index(n) for n in FINGER_JOINT_NAMES if n in joint_names],
                device=self.device
            )

            self.default_leg = torch.tensor(
                [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
                device=self.device
            )
            self.default_arm = self.robot.data.default_joint_pos[0, self.arm_idx].clone()

            joint_limits = self.robot.root_physx_view.get_dof_limits()
            self.finger_lower = torch.tensor(
                [joint_limits[0, i, 0].item() for i in self.finger_idx],
                device=self.device
            )
            self.finger_upper = torch.tensor(
                [joint_limits[0, i, 1].item() for i in self.finger_idx],
                device=self.device
            )

            body_names = self.robot.body_names
            self.palm_idx = None
            for i, name in enumerate(body_names):
                if "right_palm" in name.lower():
                    self.palm_idx = i
                    break
            if self.palm_idx is None:
                self.palm_idx = len(body_names) - 1

            print(f"[Simplified] Leg joints: {len(self.leg_idx)}")
            print(f"[Simplified] Arm joints: {len(self.arm_idx)}")
            print(f"[Simplified] Finger joints: {len(self.finger_idx)} (FIXED OPEN)")
            print(f"[Simplified] Palm body idx: {self.palm_idx}")

            # Commands
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

            self.phase = torch.zeros(self.num_envs, device=self.device)

            # Action history - NO FINGER ACTIONS
            self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
            self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
            self._prev_prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

            self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

            # Curriculum
            self.curr_level = 0
            self.reach_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.total_reaches = 0
            self.stage_reaches = 0
            self.stage_steps = 0
            self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            self.loco_reward = torch.zeros(self.num_envs, device=self.device)
            self.arm_reward = torch.zeros(self.num_envs, device=self.device)

            self._markers_initialized = False

        @property
        def robot(self):
            return self.scene["robot"]

        def _init_markers(self):
            if self._markers_initialized:
                return
            self.target_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/TargetMarkers",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                        ),
                    },
                )
            )
            self.ee_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/EEMarkers",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    },
                )
            )
            self._markers_initialized = True

        def _compute_palm_ee(self) -> tuple:
            palm_pos = self.robot.data.body_pos_w[:, self.palm_idx]
            palm_quat = self.robot.data.body_quat_w[:, self.palm_idx]
            palm_forward = get_palm_forward(palm_quat)
            ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_forward
            return ee_pos, palm_quat

        def _sample_commands(self, env_ids):
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]
            self.already_reached[env_ids] = False

            self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*lv["vx"])
            self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*lv["vy"])
            self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*lv["vyaw"])
            self.height_cmd[env_ids] = HEIGHT_DEFAULT

            sampling_mode = lv.get("sampling_mode", "absolute")
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]
            ee_world, _ = self._compute_palm_ee()
            ee_world = ee_world[env_ids]
            current_ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

            if sampling_mode == "relative":
                azimuth = torch.empty(n, device=self.device).uniform_(-0.8, 0.8)
                elevation = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
                radius = torch.empty(n, device=self.device).uniform_(*lv["arm_radius"])
                offset_x = radius * torch.cos(elevation) * torch.cos(azimuth)
                offset_y = radius * torch.cos(elevation) * torch.sin(azimuth)
                offset_z = radius * torch.sin(elevation)
                target_x = current_ee_body[:, 0] + offset_x
                target_y = current_ee_body[:, 1] + offset_y
                target_z = current_ee_body[:, 2] + offset_z
            elif sampling_mode == "mixed":
                use_absolute = torch.rand(n, device=self.device) > 0.5
                workspace_radius = lv.get("workspace_radius", (0.18, 0.35))
                rel_azimuth = torch.empty(n, device=self.device).uniform_(-0.8, 0.8)
                rel_elevation = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
                rel_radius = torch.empty(n, device=self.device).uniform_(*lv["arm_radius"])
                rel_x = current_ee_body[:, 0] + rel_radius * torch.cos(rel_elevation) * torch.cos(rel_azimuth)
                rel_y = current_ee_body[:, 1] + rel_radius * torch.cos(rel_elevation) * torch.sin(rel_azimuth)
                rel_z = current_ee_body[:, 2] + rel_radius * torch.sin(rel_elevation)
                abs_azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
                abs_elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
                abs_radius = torch.empty(n, device=self.device).uniform_(*workspace_radius)
                abs_x = abs_radius * torch.cos(abs_elevation) * torch.cos(abs_azimuth) + self.shoulder_offset[0]
                abs_y = -abs_radius * torch.cos(abs_elevation) * torch.sin(abs_azimuth) + self.shoulder_offset[1]
                abs_z = abs_radius * torch.sin(abs_elevation) + self.shoulder_offset[2]
                target_x = torch.where(use_absolute, abs_x, rel_x)
                target_y = torch.where(use_absolute, abs_y, rel_y)
                target_z = torch.where(use_absolute, abs_z, rel_z)
            else:
                workspace_radius = lv.get("workspace_radius", (0.18, 0.40))
                azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
                elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
                radius = torch.empty(n, device=self.device).uniform_(*workspace_radius)
                target_x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
                target_y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
                target_z = radius * torch.sin(elevation) + self.shoulder_offset[2]

            self.target_pos_body[env_ids, 0] = target_x.clamp(0.05, 0.55)
            self.target_pos_body[env_ids, 1] = target_y.clamp(-0.55, 0.10)
            self.target_pos_body[env_ids, 2] = target_z.clamp(-0.25, 0.55)

        def get_loco_obs(self) -> torch.Tensor:
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
                lin_vel_b, ang_vel_b, proj_gravity, joint_pos, joint_vel,
                self.height_cmd.unsqueeze(-1), self.vel_cmd, gait_phase,
                self.prev_leg_actions, self.torso_cmd, torso_euler,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num()

        def get_arm_obs(self) -> torch.Tensor:
            robot = self.robot
            root_pos = robot.data.root_pos_w
            root_quat = robot.data.root_quat_w
            lv = CURRICULUM[self.curr_level]

            arm_pos = robot.data.joint_pos[:, self.arm_idx]
            arm_vel = robot.data.joint_vel[:, self.arm_idx] * 0.1
            finger_pos = robot.data.joint_pos[:, self.finger_idx]

            ee_world, palm_quat = self._compute_palm_ee()
            ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)
            ee_vel_world = (ee_world - self.prev_ee_pos) / 0.02
            ee_vel_body = quat_apply_inverse(root_quat, ee_vel_world)

            finger_normalized = (finger_pos - self.finger_lower) / (self.finger_upper - self.finger_lower + 1e-6)
            gripper_closed_ratio = finger_normalized.mean(dim=-1, keepdim=True)
            finger_vel = robot.data.joint_vel[:, self.finger_idx]
            grip_force = (finger_vel.abs().mean(dim=-1, keepdim=True) * gripper_closed_ratio).clamp(0, 1)

            target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
            dist_to_target = torch.norm(ee_world - target_world, dim=-1, keepdim=True)
            contact_detected = (dist_to_target < 0.08).float()

            target_body = self.target_pos_body
            pos_error = target_body - ee_body
            pos_dist = pos_error.norm(dim=-1, keepdim=True) / 0.5
            orient_err = compute_orientation_error(palm_quat).unsqueeze(-1) / np.pi

            orient_threshold = lv.get("orient_threshold", 1.0) if lv["use_orientation"] else 1.0
            pos_threshold = lv["pos_threshold"]
            target_reached = ((dist_to_target < pos_threshold) &
                             (orient_err * np.pi < orient_threshold)).float()

            current_height = root_pos[:, 2:3]
            height_cmd_obs = self.height_cmd.unsqueeze(-1)
            height_err = (height_cmd_obs - current_height) / 0.4

            estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
            object_in_hand_obs = torch.zeros(self.num_envs, 1, device=self.device)
            object_rel_ee_obs = torch.zeros(self.num_envs, 3, device=self.device)

            lin_vel_b = quat_apply_inverse(root_quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w)
            lin_vel_xy = lin_vel_b[:, :2]
            ang_vel_z = ang_vel_b[:, 2:3]

            obs = torch.cat([
                arm_pos, arm_vel, finger_pos,
                ee_body, ee_vel_body, palm_quat, grip_force, gripper_closed_ratio, contact_detected,
                target_body, pos_error, pos_dist, orient_err, target_reached,
                height_cmd_obs, current_height, height_err,
                estimated_load, object_in_hand_obs, object_rel_ee_obs,
                lin_vel_xy, ang_vel_z,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num()

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()
            leg_actions = actions[:, :12]
            arm_actions = actions[:, 12:17]  # Only 5 arm joints, NO FINGERS

            target_pos = self.robot.data.default_joint_pos.clone()
            target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
            target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5

            # FINGERS ALWAYS OPEN - NO CONTROL
            target_pos[:, self.finger_idx] = self.finger_lower

            self.robot.set_joint_position_target(target_pos)
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            ee_pos, palm_quat = self._compute_palm_ee()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            lv = CURRICULUM[self.curr_level]
            reach_threshold = lv["pos_threshold"]
            if lv["use_orientation"]:
                orient_err = compute_orientation_error(palm_quat)
                orient_threshold = lv.get("orient_threshold", 0.5)
                reached = (dist < reach_threshold) & (orient_err < orient_threshold)
            else:
                reached = dist < reach_threshold

            new_reaches = reached & ~self.already_reached
            if new_reaches.any():
                reached_ids = torch.where(new_reaches)[0]
                self.reach_count[reached_ids] += 1
                self.total_reaches += len(reached_ids)
                self.stage_reaches += len(reached_ids)
                self.already_reached[reached_ids] = True
                self._sample_commands(reached_ids)

            self._init_markers()
            default_quat = torch.tensor([[1, 0, 0, 0]], device=self.device).expand(self.num_envs, -1)
            self.target_markers.visualize(translations=target_world, orientations=default_quat)
            self.ee_markers.visualize(translations=ee_pos, orientations=default_quat)

            self.prev_ee_pos = ee_pos.clone()
            self._prev_prev_leg_actions = self._prev_leg_actions.clone()
            self._prev_prev_arm_actions = self._prev_arm_actions.clone()
            self._prev_leg_actions = self.prev_leg_actions.clone()
            self._prev_arm_actions = self.prev_arm_actions.clone()
            self.prev_leg_actions = leg_actions.clone()
            self.prev_arm_actions = arm_actions.clone()

            self.stage_steps += 1

        def _apply_action(self):
            pass

        def _get_observations(self) -> dict:
            return {"policy": self.get_loco_obs()}

        def compute_loco_reward(self) -> torch.Tensor:
            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w

            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

            r_vx = torch.exp(-2.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
            r_vy = torch.exp(-3.0 * (lin_vel_b[:, 1] - self.vel_cmd[:, 1]) ** 2)
            r_vyaw = torch.exp(-2.0 * (ang_vel_b[:, 2] - self.vel_cmd[:, 2]) ** 2)

            height_error = pos[:, 2] - self.height_cmd
            r_height = torch.exp(-10.0 * height_error ** 2)

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
            proj_gravity = quat_apply_inverse(quat, gravity_vec)
            r_orientation = torch.exp(-5.0 * (proj_gravity[:, 0]**2 + proj_gravity[:, 1]**2))

            joint_pos = robot.data.joint_pos[:, self.leg_idx]
            left_knee, right_knee = joint_pos[:, 6], joint_pos[:, 7]
            left_swing = (self.phase < 0.5).float()
            right_swing = (self.phase >= 0.5).float()
            knee_err = (
                (left_knee - (left_swing * 0.6 + (1 - left_swing) * 0.3)) ** 2 +
                (right_knee - (right_swing * 0.6 + (1 - right_swing) * 0.3)) ** 2
            )
            r_gait = torch.exp(-3.0 * knee_err)

            lateral_vel = lin_vel_b[:, :2].norm(dim=-1)
            r_com_stability = torch.exp(-2.0 * lateral_vel)

            hip_roll_error = joint_pos[:, 2:4].pow(2).sum(-1)
            ankle_roll_error = joint_pos[:, 10:12].pow(2).sum(-1)
            hip_yaw_error = joint_pos[:, 4:6].pow(2).sum(-1)
            posture_error = hip_roll_error * 2.0 + ankle_roll_error * 1.5 + hip_yaw_error * 1.0
            r_leg_posture = torch.exp(-3.0 * posture_error)

            vel_cmd_magnitude = self.vel_cmd.abs().sum(dim=-1)
            actual_vel_magnitude = lin_vel_b[:, :2].abs().sum(dim=-1) + ang_vel_b[:, 2].abs()
            is_standing_cmd = vel_cmd_magnitude < 0.1
            r_standing_still = torch.where(
                is_standing_cmd,
                torch.exp(-5.0 * actual_vel_magnitude),
                torch.ones_like(actual_vel_magnitude)
            )

            ankle_pitch = joint_pos[:, 8:10]
            ankle_roll = joint_pos[:, 10:12]
            ankle_deviation = ankle_pitch.pow(2).sum(-1) + ankle_roll.pow(2).sum(-1) * 1.5
            r_foot_stability = torch.exp(-3.0 * ankle_deviation)

            leg_diff = self.prev_leg_actions - self._prev_leg_actions
            p_action_rate = leg_diff.pow(2).sum(-1)

            leg_accel = self.prev_leg_actions - 2 * self._prev_leg_actions + self._prev_prev_leg_actions
            p_jerk = leg_accel.pow(2).sum(-1)

            leg_vel = robot.data.joint_vel[:, self.leg_idx]
            p_energy = (leg_vel.abs() * self.prev_leg_actions.abs()).sum(-1)

            loco_reward = (
                LOCO_REWARD_WEIGHTS["vx"] * r_vx +
                LOCO_REWARD_WEIGHTS["vy"] * r_vy +
                LOCO_REWARD_WEIGHTS["vyaw"] * r_vyaw +
                LOCO_REWARD_WEIGHTS["height"] * r_height +
                LOCO_REWARD_WEIGHTS["orientation"] * r_orientation +
                LOCO_REWARD_WEIGHTS["gait"] * r_gait +
                LOCO_REWARD_WEIGHTS["com_stability"] * r_com_stability +
                LOCO_REWARD_WEIGHTS["leg_posture"] * r_leg_posture +
                LOCO_REWARD_WEIGHTS["standing_still"] * r_standing_still +
                LOCO_REWARD_WEIGHTS["foot_stability"] * r_foot_stability +
                LOCO_REWARD_WEIGHTS["action_rate"] * p_action_rate +
                LOCO_REWARD_WEIGHTS["jerk"] * p_jerk +
                LOCO_REWARD_WEIGHTS["energy"] * p_energy +
                LOCO_REWARD_WEIGHTS["alive"]
            )

            return loco_reward.clamp(-5, 25)

        def compute_arm_reward(self) -> torch.Tensor:
            """Compute ARM reward - NO GRIPPER!"""
            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w
            lv = CURRICULUM[self.curr_level]

            ee_pos, palm_quat = self._compute_palm_ee()
            target_world = pos + quat_apply(quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            r_distance = torch.exp(-8.0 * dist)

            reach_threshold = lv["pos_threshold"]
            r_reaching = torch.sigmoid((reach_threshold - dist) * 30.0)

            r_final_push = torch.exp(-15.0 * dist) * torch.sigmoid((0.08 - dist) * 25.0)

            arm_diff = self.prev_arm_actions - self._prev_arm_actions
            r_smooth = torch.exp(-1.0 * arm_diff.pow(2).sum(-1))

            if lv["use_orientation"]:
                orient_err = compute_orientation_error(palm_quat)
                r_palm_orient = torch.exp(-3.0 * orient_err)
            else:
                r_palm_orient = torch.zeros(self.num_envs, device=self.device)

            # NO GRIPPER REWARD!

            ee_body = quat_apply_inverse(quat, ee_pos - pos)
            x_violation = torch.clamp(0.05 - ee_body[:, 0], min=0) + torch.clamp(ee_body[:, 0] - 0.55, min=0)
            y_violation = torch.clamp(-0.55 - ee_body[:, 1], min=0) + torch.clamp(ee_body[:, 1] - 0.10, min=0)
            z_violation = torch.clamp(-0.25 - ee_body[:, 2], min=0) + torch.clamp(ee_body[:, 2] - 0.55, min=0)
            p_workspace = (x_violation + y_violation + z_violation) * 5.0

            arm_diff = self.prev_arm_actions - self._prev_arm_actions
            p_action_rate = arm_diff.pow(2).sum(-1)

            arm_accel = self.prev_arm_actions - 2 * self._prev_arm_actions + self._prev_prev_arm_actions
            p_jerk = arm_accel.pow(2).sum(-1)

            arm_reward = (
                ARM_REWARD_WEIGHTS["distance"] * r_distance +
                ARM_REWARD_WEIGHTS["reaching"] * r_reaching +
                ARM_REWARD_WEIGHTS["final_push"] * r_final_push +
                ARM_REWARD_WEIGHTS["smooth"] * r_smooth +
                ARM_REWARD_WEIGHTS["palm_orient"] * r_palm_orient +
                # NO GRIPPER REWARD!
                ARM_REWARD_WEIGHTS["action_rate"] * p_action_rate +
                ARM_REWARD_WEIGHTS["jerk"] * p_jerk +
                ARM_REWARD_WEIGHTS["workspace_violation"] * p_workspace +
                ARM_REWARD_WEIGHTS["alive"]
            )

            return arm_reward.clamp(-5, 30)

        def _get_rewards(self) -> torch.Tensor:
            self.loco_reward = self.compute_loco_reward()
            self.arm_reward = self.compute_arm_reward()
            combined = self.loco_reward + self.arm_reward

            phase_name = "P1-Stand" if self.curr_level < 5 else "P2-Walk"

            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w
            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ee_pos, _ = self._compute_palm_ee()
            target_world = pos + quat_apply(quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            self.extras = {
                "R/loco_total": self.loco_reward.mean().item(),
                "R/arm_total": self.arm_reward.mean().item(),
                "M/height": pos[:, 2].mean().item(),
                "M/vx": lin_vel_b[:, 0].mean().item(),
                "M/ee_dist": dist.mean().item(),
                "M/reaches": self.total_reaches,
                "curriculum_level": self.curr_level,
                "phase": phase_name,
            }

            return combined.clamp(-10, 50)

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
            self._prev_leg_actions[env_ids] = 0
            self._prev_arm_actions[env_ids] = 0
            self._prev_prev_leg_actions[env_ids] = 0
            self._prev_prev_arm_actions[env_ids] = 0
            self.prev_ee_pos[env_ids] = 0
            self.reach_count[env_ids] = 0
            self.already_reached[env_ids] = False

        def update_curriculum(self, mean_reward):
            lv = CURRICULUM[self.curr_level]
            if lv["min_reaches"] is None:
                return

            min_steps = lv.get("min_steps", 0)
            if self.stage_steps >= min_steps and self.stage_reaches >= lv["min_reaches"]:
                success_rate = self.stage_reaches / max(self.stage_steps, 1)
                if success_rate >= lv["success_rate"]:
                    if self.curr_level < len(CURRICULUM) - 1:
                        self.curr_level += 1
                        new_lv = CURRICULUM[self.curr_level]

                        phase_msg = ""
                        if self.curr_level == 5:
                            phase_msg = " 🚶 PHASE 2: WALKING + REACHING!"

                        print(f"\n{'='*60}")
                        print(f"🎯 LEVEL UP! Level {self.curr_level}{phase_msg}")
                        print(f"   vx={new_lv['vx']}, pos_thresh={new_lv['pos_threshold']}")
                        print(f"   orient={new_lv['use_orientation']}")
                        print(f"   Reaches: {self.stage_reaches}, SR: {success_rate:.2%}")
                        print(f"{'='*60}\n")

                        self.stage_reaches = 0
                        self.stage_steps = 0

    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    return Stage6SimplifiedEnv(cfg)


# ============================================================================
# WEIGHT TRANSFER WITH OBS FILLER
# ============================================================================

def transfer_weights_with_filler(net, stage3_path, stage5_path, device):
    """
    Transfer weights from Stage 3 (loco) and/or Stage 5 (arm) checkpoints.
    Uses filler logic for obs dimension mismatches.
    """

    # Stage 3: Loco weights (57 obs → 12 act)
    if stage3_path is not None:
        print(f"\n[Transfer] Loading Stage 3 (Loco): {stage3_path}")
        try:
            ckpt = torch.load(stage3_path, map_location=device, weights_only=False)

            # RSL-RL format
            if "actor_critic" in ckpt:
                s3_state = ckpt["actor_critic"]
            elif "model" in ckpt:
                s3_state = ckpt["model"]
            else:
                s3_state = ckpt

            transferred = 0

            for key in s3_state:
                # Actor weights
                if key.startswith("actor."):
                    new_key = "loco_actor.net." + key[6:]
                    if new_key in net.state_dict():
                        src_shape = s3_state[key].shape
                        dst_shape = net.state_dict()[new_key].shape
                        if src_shape == dst_shape:
                            net.state_dict()[new_key].copy_(s3_state[key])
                            transferred += 1
                        else:
                            print(f"   [FILLER] {key}: {src_shape} → {dst_shape} (padding)")
                            # Pad with zeros if needed
                            if len(src_shape) == 2:  # Weight matrix
                                min_rows = min(src_shape[0], dst_shape[0])
                                min_cols = min(src_shape[1], dst_shape[1])
                                net.state_dict()[new_key][:min_rows, :min_cols].copy_(
                                    s3_state[key][:min_rows, :min_cols]
                                )
                                transferred += 1
                            elif len(src_shape) == 1:  # Bias vector
                                min_len = min(src_shape[0], dst_shape[0])
                                net.state_dict()[new_key][:min_len].copy_(
                                    s3_state[key][:min_len]
                                )
                                transferred += 1

                # Critic weights
                elif key.startswith("critic."):
                    new_key = "loco_critic.net." + key[7:]
                    if new_key in net.state_dict():
                        src_shape = s3_state[key].shape
                        dst_shape = net.state_dict()[new_key].shape
                        if src_shape == dst_shape:
                            net.state_dict()[new_key].copy_(s3_state[key])
                            transferred += 1
                        else:
                            print(f"   [FILLER] {key}: {src_shape} → {dst_shape} (padding)")
                            if len(src_shape) == 2:
                                min_rows = min(src_shape[0], dst_shape[0])
                                min_cols = min(src_shape[1], dst_shape[1])
                                net.state_dict()[new_key][:min_rows, :min_cols].copy_(
                                    s3_state[key][:min_rows, :min_cols]
                                )
                                transferred += 1
                            elif len(src_shape) == 1:
                                min_len = min(src_shape[0], dst_shape[0])
                                net.state_dict()[new_key][:min_len].copy_(
                                    s3_state[key][:min_len]
                                )
                                transferred += 1

                # Log std
                elif key == "log_std" or key == "std":
                    if "loco_actor.log_std" in net.state_dict():
                        src_shape = s3_state[key].shape
                        dst_shape = net.state_dict()["loco_actor.log_std"].shape
                        min_len = min(src_shape[0], dst_shape[0])
                        net.state_dict()["loco_actor.log_std"][:min_len].copy_(
                            s3_state[key][:min_len]
                        )
                        transferred += 1

            print(f"[Transfer] Stage 3 → Loco: {transferred} parameters transferred")

        except Exception as e:
            print(f"[Transfer] Stage 3 load failed: {e}")

    # Stage 5: Arm weights (if available)
    if stage5_path is not None:
        print(f"\n[Transfer] Loading Stage 5 (Arm): {stage5_path}")
        try:
            ckpt = torch.load(stage5_path, map_location=device, weights_only=False)

            if "actor_critic" in ckpt:
                s5_state = ckpt["actor_critic"]
            elif "model" in ckpt:
                s5_state = ckpt["model"]
            else:
                s5_state = ckpt

            transferred = 0

            # Look for arm-related weights
            for key in s5_state:
                # Try to find arm actor weights
                if "arm" in key.lower() and "actor" in key.lower():
                    new_key = key.replace("arm_actor.", "arm_actor.net.")
                    if new_key in net.state_dict():
                        src_shape = s5_state[key].shape
                        dst_shape = net.state_dict()[new_key].shape
                        if src_shape == dst_shape:
                            net.state_dict()[new_key].copy_(s5_state[key])
                            transferred += 1
                        else:
                            print(f"   [FILLER] {key}: {src_shape} → {dst_shape}")

            print(f"[Transfer] Stage 5 → Arm: {transferred} parameters transferred")

        except Exception as e:
            print(f"[Transfer] Stage 5 load failed: {e}")

    if stage3_path is None and stage5_path is None:
        print("[Transfer] No checkpoints provided, starting fresh")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    device = "cuda:0"

    print(f"\n[INFO] Creating environment with {args_cli.num_envs} envs...")
    env = create_env(args_cli.num_envs, device)

    print(f"[INFO] Creating DUAL Actor-Critic network (SIMPLIFIED)...")
    print("       LocoActor (57→12) + LocoCritic (57→1)")
    print("       ArmActor (52→5)   + ArmCritic (52→1)  ← NO FINGERS")
    net = DualActorCritic(loco_obs=57, arm_obs=52, loco_act=12, arm_act=5).to(device)

    # Transfer weights from Stage 3 and/or Stage 5
    transfer_weights_with_filler(net, args_cli.stage3_checkpoint, args_cli.stage5_checkpoint, device)

    start_iter = 0
    if args_cli.checkpoint:
        print(f"\n[INFO] Resuming from: {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0)
        env.curr_level = ckpt.get("curriculum_level", 0)

    ppo = DualPPO(net, device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"\n[INFO] Logging to: {log_dir}")

    best_reward = float('-inf')

    net.loco_actor.log_std.data.fill_(np.log(0.5))
    net.arm_actor.log_std.data.fill_(np.log(0.5))

    obs, _ = env.reset()

    print("\n" + "=" * 80)
    print("STARTING SIMPLIFIED TRAINING - 10 LEVELS, NO GRIPPER")
    print("  Phase 1 (Level 0-4): Standing + Reaching")
    print("  Phase 2 (Level 5-9): Walking + Reaching + Orientation")
    print("  Target: ~15000 iterations for full curriculum")
    print("=" * 80 + "\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        loco_obs_buf, arm_obs_buf = [], []
        loco_act_buf, arm_act_buf = [], []
        loco_logp_buf, arm_logp_buf = [], []
        loco_val_buf, arm_val_buf = [], []
        loco_rew_buf, arm_rew_buf = [], []
        done_buf = []

        rollout_steps = 24

        for _ in range(rollout_steps):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            with torch.no_grad():
                loco_action, arm_action, loco_logp, arm_logp = net.get_actions(loco_obs, arm_obs)
                loco_value = net.loco_critic(loco_obs)
                arm_value = net.arm_critic(arm_obs)

            loco_obs_buf.append(loco_obs)
            arm_obs_buf.append(arm_obs)
            loco_act_buf.append(loco_action)
            arm_act_buf.append(arm_action)
            loco_logp_buf.append(loco_logp)
            arm_logp_buf.append(arm_logp)
            loco_val_buf.append(loco_value)
            arm_val_buf.append(arm_value)

            # Combine actions: 12 leg + 5 arm = 17 total
            actions = torch.cat([loco_action, arm_action], dim=-1)
            obs_dict, combined_reward, terminated, truncated, _ = env.step(actions)

            loco_rew_buf.append(env.loco_reward)
            arm_rew_buf.append(env.arm_reward)
            done_buf.append((terminated | truncated).float())

        loco_obs_buf = torch.stack(loco_obs_buf)
        arm_obs_buf = torch.stack(arm_obs_buf)
        loco_act_buf = torch.stack(loco_act_buf)
        arm_act_buf = torch.stack(arm_act_buf)
        loco_logp_buf = torch.stack(loco_logp_buf)
        arm_logp_buf = torch.stack(arm_logp_buf)
        loco_val_buf = torch.stack(loco_val_buf)
        arm_val_buf = torch.stack(arm_val_buf)
        loco_rew_buf = torch.stack(loco_rew_buf)
        arm_rew_buf = torch.stack(arm_rew_buf)
        done_buf = torch.stack(done_buf)

        with torch.no_grad():
            final_loco_obs = env.get_loco_obs()
            final_arm_obs = env.get_arm_obs()
            loco_next_val = net.loco_critic(final_loco_obs)
            arm_next_val = net.arm_critic(final_arm_obs)

        loco_adv, loco_ret = ppo.gae(loco_rew_buf, loco_val_buf, done_buf, loco_next_val)
        arm_adv, arm_ret = ppo.gae(arm_rew_buf, arm_val_buf, done_buf, arm_next_val)

        loco_loss = ppo.update_loco(
            loco_obs_buf.view(-1, 57),
            loco_act_buf.view(-1, 12),
            loco_logp_buf.view(-1),
            loco_ret.view(-1),
            loco_adv.view(-1),
            loco_val_buf.view(-1),
        )

        arm_loss = ppo.update_arm(
            arm_obs_buf.view(-1, 52),
            arm_act_buf.view(-1, 5),  # Only 5 arm actions!
            arm_logp_buf.view(-1),
            arm_ret.view(-1),
            arm_adv.view(-1),
            arm_val_buf.view(-1),
        )

        progress = iteration / args_cli.max_iterations
        loco_std = 0.5 + (0.15 - 0.5) * progress
        arm_std = 0.5 + (0.15 - 0.5) * progress
        net.loco_actor.log_std.data.fill_(np.log(loco_std))
        net.arm_actor.log_std.data.fill_(np.log(arm_std))

        mean_loco_reward = loco_rew_buf.mean().item()
        mean_arm_reward = arm_rew_buf.mean().item()
        mean_reward = mean_loco_reward + mean_arm_reward

        env.update_curriculum(mean_reward)

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "total_reaches": env.total_reaches,
            }, f"{log_dir}/model_best.pt")

        writer.add_scalar("Train/reward", mean_reward, iteration)
        writer.add_scalar("Train/loco_reward", mean_loco_reward, iteration)
        writer.add_scalar("Train/arm_reward", mean_arm_reward, iteration)
        writer.add_scalar("Train/best_reward", best_reward, iteration)
        writer.add_scalar("Loss/loco", loco_loss, iteration)
        writer.add_scalar("Loss/arm", arm_loss, iteration)
        writer.add_scalar("Curriculum/level", env.curr_level, iteration)
        writer.add_scalar("Curriculum/total_reaches", env.total_reaches, iteration)

        for key, val in env.extras.items():
            if isinstance(val, (int, float)):
                writer.add_scalar(f"Env/{key}", val, iteration)

        if iteration % 10 == 0:
            phase = env.extras.get("phase", "P1")
            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:5.1f} (L={mean_loco_reward:5.1f} A={mean_arm_reward:5.1f}) | "
                f"Best={best_reward:5.1f} | "
                f"Lv={env.curr_level:2d} ({phase}) | "
                f"Reaches={env.total_reaches} | "
                f"EE={env.extras.get('M/ee_dist', 0):.3f}"
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
    print("TRAINING COMPLETE!")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Total Reaches: {env.total_reaches}")
    print(f"  Log Dir: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
    simulation_app.close()