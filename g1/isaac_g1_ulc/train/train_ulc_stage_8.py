"""
ULC G1 Stage 8: Position Perfection + Orientation Training
============================================================
DUAL ACTOR-CRITIC - LOCO FROZEN, ARM FINE-TUNED FROM STAGE 7

BUILDS ON STAGE 7: Fine-tunes arm reaching to achieve >90% validated rate
at 4cm threshold, then adds palm orientation control.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────┐
│     LOCO BRANCH (FROZEN)    │    ARM BRANCH (FINE-TUNE)   │
├─────────────────────────────┼────────────────────────────┤
│  LocoActor (57→12) FROZEN   │  ArmActor (55→5) FROM ST7   │
│  LocoCritic (57→1) FROZEN   │  ArmCritic (55→1) FROM ST7  │
└─────────────────────────────┴────────────────────────────┘

ARM OBS: 55 (same as Stage 7)

CURRICULUM (11 levels, 2 phases):
- Phase 1 (Level 0-5): Position perfection (4cm, >90% rate)
- Phase 2 (Level 6-10): Palm orientation (progressive cone widening)

KEY DIFFERENCE FROM STAGE 7:
- Arm weights FINE-TUNED (not fresh) — preserves reaching skill
- Lower initial LR (1e-4 vs 3e-4) and exploration (log(0.3) vs log(0.8))
- Phase 2 transition: LR bump + exploration bump + critic reinit
- Two-component orientation reward (continuous + bonus)
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
# REWARD WEIGHTS - LOCO UNCHANGED, ARM MOVEMENT-CENTRIC
# ============================================================================

# LOCOMOTION REWARDS - Same as Stage 6 (frozen branch)
LOCO_REWARD_WEIGHTS = {
    "vx": 5.0,
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

# ARM REWARDS - MOVEMENT-CENTRIC (anti-gaming)
ARM_REWARD_WEIGHTS = {
    "velocity_toward": 8.0,   # NEW: dot(ee_vel, direction_to_target)
    "progress": 6.0,          # NEW: (initial_dist - current_dist) / initial_dist
    "distance": 2.0,          # REDUCED from 4.0
    "reaching": 10.0,         # KEPT: sigmoid at threshold
    "final_push": 4.0,        # REDUCED from 6.0
    "smooth": 1.0,            # REDUCED from 3.0, only when close
    "stillness_penalty": -2.0, # NEW: penalize stillness when far
    "palm_orient": 7.0,       # Phase 2: 4.0 continuous + 3.0 bonus (stronger than Stage 7's 3.0)
    # Penalties
    "action_rate": -0.05,
    "jerk": -0.03,
    "workspace_violation": -2.0,
    "alive": 0.3,
}


# ============================================================================
# ANTI-GAMING CURRICULUM (11 levels, 2 phases)
# ============================================================================

def create_antigaming_curriculum():
    """
    11-level curriculum with 2 phases

    Phase 1 (Level 0-5): Position perfection — 4cm threshold, >90% validated rate
    Phase 2 (Level 6-10): Palm orientation — progressive cone widening
    """
    levels = []

    # ============================================================
    # PHASE 1: POSITION PERFECTION (Levels 0-5)
    # No orientation check — focus purely on position
    # ============================================================

    # Level 0: Standing + Easy reaching (warm up from Stage 7 weights)
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.08,
        "min_target_distance": 0.10,
        "min_displacement": 0.05,
        "max_reach_steps": 150,
        "use_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.50,
        "min_validated_reaches": 5000,
        "min_steps": 200000,
        "description": "Phase1: Stand + Easy reach (0.08m thresh, 50% rate)",
    })

    # Level 1: Tighter threshold
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.06,
        "min_target_distance": 0.12,
        "min_displacement": 0.07,
        "max_reach_steps": 140,
        "use_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.50,
        "min_validated_reaches": 6000,
        "min_steps": 250000,
        "description": "Phase1: Stand + Medium reach (0.06m thresh, 50% rate)",
    })

    # Level 2: Getting precise
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.05,
        "min_target_distance": 0.14,
        "min_displacement": 0.08,
        "max_reach_steps": 130,
        "use_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.45,
        "min_validated_reaches": 7000,
        "min_steps": 300000,
        "description": "Phase1: Stand + Precise reach (0.05m thresh, 45% rate)",
    })

    # Level 3: Target precision (4cm!)
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.04,
        "min_target_distance": 0.15,
        "min_displacement": 0.10,
        "max_reach_steps": 120,
        "use_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.40,
        "min_validated_reaches": 8000,
        "min_steps": 350000,
        "description": "Phase1: Stand + 4cm reach (40% rate)",
    })

    # Level 4: Walking + 4cm
    levels.append({
        "vx_range": (0.0, 0.3), "vy_range": (-0.05, 0.05), "vyaw_range": (-0.10, 0.10),
        "pos_threshold": 0.04,
        "min_target_distance": 0.15,
        "min_displacement": 0.10,
        "max_reach_steps": 130,
        "use_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.35,
        "min_validated_reaches": 8000,
        "min_steps": 350000,
        "description": "Phase1: Walk(0.3) + 4cm reach (35% rate)",
    })

    # Level 5: Faster walking + 4cm
    levels.append({
        "vx_range": (0.0, 0.5), "vy_range": (-0.07, 0.07), "vyaw_range": (-0.13, 0.13),
        "pos_threshold": 0.04,
        "min_target_distance": 0.16,
        "min_displacement": 0.10,
        "max_reach_steps": 140,
        "use_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.30,
        "min_validated_reaches": 9000,
        "min_steps": 400000,
        "description": "Phase1: Walk(0.5) + 4cm reach (30% rate)",
    })

    # ============================================================
    # PHASE 2: ORIENTATION (Levels 6-10)
    # Position threshold relaxed to 5cm, orientation progressively tightens
    # ============================================================

    # Level 6: Stand + Easy orientation (palm down, very loose)
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.05,
        "orient_threshold": 2.5,  # ~143 deg — very loose
        "min_target_distance": 0.12,
        "min_displacement": 0.08,
        "max_reach_steps": 150,
        "use_orientation": True,
        "variable_orientation": False,  # Palm down only
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.25,
        "min_validated_reaches": 8000,
        "min_steps": 400000,
        "description": "Phase2: Stand + Orient palm_down (2.5rad, 25% rate)",
    })

    # Level 7: Tighter palm down
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.05,
        "orient_threshold": 2.0,  # ~115 deg
        "min_target_distance": 0.14,
        "min_displacement": 0.08,
        "max_reach_steps": 150,
        "use_orientation": True,
        "variable_orientation": False,
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.22,
        "min_validated_reaches": 8000,
        "min_steps": 400000,
        "description": "Phase2: Stand + Orient palm_down (2.0rad, 22% rate)",
    })

    # Level 8: Variable orient — 20 deg cone
    levels.append({
        "vx_range": (0.0, 0.0), "vy_range": (0.0, 0.0), "vyaw_range": (0.0, 0.0),
        "pos_threshold": 0.05,
        "orient_threshold": 1.5,  # ~86 deg
        "min_target_distance": 0.14,
        "min_displacement": 0.08,
        "max_reach_steps": 160,
        "use_orientation": True,
        "variable_orientation": True,
        "orient_sample_range": 0.35,  # ~20 deg cone
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.20,
        "min_validated_reaches": 9000,
        "min_steps": 450000,
        "description": "Phase2: Stand + Variable orient 20deg (1.5rad, 20% rate)",
    })

    # Level 9: Walk + 40 deg cone
    levels.append({
        "vx_range": (0.0, 0.3), "vy_range": (-0.05, 0.05), "vyaw_range": (-0.10, 0.10),
        "pos_threshold": 0.05,
        "orient_threshold": 1.2,  # ~69 deg
        "min_target_distance": 0.15,
        "min_displacement": 0.10,
        "max_reach_steps": 170,
        "use_orientation": True,
        "variable_orientation": True,
        "orient_sample_range": 0.70,  # ~40 deg cone
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": 0.18,
        "min_validated_reaches": 9000,
        "min_steps": 450000,
        "description": "Phase2: Walk(0.3) + Variable orient 40deg (1.2rad, 18% rate)",
    })

    # Level 10: Walk + 60 deg cone (FINAL)
    levels.append({
        "vx_range": (0.0, 0.5), "vy_range": (-0.07, 0.07), "vyaw_range": (-0.13, 0.13),
        "pos_threshold": 0.05,
        "orient_threshold": 1.0,  # ~57 deg
        "min_target_distance": 0.16,
        "min_displacement": 0.10,
        "max_reach_steps": 180,
        "use_orientation": True,
        "variable_orientation": True,
        "orient_sample_range": 1.05,  # ~60 deg cone
        "workspace_radius": (0.18, 0.40),
        "validated_reach_rate": None,  # FINAL level
        "min_validated_reaches": None,
        "min_steps": None,
        "description": "Phase2 FINAL: Walk(0.5) + Variable orient 60deg (1.0rad)",
    })

    return levels


CURRICULUM = create_antigaming_curriculum()


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 8: Position Perfection + Orientation Training")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--stage7_checkpoint", type=str, default=None,
                        help="Stage 7 checkpoint (arm fine-tuned, loco frozen). Required for fresh start.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from Stage 8 checkpoint")
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage8_orient")
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


def compute_orientation_error(palm_quat: torch.Tensor, target_dir: torch.Tensor = None) -> torch.Tensor:
    forward = get_palm_forward(palm_quat)
    if target_dir is None:
        target_dir = torch.zeros_like(forward)
        target_dir[:, 2] = -1.0
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.acos(dot)


print("=" * 80)
print("ULC G1 STAGE 8: POSITION PERFECTION + ORIENTATION TRAINING")
print("=" * 80)
print("ANTI-GAMING MECHANISMS (inherited from Stage 7):")
print("  1. Absolute-only sampling + min distance enforcement")
print("  2. 3-condition reach validation (position + displacement + time)")
print("  3. Validated reach rate for curriculum advancement")
print("  4. Movement-centric rewards (velocity_toward, progress, stillness_penalty)")
print("  5. Strict 11-level curriculum (2 phases)")
print()
print("Architecture:")
print("  LocoActor (57->12) FROZEN    + LocoCritic (57->1) FROZEN")
print("  ArmActor  (55->5)  FINE-TUNE + ArmCritic  (55->1) FINE-TUNE")
print(f"\nCurriculum: {len(CURRICULUM)} levels")
print("  Phase 1 (0-5): Position Perfection (4cm threshold)")
print("  Phase 2 (6-10): Palm Orientation (progressive cone)")
print("=" * 80)


# ============================================================================
# DUAL ACTOR-CRITIC NETWORKS
# ============================================================================

class LocoActor(nn.Module):
    """Locomotion policy: 57 obs -> 12 leg actions (FROZEN)"""
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
    """Locomotion value function: 57 obs -> 1 value (FROZEN)"""
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
    """Arm policy: 55 obs -> 5 actions (FINE-TUNED from Stage 7)"""
    def __init__(self, num_obs=55, num_act=5, hidden=[256, 256, 128]):
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
    """Arm value function: 55 obs -> 1 value (FINE-TUNED from Stage 7)"""
    def __init__(self, num_obs=55, hidden=[256, 256, 128]):
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
    Dual Actor-Critic - Stage 8 (LOCO FROZEN, ARM FINE-TUNED)

    Architecture:
    - LocoActor (57->12) + LocoCritic (57->1)  -- FROZEN from Stage 6/7
    - ArmActor (55->5)   + ArmCritic (55->1)   -- FINE-TUNED from Stage 7
    """
    def __init__(self, loco_obs=57, arm_obs=55, loco_act=12, arm_act=5):
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
        # Loco always deterministic (frozen)
        loco_mean = self.loco_actor(loco_obs)
        arm_mean = self.arm_actor(arm_obs)

        if deterministic:
            return loco_mean, arm_mean

        # Only arm uses stochastic exploration
        arm_std = self.arm_actor.log_std.clamp(-2, 1).exp()
        arm_dist = torch.distributions.Normal(arm_mean, arm_std)
        arm_action = arm_dist.sample()
        arm_logp = arm_dist.log_prob(arm_action).sum(-1)

        # Loco: no noise needed (frozen), but return logp=0 for compatibility
        loco_std = self.loco_actor.log_std.clamp(-2, 1).exp()
        loco_dist = torch.distributions.Normal(loco_mean, loco_std)
        loco_action = loco_dist.sample()
        loco_logp = loco_dist.log_prob(loco_action).sum(-1)

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
# DUAL PPO TRAINER (ARM ONLY - LOCO FROZEN)
# ============================================================================

class DualPPO:
    def __init__(self, net, device, arm_lr=3e-4):
        self.net = net
        self.device = device

        # Loco optimizer: NO-OP (frozen, but we keep structure for compatibility)
        # We don't create loco_opt since loco is frozen

        # Arm optimizer: active training
        self.arm_opt = torch.optim.AdamW(
            list(net.arm_actor.parameters()) + list(net.arm_critic.parameters()),
            lr=arm_lr, weight_decay=1e-5
        )

        self.arm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.arm_opt, args_cli.max_iterations, eta_min=5e-5
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
        action_space = 17  # 12 leg + 5 arm
        observation_space = 57
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=4.0)

    class Stage8OrientEnv(DirectRLEnv):
        """Stage 8 environment with anti-gaming mechanisms + orientation training"""
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

            print(f"[Stage8] Leg joints: {len(self.leg_idx)}")
            print(f"[Stage8] Arm joints: {len(self.arm_idx)}")
            print(f"[Stage8] Finger joints: {len(self.finger_idx)} (FIXED OPEN)")
            print(f"[Stage8] Palm body idx: {self.palm_idx}")

            # Commands
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.target_pos_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_orient_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_orient_body[:, 2] = -1.0  # Default: palm down
            self.shoulder_offset = SHOULDER_OFFSET.to(self.device)

            self.phase = torch.zeros(self.num_envs, device=self.device)

            # Action history
            self.prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self.prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
            self._prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)
            self._prev_prev_leg_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_prev_arm_actions = torch.zeros(self.num_envs, 5, device=self.device)

            self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

            # ══════════════════════════════════════════════════════════════
            # ANTI-GAMING STATE BUFFERS
            # ══════════════════════════════════════════════════════════════
            self.ee_pos_at_spawn = torch.zeros(self.num_envs, 3, device=self.device)
            self.steps_since_spawn = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.initial_dist = torch.zeros(self.num_envs, device=self.device)

            # Curriculum tracking
            self.curr_level = 0
            self.reach_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.total_reaches = 0
            self.validated_reaches = 0         # Only displacement-verified reaches
            self.timed_out_targets = 0         # Targets that timed out
            self.total_attempts = 0            # validated_reaches + timed_out_targets
            self.stage_validated_reaches = 0
            self.stage_timed_out = 0
            self.stage_steps = 0
            self.already_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            self.loco_reward = torch.zeros(self.num_envs, device=self.device)
            self.arm_reward = torch.zeros(self.num_envs, device=self.device)

            # Reward component tracking
            self.loco_reward_components = {}
            self.arm_reward_components = {}
            self.behavior_metrics = {}

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
            """ANTI-GAMING: Absolute-only sampling + min distance enforcement"""
            n = len(env_ids)
            lv = CURRICULUM[self.curr_level]
            self.already_reached[env_ids] = False

            # Velocity commands
            self.vel_cmd[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*lv["vx_range"])
            self.vel_cmd[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*lv["vy_range"])
            self.vel_cmd[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*lv["vyaw_range"])
            self.height_cmd[env_ids] = HEIGHT_DEFAULT

            # Get current EE position in body frame
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]
            ee_world, _ = self._compute_palm_ee()
            ee_world = ee_world[env_ids]
            current_ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

            # ANTI-GAMING: ABSOLUTE-ONLY SAMPLING (no relative, no mixed!)
            workspace_radius = lv.get("workspace_radius", (0.18, 0.40))
            azimuth = torch.empty(n, device=self.device).uniform_(-0.3, 1.2)
            elevation = torch.empty(n, device=self.device).uniform_(-0.4, 0.6)
            radius = torch.empty(n, device=self.device).uniform_(*workspace_radius)
            target_x = radius * torch.cos(elevation) * torch.cos(azimuth) + self.shoulder_offset[0]
            target_y = -radius * torch.cos(elevation) * torch.sin(azimuth) + self.shoulder_offset[1]
            target_z = radius * torch.sin(elevation) + self.shoulder_offset[2]

            # Clamp to workspace bounds
            target_x = target_x.clamp(0.05, 0.55)
            target_y = target_y.clamp(-0.55, 0.10)
            target_z = target_z.clamp(-0.25, 0.55)

            target_body = torch.stack([target_x, target_y, target_z], dim=-1)

            # ANTI-GAMING: Enforce minimum distance from current EE
            min_target_distance = lv["min_target_distance"]
            dist_to_ee = torch.norm(target_body - current_ee_body, dim=-1)
            too_close = dist_to_ee < min_target_distance

            if too_close.any():
                # Push targets outward along ee-to-target direction
                direction = (target_body - current_ee_body)
                direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                direction = direction / direction_norm
                pushed_target = current_ee_body + min_target_distance * direction
                # Re-clamp pushed targets
                pushed_target[:, 0] = pushed_target[:, 0].clamp(0.05, 0.55)
                pushed_target[:, 1] = pushed_target[:, 1].clamp(-0.55, 0.10)
                pushed_target[:, 2] = pushed_target[:, 2].clamp(-0.25, 0.55)
                target_body = torch.where(
                    too_close.unsqueeze(-1).expand_as(target_body),
                    pushed_target,
                    target_body
                )

            self.target_pos_body[env_ids] = target_body

            # Target orientation
            variable_orient = lv.get("variable_orientation", False)
            if variable_orient:
                orient_range = lv.get("orient_sample_range", 0.5)
                theta = torch.empty(n, device=self.device).uniform_(0, orient_range)
                phi = torch.empty(n, device=self.device).uniform_(0, 2 * np.pi)
                dir_x = torch.sin(theta) * torch.cos(phi)
                dir_y = torch.sin(theta) * torch.sin(phi)
                dir_z = -torch.cos(theta)
                self.target_orient_body[env_ids, 0] = dir_x
                self.target_orient_body[env_ids, 1] = dir_y
                self.target_orient_body[env_ids, 2] = dir_z
            else:
                self.target_orient_body[env_ids, 0] = 0.0
                self.target_orient_body[env_ids, 1] = 0.0
                self.target_orient_body[env_ids, 2] = -1.0

            # ANTI-GAMING: Record spawn state for validation
            self.ee_pos_at_spawn[env_ids] = current_ee_body
            self.steps_since_spawn[env_ids] = 0
            # Compute initial distance to target
            self.initial_dist[env_ids] = torch.norm(
                target_body - current_ee_body, dim=-1
            ).clamp(min=0.01)  # Avoid division by zero

        def get_loco_obs(self) -> torch.Tensor:
            """57 dims - Same as Stage 6 (frozen branch)"""
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
            """55 dims = 52 (Stage 6 base) + 3 new anti-gaming obs"""
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

            orient_err = compute_orientation_error(palm_quat, self.target_orient_body).unsqueeze(-1) / np.pi

            orient_threshold = lv.get("orient_threshold", 1.0) if lv["use_orientation"] else 1.0
            pos_threshold = lv["pos_threshold"]
            target_reached = ((dist_to_target < pos_threshold) &
                             (orient_err * np.pi < orient_threshold)).float()

            current_height = root_pos[:, 2:3]
            height_cmd_obs = self.height_cmd.unsqueeze(-1)
            height_err = (height_cmd_obs - current_height) / 0.4

            estimated_load = torch.zeros(self.num_envs, 3, device=self.device)
            object_in_hand_obs = torch.zeros(self.num_envs, 1, device=self.device)
            target_orient_obs = self.target_orient_body

            lin_vel_b = quat_apply_inverse(root_quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w)
            lin_vel_xy = lin_vel_b[:, :2]
            ang_vel_z = ang_vel_b[:, 2:3]

            # === 3 NEW ANTI-GAMING OBSERVATIONS ===
            # Normalized steps since spawn (0 to 1 over max_reach_steps)
            max_steps = float(lv["max_reach_steps"])
            steps_norm = (self.steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)

            # EE displacement from spawn position (how much arm has moved)
            ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1, keepdim=True)

            # Initial distance to target (at spawn time)
            initial_dist_obs = self.initial_dist.unsqueeze(-1) / 0.5  # Normalize

            obs = torch.cat([
                arm_pos, arm_vel, finger_pos,
                ee_body, ee_vel_body, palm_quat, grip_force, gripper_closed_ratio, contact_detected,
                target_body, pos_error, pos_dist, orient_err, target_reached,
                height_cmd_obs, current_height, height_err,
                estimated_load, object_in_hand_obs, target_orient_obs,
                lin_vel_xy, ang_vel_z,
                # 3 new anti-gaming observations
                steps_norm, ee_displacement, initial_dist_obs,
            ], dim=-1)
            return obs.clamp(-10, 10).nan_to_num()

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()
            leg_actions = actions[:, :12]
            arm_actions = actions[:, 12:17]

            target_pos = self.robot.data.default_joint_pos.clone()
            target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4
            target_pos[:, self.arm_idx] = self.default_arm + arm_actions * 0.5

            # FINGERS ALWAYS OPEN
            target_pos[:, self.finger_idx] = self.finger_lower

            self.robot.set_joint_position_target(target_pos)
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

            # Increment step counter for all envs
            self.steps_since_spawn += 1

            ee_pos, palm_quat = self._compute_palm_ee()
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            ee_body = quat_apply_inverse(root_quat, ee_pos - root_pos)
            target_world = root_pos + quat_apply(root_quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            lv = CURRICULUM[self.curr_level]
            reach_threshold = lv["pos_threshold"]

            # ══════════════════════════════════════════════════════════════
            # ANTI-GAMING: 3-CONDITION REACH VALIDATION
            # ══════════════════════════════════════════════════════════════
            pos_reached = dist < reach_threshold

            if lv["use_orientation"]:
                orient_err = compute_orientation_error(palm_quat, self.target_orient_body)
                orient_threshold = lv.get("orient_threshold", 0.5)
                orient_reached = orient_err < orient_threshold
                pos_reached = pos_reached & orient_reached

            # Condition 2: Arm actually moved from spawn position
            ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1)
            min_displacement = lv["min_displacement"]
            moved_enough = ee_displacement >= min_displacement

            # Condition 3: Reached within time limit
            max_reach_steps = lv["max_reach_steps"]
            within_time = self.steps_since_spawn <= max_reach_steps

            # All 3 conditions must be met
            validated_reach = pos_reached & moved_enough & within_time
            new_reaches = validated_reach & ~self.already_reached

            if new_reaches.any():
                reached_ids = torch.where(new_reaches)[0]
                self.reach_count[reached_ids] += 1
                self.total_reaches += len(reached_ids)
                self.validated_reaches += len(reached_ids)
                self.stage_validated_reaches += len(reached_ids)
                self.total_attempts += len(reached_ids)
                self.already_reached[reached_ids] = True
                self._sample_commands(reached_ids)

            # ANTI-GAMING: Handle timed-out targets
            timed_out = (self.steps_since_spawn > max_reach_steps) & ~self.already_reached
            if timed_out.any():
                timed_out_ids = torch.where(timed_out)[0]
                self.timed_out_targets += len(timed_out_ids)
                self.stage_timed_out += len(timed_out_ids)
                self.total_attempts += len(timed_out_ids)
                # Resample for timed-out envs
                self._sample_commands(timed_out_ids)

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
            """Same as Stage 6 - used for logging only (loco is frozen)"""
            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w

            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ang_vel_b = quat_apply_inverse(quat, robot.data.root_ang_vel_w)

            r_vx = torch.exp(-4.0 * (lin_vel_b[:, 0] - self.vel_cmd[:, 0]) ** 2)
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
            is_standing_cmd = vel_cmd_magnitude < 0.05
            r_standing_still = torch.where(
                is_standing_cmd,
                torch.exp(-5.0 * actual_vel_magnitude),
                torch.zeros_like(actual_vel_magnitude)
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

            self.loco_reward_components = {
                "vx": (LOCO_REWARD_WEIGHTS["vx"] * r_vx).mean().item(),
                "height": (LOCO_REWARD_WEIGHTS["height"] * r_height).mean().item(),
                "orientation": (LOCO_REWARD_WEIGHTS["orientation"] * r_orientation).mean().item(),
            }

            self.behavior_metrics = {
                "actual_vx": lin_vel_b[:, 0].mean().item(),
                "cmd_vx": self.vel_cmd[:, 0].mean().item(),
                "actual_height": pos[:, 2].mean().item(),
            }

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
            """ANTI-GAMING: Movement-centric arm reward"""
            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w
            lv = CURRICULUM[self.curr_level]

            ee_pos, palm_quat = self._compute_palm_ee()
            ee_body = quat_apply_inverse(quat, ee_pos - pos)
            target_world = pos + quat_apply(quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            # EE velocity in body frame
            ee_vel_world = (ee_pos - self.prev_ee_pos) / 0.02
            ee_vel_body = quat_apply_inverse(quat, ee_vel_world)
            ee_speed = ee_vel_body.norm(dim=-1)

            # Direction to target in body frame
            target_body = self.target_pos_body
            direction_to_target = target_body - ee_body
            direction_norm = direction_to_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            direction_unit = direction_to_target / direction_norm

            # ══════════════════════════════════════════════════════════════
            # NEW: VELOCITY TOWARD TARGET (biggest anti-gaming signal)
            # Rewards moving toward target, penalizes moving away
            # ══════════════════════════════════════════════════════════════
            velocity_toward = (ee_vel_body * direction_unit).sum(dim=-1)
            r_velocity_toward = velocity_toward.clamp(-1.0, 1.0)

            # ══════════════════════════════════════════════════════════════
            # NEW: PROGRESS (fraction of initial distance closed)
            # ══════════════════════════════════════════════════════════════
            current_dist = dist
            r_progress = ((self.initial_dist - current_dist) / self.initial_dist.clamp(min=0.01)).clamp(0, 1)

            # ══════════════════════════════════════════════════════════════
            # REDUCED: Distance reward (still useful, but less dominant)
            # ══════════════════════════════════════════════════════════════
            r_distance = torch.exp(-8.0 * dist)

            # ══════════════════════════════════════════════════════════════
            # KEPT: Reaching sigmoid (strong signal for getting close)
            # ══════════════════════════════════════════════════════════════
            reach_threshold = lv["pos_threshold"]
            r_reaching = torch.sigmoid((reach_threshold - dist) * 30.0)

            # ══════════════════════════════════════════════════════════════
            # REDUCED: Final push (less free reward for being stationary near target)
            # ══════════════════════════════════════════════════════════════
            r_final_push = torch.exp(-15.0 * dist) * torch.sigmoid((0.08 - dist) * 25.0)

            # ══════════════════════════════════════════════════════════════
            # REDUCED: Smooth (only active when CLOSE to target, not free reward)
            # ══════════════════════════════════════════════════════════════
            arm_diff = self.prev_arm_actions - self._prev_arm_actions
            r_smooth_raw = torch.exp(-1.0 * arm_diff.pow(2).sum(-1))
            close_mask = (dist < 0.08).float()  # Only reward smoothness when close
            r_smooth = r_smooth_raw * close_mask

            # ══════════════════════════════════════════════════════════════
            # NEW: STILLNESS PENALTY (penalize not moving when target is far)
            # ══════════════════════════════════════════════════════════════
            far_mask = (dist > 0.15).float()
            r_stillness = -torch.exp(-20.0 * ee_speed) * far_mask

            # Palm orientation reward — two-component design (Phase 2 only)
            if lv.get("use_orientation", False):
                orient_err = compute_orientation_error(palm_quat, self.target_orient_body)
                # Component 1: Continuous tracking (always on during Phase 2)
                r_orient_continuous = torch.exp(-2.0 * orient_err)
                # Component 2: Bonus when close to threshold
                orient_thresh = lv.get("orient_threshold", 2.0)
                r_orient_bonus = torch.sigmoid(10.0 * (orient_thresh - orient_err))
                # Combined: 4.0 * continuous + 3.0 * bonus = up to 7.0
                r_palm_orient = 4.0/7.0 * r_orient_continuous + 3.0/7.0 * r_orient_bonus
            else:
                orient_err = compute_orientation_error(palm_quat, self.target_orient_body)
                r_palm_orient = torch.zeros(self.num_envs, device=self.device)

            # Workspace violation
            x_violation = torch.clamp(0.05 - ee_body[:, 0], min=0) + torch.clamp(ee_body[:, 0] - 0.55, min=0)
            y_violation = torch.clamp(-0.55 - ee_body[:, 1], min=0) + torch.clamp(ee_body[:, 1] - 0.10, min=0)
            z_violation = torch.clamp(-0.25 - ee_body[:, 2], min=0) + torch.clamp(ee_body[:, 2] - 0.55, min=0)
            p_workspace = (x_violation + y_violation + z_violation) * 5.0

            # Action penalties
            arm_diff2 = self.prev_arm_actions - self._prev_arm_actions
            p_action_rate = arm_diff2.pow(2).sum(-1)
            arm_accel = self.prev_arm_actions - 2 * self._prev_arm_actions + self._prev_prev_arm_actions
            p_jerk = arm_accel.pow(2).sum(-1)

            # EE displacement for monitoring
            ee_displacement = torch.norm(ee_body - self.ee_pos_at_spawn, dim=-1)

            # Store reward components for monitoring
            self.arm_reward_components = {
                "velocity_toward": (ARM_REWARD_WEIGHTS["velocity_toward"] * r_velocity_toward).mean().item(),
                "progress": (ARM_REWARD_WEIGHTS["progress"] * r_progress).mean().item(),
                "distance": (ARM_REWARD_WEIGHTS["distance"] * r_distance).mean().item(),
                "reaching": (ARM_REWARD_WEIGHTS["reaching"] * r_reaching).mean().item(),
                "final_push": (ARM_REWARD_WEIGHTS["final_push"] * r_final_push).mean().item(),
                "smooth": (ARM_REWARD_WEIGHTS["smooth"] * r_smooth).mean().item(),
                "stillness_penalty": (ARM_REWARD_WEIGHTS["stillness_penalty"] * r_stillness).mean().item(),
                "palm_orient": (ARM_REWARD_WEIGHTS["palm_orient"] * r_palm_orient).mean().item(),
                "action_rate": (ARM_REWARD_WEIGHTS["action_rate"] * p_action_rate).mean().item(),
                "jerk": (ARM_REWARD_WEIGHTS["jerk"] * p_jerk).mean().item(),
                "workspace_violation": (ARM_REWARD_WEIGHTS["workspace_violation"] * p_workspace).mean().item(),
            }

            # Behavior metrics
            self.behavior_metrics.update({
                "ee_dist_raw": dist.mean().item(),
                "ee_dist_min": dist.min().item(),
                "orient_error_raw": orient_err.mean().item(),
                "ee_speed": ee_speed.mean().item(),
                "ee_displacement": ee_displacement.mean().item(),
                "velocity_toward": velocity_toward.mean().item(),
                "progress_fraction": r_progress.mean().item(),
            })

            arm_reward = (
                ARM_REWARD_WEIGHTS["velocity_toward"] * r_velocity_toward +
                ARM_REWARD_WEIGHTS["progress"] * r_progress +
                ARM_REWARD_WEIGHTS["distance"] * r_distance +
                ARM_REWARD_WEIGHTS["reaching"] * r_reaching +
                ARM_REWARD_WEIGHTS["final_push"] * r_final_push +
                ARM_REWARD_WEIGHTS["smooth"] * r_smooth +
                ARM_REWARD_WEIGHTS["stillness_penalty"] * r_stillness +
                ARM_REWARD_WEIGHTS["palm_orient"] * r_palm_orient +
                ARM_REWARD_WEIGHTS["action_rate"] * p_action_rate +
                ARM_REWARD_WEIGHTS["jerk"] * p_jerk +
                ARM_REWARD_WEIGHTS["workspace_violation"] * p_workspace +
                ARM_REWARD_WEIGHTS["alive"]
            )

            return arm_reward.clamp(-10, 30)

        def _get_rewards(self) -> torch.Tensor:
            self.loco_reward = self.compute_loco_reward()
            self.arm_reward = self.compute_arm_reward()
            combined = self.loco_reward + self.arm_reward

            lv = CURRICULUM[self.curr_level]
            if self.curr_level < 6:
                phase_name = "P1-Position"
            else:
                phase_name = "P2-Orient"

            robot = self.robot
            quat = robot.data.root_quat_w
            pos = robot.data.root_pos_w
            lin_vel_b = quat_apply_inverse(quat, robot.data.root_lin_vel_w)
            ee_pos, _ = self._compute_palm_ee()
            target_world = pos + quat_apply(quat, self.target_pos_body)
            dist = torch.norm(ee_pos - target_world, dim=-1)

            # Validated reach rate
            if self.total_attempts > 0:
                validated_rate = self.validated_reaches / self.total_attempts
                timeout_rate = self.timed_out_targets / self.total_attempts
            else:
                validated_rate = 0.0
                timeout_rate = 0.0

            self.extras = {
                "R/loco_total": self.loco_reward.mean().item(),
                "R/arm_total": self.arm_reward.mean().item(),
                "M/height": pos[:, 2].mean().item(),
                "M/vx": lin_vel_b[:, 0].mean().item(),
                "M/ee_dist": dist.mean().item(),
                "M/reaches": self.total_reaches,
                "curriculum_level": self.curr_level,
                "phase": phase_name,

                # ARM REWARD COMPONENTS
                "RC/arm_velocity_toward": self.arm_reward_components.get("velocity_toward", 0),
                "RC/arm_progress": self.arm_reward_components.get("progress", 0),
                "RC/arm_distance": self.arm_reward_components.get("distance", 0),
                "RC/arm_reaching": self.arm_reward_components.get("reaching", 0),
                "RC/arm_final_push": self.arm_reward_components.get("final_push", 0),
                "RC/arm_smooth": self.arm_reward_components.get("smooth", 0),
                "RC/arm_stillness_penalty": self.arm_reward_components.get("stillness_penalty", 0),
                "RC/arm_palm_orient": self.arm_reward_components.get("palm_orient", 0),

                # BEHAVIOR METRICS
                "BH/ee_dist_raw": self.behavior_metrics.get("ee_dist_raw", 0),
                "BH/ee_speed": self.behavior_metrics.get("ee_speed", 0),
                "BH/ee_displacement": self.behavior_metrics.get("ee_displacement", 0),
                "BH/velocity_toward": self.behavior_metrics.get("velocity_toward", 0),
                "BH/progress_fraction": self.behavior_metrics.get("progress_fraction", 0),

                # ANTI-GAMING METRICS
                "AG/validated_reaches": self.validated_reaches,
                "AG/timed_out_targets": self.timed_out_targets,
                "AG/validated_rate": validated_rate,
                "AG/timeout_rate": timeout_rate,
                "AG/stage_validated": self.stage_validated_reaches,
                "AG/stage_timed_out": self.stage_timed_out,

                # CURRICULUM INFO
                "Curr/pos_threshold": lv["pos_threshold"],
                "Curr/min_target_distance": lv["min_target_distance"],
                "Curr/min_displacement": lv["min_displacement"],
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
            """ANTI-GAMING: Curriculum advancement based on validated reach rate.
            Returns new_level if advanced, None otherwise."""
            lv = CURRICULUM[self.curr_level]
            if lv["min_validated_reaches"] is None:
                return None

            min_steps = lv.get("min_steps", 0)
            if self.stage_steps < min_steps:
                return None

            stage_attempts = self.stage_validated_reaches + self.stage_timed_out
            if stage_attempts == 0:
                return None

            # ANTI-GAMING: Gaming detection
            if stage_attempts > 100:
                timeout_ratio = self.stage_timed_out / stage_attempts
                if timeout_ratio > 0.90:
                    print(f"\n{'!'*60}")
                    print(f"  GAMING DETECTED at Level {self.curr_level}!")
                    print(f"  Timeout ratio: {timeout_ratio:.1%} (>{90}%)")
                    print(f"  Validated: {self.stage_validated_reaches}, Timed out: {self.stage_timed_out}")
                    print(f"  NOT advancing. Robot needs to actually reach targets!")
                    print(f"{'!'*60}\n")
                    return None

            if self.stage_validated_reaches < lv["min_validated_reaches"]:
                return None

            validated_rate = self.stage_validated_reaches / stage_attempts
            if validated_rate >= lv["validated_reach_rate"]:
                if self.curr_level < len(CURRICULUM) - 1:
                    self.curr_level += 1
                    new_lv = CURRICULUM[self.curr_level]

                    phase_msg = ""
                    if self.curr_level == 4:
                        phase_msg = " -- WALKING STARTS!"
                    elif self.curr_level == 6:
                        phase_msg = " -- PHASE 2: ORIENTATION LEARNING!"

                    print(f"\n{'='*60}")
                    print(f"LEVEL UP! Level {self.curr_level}{phase_msg}")
                    print(f"   pos_thresh={new_lv['pos_threshold']}, min_dist={new_lv['min_target_distance']}")
                    print(f"   min_disp={new_lv['min_displacement']}, max_steps={new_lv['max_reach_steps']}")
                    if new_lv.get("use_orientation", False):
                        print(f"   orient_thresh={new_lv.get('orient_threshold', 'N/A')}")
                        print(f"   variable_orient={new_lv.get('variable_orientation', False)}")
                    print(f"   description: {new_lv.get('description', '')}")
                    print(f"   Validated reaches: {self.stage_validated_reaches}")
                    print(f"   Validated rate: {validated_rate:.1%}")
                    print(f"   Timed out: {self.stage_timed_out}")
                    print(f"{'='*60}\n")

                    self.stage_validated_reaches = 0
                    self.stage_timed_out = 0
                    self.stage_steps = 0

                    return self.curr_level

            return None

    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    return Stage8OrientEnv(cfg)


# ============================================================================
# CHECKPOINT LOADING (Stage 7 → Stage 8)
# ============================================================================

def load_stage7_and_setup(net, stage7_path, device):
    """Load Stage 7 checkpoint — fine-tune arm (not fresh start)"""
    print(f"\n[Transfer] Loading Stage 7: {stage7_path}")
    ckpt = torch.load(stage7_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Load ALL weights (exact match — same architecture)
    net.load_state_dict(state_dict, strict=True)

    # Freeze loco
    loco_frozen = 0
    for name, p in net.named_parameters():
        if name.startswith("loco_actor.") or name.startswith("loco_critic."):
            p.requires_grad = False
            loco_frozen += 1

    # Arm: KEEP weights, lower exploration for fine-tuning
    net.arm_actor.log_std.data.fill_(np.log(0.3))  # Lower than Stage 7's log(0.8)

    arm_trainable = sum(1 for n, p in net.named_parameters() if p.requires_grad)

    print(f"[Stage 8] Loaded Stage 7 checkpoint:")
    print(f"  Loco frozen: {loco_frozen} params")
    print(f"  Arm trainable: {arm_trainable} params (FINE-TUNED from Stage 7)")
    print(f"  Arm log_std: {net.arm_actor.log_std.data.exp().mean():.4f}")

    # Load curriculum state if available
    ckpt_level = ckpt.get("curriculum_level", 0)
    print(f"  Stage 7 was at curriculum Level {ckpt_level}")

    # Print checkpoint info
    for key in ["best_reward", "iteration", "total_reaches", "validated_reaches"]:
        if key in ckpt:
            print(f"  Stage 7 {key}: {ckpt[key]}")

    # Verify freeze
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"\n[Architecture] Trainable: {trainable:,} | Frozen: {frozen:,}")

    return ckpt


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    device = "cuda:0"

    print(f"\n[INFO] Creating environment with {args_cli.num_envs} envs...")
    env = create_env(args_cli.num_envs, device)

    print(f"[INFO] Creating Stage 8 Dual Actor-Critic...")
    print("       LocoActor (57->12) FROZEN    + LocoCritic (57->1) FROZEN")
    print("       ArmActor  (55->5)  FINE-TUNE + ArmCritic  (55->1) FINE-TUNE")
    net = DualActorCritic(loco_obs=57, arm_obs=55, loco_act=12, arm_act=5).to(device)

    start_iter = 0
    best_reward_resume = float('-inf')

    if args_cli.checkpoint:
        # Resume Stage 8 training
        print(f"\n[INFO] Resuming Stage 8 from: {args_cli.checkpoint}")
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iteration", 0)
        best_reward_resume = ckpt.get("best_reward", float('-inf'))
        env.curr_level = ckpt.get("curriculum_level", 0)
        env.total_reaches = ckpt.get("total_reaches", 0)
        env.validated_reaches = ckpt.get("validated_reaches", 0)
        env.timed_out_targets = ckpt.get("timed_out_targets", 0)
        env.total_attempts = ckpt.get("total_attempts", 0)
        print(f"   Resumed at iter={start_iter}, level={env.curr_level}, "
              f"VR={env.validated_reaches}, TO={env.timed_out_targets}, best={best_reward_resume:.2f}")

        # Re-freeze loco after loading
        for name, p in net.named_parameters():
            if name.startswith("loco_actor.") or name.startswith("loco_critic."):
                p.requires_grad = False
    else:
        # Fresh Stage 8 from Stage 7 checkpoint
        if args_cli.stage7_checkpoint is None:
            raise ValueError("--stage7_checkpoint is required for fresh Stage 8 start (not resuming)")
        load_stage7_and_setup(net, args_cli.stage7_checkpoint, device)

    # Phase 2 tracking — needed to prevent exploration/LR overrides
    phase2_started = env.curr_level >= 6  # True if resuming from Phase 2
    phase2_start_iter = start_iter if phase2_started else None

    if phase2_started:
        ppo = DualPPO(net, device, arm_lr=2e-4)  # Phase 2 LR
        remaining_iters = args_cli.max_iterations - start_iter
        ppo.arm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            ppo.arm_opt, max(remaining_iters, 1), eta_min=1e-5
        )
        print(f"[Phase 2 Resume] LR=2e-4, cosine -> 1e-5 over {remaining_iters} iters")
    else:
        ppo = DualPPO(net, device, arm_lr=1e-4)  # Phase 1 LR

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"\n[INFO] Logging to: {log_dir}")

    best_reward = best_reward_resume if args_cli.checkpoint else float('-inf')

    obs, _ = env.reset()

    print("\n" + "=" * 80)
    print("STARTING STAGE 8: POSITION PERFECTION + ORIENTATION TRAINING")
    print("  Phase 1 (Level 0-5): Position Perfection (4cm threshold)")
    print("  Phase 2 (Level 6-10): Palm Orientation (progressive cone)")
    print("  LOCO: FROZEN | ARM: FINE-TUNED from Stage 7 (55 obs)")
    print("=" * 80 + "\n")

    for iteration in range(start_iter, args_cli.max_iterations):
        arm_obs_buf = []
        arm_act_buf = []
        arm_logp_buf = []
        arm_val_buf = []
        arm_rew_buf = []
        done_buf = []

        rollout_steps = 24

        for _ in range(rollout_steps):
            loco_obs = env.get_loco_obs()
            arm_obs = env.get_arm_obs()

            with torch.no_grad():
                loco_action, arm_action, loco_logp, arm_logp = net.get_actions(loco_obs, arm_obs)
                arm_value = net.arm_critic(arm_obs)

            arm_obs_buf.append(arm_obs)
            arm_act_buf.append(arm_action)
            arm_logp_buf.append(arm_logp)
            arm_val_buf.append(arm_value)

            # Combine actions: 12 leg + 5 arm = 17 total
            actions = torch.cat([loco_action, arm_action], dim=-1)
            obs_dict, combined_reward, terminated, truncated, _ = env.step(actions)

            arm_rew_buf.append(env.arm_reward)
            done_buf.append((terminated | truncated).float())

        arm_obs_buf = torch.stack(arm_obs_buf)
        arm_act_buf = torch.stack(arm_act_buf)
        arm_logp_buf = torch.stack(arm_logp_buf)
        arm_val_buf = torch.stack(arm_val_buf)
        arm_rew_buf = torch.stack(arm_rew_buf)
        done_buf = torch.stack(done_buf)

        with torch.no_grad():
            final_arm_obs = env.get_arm_obs()
            arm_next_val = net.arm_critic(final_arm_obs)

        arm_adv, arm_ret = ppo.gae(arm_rew_buf, arm_val_buf, done_buf, arm_next_val)

        # ONLY update ARM (loco is frozen)
        arm_loss = ppo.update_arm(
            arm_obs_buf.view(-1, 55),
            arm_act_buf.view(-1, 5),
            arm_logp_buf.view(-1),
            arm_ret.view(-1),
            arm_adv.view(-1),
            arm_val_buf.view(-1),
        )

        # Arm exploration schedule — phase-aware
        if phase2_started and phase2_start_iter is not None:
            # Phase 2: 0.5 -> 0.15 (from phase2 start to end)
            phase2_iters = args_cli.max_iterations - phase2_start_iter
            phase2_progress = (iteration - phase2_start_iter) / max(phase2_iters, 1)
            arm_std = 0.5 + (0.15 - 0.5) * min(phase2_progress, 1.0)
        else:
            # Phase 1: 0.3 -> 0.15 (standard fine-tune schedule)
            progress = iteration / args_cli.max_iterations
            arm_std = 0.3 + (0.15 - 0.3) * progress
        net.arm_actor.log_std.data.fill_(np.log(max(arm_std, 0.15)))

        mean_arm_reward = arm_rew_buf.mean().item()
        mean_loco_reward = env.loco_reward.mean().item()
        mean_reward = mean_loco_reward + mean_arm_reward

        new_level = env.update_curriculum(mean_reward)

        # Phase 2 transition: orientation learning activation
        if new_level == 6 and not phase2_started:
            phase2_started = True
            phase2_start_iter = iteration
            print(f"\n{'='*70}")
            print(f"[Stage 8] PHASE 2 ACTIVATED: Orientation Learning!")
            print(f"{'='*70}")
            # Bump LR for new skill + recreate scheduler
            for pg in ppo.arm_opt.param_groups:
                pg['lr'] = 2e-4
            remaining_iters = args_cli.max_iterations - iteration
            ppo.arm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                ppo.arm_opt, remaining_iters, eta_min=1e-5
            )
            # Exploration will be handled by phase-aware schedule above
            # (starts at 0.5, decays to 0.15)
            # Reinit arm critic (value estimates wrong for new reward structure)
            for layer in net.arm_critic.net:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            print(f"  LR bumped to 2e-4 with new cosine schedule -> 1e-5")
            print(f"  Exploration: 0.5 -> 0.15 (phase-aware decay)")
            print(f"  Arm critic reinitialized")
            print(f"{'='*70}\n")

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "total_reaches": env.total_reaches,
                "validated_reaches": env.validated_reaches,
                "timed_out_targets": env.timed_out_targets,
                "total_attempts": env.total_attempts,
            }, f"{log_dir}/model_best.pt")

        writer.add_scalar("Train/reward", mean_reward, iteration)
        writer.add_scalar("Train/arm_reward", mean_arm_reward, iteration)
        writer.add_scalar("Train/loco_reward", mean_loco_reward, iteration)
        writer.add_scalar("Train/best_reward", best_reward, iteration)
        writer.add_scalar("Loss/arm", arm_loss, iteration)
        writer.add_scalar("Curriculum/level", env.curr_level, iteration)
        writer.add_scalar("Curriculum/validated_reaches", env.validated_reaches, iteration)
        writer.add_scalar("Curriculum/timed_out_targets", env.timed_out_targets, iteration)

        for key, val in env.extras.items():
            if isinstance(val, (int, float)):
                writer.add_scalar(f"Env/{key}", val, iteration)

        if iteration % 10 == 0:
            phase = env.extras.get("phase", "P1")
            v_rate = env.extras.get("AG/validated_rate", 0)
            print(
                f"#{iteration:5d} | "
                f"R={mean_reward:5.1f} (L={mean_loco_reward:5.1f} A={mean_arm_reward:5.1f}) | "
                f"Best={best_reward:5.1f} | "
                f"Lv={env.curr_level:2d} ({phase}) | "
                f"VR={env.validated_reaches} TO={env.timed_out_targets} "
                f"Rate={v_rate:.1%} | "
                f"EE={env.extras.get('M/ee_dist', 0):.3f} "
                f"Spd={env.extras.get('BH/ee_speed', 0):.3f}"
            )

        if (iteration + 1) % 500 == 0:
            torch.save({
                "model": net.state_dict(),
                "iteration": iteration,
                "best_reward": best_reward,
                "curriculum_level": env.curr_level,
                "total_reaches": env.total_reaches,
                "validated_reaches": env.validated_reaches,
                "timed_out_targets": env.timed_out_targets,
                "total_attempts": env.total_attempts,
            }, f"{log_dir}/model_{iteration + 1}.pt")

        writer.flush()

    torch.save({
        "model": net.state_dict(),
        "iteration": args_cli.max_iterations,
        "best_reward": best_reward,
        "curriculum_level": env.curr_level,
        "total_reaches": env.total_reaches,
        "validated_reaches": env.validated_reaches,
        "timed_out_targets": env.timed_out_targets,
        "total_attempts": env.total_attempts,
    }, f"{log_dir}/model_final.pt")

    writer.close()
    env.close()

    print("\n" + "=" * 80)
    print("STAGE 8 TRAINING COMPLETE!")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Final Level: {env.curr_level}")
    print(f"  Validated Reaches: {env.validated_reaches}")
    print(f"  Timed Out Targets: {env.timed_out_targets}")
    if env.total_attempts > 0:
        print(f"  Validated Rate: {env.validated_reaches / env.total_attempts:.1%}")
    print(f"  Log Dir: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
    simulation_app.close()
