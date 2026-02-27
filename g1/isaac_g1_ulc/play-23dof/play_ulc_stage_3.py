#!/usr/bin/env python3
"""
ULC G1 Stage 3 Play Script
==========================
Test trained torso control policy.

Usage:
    ./isaaclab.bat -p train_ulc_stage3_play.py \
        --checkpoint logs/ulc/ulc_g1_stage3_.../model_best.pt \
        --num_envs 4 \
        --pitch -0.3
"""

import torch
import torch.nn as nn
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 3 Play")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vx", type=float, default=0.3, help="Forward velocity command")
    parser.add_argument("--pitch", type=float, default=0.0, help="Torso pitch command (rad)")
    parser.add_argument("--roll", type=float, default=0.0, help="Torso roll command (rad)")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


args_cli = parse_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion tensor [N, 4] in (x, y, z, w) format

    Returns:
        euler: Euler angles [N, 3] in (roll, pitch, yaw) format
    """
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


print("=" * 60)
print("ULC G1 STAGE 3 - PLAY (Torso Control)")
print("=" * 60)
print(f"Commands: vx={args_cli.vx}, pitch={args_cli.pitch}, roll={args_cli.roll}")

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"


class ActorCritic(nn.Module):
    """Same architecture as training"""

    def __init__(self, num_obs=57, num_act=12, hidden=[512, 256, 128]):
        super().__init__()

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def act(self, x, deterministic=True):
        mean = self.actor(x)
        if deterministic:
            return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()


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
            usd_path=G1_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=10.0,
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
                stiffness=150.0, damping=15.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*shoulder.*", ".*elbow.*"],
                stiffness=50.0, damping=5.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint"],
                stiffness=100.0, damping=10.0,
            ),
        },
    )


@configclass
class PlayEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_space = 12
    observation_space = 57  # Stage 3
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=4, env_spacing=2.5)


class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_names = self.robot.joint_names
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
        self.default_leg = torch.tensor(
            [-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0],
            device=self.device
        )

        # Commands from CLI
        self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_cmd[:, 0] = args_cli.vx

        self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_cmd[:, 0] = args_cli.roll  # roll
        self.torso_cmd[:, 1] = args_cli.pitch  # pitch

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 12, device=self.device)

        print(f"[Env] Initialized with vx={args_cli.vx}, pitch={args_cli.pitch}, roll={args_cli.roll}")

    @property
    def robot(self):
        return self.scene["robot"]

    def get_torso_euler(self):
        quat = self.robot.data.root_quat_w
        return quat_to_euler_xyz(quat)

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        target_pos = self.robot.data.default_joint_pos.clone()
        target_pos[:, self.leg_idx] = self.default_leg + actions * 0.4
        self.robot.set_joint_position_target(target_pos)
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0
        self.prev_actions = actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
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

        torso_euler = self.get_torso_euler()

        obs = torch.cat([
            lin_vel_b,  # 3
            ang_vel_b,  # 3
            proj_gravity,  # 3
            joint_pos,  # 12
            joint_vel,  # 12
            self.height_cmd.unsqueeze(-1),  # 1
            self.vel_cmd,  # 3
            gait_phase,  # 2
            self.prev_actions,  # 12
            self.torso_cmd,  # 3
            torso_euler,  # 3
        ], dim=-1)

        return {"policy": obs.clamp(-10, 10).nan_to_num()}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

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

        default_pos = torch.tensor([[0.0, 0.0, 0.8]], device=self.device).expand(len(env_ids), -1).clone()
        default_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(len(env_ids), -1)

        self.robot.write_root_pose_to_sim(torch.cat([default_pos, default_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)

        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        self.prev_actions[env_ids] = 0


def main():
    device = "cuda:0"

    print(f"\n[INFO] Loading: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)

    if "best_reward" in checkpoint:
        print(f"[INFO] Best reward: {checkpoint['best_reward']:.2f}")
    if "iteration" in checkpoint:
        print(f"[INFO] Iteration: {checkpoint['iteration']}")
    if "curriculum_level" in checkpoint:
        print(f"[INFO] Curriculum level: {checkpoint['curriculum_level']}")

    cfg = PlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = PlayEnv(cfg)

    net = ActorCritic(57, 12).to(device)
    net.load_state_dict(checkpoint["actor_critic"])
    net.eval()
    print("[INFO] Model loaded successfully!")

    obs, _ = env.reset()
    obs = obs["policy"]

    print(f"\n[Play] vx={args_cli.vx}, pitch={args_cli.pitch:.2f}, roll={args_cli.roll:.2f}")
    print(f"       Press Ctrl+C to stop")
    print("-" * 60)

    step = 0
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = net.act(obs, deterministic=True)

            obs_dict, _, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]
            step += 1

            if step % 100 == 0:
                height = env.robot.data.root_pos_w[:, 2].mean().item()
                vx = env.robot.data.root_lin_vel_w[:, 0].mean().item()
                euler = env.get_torso_euler()
                pitch = euler[:, 1].mean().item()
                roll = euler[:, 0].mean().item()

                print(
                    f"Step {step:5d} | "
                    f"H={height:.3f}m | "
                    f"Vx={vx:.3f}m/s | "
                    f"Pitch={np.rad2deg(pitch):.1f}° (cmd={np.rad2deg(args_cli.pitch):.1f}°) | "
                    f"Roll={np.rad2deg(roll):.1f}° (cmd={np.rad2deg(args_cli.roll):.1f}°)"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()