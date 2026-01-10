"""
ULC G1 Stage 4: Play Script - Arm Control Testing
==================================================
Test trained Stage 4 model with arm commands
"""

import torch
import torch.nn as nn
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 4 Play")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    # Arm commands (per arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll)
    parser.add_argument("--left_shoulder_pitch", type=float, default=0.0)
    parser.add_argument("--left_elbow", type=float, default=-0.2)
    parser.add_argument("--right_shoulder_pitch", type=float, default=0.0)
    parser.add_argument("--right_elbow", type=float, default=-0.2)
    # Preset arm poses
    parser.add_argument("--arms_up", action="store_true", help="Both arms raised")
    parser.add_argument("--arms_front", action="store_true", help="Arms extended forward")
    parser.add_argument("--wave", action="store_true", help="Right arm wave position")
    return parser.parse_args()

args_cli = parse_args()

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

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = torch.clamp(2*(w*y - z*x), -1, 1)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return torch.stack([roll, pitch, yaw], dim=-1)

# Preset arm poses
if args_cli.arms_up:
    args_cli.left_shoulder_pitch = 1.5   # Arms up
    args_cli.right_shoulder_pitch = 1.5
    args_cli.left_elbow = 0.0
    args_cli.right_elbow = 0.0
elif args_cli.arms_front:
    args_cli.left_shoulder_pitch = 1.0   # Arms forward
    args_cli.right_shoulder_pitch = 1.0
    args_cli.left_elbow = -0.2
    args_cli.right_elbow = -0.2
elif args_cli.wave:
    args_cli.right_shoulder_pitch = 1.2  # Wave pose
    args_cli.right_elbow = 0.8

print("=" * 60)
print("ULC G1 STAGE 4 - PLAY (Arm Control)")
print("=" * 60)
print(f"Commands: vx={args_cli.vx}, pitch={args_cli.pitch}, roll={args_cli.roll}")
print(f"Left arm: shoulder_pitch={args_cli.left_shoulder_pitch}, elbow={args_cli.left_elbow}")
print(f"Right arm: shoulder_pitch={args_cli.right_shoulder_pitch}, elbow={args_cli.right_elbow}")

G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

class ActorCritic(nn.Module):
    def __init__(self, num_obs=77, num_act=22, hidden=[512, 256, 128]):
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
        return self.actor(x) if deterministic else self.actor(x) + torch.randn_like(self.actor(x)) * self.log_std.exp()

# Load checkpoint
print(f"\n[INFO] Loading: {args_cli.checkpoint}")
checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0", weights_only=False)
print(f"[INFO] Best reward: {checkpoint.get('best_reward', 'N/A')}")
print(f"[INFO] Iteration: {checkpoint.get('iteration', 'N/A')}")
print(f"[INFO] Curriculum level: {checkpoint.get('curriculum_level', 'N/A')}")

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
DEFAULT_ARM_POSE = [0.0, 0.0, 0.0, -0.2, 0.0]
RESIDUAL_SCALES = [0.5, 0.3, 0.3, 0.4, 0.3]

def create_play_env(num_envs, device):
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
                    "left_elbow_pitch_joint": -0.2, "right_elbow_pitch_joint": -0.2,
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
                    stiffness=80.0, damping=8.0,
                ),
                "torso": ImplicitActuatorCfg(
                    joint_names_expr=["torso_joint"],
                    stiffness=100.0, damping=10.0,
                ),
            },
        )

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 30.0
        action_space = 22
        observation_space = 77
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class PlayEnv(DirectRLEnv):
        cfg: EnvCfg

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
            self.leg_idx = torch.tensor([joint_names.index(n) for n in leg_names if n in joint_names], device=self.device)

            left_arm_names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_elbow_roll_joint"]
            right_arm_names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_elbow_roll_joint"]

            self.left_arm_idx = torch.tensor([joint_names.index(n) for n in left_arm_names if n in joint_names], device=self.device)
            self.right_arm_idx = torch.tensor([joint_names.index(n) for n in right_arm_names if n in joint_names], device=self.device)
            self.arm_idx = torch.cat([self.left_arm_idx, self.right_arm_idx])

            self.default_leg = torch.tensor([-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0], device=self.device)
            self.default_arm = torch.tensor(DEFAULT_ARM_POSE * 2, device=self.device)
            self.residual_scales = torch.tensor(RESIDUAL_SCALES * 2, device=self.device)

            # Fixed commands from CLI
            self.height_cmd = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.vel_cmd[:, 0] = args_cli.vx

            self.torso_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_cmd[:, 0] = args_cli.roll
            self.torso_cmd[:, 1] = args_cli.pitch

            # Arm commands from CLI
            self.left_arm_cmd = torch.zeros(self.num_envs, 5, device=self.device)
            self.left_arm_cmd[:, 0] = args_cli.left_shoulder_pitch
            self.left_arm_cmd[:, 3] = args_cli.left_elbow

            self.right_arm_cmd = torch.zeros(self.num_envs, 5, device=self.device)
            self.right_arm_cmd[:, 0] = args_cli.right_shoulder_pitch
            self.right_arm_cmd[:, 3] = args_cli.right_elbow

            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_actions = torch.zeros(self.num_envs, 22, device=self.device)

            print(f"[Env] Initialized with commands")

        @property
        def robot(self):
            return self.scene["robot"]

        def _pre_physics_step(self, actions):
            self.actions = actions.clone()

            leg_actions = actions[:, :12]
            arm_actions = actions[:, 12:]

            target_pos = self.robot.data.default_joint_pos.clone()
            target_pos[:, self.leg_idx] = self.default_leg + leg_actions * 0.4

            arm_cmd = torch.cat([self.left_arm_cmd, self.right_arm_cmd], dim=-1)
            arm_residual = arm_actions * self.residual_scales
            arm_target = arm_cmd + torch.tanh(arm_residual) * self.residual_scales
            target_pos[:, self.arm_idx] = arm_target

            self.robot.set_joint_position_target(target_pos)
            self.phase = (self.phase + GAIT_FREQUENCY * self.cfg.sim.dt * self.cfg.decimation) % 1.0
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

            leg_pos = robot.data.joint_pos[:, self.leg_idx]
            leg_vel = robot.data.joint_vel[:, self.leg_idx]
            left_arm_pos = robot.data.joint_pos[:, self.left_arm_idx]
            right_arm_pos = robot.data.joint_pos[:, self.right_arm_idx]

            gait_phase = torch.stack([torch.sin(2*np.pi*self.phase), torch.cos(2*np.pi*self.phase)], dim=-1)
            torso_euler = quat_to_euler_xyz(quat)

            obs = torch.cat([
                lin_vel_b, ang_vel_b, proj_gravity,
                leg_pos, leg_vel,
                self.height_cmd.unsqueeze(-1), self.vel_cmd, gait_phase,
                self.prev_actions[:, :12],
                self.torso_cmd, torso_euler,
                left_arm_pos, right_arm_pos,
                self.left_arm_cmd, self.right_arm_cmd,
            ], dim=-1)

            return {"policy": obs.clamp(-10, 10).nan_to_num()}

        def _get_rewards(self) -> torch.Tensor:
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self) -> tuple:
            height = self.robot.data.root_pos_w[:, 2]
            terminated = (height < 0.3) | (height > 1.2)
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
            self.phase[env_ids] = 0
            self.prev_actions[env_ids] = 0

    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    return PlayEnv(cfg)

# Create environment
env = create_play_env(args_cli.num_envs, "cuda:0")

# Load model
net = ActorCritic(77, 22).to("cuda:0")
net.load_state_dict(checkpoint["actor_critic"])
net.eval()
print("[INFO] Model loaded successfully!")

# Run
obs, _ = env.reset()
obs = obs["policy"]

print(f"\n[Play] vx={args_cli.vx}, pitch={args_cli.pitch:.2f}, roll={args_cli.roll:.2f}")
print(f"       Left arm: sp={args_cli.left_shoulder_pitch:.2f}, elbow={args_cli.left_elbow:.2f}")
print(f"       Right arm: sp={args_cli.right_shoulder_pitch:.2f}, elbow={args_cli.right_elbow:.2f}")
print("       Press Ctrl+C to stop")
print("-" * 60)

step = 0
try:
    while simulation_app.is_running():
        with torch.no_grad():
            action = net.act(obs, deterministic=True)

        obs_dict, _, terminated, truncated, _ = env.step(action)
        obs = obs_dict["policy"]
        step += 1

        if step % 100 == 0:
            robot = env.robot
            height = robot.data.root_pos_w[:, 2].mean().item()
            lin_vel = quat_apply_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w)
            vx = lin_vel[:, 0].mean().item()

            euler = quat_to_euler_xyz(robot.data.root_quat_w)
            pitch_deg = np.degrees(euler[:, 1].mean().item())
            roll_deg = np.degrees(euler[:, 0].mean().item())

            left_arm = robot.data.joint_pos[:, env.left_arm_idx]
            right_arm = robot.data.joint_pos[:, env.right_arm_idx]

            print(
                f"Step {step:5d} | "
                f"H={height:.3f}m | "
                f"Vx={vx:.2f}m/s | "
                f"Pitch={pitch_deg:.1f}° | "
                f"Roll={roll_deg:.1f}° | "
                f"L_sp={left_arm[:, 0].mean().item():.2f} | "
                f"R_sp={right_arm[:, 0].mean().item():.2f}"
            )

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")

env.close()
simulation_app.close()