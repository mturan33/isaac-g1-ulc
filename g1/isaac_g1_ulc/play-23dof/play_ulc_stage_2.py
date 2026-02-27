#!/usr/bin/env python3
"""
ULC G1 Stage 2 v2 Play Script
Matches training: obs=51, hidden=[512,256,128], Linear+LayerNorm+ELU
"""

import torch
import torch.nn as nn
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ULC G1 Stage 2 Play")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--vx", type=float, default=0.5)
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

print("=" * 60)
print("ULC G1 STAGE 2 v2 - PLAY")
print("=" * 60)

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5
G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"


class ActorCritic(nn.Module):
    """Exact match with training: Linear + LayerNorm + ELU"""

    def __init__(self, num_obs, num_act, hidden=[512, 256, 128]):
        super().__init__()

        # Actor: Linear -> LayerNorm -> ELU pattern
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, num_act))
        self.actor = nn.Sequential(*layers)

        # Critic: same pattern
        layers = []
        prev = num_obs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def act(self, x, det=True):
        mean = self.actor(x)
        if det:
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
            joint_vel={".*": 0.0},
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
    observation_space = 51
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=4)
    scene = PlaySceneCfg(num_envs=4, env_spacing=2.5)


class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        jn = self.robot.joint_names
        leg_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]

        self.leg_idx = torch.tensor([jn.index(n) for n in leg_names if n in jn], device=self.device)
        self.default_leg = torch.tensor([-0.2, -0.2, 0, 0, 0, 0, 0.4, 0.4, -0.2, -0.2, 0, 0], device=self.device)

        self.heights = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_cmd[:, 0] = args_cli.vx

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.prev_act = torch.zeros(self.num_envs, 12, device=self.device)

        print(f"[Env] Leg indices: {self.leg_idx.tolist()}")
        print(f"[Env] Velocity: vx={args_cli.vx:.2f} m/s")

    @property
    def robot(self):
        return self.scene["robot"]

    def _pre_physics_step(self, act):
        self.actions = act.clone()
        tgt = self.robot.data.default_joint_pos.clone()
        tgt[:, self.leg_idx] = self.default_leg + act * 0.4
        self.robot.set_joint_position_target(tgt)
        self.prev_act = act.clone()
        self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        from isaaclab.utils.math import quat_apply_inverse as qai

        r = self.robot
        q = r.data.root_quat_w
        lv = qai(q, r.data.root_lin_vel_w)
        av = qai(q, r.data.root_ang_vel_w)
        g = qai(q, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))

        jp = r.data.joint_pos[:, self.leg_idx]
        jv = r.data.joint_vel[:, self.leg_idx]

        gait = torch.stack([
            torch.sin(2 * np.pi * self.phase),
            torch.cos(2 * np.pi * self.phase)
        ], -1)

        # obs=51: lv(3) + av(3) + g(3) + jp(12) + jv(12) + h(1) + cmd(3) + gait(2) + prev(12)
        obs = torch.cat([lv, av, g, jp, jv, self.heights[:, None], self.vel_cmd, gait, self.prev_act], -1)

        return {"policy": obs.clamp(-10, 10).nan_to_num()}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse as qai
        h = self.robot.data.root_pos_w[:, 2]
        g = qai(self.robot.data.root_quat_w, torch.tensor([0, 0, -1.], device=self.device).expand(self.num_envs, -1))
        term = (h < 0.35) | (h > 1.1) | (g[:, :2].abs().max(-1)[0] > 0.6)
        return term, self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if len(ids) == 0:
            return
        pos = torch.tensor([[0, 0, 0.8]], device=self.device).expand(len(ids), -1).clone()
        quat = torch.tensor([[0, 0, 0, 1.]], device=self.device).expand(len(ids), -1)
        self.robot.write_root_pose_to_sim(torch.cat([pos, quat], -1), ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), ids)
        dp = self.robot.data.default_joint_pos[ids]
        self.robot.write_joint_state_to_sim(dp, torch.zeros_like(dp), None, ids)
        self.phase[ids] = torch.rand(len(ids), device=self.device)
        self.prev_act[ids] = 0


def main():
    device = "cuda:0"

    print(f"[INFO] Loading: {args_cli.checkpoint}")
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

    num_obs = 51
    num_act = 12
    print(f"[INFO] Obs: {num_obs}, Actions: {num_act}")

    net = ActorCritic(num_obs, num_act).to(device)
    net.load_state_dict(checkpoint["actor_critic"])
    net.eval()
    print("[INFO] Model loaded successfully!")

    obs, _ = env.reset()
    obs = obs["policy"]

    print(f"\n[Play] vx={args_cli.vx:.2f} m/s | Ctrl+C to stop")
    print("-" * 60)

    step = 0
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = net.act(obs, det=True)

            obs_d, _, _, _, _ = env.step(actions)
            obs = obs_d["policy"]
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