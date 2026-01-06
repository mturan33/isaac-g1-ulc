"""
ULC G1 Stage 2 v2: Improved Locomotion with Adaptive Curriculum
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from datetime import datetime

CURRICULUM_LEVELS = [
    {"vx": (0.0, 0.3), "vy": (-0.1, 0.1), "vyaw": (-0.3, 0.3), "threshold": 14.0},
    {"vx": (0.0, 0.6), "vy": (-0.2, 0.2), "vyaw": (-0.5, 0.5), "threshold": 16.0},
    {"vx": (-0.3, 1.0), "vy": (-0.3, 0.3), "vyaw": (-0.8, 0.8), "threshold": 18.0},
    {"vx": (-0.5, 1.5), "vy": (-0.5, 0.5), "vyaw": (-1.0, 1.0), "threshold": None},
]

HEIGHT_DEFAULT = 0.72
GAIT_FREQUENCY = 1.5

REWARD_WEIGHTS = {
    "vx": 4.0, "vy": 2.0, "vyaw": 2.0,
    "gait": 3.0, "clearance": 2.0, "symmetry": 2.0,
    "height": 3.0, "orientation": 4.0, "stability": 2.0,
    "smooth": -0.01, "accel": -0.001, "energy": -0.0005,
    "alive": 0.5,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=6000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="ulc_g1_stage2_v2")
    parser.add_argument("--headless", action="store_true")
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
from torch.utils.tensorboard import SummaryWriter

G1_USD = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/G1/g1.usd"

print("="*80)
print("ULC G1 STAGE 2 v2 - IMPROVED LOCOMOTION")
print("="*80)
for i, lv in enumerate(CURRICULUM_LEVELS):
    print(f"  Level {i}: vx={lv['vx']}, threshold={lv['threshold']}")


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_act, hidden=[512, 256, 128]):
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
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def act(self, x, det=False):
        mean = self.actor(x)
        if det: return mean
        std = self.log_std.clamp(-2, 1).exp()
        return torch.distributions.Normal(mean, std).sample()

    def evaluate(self, x, a):
        mean, val = self.forward(x)
        std = self.log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        return val.squeeze(-1), dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


class PPO:
    def __init__(self, net, device, lr=3e-4):
        self.net = net
        self.device = device
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, 6000, 1e-5)

    def gae(self, r, v, d, nv):
        adv = torch.zeros_like(r)
        last = 0
        for t in reversed(range(len(r))):
            nxt = nv if t == len(r)-1 else v[t+1]
            delta = r[t] + 0.99 * nxt * (1-d[t]) - v[t]
            adv[t] = last = delta + 0.99 * 0.95 * (1-d[t]) * last
        return adv, adv + v

    def update(self, obs, act, old_lp, ret, adv, old_v):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tot_a, tot_c, tot_e, n = 0, 0, 0, 0
        bs = obs.shape[0]

        for _ in range(5):
            idx = torch.randperm(bs, device=self.device)
            for i in range(0, bs, 4096):
                mb = idx[i:i+4096]
                val, lp, ent = self.net.evaluate(obs[mb], act[mb])

                ratio = (lp - old_lp[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = ratio.clamp(0.8, 1.2) * adv[mb]
                a_loss = -torch.min(s1, s2).mean()

                v_clip = old_v[mb] + (val - old_v[mb]).clamp(-0.2, 0.2)
                c_loss = 0.5 * torch.max((val - ret[mb])**2, (v_clip - ret[mb])**2).mean()

                loss = a_loss + 0.5 * c_loss - 0.01 * ent.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                tot_a += a_loss.item()
                tot_c += c_loss.item()
                tot_e += ent.mean().item()
                n += 1

        self.sched.step()
        return {"a": tot_a/n, "c": tot_c/n, "e": tot_e/n, "lr": self.sched.get_last_lr()[0]}


def create_env(num_envs, device):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0))
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=G1_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0.8),
                joint_pos={
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
                }),
            actuators={
                "legs": ImplicitActuatorCfg(joint_names_expr=[".*hip.*",".*knee.*",".*ankle.*"], stiffness=150, damping=15),
                "arms": ImplicitActuatorCfg(joint_names_expr=[".*shoulder.*",".*elbow.*"], stiffness=50, damping=5),
                "torso": ImplicitActuatorCfg(joint_names_expr=["torso_joint"], stiffness=100, damping=10),
            })

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 15.0
        action_space = 12
        observation_space = 51
        state_space = 0
        sim = sim_utils.SimulationCfg(dt=1/200, render_interval=4)
        scene = SceneCfg(num_envs=num_envs, env_spacing=2.5)

    class Env(DirectRLEnv):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            jn = self.robot.joint_names
            leg_names = ["left_hip_pitch_joint","right_hip_pitch_joint","left_hip_roll_joint","right_hip_roll_joint",
                "left_hip_yaw_joint","right_hip_yaw_joint","left_knee_joint","right_knee_joint",
                "left_ankle_pitch_joint","right_ankle_pitch_joint","left_ankle_roll_joint","right_ankle_roll_joint"]
            self.leg_idx = torch.tensor([jn.index(n) for n in leg_names if n in jn], device=self.device)
            self.default_leg = torch.tensor([-0.2,-0.2,0,0,0,0,0.4,0.4,-0.2,-0.2,0,0], device=self.device)

            self.curr_level = 0
            self.curr_hist = []
            self.heights = torch.ones(self.num_envs, device=self.device) * HEIGHT_DEFAULT
            self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.phase = torch.zeros(self.num_envs, device=self.device)
            self.prev_act = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_act = torch.zeros(self.num_envs, 12, device=self.device)
            self._prev_jvel = None
            print(f"[Env] {self.num_envs} envs, level {self.curr_level}")

        @property
        def robot(self): return self.scene["robot"]

        def update_curriculum(self, r):
            self.curr_hist.append(r)
            if len(self.curr_hist) >= 100:
                avg = np.mean(self.curr_hist[-100:])
                thr = CURRICULUM_LEVELS[self.curr_level]["threshold"]
                if thr and avg > thr and self.curr_level < len(CURRICULUM_LEVELS)-1:
                    self.curr_level += 1
                    print(f"\n*** LEVEL UP! Now {self.curr_level}, avg={avg:.2f} ***\n")
                    self.curr_hist = []

        def _sample_cmd(self, ids):
            lv = CURRICULUM_LEVELS[self.curr_level]
            n = len(ids)
            self.vel_cmd[ids,0] = torch.rand(n, device=self.device)*(lv["vx"][1]-lv["vx"][0])+lv["vx"][0]
            self.vel_cmd[ids,1] = torch.rand(n, device=self.device)*(lv["vy"][1]-lv["vy"][0])+lv["vy"][0]
            self.vel_cmd[ids,2] = torch.rand(n, device=self.device)*(lv["vyaw"][1]-lv["vyaw"][0])+lv["vyaw"][0]

        def _pre_physics_step(self, act):
            self.actions = act.clone()
            tgt = self.robot.data.default_joint_pos.clone()
            tgt[:, self.leg_idx] = self.default_leg + act * 0.4
            self.robot.set_joint_position_target(tgt)
            self.prev_act = act.clone()
            self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0

        def _apply_action(self): pass

        def _get_observations(self):
            r = self.robot
            from isaaclab.utils.math import quat_apply_inverse as qai
            q = r.data.root_quat_w
            lv = qai(q, r.data.root_lin_vel_w)
            av = qai(q, r.data.root_ang_vel_w)
            g = qai(q, torch.tensor([0,0,-1.], device=self.device).expand(self.num_envs,-1))
            jp = r.data.joint_pos[:, self.leg_idx]
            jv = r.data.joint_vel[:, self.leg_idx]
            gait = torch.stack([torch.sin(2*np.pi*self.phase), torch.cos(2*np.pi*self.phase)], -1)
            obs = torch.cat([lv, av, g, jp, jv, self.heights[:,None], self.vel_cmd, gait, self.prev_act], -1)
            return {"policy": obs.clamp(-10,10).nan_to_num()}

        def _get_rewards(self):
            r = self.robot
            from isaaclab.utils.math import quat_apply_inverse as qai
            pos = r.data.root_pos_w
            q = r.data.root_quat_w
            lv = qai(q, r.data.root_lin_vel_w)
            av = qai(q, r.data.root_ang_vel_w)
            jp = r.data.joint_pos[:, self.leg_idx]
            jv = r.data.joint_vel[:, self.leg_idx]

            # Velocity tracking
            r_vx = torch.exp(-2 * (lv[:,0] - self.vel_cmd[:,0])**2)
            r_vy = torch.exp(-3 * (lv[:,1] - self.vel_cmd[:,1])**2)
            r_vyaw = torch.exp(-2 * (av[:,2] - self.vel_cmd[:,2])**2)

            # Gait: alternating knee bend
            lk, rk = jp[:,6], jp[:,7]
            lh, rh = jp[:,0], jp[:,1]
            ph = self.phase
            l_sw = (ph < 0.5).float()
            r_sw = (ph >= 0.5).float()
            knee_err = (lk - (l_sw*0.6 + (1-l_sw)*0.3))**2 + (rk - (r_sw*0.6 + (1-r_sw)*0.3))**2
            r_gait = torch.exp(-3 * knee_err)

            # Foot clearance
            hip_err = l_sw*(lh+0.3)**2 + r_sw*(rh+0.3)**2
            r_clear = torch.exp(-2 * hip_err)

            # Symmetry
            l_act = self.actions[:, 0::2]
            r_act = self.actions[:, 1::2]
            r_sym = torch.exp(-1 * (l_act - r_act).pow(2).mean(-1))

            # Height
            r_h = torch.exp(-10 * (pos[:,2] - self.heights)**2)

            # Orientation
            g = qai(q, torch.tensor([0,0,-1.], device=self.device).expand(self.num_envs,-1))
            r_ori = torch.exp(-5 * (g[:,0]**2 + g[:,1]**2))

            # Stability
            r_stab = torch.exp(-0.5 * av.pow(2).sum(-1))

            # Penalties
            act_diff = self.actions - self._prev_act
            p_smooth = act_diff.pow(2).sum(-1)
            self._prev_act = self.actions.clone()

            if self._prev_jvel is not None:
                p_accel = (jv - self._prev_jvel).pow(2).sum(-1)
            else:
                p_accel = torch.zeros(self.num_envs, device=self.device)
            self._prev_jvel = jv.clone()

            p_energy = (jv.abs() * self.actions.abs()).sum(-1)

            rew = (REWARD_WEIGHTS["vx"]*r_vx + REWARD_WEIGHTS["vy"]*r_vy + REWARD_WEIGHTS["vyaw"]*r_vyaw +
                   REWARD_WEIGHTS["gait"]*r_gait + REWARD_WEIGHTS["clearance"]*r_clear + REWARD_WEIGHTS["symmetry"]*r_sym +
                   REWARD_WEIGHTS["height"]*r_h + REWARD_WEIGHTS["orientation"]*r_ori + REWARD_WEIGHTS["stability"]*r_stab +
                   REWARD_WEIGHTS["smooth"]*p_smooth + REWARD_WEIGHTS["accel"]*p_accel + REWARD_WEIGHTS["energy"]*p_energy +
                   REWARD_WEIGHTS["alive"])

            self.extras = {"R/vx": r_vx.mean(), "R/gait": r_gait.mean(), "R/sym": r_sym.mean(),
                          "R/h": r_h.mean(), "M/h": pos[:,2].mean(), "M/vx": lv[:,0].mean(), "curr": self.curr_level}
            return rew.clamp(-10, 30)

        def _get_dones(self):
            from isaaclab.utils.math import quat_apply_inverse as qai
            h = self.robot.data.root_pos_w[:,2]
            g = qai(self.robot.data.root_quat_w, torch.tensor([0,0,-1.], device=self.device).expand(self.num_envs,-1))
            term = (h < 0.35) | (h > 1.1) | (g[:,:2].abs().max(-1)[0] > 0.6)
            return term, self.episode_length_buf >= self.max_episode_length

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if len(ids) == 0: return
            pos = torch.tensor([[0,0,0.8]], device=self.device).expand(len(ids),-1).clone()
            quat = torch.tensor([[0,0,0,1.]], device=self.device).expand(len(ids),-1)
            self.robot.write_root_pose_to_sim(torch.cat([pos,quat],-1), ids)
            self.robot.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=self.device), ids)
            dp = self.robot.data.default_joint_pos[ids]
            self.robot.write_joint_state_to_sim(dp, torch.zeros_like(dp), None, ids)
            self._sample_cmd(ids)
            self.phase[ids] = torch.rand(len(ids), device=self.device)
            self.prev_act[ids] = 0
            self._prev_act[ids] = 0

    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    return Env(cfg), 51, 12


def train():
    device = "cuda:0"
    print(f"\n[INFO] Creating env with {args_cli.num_envs} envs...")
    env, nobs, nact = create_env(args_cli.num_envs, device)

    net = ActorCritic(nobs, nact).to(device)

    if args_cli.stage1_checkpoint:
        print(f"[INFO] Loading Stage 1: {args_cli.stage1_checkpoint}")
        ckpt = torch.load(args_cli.stage1_checkpoint, map_location=device, weights_only=False)
        state = ckpt["actor_critic"]
        cur = net.state_dict()
        for k in state:
            if k in cur and state[k].shape == cur[k].shape:
                cur[k] = state[k]
        net.load_state_dict(cur)

    ppo = PPO(net, device)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/ulc/{args_cli.experiment_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[INFO] Log: {log_dir}")

    best_rew = float('-inf')
    net.log_std.data.fill_(np.log(0.8))

    obs, _ = env.reset()
    obs = obs["policy"]
    start = datetime.now()

    for it in range(args_cli.max_iterations):
        t0 = datetime.now()

        obs_b, act_b, rew_b, done_b, val_b, lp_b = [], [], [], [], [], []

        for _ in range(24):
            with torch.no_grad():
                mean, val = net(obs)
                std = net.log_std.clamp(-2,1).exp()
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                lp = dist.log_prob(act).sum(-1)

            obs_b.append(obs)
            act_b.append(act)
            val_b.append(val.squeeze(-1))
            lp_b.append(lp)

            obs_d, rew, term, trunc, _ = env.step(act)
            obs = obs_d["policy"]

            rew_b.append(rew)
            done_b.append((term|trunc).float())

        obs_b = torch.stack(obs_b)
        act_b = torch.stack(act_b)
        rew_b = torch.stack(rew_b)
        done_b = torch.stack(done_b)
        val_b = torch.stack(val_b)
        lp_b = torch.stack(lp_b)

        with torch.no_grad():
            _, nv = net(obs)
            nv = nv.squeeze(-1)

        adv, ret = ppo.gae(rew_b, val_b, done_b, nv)

        info = ppo.update(obs_b.view(-1,nobs), act_b.view(-1,nact), lp_b.view(-1), ret.view(-1), adv.view(-1), val_b.view(-1))

        # Std anneal (LINEAR - key improvement!)
        prog = it / args_cli.max_iterations
        std = 0.8 + (0.2 - 0.8) * prog  # 0.8 -> 0.2 linearly
        net.log_std.data.fill_(np.log(std))

        mean_rew = rew_b.mean().item()
        env.update_curriculum(mean_rew)

        dt = (datetime.now() - t0).total_seconds()
        fps = 24 * args_cli.num_envs / dt

        if mean_rew > best_rew:
            best_rew = mean_rew
            torch.save({"actor_critic": net.state_dict(), "optimizer": ppo.opt.state_dict(),
                       "iteration": it, "best_reward": best_rew, "curriculum_level": env.curr_level},
                      f"{log_dir}/model_best.pt")
            print(f"[BEST] {best_rew:.2f}")

        writer.add_scalar("Train/reward", mean_rew, it)
        writer.add_scalar("Train/std", std, it)
        writer.add_scalar("Loss/actor", info["a"], it)
        writer.add_scalar("Loss/critic", info["c"], it)
        writer.add_scalar("Curriculum/level", env.curr_level, it)

        if it % 10 == 0:
            el = datetime.now() - start
            eta = el / (it+1) * (args_cli.max_iterations - it)
            print(f"#{it:5d} | R={mean_rew:6.2f} | Best={best_rew:6.2f} | Std={std:.3f} | Lv={env.curr_level} | FPS={fps:.0f} | {str(el).split('.')[0]} / {str(eta).split('.')[0]}")

        if (it+1) % 500 == 0:
            torch.save({"actor_critic": net.state_dict(), "optimizer": ppo.opt.state_dict(),
                       "iteration": it, "best_reward": best_rew, "curriculum_level": env.curr_level},
                      f"{log_dir}/model_{it+1}.pt")

        writer.flush()

    torch.save({"actor_critic": net.state_dict(), "iteration": args_cli.max_iterations,
               "best_reward": best_rew, "curriculum_level": env.curr_level}, f"{log_dir}/model_final.pt")

    writer.close()
    env.close()
    print(f"\nDone! Best={best_rew:.2f}, Level={env.curr_level}, Log={log_dir}")

if __name__ == "__main__":
    train()
    simulation_app.close()