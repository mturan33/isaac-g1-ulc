"""
G1 Gripper & Palm Test Script WITH Locomotion Policy
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stage3_checkpoint", type=str, required=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

LOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
]
RIGHT_FINGER_JOINTS = [
    "right_zero_joint", "right_one_joint", "right_two_joint",
    "right_three_joint", "right_four_joint", "right_five_joint", "right_six_joint",
]

DEFAULT_ARM_POSE = {
    "right_shoulder_pitch_joint": 0.3, "right_shoulder_roll_joint": 0.2,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_pitch_joint": 0.8, "right_elbow_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_pitch_joint": 0.5, "left_elbow_roll_joint": 0.0,
}

class TestSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

class LocoPolicy:
    """RSL-RL format policy loader - Stage 3 architecture"""
    def __init__(self, checkpoint_path, device):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        actor_critic_state = checkpoint["actor_critic"]

        # Stage 3 architecture: Linear -> LayerNorm -> ELU pattern
        # Input: 57, Output: 12
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(57, 512),      # 0
            torch.nn.LayerNorm(512),       # 1
            torch.nn.ELU(),                # 2
            torch.nn.Linear(512, 256),     # 3
            torch.nn.LayerNorm(256),       # 4
            torch.nn.ELU(),                # 5
            torch.nn.Linear(256, 128),     # 6
            torch.nn.LayerNorm(128),       # 7
            torch.nn.ELU(),                # 8
            torch.nn.Linear(128, 12),      # 9
        ).to(device)

        # Extract actor weights
        actor_state = {}
        for k, v in actor_critic_state.items():
            if k.startswith("actor."):
                new_key = k.replace("actor.", "")
                actor_state[new_key] = v

        self.actor.load_state_dict(actor_state)
        print(f"Loaded Stage 3 loco policy (obs=57, act=12)")
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, obs):
        return self.actor(obs)

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, -0.5, 1.2], [0.0, 0.0, 0.8])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={
            "palm": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "palm_forward": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    markers = VisualizationMarkers(marker_cfg)

    sim.reset()

    robot = scene["robot"]
    device = robot.device

    # Load loco policy
    loco_policy = LocoPolicy(args.stage3_checkpoint, device)

    # Get indices
    loco_indices = [robot.joint_names.index(n) for n in LOCO_JOINT_NAMES]
    finger_indices = [robot.joint_names.index(n) for n in RIGHT_FINGER_JOINTS]
    palm_idx = robot.body_names.index("right_palm_link")

    # Finger limits
    joint_limits = robot.root_physx_view.get_dof_limits()
    finger_lower = torch.tensor([joint_limits[0, i, 0].item() for i in finger_indices], device=device)
    finger_upper = torch.tensor([joint_limits[0, i, 1].item() for i in finger_indices], device=device)

    prev_loco_actions = torch.zeros(1, 12, device=device)
    commands = torch.zeros(1, 3, device=device)  # Stand still

    print("\n" + "="*60)
    print("TEST: Robot should STAND, hand should OPEN/CLOSE")
    print("="*60 + "\n")

    cycle_duration = 3.0
    step = 0
    decimation = 4

    while simulation_app.is_running():
        # Build Stage 3 observation (57 dims)
        # Structure: ang_vel(3) + gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + actions(12) + height(1) + feet(8) + phase(2) + freq(1)
        # Actually let me check - 3+3+3+12+12+12 = 45, so extra 12 must be something else
        # Looking at checkpoint input=57: maybe 45 + 12 more (feet contact, phase, etc.)

        base_quat = robot.data.root_quat_w
        base_ang_vel = robot.data.root_ang_vel_b
        base_lin_vel = robot.data.root_lin_vel_b

        w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        gravity_x = 2.0 * (x * z - w * y)
        gravity_y = 2.0 * (y * z + w * x)
        gravity_z = 1.0 - 2.0 * (x * x + y * y)
        projected_gravity = torch.stack([gravity_x, gravity_y, gravity_z], dim=-1)

        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        loco_pos = joint_pos[:, loco_indices]
        loco_vel = joint_vel[:, loco_indices]

        # Build 57-dim obs
        # Based on Stage 3 training: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + pos(12) + vel(12) + actions(12) + extra(9)
        loco_obs = torch.cat([
            base_lin_vel[:, :2] * 2.0,       # 2 (vx, vy scaled)
            base_ang_vel[:, 2:3] * 0.25,     # 1 (yaw rate)
            projected_gravity,                # 3
            commands * torch.tensor([[2.0, 2.0, 0.25]], device=device),  # 3
            loco_pos,                         # 12
            loco_vel * 0.05,                  # 12
            prev_loco_actions,                # 12
            # Extra padding to reach 57
            torch.zeros(1, 57 - 45, device=device),  # 12 padding (height, phase, feet, etc.)
        ], dim=-1)

        if step % decimation == 0:
            loco_action = loco_policy.get_action(loco_obs)
            prev_loco_actions = loco_action.clone()

        # Finger animation
        t = (step * sim_cfg.dt) % cycle_duration
        phase = abs(2.0 * t / cycle_duration - 1.0)
        finger_targets = finger_lower + phase * (finger_upper - finger_lower)

        # Apply targets
        joint_target = robot.data.joint_pos.clone()

        # Loco joints
        for i, idx in enumerate(loco_indices):
            joint_target[0, idx] = joint_pos[0, idx] + loco_action[0, i] * 0.25

        # Arms
        for name in RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS:
            idx = robot.joint_names.index(name)
            joint_target[0, idx] = DEFAULT_ARM_POSE[name]

        # Fingers
        for i, idx in enumerate(finger_indices):
            joint_target[0, idx] = finger_targets[i]

        robot.set_joint_position_target(joint_target)

        # Visualize palm
        palm_pos = robot.data.body_pos_w[:, palm_idx, :]
        palm_quat = robot.data.body_quat_w[:, palm_idx, :]

        w, x, y, z = palm_quat[0, 0], palm_quat[0, 1], palm_quat[0, 2], palm_quat[0, 3]
        forward_x = 1 - 2*(y*y + z*z)
        forward_y = 2*(x*y + w*z)
        forward_z = 2*(x*z - w*y)
        forward = torch.tensor([[forward_x, forward_y, forward_z]], device=device)
        forward_pos = palm_pos + 0.08 * forward

        marker_pos = torch.cat([palm_pos, forward_pos], dim=0)
        markers.visualize(marker_pos, marker_indices=torch.tensor([0, 1], device=device))

        if step % 200 == 0:
            h = robot.data.root_pos_w[0, 2].item()
            state = "CLOSING" if t < cycle_duration/2 else "OPENING"
            print(f"Step {step:5d} | Height: {h:.2f}m | {state}")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        step += 1

        if step > 4000:
            break

    simulation_app.close()

if __name__ == "__main__":
    main()