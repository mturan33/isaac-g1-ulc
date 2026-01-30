"""
G1 Gripper & Palm Test - FIXED BASE - Arm Extended Forward
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# G1 config - Arm extended forward so hand is visible
G1_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # Right arm - EXTENDED FORWARD (visible)
            "right_shoulder_pitch_joint": 0.8,   # Öne doğru
            "right_shoulder_roll_joint": -0.3,   # Biraz dışa
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.5,      # Hafif bükük
            "right_elbow_roll_joint": 0.0,
            # Left arm - aynı poz (simetrik)
            "left_shoulder_pitch_joint": 0.8,
            "left_shoulder_roll_joint": 0.3,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.5,
            "left_elbow_roll_joint": 0.0,
        },
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*elbow.*"],
            stiffness=100.0,
            damping=10.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*zero.*", ".*one.*", ".*two.*", ".*three.*", ".*four.*", ".*five.*", ".*six.*"],
            stiffness=20.0,
            damping=2.0,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
            stiffness=100.0,
            damping=10.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint"],
            stiffness=100.0,
            damping=10.0,
        ),
    },
)

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
]
RIGHT_FINGER_JOINTS = [
    "right_zero_joint", "right_one_joint", "right_two_joint",
    "right_three_joint", "right_four_joint", "right_five_joint", "right_six_joint",
]

# Kol sabit kalacak pozisyon
ARM_POSE = [0.8, -0.3, 0.0, 0.5, 0.0]

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
    robot = G1_FIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    # Kamerı önden ve biraz yandan görecek şekilde
    sim.set_camera_view([1.5, 0.5, 1.2], [0.0, 0.0, 1.0])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={
            "palm": sim_utils.SphereCfg(
                radius=0.03,  # Biraz büyüttüm
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "palm_forward": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "palm_down": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )
    markers = VisualizationMarkers(marker_cfg)

    sim.reset()

    robot = scene["robot"]
    device = robot.device

    finger_indices = [robot.joint_names.index(n) for n in RIGHT_FINGER_JOINTS]
    arm_indices = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    palm_idx = robot.body_names.index("right_palm_link")

    joint_limits = robot.root_physx_view.get_dof_limits()
    finger_lower = torch.tensor([joint_limits[0, i, 0].item() for i in finger_indices], device=device)
    finger_upper = torch.tensor([joint_limits[0, i, 1].item() for i in finger_indices], device=device)

    print("\n" + "="*60)
    print("G1 GRIPPER TEST - ARM EXTENDED FORWARD")
    print("="*60)
    print(f"Arm pose (fixed): {ARM_POSE}")
    print("\nMarkers: Green=Palm, Red=Forward(+X), Blue=Down(-Z)")
    print("="*60 + "\n")

    arm_targets = torch.tensor([ARM_POSE], device=device)
    cycle_duration = 2.0
    step = 0

    while simulation_app.is_running():
        t = (step * sim_cfg.dt) % cycle_duration
        phase = abs(2.0 * t / cycle_duration - 1.0)

        finger_targets = finger_lower + phase * (finger_upper - finger_lower)

        joint_target = robot.data.joint_pos.clone()

        # Kol SABİT - her zaman aynı pozda
        for i, idx in enumerate(arm_indices):
            joint_target[0, idx] = arm_targets[0, i]

        # Parmaklar açılıp kapanıyor
        for i, idx in enumerate(finger_indices):
            joint_target[0, idx] = finger_targets[i]

        robot.set_joint_position_target(joint_target)

        palm_pos = robot.data.body_pos_w[:, palm_idx, :]
        palm_quat = robot.data.body_quat_w[:, palm_idx, :]

        w, x, y, z = palm_quat[0, 0], palm_quat[0, 1], palm_quat[0, 2], palm_quat[0, 3]

        # Forward direction (+X in palm local frame)
        fwd_x = 1 - 2*(y*y + z*z)
        fwd_y = 2*(x*y + w*z)
        fwd_z = 2*(x*z - w*y)
        forward = torch.tensor([[fwd_x, fwd_y, fwd_z]], device=device)

        # Down direction (-Z in palm local frame)
        down_x = 2*(x*z + w*y)
        down_y = 2*(y*z - w*x)
        down_z = 1 - 2*(x*x + y*y)
        down = torch.tensor([[-down_x, -down_y, -down_z]], device=device)

        forward_pos = palm_pos + 0.08 * forward
        down_pos = palm_pos + 0.08 * down

        marker_pos = torch.cat([palm_pos, forward_pos, down_pos], dim=0)
        markers.visualize(marker_pos, marker_indices=torch.tensor([0, 1, 2], device=device))

        if step % 100 == 0:
            state = "CLOSING" if t < cycle_duration/2 else "OPENING"
            print(f"Step {step:4d} | {state:8s} | Palm: [{palm_pos[0,0]:.2f}, {palm_pos[0,1]:.2f}, {palm_pos[0,2]:.2f}]")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        step += 1

        if step > 2000:
            break

    print("\nTEST COMPLETE")
    simulation_app.close()

if __name__ == "__main__":
    main()