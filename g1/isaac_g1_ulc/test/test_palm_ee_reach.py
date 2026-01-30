"""
G1 Palm EE Reaching Test
- EE = palm center (red marker position = palm_pos + forward_offset)
- Orientation: palm forward should point DOWN
- Gripper: closes when near target
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import math
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Palm forward offset (distance from palm_link to fingertip center)
PALM_FORWARD_OFFSET = 0.08  # meters

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
            "right_shoulder_pitch_joint": 0.5,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.6,
            "right_elbow_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.5,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.6,
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

def get_palm_forward_vector(quat):
    """Get palm forward (+X) direction from quaternion (wxyz)"""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fwd_x = 1 - 2*(y*y + z*z)
    fwd_y = 2*(x*y + w*z)
    fwd_z = 2*(x*z - w*y)
    return torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)

def compute_ee_position(palm_pos, palm_quat, offset=PALM_FORWARD_OFFSET):
    """Compute EE position = palm + offset * forward"""
    forward = get_palm_forward_vector(palm_quat)
    return palm_pos + offset * forward

def compute_orientation_error(palm_quat):
    """
    Compute how much palm forward deviates from pointing DOWN (-Z world)
    Returns: angle error in radians (0 = perfect, pi = opposite)
    """
    forward = get_palm_forward_vector(palm_quat)
    target_dir = torch.tensor([[0.0, 0.0, -1.0]], device=forward.device)  # Down

    # Dot product: 1 = aligned, -1 = opposite
    dot = (forward * target_dir).sum(dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angle error
    angle_error = torch.acos(dot)
    return angle_error, forward

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 0.8, 1.0], [0.0, 0.0, 0.8])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={
            "ee": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red - EE
            ),
            "target": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green - Target
            ),
            "palm_down": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),  # Cyan - palm forward dir
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

    # Target position - önde ve aşağıda (robot'un önünde, bel hizasında)
    target_pos = torch.tensor([[0.35, -0.2, 0.75]], device=device)

    # Grasp threshold
    GRASP_DISTANCE = 0.05  # 5cm

    print("\n" + "="*70)
    print("G1 PALM EE REACHING TEST")
    print("="*70)
    print(f"Target position: {target_pos[0].tolist()}")
    print(f"Palm forward offset: {PALM_FORWARD_OFFSET}m")
    print(f"Grasp distance: {GRASP_DISTANCE}m")
    print("\nMarkers:")
    print("  RED = End-Effector (palm center)")
    print("  GREEN = Target")
    print("  CYAN = Palm forward direction indicator")
    print("\nGoal: EE reaches target + palm points DOWN + gripper closes")
    print("="*70 + "\n")

    # Simple IK-like arm control (manual for demo)
    # We'll animate the arm to reach the target
    arm_sequence = [
        [0.5, -0.2, 0.0, 0.6, 0.0],   # Start
        [0.8, -0.1, 0.2, 0.9, 0.0],   # Reach forward-down
        [1.0, 0.0, 0.3, 1.1, 0.0],    # More forward
        [1.2, 0.1, 0.2, 1.0, 0.0],    # Adjust
        [1.1, 0.0, 0.0, 0.9, 0.3],    # Final reach
    ]

    step = 0
    phase_duration = 1.0  # seconds per phase
    gripper_closed = False

    while simulation_app.is_running():
        # Determine arm pose from sequence
        total_time = step * sim_cfg.dt
        phase_idx = min(int(total_time / phase_duration), len(arm_sequence) - 1)

        # Interpolate between phases
        phase_progress = (total_time % phase_duration) / phase_duration
        if phase_idx < len(arm_sequence) - 1:
            arm_start = torch.tensor(arm_sequence[phase_idx], device=device)
            arm_end = torch.tensor(arm_sequence[phase_idx + 1], device=device)
            arm_target = arm_start + phase_progress * (arm_end - arm_start)
        else:
            arm_target = torch.tensor(arm_sequence[-1], device=device)

        # Get current palm state
        palm_pos = robot.data.body_pos_w[:, palm_idx, :]
        palm_quat = robot.data.body_quat_w[:, palm_idx, :]

        # Compute EE position (palm center)
        ee_pos = compute_ee_position(palm_pos, palm_quat)

        # Compute distance to target
        dist_to_target = torch.norm(ee_pos - target_pos, dim=-1)

        # Compute orientation error
        orient_error, palm_forward = compute_orientation_error(palm_quat)

        # Gripper control: close if near target
        if dist_to_target[0] < GRASP_DISTANCE:
            gripper_closed = True

        # Set finger positions
        if gripper_closed:
            finger_targets = finger_upper  # Closed
        else:
            finger_targets = finger_lower  # Open

        # Build joint targets
        joint_target = robot.data.joint_pos.clone()

        for i, idx in enumerate(arm_indices):
            joint_target[0, idx] = arm_target[i]

        for i, idx in enumerate(finger_indices):
            joint_target[0, idx] = finger_targets[i]

        robot.set_joint_position_target(joint_target)

        # Visualize
        palm_forward_pos = ee_pos + 0.05 * palm_forward
        marker_pos = torch.cat([ee_pos, target_pos, palm_forward_pos], dim=0)
        markers.visualize(marker_pos, marker_indices=torch.tensor([0, 1, 2], device=device))

        # Print status
        if step % 50 == 0:
            gripper_state = "CLOSED" if gripper_closed else "OPEN"
            orient_deg = math.degrees(orient_error[0].item())
            print(f"Step {step:4d} | Phase {phase_idx} | Dist: {dist_to_target[0]:.3f}m | Orient err: {orient_deg:5.1f}° | Gripper: {gripper_state}")
            print(f"         | EE: [{ee_pos[0,0]:.2f}, {ee_pos[0,1]:.2f}, {ee_pos[0,2]:.2f}] | Palm fwd: [{palm_forward[0,0]:.2f}, {palm_forward[0,1]:.2f}, {palm_forward[0,2]:.2f}]")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        step += 1

        if step > 3000:
            break

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print(f"Final distance to target: {dist_to_target[0]:.3f}m")
    print(f"Final orientation error: {math.degrees(orient_error[0].item()):.1f}°")
    print(f"Gripper state: {'CLOSED' if gripper_closed else 'OPEN'}")
    print("="*70)
    simulation_app.close()

if __name__ == "__main__":
    main()