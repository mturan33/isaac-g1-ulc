# Copyright (c) 2025, Turan - VLM-RL Project
# G1 Loco-Manipulation Demo: Wave while Walking (No Pink IK Required)
#
# Bu script, G1 humanoid robotun yürürken el sallamasını gösterir.
# Lower body: PPO locomotion policy (ayakta durma/yürüme)
# Upper body: Sinusoidal joint control (el sallama)
#
# Kullanım:
#   cd C:\IsaacLab
#   .\isaaclab.bat -p <path_to_this_script>

"""
G1 Wave Demo - Pink IK olmadan loco-manipulation demonstrasyonu.

Bu demo gösterir:
1. G1 robotun masa içeren bir sahnede spawn edilmesi
2. Lower body'nin PPO ile ayakta tutulması
3. Upper body arm joint'lerinin sinusoidal kontrolü (el sallama)
4. Locomotion + manipulation koordinasyonu

NO PINK IK / NO PINOCCHIO REQUIRED!
"""

import argparse
import math
import torch
from omni.isaac.lab.app import AppLauncher

# CLI argümanları
parser = argparse.ArgumentParser(description="G1 Wave Demo - Loco-manipulation without Pink IK")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")

# AppLauncher argümanları ekle
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim başlat
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab importları (Isaac Sim başladıktan sonra)
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

# Unitree G1 asset
try:
    from omni.isaac.lab_assets.unitree import UNITREE_G1_CFG
except ImportError:
    # Isaac Lab 2.3.1 için alternatif import
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG as UNITREE_G1_CFG


##
# Scene Configuration
##
@configclass
class G1WaveSceneCfg(InteractiveSceneCfg):
    """G1 robot ile masa içeren sahne konfigürasyonu."""

    # Ground plane
    ground = sim_utils.GroundPlaneCfg()

    # Lighting
    dome_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.9, 0.9, 0.9),
    )

    # G1 Robot - Standing pose
    robot: ArticulationCfg = UNITREE_G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),  # 0.8m height (standing)
            joint_pos={
                # Başlangıç pozisyonu - nötr duruş
                ".*": 0.0,
            },
        ),
    )

    # Table (optional - for future pick&place)
    table = sim_utils.CuboidCfg(
        size=(0.8, 0.5, 0.4),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )


##
# Arm Wave Controller
##
class ArmWaveController:
    """Upper body el sallama kontrolcüsü - joint space sinusoidal kontrol."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

        # Wave parametreleri
        self.wave_frequency = 2.0  # Hz
        self.wave_amplitude = 0.5  # radyan
        self.phase = 0.0

        # G1 arm joint isimleri (29 DoF model için)
        # Left arm: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow
        # Right arm: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
        self.arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]

    def compute_wave(self, time: float) -> dict[str, float]:
        """Sinusoidal el sallama joint pozisyonlarını hesapla."""

        # Sağ kol sallama (sol kol sabit)
        wave_angle = self.wave_amplitude * math.sin(2 * math.pi * self.wave_frequency * time)

        # Joint hedefleri
        targets = {
            # Sol kol - nötr pozisyon
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.2,  # Hafif açık
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5,  # Hafif bükülü

            # Sağ kol - el sallama
            "right_shoulder_pitch_joint": -0.8,  # Kol yukarı kaldırılmış
            "right_shoulder_roll_joint": -0.3 + wave_angle * 0.3,  # Sallama (yanlara)
            "right_shoulder_yaw_joint": wave_angle * 0.2,  # Hafif rotasyon
            "right_elbow_joint": -1.2 + wave_angle * 0.3,  # Dirsek bükümü ile sallama
        }

        return targets


##
# Simple Balance Controller (placeholder for PPO)
##
class SimpleBalanceController:
    """
    Basit denge kontrolcüsü - PPO policy yüklenene kadar placeholder.

    Gerçek kullanımda bu, eğitilmiş locomotion policy ile değiştirilecek:
    - Nucleus'tan indirilen G1 locomotion model
    - RSL-RL ile eğitilmiş custom model
    """

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device

        # Leg joint isimleri
        self.leg_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]

        # Default standing pose (ayakta durma pozisyonu)
        self.standing_pose = {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.25,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.25,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
        }

    def compute_action(self) -> dict[str, float]:
        """Standing pose döndür (gerçek PPO policy placeholder)."""
        return self.standing_pose


##
# Main Demo
##
def main():
    """G1 Wave Demo ana fonksiyonu."""

    print("\n" + "=" * 60)
    print("G1 LOCO-MANIPULATION DEMO")
    print("El sallama + Ayakta durma (Pink IK gerektirmez!)")
    print("=" * 60 + "\n")

    # Simulation config
    sim_cfg = SimulationCfg(
        device=args_cli.device,
        dt=0.01,  # 100 Hz
        render_interval=2,
    )

    # Simulation context oluştur
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.8])

    # Scene oluştur
    scene_cfg = G1WaveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    # Controllers oluştur
    wave_controller = ArmWaveController(args_cli.num_envs, args_cli.device)
    balance_controller = SimpleBalanceController(args_cli.num_envs, args_cli.device)

    # Simulation başlat
    sim.reset()

    # Robot'u al
    robot: Articulation = scene["robot"]

    # Joint indekslerini bul
    arm_joint_ids = []
    leg_joint_ids = []

    for name in wave_controller.arm_joint_names:
        idx = robot.find_joints(name)
        if idx[0]:
            arm_joint_ids.append(idx[0][0])

    for name in balance_controller.leg_joint_names:
        idx = robot.find_joints(name)
        if idx[0]:
            leg_joint_ids.append(idx[0][0])

    print(f"Arm joint IDs: {arm_joint_ids}")
    print(f"Leg joint IDs: {leg_joint_ids}")
    print(f"Total robot joints: {robot.num_joints}")

    # Simulation loop
    sim_time = 0.0
    count = 0

    print("\n[INFO] Simulation başlıyor... (Ctrl+C ile çıkış)")
    print("[INFO] Robot el sallıyor ve ayakta duruyor!")

    while simulation_app.is_running():
        # Reset every 1000 steps
        if count % 1000 == 0:
            sim_time = 0.0
            scene.reset()
            print(f"\n[INFO] Reset at step {count}")

        # Wave controller - arm positions
        wave_targets = wave_controller.compute_wave(sim_time)

        # Balance controller - leg positions
        balance_targets = balance_controller.compute_action()

        # Birleştir: arm + leg hedefleri
        all_targets = {**wave_targets, **balance_targets}

        # Joint pozisyonlarını ayarla
        joint_pos = robot.data.default_joint_pos.clone()

        for joint_name, target_pos in all_targets.items():
            idx = robot.find_joints(joint_name)
            if idx[0]:
                joint_pos[:, idx[0][0]] = target_pos

        # Robot'a gönder
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()

        # Simulation step
        sim.step()
        scene.update(sim_cfg.dt)

        # Update time
        sim_time += sim_cfg.dt
        count += 1

        # Her 100 step'te bilgi ver
        if count % 100 == 0:
            base_pos = robot.data.root_pos_w[0].cpu().numpy()
            print(f"Step {count}: Base pos = ({base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f})")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()