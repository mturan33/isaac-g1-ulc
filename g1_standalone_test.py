"""
G1 Standalone Test - DDS gerektirmez, sadece simülasyon
"""

import os

# PROJECT_ROOT'u ayarla
os.environ["PROJECT_ROOT"] = r"C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\unitree_sim_isaaclab"

from isaacsim import SimulationApp

# Simulation başlat
simulation_app = SimulationApp({"headless": False})

# Isaac imports (SimulationApp'ten sonra!)
import torch
import omni.usd
from pxr import UsdGeom, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# G1 config'i import et
from external.unitree_robots.unitree import G129_CFG_WITH_DEX1_BASE_FIX


def main():
    """G1 robot'u basitçe yükle ve göster."""

    # Simulation context - design mode'da başlat
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Stage'i ayarla
    sim.set_camera_view(eye=[3.0, 3.0, 2.5], target=[0.0, 0.0, 0.75])

    # Ground plane ekle
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light ekle
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg_light.func("/World/Light", cfg_light)

    # G1 Robot'u spawn et
    robot_cfg = G129_CFG_WITH_DEX1_BASE_FIX.copy()
    robot_cfg.prim_path = "/World/G1"
    robot = Articulation(cfg=robot_cfg)

    # Scene'i oluştur ve başlat
    sim.reset()

    # Robot bilgilerini yazdır
    print("\n" + "="*50)
    print("G1 Robot başarıyla yüklendi!")
    print(f"Joint sayısı: {robot.num_joints}")
    print(f"Body sayısı: {robot.num_bodies}")
    print(f"Joint isimleri: {robot.joint_names[:10]}...")  # İlk 10
    print("="*50 + "\n")
    print("Simülasyonu kapatmak için pencereyi kapat.")

    # Robot'u başlangıç pozisyonuna getir
    robot.reset()

    # Ana döngü
    count = 0
    while simulation_app.is_running():
        # Her 1000 step'te bir bilgi yazdır
        if count % 1000 == 0 and count > 0:
            print(f"[Step {count}] Simülasyon devam ediyor...")

        # Simülasyon adımı
        sim.step()
        robot.update(sim.get_physics_dt())
        count += 1

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()