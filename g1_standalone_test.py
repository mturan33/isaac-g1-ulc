"""
G1 Standalone Test - Minimal working example (FIXED)
"""

import os
os.environ["PROJECT_ROOT"] = r"C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\unitree_sim_isaaclab"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import torch
from pxr import UsdLux, UsdGeom, Gf, UsdPhysics
import omni.usd
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from external.unitree_robots.unitree import G129_CFG_WITH_DEX1_BASE_FIX


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.75])
    stage = omni.usd.get_context().get_stage()

    # Ground Plane
    ground_prim = stage.DefinePrim("/World/groundPlane", "Xform")
    plane_prim = stage.DefinePrim("/World/groundPlane/CollisionMesh", "Mesh")
    mesh = UsdGeom.Mesh(plane_prim)
    size = 50.0
    mesh.GetPointsAttr().Set([(-size, 0, -size), (size, 0, -size), (size, 0, size), (-size, 0, size)])
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(plane_prim)
    UsdPhysics.MeshCollisionAPI.Apply(plane_prim)

    # Light
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.GetIntensityAttr().Set(1000.0)

    # G1 Robot - 0.8m above ground
    robot_cfg = G129_CFG_WITH_DEX1_BASE_FIX.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.init_state.pos = (0.0, 0.8, 0.0)
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    print(f"\n{'='*60}\nG1 Robot yüklendi! Joints: {robot.num_joints}, Bodies: {robot.num_bodies}\n{'='*60}\n")
    robot.reset()

    count = 0
    while simulation_app.is_running():
        if count % 1000 == 0:
            pos = robot.data.root_pos_w[0]
            print(f"[Step {count}] Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        sim.step()
        robot.update(sim.get_physics_dt())
        count += 1

    simulation_app.close()

if __name__ == "__main__":
    main()