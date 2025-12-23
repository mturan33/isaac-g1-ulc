import sys
sys.path.insert(0, r'C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl')

try:
    from external import unitree_robots
    print("OK: unitree_robots imported")
except Exception as e:
    print(f"FAIL: unitree_robots - {e}")

try:
    from external import unitree_dds
    print("OK: unitree_dds imported")
except Exception as e:
    print(f"FAIL: unitree_dds - {e}")

try:
    from external import layeredcontrol
    print("OK: layeredcontrol imported")
except Exception as e:
    print(f"FAIL: layeredcontrol - {e}")

try:
    from external import image_server
    print("OK: image_server imported")
except Exception as e:
    print(f"FAIL: image_server - {e}")

print("\nImport testi tamamlandi!")
