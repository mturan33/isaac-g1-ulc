# vlm_integration.py
class HierarchicalVLMController:
    def __init__(self):
        self.vlm = Florence2Integration()
        self.semantic_map = MomaGraphSemanticMap()
        self.navigator = LoGoPlanner()  # Veya basit heuristic başla

    def step(self, rgb, depth, task_description):
        # 1. VLM inference (cached, ~2Hz)
        if self.should_update_vlm():
            goals = self.vlm.get_grounded_objects(rgb, task_description)
            self.semantic_map.update(rgb, depth, task_description)

        # 2. Navigation (20Hz)
        velocity_cmd = self.navigator.plan(
            self.semantic_map.get_goal_position(),
            depth
        )

        return velocity_cmd  # (vx, vy, wz) → PPO'ya gider