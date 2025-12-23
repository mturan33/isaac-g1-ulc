# Third-Party Licenses and Acknowledgements

This project incorporates code and assets from the following open-source projects:

---

## 1. unitree_sim_isaaclab

**Repository:** https://github.com/unitreerobotics/unitree_sim_isaaclab

**Copyright:** Copyright 2025 HangZhou YuShu TECHNOLOGY CO., LTD. ("Unitree Robotics")

**License:** Apache License, Version 2.0

**Used Components:**
- G1 robot configurations (`robots/` directory)
- DDS communication modules (`dds/` directory)
- Task environment templates (`tasks/` directory)
- Low-level control interfaces (`layeredcontrol/` directory)
- Image server utilities (`image_server/` directory)

**Modifications Made:**
- Adapted robot configurations for VLM-RL hierarchical control
- Extended task environments with semantic mapping integration
- Modified observation spaces for VLM input processing
- Added Turkish language command processing

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 2. Isaac Lab

**Repository:** https://github.com/isaac-sim/IsaacLab

**Copyright:** Copyright (c) 2022-2025, The Isaac Lab Project Developers.

**License:** BSD 3-Clause License

**Used Components:**
- Core simulation framework
- RL environment interfaces
- Sensor simulation modules

---

## 3. RSL-RL

**Repository:** https://github.com/leggedrobotics/rsl_rl

**Copyright:** Copyright (c) 2021 Robotic Systems Lab, ETH Zurich

**License:** BSD 3-Clause License

**Used Components:**
- PPO algorithm implementation reference
- Training infrastructure patterns

---

## 4. Florence-2 / Molmo2

**Florence-2 Repository:** https://huggingface.co/microsoft/Florence-2-large

**Copyright:** Microsoft Corporation

**License:** MIT License

**Used Components:**
- Vision-Language Model for semantic understanding
- Object grounding and scene description

---

## 5. Flow Matching Implementation

**Reference:** "Flow Matching for Generative Modeling" (Lipman et al., 2023)

**Used Components:**
- Trajectory generation for smooth manipulation
- Conditional flow matching for arm control

---

## Original Contributions

The following components are original work by the author (Mehmet Turan):

- **Hierarchical VLM-RL Architecture:** Integration of VLM planning with RL execution
- **Semantic Mapping System:** Real-time object tracking without per-frame VLM calls
- **Turkish Language Commands:** Natural language processing for Turkish instructions
- **Whole-Body Coordination:** Balance maintenance during manipulation tasks
- **Custom PPO Implementation:** Adapted for humanoid locomotion with manipulation

---

## Contact

For questions about licensing or attribution:
- **Author:** Mehmet Turan
- **Project:** VLM-RL G1 Humanoid System
- **Purpose:** Graduate School Application / RSS 2026 Workshop Submission

---

*Last Updated: December 23, 2025*
