# \# 🤖 Go2 VLM-RL Navigation

# 

# Language-conditioned quadruped navigation using Vision-Language Models (VLM) and Reinforcement Learning.

# 

# \## 📋 Overview

# 

# This project implements a hierarchical VLM-RL system for Unitree Go2 robot navigation:

# 

# ```

# "Mavi sandalyeye git" (Go to the blue chair)

# &nbsp;        │

# &nbsp;   ┌────▼────┐

# &nbsp;   │   VLM   │ ← RGB Image (640×480)

# &nbsp;   │ Phi-3-V │ → Target: {x, y, confidence}

# &nbsp;   └────┬────┘

# &nbsp;        │ Every ~1 second

# &nbsp;   ┌────▼────────┐

# &nbsp;   │ Nav Control │ → cmd\_vel: \[vx, vy, vyaw]

# &nbsp;   └────┬────────┘

# &nbsp;        │ Every 20ms

# &nbsp;   ┌────▼────────┐

# &nbsp;   │ Student RL  │ ← Depth (64×64) + Proprio (48)

# &nbsp;   │   Policy    │ → Joint Actions (12 DoF)

# &nbsp;   └────┬────────┘

# &nbsp;        │

# &nbsp;     \[GO2 🐕]

# ```

# 

# \## 🛠️ Requirements

# 

# \- Python 3.10+

# \- PyTorch 2.0+

# \- Transformers 4.40+

# \- NVIDIA GPU with 12GB+ VRAM

# \- Isaac Lab (for simulation)

# 

# \## 📦 Installation

# 

# ```powershell

# \# Install dependencies

# pip install transformers accelerate pillow

# 

# \# Clone/copy files to Isaac Lab

# \# C:\\IsaacLab\\source\\isaaclab\_tasks\\isaaclab\_tasks\\direct\\go2\_vlm\_rl\\

# ```

# 

# \## 🚀 Quick Start

# 

# \### 1. Test Scene Generator

# ```powershell

# python scene\_generator.py

# ```

# 

# \### 2. Test VLM (requires ~8GB VRAM)

# ```powershell

# python vlm\_wrapper.py test\_scene.png "mavi sandalyeye git"

# ```

# 

# \### 3. Full Demo

# ```powershell

# python run\_vlm\_nav.py --test-vlm --image test\_scene.png

# python run\_vlm\_nav.py --test-scene

# ```

# 

# \## 📁 File Structure

# 

# ```

# go2\_vlm\_rl/

# ├── vlm\_wrapper.py          # Phi-3-Vision wrapper (Windows compatible)

# ├── scene\_generator.py      # Random colored objects spawner

# ├── run\_vlm\_nav.py         # Full pipeline demo

# ├── test\_scene.png         # Test image

# └── README.md              # This file

# ```

# 

# \## 🎨 Supported Objects \& Colors

# 

# \*\*Objects:\*\* chair, table, cabinet, sofa, box, ball, cylinder, cone

# 

# \*\*Colors:\*\* blue, red, green, yellow, orange, purple, white, pink, brown, cyan

# 

# \*\*Languages:\*\* Turkish and English commands supported!

# 

# | Turkish | English |

# |---------|---------|

# | mavi sandalyeye git | go to the blue chair |

# | kırmızı masayı bul | find the red table |

# | yeşil kutuya git | navigate to green box |

# 

# \## ⚙️ Configuration

# 

# \### VLM Settings

# \- Model: `microsoft/Phi-3-vision-128k-instruct`

# \- VRAM: ~8-10GB

# \- Inference: ~100-200ms per call

# \- Attention: SDPA (Windows compatible, no Flash Attention needed)

# 

# \### Navigation Controller

# \- Max linear velocity: 1.0 m/s

# \- Max angular velocity: 1.0 rad/s

# \- Goal threshold: 0.15m

# 

# \## 🔧 Troubleshooting

# 

# \### Flash Attention Error

# The code uses `\_attn\_implementation="eager"` which works on Windows without Flash Attention.

# 

# \### VRAM Issues

# If running out of memory:

# 1\. Images are automatically resized to 768px max

# 2\. Use `low\_memory=True` in VLMWrapper

# 3\. Close other GPU applications

# 

# \### Model Download

# First run downloads ~8GB model. This is cached in `~/.cache/huggingface/`.

# 

# \## 📊 Architecture

# 

# \### High-Level (VLM)

# \- \*\*Input:\*\* RGB image + text command

# \- \*\*Output:\*\* Target coordinates (x, y) + confidence

# \- \*\*Frequency:\*\* Every 20-30 simulation steps (~1 second)

# 

# \### Low-Level (RL Policy)

# \- \*\*Input:\*\* Depth image (64×64) + proprioception (48) + cmd\_vel (3)

# \- \*\*Output:\*\* Joint positions (12 DoF)

# \- \*\*Frequency:\*\* Every simulation step (50Hz)

# 

# \## 🔮 Roadmap

# 

# \- \[x] VLM wrapper (Phi-3-Vision)

# \- \[x] Scene generator with colored objects

# \- \[x] Navigation controller

# \- \[ ] Isaac Lab environment integration

# \- \[ ] Full pipeline testing

# \- \[ ] Real robot deployment

# 

# \## 📚 References

# 

# \- \[Berkeley Quadruped LLM](https://arxiv.org/abs/2404.05291) - ICRA 2024

# \- \[Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)

# \- \[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

# 

# \## 📝 License

# 

# MIT License

# 

# \## 👤 Author

# 

# Mehmet Turan Yardımcı

# \- GitHub: \[@mturan33](https://github.com/mturan33)

# \- LinkedIn: \[/in/mehmetturanyardimci](https://linkedin.com/in/mehmetturanyardimci)

