# Installation Guide

## Requirements

- Google Colab with T4 GPU (recommended)
- OR local machine with Python 3.10+, NVIDIA GPU optional

---

## Option A — Google Colab (recommended)

### Step 1: Open the notebook

Upload `COMPOSE_FINAL_V4.ipynb` to [colab.research.google.com](https://colab.research.google.com)

Switch runtime: **Runtime → Change runtime type → T4 GPU → Save**

### Step 2: Run Cell 1 — Install dependencies

```python
!pip install numpy==1.26.4 -q --force-reinstall
!pip install 'gradio>=3.50,<4.0' 'transformers>=4.35.0' huggingface_hub \
             torch torchvision ultralytics pybullet matplotlib Pillow scipy -q

import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

This installs all dependencies and auto-restarts the kernel. Wait for the restart.

### Step 3: Run Cell 2 — Clone repo

```python
import os, sys
os.system('git clone https://github.com/k2406/compose-axion-ax-hackathon-2026-full-solution-template.git')
os.chdir('compose-axion-ax-hackathon-2026-full-solution-template/src')
sys.path.insert(0, os.getcwd())
```

### Step 4: Run Cell 3 — Train attribute MLP

```python
from perception import train_attribute_mlp
mlp = train_attribute_mlp(save_path='mlp_weights.pth', n_samples=800, epochs=40)
```

Expected time: ~90 seconds on T4 GPU.

### Step 5: Run Cell 4 — Load models

```python
from perception import Perceptor
perceptor = Perceptor(mlp_weights='mlp_weights.pth')
perceptor.load()  # downloads YOLO (~50MB) and DINOv2 (~1.2GB)
```

### Step 6: Run Cell 5 — Benchmark

```python
!cd /content/compose-axion.../src && python evaluate.py
```

Expected output: 6/6 PASS, TSR 100%, Novel colour gen 100%

### Step 7: Run Cell 6 — Launch GUI

```python
demo.launch(share=True)
# Prints: Running on public URL: https://xxx.gradio.live
```

---

## Option B — Local (Mac/Linux)

```bash
git clone https://github.com/k2406/compose-axion-ax-hackathon-2026-full-solution-template.git
cd compose-axion-ax-hackathon-2026-full-solution-template/src
python -m venv env && source env/bin/activate
pip install -r requirements.txt
python train_attribute_mlp.py
python evaluate.py
python app.py
# Open http://localhost:7860
```

---

## Dependencies (requirements.txt)

```
gradio>=3.50,<4.0
transformers>=4.35.0
huggingface_hub
torch==2.3.0
torchvision==0.18.0
ultralytics==8.2.0
pybullet==3.2.6
matplotlib==3.9.0
numpy==1.26.4
Pillow==10.4.0
scipy==1.13.0
```

---

## Common Issues

**numpy version conflict:** Always run Cell 1 in a fresh kernel session. The `--force-reinstall` flag and kernel restart ensure numpy 1.26.4 is loaded cleanly.

**Shape.BLOCK AttributeError:** Run this fix before Cell 3:
```python
!sed -i 's/Shape\.BLOCK/Shape.CUBE/g' src/perception.py src/simulation/pybullet_env.py
```

**PyBullet falls back to matplotlib:** This is normal in Colab environments without display. The system automatically falls back to matplotlib rendering — all functionality is preserved.

**Gradio URL not appearing:** Ensure `share=True` is set in `demo.launch()`. The public URL expires after 72 hours.
