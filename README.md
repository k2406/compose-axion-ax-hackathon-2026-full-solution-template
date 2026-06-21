# COMPOSE: Compositional Object Manipulation via Semantic Embeddings

**Problem Statement Number** - 1

**Problem Statement Title** - Build a Vision-Language-Action Robotic System for Natural Language Object Manipulation

**Team name** - Axion

**Team members (Names)** - Kaushal H

**Institute/College Name** - RV College of Engineering, Bengaluru, Karnataka - 560059

**Final Presentation Google Drive Link** - [FILL AFTER JUN 21]

**Full Submission Demo Video Link** - https://youtu.be/IvE0EpuIrds

**Setup & Result Reproducibility Video Link** - https://youtu.be/rhQh9Mf5F80

---

## Project Artefacts

### Technical Documentation
See the `docs/` folder:
- [Architecture](docs/architecture.md)
- [AX — Open Weight Models & Agentic AI](docs/ax.md)
- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)

### Source Code
All source code is in the `src/` folder. Run using `COMPOSE_FINAL_V4.ipynb` on Google Colab T4 GPU.

### Models Used
- DINOv2-large: https://huggingface.co/facebook/dinov2-large
- YOLOv8m: https://huggingface.co/Ultralytics/YOLOv8
- BERT-base-uncased: https://huggingface.co/google-bert/bert-base-uncased

### Models Published
Custom AttributeMLP (3-head shape/colour/size classifier trained on synthetic data) — weights generated at runtime via `src/train_attribute_mlp.py`. Not published separately as it is a lightweight adapter on top of DINOv2.

### Datasets Used
- COCO: https://cocodataset.org (YOLOv8 pretrained backbone)
- Synthetic PyBullet scenes: Generated at runtime via `src/train_attribute_mlp.py` — 800 samples, no external download required

### Datasets Published
Synthetic attribute classification dataset generated at runtime. Available via `src/train_attribute_mlp.py` under Apache 2.0 license.

---

## Attribution
Built from scratch during the Samsung EnnovateX AX Hackathon 2026. No existing open source project was used as a base. All OSS libraries used are listed in `src/requirements.txt`.

Claude (Anthropic) was used as a coding assistant during development, as permitted under the hackathon's AI development guidelines.
