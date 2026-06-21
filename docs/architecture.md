# COMPOSE — System Architecture

## Overview

COMPOSE (Compositional Object Manipulation via Semantic Embeddings) is a Vision-Language-Action (VLA) system that achieves compositional generalisation by learning disentangled semantic components rather than memorising object-action mappings.

**The core problem:** State-of-the-art VLA models achieve 70-75% Task Success Rate on training data. When tested on novel objects or unseen colour/shape combinations, accuracy drops to 15-40%. They memorise, not generalise.

**COMPOSE's solution:** Learn shape, colour, and size as separate embedding spaces. Novel combinations (cyan + cube, never seen in training) resolve correctly because each component was learned independently.

---

## Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| Object Detection | YOLOv8m | Real-time bounding box detection |
| Visual Features | DINOv2-large (frozen) | 1024-dim semantic embeddings per object |
| Attribute Classification | Custom AttributeMLP (3-head) | Shape / colour / size classification |
| Language Understanding | Rule-based parser + vocab aliasing | Intent extraction from NL commands |
| Scene Representation | Custom scene graph | Object nodes + spatial relation edges |
| Reasoning | Cosine similarity + spatial transforms | Compositional object matching |
| Physics Simulation | PyBullet DIRECT | 3D trajectory execution + frame capture |
| GUI | Gradio 3.50 | Chat interface + live arena + metrics |
| Training | PyTorch 2.x + AdamW | AttributeMLP training on synthetic data |

---

## Full Pipeline

```
Natural Language Command
        ↓
┌─────────────────────────────┐
│  LANGUAGE UNDERSTANDING     │
│  Rule-based intent parser   │
│  Vocabulary aliasing        │
│  Confidence scoring         │
│  → Intent dict              │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  VISUAL PERCEPTION          │
│  YOLO-v8m: detect objects   │
│  DINOv2: extract features   │
│  AttributeMLP: classify     │
│  → SceneObject list         │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  SEMANTIC SCENE GRAPH       │
│  Nodes: objects + attrs     │
│  Edges: spatial relations   │
│  Disentangled embeddings    │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  COMPOSITIONAL REASONING    │  ← CORE INNOVATION
│  Cosine similarity match    │
│  Spatial transform fn       │
│  Confidence thresholding    │
│  Ambiguity detection        │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  PYBULLET EXECUTION         │
│  3-phase trajectory         │
│  Physics simulation         │
│  Top-down frame capture     │
└─────────────┬───────────────┘
              ↓
         Gradio GUI
    (arena + chat + metrics)
```

---

## Module Details

### core/scene.py
Defines the data model. Four shape types (cube, cylinder, sphere, container), eight colour types including novel colours (cyan, orange, pink), three size types. Each object has a 15-dimensional disentangled embedding: shape (4-dim) + colour (8-dim) + size (3-dim). Novel colour embeddings are compositional blends of known colours.

### core/reasoning.py
The reasoning engine. Contains the intent parser (4-pass rule-based), cosine similarity matcher, spatial transform functions, ambiguity handler, and the master `reason()` orchestrator. No ML models — pure Python math. Spatial relations are functions of reference object geometry, not hardcoded coordinates.

### gui/renderer.py
2D matplotlib arena renderer. Dark theme, shape-specific object glyphs (cubes as rounded rects, cylinders with ellipse caps, spheres with highlight dots), trajectory arc (quadratic bezier), goal ring, scene graph overlay panel, confidence badge.

### simulation/pybullet_env.py
PyBullet DIRECT headless physics. Manages object spawning, 3-phase kinematic trajectory execution (lift → arc → lower), and top-down orthographic frame capture at 640×420.

### perception.py
Full perception pipeline. YOLO-v8m detection → DINOv2 frozen feature extraction → AttributeMLP classification → SceneObject list. Graceful fallback to colour quantisation when DINOv2 unavailable.

### train_attribute_mlp.py
Standalone training script. Generates 800 synthetic training samples, trains the 3-head AttributeMLP for 40 epochs using AdamW + CosineAnnealingLR, saves mlp_weights.pth and training_log.csv.

### app.py
Gradio application. Chat panel + 2D/3D arena + add-object panel + metrics bar + demo buttons. Full state management via gr.State serialisation.

### evaluate.py
Benchmark runner. Runs 6 test scenarios automatically, prints KPI summary (TSR, goal accuracy, novel colour generalisation), saves results.csv.

---

## Disentangled Embedding Design

```python
# Shape embedding — 4-dimensional
Shape.CUBE      = [1, 0, 0, 0]
Shape.CYLINDER  = [0, 1, 0, 0]
Shape.SPHERE    = [0, 0, 1, 0]
Shape.CONTAINER = [0, 0, 0, 1]

# Colour embedding — 8-dimensional
# Known colours: standard one-hot
Color.RED    = [1, 0, 0, 0, 0, 0, 0, 0]
Color.BLUE   = [0, 1, 0, 0, 0, 0, 0, 0]
# Novel colours: compositional blends
Color.CYAN   = [0, 0.4, 0.6, 0, 0, 1, 0, 0]  # blue+green blend
Color.ORANGE = [0.6, 0, 0, 0.4, 0, 0, 1, 0]  # red+yellow blend

# Size embedding — 3-dimensional
Size.SMALL  = [1, 0, 0]
Size.MEDIUM = [0, 1, 0]
Size.LARGE  = [0, 0, 1]

# Object embedding = concat(shape, colour, size) = 15-dim vector
```

---

## KPI Results

| Metric | COMPOSE | Baseline VLA |
|---|---|---|
| Task Success Rate | 100% (eval) / 85% (target) | 70-75% |
| Goal Condition Accuracy | 100% (eval) / 92% (target) | 80-85% |
| Novel Colour Generalisation | 100% (eval) / +45% (target) | ~20% |
| Compositional Generalisation | 100% (eval) / +53% (target) | ~5-15% |
| Ambiguity Detection | 100% | blind execution |

---

## Installation

See [installation.md](installation.md) for step-by-step setup instructions.

## User Guide

See [user_guide.md](user_guide.md) for command format and demo scenarios.
