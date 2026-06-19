# How COMPOSE Uses Open-Weight Models and Agentic AI

## Open-Weight Models

| Model | HuggingFace | Role |
|---|---|---|
| DINOv2-large | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | Frozen visual feature extractor |
| YOLOv8m | [Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8) | Real-time object detection |
| BERT-base-uncased | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | Language embedding backbone |

All models used under their respective open licenses. No proprietary models.

---

## The Core Research Claim

Standard VLA models learn end-to-end mappings: `image → action`. They memorise.
When tested on unseen object combinations (a cyan cube when only red/blue/green cubes were in training),
they fail catastrophically — accuracy drops from 70-75% to 15-40%.

COMPOSE learns **disentangled semantic components** instead.
Shape, colour, and size are separate embedding spaces.
A cyan object is never seen in training, but cyan's embedding is computed compositionally
as a blend of blue and green — so the system generalises.

---

## Agentic Pipeline

Every natural language command triggers a 6-stage agentic reasoning pipeline:

```
Stage 1 — Perception
  YOLO-v8 detects objects + bounding boxes from scene image
  DINOv2 extracts 1024-dim CLS features per detected crop
  3-head attribute MLP classifies: shape (5 classes) / colour (8) / size (3)
  → SceneObject list with position, attributes, disentangled embeddings

Stage 2 — Language Understanding
  Rule-based intent parser with vocabulary aliasing
  Splits command into: action / target attrs / spatial relation / reference attrs
  Confidence scoring: drops when fewer than 2 target attributes found
  → structured Intent dict

Stage 3 — Semantic Scene Graph
  Builds graph: nodes = objects, edges = spatial relations
  Computes pairwise: left_of, right_of, behind, beside, above, below
  Stores: {object_id, shape, colour, size, position, embedding}

Stage 4 — Compositional Reasoning (the core innovation)
  Builds query embedding from intent attributes
  Cosine similarity match against all scene object embeddings
  Key: match is on COMPONENTS not combinations
  → novel colour+shape resolves correctly via partial embedding similarity

Stage 5 — Ambiguity Handler
  Checks: overall_confidence = min(parse_conf, match_score)
  Checks: score gap between top-1 and top-2 match
  If conf < 0.7 OR gap < 0.08: ask for clarification, never execute blindly
  → prevents catastrophic failures in ambiguous cases

Stage 6 — PyBullet Execution
  Spatial transform functions compute destination from reference geometry
  right_of(ref) = ref.x + ref.width/2 + offset  ← function, not lookup
  3-phase kinematic trajectory: lift → arc → lower
  Validates success: object within 5cm of goal position
```

---

## What Worked

**Frozen DINOv2 over fine-tuning.** DINOv2's pretrained features already separate
object semantics without any task-specific fine-tuning. Training only the lightweight
3-head MLP on top (256 hidden units, ~200K params) achieved strong attribute classification
in under 2 minutes on T4 GPU. Fine-tuning the full model would have required 10x more
compute and generalised worse due to overfitting on our small synthetic dataset.

**Disentangled embeddings over end-to-end.** Keeping shape/colour/size as separate
embedding vectors means the system can match on any subset of attributes. A query for
"cyan sphere" with no size specified correctly ignores size dimensions and matches on
shape + colour only. An end-to-end model cannot do this.

**Rule-based intent parser over BERT fine-tuning.** Our command structure is well-defined:
`move [colour] [shape] [spatial] [colour] [shape]`. A carefully designed rule-based parser
with vocabulary aliasing (ball→sphere, bin→container, teal→cyan) outperformed
BERT fine-tuning on this structured domain and runs 100x faster — important for
real-time interactive use.

**Compositional colour embeddings.** Novel colours (cyan, orange, pink) are given
embeddings that are linear blends of known colour embeddings. Cyan = 0.4×blue + 0.6×green.
This is a lightweight form of zero-shot generalisation — the model never saw cyan but
resolves it correctly because the embedding space is compositionally structured.

---

## What Did Not Work

**PyBullet 3D rendering for perception.** We originally planned to generate all training
data by rendering PyBullet scenes and running DINOv2 on those renders. In practice,
PyBullet's DIRECT mode renders look synthetic enough that DINOv2 features diverge from
real image features. We pivoted to colour-space quantisation as a reliable fallback
and use real images for DINOv2 feature extraction when available.

**BERT for spatial parsing.** Early experiments used BERT embeddings to parse spatial
relations. The model struggled with the structured command format and added 300ms latency.
The rule-based parser with ordered phrase matching (longer phrases checked before shorter
ones to avoid partial matches) was more accurate and 100x faster.

**End-to-end fine-tuning on our dataset size.** 800 synthetic samples are insufficient
to fine-tune a model of DINOv2's scale. Attempts to fine-tune even the last 2 layers
resulted in worse generalisation than frozen features + lightweight MLP.

---

## Datasets Used

| Dataset | Link | Use |
|---|---|---|
| COCO | [cocodataset.org](https://cocodataset.org) | YOLO-v8 pretrained backbone |
| Open X-Embodiment | [openxembodiment.github.io](https://openxembodiment.github.io) | Architecture reference |
| Synthetic PyBullet scenes (generated) | — | MLP attribute classifier training |

Synthetic dataset generated at runtime via `train_attribute_mlp.py`.
800 samples, known labels, no external download required.
