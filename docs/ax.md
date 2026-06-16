# How COMPOSE Uses Open-Weight Models and Agentic AI

> **Note:** This document will be completed by Jun 20. Outline below.

## Open-Weight Models Used

| Model | HuggingFace Link | Role |
|---|---|---|
| DINOv2-large | https://huggingface.co/facebook/dinov2-large | Semantic visual embeddings |
| YOLOv8m | https://huggingface.co/Ultralytics/YOLOv8 | Real-time object detection |
| BERT-base-uncased | https://huggingface.co/google-bert/bert-base-uncased | Language understanding |

## Agentic AI Architecture

<!-- TO COMPLETE Jun 20 -->
- Reasoning pipeline
- Tool chaining: perception → language → scene graph → execution
- Confidence-based ambiguity handling
- What worked / what didn't

## Key Technical Decisions

<!-- TO COMPLETE Jun 20 -->
- Why disentangled embeddings over end-to-end VLA
- Why frozen DINOv2 over fine-tuning
- Why rule-based intent parser over full BERT fine-tune
