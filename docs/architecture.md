# COMPOSE Architecture

> Full documentation coming Jun 20.

## System Overview

COMPOSE (Compositional Object Manipulation via Semantic Embeddings) is a Vision-Language-Action system that achieves compositional generalisation by learning disentangled semantic components rather than memorising object-action mappings.

## Pipeline

```
Natural language command
        ↓
Language Understanding (BERT + Intent Parser)
        ↓
Visual Perception (YOLO-v8 + DINOv2 + Attribute MLP)
        ↓
Semantic Scene Graph Builder
        ↓
Compositional Reasoning Engine  ←— core innovation
        ↓
Ambiguity Handler (conf < 0.7 → ask user)
        ↓
Execution (trajectory planning → goal validation)
```

## Module Details

<!-- TO COMPLETE Jun 20 -->
