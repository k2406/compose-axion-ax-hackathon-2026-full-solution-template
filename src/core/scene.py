"""
COMPOSE - Core Scene & Object Model
Handles all object state, scene graph construction, and spatial relationships.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ── Vocabulary ──────────────────────────────────────────────────────────────

class Shape(str, Enum):
    CUBE      = "cube"
    CYLINDER  = "cylinder"
    SPHERE    = "sphere"
    CONTAINER = "container"

class Color(str, Enum):
    RED    = "red"
    BLUE   = "blue"
    GREEN  = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    CYAN   = "cyan"      # unseen in "training"
    ORANGE = "orange"    # unseen in "training"
    PINK   = "pink"      # unseen in "training"

class Size(str, Enum):
    SMALL  = "small"
    MEDIUM = "medium"
    LARGE  = "large"

COLOR_HEX = {
    Color.RED:    "#e74c3c",
    Color.BLUE:   "#2980b9",
    Color.GREEN:  "#27ae60",
    Color.YELLOW: "#f1c40f",
    Color.PURPLE: "#8e44ad",
    Color.CYAN:   "#16a085",
    Color.ORANGE: "#e67e22",
    Color.PINK:   "#e91e8c",
}

SIZE_PX = {
    Size.SMALL:  32,
    Size.MEDIUM: 44,
    Size.LARGE:  58,
}

# ── Disentangled Embedding Vectors ──────────────────────────────────────────
# Each attribute is a separate embedding space — this IS the core innovation.

SHAPE_EMBED = {
    Shape.CUBE:      np.array([1.0, 0.0, 0.0, 0.0]),
    Shape.CYLINDER:  np.array([0.0, 1.0, 0.0, 0.0]),
    Shape.SPHERE:    np.array([0.0, 0.0, 1.0, 0.0]),
    Shape.CONTAINER: np.array([0.0, 0.0, 0.0, 1.0]),
}

COLOR_EMBED = {
    Color.RED:    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    Color.BLUE:   np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    Color.GREEN:  np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    Color.YELLOW: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    Color.PURPLE: np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    # Novel colours — compositional blends, never in "training"
    Color.CYAN:   np.array([0.0, 0.4, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0]),
    Color.ORANGE: np.array([0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 1.0, 0.0]),
    Color.PINK:   np.array([0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.0]),
}

SIZE_EMBED = {
    Size.SMALL:  np.array([1.0, 0.0, 0.0]),
    Size.MEDIUM: np.array([0.0, 1.0, 0.0]),
    Size.LARGE:  np.array([0.0, 0.0, 1.0]),
}

KNOWN_COLORS = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.PURPLE}
NOVEL_COLORS = {Color.CYAN, Color.ORANGE, Color.PINK}


# ── Object ───────────────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    obj_id: str
    shape:  Shape
    color:  Color
    size:   Size
    x:      float
    y:      float

    @property
    def px_size(self) -> int:
        return SIZE_PX[self.size]

    @property
    def color_hex(self) -> str:
        return COLOR_HEX[self.color]

    @property
    def label(self) -> str:
        return f"{self.color.value} {self.shape.value}"

    @property
    def embedding(self) -> np.ndarray:
        return np.concatenate([
            SHAPE_EMBED[self.shape],
            COLOR_EMBED[self.color],
            SIZE_EMBED[self.size],
        ])

    def is_novel(self) -> bool:
        return self.color in NOVEL_COLORS

    def bbox(self):
        h = self.px_size
        return (self.x - h/2, self.y - h/2, self.x + h/2, self.y + h/2)


# ── Scene ────────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    objects:  list[SceneObject] = field(default_factory=list)
    canvas_w: float = 640.0
    canvas_h: float = 420.0

    def get(self, obj_id: str) -> Optional[SceneObject]:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        return None

    def spatial_relation(self, a: SceneObject, b: SceneObject) -> str:
        dx = a.x - b.x
        dy = a.y - b.y
        if abs(dx) > abs(dy):
            return "right_of" if dx > 0 else "left_of"
        return "below" if dy > 0 else "above"

    def edges(self) -> list[dict]:
        result = []
        for i, a in enumerate(self.objects):
            for b in self.objects[i+1:]:
                dist = np.hypot(a.x - b.x, a.y - b.y)
                rel  = self.spatial_relation(a, b)
                result.append({
                    "from": a.obj_id, "to": b.obj_id,
                    "relation": rel, "distance": round(dist, 1)
                })
        return result

    def next_obj_id(self) -> str:
        existing = {o.obj_id for o in self.objects}
        i = 0
        while f"obj_{i}" in existing:
            i += 1
        return f"obj_{i}"

    def free_position(self) -> tuple[float, float]:
        """Find a position that doesn't heavily overlap existing objects."""
        import random
        margin = 60
        for _ in range(50):
            x = random.uniform(margin, self.canvas_w - margin)
            y = random.uniform(margin, self.canvas_h - margin)
            too_close = any(
                np.hypot(o.x - x, o.y - y) < 80
                for o in self.objects
            )
            if not too_close:
                return x, y
        return (
            random.uniform(margin, self.canvas_w - margin),
            random.uniform(margin, self.canvas_h - margin),
        )

    def summary(self) -> str:
        lines = [f"Scene: {len(self.objects)} objects"]
        for o in self.objects:
            novel_tag = " [NOVEL]" if o.is_novel() else ""
            lines.append(
                f"  {o.obj_id}: {o.label} ({o.size.value}){novel_tag} @ ({int(o.x)},{int(o.y)})"
            )
        return "\n".join(lines)


# ── Default scene ─────────────────────────────────────────────────────────────

def make_default_scene() -> Scene:
    return Scene(objects=[
        SceneObject("obj_0", Shape.CUBE,      Color.RED,    Size.MEDIUM, x=140, y=160),
        SceneObject("obj_1", Shape.CUBE,      Color.BLUE,   Size.LARGE,  x=320, y=200),
        SceneObject("obj_2", Shape.CYLINDER,  Color.GREEN,  Size.SMALL,  x=480, y=140),
        SceneObject("obj_3", Shape.CONTAINER, Color.YELLOW, Size.LARGE,  x=240, y=310),
        SceneObject("obj_4", Shape.SPHERE,    Color.CYAN,   Size.SMALL,  x=420, y=310),
        SceneObject("obj_5", Shape.CONTAINER, Color.PURPLE, Size.MEDIUM, x=530, y=280),
    ])
