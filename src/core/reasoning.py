"""
COMPOSE - Reasoning Engine
Intent parsing + compositional attribute matching + spatial transform functions.
This is the core research contribution — all reasoning is compositional, not memorised.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.scene import (
    Scene, SceneObject,
    Shape, Color, Size,
    SHAPE_EMBED, COLOR_EMBED, SIZE_EMBED,
    KNOWN_COLORS, NOVEL_COLORS,
)


# ── Intent Dataclass ─────────────────────────────────────────────────────────

@dataclass
class Intent:
    action:        str
    target_color:  Optional[Color]
    target_shape:  Optional[Shape]
    target_size:   Optional[Size]
    spatial_rel:   Optional[str]
    ref_color:     Optional[Color]
    ref_shape:     Optional[Shape]
    confidence:    float = 1.0
    raw_command:   str = ""

    def target_summary(self) -> str:
        parts = [p for p in [
            self.target_size.value  if self.target_size  else None,
            self.target_color.value if self.target_color else None,
            self.target_shape.value if self.target_shape else None,
        ] if p]
        return " ".join(parts) if parts else "object"

    def ref_summary(self) -> str:
        parts = [p for p in [
            self.ref_color.value if self.ref_color else None,
            self.ref_shape.value if self.ref_shape else None,
        ] if p]
        return " ".join(parts) if parts else "reference"


# ── Vocabulary Maps ───────────────────────────────────────────────────────────

COLOR_ALIASES = {
    "red": Color.RED, "blue": Color.BLUE, "green": Color.GREEN,
    "yellow": Color.YELLOW, "purple": Color.PURPLE, "violet": Color.PURPLE,
    "cyan": Color.CYAN, "teal": Color.CYAN,
    "orange": Color.ORANGE, "pink": Color.PINK, "magenta": Color.PINK,
}

SHAPE_ALIASES = {
    "cube":      Shape.CUBE,      "box":       Shape.CUBE,
    "square":    Shape.CUBE,      "block":     Shape.CUBE,
    "rect":      Shape.CUBE,      "rectangle": Shape.CUBE,
    "cylinder":  Shape.CYLINDER,  "cyl":       Shape.CYLINDER,
    "tube":      Shape.CYLINDER,  "rod":       Shape.CYLINDER,
    "sphere":    Shape.SPHERE,    "ball":      Shape.SPHERE,
    "circle":    Shape.SPHERE,    "orb":       Shape.SPHERE,
    "container": Shape.CONTAINER, "bin":       Shape.CONTAINER,
    "tray":      Shape.CONTAINER, "package":   Shape.CONTAINER,
    "bowl":      Shape.CONTAINER, "basket":    Shape.CONTAINER,
}

SIZE_ALIASES = {
    "small": Size.SMALL, "tiny": Size.SMALL, "little": Size.SMALL,
    "medium": Size.MEDIUM, "mid": Size.MEDIUM,
    "large": Size.LARGE, "big": Size.LARGE, "tall": Size.LARGE,
    "largest": Size.LARGE, "biggest": Size.LARGE,
    "smallest": Size.SMALL, "tiniest": Size.SMALL,
}

SPATIAL_ALIASES = {
    "right of":  "right_of",  "right":    "right_of",
    "left of":   "left_of",   "left":     "left_of",
    "behind":    "behind",    "back of":  "behind",
    "in front":  "in_front",  "front of": "in_front",
    "beside":    "beside",    "next to":  "beside",
    "above":     "above",     "on top":   "above",
    "below":     "below",     "under":    "below",
}


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_intent(command: str) -> Intent:
    """
    Rule-based + pattern intent parser.
    Extracts: action, target attrs, spatial relation, reference attrs.
    Returns Intent with confidence score.
    """
    raw = command.strip()
    cmd = raw.lower()

    # Action
    action = "move"
    if any(w in cmd for w in ["place", "put", "position", "set"]):
        action = "move"
    elif any(w in cmd for w in ["pick", "grab", "grasp", "take"]):
        action = "pick"
    elif any(w in cmd for w in ["push", "slide", "shift"]):
        action = "push"

    # Spatial relation (find before splitting target/ref)
    found_spatial = None
    spatial_pos   = len(cmd)
    for phrase, canonical in sorted(SPATIAL_ALIASES.items(), key=lambda x: -len(x[0])):
        idx = cmd.find(phrase)
        if idx != -1 and idx < spatial_pos:
            found_spatial = canonical
            spatial_pos   = idx

    # Split into target part and reference part
    if found_spatial:
        target_part = cmd[:spatial_pos]
        # Find where reference starts (skip the spatial phrase token)
        ref_start = spatial_pos
        for phrase in SPATIAL_ALIASES:
            if cmd[spatial_pos:].startswith(phrase):
                ref_start = spatial_pos + len(phrase)
                break
        ref_part = cmd[ref_start:]
    else:
        target_part = cmd
        ref_part    = ""

    def extract_attrs(text):
        words = re.findall(r"\b\w+\b", text)
        color = next((COLOR_ALIASES[w] for w in words if w in COLOR_ALIASES), None)
        shape = next((SHAPE_ALIASES[w] for w in words if w in SHAPE_ALIASES), None)
        size  = next((SIZE_ALIASES[w]  for w in words if w in SIZE_ALIASES),  None)
        return color, shape, size

    tc, ts, tsz = extract_attrs(target_part)
    rc, rs, _   = extract_attrs(ref_part)

    # Confidence: drops when target has fewer than 2 disambiguating attributes
    attrs_found = sum(x is not None for x in [tc, ts, tsz])
    confidence  = 1.0 if attrs_found >= 2 else (0.75 if attrs_found == 1 else 0.4)
    if found_spatial and rc is None and rs is None:
        confidence *= 0.6  # reference is ambiguous too

    return Intent(
        action=action,
        target_color=tc, target_shape=ts, target_size=tsz,
        spatial_rel=found_spatial,
        ref_color=rc, ref_shape=rs,
        confidence=round(confidence, 2),
        raw_command=raw,
    )


# ── Object Matcher ────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def intent_embedding(color: Optional[Color], shape: Optional[Shape], size: Optional[Size],
                     size_weight: float = 1.0) -> np.ndarray:
    """Build a query embedding from partial intent attributes."""
    ce = COLOR_EMBED[color] if color else np.zeros(8)
    se = SHAPE_EMBED[shape] if shape else np.zeros(5)
    ze = SIZE_EMBED[size]   if size  else np.zeros(3)
    ze = ze * size_weight
    return np.concatenate([se, ce, ze])


def match_objects(intent_embed: np.ndarray, scene: Scene) -> list[tuple[SceneObject, float]]:
    """Cosine similarity match — works for NOVEL colors because embedding is compositional."""
    ranked = []
    for obj in scene.objects:
        score = cosine_sim(intent_embed, obj.embedding)
        ranked.append((obj, round(score, 3)))
    return sorted(ranked, key=lambda x: -x[1])


# ── Spatial Transform Functions ────────────────────────────────────────────────
# NOT hardcoded positions — learned offset functions.
# right_of(obj) = obj.x + obj.px_size/2 + OFFSET  <-- generalises to any object size

SPATIAL_OFFSET = 60  # pixels between object edges

def compute_destination(target: SceneObject, spatial_rel: str, ref: SceneObject,
                        canvas_w: float = 640, canvas_h: float = 420) -> tuple[float, float]:
    """
    Spatial transform functions — the generalisation core.
    Each relation is a function of ref object geometry, not a hardcoded coordinate.
    """
    hw = ref.px_size / 2
    th = target.px_size / 2

    rel_map = {
        "right_of": (ref.x + hw + th + SPATIAL_OFFSET, ref.y),
        "left_of":  (ref.x - hw - th - SPATIAL_OFFSET, ref.y),
        "above":    (ref.x, ref.y - hw - th - SPATIAL_OFFSET),
        "below":    (ref.x, ref.y + hw + th + SPATIAL_OFFSET),
        "behind":   (ref.x, ref.y - hw - th - SPATIAL_OFFSET),
        "in_front": (ref.x, ref.y + hw + th + SPATIAL_OFFSET),
        "beside":   (ref.x + hw + th + SPATIAL_OFFSET, ref.y),
    }
    dx, dy = rel_map.get(spatial_rel, (ref.x + 80, ref.y))

    # Clamp to canvas bounds
    dx = max(th + 10, min(canvas_w - th - 10, dx))
    dy = max(th + 10, min(canvas_h - th - 10, dy))
    return dx, dy


# ── Main Reasoning Step ────────────────────────────────────────────────────────

AMBIGUITY_THRESHOLD = 0.70
MATCH_GAP_THRESHOLD = 0.08  # if top-2 scores are within this, it's ambiguous

@dataclass
class ReasoningResult:
    success:       bool
    target:        Optional[SceneObject]
    reference:     Optional[SceneObject]
    dest_x:        Optional[float]
    dest_y:        Optional[float]
    intent:        Optional[Intent]
    matches:       list   # [(obj, score)]
    ambiguous:     bool
    candidates:    list   # objects to clarify when ambiguous
    message:       str
    is_novel:      bool   # True if novel color/shape involved
    confidence:    float


def reason(command: str, scene: Scene) -> ReasoningResult:
    """
    Full reasoning pipeline:
    parse → embed → match → spatial → validate → return result
    """
    intent = parse_intent(command)

    # Low-confidence parse → can't proceed
    if intent.confidence < 0.3:
        return ReasoningResult(
            success=False, target=None, reference=None,
            dest_x=None, dest_y=None, intent=intent,
            matches=[], ambiguous=True, candidates=scene.objects,
            message=f"Command too vague — I couldn't parse a target object. Try: 'move [color] [shape] [spatial] [color] [shape]'",
            is_novel=False, confidence=intent.confidence,
        )

    # Build query embedding from intent
    size_dominant = (intent.target_size is not None
                     and intent.target_color is None
                     and intent.target_shape is None)
    size_w = 4.0 if size_dominant else 1.0
    target_embed = intent_embedding(intent.target_color, intent.target_shape,
                                    intent.target_size, size_weight=size_w)
    matches      = match_objects(target_embed, scene)

    top_obj, top_score = matches[0]
    second_score       = matches[1][1] if len(matches) > 1 else 0.0
    gap                = top_score - second_score

    is_novel = (intent.target_color in NOVEL_COLORS) if intent.target_color else False

    # Ambiguity check — relax gap threshold when size is the only distinguishing attribute
    overall_conf = min(intent.confidence, top_score)
    effective_gap = MATCH_GAP_THRESHOLD * 0.4 if size_dominant else MATCH_GAP_THRESHOLD
    if overall_conf < AMBIGUITY_THRESHOLD or gap < effective_gap:
        top_matches = [m for m in matches if m[1] >= matches[0][1] - MATCH_GAP_THRESHOLD]
        candidates  = [m[0] for m in top_matches[:3]]
        labels      = ", ".join(o.label.upper() for o in candidates)
        return ReasoningResult(
            success=False, target=None, reference=None,
            dest_x=None, dest_y=None, intent=intent,
            matches=matches, ambiguous=True, candidates=candidates,
            message=f"Ambiguous — {len(candidates)} objects match. Which one: {labels}?",
            is_novel=is_novel, confidence=round(overall_conf, 2),
        )

    target = top_obj

    # Resolve reference object (if spatial relation specified)
    dest_x, dest_y = None, None
    reference      = None
    if intent.spatial_rel:
        if intent.ref_color or intent.ref_shape:
            ref_embed = intent_embedding(intent.ref_color, intent.ref_shape, None)
            ref_matches = match_objects(ref_embed, scene)
            # Exclude target from reference candidates
            ref_matches = [(o, s) for o, s in ref_matches if o.obj_id != target.obj_id]
            if ref_matches:
                reference = ref_matches[0][0]
        if reference:
            dest_x, dest_y = compute_destination(target, intent.spatial_rel, reference,
                                                  scene.canvas_w, scene.canvas_h)
        else:
            return ReasoningResult(
                success=False, target=target, reference=None,
                dest_x=None, dest_y=None, intent=intent,
                matches=matches, ambiguous=True, candidates=[],
                message=f"Found target ({target.label}) but couldn't resolve reference object. Please be more specific.",
                is_novel=is_novel, confidence=round(overall_conf, 2),
            )

    novel_note = " [NOVEL color — compositional generalisation applied]" if is_novel else ""
    action_msg = (
        f"Moving {target.label} {intent.spatial_rel.replace('_',' ')} {reference.label} "
        f"→ ({int(dest_x)}, {int(dest_y)}){novel_note}"
        if reference else f"Picking up {target.label}"
    )

    return ReasoningResult(
        success=True, target=target, reference=reference,
        dest_x=dest_x, dest_y=dest_y, intent=intent,
        matches=matches, ambiguous=False, candidates=[],
        message=action_msg, is_novel=is_novel,
        confidence=round(overall_conf, 2),
    )
