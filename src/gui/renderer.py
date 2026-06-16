"""
COMPOSE - Arena Renderer
Draws the 2D top-down workspace using matplotlib.
Shows objects, trajectory, goal ring, scene graph overlay.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as pe

from core.scene import Scene, SceneObject, COLOR_HEX, SIZE_PX
from core.reasoning import ReasoningResult


# ── Palette ───────────────────────────────────────────────────────────────────
BG_COLOR      = "#0d0f1a"
GRID_COLOR    = "#1a1d2e"
GRID_ALPHA    = 0.6
TEXT_COLOR    = "#e8eaf0"
MUTED_COLOR   = "#6b7280"
ACCENT_ARROW  = "#f59e0b"
GOAL_COLOR    = "#f59e0b"
SHADOW_COLOR  = "#000000"


def draw_scene(
    scene: Scene,
    result: ReasoningResult = None,
    trajectory_frames: int = 0,   # 0 = not animating, 1–10 = progress
    canvas_w: float = 640,
    canvas_h: float = 420,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(8, 5.25))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── Grid ──────────────────────────────────────────────────────────────────
    for x in range(0, int(canvas_w) + 1, 80):
        ax.axvline(x, color=GRID_COLOR, linewidth=0.5, alpha=GRID_ALPHA)
    for y in range(0, int(canvas_h) + 1, 60):
        ax.axhline(y, color=GRID_COLOR, linewidth=0.5, alpha=GRID_ALPHA)

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)   # Flip y so top-left is origin
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Objects ───────────────────────────────────────────────────────────────
    for obj in scene.objects:
        _draw_object(ax, obj, result)

    # ── Trajectory + Goal ─────────────────────────────────────────────────────
    if result and result.success and result.dest_x is not None:
        tx, ty = result.target.x, result.target.y
        dx, dy = result.dest_x, result.dest_y

        # Dashed trajectory arc
        t_vals = np.linspace(0, 1, 40)
        ctrl_x = (tx + dx) / 2
        ctrl_y = min(ty, dy) - 60   # arc upward
        arc_x = (1 - t_vals)**2 * tx + 2*(1-t_vals)*t_vals*ctrl_x + t_vals**2*dx
        arc_y = (1 - t_vals)**2 * ty + 2*(1-t_vals)*t_vals*ctrl_y + t_vals**2*dy
        ax.plot(arc_x, arc_y, color=ACCENT_ARROW, linewidth=1.5,
                linestyle="--", alpha=0.7, zorder=5)

        # Arrow head at destination
        ax.annotate("", xy=(dx, dy), xytext=(arc_x[-3], arc_y[-3]),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_ARROW,
                                    lw=1.5, mutation_scale=14), zorder=6)

        # Goal ring
        goal_ring = Circle((dx, dy), result.target.px_size / 2 + 6,
                            fill=False, edgecolor=GOAL_COLOR,
                            linewidth=1.5, linestyle="--", alpha=0.8, zorder=4)
        ax.add_patch(goal_ring)

        # Goal label
        ax.text(dx, dy - result.target.px_size/2 - 14, "GOAL",
                ha="center", va="center", fontsize=7, color=GOAL_COLOR,
                fontfamily="monospace", alpha=0.9)

    # ── Scene graph overlay (bottom-left) ─────────────────────────────────────
    _draw_scene_graph(ax, scene, canvas_w, canvas_h)

    # ── Confidence badge (top-right) ──────────────────────────────────────────
    if result:
        conf_color = "#22c55e" if result.confidence >= 0.7 else "#f59e0b"
        label = f"conf {result.confidence:.2f}"
        if result.is_novel:
            label += "  NOVEL"
        ax.text(canvas_w - 8, 12, label, ha="right", va="top",
                fontsize=8, color=conf_color, fontfamily="monospace")

    # ── Title bar ─────────────────────────────────────────────────────────────
    ax.text(8, 8, "COMPOSE  ·  workspace", ha="left", va="top",
            fontsize=8, color=MUTED_COLOR, fontfamily="monospace")

    plt.tight_layout(pad=0)
    return fig


def _draw_object(ax, obj: SceneObject, result: ReasoningResult = None):
    """Draw a single object with shape-appropriate glyph."""
    x, y   = obj.x, obj.y
    sz     = obj.px_size
    half   = sz / 2
    color  = obj.color_hex
    is_target = result and result.target and result.target.obj_id == obj.obj_id
    is_ref    = result and result.reference and result.reference.obj_id == obj.obj_id

    border_color = "#ffffff" if is_target else ("#aaaaff" if is_ref else "#00000044")
    border_lw    = 2.0 if (is_target or is_ref) else 0.5
    alpha        = 1.0

    from core.scene import Shape
    shape = obj.shape

    if shape in (Shape.CUBE, Shape.BLOCK, Shape.CONTAINER):
        rx = 4 if shape == Shape.CUBE else (2 if shape == Shape.BLOCK else 6)
        box = FancyBboxPatch(
            (x - half, y - half), sz, sz,
            boxstyle=f"round,pad=0,rounding_size={rx}",
            facecolor=color, edgecolor=border_color,
            linewidth=border_lw, alpha=alpha, zorder=3,
        )
        ax.add_patch(box)

    elif shape == Shape.CYLINDER:
        rect = mpatches.Rectangle((x - half * 0.6, y - half), sz * 0.6, sz,
                                   facecolor=color, edgecolor=border_color,
                                   linewidth=border_lw, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        top = mpatches.Ellipse((x, y - half), sz * 0.6, sz * 0.2,
                                facecolor=_lighten(color), edgecolor=border_color,
                                linewidth=border_lw * 0.5, alpha=alpha, zorder=4)
        ax.add_patch(top)

    elif shape == Shape.SPHERE:
        circ = Circle((x, y), half,
                      facecolor=color, edgecolor=border_color,
                      linewidth=border_lw, alpha=alpha, zorder=3)
        ax.add_patch(circ)
        # Highlight dot
        hi = Circle((x - half * 0.25, y - half * 0.25), half * 0.2,
                    facecolor="white", alpha=0.25, zorder=4)
        ax.add_patch(hi)

    # Label
    label_lines = obj.color.value[:3].upper() + "\n" + obj.shape.value[:3].upper()
    ax.text(x, y, label_lines, ha="center", va="center",
            fontsize=max(5, sz * 0.18), color="white", fontweight="bold",
            fontfamily="monospace", zorder=5,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    # Novel badge
    if obj.is_novel():
        ax.text(x + half - 2, y - half + 2, "N", ha="right", va="top",
                fontsize=6, color="#f59e0b", fontfamily="monospace", zorder=6)


def _draw_scene_graph(ax, scene: Scene, canvas_w: float, canvas_h: float):
    """Compact scene graph panel in bottom-left corner."""
    px, py = 10, canvas_h - 10
    line_h = 12
    panel_h = len(scene.objects) * line_h + 24
    panel_w = 165

    bg = FancyBboxPatch((px - 2, py - panel_h), panel_w, panel_h,
                         boxstyle="round,pad=2", facecolor="#0d0f1acc",
                         edgecolor="#ffffff22", linewidth=0.5, zorder=7)
    ax.add_patch(bg)

    ax.text(px + 4, py - panel_h + 10, "scene graph",
            fontsize=7, color="#888", fontfamily="monospace",
            fontweight="bold", zorder=8)

    for i, obj in enumerate(scene.objects):
        row_y = py - panel_h + 20 + i * line_h
        dot = Circle((px + 8, row_y + 4), 3,
                     facecolor=obj.color_hex, edgecolor="none", zorder=8)
        ax.add_patch(dot)
        novel_tag = " [N]" if obj.is_novel() else ""
        ax.text(px + 15, row_y + 4,
                f"{obj.label}  ({int(obj.x)},{int(obj.y)}){novel_tag}",
                ha="left", va="center", fontsize=6.5,
                color=TEXT_COLOR, fontfamily="monospace", zorder=8)


def _lighten(hex_color: str, amount: float = 0.3) -> str:
    """Lighten a hex color by blending with white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def fig_to_pil(fig):
    """Convert matplotlib fig to PIL Image for Gradio."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
