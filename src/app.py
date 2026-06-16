"""
COMPOSE - Main Gradio Application
Chat interface + live 2D arena + scene graph + metrics bar.
Run: python app.py  or  in Colab: exec(open('app.py').read())
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import copy
import gradio as gr

from core.scene import Scene, SceneObject, make_default_scene, Color, Shape, Size
from core.reasoning import reason, ReasoningResult, NOVEL_COLORS
from gui.renderer import draw_scene, fig_to_pil


# ── Global State ───────────────────────────────────────────────────────────────
# Gradio shares state via gr.State — we clone the scene on each command.

def get_fresh_scene() -> Scene:
    return make_default_scene()


# ── Metrics tracker ────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.total        = 0
        self.successes    = 0
        self.novel_total  = 0
        self.novel_ok     = 0
        self.comp_total   = 0   # commands requiring compositional reasoning
        self.comp_ok      = 0

    def update(self, result: ReasoningResult):
        self.total += 1
        if result.success:
            self.successes += 1
        if result.is_novel:
            self.novel_total += 1
            if result.success:
                self.novel_ok += 1
        if result.intent and result.intent.spatial_rel:
            self.comp_total += 1
            if result.success:
                self.comp_ok += 1

    @property
    def tsr(self) -> str:
        if self.total == 0: return "—"
        return f"{self.successes/self.total*100:.0f}%"

    @property
    def novel_gen(self) -> str:
        if self.novel_total == 0: return "—"
        return f"+{self.novel_ok/self.novel_total*100:.0f}%"

    @property
    def comp_gen(self) -> str:
        if self.comp_total == 0: return "—"
        return f"{self.comp_ok/self.comp_total*100:.0f}%"

    @property
    def goal_acc(self) -> str:
        # Approximation: successful spatial commands / total spatial
        if self.comp_total == 0: return "—"
        return f"{self.comp_ok/self.comp_total*100:.0f}%"


# ── Command handler ────────────────────────────────────────────────────────────

def handle_command(command: str, history: list, scene_state: dict, metrics_state: dict):
    """
    Main pipeline: parse → reason → execute → update scene → return UI updates.
    Returns: (updated_history, arena_image, scene_state, metrics_state,
              tsr, goal_acc, novel_gen, comp_gen)
    """
    if not command.strip():
        return history, None, scene_state, metrics_state, "—", "—", "—", "—"

    # Deserialise scene from state dict
    scene = _dict_to_scene(scene_state)
    metrics = _dict_to_metrics(metrics_state)

    # Add user message
    history = history + [{"role": "user", "content": command}]

    # Run reasoning pipeline
    result = reason(command, scene)
    metrics.update(result)

    # Build assistant response
    response_parts = []

    # Intent chips (shown as formatted text in Gradio)
    if result.intent:
        i = result.intent
        chips = []
        if i.action:                chips.append(f"`action:{i.action}`")
        if i.target_color:          chips.append(f"`target:{i.target_color.value}`")
        if i.target_shape:          chips.append(f"`shape:{i.target_shape.value}`")
        if i.target_size:           chips.append(f"`size:{i.target_size.value}`")
        if i.spatial_rel:           chips.append(f"`spatial:{i.spatial_rel}`")
        if i.ref_color or i.ref_shape:
            ref = " ".join(filter(None,[
                i.ref_color.value  if i.ref_color  else None,
                i.ref_shape.value if i.ref_shape else None,
            ]))
            chips.append(f"`ref:{ref}`")
        chips.append(f"`conf:{result.confidence}`")
        response_parts.append("**Parsed intent:** " + "  ".join(chips))

    if result.is_novel:
        response_parts.append("**NOVEL** — unseen color detected. Applying compositional generalisation via disentangled embeddings.")

    if result.ambiguous:
        response_parts.append(f"**Ambiguous** — {result.message}")
    elif result.success:
        response_parts.append(f"**Executing:** {result.message}")
        # Apply movement to scene
        if result.dest_x is not None:
            obj = scene.get(result.target.obj_id)
            if obj:
                obj.x = result.dest_x
                obj.y = result.dest_y
    else:
        response_parts.append(f"**Error:** {result.message}")

    response_parts.append(f"TSR this session: **{metrics.tsr}**")

    history = history + [{"role": "assistant", "content": "\n\n".join(response_parts)}]

    # Render arena (show trajectory before finalising position)
    render_result = result if not result.success else None  # show arc on next render
    fig = draw_scene(scene, result if result.success and result.dest_x else None)
    img = fig_to_pil(fig)

    return (
        history, img,
        _scene_to_dict(scene),
        _metrics_to_dict(metrics),
        metrics.tsr, metrics.goal_acc, metrics.novel_gen, metrics.comp_gen,
    )


def reset_scene(metrics_state):
    scene   = get_fresh_scene()
    metrics = Metrics()
    fig     = draw_scene(scene)
    img     = fig_to_pil(fig)
    return (
        [], img,
        _scene_to_dict(scene),
        _metrics_to_dict(metrics),
        "—", "—", "—", "—",
    )


def load_demo(demo_idx: int, history: list, scene_state: dict, metrics_state: dict):
    """Load one of the 5 pre-set demo commands."""
    demos = [
        "move red cube right of blue block",
        "move smallest object behind yellow container",
        "move cyan sphere left of green cylinder",
        "move purple container beside red cube",
        "move blue block in front of yellow container",
    ]
    cmd = demos[int(demo_idx)]
    return handle_command(cmd, history, scene_state, metrics_state)


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _scene_to_dict(scene: Scene) -> dict:
    return {
        "canvas_w": scene.canvas_w,
        "canvas_h": scene.canvas_h,
        "objects": [
            {
                "obj_id": o.obj_id,
                "shape":  o.shape.value,
                "color":  o.color.value,
                "size":   o.size.value,
                "x": o.x, "y": o.y,
            }
            for o in scene.objects
        ],
    }

def _dict_to_scene(d: dict) -> Scene:
    objs = [
        SceneObject(
            obj_id=o["obj_id"],
            shape=Shape(o["shape"]),
            color=Color(o["color"]),
            size=Size(o["size"]),
            x=o["x"], y=o["y"],
        )
        for o in d.get("objects", [])
    ]
    return Scene(objects=objs, canvas_w=d.get("canvas_w", 640), canvas_h=d.get("canvas_h", 420))

def _metrics_to_dict(m: Metrics) -> dict:
    return {k: v for k, v in vars(m).items()}

def _dict_to_metrics(d: dict) -> Metrics:
    m = Metrics()
    for k, v in d.items():
        setattr(m, k, v)
    return m


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
.metric-box { text-align: center; padding: 8px 0; }
.metric-val { font-size: 22px; font-weight: 600; }
.demo-btn { font-size: 12px !important; padding: 4px 8px !important; }
footer { display: none !important; }
"""

def build_ui():
    initial_scene   = get_fresh_scene()
    initial_metrics = Metrics()
    initial_fig     = draw_scene(initial_scene)
    initial_img     = fig_to_pil(initial_fig)

    with gr.Blocks(css=CUSTOM_CSS, title="COMPOSE") as demo:

        # ── State ──────────────────────────────────────────────────────────────
        scene_state   = gr.State(_scene_to_dict(initial_scene))
        metrics_state = gr.State(_metrics_to_dict(initial_metrics))

        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown("""
# COMPOSE
**Compositional Object Manipulation via Semantic Embeddings**  
Type a natural language command to manipulate objects in the workspace.
""")

        # ── Main layout ────────────────────────────────────────────────────────
        with gr.Row():

            # Left: Arena
            with gr.Column(scale=3):
                arena = gr.Image(
                    value=initial_img,
                    label="Workspace",
                    show_label=False,
                    height=420,
                    interactive=False,
                )

                # Metrics row
                with gr.Row():
                    tsr_box      = gr.Textbox(value="—", label="Task success rate",  interactive=False, elem_classes=["metric-box"])
                    goal_box     = gr.Textbox(value="—", label="Goal accuracy",       interactive=False, elem_classes=["metric-box"])
                    novel_box    = gr.Textbox(value="—", label="Novel color gen.",    interactive=False, elem_classes=["metric-box"])
                    comp_box     = gr.Textbox(value="—", label="Compositional gen.",  interactive=False, elem_classes=["metric-box"])

                # Demo buttons
                with gr.Row():
                    gr.Markdown("**Quick demos:**")
                with gr.Row():
                    demo_btns = [
                        gr.Button(f"Demo {i+1}", elem_classes=["demo-btn"], size="sm")
                        for i in range(5)
                    ]

                gr.Markdown("""
**Demo scenarios:**
1. `move red cube right of blue block` — baseline  
2. `move smallest object behind yellow container` — attribute reasoning  
3. `move cyan sphere left of green cylinder` — **unseen color** (novel generalisation)  
4. `move purple container beside red cube` — novel composition  
5. `move blue block in front of yellow container` — multi-constraint
""")

            # Right: Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="COMPOSE chat",
                    height=420,
                    type="messages",
                    show_copy_button=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    cmd_input = gr.Textbox(
                        placeholder="move red cube right of blue block",
                        label="",
                        scale=4,
                        show_label=False,
                        submit_btn=True,
                    )
                with gr.Row():
                    reset_btn = gr.Button("Reset scene", variant="secondary", size="sm")

        # ── Output list helper ─────────────────────────────────────────────────
        outputs = [chatbot, arena, scene_state, metrics_state,
                   tsr_box, goal_box, novel_box, comp_box]

        # ── Events ────────────────────────────────────────────────────────────
        cmd_input.submit(
            handle_command,
            inputs=[cmd_input, chatbot, scene_state, metrics_state],
            outputs=outputs,
        ).then(lambda: "", outputs=cmd_input)

        reset_btn.click(
            reset_scene,
            inputs=[metrics_state],
            outputs=outputs,
        )

        for idx, btn in enumerate(demo_btns):
            btn.click(
                load_demo,
                inputs=[gr.Number(value=idx, visible=False), chatbot, scene_state, metrics_state],
                outputs=outputs,
            )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=True,          # gives a public URL — works in Colab
        server_port=7860,
        show_error=True,
    )
