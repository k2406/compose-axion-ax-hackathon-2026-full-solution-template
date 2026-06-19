"""
COMPOSE - Main Gradio Application
Full integration: image upload → YOLO perception → reasoning → PyBullet execution → animated GUI
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import gradio as gr
from PIL import Image

from core.scene import Scene, SceneObject, make_default_scene, Color, Shape, Size
from core.reasoning import reason, ReasoningResult, NOVEL_COLORS
from gui.renderer import draw_scene, fig_to_pil


# ── Lazy imports for heavy models ─────────────────────────────────────────────
_perceptor = None
_bullet_env = None

def get_perceptor():
    global _perceptor
    if _perceptor is None:
        from perception import Perceptor
        _perceptor = Perceptor()
        _perceptor.load()
    return _perceptor

def get_bullet():
    global _bullet_env
    if _bullet_env is None:
        from simulation.pybullet_env import BulletEnv
        _bullet_env = BulletEnv()
        _bullet_env.start()
    return _bullet_env


# ── Metrics ───────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.total = self.successes = 0
        self.novel_total = self.novel_ok = 0
        self.comp_total = self.comp_ok = 0

    def update(self, result: ReasoningResult):
        self.total += 1
        if result.success:
            self.successes += 1
        if result.is_novel:
            self.novel_total += 1
            if result.success: self.novel_ok += 1
        if result.intent and result.intent.spatial_rel:
            self.comp_total += 1
            if result.success: self.comp_ok += 1

    @property
    def tsr(self):
        return f"{self.successes/self.total*100:.0f}%" if self.total else "—"
    @property
    def novel_gen(self):
        return f"+{self.novel_ok/self.novel_total*100:.0f}%" if self.novel_total else "—"
    @property
    def comp_gen(self):
        return f"{self.comp_ok/self.comp_total*100:.0f}%" if self.comp_total else "—"
    @property
    def goal_acc(self):
        return f"{self.comp_ok/self.comp_total*100:.0f}%" if self.comp_total else "—"


# ── Scene serialisation ───────────────────────────────────────────────────────

def scene_to_dict(scene: Scene) -> dict:
    return {
        "canvas_w": scene.canvas_w, "canvas_h": scene.canvas_h,
        "objects": [{
            "obj_id": o.obj_id, "shape": o.shape.value,
            "color": o.color.value, "size": o.size.value,
            "x": o.x, "y": o.y,
        } for o in scene.objects],
    }

def dict_to_scene(d: dict) -> Scene:
    return Scene(
        objects=[SceneObject(
            obj_id=o["obj_id"], shape=Shape(o["shape"]),
            color=Color(o["color"]), size=Size(o["size"]),
            x=o["x"], y=o["y"],
        ) for o in d.get("objects", [])],
        canvas_w=d.get("canvas_w", 640),
        canvas_h=d.get("canvas_h", 420),
    )

def metrics_to_dict(m: Metrics) -> dict:
    return vars(m).copy()

def dict_to_metrics(d: dict) -> Metrics:
    m = Metrics()
    for k, v in d.items(): setattr(m, k, v)
    return m


# ── Image upload handler ──────────────────────────────────────────────────────

def handle_image_upload(image, scene_state, metrics_state):
    """Run perception pipeline on uploaded image → populate scene."""
    if image is None:
        return scene_state, metrics_state, None, "—", "—", "—", "—"

    try:
        pil = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image
        perceptor = get_perceptor()
        scene = perceptor.image_to_scene(pil)

        # Load into PyBullet
        try:
            env = get_bullet()
            env.load_scene(scene)
            frame = env.capture_frame()
            if frame is not None:
                arena_img = Image.fromarray(frame)
            else:
                raise ValueError("no frame")
        except Exception:
            fig = draw_scene(scene)
            arena_img = fig_to_pil(fig)

        metrics = Metrics()
        return (
            scene_to_dict(scene), metrics_to_dict(metrics),
            arena_img, "—", "—", "—", "—",
        )

    except Exception as e:
        print(f"[upload] error: {e}")
        scene = make_default_scene()
        fig = draw_scene(scene)
        return (
            scene_to_dict(scene), metrics_to_dict(Metrics()),
            fig_to_pil(fig), "—", "—", "—", "—",
        )


# ── Command handler ───────────────────────────────────────────────────────────

def handle_command(command, history, scene_state, metrics_state):
    if not command.strip():
        return history, None, scene_state, metrics_state, "—", "—", "—", "—"

    scene   = dict_to_scene(scene_state)
    metrics = dict_to_metrics(metrics_state)

    history = history + [{"role": "user", "content": command}]
    result  = reason(command, scene)
    metrics.update(result)

    # Build response
    parts = []
    if result.intent:
        i = result.intent
        chips = []
        if i.action:       chips.append(f"`action:{i.action}`")
        if i.target_color: chips.append(f"`target:{i.target_color.value}`")
        if i.target_shape: chips.append(f"`shape:{i.target_shape.value}`")
        if i.target_size:  chips.append(f"`size:{i.target_size.value}`")
        if i.spatial_rel:  chips.append(f"`spatial:{i.spatial_rel}`")
        if i.ref_color or i.ref_shape:
            ref = " ".join(filter(None, [
                i.ref_color.value  if i.ref_color  else None,
                i.ref_shape.value if i.ref_shape else None,
            ]))
            chips.append(f"`ref:{ref}`")
        chips.append(f"`conf:{result.confidence}`")
        parts.append("**Parsed intent:** " + "  ".join(chips))

    if result.is_novel:
        parts.append("🔬 **NOVEL** — unseen colour detected. Compositional generalisation applied via disentangled embeddings.")

    if result.ambiguous:
        parts.append(f"⚠️ **Ambiguous** — {result.message}")
    elif result.success:
        parts.append(f"✅ **Executing:** {result.message}")
        # Execute in PyBullet
        arena_img = _execute_and_render(result, scene)
    else:
        parts.append(f"❌ **Error:** {result.message}")

    parts.append(f"TSR this session: **{metrics.tsr}**")
    history = history + [{"role": "assistant", "content": "\n\n".join(parts)}]

    if not result.success:
        fig = draw_scene(scene, result)
        arena_img = fig_to_pil(fig)

    return (
        history, arena_img,
        scene_to_dict(scene), metrics_to_dict(metrics),
        metrics.tsr, metrics.goal_acc, metrics.novel_gen, metrics.comp_gen,
    )


def _execute_and_render(result: ReasoningResult, scene: Scene):
    """Move object in PyBullet and return final frame. Falls back to matplotlib."""
    try:
        env = get_bullet()
        frames, success = env.move_object(
            result.target.obj_id,
            result.dest_x, result.dest_y,
        )
        # Update scene object position
        obj = scene.get(result.target.obj_id)
        if obj:
            obj.x = result.dest_x
            obj.y = result.dest_y

        frame = env.capture_frame()
        if frame is not None:
            return Image.fromarray(frame)
    except Exception as e:
        print(f"[PyBullet] {e} — falling back to matplotlib")

    # Matplotlib fallback
    obj = scene.get(result.target.obj_id)
    if obj:
        obj.x = result.dest_x
        obj.y = result.dest_y
    fig = draw_scene(scene, result)
    return fig_to_pil(fig)


# ── Reset + demo ──────────────────────────────────────────────────────────────

def reset_scene(metrics_state):
    scene   = make_default_scene()
    metrics = Metrics()
    try:
        env = get_bullet()
        env.load_scene(scene)
        frame = env.capture_frame()
        arena_img = Image.fromarray(frame) if frame is not None else fig_to_pil(draw_scene(scene))
    except Exception:
        arena_img = fig_to_pil(draw_scene(scene))
    return (
        [], arena_img,
        scene_to_dict(scene), metrics_to_dict(metrics),
        "—", "—", "—", "—",
    )

def load_demo(demo_idx, history, scene_state, metrics_state):
    demos = [
        "move red cube right of blue block",
        "move green cylinder behind yellow container",
        "move cyan sphere left of green cylinder",
        "move purple container beside red cube",
        "move blue block in front of yellow container",
    ]
    return handle_command(demos[int(demo_idx)], history, scene_state, metrics_state)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

def build_ui():
    scene0   = make_default_scene()
    metrics0 = Metrics()
    fig0     = draw_scene(scene0)
    img0     = fig_to_pil(fig0)

    with gr.Blocks(css=CSS, title="COMPOSE") as demo:
        scene_state   = gr.State(scene_to_dict(scene0))
        metrics_state = gr.State(metrics_to_dict(metrics0))

        gr.Markdown("""# COMPOSE
**Compositional Object Manipulation via Semantic Embeddings**
Upload a scene image or use the default workspace. Type natural language commands to manipulate objects.""")

        with gr.Row():
            # Left: arena + upload
            with gr.Column(scale=3):
                arena = gr.Image(value=img0, label="Workspace",
                                 show_label=False, height=420, interactive=False)

                with gr.Accordion("📷 Upload your own scene image", open=False):
                    img_upload = gr.Image(label="Scene image (JPG/PNG)",
                                          type="numpy", height=200)
                    upload_btn = gr.Button("Detect objects from image", variant="primary")

                with gr.Row():
                    tsr_box   = gr.Textbox(value="—", label="Task success",     interactive=False)
                    goal_box  = gr.Textbox(value="—", label="Goal accuracy",    interactive=False)
                    novel_box = gr.Textbox(value="—", label="Novel colour gen", interactive=False)
                    comp_box  = gr.Textbox(value="—", label="Comp. gen",        interactive=False)

                with gr.Row():
                    demo_btns = [gr.Button(f"Demo {i+1}", size="sm") for i in range(5)]

                gr.Markdown("""**Demos:** `1` baseline · `2` attribute · `3` **NOVEL colour** · `4` novel comp · `5` multi-constraint""")

            # Right: chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="COMPOSE", height=460,
                                     type="messages", bubble_full_width=False)
                cmd_input = gr.Textbox(placeholder="move red cube right of blue block",
                                       show_label=False, submit_btn=True)
                reset_btn = gr.Button("Reset scene", variant="secondary", size="sm")

        outputs = [chatbot, arena, scene_state, metrics_state,
                   tsr_box, goal_box, novel_box, comp_box]

        upload_outputs = [scene_state, metrics_state, arena,
                          tsr_box, goal_box, novel_box, comp_box]

        # Events
        cmd_input.submit(handle_command,
                         inputs=[cmd_input, chatbot, scene_state, metrics_state],
                         outputs=outputs).then(lambda: "", outputs=cmd_input)

        upload_btn.click(handle_image_upload,
                         inputs=[img_upload, scene_state, metrics_state],
                         outputs=upload_outputs)

        reset_btn.click(reset_scene,
                        inputs=[metrics_state], outputs=outputs)

        for idx, btn in enumerate(demo_btns):
            btn.click(load_demo,
                      inputs=[gr.Number(value=idx, visible=False),
                               chatbot, scene_state, metrics_state],
                      outputs=outputs)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True, server_port=7860, show_error=True)
