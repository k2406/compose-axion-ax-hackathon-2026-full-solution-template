"""
COMPOSE - Main Gradio Application
Full integration: scene management + add objects + PyBullet execution + animated GUI
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.scene import (
    Scene, SceneObject, make_default_scene,
    Color, Shape, Size,
)
from core.reasoning import reason, ReasoningResult, NOVEL_COLORS
from gui.renderer import draw_scene, fig_to_pil


# ── Lazy model singletons ─────────────────────────────────────────────────────
_perceptor  = None
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
        self.comp_total  = self.comp_ok  = 0

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


# ── State serialisation ───────────────────────────────────────────────────────

def scene_to_dict(scene: Scene) -> dict:
    return {
        "canvas_w": scene.canvas_w,
        "canvas_h": scene.canvas_h,
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
    for k, v in d.items():
        setattr(m, k, v)
    return m


# ── Rendering ─────────────────────────────────────────────────────────────────

def get_arena(scene: Scene, result=None) -> np.ndarray:
    try:
        env = get_bullet()
        frame = env.capture_frame()
        if frame is not None:
            return frame
    except Exception:
        pass
    img = fig_to_pil(draw_scene(scene, result))
    plt.close('all')
    return np.array(img)


# ── Add object handler ────────────────────────────────────────────────────────

def handle_add_object(color_str, shape_str, size_str, scene_state, metrics_state):
    """Add a new user-specified object to the scene."""
    scene   = dict_to_scene(scene_state)
    metrics = dict_to_metrics(metrics_state)

    try:
        color = Color(color_str.lower())
        shape = Shape(shape_str.lower())
        size  = Size(size_str.lower())
    except ValueError as e:
        return scene_state, metrics_state, get_arena(scene), \
               metrics.tsr, metrics.goal_acc, metrics.novel_gen, metrics.comp_gen

    obj_id  = scene.next_obj_id()
    x, y    = scene.free_position()
    new_obj = SceneObject(obj_id=obj_id, shape=shape, color=color,
                          size=size, x=x, y=y)
    scene.objects.append(new_obj)

    # Sync with PyBullet if running
    try:
        env = get_bullet()
        env._spawn_object(new_obj)
        env._settle(40)
    except Exception:
        pass

    arena_img = get_arena(scene)
    novel_note = " (NOVEL colour)" if new_obj.is_novel() else ""
    # Return scene graph summary as status
    return (
        scene_to_dict(scene), metrics_to_dict(metrics),
        arena_img,
        metrics.tsr, metrics.goal_acc, metrics.novel_gen, metrics.comp_gen,
        f"Added: {new_obj.label} ({size_str}){novel_note} at ({int(x)}, {int(y)})",
    )


# ── Command handler ───────────────────────────────────────────────────────────

def handle_command(command, history, scene_state, metrics_state):
    if not command.strip():
        scene = dict_to_scene(scene_state)
        return history, get_arena(scene), scene_state, metrics_state, "—", "—", "—", "—"

    scene   = dict_to_scene(scene_state)
    metrics = dict_to_metrics(metrics_state)
    result  = reason(command, scene)
    metrics.update(result)

    parts = []
    if result.intent:
        i = result.intent
        chips = " ".join(filter(None, [
            f"[action:{i.action}]"             if i.action       else None,
            f"[target:{i.target_color.value}]" if i.target_color else None,
            f"[shape:{i.target_shape.value}]"  if i.target_shape else None,
            f"[size:{i.target_size.value}]"    if i.target_size  else None,
            f"[spatial:{i.spatial_rel}]"        if i.spatial_rel  else None,
            f"[conf:{result.confidence}]",
        ]))
        parts.append(f"Parsed: {chips}")

    if result.is_novel:
        parts.append("NOVEL — unseen colour detected. Compositional generalisation applied via disentangled embeddings.")

    if result.ambiguous:
        parts.append(f"Ambiguous — {result.message}")
        arena_img = get_arena(scene)
    elif result.success:
        parts.append(f"Executing: {result.message}")
        obj = scene.get(result.target.obj_id)
        if obj:
            obj.x = result.dest_x
            obj.y = result.dest_y
        try:
            env = get_bullet()
            env.move_object(result.target.obj_id, result.dest_x, result.dest_y)
            arena_img = get_arena(scene)
        except Exception as e:
            print(f"[PyBullet] {e}")
            arena_img = np.array(fig_to_pil(draw_scene(scene, result)))
            plt.close('all')
    else:
        parts.append(f"Error: {result.message}")
        arena_img = get_arena(scene)

    parts.append(f"TSR: {metrics.tsr}")
    history = history + [[command, "\n".join(parts)]]

    return (
        history, arena_img,
        scene_to_dict(scene), metrics_to_dict(metrics),
        metrics.tsr, metrics.goal_acc, metrics.novel_gen, metrics.comp_gen,
    )


# ── Reset + demo ──────────────────────────────────────────────────────────────

def reset_scene(metrics_state):
    scene   = make_default_scene()
    metrics = Metrics()
    try:
        env = get_bullet()
        env.load_scene(scene)
    except Exception:
        pass
    return (
        [], get_arena(scene),
        scene_to_dict(scene), metrics_to_dict(metrics),
        "—", "—", "—", "—", "",
    )

def load_demo(demo_idx, history, scene_state, metrics_state):
    demos = [
        "move red cube right of blue cube",
        "move green cylinder behind yellow container",
        "move cyan sphere left of green cylinder",
        "move purple container beside red cube",
        "move blue cube in front of yellow container",
    ]
    cmd = demos[int(demo_idx)]
    result = handle_command(cmd, history, scene_state, metrics_state)
    # handle_command returns 8 values, add empty status for 9th
    return result + ("",)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

# Dropdown choices
COLORS  = [c.value for c in Color]
SHAPES  = [s.value for s in Shape]
SIZES   = [s.value for s in Size]

def build_ui():
    scene0   = make_default_scene()
    metrics0 = Metrics()
    img0     = get_arena(scene0)

    with gr.Blocks(css=CSS, title="COMPOSE") as demo:
        scene_state   = gr.State(scene_to_dict(scene0))
        metrics_state = gr.State(metrics_to_dict(metrics0))

        gr.Markdown("""# COMPOSE
**Compositional Object Manipulation via Semantic Embeddings**
Type a natural language command to manipulate objects, or add new objects to the scene.""")

        with gr.Row():

            # ── Left: arena + controls ──────────────────────────────────────
            with gr.Column(scale=3):
                arena = gr.Image(
                    value=img0, show_label=False,
                    height=420, interactive=False,
                )

                # Metrics
                with gr.Row():
                    tsr_box   = gr.Textbox(value="—", label="Task success",     interactive=False)
                    goal_box  = gr.Textbox(value="—", label="Goal accuracy",    interactive=False)
                    novel_box = gr.Textbox(value="—", label="Novel colour gen", interactive=False)
                    comp_box  = gr.Textbox(value="—", label="Comp. gen",        interactive=False)

                # Add object panel
                with gr.Accordion("➕ Add object to scene", open=False):
                    with gr.Row():
                        color_dd = gr.Dropdown(
                            choices=COLORS, value="orange",
                            label="Colour", interactive=True,
                        )
                        shape_dd = gr.Dropdown(
                            choices=SHAPES, value="cube",
                            label="Shape", interactive=True,
                        )
                        size_dd  = gr.Dropdown(
                            choices=SIZES, value="medium",
                            label="Size", interactive=True,
                        )
                    add_btn    = gr.Button("Add to scene", variant="primary")
                    add_status = gr.Textbox(value="", label="Status", interactive=False)

                # Demo buttons
                with gr.Row():
                    demo_btns = [gr.Button(f"Demo {i+1}") for i in range(5)]

                gr.Markdown("""**Demos:**
`1` move red cube right of blue cube
`2` move green cylinder behind yellow container
`3` move cyan sphere left of green cylinder ← **NOVEL**
`4` move purple container beside red cube
`5` move blue cube in front of yellow container""")

            # ── Right: chat ─────────────────────────────────────────────────
            with gr.Column(scale=2):
                chatbot   = gr.Chatbot(height=480)
                cmd_input = gr.Textbox(
                    placeholder="move red cube right of blue cube",
                    show_label=False,
                )
                send_btn  = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset scene", variant="secondary")

        # ── Output lists ──────────────────────────────────────────────────────
        cmd_outputs = [
            chatbot, arena, scene_state, metrics_state,
            tsr_box, goal_box, novel_box, comp_box,
        ]
        add_outputs = [
            scene_state, metrics_state, arena,
            tsr_box, goal_box, novel_box, comp_box,
            add_status,
        ]
        reset_outputs = cmd_outputs + [add_status]

        # ── Events ────────────────────────────────────────────────────────────
        send_btn.click(
            handle_command,
            inputs=[cmd_input, chatbot, scene_state, metrics_state],
            outputs=cmd_outputs,
        )
        cmd_input.submit(
            handle_command,
            inputs=[cmd_input, chatbot, scene_state, metrics_state],
            outputs=cmd_outputs,
        )
        add_btn.click(
            handle_add_object,
            inputs=[color_dd, shape_dd, size_dd, scene_state, metrics_state],
            outputs=add_outputs,
        )
        reset_btn.click(
            reset_scene,
            inputs=[metrics_state],
            outputs=reset_outputs,
        )
        for idx, btn in enumerate(demo_btns):
            btn.click(
                load_demo,
                inputs=[
                    gr.Number(value=idx, visible=False),
                    chatbot, scene_state, metrics_state,
                ],
                outputs=cmd_outputs + [add_status],
            )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True, server_port=7860, show_error=True)
