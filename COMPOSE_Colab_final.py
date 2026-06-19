import subprocess, os, sys

# ── 1. Fix numpy first ────────────────────────────────────────────────────────
subprocess.run(["pip", "install", "numpy==1.26.4", "-q"], check=True)

# ── 2. Install everything else ────────────────────────────────────────────────
subprocess.run(["pip", "install",
    "gradio==3.50.2",
    "huggingface_hub==0.19.4",
    "torch==2.3.0",
    "torchvision==0.18.0",
    "transformers==4.44.0",
    "ultralytics==8.2.0",
    "pybullet==3.2.6",
    "matplotlib==3.9.0",
    "Pillow==10.4.0",
    "scipy==1.13.0",
    "-q"
], check=True)
print("All packages installed")

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
if not os.path.exists('/content/compose-axion-ax-hackathon-2026-full-solution-template'):
    os.system("git clone https://github.com/k2406/compose-axion-ax-hackathon-2026-full-solution-template.git")
os.chdir('/content/compose-axion-ax-hackathon-2026-full-solution-template/src')
sys.path.insert(0, os.getcwd())
print("Working dir:", os.getcwd())

# ── 4. Reload numpy with correct version ──────────────────────────────────────
import importlib
import numpy as np
importlib.reload(np)
print("numpy:", np.__version__)

# ── 5. Train MLP ──────────────────────────────────────────────────────────────
from perception import train_attribute_mlp
print("\nTraining attribute MLP on T4 GPU...")
mlp = train_attribute_mlp(save_path="mlp_weights.pth", n_samples=800, epochs=40)
print("MLP ready — mlp_weights.pth saved")

# ── 6. Load Perceptor ─────────────────────────────────────────────────────────
from perception import Perceptor
perceptor = Perceptor(mlp_weights="mlp_weights.pth")
perceptor.load()
print("Perceptor ready")

# ── 7. Load PyBullet ──────────────────────────────────────────────────────────
from simulation.pybullet_env import BulletEnv
from core.scene import make_default_scene
bullet = BulletEnv()
bullet.start()
scene0 = make_default_scene()
bullet.load_scene(scene0)
print("PyBullet ready")

# ── 8. Test PyBullet frame ────────────────────────────────────────────────────
frame = bullet.capture_frame()
if frame is not None:
    print("PyBullet rendering: OK — frame shape:", frame.shape)
else:
    print("PyBullet rendering: falling back to matplotlib")

# ── 9. Run benchmark ──────────────────────────────────────────────────────────
print("\nRunning evaluate.py...")
os.system("python evaluate.py")

# ── 10. Inject models into app ────────────────────────────────────────────────
import app as _app
_app._perceptor  = perceptor
_app._bullet_env = bullet
print("Models injected into app")

# ── 11. Setup GUI ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.scene import make_default_scene, Scene, SceneObject, Color, Shape, Size
from core.reasoning import reason
from gui.renderer import draw_scene, fig_to_pil
from app import scene_to_dict, dict_to_scene, metrics_to_dict, dict_to_metrics, Metrics

scene0   = make_default_scene()
bullet.load_scene(scene0)
metrics0 = Metrics()

def get_arena(scene, result=None):
    try:
        frame = bullet.capture_frame()
        if frame is not None:
            return frame
    except:
        pass
    return np.array(fig_to_pil(draw_scene(scene, result)))

img0 = get_arena(scene0)

# ── 12. Command handler ───────────────────────────────────────────────────────
def handle_command_v2(command, history, scene_state, metrics_state):
    if not command.strip():
        return history, img0, scene_state, metrics_state, "—", "—", "—", "—"

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
        # Update scene position
        obj = scene.get(result.target.obj_id)
        if obj:
            obj.x = result.dest_x
            obj.y = result.dest_y
        # PyBullet execution
        try:
            frames, success = bullet.move_object(
                result.target.obj_id, result.dest_x, result.dest_y
            )
            arena_img = get_arena(scene)
            if success:
                parts.append("PyBullet: object reached goal position.")
        except Exception as e:
            print(f"[PyBullet] {e}")
            arena_img = np.array(fig_to_pil(draw_scene(scene, result)))
    else:
        parts.append(f"Error: {result.message}")
        arena_img = get_arena(scene)

    parts.append(f"TSR: {metrics.tsr}")
    history = history + [[command, "\n".join(parts)]]
    plt.close('all')

    return (
        history, arena_img,
        scene_to_dict(scene), metrics_to_dict(metrics),
        metrics.tsr, metrics.goal_acc, metrics.novel_gen, metrics.comp_gen,
    )

def reset_scene_v2(metrics_state):
    scene   = make_default_scene()
    metrics = Metrics()
    bullet.load_scene(scene)
    img = get_arena(scene)
    plt.close('all')
    return ([], img, scene_to_dict(scene), metrics_to_dict(metrics),
            "—", "—", "—", "—")

def load_demo_v2(demo_idx, history, scene_state, metrics_state):
    demos = [
        "move red cube right of blue block",
        "move green cylinder behind yellow container",
        "move cyan sphere left of green cylinder",
        "move purple container beside red cube",
        "move blue block in front of yellow container",
    ]
    return handle_command_v2(demos[int(demo_idx)], history, scene_state, metrics_state)

# ── 13. Launch Gradio ─────────────────────────────────────────────────────────
import gradio as gr
print("Gradio version:", gr.__version__)

with gr.Blocks(title="COMPOSE") as demo:
    scene_state   = gr.State(scene_to_dict(scene0))
    metrics_state = gr.State(metrics_to_dict(metrics0))

    gr.Markdown("""# COMPOSE
**Compositional Object Manipulation via Semantic Embeddings**
Type a natural language command to manipulate objects in the workspace.""")

    with gr.Row():
        with gr.Column(scale=3):
            arena = gr.Image(
                value=img0,
                show_label=False,
                height=420,
                interactive=False,
            )
            with gr.Row():
                tsr_box   = gr.Textbox(value="—", label="Task success",     interactive=False)
                goal_box  = gr.Textbox(value="—", label="Goal accuracy",    interactive=False)
                novel_box = gr.Textbox(value="—", label="Novel colour gen", interactive=False)
                comp_box  = gr.Textbox(value="—", label="Comp. gen",        interactive=False)
            with gr.Row():
                demo_btns = [gr.Button(f"Demo {i+1}") for i in range(5)]
            gr.Markdown("""**Demo commands:**
1. `move red cube right of blue block` — baseline
2. `move green cylinder behind yellow container` — attribute reasoning
3. `move cyan sphere left of green cylinder` — **NOVEL colour**
4. `move purple container beside red cube` — novel composition
5. `move blue block in front of yellow container` — multi-constraint""")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=460)
            cmd_input = gr.Textbox(
                placeholder="move red cube right of blue block",
                show_label=False,
            )
            send_btn  = gr.Button("Send", variant="primary")
            reset_btn = gr.Button("Reset scene", variant="secondary")

    outputs = [chatbot, arena, scene_state, metrics_state,
               tsr_box, goal_box, novel_box, comp_box]

    send_btn.click(handle_command_v2,
                   inputs=[cmd_input, chatbot, scene_state, metrics_state],
                   outputs=outputs)
    cmd_input.submit(handle_command_v2,
                     inputs=[cmd_input, chatbot, scene_state, metrics_state],
                     outputs=outputs)
    reset_btn.click(reset_scene_v2,
                    inputs=[metrics_state], outputs=outputs)
    for idx, btn in enumerate(demo_btns):
        btn.click(load_demo_v2,
                  inputs=[gr.Number(value=idx, visible=False),
                          chatbot, scene_state, metrics_state],
                  outputs=outputs)

demo.launch(share=True, debug=True)
