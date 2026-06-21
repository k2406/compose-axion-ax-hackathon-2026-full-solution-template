"""
COMPOSE - Perception Pipeline
YOLO-v8 object detection → DINOv2 frozen feature extraction →
3-head attribute MLP (shape / colour / size) → SceneObject list

Usage:
    from perception import Perceptor
    p = Perceptor()
    p.load()
    scene = p.image_to_scene(pil_image)
"""

import os
import numpy as np
from PIL import Image
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from transformers import AutoModel
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False

from core.scene import (
    Scene, SceneObject,
    Shape, Color, Size,
    COLOR_HEX, SIZE_PX,
)


# ── Label maps (YOLO COCO → COMPOSE vocabulary) ───────────────────────────────
COCO_TO_SHAPE = {
    "cup":        Shape.CYLINDER,
    "bottle":     Shape.CYLINDER,
    "bowl":       Shape.CONTAINER,
    "box":        Shape.CUBE,
    "book":       Shape.CUBE,
    "cell phone": Shape.CUBE,
    "remote":     Shape.CUBE,
    "vase":       Shape.CYLINDER,
    "sports ball":Shape.SPHERE,
    "orange":     Shape.SPHERE,
    "apple":      Shape.SPHERE,
}

# Colour quantisation — maps dominant RGB to COMPOSE Color
COLOUR_CENTROIDS = {
    Color.RED:    np.array([200,  50,  50]),
    Color.BLUE:   np.array([ 50, 100, 200]),
    Color.GREEN:  np.array([ 50, 180,  80]),
    Color.YELLOW: np.array([220, 200,  50]),
    Color.PURPLE: np.array([140,  60, 180]),
    Color.CYAN:   np.array([ 30, 180, 180]),
    Color.ORANGE: np.array([220, 130,  40]),
    Color.PINK:   np.array([220,  80, 160]),
}


# ── 3-Head Attribute MLP ──────────────────────────────────────────────────────

class AttributeMLP(nn.Module):
    """
    Three independent classification heads on top of frozen DINOv2 features.
    Input:  DINOv2 feature vector (1024-dim for dinov2-large)
    Output: shape logits (5), colour logits (8), size logits (3)
    """
    FEAT_DIM   = 1024
    N_SHAPES   = len(Shape)
    N_COLOURS  = len(Color)
    N_SIZES    = len(Size)

    def __init__(self):
        super().__init__()
        hidden = 256
        self.shared = nn.Sequential(
            nn.Linear(self.FEAT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_shape  = nn.Linear(hidden, self.N_SHAPES)
        self.head_colour = nn.Linear(hidden, self.N_COLOURS)
        self.head_size   = nn.Linear(hidden, self.N_SIZES)

    def forward(self, x):
        h = self.shared(x)
        return (
            self.head_shape(h),
            self.head_colour(h),
            self.head_size(h),
        )


# ── DINOv2 transform ──────────────────────────────────────────────────────────

DINO_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ── Colour quantisation helper ────────────────────────────────────────────────

def dominant_colour(crop: Image.Image) -> Color:
    """Find closest COMPOSE colour by dominant RGB in centre crop."""
    w, h  = crop.size
    cx, cy = w // 4, h // 4
    centre = crop.crop((cx, cy, w - cx, h - cy))
    arr    = np.array(centre.convert("RGB"), dtype=float)
    mean   = arr.reshape(-1, 3).mean(axis=0)

    best_color = Color.RED
    best_dist  = float("inf")
    for color, centroid in COLOUR_CENTROIDS.items():
        dist = float(np.linalg.norm(mean - centroid))
        if dist < best_dist:
            best_dist  = dist
            best_color = color
    return best_color


# ── Main Perceptor class ──────────────────────────────────────────────────────

class Perceptor:
    """
    Full perception pipeline.
    Falls back gracefully if YOLO/DINOv2 unavailable.
    """

    def __init__(self, mlp_weights: Optional[str] = None):
        self.yolo       = None
        self.dino       = None
        self.mlp        = AttributeMLP()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp_weights = mlp_weights
        self._loaded    = False

    def load(self):
        """Download and initialise models. Call once at startup."""
        print(f"[Perceptor] Loading on {self.device}")

        # YOLO-v8 medium
        if YOLO_AVAILABLE:
            print("[Perceptor] Loading YOLOv8m...")
            self.yolo = YOLO("yolov8m.pt")
        else:
            print("[Perceptor] YOLO not available — using colour-based fallback")

        # DINOv2 large frozen
        if DINO_AVAILABLE:
            print("[Perceptor] Loading DINOv2-large (frozen)...")
            self.dino = AutoModel.from_pretrained("facebook/dinov2-large")
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino = self.dino.to(self.device)
        else:
            print("[Perceptor] DINOv2 not available — using colour fallback")

        # Attribute MLP
        self.mlp = self.mlp.to(self.device)
        if self.mlp_weights and os.path.exists(self.mlp_weights):
            print(f"[Perceptor] Loading MLP weights from {self.mlp_weights}")
            self.mlp.load_state_dict(
                torch.load(self.mlp_weights, map_location=self.device)
            )
        else:
            print("[Perceptor] No MLP weights — using random init + colour fallback")
        self.mlp.eval()

        self._loaded = True
        print("[Perceptor] Ready")

    # ── Inference ─────────────────────────────────────────────────────────────

    def image_to_scene(self, image: Image.Image,
                        canvas_w: float = 640,
                        canvas_h: float = 420) -> Scene:
        """
        Full pipeline: image → detected objects → Scene.
        image: PIL Image (RGB)
        """
        if not self._loaded:
            self.load()

        img_w, img_h = image.size
        detections   = self._detect(image)

        objects = []
        used_colours = set()

        for idx, det in enumerate(detections[:6]):  # max 6 objects
            x1, y1, x2, y2 = det["bbox"]
            crop  = image.crop((x1, y1, x2, y2)).convert("RGB")

            # Map bbox centre to canvas coordinates
            cx_img = (x1 + x2) / 2
            cy_img = (y1 + y2) / 2
            cx = cx_img / img_w * canvas_w
            cy = cy_img / img_h * canvas_h

            # Size from bbox area relative to image
            area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
            size = (Size.LARGE if area_ratio > 0.08
                    else Size.SMALL if area_ratio < 0.03
                    else Size.MEDIUM)

            # Shape from YOLO class label
            yolo_label = det.get("label", "")
            shape = COCO_TO_SHAPE.get(yolo_label, Shape.CUBE)

            # Colour from DINOv2 + MLP or fallback
            colour = self._classify_colour(crop, used_colours)
            used_colours.add(colour)

            objects.append(SceneObject(
                obj_id=f"obj_{idx}",
                shape=shape,
                color=colour,
                size=size,
                x=cx, y=cy,
            ))

        if not objects:
            print("[Perceptor] No objects detected — returning default scene")
            from core.scene import make_default_scene
            return make_default_scene()

        return Scene(objects=objects, canvas_w=canvas_w, canvas_h=canvas_h)

    def _detect(self, image: Image.Image) -> list[dict]:
        """Run YOLO detection. Falls back to grid-based mock if unavailable."""
        if self.yolo is None:
            return self._mock_detect(image)

        results = self.yolo(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = results.names[cls]
            if conf > 0.3:
                detections.append({
                    "bbox":  [x1, y1, x2, y2],
                    "conf":  conf,
                    "label": label,
                })
        return detections

    def _mock_detect(self, image: Image.Image) -> list[dict]:
        """
        Fallback when YOLO unavailable.
        Divides image into grid cells and treats each cell as an object.
        """
        w, h    = image.size
        rows, cols = 2, 3
        detections = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * w // cols
                y1 = r * h // rows
                x2 = (c + 1) * w // cols
                y2 = (r + 1) * h // rows
                detections.append({
                    "bbox":  [x1, y1, x2, y2],
                    "conf":  0.9,
                    "label": "box",
                })
        return detections

    def _classify_colour(self, crop: Image.Image,
                          used_colours: set) -> Color:
        """
        Classify crop colour.
        If DINOv2 + MLP available: use feature-based classification.
        Otherwise: dominant colour quantisation.
        """
        if self.dino is not None:
            try:
                tensor = DINO_TRANSFORM(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.dino(pixel_values=tensor).last_hidden_state
                    feat = feat[:, 0, :]   # CLS token
                _, colour_logits, _ = self.mlp(feat)
                colour_idx = colour_logits.argmax(dim=1).item()
                colour     = list(Color)[colour_idx]
                if colour not in used_colours:
                    return colour
            except Exception as e:
                print(f"[Perceptor] DINOv2 classify failed: {e}")

        # Fallback: dominant colour, avoid duplicates
        colour = dominant_colour(crop)
        if colour in used_colours:
            remaining = [c for c in Color if c not in used_colours]
            colour = remaining[0] if remaining else colour
        return colour

    def extract_features(self, crop: Image.Image) -> Optional[np.ndarray]:
        """Extract raw DINOv2 CLS features for a single crop."""
        if self.dino is None:
            return None
        tensor = DINO_TRANSFORM(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.dino(pixel_values=tensor).last_hidden_state
            return feat[:, 0, :].cpu().numpy()


# ── Synthetic data generation + MLP training ──────────────────────────────────

def generate_synthetic_data(n_samples: int = 800) -> tuple:
    """
    Generate synthetic training data for the attribute MLP.
    Each sample: a 1024-dim noise vector (simulating DINOv2 features)
    + colour/shape/size labels.
    Replace with real DINOv2 features extracted from PyBullet renders
    for better accuracy.
    """
    shapes  = list(Shape)
    colours = list(Color)
    sizes   = list(Size)

    X, y_shape, y_colour, y_size = [], [], [], []

    for _ in range(n_samples):
        s_idx = np.random.randint(len(shapes))
        c_idx = np.random.randint(len(colours))
        z_idx = np.random.randint(len(sizes))

        # Simulate class-discriminative features
        feat  = np.random.randn(1024) * 0.3
        feat[s_idx * 200:(s_idx + 1) * 200] += 2.0   # shape signal
        feat[500 + c_idx * 60:500 + (c_idx + 1) * 60] += 2.0  # colour signal
        feat[980 + z_idx * 14:980 + (z_idx + 1) * 14] += 2.0  # size signal

        X.append(feat)
        y_shape.append(s_idx)
        y_colour.append(c_idx)
        y_size.append(z_idx)

    return (
        np.array(X, dtype=np.float32),
        np.array(y_shape), np.array(y_colour), np.array(y_size),
    )


def train_attribute_mlp(save_path: str = "mlp_weights.pth",
                         n_samples: int = 800,
                         epochs: int = 30) -> AttributeMLP:
    """
    Train the 3-head MLP on synthetic data.
    In production: replace generate_synthetic_data() with real
    DINOv2 features extracted from PyBullet-rendered scenes.
    """
    print(f"[MLP] Generating {n_samples} synthetic samples...")
    X, ys, yc, yz = generate_synthetic_data(n_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttributeMLP().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    X_t  = torch.tensor(X).to(device)
    ys_t = torch.tensor(ys, dtype=torch.long).to(device)
    yc_t = torch.tensor(yc, dtype=torch.long).to(device)
    yz_t = torch.tensor(yz, dtype=torch.long).to(device)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        ls, lc, lz = model(X_t)
        loss = loss_fn(ls, ys_t) + loss_fn(lc, yc_t) + loss_fn(lz, yz_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[MLP] Weights saved to {save_path}")
    model.eval()
    return model
