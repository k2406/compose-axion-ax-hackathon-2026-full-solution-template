"""
COMPOSE - PyBullet Simulation Environment
Manages the 3D physics scene, object loading, trajectory execution,
and frame capture for the GUI arena feed.
"""

import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

# PyBullet import with fallback
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from core.scene import SceneObject, Scene, Shape, Color, Size, COLOR_HEX


# ── Physics constants ─────────────────────────────────────────────────────────
PLANE_HEIGHT    = 0.0
TABLE_HEIGHT    = 0.02
OBJECT_FRICTION = 0.8
GRAVITY         = -9.81
SIM_STEPS       = 240          # steps per second
TRAJECTORY_STEPS = 60          # frames for smooth movement
SUCCESS_THRESHOLD = 0.05       # metres — goal reached if within this


# ── Colour → RGBA ─────────────────────────────────────────────────────────────
RGBA_MAP = {
    Color.RED:    [0.90, 0.20, 0.20, 1.0],
    Color.BLUE:   [0.16, 0.50, 0.72, 1.0],
    Color.GREEN:  [0.15, 0.68, 0.38, 1.0],
    Color.YELLOW: [0.95, 0.77, 0.06, 1.0],
    Color.PURPLE: [0.56, 0.27, 0.68, 1.0],
    Color.CYAN:   [0.09, 0.63, 0.52, 1.0],
    Color.ORANGE: [0.90, 0.50, 0.13, 1.0],
    Color.PINK:   [0.91, 0.12, 0.55, 1.0],
}

# Shape → (half-extents or radius, height)
SHAPE_DIMS = {
    Shape.CUBE:      {"type": "box",      "half": [0.04, 0.04, 0.04]},
    Shape.BLOCK:     {"type": "box",      "half": [0.06, 0.04, 0.03]},
    Shape.CONTAINER: {"type": "box",      "half": [0.06, 0.06, 0.05]},
    Shape.CYLINDER:  {"type": "cylinder", "radius": 0.03, "height": 0.08},
    Shape.SPHERE:    {"type": "sphere",   "radius": 0.04},
}

SIZE_SCALE = {
    Size.SMALL:  0.7,
    Size.MEDIUM: 1.0,
    Size.LARGE:  1.4,
}

# Canvas → world coordinate mapping (canvas 640×420 → world ±0.4m)
CANVAS_W, CANVAS_H = 640.0, 420.0
WORLD_SCALE = 0.001   # 1 pixel ≈ 1mm


def canvas_to_world(cx: float, cy: float) -> tuple[float, float]:
    """Convert 2D canvas coords to PyBullet XY plane coords."""
    wx = (cx - CANVAS_W / 2) * WORLD_SCALE
    wy = (CANVAS_H / 2 - cy) * WORLD_SCALE
    return wx, wy


def world_to_canvas(wx: float, wy: float) -> tuple[float, float]:
    """Convert PyBullet world coords back to canvas coords."""
    cx = wx / WORLD_SCALE + CANVAS_W / 2
    cy = CANVAS_H / 2 - wy / WORLD_SCALE
    return cx, cy


@dataclass
class SimObject:
    obj_id:    str
    body_id:   int           # PyBullet body id
    scene_obj: SceneObject
    world_x:   float
    world_y:   float
    world_z:   float


class BulletEnv:
    """
    Manages a PyBullet DIRECT (headless) simulation.
    Objects are placed from a COMPOSE Scene, physics is stepped for
    settling, and frames are captured via the debug camera.
    """

    def __init__(self):
        self.client   = None
        self.bodies:  dict[str, SimObject] = {}
        self.plane_id = None
        self._ready   = False

        if not PYBULLET_AVAILABLE:
            print("[BulletEnv] PyBullet not available — running in mock mode")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not PYBULLET_AVAILABLE:
            self._ready = True
            return
        if self.client is not None:
            p.disconnect(self.client)

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)

        # Ground plane
        self.plane_id = p.loadURDF(
            "plane.urdf", [0, 0, PLANE_HEIGHT],
            physicsClientId=self.client
        )
        self._ready = True

    def reset(self):
        self.bodies.clear()
        if self.client is not None:
            p.resetSimulation(physicsClientId=self.client)
            p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)
            self.plane_id = p.loadURDF(
                "plane.urdf", [0, 0, PLANE_HEIGHT],
                physicsClientId=self.client
            )

    def stop(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
        self._ready = False

    # ── Scene loading ──────────────────────────────────────────────────────────

    def load_scene(self, scene: Scene):
        """Load all scene objects into PyBullet."""
        self.reset()
        for obj in scene.objects:
            self._spawn_object(obj)
        self._settle(steps=120)

    def _spawn_object(self, obj: SceneObject) -> int:
        if not PYBULLET_AVAILABLE:
            self.bodies[obj.obj_id] = SimObject(
                obj_id=obj.obj_id, body_id=-1,
                scene_obj=obj,
                world_x=0, world_y=0, world_z=0,
            )
            return -1

        wx, wy = canvas_to_world(obj.x, obj.y)
        scale  = SIZE_SCALE[obj.size]
        dims   = SHAPE_DIMS[obj.shape]
        rgba   = RGBA_MAP.get(obj.color, [0.5, 0.5, 0.5, 1.0])

        if dims["type"] == "box":
            half = [h * scale for h in dims["half"]]
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                             physicsClientId=self.client)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                          rgbaColor=rgba,
                                          physicsClientId=self.client)
            wz = TABLE_HEIGHT + half[2]

        elif dims["type"] == "cylinder":
            r  = dims["radius"] * scale
            h  = dims["height"] * scale
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                             physicsClientId=self.client)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                          rgbaColor=rgba,
                                          physicsClientId=self.client)
            wz = TABLE_HEIGHT + h / 2

        else:  # sphere
            r  = dims["radius"] * scale
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=r,
                                             physicsClientId=self.client)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                          rgbaColor=rgba,
                                          physicsClientId=self.client)
            wz = TABLE_HEIGHT + r

        body_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[wx, wy, wz],
            physicsClientId=self.client,
        )
        p.changeDynamics(body_id, -1, lateralFriction=OBJECT_FRICTION,
                         physicsClientId=self.client)

        self.bodies[obj.obj_id] = SimObject(
            obj_id=obj.obj_id, body_id=body_id,
            scene_obj=obj, world_x=wx, world_y=wy, world_z=wz,
        )
        return body_id

    def _settle(self, steps: int = 120):
        if not PYBULLET_AVAILABLE or self.client is None:
            return
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client)

    # ── Movement ──────────────────────────────────────────────────────────────

    def move_object(self, obj_id: str, dest_canvas_x: float,
                    dest_canvas_y: float) -> list[np.ndarray]:
        """
        Move object from current position to destination.
        Returns list of frame captures (PIL Images) for animation.
        Uses kinematic trajectory — object lifted, moved, lowered.
        """
        if obj_id not in self.bodies:
            return []

        sim_obj = self.bodies[obj_id]
        dest_wx, dest_wy = canvas_to_world(dest_canvas_x, dest_canvas_y)

        if not PYBULLET_AVAILABLE or self.client is None:
            # Update mock position
            sim_obj.world_x = dest_wx
            sim_obj.world_y = dest_wy
            sim_obj.scene_obj.x = dest_canvas_x
            sim_obj.scene_obj.y = dest_canvas_y
            return []

        body_id = sim_obj.body_id
        start_x, start_y, start_z = sim_obj.world_x, sim_obj.world_y, sim_obj.world_z
        lift_z  = start_z + 0.12   # lift height

        # Kinematic: disable dynamics during move
        p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client)

        frames = []
        trajectory = self._build_trajectory(
            start_x, start_y, start_z,
            dest_wx, dest_wy, start_z,
            lift_z, TRAJECTORY_STEPS,
        )

        for pos in trajectory:
            p.resetBasePositionAndOrientation(
                body_id, pos, [0,0,0,1],
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)

        # Settle after placement
        self._settle(60)

        # Validate success
        final_pos, _ = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self.client
        )
        dist = np.hypot(final_pos[0] - dest_wx, final_pos[1] - dest_wy)
        success = dist < SUCCESS_THRESHOLD

        # Update scene object position
        sim_obj.world_x = final_pos[0]
        sim_obj.world_y = final_pos[1]
        sim_obj.scene_obj.x = dest_canvas_x
        sim_obj.scene_obj.y = dest_canvas_y

        return frames, success

    def _build_trajectory(self, sx, sy, sz, dx, dy, dz,
                           lift_z, n_steps) -> list:
        """
        3-phase trajectory: lift → arc → lower.
        Phase 1 (25%): lift straight up
        Phase 2 (50%): move horizontally at lift height
        Phase 3 (25%): lower to destination
        """
        traj = []
        p1 = int(n_steps * 0.25)
        p2 = int(n_steps * 0.75)

        for i in range(n_steps):
            if i < p1:
                t = i / p1
                traj.append([sx, sy, sz + (lift_z - sz) * t])
            elif i < p2:
                t = (i - p1) / (p2 - p1)
                traj.append([
                    sx + (dx - sx) * t,
                    sy + (dy - sy) * t,
                    lift_z,
                ])
            else:
                t = (i - p2) / (n_steps - p2)
                traj.append([dx, dy, lift_z + (dz - lift_z) * t])

        return traj

    # ── Rendering ─────────────────────────────────────────────────────────────

    def capture_frame(self, width=640, height=420) -> Optional[np.ndarray]:
        """Capture top-down orthographic frame from PyBullet camera."""
        if not PYBULLET_AVAILABLE or self.client is None:
            return None

        view_matrix = p.computeViewMatrix(
            cameraEyePosition   = [0, 0, 0.8],
            cameraTargetPosition = [0, 0, 0],
            cameraUpVector       = [0, 1, 0],
            physicsClientId      = self.client,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height,
            nearVal=0.1, farVal=10.0,
            physicsClientId=self.client,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self.client,
        )
        return np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def get_object_canvas_positions(self) -> dict[str, tuple[float, float]]:
        """Return current canvas positions of all objects."""
        positions = {}
        for obj_id, sim_obj in self.bodies.items():
            if PYBULLET_AVAILABLE and self.client is not None:
                pos, _ = p.getBasePositionAndOrientation(
                    sim_obj.body_id, physicsClientId=self.client
                )
                cx, cy = world_to_canvas(pos[0], pos[1])
            else:
                cx, cy = sim_obj.scene_obj.x, sim_obj.scene_obj.y
            positions[obj_id] = (cx, cy)
        return positions


# ── Singleton ─────────────────────────────────────────────────────────────────
_env: Optional[BulletEnv] = None

def get_env() -> BulletEnv:
    global _env
    if _env is None:
        _env = BulletEnv()
        _env.start()
    return _env
