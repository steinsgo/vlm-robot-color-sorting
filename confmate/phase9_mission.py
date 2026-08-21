"""Phase 9: confidence-gated multi-object peg-hole insertion.

This module is a simulation prototype that connects the ConfMate matching
pipeline to the Panda pick-and-insert motion used by the original demo.  The
scene is intentionally separate from color sorting.  Four seeded objects are
placed at shuffled reachable slots, their corresponding holes are shuffled
independently, and the robot must accept and complete all four matches for a
mission pass.

The collision-enabled hole is a low-profile socket proxy.  The peg remains a
dynamic body throughout the pick, transport, release, and settling phases;
there is no pose teleport into the hole.  This is useful for checking the
selected action and final pose, but it is not a complete contact-rich
insertion controller or a hardware success claim.  ``oracle`` uses the
analytic pair label only as a controller/debug baseline; it is not a VLM
result.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from .artifacts import save_video
from .phase4 import (
    CLIPImageImageMatcher,
    ObservationInputs,
    _rank,
    _softmax_confidence,
    chamfer_scores,
    multi_view_clip_scores,
)


MISSION_SHAPES = ("cylinder", "sphere", "square", "cuboid")
SURFACE_Z = 0.625
HOLE_FLOOR_THICKNESS = 0.012
BOX_HOLE_WALL_HEIGHT = 0.045
RELEASE_CLEARANCE = 0.040
PICK_HEIGHT_OFFSETS = {
    "cylinder": 0.015,
    "sphere": 0.015,
    "square": 0.025,
    "cuboid": 0.025,
}
# The cuboid is the most sensitive shape to the residual lateral error of the
# Panda IK path: its short side leaves less opening margin after GUI_SCALE is
# applied.  Keep the extra margin explicit and small rather than accepting a
# peg that is outside the opening.  The final release still requires geometric
# containment, floor support, stability, and zero wall contacts.
CUBOID_HOLE_EXTRA_CLEARANCE = 0.003
GUI_SCALE = 0.60
GUI_Z_SCALE = 0.75
PEG_SLOTS = (
    (0.25, 0.25),
    (0.36, 0.25),
    (0.47, 0.25),
    (0.58, 0.25),
)
HOLE_SLOTS = (
    (0.28, -0.16),
    (0.41, -0.16),
    (0.54, -0.16),
    (0.67, -0.16),
)
DOWNWARD_ORIENTATION = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])


def _box(half_x: float, half_y: float, half_z: float) -> dict:
    return {
        "kind": "box",
        "half_extents": [float(half_x), float(half_y), float(half_z)],
        "offset": [0.0, 0.0, 0.0],
    }


def _round_part(kind: str, radius: float, height: float) -> dict:
    return {
        "kind": kind,
        "radius": float(radius),
        "height": float(height),
        "offset": [0.0, 0.0, 0.0],
    }


def _ring_part(inner_radius: float, outer_radius: float, height: float) -> dict:
    return {
        "kind": "ring",
        "inner_radius": float(inner_radius),
        "outer_radius": float(outer_radius),
        "height": float(height),
        "offset": [0.0, 0.0, 0.0],
    }


def _ring_mesh(
    inner_radius: float,
    outer_radius: float,
    height: float,
    segments: int = 24,
) -> Tuple[List[List[float]], List[int]]:
    """Build a closed annulus mesh for the visual hole marker."""
    vertices: List[List[float]] = []
    for z in (-height / 2.0, height / 2.0):
        for radius in (inner_radius, outer_radius):
            for index in range(segments):
                angle = 2.0 * math.pi * index / segments
                vertices.append([radius * math.cos(angle), radius * math.sin(angle), z])

    def vertex(layer: int, radius_layer: int, index: int) -> int:
        return layer * segments * 2 + radius_layer * segments + (index % segments)

    indices: List[int] = []
    for index in range(segments):
        next_index = index + 1
        # Top and bottom annulus surfaces.
        indices.extend([vertex(1, 0, index), vertex(1, 1, index), vertex(1, 1, next_index)])
        indices.extend([vertex(1, 0, index), vertex(1, 1, next_index), vertex(1, 0, next_index)])
        indices.extend([vertex(0, 0, index), vertex(0, 1, next_index), vertex(0, 1, index)])
        indices.extend([vertex(0, 0, index), vertex(0, 0, next_index), vertex(0, 1, next_index)])
        # Outer and inner cylindrical sides.
        indices.extend([vertex(0, 1, index), vertex(1, 1, next_index), vertex(1, 1, index)])
        indices.extend([vertex(0, 1, index), vertex(0, 1, next_index), vertex(1, 1, next_index)])
        indices.extend([vertex(0, 0, index), vertex(1, 0, index), vertex(1, 0, next_index)])
        indices.extend([vertex(0, 0, index), vertex(1, 0, next_index), vertex(0, 0, next_index)])
    return vertices, indices


def _shape_parts(family: str, role: str, clearance: float = 0.004) -> List[dict]:
    """Return one simple 2.5-D peg or low-profile hole marker.

    ``clearance`` is a per-side linear opening clearance in metres.  Hole
    markers are deliberately shallow; the collision perimeter and floor are
    created separately by :func:`_create_hole_proxy`.
    """
    if family not in MISSION_SHAPES:
        raise ValueError(f"Unknown Phase 9 shape: {family}")
    if role not in {"peg", "hole"}:
        raise ValueError(f"Unknown role: {role}")
    if clearance <= 0.0:
        raise ValueError("clearance must be positive")

    peg_height = 0.12
    if family == "cylinder":
        if role == "peg":
            return [_round_part("cylinder", 0.040, peg_height)]
        inner = 0.040 + clearance
        return [_ring_part(inner, inner + 0.007, 0.035)]
    if family == "sphere":
        if role == "peg":
            return [_round_part("sphere", 0.038, 0.076)]
        inner = 0.038 + clearance
        return [_ring_part(inner, inner + 0.007, 0.035)]
    if family == "square":
        if role == "peg":
            return [_box(0.042, 0.042, peg_height / 2.0)]
        return [_box(0.042 + clearance, 0.042 + clearance, 0.009)]
    if family == "cuboid":
        if role == "peg":
            return [_box(0.055, 0.032, peg_height / 2.0)]
        hole_clearance = clearance + CUBOID_HOLE_EXTRA_CLEARANCE
        return [_box(0.055 + hole_clearance, 0.032 + hole_clearance, 0.009)]
    raise AssertionError(f"Unhandled Phase 9 shape: {family}")


def _scaled_parts(parts: Sequence[dict]) -> List[dict]:
    scaled: List[dict] = []
    for part in parts:
        copy = dict(part)
        copy["offset"] = [
            float(part["offset"][0]) * GUI_SCALE,
            float(part["offset"][1]) * GUI_SCALE,
            float(part["offset"][2]) * GUI_Z_SCALE,
        ]
        if part["kind"] == "box":
            copy["half_extents"] = [
                float(part["half_extents"][0]) * GUI_SCALE,
                float(part["half_extents"][1]) * GUI_SCALE,
                float(part["half_extents"][2]) * GUI_Z_SCALE,
            ]
        elif part["kind"] == "ring":
            copy["inner_radius"] = float(part["inner_radius"]) * GUI_SCALE
            copy["outer_radius"] = float(part["outer_radius"]) * GUI_SCALE
            copy["height"] = float(part["height"]) * GUI_Z_SCALE
        else:
            copy["radius"] = float(part["radius"]) * GUI_SCALE
            copy["height"] = float(part["height"]) * GUI_Z_SCALE
        scaled.append(copy)
    return scaled


def _vertical_half_height(parts: Sequence[dict]) -> float:
    return max(
        (
            float(part["half_extents"][2])
            if part["kind"] == "box"
            else (
                float(part["height"]) / 2.0
                if part["kind"] != "sphere"
                else float(part["radius"])
            )
        )
        for part in parts
    )


def _footprint_bounds(parts: Sequence[dict]) -> Tuple[float, float, float, float]:
    extents = []
    for part in parts:
        offset_x, offset_y = part["offset"][:2]
        if part["kind"] == "box":
            half_x, half_y = part["half_extents"][:2]
        elif part["kind"] in {"cylinder", "sphere"}:
            half_x = half_y = float(part["radius"])
        elif part["kind"] == "ring":
            half_x = half_y = float(part["outer_radius"])
        else:
            raise ValueError(f"Unsupported Phase 9 part kind: {part['kind']}")
        extents.append(
            (offset_x - half_x, offset_x + half_x, offset_y - half_y, offset_y + half_y)
        )
    if not extents:
        raise ValueError("Cannot build an empty peg-hole footprint")
    return (
        min(item[0] for item in extents),
        max(item[1] for item in extents),
        min(item[2] for item in extents),
        max(item[3] for item in extents),
    )


def _rotate_xy(x: float, y: float, yaw_deg: float) -> Tuple[float, float]:
    angle = math.radians(yaw_deg)
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )


def _create_visual_shape(
    parts: Sequence[dict],
    color: Sequence[float],
) -> int:
    if len(parts) == 1 and parts[0]["kind"] == "ring":
        part = parts[0]
        vertices, indices = _ring_mesh(
            part["inner_radius"],
            part["outer_radius"],
            part["height"],
        )
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=color,
        )
    elif len(parts) == 1 and parts[0]["kind"] in {"cylinder", "sphere"}:
        part = parts[0]
        geometry = p.GEOM_CYLINDER if part["kind"] == "cylinder" else p.GEOM_SPHERE
        visual_id = p.createVisualShape(
            geometry,
            radius=part["radius"],
            **({"length": part["height"]} if part["kind"] == "cylinder" else {}),
            rgbaColor=color,
        )
    else:
        if not all(part["kind"] == "box" for part in parts):
            raise ValueError("Phase 9 visualizer supports boxes, cylinders, spheres, and rings")
        visual_id = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * len(parts),
            halfExtents=[part["half_extents"] for part in parts],
            visualFramePositions=[part["offset"] for part in parts],
            rgbaColors=[color] * len(parts),
        )
    return visual_id


def _create_visual_body(
    parts: Sequence[dict],
    position: Sequence[float],
    yaw_deg: float,
    color: Sequence[float],
) -> int:
    visual_id = _create_visual_shape(parts, color)
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)])
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=orientation,
    )


def _create_collision_shape(parts: Sequence[dict]) -> int:
    if len(parts) == 1 and parts[0]["kind"] in {"cylinder", "sphere"}:
        part = parts[0]
        geometry = p.GEOM_CYLINDER if part["kind"] == "cylinder" else p.GEOM_SPHERE
        return p.createCollisionShape(
            geometry,
            radius=part["radius"],
            **({"height": part["height"]} if part["kind"] == "cylinder" else {}),
        )
    if not all(part["kind"] == "box" for part in parts):
        raise ValueError("Phase 9 collision model supports boxes, cylinders, and spheres")
    return p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * len(parts),
        halfExtents=[part["half_extents"] for part in parts],
        collisionFramePositions=[part["offset"] for part in parts],
    )


def _create_hole_proxy(
    parts: Sequence[dict],
    position: Sequence[float],
    yaw_deg: float,
    surface_z: float,
    color: Sequence[float],
) -> Tuple[int, List[int], int, float]:
    """Create a low hole marker and a collision-enabled ring/box proxy."""
    min_x, max_x, min_y, max_y = _footprint_bounds(parts)
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    half_x = 0.5 * (max_x - min_x)
    half_y = 0.5 * (max_y - min_y)
    wall_thickness = 0.007
    floor_thickness = HOLE_FLOOR_THICKNESS
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)])
    wall_ids: List[int] = []
    is_ring = len(parts) == 1 and parts[0]["kind"] == "ring"
    if is_ring:
        ring = parts[0]
        wall_height = float(ring["height"])
        ring_segments = 16
        radial_half = 0.5 * (ring["outer_radius"] - ring["inner_radius"])
        tangential_half = ring["outer_radius"] * math.pi / ring_segments * 1.10
        ring_mid_radius = 0.5 * (ring["outer_radius"] + ring["inner_radius"])
        for index in range(ring_segments):
            angle_deg = 360.0 * index / ring_segments
            angle = math.radians(angle_deg)
            local_position = [
                ring_mid_radius * math.cos(angle),
                ring_mid_radius * math.sin(angle),
            ]
            offset_x, offset_y = _rotate_xy(local_position[0], local_position[1], yaw_deg)
            segment_orientation = p.getQuaternionFromEuler(
                # The local x-axis is the radial direction and the local
                # y-axis is tangent to the ring.  Adding 90 degrees here
                # would swap those axes and make the tangential half-width
                # intrude into the opening, producing false wall contacts.
                [0.0, 0.0, math.radians(yaw_deg + angle_deg)]
            )
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[radial_half, tangential_half, wall_height / 2.0],
            )
            wall_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=-1,
                    basePosition=[
                        position[0] + offset_x,
                        position[1] + offset_y,
                        surface_z + wall_height / 2.0,
                    ],
                    baseOrientation=segment_orientation,
                )
            )
    else:
        wall_height = BOX_HOLE_WALL_HEIGHT
        wall_specs = [
            ([half_x + wall_thickness, wall_thickness, wall_height / 2.0], [centre_x, min_y - wall_thickness]),
            ([half_x + wall_thickness, wall_thickness, wall_height / 2.0], [centre_x, max_y + wall_thickness]),
            ([wall_thickness, half_y + wall_thickness, wall_height / 2.0], [min_x - wall_thickness, centre_y]),
            ([wall_thickness, half_y + wall_thickness, wall_height / 2.0], [max_x + wall_thickness, centre_y]),
        ]
        wall_color = [
            min(1.0, color[0] * 0.72),
            min(1.0, color[1] * 0.72),
            min(1.0, color[2] * 0.72),
            1.0,
        ]
        for half_extents, local_position in wall_specs:
            offset_x, offset_y = _rotate_xy(local_position[0], local_position[1], yaw_deg)
            collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=wall_color)
            wall_ids.append(
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=visual_id,
                    basePosition=[
                        position[0] + offset_x,
                        position[1] + offset_y,
                        surface_z + wall_height / 2.0,
                    ],
                    baseOrientation=orientation,
                )
            )

    floor_half_x = (parts[0]["outer_radius"] if is_ring else half_x) + wall_thickness
    floor_half_y = (parts[0]["outer_radius"] if is_ring else half_y) + wall_thickness
    floor_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[floor_half_x, floor_half_y, floor_thickness / 2.0],
    )
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[floor_half_x, floor_half_y, floor_thickness / 2.0],
        rgbaColor=[0.035, 0.035, 0.045, 1.0],
    )
    centre_offset_x, centre_offset_y = _rotate_xy(centre_x, centre_y, yaw_deg)
    floor_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[
            position[0] + centre_offset_x,
            position[1] + centre_offset_y,
            # The table has a solid collision mesh at SURFACE_Z.  Put the
            # proxy floor just above that surface so a released dynamic peg
            # can actually fall onto this hole's floor instead of colliding
            # with the table underneath it.
            surface_z + floor_thickness / 2.0,
        ],
        baseOrientation=orientation,
    )
    visual_id = _create_visual_body(
        parts,
        [position[0], position[1], surface_z + _vertical_half_height(parts)],
        yaw_deg,
        color,
    )
    return visual_id, wall_ids, floor_id, surface_z + floor_thickness


def _mask_crop(image: np.ndarray, segmentation: np.ndarray, body_id: int) -> Image.Image:
    mask = (segmentation.astype(np.int64) & 0x00FFFFFF) == body_id
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return Image.fromarray(image.copy()).convert("RGB")
    x1 = max(0, int(xs.min()) - 8)
    y1 = max(0, int(ys.min()) - 8)
    x2 = min(image.shape[1], int(xs.max()) + 9)
    y2 = min(image.shape[0], int(ys.max()) + 9)
    crop = image[y1:y2, x1:x2].copy()
    local_mask = mask[y1:y2, x1:x2]
    crop[~local_mask] = 255
    return Image.fromarray(crop).convert("RGB")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def decide_action(
    scores: Dict[str, float],
    min_confidence: float,
    min_margin: float,
    temperature: float = 0.10,
) -> dict:
    """Apply tie-aware selective prediction before robot motion."""
    predicted, ranked, top_score, margin = _rank(scores)
    confidence = _softmax_confidence(scores, temperature=temperature)
    if predicted is not None and len(scores) == 1:
        # Once three pairs have been completed, the remaining hole is the only
        # admissible candidate.  There is no second score from which to form a
        # meaningful margin, so the one-candidate decision is unambiguous.
        margin = 1.0
    if predicted is None:
        return {
            "execute": False,
            "prediction_status": "uncertain",
            "uncertainty_reason": "score_tie",
            "selected_hole_id": None,
            "ranked_hole_ids": ranked,
            "top_score": top_score,
            "margin": margin,
            "confidence": confidence,
        }
    if confidence < min_confidence:
        reason = "confidence_below_threshold"
    elif margin < min_margin:
        reason = "margin_below_threshold"
    else:
        reason = None
    return {
        "execute": reason is None,
        "prediction_status": "predicted" if reason is None else "abstained",
        "uncertainty_reason": reason,
        "selected_hole_id": predicted if reason is None else None,
        "ranked_hole_ids": ranked,
        "top_score": top_score,
        "margin": margin,
        "confidence": confidence,
    }


@dataclass
class Phase9Config:
    """Configuration for the four-object confidence-aware mission."""

    seed: int = 17
    shape_families: Tuple[str, ...] = MISSION_SHAPES
    num_objects: int = 4
    clearance: float = 0.004
    slot_jitter: float = 0.018
    matcher: str = "clip"
    model_name: str = "openai/clip-vit-base-patch32"
    gui: bool = True
    keep_open: bool = False
    save_video: bool = True
    output_dir: str = "runs/phase9"
    video_fps: int = 8
    min_confidence: float = 0.30
    min_margin: float = 0.010
    confidence_temperature: float = 0.50
    motion_steps: int = 180
    settle_steps: int = 180
    sleep: bool = True
    # Compatibility fields retained for scripts written against the original
    # single-object demo.  The multi-object mission does not use them.
    family: str = "rectangle"
    num_candidates: int = 3

    def __post_init__(self) -> None:
        self.shape_families = tuple(self.shape_families)

    def validate(self) -> None:
        if self.num_objects != 4:
            raise ValueError("Phase 9 mission requires exactly four objects")
        if len(self.shape_families) != self.num_objects:
            raise ValueError("shape_families must contain exactly four shapes")
        if len(set(self.shape_families)) != len(self.shape_families):
            raise ValueError("shape_families must be unique for the default mission")
        unknown = set(self.shape_families) - set(MISSION_SHAPES)
        if unknown:
            raise ValueError(f"Unknown Phase 9 shape(s): {sorted(unknown)}")
        if self.matcher not in {"chamfer", "clip", "oracle"}:
            raise ValueError("matcher must be one of: chamfer, clip, oracle")
        if self.clearance <= 0.0:
            raise ValueError("clearance must be positive")
        if self.slot_jitter < 0.0 or self.slot_jitter > 0.04:
            raise ValueError("slot_jitter must be in [0, 0.04]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_margin < 0.0:
            raise ValueError("min_margin must be non-negative")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.motion_steps < 30 or self.settle_steps < 1:
            raise ValueError("motion_steps and settle_steps are too small")


class MultiObjectPegHoleScene:
    """Seeded four-peg/four-hole Panda scene."""

    def __init__(self, config: Phase9Config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.robot_id: Optional[int] = None
        self.pegs: Dict[str, dict] = {}
        self.holes: Dict[str, dict] = {}
        self.peg_order: List[str] = []
        self.hole_order: List[str] = []
        self.completed_pegs: Set[str] = set()
        self.completed_holes: Set[str] = set()
        self._label_ids: Dict[str, int] = {}

    @staticmethod
    def _yaw(rng: random.Random) -> float:
        return float(rng.choice([0, 15, 30, 45, 60, 90, 120, 150, 180]))

    def _jittered_slot(self, slot: Sequence[float]) -> List[float]:
        amount = self.config.slot_jitter
        return [
            float(slot[0] + self.rng.uniform(-amount, amount)),
            float(slot[1] + self.rng.uniform(-amount, amount)),
        ]

    def _pair_fit(self, peg: dict, hole: dict) -> dict:
        family_match = peg["shape_family"] == hole["shape_family"]
        yaw_delta = abs(((hole["yaw_deg"] - peg["yaw_deg"] + 180.0) % 360.0) - 180.0)
        yaw_relevant = peg["shape_family"] in {"square", "cuboid"}
        yaw_aligned = yaw_delta <= 1e-6 or not yaw_relevant
        requires_air_rotation = bool(yaw_relevant and not yaw_aligned)
        compatible = (
            family_match
            and hole["target_peg_id"] == peg["peg_id"]
        )
        configured_clearance = self.config.clearance
        shape_extra_clearance = (
            CUBOID_HOLE_EXTRA_CLEARANCE
            if peg["shape_family"] == "cuboid"
            else 0.0
        )
        return {
            "compatible": bool(compatible),
            "family_match": bool(family_match),
            "yaw_aligned": bool(yaw_aligned),
            "yaw_delta_deg": float(yaw_delta),
            "requires_air_rotation": requires_air_rotation,
            "target_hole_for_peg": hole["target_peg_id"] == peg["peg_id"],
            "fit_margin": (
                float(2.0 * (configured_clearance + shape_extra_clearance))
                if compatible
                else None
            ),
            "reason": (
                "positive_configured_clearance_requires_air_rotation"
                if compatible and requires_air_rotation
                else "positive_configured_clearance"
                if compatible
                else "not_the_corresponding_hole"
            ),
        }

    def build(self) -> None:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0])
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [0.0, 0.0, 0.626],
            useFixedBase=True,
        )

        peg_slots = list(PEG_SLOTS)
        hole_slots = list(HOLE_SLOTS)
        self.rng.shuffle(peg_slots)
        hole_ids = [f"hole_{index:03d}" for index in range(self.config.num_objects)]
        self.rng.shuffle(hole_ids)
        self.peg_order = [f"peg_{index:03d}" for index in range(self.config.num_objects)]
        self.rng.shuffle(self.peg_order)

        for index, family in enumerate(self.config.shape_families):
            peg_id = f"peg_{index:03d}"
            parts = _scaled_parts(_shape_parts(family, "peg", self.config.clearance))
            position = self._jittered_slot(peg_slots[index])
            yaw_deg = self._yaw(self.rng)
            half_height = _vertical_half_height(parts)
            visual_shape_id = _create_visual_shape(parts, [0.10, 0.30, 0.90, 1.0])
            collision_id = _create_collision_shape(parts)
            body_id = p.createMultiBody(
                baseMass=0.08,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[position[0], position[1], SURFACE_Z + half_height],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)]),
            )
            p.changeDynamics(
                body_id,
                -1,
                lateralFriction=0.85,
                spinningFriction=0.15,
                rollingFriction=0.15,
                restitution=0.0,
            )
            self.pegs[peg_id] = {
                "peg_id": peg_id,
                "shape_family": family,
                "position": position,
                "yaw_deg": yaw_deg,
                "parts": parts,
                "body_id": body_id,
                "visual_id": body_id,
            }

        for index, (peg_id, hole_id) in enumerate(zip(sorted(self.pegs), hole_ids)):
            peg = self.pegs[peg_id]
            # Hole positions are fixed across seeds; only IDs and pair
            # assignment are shuffled so the board remains reproducible.
            position = [float(hole_slots[index][0]), float(hole_slots[index][1])]
            hole_parts = _scaled_parts(
                _shape_parts(peg["shape_family"], "hole", self.config.clearance)
            )
            # The matching hole may have a different yaw.  The controller
            # must align the held peg in the air before release; the oracle
            # labels this as a motor-planning step rather than rejecting the
            # otherwise compatible shape pair.
            hole_yaw = self._yaw(self.rng)
            visual_id, wall_ids, floor_id, floor_top_z = _create_hole_proxy(
                hole_parts,
                position,
                hole_yaw,
                SURFACE_Z,
                [0.18, 0.18, 0.22, 1.0],
            )
            self.holes[hole_id] = {
                "hole_id": hole_id,
                "shape_family": peg["shape_family"],
                "position": position,
                "yaw_deg": hole_yaw,
                "parts": hole_parts,
                "target_peg_id": peg_id,
                "visual_id": visual_id,
                "wall_ids": wall_ids,
                "floor_id": floor_id,
                "floor_top_z": floor_top_z,
            }

        self.hole_order = list(hole_ids)
        for peg_id, peg in self.pegs.items():
            peg["target_hole_id"] = next(
                hole_id for hole_id, hole in self.holes.items() if hole["target_peg_id"] == peg_id
            )
            peg["candidate_fit"] = {
                hole_id: self._pair_fit(peg, hole) for hole_id, hole in self.holes.items()
            }

        p.stepSimulation()
        self._set_debug_camera()
        self._add_scene_labels()

    def _set_debug_camera(self) -> None:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.55,
            cameraYaw=42,
            cameraPitch=-38,
            cameraTargetPosition=[0.60, 0.0, 0.66],
        )

    def _add_scene_labels(self) -> None:
        for peg_id, peg in self.pegs.items():
            x, y = peg["position"]
            self._label_ids[peg_id] = p.addUserDebugText(
                peg_id.upper(),
                [x, y, SURFACE_Z + 0.14],
                textColorRGB=[0.25, 0.55, 1.0],
                textSize=1.0,
            )

    def ignore_robot_collision_with_completed_peg(self, peg_id: str) -> None:
        """Keep a placed peg visible/physical without blocking the next pick."""
        peg_body_id = int(self.pegs[peg_id]["body_id"])
        for link_index in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(
                self.robot_id,
                peg_body_id,
                linkIndexA=link_index,
                linkIndexB=-1,
                enableCollision=0,
            )
        self.pegs[peg_id]["robot_collision_disabled_after_insert"] = True
        for hole_id, hole in self.holes.items():
            x, y = hole["position"]
            self._label_ids[hole_id] = p.addUserDebugText(
                hole_id.upper(),
                [x, y, SURFACE_Z + 0.14],
                textColorRGB=[0.75, 0.75, 0.75],
                textSize=1.0,
            )

    def _camera(self, view_id: str) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        if view_id == "top":
            return (0.60, 0.0, 2.30), (0.60, 0.0, 0.0), (0.0, 1.0, 0.0)
        if view_id == "oblique":
            return (1.70, -1.65, 1.55), (0.60, 0.0, 0.03), (0.0, 0.0, 1.0)
        raise ValueError(f"Unknown Phase 9 view: {view_id}")

    def capture_observation(
        self,
        peg_id: str,
        active_hole_ids: Iterable[str],
    ) -> Tuple[Dict[str, Image.Image], Dict[str, Dict[str, Image.Image]]]:
        peg = self.pegs[peg_id]
        active = list(active_hole_ids)
        rgb_by_view: Dict[str, Image.Image] = {}
        crops_by_view: Dict[str, Dict[str, Image.Image]] = {}
        for view_id in ("top", "oblique"):
            eye, target, up = self._camera(view_id)
            view_matrix = p.computeViewMatrix(eye, target, up)
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=55.0,
                aspect=640 / 480,
                nearVal=0.01,
                farVal=4.0,
            )
            _, _, rgba, _, segmentation = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            )
            rgb = np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
            segmentation_array = np.asarray(segmentation, dtype=np.int32).reshape(480, 640)
            crops = {
                "peg": _mask_crop(rgb, segmentation_array, int(peg["visual_id"]))
            }
            for hole_id in active:
                crops[hole_id] = _mask_crop(
                    rgb,
                    segmentation_array,
                    int(self.holes[hole_id]["visual_id"]),
                )
            rgb_by_view[view_id] = Image.fromarray(rgb).convert("RGB")
            crops_by_view[view_id] = crops
        return rgb_by_view, crops_by_view

    def capture_frame(self) -> Image.Image:
        eye, target, up = self._camera("oblique")
        view_matrix = p.computeViewMatrix(eye, target, up)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=640 / 480,
            nearVal=0.01,
            farVal=4.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
        return Image.fromarray(rgb).convert("RGB")

    def annotate_scores(self, peg_id: str, scores: Dict[str, float], decision: dict) -> None:
        peg = self.pegs[peg_id]
        for hole_id, hole in self.holes.items():
            if hole_id in self.completed_holes:
                continue
            x, y = hole["position"]
            selected = hole_id == decision.get("selected_hole_id")
            color = [0.2, 1.0, 0.2] if selected else [0.75, 0.75, 0.75]
            text = f"{hole_id} {scores.get(hole_id, 0.0):.3f}"
            if selected:
                text = "SELECTED " + text
            self._label_ids[hole_id] = p.addUserDebugText(
                text,
                [x, y, SURFACE_Z + 0.19],
                textColorRGB=color,
                textSize=0.95,
            )
        p.addUserDebugText(
            f"{peg_id} conf={decision.get('confidence', 0.0):.3f} margin={decision.get('margin', 0.0):.3f}",
            [peg["position"][0], peg["position"][1], SURFACE_Z + 0.19],
            textColorRGB=[0.3, 0.75, 1.0],
            textSize=0.95,
        )


class PandaInsertionController:
    """Panda IK/gripper sequence for one active peg."""

    GRIPPER_FINGER_LINKS = (9, 10)

    def __init__(self, scene: MultiObjectPegHoleScene, peg_id: str):
        self.scene = scene
        self.config = scene.config
        self.robot_id = int(scene.robot_id)
        self.peg_id = peg_id
        self.peg = scene.pegs[peg_id]
        self.peg_body_id = int(self.peg["body_id"])
        self.sleep = bool(self.config.sleep and self.config.gui)
        self.motion_steps = int(self.config.motion_steps)
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        self.frames: List[Image.Image] = []

    def _record(self, force: bool = False) -> None:
        if force or not self.frames or len(self.frames) % 8 == 0:
            self.frames.append(self.scene.capture_frame())

    def _step(self, count: int, record: bool = True) -> None:
        for _ in range(max(1, count)):
            p.stepSimulation()
            if record:
                self._record()
            if self.sleep:
                time.sleep(1.0 / 240.0)

    def move_to_position(
        self,
        target_position: Sequence[float],
        target_orientation: Optional[Sequence[float]] = None,
    ) -> None:
        orientation = target_orientation or DOWNWARD_ORIENTATION
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=11,
            targetPosition=list(target_position),
            targetOrientation=orientation,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        for joint_index in range(7):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_index],
                force=500,
                maxVelocity=1.0,
            )
        self._step(self.motion_steps)

    def control_gripper(self, position: float) -> None:
        for joint_index in (9, 10):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=position,
                force=200,
                maxVelocity=0.5,
            )
        self._step(75)

    def _hold_gripper(self, joint_positions: Dict[int, float]) -> None:
        """Hold the measured grasp opening instead of continuing to close."""
        for joint_index in self.GRIPPER_FINGER_LINKS:
            target = float(joint_positions[joint_index])
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200,
                maxVelocity=0.5,
            )

    def set_initial_pose(self) -> None:
        initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        # A direct joint interpolation from the previous hole can sweep the
        # wrist across the row of still-unpicked pegs.  Route the empty hand
        # through a high Cartesian waypoint first; this preserves collision
        # checking while preventing the robot from perturbing the next peg.
        self.move_to_position(
            [0.45, 0.0, 1.05],
            DOWNWARD_ORIENTATION,
        )
        for joint_index, target in enumerate(initial_joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=500,
                maxVelocity=2.0,
            )
        # The original demo gives the arm a longer return-to-home window.
        # This matters after an in-air wrist rotation: 180 steps can leave
        # joint 7 on the previous branch, causing the next IK approach to
        # miss the peg entirely.
        self._step(max(480, self.motion_steps))

    def _attach_grasp_constraint(
        self,
        required_contact_links: Optional[Set[int]] = None,
    ) -> Optional[int]:
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.peg_body_id)
        if not contacts:
            return None
        contact_links = {int(point[3]) for point in contacts}
        if required_contact_links and not required_contact_links.issubset(contact_links):
            return None
        link_state = p.getLinkState(self.robot_id, 11, computeForwardKinematics=True)
        parent_position, parent_orientation = link_state[4], link_state[5]
        object_position, object_orientation = p.getBasePositionAndOrientation(self.peg_body_id)
        inverse_position, inverse_orientation = p.invertTransform(
            parent_position,
            parent_orientation,
        )
        parent_frame_position, parent_frame_orientation = p.multiplyTransforms(
            inverse_position,
            inverse_orientation,
            object_position,
            object_orientation,
        )
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=11,
            childBodyUniqueId=self.peg_body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=parent_frame_position,
            childFramePosition=[0.0, 0.0, 0.0],
            parentFrameOrientation=parent_frame_orientation,
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        p.changeConstraint(constraint_id, maxForce=1000.0)
        return constraint_id

    def _set_grasp_collision_enabled(
        self,
        enabled: bool,
        link_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Toggle selected robot/peg contacts during a validated grasp."""
        links = (
            list(link_indices)
            if link_indices is not None
            else list(range(-1, p.getNumJoints(self.robot_id)))
        )
        for link_index in links:
            p.setCollisionFilterPair(
                self.robot_id,
                self.peg_body_id,
                linkIndexA=link_index,
                linkIndexB=-1,
                enableCollision=1 if enabled else 0,
            )

    def _finger_midpoint_xy(self) -> Tuple[float, float]:
        finger_positions = [
            p.getLinkState(
                self.robot_id,
                link_index,
                computeForwardKinematics=True,
            )[4]
            for link_index in self.GRIPPER_FINGER_LINKS
        ]
        return (
            0.5 * (finger_positions[0][0] + finger_positions[1][0]),
            0.5 * (finger_positions[0][1] + finger_positions[1][1]),
        )

    def _align_gripper_for_pick(
        self,
        peg_position: Sequence[float],
        target_orientation: Sequence[float],
    ) -> dict:
        """Center the open fingers over the peg before lowering to contact."""
        approach_position = [
            float(peg_position[0]),
            float(peg_position[1]),
            float(peg_position[2] + 0.18),
        ]
        approach_error = float("inf")
        for _ in range(3):
            self.move_to_position(approach_position, target_orientation)
            midpoint_x, midpoint_y = self._finger_midpoint_xy()
            error_x = float(peg_position[0] - midpoint_x)
            error_y = float(peg_position[1] - midpoint_y)
            approach_error = math.hypot(error_x, error_y)
            if approach_error <= 0.006:
                break
            approach_position[0] += error_x
            approach_position[1] += error_y

        # Keep the open-finger descent above the peg's upper corner band.  A
        # lower approach can let the collision mesh roll a narrow cuboid
        # before the close command; the post-close contact validator still
        # decides whether the grasp is real.
        pick_height_offset = float(PICK_HEIGHT_OFFSETS[self.peg["shape_family"]])
        pick_position = [
            approach_position[0],
            approach_position[1],
            float(peg_position[2] + pick_height_offset),
        ]
        # Do not chase a low-height IK residual by repeatedly translating the
        # closed-volume target.  That can push a peg sideways before contact.
        # The physical contact validator below decides whether this single
        # centered lowering produced a real grasp.
        self.move_to_position(pick_position, target_orientation)
        midpoint_x, midpoint_y = self._finger_midpoint_xy()
        pre_contact_error = math.hypot(
            float(peg_position[0] - midpoint_x),
            float(peg_position[1] - midpoint_y),
        )
        return {
            "approach_position": approach_position,
            "pick_position": pick_position,
            "pick_height_offset": float(pick_height_offset),
            "approach_xy_error": float(approach_error),
            "pre_contact_xy_error": float(pre_contact_error),
        }

    def _validate_physical_grasp(self, contacts: Sequence[tuple]) -> dict:
        """Reject top/one-sided contacts before creating the lift constraint."""
        contacts_by_link = {
            link_index: [point for point in contacts if int(point[3]) == link_index]
            for link_index in self.GRIPPER_FINGER_LINKS
        }
        finger_positions = {
            link_index: p.getLinkState(
                self.robot_id,
                link_index,
                computeForwardKinematics=True,
            )[4]
            for link_index in self.GRIPPER_FINGER_LINKS
        }
        peg_position, peg_orientation = p.getBasePositionAndOrientation(self.peg_body_id)
        peg_half_height = _vertical_half_height(self.peg["parts"])
        joint_positions = {
            link_index: float(p.getJointState(self.robot_id, link_index)[0])
            for link_index in self.GRIPPER_FINGER_LINKS
        }
        finger_gap = math.hypot(
            finger_positions[9][0] - finger_positions[10][0],
            finger_positions[9][1] - finger_positions[10][1],
        )
        midpoint = (
            0.5 * (finger_positions[9][0] + finger_positions[10][0]),
            0.5 * (finger_positions[9][1] + finger_positions[10][1]),
        )
        centre_offset = math.hypot(
            peg_position[0] - midpoint[0],
            peg_position[1] - midpoint[1],
        )

        side_contacts = {
            link_index: [
                point
                for point in contacts_by_link[link_index]
                if abs(float(point[7][2])) <= 0.75 and float(point[9]) >= 1.0
            ]
            for link_index in self.GRIPPER_FINGER_LINKS
        }
        side_forces = {
            link_index: max(
                [float(point[9]) for point in side_contacts[link_index]] or [0.0]
            )
            for link_index in self.GRIPPER_FINGER_LINKS
        }
        mean_normals = {}
        for link_index in self.GRIPPER_FINGER_LINKS:
            vectors = [point[7] for point in side_contacts[link_index]]
            if not vectors:
                mean_normals[link_index] = (0.0, 0.0, 0.0)
                continue
            mean = [sum(vector[index] for vector in vectors) for index in range(3)]
            norm = math.sqrt(sum(value * value for value in mean)) or 1.0
            mean_normals[link_index] = tuple(value / norm for value in mean)
        normal_dot = sum(
            mean_normals[9][index] * mean_normals[10][index] for index in range(3)
        )
        roll, pitch, yaw = p.getEulerFromQuaternion(peg_orientation)
        # A real two-finger grasp may begin with a modest tilt.  The later
        # constrained air-alignment step corrects it; reject only a severe
        # tilt that is more consistent with a top/edge collision than a
        # usable grasp.
        upright = abs(roll) <= math.radians(30.0) and abs(pitch) <= math.radians(30.0)
        expected_z = SURFACE_Z + peg_half_height
        height_error = abs(float(peg_position[2]) - expected_z)
        criteria = {
            "both_finger_contacts": all(contacts_by_link[index] for index in self.GRIPPER_FINGER_LINKS),
            "side_contact_on_both_fingers": all(side_contacts[index] for index in self.GRIPPER_FINGER_LINKS),
            "opposing_contact_normals": normal_dot <= -0.25,
            "finger_gap_reasonable": 0.012 <= finger_gap <= 0.080,
            "peg_between_fingers": centre_offset <= 0.012,
            "not_fully_closed": not (
                joint_positions[9] <= 0.0045 and joint_positions[10] <= 0.0045
            ),
            "peg_upright": upright,
            "peg_supported_height": height_error <= 0.018,
        }
        return {
            "valid": bool(all(criteria.values())),
            "criteria": criteria,
            "finger_gap": float(finger_gap),
            "finger_midpoint": [float(midpoint[0]), float(midpoint[1])],
            "peg_centre_offset": float(centre_offset),
            "joint_positions": joint_positions,
            "side_contact_forces": side_forces,
            "contact_normal_dot": float(normal_dot),
            "peg_height_error": float(height_error),
            "peg_euler": [float(roll), float(pitch), float(yaw)],
            "peg_position": list(peg_position),
        }

    @staticmethod
    def _relative_object_pose(
        ee_position: Sequence[float],
        ee_orientation: Sequence[float],
        object_position: Sequence[float],
        object_orientation: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        """Return the peg pose expressed in the end-effector frame."""
        inverse_position, inverse_orientation = p.invertTransform(
            ee_position,
            ee_orientation,
        )
        relative_position, relative_orientation = p.multiplyTransforms(
            inverse_position,
            inverse_orientation,
            object_position,
            object_orientation,
        )
        return list(relative_position), list(relative_orientation)

    @staticmethod
    def _ee_pose_for_object_pose(
        object_position: Sequence[float],
        object_orientation: Sequence[float],
        relative_position: Sequence[float],
        relative_orientation: Sequence[float],
    ) -> Tuple[List[float], List[float]]:
        """Solve the EE pose that places a constrained peg at a target pose."""
        inverse_relative_orientation = p.invertTransform(
            [0.0, 0.0, 0.0],
            relative_orientation,
        )[1]
        ee_orientation = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            object_orientation,
            [0.0, 0.0, 0.0],
            inverse_relative_orientation,
        )[1]
        relative_world_position = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            ee_orientation,
            relative_position,
            [0.0, 0.0, 0.0, 1.0],
        )[0]
        ee_position = [
            float(object_position[index] - relative_world_position[index])
            for index in range(3)
        ]
        return ee_position, list(ee_orientation)

    @staticmethod
    def _angle_delta(first: float, second: float) -> float:
        return abs(((first - second + math.pi) % (2.0 * math.pi)) - math.pi)

    @staticmethod
    def _orientation_angle(
        actual_orientation: Sequence[float],
        target_orientation: Sequence[float],
    ) -> float:
        inverse_target = p.invertTransform(
            [0.0, 0.0, 0.0],
            target_orientation,
        )[1]
        delta = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            inverse_target,
            [0.0, 0.0, 0.0],
            actual_orientation,
        )[1]
        scalar = max(-1.0, min(1.0, abs(float(delta[3]))))
        return float(2.0 * math.acos(scalar))

    def _shape_orientation_ok(
        self,
        orientation: Sequence[float],
        target_orientation: Sequence[float],
    ) -> bool:
        """Check uprightness and yaw according to the peg's symmetry."""
        if self.peg["shape_family"] == "sphere":
            return True
        roll, pitch, _ = p.getEulerFromQuaternion(orientation)
        upright = abs(roll) <= math.radians(4.0) and abs(pitch) <= math.radians(4.0)
        if self.peg["shape_family"] == "cylinder":
            return upright
        return bool(upright and self._orientation_angle(orientation, target_orientation) <= math.radians(5.0))

    def _move_held_object_to_pose(
        self,
        object_position: Sequence[float],
        object_orientation: Sequence[float],
        relative_position: Sequence[float],
        relative_orientation: Sequence[float],
        correction_passes: int = 2,
    ) -> dict:
        """Move a constrained peg to a world pose using EE-frame compensation.

        The original Phase 9 path commanded the EE to the hole coordinates,
        even though the peg was offset from the EE at grasp time.  This helper
        solves the rigid transform instead and applies a small closed-loop
        correction for normal IK residuals.  It never writes the peg base
        pose.
        """
        target_position = [float(value) for value in object_position]
        target_orientation = list(object_orientation)
        actual_position: Sequence[float] = target_position
        actual_orientation: Sequence[float] = target_orientation
        for _ in range(max(1, correction_passes)):
            command_position, command_orientation = self._ee_pose_for_object_pose(
                target_position,
                target_orientation,
                relative_position,
                relative_orientation,
            )
            self.move_to_position(command_position, command_orientation)
            actual_position, actual_orientation = p.getBasePositionAndOrientation(
                self.peg_body_id
            )
            position_error = [
                target_position[index] - actual_position[index] for index in range(3)
            ]
            orientation_error = self._orientation_angle(
                actual_orientation,
                target_orientation,
            )
            if math.sqrt(sum(value * value for value in position_error)) <= 0.0035 and orientation_error <= math.radians(2.0):
                break
            # Compensate only the residual position.  The next pass still
            # uses the same desired orientation and rigid grasp transform.
            target_position = [
                target_position[index] + max(-0.025, min(0.025, position_error[index]))
                for index in range(3)
            ]
        final_position, final_orientation = p.getBasePositionAndOrientation(
            self.peg_body_id
        )
        return {
            "position": list(final_position),
            "orientation": list(final_orientation),
            "position_error": float(
                math.sqrt(
                    sum(
                        (final_position[index] - object_position[index]) ** 2
                        for index in range(3)
                    )
                )
            ),
            "xy_error": float(
                math.hypot(
                    final_position[0] - object_position[0],
                    final_position[1] - object_position[1],
                )
            ),
            "orientation_error_rad": self._orientation_angle(
                final_orientation,
                object_orientation,
            ),
        }

    def _inside_hole_opening(
        self,
        hole: dict,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> bool:
        """Check geometric containment, not just centre-point proximity."""
        hole_part = hole["parts"][0]
        peg_part = self.peg["parts"][0]
        delta_x = float(position[0] - hole["position"][0])
        delta_y = float(position[1] - hole["position"][1])
        tolerance = 0.0015
        if hole_part["kind"] == "ring":
            peg_radius = float(peg_part["radius"])
            allowed_centre_radius = float(hole_part["inner_radius"] - peg_radius)
            return math.hypot(delta_x, delta_y) <= allowed_centre_radius + tolerance

        hole_yaw = math.radians(float(hole["yaw_deg"]))
        local_x, local_y = _rotate_xy(delta_x, delta_y, -float(hole["yaw_deg"]))
        actual_yaw = p.getEulerFromQuaternion(orientation)[2]
        yaw_delta = actual_yaw - hole_yaw
        peg_half_x, peg_half_y = peg_part["half_extents"][:2]
        cosine = abs(math.cos(yaw_delta))
        sine = abs(math.sin(yaw_delta))
        projected_x = cosine * peg_half_x + sine * peg_half_y
        projected_y = sine * peg_half_x + cosine * peg_half_y
        hole_half_x, hole_half_y = hole_part["half_extents"][:2]
        return bool(
            abs(local_x) + projected_x <= hole_half_x + tolerance
            and abs(local_y) + projected_y <= hole_half_y + tolerance
        )

    def pick_and_insert(self, hole_id: str) -> dict:
        hole = self.scene.holes[hole_id]
        peg_position, _ = p.getBasePositionAndOrientation(self.peg_body_id)
        peg_half_height = _vertical_half_height(self.peg["parts"])
        grasp_ee_orientation = p.getQuaternionFromEuler(
            [math.pi, 0.0, math.radians(self.peg["yaw_deg"])]
        )

        self.set_initial_pose()
        self.control_gripper(self.gripper_open_position)
        grasp_alignment = self._align_gripper_for_pick(
            peg_position,
            grasp_ee_orientation,
        )
        self.control_gripper(self.gripper_close_position)
        self._step(45)
        grasp_contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.peg_body_id,
        )
        grasp_contact_count = len(grasp_contacts)
        grasp_contact_link_indices = sorted({int(point[3]) for point in grasp_contacts})
        grasp_finger_link_indices = sorted(
            set(grasp_contact_link_indices).intersection(self.GRIPPER_FINGER_LINKS)
        )
        grasp_validation = self._validate_physical_grasp(grasp_contacts)
        relative_position: List[float] = []
        relative_orientation: List[float] = []
        grasp_constraint_id: Optional[int] = None
        grasp_lift = 0.0
        grasped = False
        transport_finger_joint_positions: Dict[int, float] = {}
        transport_grasp_contact_count = 0
        if grasp_validation["valid"]:
            grasp_ee_position, grasp_ee_orientation_actual = p.getLinkState(
                self.robot_id,
                11,
                computeForwardKinematics=True,
            )[4:6]
            grasp_object_position, grasp_object_orientation = p.getBasePositionAndOrientation(
                self.peg_body_id
            )
            relative_position, relative_orientation = self._relative_object_pose(
                grasp_ee_position,
                grasp_ee_orientation_actual,
                grasp_object_position,
                grasp_object_orientation,
            )
            grasp_constraint_id = self._attach_grasp_constraint(
                required_contact_links=set(self.GRIPPER_FINGER_LINKS),
            )
            if grasp_constraint_id is not None:
                # Preserve the measured two-finger opening.  The close command
                # used to remain active here, so the fingers continued toward
                # zero after the fixed hand/object constraint was created.
                # That made the visual fingers pass through the peg.  Keep
                # Keep the robot/peg internal collision disabled during the
                # constrained transport.  The grasp was already validated by
                # real finger contacts; keeping those contacts active while
                # also enforcing the fixed hand/object transform lets the
                # contact solver push the peg sideways.  Holding the measured
                # finger opening prevents the visual fingers from closing
                # through the peg without introducing that solver conflict.
                grasp_hold_joint_positions = {
                    int(link_index): float(position)
                    for link_index, position in grasp_validation["joint_positions"].items()
                }
                self._hold_gripper(grasp_hold_joint_positions)
                self._set_grasp_collision_enabled(False)
                self.move_to_position(
                    [peg_position[0], peg_position[1], SURFACE_Z + 0.24],
                    grasp_ee_orientation,
                )
                lifted_position, _ = p.getBasePositionAndOrientation(
                    self.peg_body_id
                )
                grasp_lift = float(
                    lifted_position[2] - (SURFACE_Z + peg_half_height)
                )
                transport_finger_joint_positions = {
                    int(link_index): float(p.getJointState(self.robot_id, link_index)[0])
                    for link_index in self.GRIPPER_FINGER_LINKS
                }
                transport_grasp_contact_count = len(
                    p.getContactPoints(
                        bodyA=self.robot_id,
                        bodyB=self.peg_body_id,
                    )
                )
                grasped = bool(grasp_lift > 0.035)
            else:
                grasp_hold_joint_positions = {}
        else:
            grasp_hold_joint_positions = {}

        if not grasped:
            if grasp_constraint_id is not None:
                p.removeConstraint(grasp_constraint_id)
                self._set_grasp_collision_enabled(True)
            self.control_gripper(self.gripper_open_position)
            self.move_to_position(
                grasp_alignment["approach_position"],
                grasp_ee_orientation,
            )
            final_position, final_orientation = p.getBasePositionAndOrientation(
                self.peg_body_id
            )
            return {
                "status": "failed",
                "peg_id": self.peg_id,
                "hole_id": hole_id,
                "grasped": False,
                "grasp_contact_count": grasp_contact_count,
                "grasp_contact_link_indices": grasp_contact_link_indices,
                "grasp_finger_link_indices": grasp_finger_link_indices,
                "grasp_lift": float(grasp_lift),
                "grasp_constraint_used": grasp_constraint_id is not None,
                "grasp_hold_joint_positions": grasp_hold_joint_positions,
                "transport_finger_joint_positions": transport_finger_joint_positions,
                "transport_grasp_contact_count": transport_grasp_contact_count,
                "grasp_validation": grasp_validation,
                "grasp_alignment": grasp_alignment,
                "inserted": False,
                "correct_target": hole["target_peg_id"] == self.peg_id,
                "insertion_attempted": False,
                "collision_proxy_used": False,
                "outcome_reason": (
                    "grasp_geometry_invalid"
                    if not grasp_validation["valid"]
                    else "grasp_not_confirmed"
                ),
                "target_position": list(hole["position"]),
                "final_position": list(final_position),
                "final_orientation": list(final_orientation),
                "wall_contact_count": 0,
                "floor_contact_count": 0,
                "physics_release_used": False,
                "teleport_used": False,
            }

        target_position = [
            float(hole["position"][0]),
            float(hole["position"][1]),
        ]
        target_orientation = p.getQuaternionFromEuler(
            [0.0, 0.0, math.radians(hole["yaw_deg"])]
        )
        yaw_delta_deg = abs(
            ((hole["yaw_deg"] - self.peg["yaw_deg"] + 180.0) % 360.0) - 180.0
        )
        floor_top_z = float(hole["floor_top_z"])
        expected_z = floor_top_z + peg_half_height
        wall_height = (
            float(hole["parts"][0]["height"])
            if hole["parts"][0]["kind"] == "ring"
            else BOX_HOLE_WALL_HEIGHT
        )
        # Hold the peg above the wall rim before opening the fingers.  The
        # old release height intersected the low box walls while the peg was
        # still constrained, which let the collision solver push it sideways
        # before the physical drop even began.
        release_height = max(
            expected_z + RELEASE_CLEARANCE,
            SURFACE_Z + wall_height + peg_half_height + RELEASE_CLEARANCE,
        )
        release_position = [
            target_position[0],
            target_position[1],
            release_height,
        ]
        high_object_position = [
            target_position[0],
            target_position[1],
            max(SURFACE_Z + 0.24, release_position[2] + 0.13),
        ]

        # Keep the peg attached while translating and rotating it in the air.
        # First make the held object upright at a safe height, then rotate in
        # small yaw increments.  A single large IK jump can leave the Panda's
        # wrist several degrees off target, which is enough to jam a cuboid.
        held_position, held_orientation = p.getBasePositionAndOrientation(
            self.peg_body_id
        )
        held_yaw = p.getEulerFromQuaternion(held_orientation)[2]
        target_yaw = math.radians(float(hole["yaw_deg"]))
        yaw_delta = ((target_yaw - held_yaw + math.pi) % (2.0 * math.pi)) - math.pi
        self._move_held_object_to_pose(
            high_object_position,
            p.getQuaternionFromEuler([0.0, 0.0, held_yaw]),
            relative_position,
            relative_orientation,
            correction_passes=2,
        )
        air_rotation_steps = max(1, int(math.ceil(abs(yaw_delta) / math.radians(15.0))))
        for rotation_index in range(1, air_rotation_steps + 1):
            rotation_yaw = held_yaw + yaw_delta * rotation_index / air_rotation_steps
            self._move_held_object_to_pose(
                high_object_position,
                p.getQuaternionFromEuler([0.0, 0.0, rotation_yaw]),
                relative_position,
                relative_orientation,
                correction_passes=2,
            )
        # Do not jump directly from the high transport pose to the rim.  For
        # an asymmetric grasp, that large Cartesian z change can make Panda
        # IK select a different wrist branch and move the constrained peg
        # sideways.  A short intermediate descent keeps the same branch while
        # preserving the measured hand/peg rigid transform.
        release_intermediate_position = [
            target_position[0],
            target_position[1],
            release_position[2] + 0.060,
        ]
        self._move_held_object_to_pose(
            release_intermediate_position,
            target_orientation,
            relative_position,
            relative_orientation,
            correction_passes=3,
        )
        release_alignment = self._move_held_object_to_pose(
            release_position,
            target_orientation,
            relative_position,
            relative_orientation,
            correction_passes=5,
        )
        pre_release_position = release_alignment["position"]
        pre_release_orientation = release_alignment["orientation"]
        pre_release_ee_position, pre_release_ee_orientation = p.getLinkState(
            self.robot_id,
            11,
            computeForwardKinematics=True,
        )[4:6]
        pre_release_inside_opening = self._inside_hole_opening(
            hole,
            pre_release_position,
            pre_release_orientation,
        )
        pre_release_alignment_ok = bool(
            release_alignment["xy_error"] <= 0.006
            and pre_release_inside_opening
            and self._shape_orientation_ok(pre_release_orientation, target_orientation)
        )
        if grasp_constraint_id is not None:
            p.removeConstraint(grasp_constraint_id)
        # From this point onward the peg is a free dynamic body.  Opening the
        # fingers and retreating lets gravity, the ring/box walls, and the
        # hole floor determine the final pose.  There is deliberately no
        # resetBasePositionAndOrientation call here.
        self.control_gripper(self.gripper_open_position)
        self._set_grasp_collision_enabled(True)
        retreat_ee_position, retreat_ee_orientation = self._ee_pose_for_object_pose(
            high_object_position,
            target_orientation,
            relative_position,
            relative_orientation,
        )
        self.move_to_position(
            retreat_ee_position,
            retreat_ee_orientation,
        )
        self._step(self.config.settle_steps)

        final_position, final_orientation = p.getBasePositionAndOrientation(self.peg_body_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.peg_body_id)
        wall_contacts = [
            point
            for wall_id in hole["wall_ids"]
            for point in p.getContactPoints(bodyA=self.peg_body_id, bodyB=wall_id)
        ]
        floor_contacts = p.getContactPoints(
            bodyA=self.peg_body_id,
            bodyB=hole["floor_id"],
        )
        displacement = math.hypot(
            final_position[0] - target_position[0],
            final_position[1] - target_position[1],
        )
        height_error = abs(final_position[2] - expected_z)
        speed = math.sqrt(sum(value * value for value in linear_velocity))
        angular_speed = math.sqrt(sum(value * value for value in angular_velocity))
        inside_opening = self._inside_hole_opening(
            hole,
            final_position,
            final_orientation,
        )
        floor_supported = bool(floor_contacts)
        stable = bool(
            speed <= 0.03
            and angular_speed <= 0.08
            and height_error <= 0.018
        )
        orientation_ok = self._shape_orientation_ok(
            final_orientation,
            target_orientation,
        )
        correct_target = hole["target_peg_id"] == self.peg_id
        inserted = bool(
            grasped
            and correct_target
            and pre_release_alignment_ok
            and not wall_contacts
            and inside_opening
            and floor_supported
            and stable
            and orientation_ok
        )
        if inserted:
            outcome_reason = "inserted_physics_release"
        elif not pre_release_alignment_ok:
            outcome_reason = "release_alignment_failed"
        elif wall_contacts:
            outcome_reason = "wall_contact_after_release"
        elif not inside_opening:
            outcome_reason = "peg_not_inside_opening"
        elif not floor_supported:
            outcome_reason = "hole_floor_not_contacted"
        elif not stable:
            outcome_reason = "release_not_stable"
        elif not orientation_ok:
            outcome_reason = "release_orientation_misaligned"
        else:
            outcome_reason = "wrong_target_hole"
        return {
            "status": "executed",
            "peg_id": self.peg_id,
            "hole_id": hole_id,
            "grasped": grasped,
            "grasp_contact_count": grasp_contact_count,
            "grasp_contact_link_indices": grasp_contact_link_indices,
            "grasp_finger_link_indices": grasp_finger_link_indices,
            "grasp_validation": grasp_validation,
            "grasp_hold_joint_positions": grasp_hold_joint_positions,
            "transport_finger_joint_positions": transport_finger_joint_positions,
            "transport_grasp_contact_count": transport_grasp_contact_count,
            "grasp_alignment": grasp_alignment,
            "grasp_lift": float(grasp_lift),
            "grasp_constraint_used": grasp_constraint_id is not None,
            "inserted": inserted,
            "correct_target": correct_target,
            "insertion_attempted": True,
            "collision_proxy_used": True,
            "outcome_reason": outcome_reason,
            "target_position": list(target_position),
            "target_hole_yaw_deg": float(hole["yaw_deg"]),
            "air_rotation_used": bool(yaw_delta_deg > 1e-6),
            "yaw_delta_deg": float(yaw_delta_deg),
            "air_rotation_steps": int(air_rotation_steps),
            "grasp_relative_position": list(relative_position),
            "grasp_relative_orientation": list(relative_orientation),
            "pre_release_position": list(pre_release_position),
            "pre_release_orientation": list(pre_release_orientation),
            "pre_release_ee_position": list(pre_release_ee_position),
            "pre_release_ee_orientation": list(pre_release_ee_orientation),
            "release_position_commanded": list(release_position),
            "pre_release_xy_error": float(release_alignment["xy_error"]),
            "pre_release_pose_error": float(release_alignment["position_error"]),
            "pre_release_inside_opening": pre_release_inside_opening,
            "release_pose_alignment_used": True,
            "physics_release_used": True,
            "teleport_used": False,
            "final_position": list(final_position),
            "final_orientation": list(final_orientation),
            "displacement": float(displacement),
            "expected_z": float(expected_z),
            "height_error": float(height_error),
            "linear_speed": float(speed),
            "angular_speed": float(angular_speed),
            "wall_contact_count": len(wall_contacts),
            "floor_contact_count": len(floor_contacts),
            "inside_hole_opening": inside_opening,
            "floor_supported": floor_supported,
            "stable": stable,
            "orientation_ok": orientation_ok,
            "settle_steps": int(self.config.settle_steps),
        }


def _score_peg(
    scene: MultiObjectPegHoleScene,
    peg_id: str,
    active_hole_ids: Sequence[str],
    matcher_name: str,
    model_name: str,
    cache_dir: Optional[Path],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    _, crops_by_view = scene.capture_observation(peg_id, active_hole_ids)
    peg_by_view = {view: crops_by_view[view]["peg"] for view in ("top", "oblique")}
    holes_by_view = {
        view: {hole_id: crops_by_view[view][hole_id] for hole_id in active_hole_ids}
        for view in ("top", "oblique")
    }
    if matcher_name == "chamfer":
        per_view = [
            chamfer_scores(peg_by_view[view], holes_by_view[view])
            for view in ("top", "oblique")
        ]
        scores = {
            hole_id: float(np.mean([view_scores[hole_id] for view_scores in per_view]))
            for hole_id in active_hole_ids
        }
        definition = "mean negative Chamfer boundary distance over two views"
    elif matcher_name == "clip":
        matcher = CLIPImageImageMatcher(model_name, cache_dir=cache_dir)
        inputs = ObservationInputs(
            observation_mode="gui_render_mask_assisted",
            view_ids=("top", "oblique"),
            peg_by_view=peg_by_view,
            holes_by_view=holes_by_view,
        )
        scores = multi_view_clip_scores(matcher, inputs)
        definition = f"mean image-image CLIP cosine over two views ({model_name})"
    else:
        scores = {
            hole_id: 0.95 if scene.pegs[peg_id]["candidate_fit"][hole_id]["compatible"] else 0.05
            for hole_id in active_hole_ids
        }
        definition = "debug-only analytic corresponding-pair oracle; not a VLM result"
    return scores, {
        "confidence_definition": definition,
        "candidate_source": "all remaining holes; other target holes act as distractors",
    }


def _run_dir(output_dir: Path, seed: int) -> Path:
    run_dir = output_dir / f"phase9_seed{seed}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_dir / f"phase9_seed{seed}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_phase9(config: Phase9Config, project_root: Optional[Path] = None) -> Path:
    """Run one four-object confidence-gated insertion mission."""
    config.validate()
    project_root = project_root or Path.cwd()
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _run_dir(output_root, config.seed)

    connection_id = p.connect(p.GUI if config.gui else p.DIRECT)
    if connection_id < 0:
        raise RuntimeError("Could not connect to PyBullet")

    result = {
        "schema_version": "phase9.v3",
        "status": "error",
        "mission_pass": False,
        "config": asdict(config),
        "connection_mode": "GUI" if config.gui else "DIRECT",
    }
    try:
        scene = MultiObjectPegHoleScene(config)
        scene.build()
        frames: List[Image.Image] = [scene.capture_frame()]
        steps: List[dict] = []
        active_holes = set(scene.hole_order)
        mission_failed = False

        for sequence_index, peg_id in enumerate(scene.peg_order):
            active_ids = [hole_id for hole_id in scene.hole_order if hole_id in active_holes]
            scores, score_metadata = _score_peg(
                scene,
                peg_id,
                active_ids,
                config.matcher,
                config.model_name,
                project_root / ".hf-cache-temp",
            )
            decision = decide_action(
                scores,
                min_confidence=config.min_confidence,
                min_margin=config.min_margin,
                temperature=config.confidence_temperature,
            )
            scene.annotate_scores(peg_id, scores, decision)
            frames.append(scene.capture_frame())
            step = {
                "sequence_index": sequence_index,
                "peg_id": peg_id,
                "shape_family": scene.pegs[peg_id]["shape_family"],
                "peg_position": list(scene.pegs[peg_id]["position"]),
                "peg_yaw_deg": scene.pegs[peg_id]["yaw_deg"],
                "target_hole_id": scene.pegs[peg_id]["target_hole_id"],
                "candidate_hole_ids": active_ids,
                "candidate_fit": {
                    hole_id: scene.pegs[peg_id]["candidate_fit"][hole_id]
                    for hole_id in active_ids
                },
                "scores": scores,
                "decision": decision,
                "score_metadata": score_metadata,
            }
            if decision["execute"]:
                controller = PandaInsertionController(scene, peg_id)
                action = controller.pick_and_insert(decision["selected_hole_id"])
                action["expected_target_hole_id"] = scene.pegs[peg_id]["target_hole_id"]
                action["selected_hole_is_target"] = (
                    decision["selected_hole_id"] == scene.pegs[peg_id]["target_hole_id"]
                )
                step["action"] = action
                frames.extend(controller.frames)
                if action["inserted"] and action["selected_hole_is_target"]:
                    scene.completed_pegs.add(peg_id)
                    scene.completed_holes.add(decision["selected_hole_id"])
                    active_holes.remove(decision["selected_hole_id"])
                    scene.ignore_robot_collision_with_completed_peg(peg_id)
                    action["robot_collision_disabled_after_insert"] = True
                    step["status"] = "success"
                else:
                    step["status"] = "failed"
                    mission_failed = True
            else:
                step["action"] = {
                    "status": "abstained",
                    "reason": decision["uncertainty_reason"],
                }
                step["status"] = "abstained"
                mission_failed = True
            steps.append(step)
            frames.append(scene.capture_frame())

        object_records = []
        for peg_id in sorted(scene.pegs):
            peg = scene.pegs[peg_id]
            hole_id = peg["target_hole_id"]
            hole = scene.holes[hole_id]
            object_records.append(
                {
                    "peg_id": peg_id,
                    "shape_family": peg["shape_family"],
                    "peg_position": list(peg["position"]),
                    "peg_yaw_deg": peg["yaw_deg"],
                    "target_hole_id": hole_id,
                    "hole_position": list(hole["position"]),
                    "hole_yaw_deg": hole["yaw_deg"],
                }
            )
        mission_pass = len(scene.completed_pegs) == config.num_objects and not mission_failed
        result.update(
            {
                "status": "ok",
                "episode_seed": config.seed,
                "geometry": {
                    "configured_per_side_clearance_m": config.clearance,
                    "effective_proxy_per_side_clearance_m": config.clearance * GUI_SCALE,
                    "cuboid_extra_hole_clearance_m": CUBOID_HOLE_EXTRA_CLEARANCE,
                    "cuboid_effective_proxy_per_side_clearance_m": (
                        (config.clearance + CUBOID_HOLE_EXTRA_CLEARANCE) * GUI_SCALE
                    ),
                    "common_gui_linear_scale": GUI_SCALE,
                    "circular_ring_segments": 16,
                    "pick_height_offsets_m": dict(PICK_HEIGHT_OFFSETS),
                },
                "mission_pass": mission_pass,
                "mission": {
                    "object_count": config.num_objects,
                    "completed_count": len(scene.completed_pegs),
                    "completed_peg_ids": sorted(scene.completed_pegs),
                    "completed_hole_ids": sorted(scene.completed_holes),
                    "remaining_hole_ids": sorted(active_holes),
                    "pass_definition": (
                        "all four pegs accepted by the confidence gate, grasped, matched to "
                        "their corresponding holes, and passed the collision-proxy pose check"
                    ),
                },
                "objects": object_records,
                "sequence": scene.peg_order,
                "steps": steps,
                "artifacts": {"summary": "summary.json"},
                "limitation": (
                    "simulation prototype using a perimeter-wall/floor collision proxy; "
                    "oracle is analytic debug control, confidence is not calibrated, and "
                    "the controller is not a complete contact-rich insertion policy"
                ),
            }
        )
        if config.save_video:
            video_path = run_dir / "phase9_mission.gif"
            save_video(frames, video_path, config.video_fps)
            result["artifacts"]["video"] = video_path.name
        _write_json(run_dir / "summary.json", result)
        if config.gui and config.keep_open:
            input("Phase 9 mission complete. Press Enter to close the PyBullet window... ")
        return run_dir
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(run_dir / "summary.json", result)
        raise
    finally:
        if p.isConnected(connection_id):
            p.disconnect(connection_id)


__all__ = [
    "MISSION_SHAPES",
    "Phase9Config",
    "decide_action",
    "run_phase9",
]
