"""Phase 9 GUI demo: confidence-aware Panda peg-hole insertion.

This module connects the ConfMate matching task to the Panda manipulation
logic used by the original color-sorting demo. It intentionally keeps the
experiment separate from the original scene: a camera observation is scored
first, the action is gated by confidence, and only an accepted prediction is
passed to the Panda pick-and-insert controller.

The physical scene uses collision-enabled hole proxies. Therefore the final
insertion result is a GUI demonstration and a physics-assisted validation of
the action sequence; it is not a hardware success claim or a calibrated VLM
confidence estimate.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from .artifacts import save_video
from .peg_hole import SHAPE_FAMILIES, geometric_fit_check, shape_parts
from .phase4 import (
    CLIPImageImageMatcher,
    _rank,
    _softmax_confidence,
    chamfer_scores,
    multi_view_clip_scores,
)


GUI_SCALE = 0.60
GUI_Z_SCALE = 0.75
GUI_TARGET_CLEARANCE_SCALE = 1.15
SURFACE_Z = 0.625
PEG_POSITION = (0.60, 0.27)
CANDIDATE_POSITIONS = (
    (0.35, -0.20),
    (0.60, -0.20),
    (0.85, -0.20),
)
DOWNWARD_ORIENTATION = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])


@dataclass
class Phase9Config:
    """Runtime configuration for the interactive insertion demo."""

    seed: int = 17
    family: str = "rectangle"
    num_candidates: int = 3
    matcher: str = "clip"
    model_name: str = "openai/clip-vit-base-patch32"
    gui: bool = True
    keep_open: bool = False
    save_video: bool = True
    output_dir: str = "runs/phase9"
    video_fps: int = 8
    min_confidence: float = 0.35
    min_margin: float = 0.015
    confidence_temperature: float = 0.50
    motion_steps: int = 240
    settle_steps: int = 180
    sleep: bool = True

    def validate(self) -> None:
        if self.family not in SHAPE_FAMILIES:
            raise ValueError(f"Unknown shape family: {self.family}")
        if self.num_candidates < 2 or self.num_candidates > len(CANDIDATE_POSITIONS):
            raise ValueError(
                f"num_candidates must be between 2 and {len(CANDIDATE_POSITIONS)}"
            )
        if self.matcher not in {"chamfer", "clip", "oracle"}:
            raise ValueError("matcher must be one of: chamfer, clip, oracle")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_margin < 0.0:
            raise ValueError("min_margin must be non-negative")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.motion_steps < 30 or self.settle_steps < 1:
            raise ValueError("motion_steps and settle_steps are too small")


def _scaled_parts(parts: Sequence[dict], scale_multiplier: float = 1.0) -> List[dict]:
    scaled = []
    for part in parts:
        copy = dict(part)
        copy["offset"] = [
            float(part["offset"][0]) * GUI_SCALE * scale_multiplier,
            float(part["offset"][1]) * GUI_SCALE * scale_multiplier,
            float(part["offset"][2]) * GUI_Z_SCALE,
        ]
        if part["kind"] == "box":
            copy["half_extents"] = [
                float(part["half_extents"][0]) * GUI_SCALE * scale_multiplier,
                float(part["half_extents"][1]) * GUI_SCALE * scale_multiplier,
                float(part["half_extents"][2]) * GUI_Z_SCALE,
            ]
        else:
            copy["radius"] = float(part["radius"]) * GUI_SCALE * scale_multiplier
            copy["height"] = float(part["height"]) * GUI_Z_SCALE
        scaled.append(copy)
    return scaled


def _footprint_bounds(parts: Sequence[dict]) -> Tuple[float, float, float, float]:
    extents = []
    for part in parts:
        offset_x, offset_y = part["offset"][:2]
        if part["kind"] == "box":
            half_x, half_y = part["half_extents"][:2]
        elif part["kind"] == "cylinder":
            half_x = half_y = part["radius"]
        else:
            raise ValueError(f"Unsupported part kind: {part['kind']}")
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


def _create_visual_body(
    parts: Sequence[dict],
    position: Sequence[float],
    yaw_deg: float,
    color: Sequence[float],
) -> int:
    if len(parts) == 1 and parts[0]["kind"] == "cylinder":
        part = parts[0]
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=part["radius"],
            length=part["height"],
            rgbaColor=color,
        )
    else:
        if not all(part["kind"] == "box" for part in parts):
            raise ValueError("GUI visualizer supports box arrays or one cylinder")
        visual_id = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * len(parts),
            halfExtents=[part["half_extents"] for part in parts],
            visualFramePositions=[part["offset"] for part in parts],
            rgbaColors=[color] * len(parts),
        )
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)])
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=orientation,
    )


def _create_collision_shape(parts: Sequence[dict]) -> int:
    if len(parts) == 1 and parts[0]["kind"] == "cylinder":
        part = parts[0]
        return p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=part["radius"],
            height=part["height"],
        )
    if not all(part["kind"] == "box" for part in parts):
        raise ValueError("GUI collision model supports box arrays or one cylinder")
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
) -> Tuple[int, List[int], int]:
    """Create a visual hole plus a collision-enabled perimeter/floor proxy."""
    min_x, max_x, min_y, max_y = _footprint_bounds(parts)
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    half_x = 0.5 * (max_x - min_x)
    half_y = 0.5 * (max_y - min_y)
    wall_height = 0.13
    wall_thickness = 0.009
    floor_thickness = 0.012
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw_deg)])

    wall_specs = [
        ([half_x + wall_thickness, wall_thickness, wall_height / 2.0], [centre_x, min_y - wall_thickness]),
        ([half_x + wall_thickness, wall_thickness, wall_height / 2.0], [centre_x, max_y + wall_thickness]),
        ([wall_thickness, half_y + wall_thickness, wall_height / 2.0], [min_x - wall_thickness, centre_y]),
        ([wall_thickness, half_y + wall_thickness, wall_height / 2.0], [max_x + wall_thickness, centre_y]),
    ]
    wall_ids = []
    wall_color = [
        min(1.0, color[0] * 0.75),
        min(1.0, color[1] * 0.75),
        min(1.0, color[2] * 0.75),
        1.0,
    ]
    for half_extents, local_position in wall_specs:
        offset_x, offset_y = _rotate_xy(local_position[0], local_position[1], yaw_deg)
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=wall_color,
        )
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

    floor_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_x + wall_thickness, half_y + wall_thickness, floor_thickness / 2.0],
    )
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_x + wall_thickness, half_y + wall_thickness, floor_thickness / 2.0],
        rgbaColor=[0.06, 0.06, 0.08, 1.0],
    )
    centre_offset_x, centre_offset_y = _rotate_xy(centre_x, centre_y, yaw_deg)
    floor_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[
            position[0] + centre_offset_x,
            position[1] + centre_offset_y,
            surface_z - floor_thickness / 2.0,
        ],
        baseOrientation=orientation,
    )

    visual_height = max(
        part["half_extents"][2] if part["kind"] == "box" else part["height"] / 2.0
        for part in parts
    )
    visual_id = _create_visual_body(
        parts,
        [position[0], position[1], surface_z + visual_height],
        yaw_deg,
        color,
    )
    return visual_id, wall_ids, floor_id


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
    """Apply tie-aware selective prediction before any robot motion."""
    predicted, ranked, top_score, margin = _rank(scores)
    confidence = _softmax_confidence(scores, temperature=temperature)
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


class GuiPegHoleScene:
    """Panda/table/peg-hole scene used by the Phase 9 demo."""

    def __init__(self, config: Phase9Config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.robot_id: Optional[int] = None
        self.peg_body_id: Optional[int] = None
        self.peg_parts: List[dict] = []
        self.peg_position = [PEG_POSITION[0], PEG_POSITION[1]]
        self.peg_yaw_deg = 0.0
        self.hole_order: List[str] = []
        self.target_hole_id = ""
        self.candidate_info: Dict[str, dict] = {}
        self._label_ids: Dict[str, int] = {}

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

        family = self.config.family
        self.peg_yaw_deg = float(self._yaw_for_family(family))
        self.peg_parts = _scaled_parts(shape_parts(family, "peg"))
        peg_height = self._vertical_half_height(self.peg_parts) * 2.0
        peg_visual = _create_visual_body(
            self.peg_parts,
            [self.peg_position[0], self.peg_position[1], SURFACE_Z + peg_height / 2.0],
            self.peg_yaw_deg,
            [0.10, 0.30, 0.90, 1.0],
        )
        peg_collision = _create_collision_shape(self.peg_parts)
        self.peg_body_id = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=peg_collision,
            baseVisualShapeIndex=peg_visual,
            basePosition=[
                self.peg_position[0],
                self.peg_position[1],
                SURFACE_Z + peg_height / 2.0,
            ],
            baseOrientation=p.getQuaternionFromEuler(
                [0.0, 0.0, math.radians(self.peg_yaw_deg)]
            ),
        )
        p.changeDynamics(
            self.peg_body_id,
            -1,
            lateralFriction=0.85,
            spinningFriction=0.15,
            rollingFriction=0.15,
            restitution=0.0,
        )

        self.hole_order = [f"hole_{index:03d}" for index in range(self.config.num_candidates)]
        self.rng.shuffle(self.hole_order)
        positions = list(CANDIDATE_POSITIONS[: self.config.num_candidates])
        self.rng.shuffle(positions)
        self.target_hole_id = self.rng.choice(self.hole_order)
        same_family_candidates = [item for item in self.hole_order if item != self.target_hole_id]
        same_family_distractor = self.rng.choice(same_family_candidates)
        different_families = [item for item in SHAPE_FAMILIES if item != family]

        for index, hole_id in enumerate(self.hole_order):
            position = positions[index]
            is_target = hole_id == self.target_hole_id
            if is_target:
                hole_family = family
                hole_variant = "target"
                hole_yaw = self.peg_yaw_deg
                color = [0.18, 0.18, 0.22, 1.0]
            elif hole_id == same_family_distractor:
                hole_family = family
                hole_variant = "distractor"
                hole_yaw = float(self.rng.choice([0, 30, 60, 90, 120, 150]))
                color = [0.22, 0.22, 0.25, 1.0]
            else:
                hole_family = self.rng.choice(different_families)
                hole_variant = "distractor"
                hole_yaw = float(self.rng.choice([0, 30, 60, 90, 120, 150]))
                color = [0.25, 0.25, 0.28, 1.0]

            fit = geometric_fit_check(
                family,
                hole_family,
                hole_variant,
                self.peg_yaw_deg,
                hole_yaw,
            )
            hole_scale = GUI_TARGET_CLEARANCE_SCALE if hole_variant == "target" else 1.0
            hole_parts = _scaled_parts(
                shape_parts(hole_family, "hole", variant=hole_variant),
                scale_multiplier=hole_scale,
            )
            visual_id, wall_ids, floor_id = _create_hole_proxy(
                hole_parts,
                position,
                hole_yaw,
                SURFACE_Z,
                color,
            )
            self.candidate_info[hole_id] = {
                "position": list(position),
                "hole_family": hole_family,
                "hole_variant": hole_variant,
                "hole_yaw_deg": hole_yaw,
                "fit": fit,
                "visual_id": visual_id,
                "wall_ids": wall_ids,
                "floor_id": floor_id,
                "hole_parts": hole_parts,
            }

        self._set_debug_camera()
        self._add_scene_labels()

    @staticmethod
    def _vertical_half_height(parts: Sequence[dict]) -> float:
        return max(
            part["half_extents"][2] if part["kind"] == "box" else part["height"] / 2.0
            for part in parts
        )

    def _yaw_for_family(self, family: str) -> float:
        if family == "cylinder":
            return 0.0
        if family == "cross":
            return float(self.rng.choice([0, 15, 30, 45]))
        if family == "rectangle":
            return float(self.rng.choice([0, 90]))
        return float(self.rng.choice([0, 30, 60, 90, 120, 180]))

    def _set_debug_camera(self) -> None:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.55,
            cameraYaw=42,
            cameraPitch=-38,
            cameraTargetPosition=[0.60, 0.0, 0.66],
        )

    def _add_scene_labels(self) -> None:
        for hole_id, info in self.candidate_info.items():
            x, y = info["position"]
            self._label_ids[hole_id] = p.addUserDebugText(
                hole_id,
                [x, y, SURFACE_Z + 0.18],
                textColorRGB=[0.75, 0.75, 0.75],
                textSize=1.15,
            )
        self._label_ids["peg"] = p.addUserDebugText(
            "PEG",
            [self.peg_position[0], self.peg_position[1], SURFACE_Z + 0.17],
            textColorRGB=[0.25, 0.55, 1.0],
            textSize=1.15,
        )

    def annotate_scores(self, scores: Dict[str, float], decision: dict) -> None:
        for hole_id, info in self.candidate_info.items():
            x, y = info["position"]
            selected = hole_id == decision.get("selected_hole_id")
            color = [0.2, 1.0, 0.2] if selected else [0.75, 0.75, 0.75]
            text = f"{hole_id} score={scores.get(hole_id, 0.0):.3f}"
            if selected:
                text = "SELECTED " + text
            if hole_id == self.target_hole_id:
                text += " [target for demo check]"
            old_id = self._label_ids.get(hole_id, -1)
            self._label_ids[hole_id] = p.addUserDebugText(
                text,
                [x, y, SURFACE_Z + 0.19],
                textColorRGB=color,
                textSize=1.0,
                replaceItemUniqueId=old_id,
            )
        p.addUserDebugText(
            f"{'EXECUTE INSERTION' if decision.get('execute') else 'ABSTAIN'} | confidence={decision['confidence']:.3f} margin={decision['margin']:.3f}",
            [0.60, 0.42, 1.05],
            textColorRGB=[0.2, 1.0, 0.2] if decision.get("execute") else [1.0, 0.75, 0.2],
            textSize=1.25,
        )

    def _camera(self, view: str) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        if view == "top":
            return [0.60, 0.02, 2.20], [0.60, 0.0, SURFACE_Z], [0.0, 1.0, 0.0]
        if view == "oblique":
            return [1.35, -1.15, 1.45], [0.60, 0.0, SURFACE_Z], [0.0, 0.0, 1.0]
        raise ValueError(f"Unknown view: {view}")

    def capture_observation(self, view: str) -> Tuple[Image.Image, Dict[str, Image.Image]]:
        eye, target, up = self._camera(view)
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
        peg = _mask_crop(rgb, segmentation_array, int(self.peg_body_id))
        holes = {
            hole_id: _mask_crop(rgb, segmentation_array, int(info["visual_id"]))
            for hole_id, info in self.candidate_info.items()
        }
        return Image.fromarray(rgb).convert("RGB"), {"peg": peg, **holes}

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


class PandaInsertionController:
    """IK/gripper controller adapted from the original Panda demo."""

    def __init__(self, scene: GuiPegHoleScene, sleep: bool = True, motion_steps: int = 240):
        self.scene = scene
        self.robot_id = int(scene.robot_id)
        self.peg_body_id = int(scene.peg_body_id)
        self.sleep = bool(sleep)
        self.motion_steps = int(motion_steps)
        self.home_position = [0.30, 0.0, 0.85]
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        self.frames: List[Image.Image] = []

    def _record(self, force: bool = False) -> None:
        if force or not self.frames or len(self.frames) % 4 == 0:
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
        self._step(90)

    def set_initial_pose(self) -> None:
        initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for joint_index, target in enumerate(initial_joint_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=500,
            )
        self._step(240)

    def _attach_grasp_constraint(self) -> Optional[int]:
        """Stabilize a confirmed finger contact while transporting the peg.

        The original demo relies on frictional finger contact.  Small keyed
        parts can rotate under the same position-controlled gripper, so this
        fixed grasp constraint is created only after the gripper has closed
        and PyBullet reports robot/peg contact.  It is removed before release
        and is recorded explicitly in the result.
        """
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.peg_body_id)
        if not contacts:
            return None
        link_state = p.getLinkState(
            self.robot_id,
            11,
            computeForwardKinematics=True,
        )
        parent_position, parent_orientation = link_state[4], link_state[5]
        object_position, object_orientation = p.getBasePositionAndOrientation(
            self.peg_body_id
        )
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
        return p.createConstraint(
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

    def pick_and_insert(self, hole_id: str, settle_steps: int) -> dict:
        info = self.scene.candidate_info[hole_id]
        peg_pos, _ = p.getBasePositionAndOrientation(self.peg_body_id)
        peg_half_height = self.scene._vertical_half_height(self.scene.peg_parts)
        yaw_orientation = p.getQuaternionFromEuler(
            [math.pi, 0.0, math.radians(self.scene.peg_yaw_deg)]
        )

        self.set_initial_pose()
        self.control_gripper(self.gripper_open_position)
        self.move_to_position(
            [peg_pos[0], peg_pos[1], peg_pos[2] + 0.18],
            yaw_orientation,
        )
        self.move_to_position(
            [peg_pos[0], peg_pos[1], peg_pos[2] + 0.015],
            yaw_orientation,
        )
        self.control_gripper(self.gripper_close_position)
        self._step(45)
        grasp_constraint_id = self._attach_grasp_constraint()
        self.move_to_position(
            [peg_pos[0], peg_pos[1], SURFACE_Z + 0.24],
            yaw_orientation,
        )
        lifted_position, _ = p.getBasePositionAndOrientation(self.peg_body_id)
        grasp_lift = float(lifted_position[2] - (SURFACE_Z + peg_half_height))
        grasped = grasp_lift > 0.04

        target_position = info["position"]
        self.move_to_position(
            [target_position[0], target_position[1], SURFACE_Z + 0.23],
            yaw_orientation,
        )
        self.move_to_position(
            [target_position[0], target_position[1], SURFACE_Z + peg_half_height + 0.012],
            yaw_orientation,
        )
        self.control_gripper(self.gripper_open_position)
        if grasp_constraint_id is not None:
            p.removeConstraint(grasp_constraint_id)
        self._step(settle_steps)

        final_position, final_orientation = p.getBasePositionAndOrientation(self.peg_body_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.peg_body_id)
        wall_contacts = [
            point
            for wall_id in info["wall_ids"]
            for point in p.getContactPoints(bodyA=self.peg_body_id, bodyB=wall_id)
        ]
        displacement = math.hypot(
            final_position[0] - target_position[0],
            final_position[1] - target_position[1],
        )
        expected_z = SURFACE_Z + peg_half_height
        height_error = abs(final_position[2] - expected_z)
        speed = math.sqrt(sum(value * value for value in linear_velocity))
        inserted = bool(
            grasped
            and not wall_contacts
            and displacement <= 0.045
            and height_error <= 0.035
            and speed <= 0.10
        )
        reason = "inserted" if inserted else "failed_contact_or_pose_check"
        self.move_to_position(
            [target_position[0], target_position[1], SURFACE_Z + 0.23],
            yaw_orientation,
        )
        return {
            "status": "executed",
            "hole_id": hole_id,
            "grasped": grasped,
            "grasp_lift": float(grasp_lift),
            "grasp_constraint_used": grasp_constraint_id is not None,
            "inserted": inserted,
            "outcome_reason": reason,
            "target_position": list(target_position),
            "final_position": list(final_position),
            "final_orientation": list(final_orientation),
            "displacement": float(displacement),
            "expected_z": float(expected_z),
            "height_error": float(height_error),
            "linear_speed": float(speed),
            "angular_speed": float(math.sqrt(sum(value * value for value in angular_velocity))),
            "wall_contact_count": len(wall_contacts),
            "settle_steps": int(settle_steps),
        }


def _score_scene(
    scene: GuiPegHoleScene,
    matcher_name: str,
    model_name: str,
    cache_dir: Optional[Path],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    observations = {}
    for view in ("top", "oblique"):
        _, crops = scene.capture_observation(view)
        observations[view] = crops
    peg_by_view = {view: crops["peg"] for view, crops in observations.items()}
    holes_by_view = {
        view: {hole_id: crop for hole_id, crop in crops.items() if hole_id != "peg"}
        for view, crops in observations.items()
    }
    if matcher_name == "chamfer":
        per_view = [
            chamfer_scores(peg_by_view[view], holes_by_view[view])
            for view in ("top", "oblique")
        ]
        candidate_ids = list(per_view[0])
        scores = {
            candidate_id: float(
                np.mean([view_scores[candidate_id] for view_scores in per_view])
            )
            for candidate_id in candidate_ids
        }
        definition = "mean negative Chamfer boundary distance over two GUI views"
    elif matcher_name == "clip":
        matcher = CLIPImageImageMatcher(model_name, cache_dir=cache_dir)
        from .phase4 import ObservationInputs

        inputs = ObservationInputs(
            observation_mode="gui_render_mask_assisted",
            view_ids=("top", "oblique"),
            peg_by_view=peg_by_view,
            holes_by_view=holes_by_view,
        )
        scores = multi_view_clip_scores(matcher, inputs)
        definition = f"mean L2-normalized image-image CLIP cosine over two GUI views ({model_name})"
    else:
        scores = {
            hole_id: 0.95 if info["fit"]["compatible"] else 0.05
            for hole_id, info in scene.candidate_info.items()
        }
        definition = "debug-only analytic-fit oracle; not a VLM result"
    return scores, {"confidence_definition": definition}


def run_phase9(config: Phase9Config, project_root: Optional[Path] = None) -> Path:
    """Run one confidence-gated GUI/headless insertion episode."""
    config.validate()
    project_root = project_root or Path.cwd()
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_index = 1
    run_dir = output_root / f"phase9_seed{config.seed}"
    while run_dir.exists():
        run_index += 1
        run_dir = output_root / f"phase9_seed{config.seed}_{run_index}"
    run_dir.mkdir(parents=True, exist_ok=False)

    connection_id = p.connect(p.GUI if config.gui else p.DIRECT)
    if connection_id < 0:
        raise RuntimeError("Could not connect to PyBullet")
    result = {
        "schema_version": "phase9.v1",
        "status": "error",
        "config": asdict(config),
        "connection_mode": "GUI" if config.gui else "DIRECT",
    }
    try:
        scene = GuiPegHoleScene(config)
        scene.build()
        frames = [scene.capture_frame()]
        scores, score_metadata = _score_scene(
            scene,
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
        scene.annotate_scores(scores, decision)
        frames.append(scene.capture_frame())
        action_result = {"status": "abstained", "reason": decision["uncertainty_reason"]}
        if decision["execute"]:
            controller = PandaInsertionController(
                scene,
                sleep=config.sleep and config.gui,
                motion_steps=config.motion_steps,
            )
            action_result = controller.pick_and_insert(
                decision["selected_hole_id"],
                settle_steps=config.settle_steps,
            )
            frames.extend(controller.frames)
        frames.append(scene.capture_frame())
        result.update(
            {
                "status": "ok",
                "episode_seed": config.seed,
                "shape_family": config.family,
                "target_hole_id_for_demo_check": scene.target_hole_id,
                "candidate_hole_ids": scene.hole_order,
                "candidate_fit": {
                    hole_id: info["fit"] for hole_id, info in scene.candidate_info.items()
                },
                "scores": scores,
                "decision": decision,
                "score_metadata": score_metadata,
                "action": action_result,
                "artifacts": {"summary": "summary.json"},
                "limitation": (
                    "GUI Panda action with a collision-enabled perimeter/floor hole proxy; "
                    "not a hardware controller or calibrated confidence evaluation"
                ),
            }
        )
        if config.save_video:
            video_path = run_dir / "phase9_demo.gif"
            save_video(frames, video_path, config.video_fps)
            result["artifacts"]["video"] = video_path.name
        _write_json(run_dir / "summary.json", result)
        if config.gui and config.keep_open:
            input("Phase 9 demo complete. Press Enter to close the PyBullet window... ")
        return run_dir
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(run_dir / "summary.json", result)
        raise
    finally:
        if p.isConnected(connection_id):
            p.disconnect(connection_id)
