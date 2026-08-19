"""Controlled peg-hole scene generation for Phase 3.

The generator deliberately keeps the scene geometric and deterministic. It
does not claim to solve insertion dynamics; it produces RGB views, rendered
segmentation artifacts, observation-mode inputs, and explicit ground truth
for the matching stages that follow.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


SHAPE_FAMILIES = (
    "cylinder",
    "rectangle",
    "keyed",
    "cross",
    "L",
    "asymmetric",
)

# The test split is shape-family disjoint from train and validation.
SPLIT_FAMILIES = {
    "train": ("cylinder", "rectangle", "keyed", "cross"),
    "val": ("L",),
    "test": ("asymmetric",),
}

VIEW_CONFIGS = {
    "top": {
        "eye": (0.0, 0.0, 2.35),
        "target": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
    },
    "oblique": {
        "eye": (1.45, -1.55, 1.55),
        "target": (0.0, 0.0, 0.02),
        "up": (0.0, 0.0, 1.0),
    },
}

OBSERVATION_MODE_SPECS = {
    "oracle_crop": {
        "source": "generated ground-truth object mask",
        "uses_ground_truth_mask": True,
        "uses_rendered_segmentation": False,
        "input_description": "masked peg and candidate-hole crops",
    },
    "render_mask_assisted": {
        "source": "PyBullet segmentation buffer",
        "uses_ground_truth_mask": False,
        "uses_rendered_segmentation": True,
        "input_description": "segmentation-derived peg and candidate-hole crops",
    },
    "rgb_only": {
        "source": "RGB camera frame",
        "uses_ground_truth_mask": False,
        "uses_rendered_segmentation": False,
        "input_description": "raw RGB frame only; no candidate labels or object IDs",
    },
}


@dataclass
class PegHoleConfig:
    seed: int = 17
    episodes_per_family: int = 2
    episodes_per_split: Optional[Dict[str, int]] = None
    num_candidates: int = 3
    image_width: int = 480
    image_height: int = 360
    output_dir: str = "datasets/peg_hole_v2"
    views: Tuple[str, ...] = ("top", "oblique")
    observation_modes: Tuple[str, ...] = tuple(OBSERVATION_MODE_SPECS)

    @classmethod
    def from_yaml(cls, path: Path) -> "PegHoleConfig":
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if "peg_hole" in payload:
            payload = payload["peg_hole"] or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")

        values = dict(payload)
        if "views" in values:
            values["views"] = tuple(values["views"])
        if "observation_modes" in values:
            values["observation_modes"] = tuple(values["observation_modes"])
        valid_fields = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in values.items() if key in valid_fields})
        config.validate()
        return config

    def validate(self) -> None:
        if self.episodes_per_family < 1:
            raise ValueError("episodes_per_family must be positive")
        if self.episodes_per_split is not None:
            unknown_splits = set(self.episodes_per_split) - set(SPLIT_FAMILIES)
            if unknown_splits:
                raise ValueError(f"Unknown episode split(s): {sorted(unknown_splits)}")
            if any(int(count) < 1 for count in self.episodes_per_split.values()):
                raise ValueError("episodes_per_split counts must be positive")
        if self.num_candidates < 2:
            raise ValueError("num_candidates must be at least 2")
        if self.num_candidates > 8:
            raise ValueError("num_candidates must be at most 8 for the configured board layout")
        if self.image_width < 64 or self.image_height < 64:
            raise ValueError("image dimensions are too small")
        unknown_views = set(self.views) - set(VIEW_CONFIGS)
        if unknown_views:
            raise ValueError(f"Unknown views: {sorted(unknown_views)}")
        unknown_modes = set(self.observation_modes) - set(OBSERVATION_MODE_SPECS)
        if unknown_modes:
            raise ValueError(f"Unknown observation modes: {sorted(unknown_modes)}")

    def update_from_cli(
        self,
        *,
        seed: Optional[int] = None,
        episodes_per_family: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> "PegHoleConfig":
        if seed is not None:
            self.seed = seed
        if episodes_per_family is not None:
            self.episodes_per_family = episodes_per_family
        if output_dir is not None:
            self.output_dir = output_dir
        self.validate()
        return self

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["views"] = list(self.views)
        payload["observation_modes"] = list(self.observation_modes)
        return payload


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_output_root(path: Path) -> Path:
    """Avoid overwriting an existing dataset from a previous seed."""
    if path.exists() and any(path.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = path.with_name(f"{path.name}_{stamp}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _box(half_x: float, half_y: float, half_z: float, offset=(0.0, 0.0, 0.0)) -> dict:
    return {
        "kind": "box",
        "half_extents": [half_x, half_y, half_z],
        "offset": list(offset),
    }


def _cylinder(radius: float, height: float) -> dict:
    return {"kind": "cylinder", "radius": radius, "height": height, "offset": [0.0, 0.0, 0.0]}


def _scale_parts(parts: Sequence[dict], scale: float, z_scale: float = 1.0) -> List[dict]:
    scaled = []
    for part in parts:
        copy = dict(part)
        copy["offset"] = list(part["offset"])
        if part["kind"] == "box":
            copy["half_extents"] = [
                part["half_extents"][0] * scale,
                part["half_extents"][1] * scale,
                part["half_extents"][2] * z_scale,
            ]
        else:
            copy["radius"] = part["radius"] * scale
            copy["height"] = part["height"] * z_scale
        scaled.append(copy)
    return scaled


def shape_parts(family: str, role: str = "peg", variant: str = "target") -> List[dict]:
    """Return simple 2.5-D geometry pieces for one shape family."""
    if family not in SHAPE_FAMILIES:
        raise ValueError(f"Unknown shape family: {family}")
    if role not in {"peg", "hole"}:
        raise ValueError(f"Unknown role: {role}")

    height = 0.16 if role == "peg" else 0.025
    z_half = height / 2.0
    if family == "cylinder":
        parts = [_cylinder(0.09, height)]
    elif family == "rectangle":
        parts = [_box(0.105, 0.07, z_half)]
    elif family == "keyed":
        parts = [
            _box(0.095, 0.07, z_half),
            _box(0.035, 0.025, z_half, offset=(0.115, 0.0, 0.0)),
        ]
    elif family == "cross":
        parts = [
            _box(0.035, 0.12, z_half),
            _box(0.12, 0.035, z_half),
        ]
    elif family == "L":
        parts = [
            _box(0.035, 0.12, z_half, offset=(-0.085, 0.0, 0.0)),
            _box(0.12, 0.035, z_half, offset=(0.0, -0.085, 0.0)),
        ]
    else:  # asymmetric
        parts = [
            _box(0.08, 0.055, z_half, offset=(-0.025, 0.0, 0.0)),
            _box(0.035, 0.02, z_half, offset=(0.095, 0.035, 0.0)),
            _box(0.02, 0.04, z_half, offset=(0.075, -0.06, 0.0)),
        ]

    # The target hole is slightly larger to represent clearance. Distractors
    # are intentionally close in scale but not functionally compatible.
    if role == "hole":
        parts = _scale_parts(parts, 1.28, z_scale=1.0)
        if variant == "distractor":
            parts = _scale_parts(parts, 0.73, z_scale=1.0)
    return parts


def geometric_fit_check(
    peg_family: str,
    hole_family: str,
    hole_variant: str,
    peg_yaw_deg: float,
    hole_yaw_deg: float,
    clearance_epsilon: float = 1e-6,
) -> dict:
    """Check analytic footprint clearance for one candidate pair.

    This is a geometric compatibility label, not insertion dynamics. It makes
    the distinction between appearance matching and nominal fit explicit.
    """
    family_match = peg_family == hole_family
    variant_is_target = hole_variant == "target"
    yaw_delta = abs(((hole_yaw_deg - peg_yaw_deg + 180.0) % 360.0) - 180.0)
    yaw_aligned = yaw_delta <= 1e-6
    base = {
        "fit_margin": None,
        "family_match": family_match,
        "variant_is_target": variant_is_target,
        "yaw_aligned": yaw_aligned,
    }
    if not family_match:
        return {**base, "compatible": False, "reason": "shape_family_mismatch"}
    if not variant_is_target:
        return {**base, "compatible": False, "reason": "clearance_variant_is_distractor"}
    if not yaw_aligned:
        return {**base, "compatible": False, "reason": "yaw_misaligned"}

    peg_parts = shape_parts(peg_family, "peg")
    hole_parts = shape_parts(hole_family, "hole", variant=hole_variant)
    if len(peg_parts) != len(hole_parts):
        return {**base, "compatible": False, "reason": "part_count_mismatch"}

    margins = []
    for peg_part, hole_part in zip(peg_parts, hole_parts):
        if peg_part["kind"] != hole_part["kind"]:
            return {**base, "compatible": False, "reason": "part_kind_mismatch"}
        if peg_part["kind"] == "box":
            peg_size = [2.0 * value for value in peg_part["half_extents"][:2]]
            hole_size = [2.0 * value for value in hole_part["half_extents"][:2]]
        else:
            peg_size = [2.0 * peg_part["radius"]] * 2
            hole_size = [2.0 * hole_part["radius"]] * 2
        margins.extend(hole - peg for hole, peg in zip(hole_size, peg_size))
    fit_margin = float(min(margins)) if margins else 0.0
    compatible = fit_margin > clearance_epsilon
    return {
        **base,
        "compatible": compatible,
        "reason": "positive_analytic_clearance" if compatible else "non_positive_clearance",
        "fit_margin": fit_margin,
    }


def _candidate_positions(num_candidates: int) -> List[Tuple[float, float]]:
    positions = [
        (-0.65, -0.42), (-0.22, -0.42), (0.22, -0.42), (0.65, -0.42),
        (-0.65, -0.10), (-0.22, -0.10), (0.22, -0.10), (0.65, -0.10),
    ]
    return positions[:num_candidates]


def _yaw_for_family(family: str, rng: random.Random) -> Tuple[Optional[float], Optional[int]]:
    if family == "cylinder":
        return None, None
    if family == "cross":
        return float(rng.choice([0, 15, 30, 45])), 4
    if family == "rectangle":
        return float(rng.choice([0, 90])), 2
    return float(rng.choice([0, 30, 60, 90, 120, 180])), 1


def _create_visual_body(parts: Sequence[dict], position: Sequence[float], yaw_deg: float, color) -> int:
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
            raise ValueError("Only box arrays or a single cylinder are supported")
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


def _create_board() -> int:
    collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.9, 0.6, 0.025])
    visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.9, 0.6, 0.025],
        rgbaColor=[0.70, 0.55, 0.35, 1.0],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[0.0, 0.0, 0.0],
    )


def _mask_for_body(segmentation: np.ndarray, body_id: int) -> np.ndarray:
    # PyBullet stores the object ID in the low 24 bits and the link ID above it.
    return (segmentation.astype(np.int64) & 0x00FFFFFF) == body_id


def _crop(image: np.ndarray, mask: np.ndarray, padding: int) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return Image.fromarray(image.copy())
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(image.shape[1], int(xs.max()) + padding + 1)
    y2 = min(image.shape[0], int(ys.max()) + padding + 1)
    cropped = image[y1:y2, x1:x2].copy()
    local_mask = mask[y1:y2, x1:x2]
    cropped[~local_mask] = 255
    return Image.fromarray(cropped)


def _save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


class PegHoleScene:
    """One deterministic PyBullet scene and its camera observations."""

    def __init__(self, config: PegHoleConfig, scene_seed: int, family: str, episode_id: str):
        self.config = config
        self.scene_seed = scene_seed
        self.family = family
        self.episode_id = episode_id
        self.rng = random.Random(scene_seed)
        self.body_ids: Dict[str, int] = {}
        self.hole_order: List[str] = []
        self.target_hole_id: str = ""
        self.candidate_fit: Dict[str, dict] = {}
        self.candidate_positions: Dict[str, Tuple[float, float]] = {}
        self.target_yaw_deg, self.yaw_symmetry_order = _yaw_for_family(family, self.rng)

    def build(self) -> dict:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.body_ids = {}
        self.hole_order = []
        self.candidate_fit = {}
        self.candidate_positions = {}
        _create_board()

        peg_yaw = self.target_yaw_deg or 0.0
        self.body_ids["peg_000"] = _create_visual_body(
            shape_parts(self.family, "peg"),
            position=[0.0, 0.30, 0.14],
            yaw_deg=peg_yaw,
            color=[0.10, 0.30, 0.90, 1.0],
        )

        self.hole_order = [
            f"hole_{index:03d}" for index in range(self.config.num_candidates)
        ]
        self.rng.shuffle(self.hole_order)
        hole_positions = _candidate_positions(self.config.num_candidates)
        self.rng.shuffle(hole_positions)
        self.target_hole_id = self.rng.choice(self.hole_order)
        same_family_distractor_id = self.rng.choice(
            [hole_id for hole_id in self.hole_order if hole_id != self.target_hole_id]
        )
        different_families = [family for family in SHAPE_FAMILIES if family != self.family]

        for order_index, hole_id in enumerate(self.hole_order):
            self.candidate_positions[hole_id] = hole_positions[order_index]
            is_target = hole_id == self.target_hole_id
            if is_target:
                hole_family = self.family
                hole_variant = "target"
                parts = shape_parts(hole_family, "hole", variant=hole_variant)
                hole_yaw = peg_yaw
            elif hole_id == same_family_distractor_id:
                # A close-size, same-family distractor that still cannot mate.
                hole_family = self.family
                hole_variant = "distractor"
                parts = shape_parts(hole_family, "hole", variant=hole_variant)
                hole_yaw = float(self.rng.choice([0, 30, 60, 90, 120, 150]))
            else:
                hole_family = self.rng.choice(different_families)
                hole_variant = "distractor"
                parts = shape_parts(hole_family, "hole", variant=hole_variant)
                hole_yaw = float(self.rng.choice([0, 30, 60, 90, 120, 150]))
            self.candidate_fit[hole_id] = {
                **geometric_fit_check(
                    self.family,
                    hole_family,
                    hole_variant,
                    peg_yaw,
                    hole_yaw,
                ),
                "peg_family": self.family,
                "hole_family": hole_family,
                "hole_variant": hole_variant,
            }
            x, y = self.candidate_positions[hole_id]
            self.body_ids[hole_id] = _create_visual_body(
                parts,
                position=[x, y, 0.0625],
                yaw_deg=hole_yaw,
                color=[0.055, 0.055, 0.065, 1.0],
            )

        p.stepSimulation()
        return {
            "peg_id": "peg_000",
            "target_hole_id": self.target_hole_id,
            "candidate_hole_ids": list(self.hole_order),
            "candidate_fit": self.candidate_fit,
            "target_yaw_deg": self.target_yaw_deg,
            "yaw_symmetry_order": self.yaw_symmetry_order,
        }

    def render(self, view_id: str) -> Tuple[np.ndarray, np.ndarray]:
        view = VIEW_CONFIGS[view_id]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=view["eye"],
            cameraTargetPosition=view["target"],
            cameraUpVector=view["up"],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=self.config.image_width / self.config.image_height,
            nearVal=0.01,
            farVal=4.0,
        )
        _, _, rgba, _, segmentation = p.getCameraImage(
            width=self.config.image_width,
            height=self.config.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        rgba_array = np.asarray(rgba, dtype=np.uint8).reshape(
            self.config.image_height, self.config.image_width, 4
        )
        segmentation_array = np.asarray(segmentation, dtype=np.int32).reshape(
            self.config.image_height, self.config.image_width
        )
        return rgba_array[:, :, :3], segmentation_array

    def ground_truth(self, split: str, view_ids: Iterable[str]) -> dict:
        return {
            "episode_id": self.episode_id,
            "split": split,
            "seed": self.scene_seed,
            "shape_family": self.family,
            "geometry_instance_id": f"{self.family}_instance_{self.scene_seed}",
            "peg_id": "peg_000",
            "target_hole_id": self.target_hole_id,
            "target_yaw_deg": self.target_yaw_deg,
            "yaw_symmetry_order": self.yaw_symmetry_order,
            "candidate_hole_ids": list(self.hole_order),
            "valid_target_hole_ids": [self.target_hole_id],
            "candidate_fit": self.candidate_fit,
            "fit_check_definition": (
                "analytic ordered-part footprint clearance; not insertion dynamics"
            ),
            "occlusion_ratio": 0.0,
            "view_ids": list(view_ids),
            "observation_modes": list(self.config.observation_modes),
        }


def _write_observations(
    episode_dir: Path,
    scene: PegHoleScene,
    rendered: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ground_truth: dict,
) -> None:
    views_dir = episode_dir / "views"
    observations_dir = episode_dir / "observations"
    views_dir.mkdir(parents=True, exist_ok=True)
    observations_dir.mkdir(parents=True, exist_ok=True)

    for view_id, (rgb, segmentation) in rendered.items():
        view_dir = views_dir / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        _save_rgb(view_dir / "rgb.png", rgb)
        np.save(view_dir / "segmentation.npy", segmentation)

        for mode in scene.config.observation_modes:
            mode_dir = observations_dir / mode / view_id
            mode_dir.mkdir(parents=True, exist_ok=True)
            spec = OBSERVATION_MODE_SPECS[mode]
            metadata = {
                "episode_id": scene.episode_id,
                "view_id": view_id,
                "observation_mode": mode,
                "source": spec["source"],
                "uses_ground_truth_mask": spec["uses_ground_truth_mask"],
                "uses_rendered_segmentation": spec["uses_rendered_segmentation"],
                "candidate_count": len(scene.hole_order),
                "input_description": spec["input_description"],
            }
            if mode == "rgb_only":
                _save_rgb(mode_dir / "rgb.png", rgb)
            else:
                for index, object_name in enumerate(["peg_000"] + scene.hole_order):
                    body_id = scene.body_ids[object_name]
                    mask = _mask_for_body(segmentation, body_id)
                    padding = 10 if mode == "oracle_crop" else 4
                    filename = "peg.png" if object_name == "peg_000" else f"candidate_{index - 1:03d}.png"
                    _crop(rgb, mask, padding).save(mode_dir / filename)
            _write_json(mode_dir / "metadata.json", metadata)

    _write_json(episode_dir / "ground_truth.json", ground_truth)
    scene_metadata = {
        "episode_id": scene.episode_id,
        "shape_family": scene.family,
        "body_ids_are_generation_artifacts": True,
        "objects": {name: {"body_id": body_id} for name, body_id in scene.body_ids.items()},
        "candidate_positions": {
            name: list(position) for name, position in scene.candidate_positions.items()
        },
        "candidate_fit": scene.candidate_fit,
    }
    _write_json(episode_dir / "scene_metadata.json", scene_metadata)


def generate_dataset(config: PegHoleConfig, project_root: Optional[Path] = None) -> Path:
    """Generate a fixed-seed, shape-family-aware peg-hole dataset."""
    config.validate()
    project_root = project_root or Path.cwd()
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = _prepare_output_root(output_root)

    master_rng = random.Random(config.seed)
    manifest = {
        "dataset_name": "confmate_peg_hole_v2",
        "generator_version": "phase3.5",
        "seed": config.seed,
        "config": config.to_dict(),
        "split_policy": {
            "type": "shape_family_disjoint",
            "families": {split: list(families) for split, families in SPLIT_FAMILIES.items()},
            "test_is_disjoint_from_train_and_val": True,
        },
        "episode_independence": {
            "unit": "one deterministic PyBullet scene per scene_seed",
            "scene_seed_source": "master seed PRNG draw for each generated scene",
            "test_episode_count": (
                config.episodes_per_split.get("test", config.episodes_per_family)
                if config.episodes_per_split
                else config.episodes_per_family
            ),
        },
        "candidate_assignment_policy": {
            "target_hole_id": "uniformly sampled from candidate IDs per episode",
            "candidate_order": "shuffled per episode before crop-file assignment",
            "candidate_positions": "shuffled per episode",
            "tie_policy": "evaluation must abstain when the top two scores are equal",
        },
        "observation_modes": OBSERVATION_MODE_SPECS,
        "episodes": [],
    }

    connection_id = p.connect(p.DIRECT)
    if connection_id < 0:
        raise RuntimeError("Could not connect to PyBullet in DIRECT mode")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        for split, families in SPLIT_FAMILIES.items():
            for family in families:
                episodes_per_family = (
                    config.episodes_per_split.get(split, config.episodes_per_family)
                    if config.episodes_per_split
                    else config.episodes_per_family
                )
                for episode_index in range(episodes_per_family):
                    scene_seed = master_rng.randrange(0, 2**31 - 1)
                    episode_id = f"{split}_{family}_{episode_index:03d}"
                    scene = PegHoleScene(config, scene_seed, family, episode_id)
                    scene.build()
                    rendered = {view: scene.render(view) for view in config.views}
                    ground_truth = scene.ground_truth(split, config.views)
                    episode_dir = output_root / split / episode_id
                    _write_observations(episode_dir, scene, rendered, ground_truth)
                    manifest["episodes"].append(
                        {
                            "episode_id": episode_id,
                            "split": split,
                            "shape_family": family,
                            "geometry_instance_id": ground_truth["geometry_instance_id"],
                            "ground_truth": str(episode_dir.relative_to(output_root) / "ground_truth.json"),
                        }
                    )
    finally:
        p.disconnect(connection_id)

    manifest["split_counts"] = {
        split: sum(1 for episode in manifest["episodes"] if episode["split"] == split)
        for split in SPLIT_FAMILIES
    }
    _write_json(output_root / "manifest.json", manifest)
    return output_root
