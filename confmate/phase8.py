"""PyBullet collision-proxy insertion validation for Phase 8.

The Phase 3 scenes use visual-only hole bodies.  This module reconstructs each
candidate as a small collision experiment: a peg is dropped into a bounded
opening with a floor and perimeter walls.  The result is intentionally called
a collision proxy because the wall ring approximates composite hole
footprints; it is a physics-assisted fit test, not a full contact-rich robot
insertion controller.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pybullet as p

from .metrics import binary_summary
from .peg_hole import shape_parts
from .phase4 import EpisodeRecord, load_episodes


@dataclass
class InsertionConfig:
    steps: int = 720
    time_step: float = 1.0 / 240.0
    start_height: float = 0.35
    floor_z: float = 0.025
    floor_thickness: float = 0.025
    wall_height: float = 0.20
    wall_thickness: float = 0.012
    position_tolerance: float = 0.025
    height_tolerance: float = 0.02
    velocity_tolerance: float = 0.08


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        extents.append((offset_x - half_x, offset_x + half_x, offset_y - half_y, offset_y + half_y))
    if not extents:
        raise ValueError("Cannot build a collision proxy for an empty footprint")
    return (
        min(item[0] for item in extents),
        max(item[1] for item in extents),
        min(item[2] for item in extents),
        max(item[3] for item in extents),
    )


def _collision_shape(parts: Sequence[dict]) -> int:
    if len(parts) == 1 and parts[0]["kind"] == "cylinder":
        part = parts[0]
        return p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=part["radius"],
            height=part["height"],
        )
    if not all(part["kind"] == "box" for part in parts):
        raise ValueError("Collision proxy supports box arrays or a single cylinder")
    return p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * len(parts),
        halfExtents=[part["half_extents"] for part in parts],
        collisionFramePositions=[part["offset"] for part in parts],
    )


def _rotate_xy(x: float, y: float, yaw_deg: float) -> Tuple[float, float]:
    angle = math.radians(yaw_deg)
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )


def _create_hole_proxy(
    hole_family: str,
    hole_variant: str,
    position: Sequence[float],
    hole_yaw_deg: float,
    config: InsertionConfig,
) -> Tuple[int, int, dict]:
    parts = shape_parts(hole_family, "hole", variant=hole_variant)
    min_x, max_x, min_y, max_y = _footprint_bounds(parts)
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    half_x = 0.5 * (max_x - min_x)
    half_y = 0.5 * (max_y - min_y)
    wall_z = config.floor_z + 0.5 * config.wall_height
    t = config.wall_thickness
    wall_half_extents = [
        [half_x + t, t, 0.5 * config.wall_height],
        [half_x + t, t, 0.5 * config.wall_height],
        [t, half_y + t, 0.5 * config.wall_height],
        [t, half_y + t, 0.5 * config.wall_height],
    ]
    wall_positions = [
        [centre_x, min_y - t, wall_z],
        [centre_x, max_y + t, wall_z],
        [min_x - t, centre_y, wall_z],
        [max_x + t, centre_y, wall_z],
    ]
    wall_shape = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * 4,
        halfExtents=wall_half_extents,
        collisionFramePositions=wall_positions,
    )
    orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(hole_yaw_deg)])
    wall_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=wall_shape,
        basePosition=[position[0], position[1], 0.0],
        baseOrientation=orientation,
    )

    floor_half_z = 0.5 * config.floor_thickness
    floor_shape = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX],
        halfExtents=[[half_x + t, half_y + t, floor_half_z]],
        collisionFramePositions=[[centre_x, centre_y, 0.0]],
    )
    floor_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_shape,
        basePosition=[position[0], position[1], config.floor_z - floor_half_z],
        baseOrientation=orientation,
    )
    return wall_body, floor_body, {
        "bounds": [min_x, max_x, min_y, max_y],
        "opening_size": [2.0 * half_x, 2.0 * half_y],
        "proxy_definition": "axis-aligned perimeter wall ring plus collision floor",
    }


def _vertical_half_height(parts: Sequence[dict]) -> float:
    return max(
        part["half_extents"][2] if part["kind"] == "box" else 0.5 * part["height"]
        for part in parts
    )


def run_insertion_trial(
    peg_family: str,
    hole_family: str,
    hole_variant: str,
    peg_yaw_deg: float,
    hole_yaw_deg: float,
    position: Sequence[float],
    analytic_fit: bool,
    config: InsertionConfig,
) -> dict:
    """Drop one peg into one collision-proxy opening and record the outcome."""
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(config.time_step)
    p.setPhysicsEngineParameter(numSolverIterations=100)

    peg_parts = shape_parts(peg_family, "peg")
    peg_shape = _collision_shape(peg_parts)
    peg_orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(peg_yaw_deg)])
    peg_body = p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=peg_shape,
        basePosition=[position[0], position[1], config.start_height],
        baseOrientation=peg_orientation,
    )
    p.changeDynamics(peg_body, -1, lateralFriction=0.65, spinningFriction=0.1, rollingFriction=0.1)
    wall_body, floor_body, proxy = _create_hole_proxy(
        hole_family,
        hole_variant,
        position,
        hole_yaw_deg,
        config,
    )

    wall_contact_seen = False
    max_wall_contacts = 0
    for _ in range(max(1, config.steps)):
        p.stepSimulation()
        wall_contacts = p.getContactPoints(bodyA=peg_body, bodyB=wall_body)
        wall_contact_seen = wall_contact_seen or bool(wall_contacts)
        max_wall_contacts = max(max_wall_contacts, len(wall_contacts))

    final_position, final_orientation = p.getBasePositionAndOrientation(peg_body)
    linear_velocity, angular_velocity = p.getBaseVelocity(peg_body)
    wall_closest = p.getClosestPoints(peg_body, wall_body, distance=0.2)
    wall_distance = min((float(point[8]) for point in wall_closest), default=None)
    peg_height = _vertical_half_height(peg_parts)
    expected_z = config.floor_z + peg_height
    displacement = math.hypot(final_position[0] - position[0], final_position[1] - position[1])
    speed = math.sqrt(sum(value * value for value in linear_velocity))
    height_error = abs(final_position[2] - expected_z)
    inserted = bool(
        not wall_contact_seen
        and displacement <= config.position_tolerance
        and height_error <= config.height_tolerance
        and speed <= config.velocity_tolerance
    )
    if wall_contact_seen:
        reason = "perimeter_wall_contact"
    elif displacement > config.position_tolerance:
        reason = "lateral_displacement"
    elif height_error > config.height_tolerance:
        reason = "not_at_insertion_height"
    elif speed > config.velocity_tolerance:
        reason = "not_settled"
    else:
        reason = "collision_proxy_inserted"
    return {
        "peg_family": peg_family,
        "hole_family": hole_family,
        "hole_variant": hole_variant,
        "peg_yaw_deg": peg_yaw_deg,
        "hole_yaw_deg": hole_yaw_deg,
        "analytic_fit": bool(analytic_fit),
        "inserted": inserted,
        "outcome_reason": reason,
        "position": list(position),
        "final_position": list(final_position),
        "final_orientation": list(final_orientation),
        "displacement": float(displacement),
        "expected_z": float(expected_z),
        "height_error": float(height_error),
        "linear_speed": float(speed),
        "angular_speed": float(math.sqrt(sum(value * value for value in angular_velocity))),
        "wall_contact_seen": bool(wall_contact_seen),
        "max_wall_contacts": int(max_wall_contacts),
        "wall_distance_at_end": wall_distance,
        "proxy": proxy,
        "steps": int(max(1, config.steps)),
    }


def _candidate_metadata(record: EpisodeRecord, candidate_id: str) -> Tuple[List[float], dict]:
    scene_metadata = _read_json(record.episode_dir / "scene_metadata.json")
    positions = scene_metadata.get("candidate_positions", {})
    if candidate_id not in positions:
        raise ValueError(f"Missing candidate position for {record.episode_id}/{candidate_id}")
    fit = record.ground_truth.get("candidate_fit", {}).get(candidate_id)
    if fit is None:
        raise ValueError(f"Missing candidate fit label for {record.episode_id}/{candidate_id}")
    return list(positions[candidate_id]), fit


def evaluate_dataset(
    dataset_root: Path,
    output_root: Path,
    split: str = "test",
    config: Optional[InsertionConfig] = None,
    max_episodes: Optional[int] = None,
) -> Path:
    config = config or InsertionConfig()
    records = load_episodes(dataset_root, split)
    if max_episodes is not None:
        records = records[:max_episodes]
    connection_id = p.connect(p.DIRECT)
    if connection_id < 0:
        raise RuntimeError("Could not connect to PyBullet in DIRECT mode")

    episode_results = []
    try:
        for record in records:
            candidate_trials = []
            peg_yaw = float(record.ground_truth.get("target_yaw_deg") or 0.0)
            for candidate_id in record.ground_truth["candidate_hole_ids"]:
                position, fit = _candidate_metadata(record, candidate_id)
                candidate_trials.append(
                    {
                        "candidate_id": candidate_id,
                        **run_insertion_trial(
                            peg_family=record.ground_truth["shape_family"],
                            hole_family=fit["hole_family"],
                            hole_variant=fit["hole_variant"],
                            peg_yaw_deg=float(fit.get("peg_yaw_deg", peg_yaw)),
                            hole_yaw_deg=float(
                                fit.get(
                                    "hole_yaw_deg",
                                    peg_yaw if fit.get("yaw_aligned") else 0.0,
                                )
                            ),
                            position=position,
                            analytic_fit=bool(fit["compatible"]),
                            config=config,
                        ),
                        "target_candidate": candidate_id == record.ground_truth["target_hole_id"],
                    }
                )
            episode_results.append(
                {
                    "episode_id": record.episode_id,
                    "split": record.split,
                    "seed": record.seed,
                    "shape_family": record.ground_truth["shape_family"],
                    "target_hole_id": record.ground_truth["target_hole_id"],
                    "candidate_trials": candidate_trials,
                    "target_inserted": any(
                        trial["target_candidate"] and trial["inserted"]
                        for trial in candidate_trials
                    ),
                    "all_distractors_rejected": all(
                        not trial["inserted"]
                        for trial in candidate_trials
                        if not trial["target_candidate"]
                    ),
                }
            )
    finally:
        p.disconnect(connection_id)

    all_trials = [trial for episode in episode_results for trial in episode["candidate_trials"]]
    distractor_trials = [trial for trial in all_trials if not trial["target_candidate"]]
    target_trials = [trial for trial in all_trials if trial["target_candidate"]]
    fit_agreement = binary_summary(trial["inserted"] == trial["analytic_fit"] for trial in all_trials)
    target_success = binary_summary(trial["inserted"] for trial in target_trials)
    distractor_rejection = binary_summary(not trial["inserted"] for trial in distractor_trials)
    episode_rejection = binary_summary(episode["all_distractors_rejected"] for episode in episode_results)
    aggregate = {
        "schema_version": "phase8.v1",
        "dataset_root": str(dataset_root),
        "split": split,
        "num_episodes": len(episode_results),
        "num_candidate_trials": len(all_trials),
        "candidate_fit_agreement": fit_agreement["accuracy"],
        "candidate_fit_agreement_ci95": fit_agreement["accuracy_ci95"],
        "target_insertion_success_rate": target_success["accuracy"],
        "target_insertion_success_ci95": target_success["accuracy_ci95"],
        "distractor_rejection_rate": distractor_rejection["accuracy"],
        "distractor_rejection_ci95": distractor_rejection["accuracy_ci95"],
        "episode_all_distractors_rejected_rate": episode_rejection["accuracy"],
        "episode_all_distractors_rejected_ci95": episode_rejection["accuracy_ci95"],
        "wall_contact_rate": sum(trial["wall_contact_seen"] for trial in all_trials) / max(len(all_trials), 1),
        "outcome_counts": {
            reason: sum(trial["outcome_reason"] == reason for trial in all_trials)
            for reason in sorted({trial["outcome_reason"] for trial in all_trials})
        },
        "geometry_definition": "collision proxy with perimeter wall ring and floor; not a full insertion controller",
        "success_definition": "no perimeter wall contact, settled near floor height, and bounded lateral displacement",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episode_results:
            handle.write(json.dumps(episode, sort_keys=True) + "\n")
            episode_path = episodes_dir / split / f"{episode['episode_id']}.json"
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            with episode_path.open("w", encoding="utf-8") as episode_handle:
                json.dump(episode, episode_handle, indent=2, sort_keys=True)
                episode_handle.write("\n")
    with (output_root / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump({"aggregate": aggregate, "config": config.__dict__}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_root
