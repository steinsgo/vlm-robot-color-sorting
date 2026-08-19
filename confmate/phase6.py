"""Confidence, abstention, and partial-observation evaluation for Phase 6.

The Phase 3 dataset contains two rendered views and three native candidates.
This module keeps those native inputs intact and creates deterministic evaluation
variants for occlusion, a synthetic third view, and larger candidate pools.
Those variants are labeled in every result so they are not confused with new
PyBullet camera observations.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .phase4 import EpisodeRecord, load_episodes, load_observation
from .phase4 import chamfer_distance
from .metrics import binary_summary


OCCLUSION_LEVELS = {
    "none": 0.0,
    "light": 0.15,
    "moderate": 0.35,
    "heavy": 0.55,
}


@dataclass
class CandidateAsset:
    source_episode_id: str
    source_family: str
    candidate_id: str
    images_by_view: Dict[str, Image.Image]


def _rank(scores: Mapping[str, float]) -> Tuple[Optional[str], List[str], float, float]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return None, [], 0.0, 0.0
    top1 = float(ranked[0][1])
    top2 = float(ranked[1][1]) if len(ranked) > 1 else top1
    margin = top1 - top2
    tied = len(ranked) > 1 and math.isclose(top1, top2, rel_tol=1e-8, abs_tol=1e-8)
    return (None if tied else ranked[0][0]), [name for name, _ in ranked], top1, margin


def _margin_confidence(top1: float, top2: float) -> float:
    """Normalize the top-1/top-2 score gap to [0, 1]."""
    margin = max(0.0, top1 - top2)
    denominator = abs(top1) + abs(top2) + 1e-8
    return float(np.clip(margin / denominator, 0.0, 1.0))


def _softmax_confidence(scores: Mapping[str, float], temperature: float = 0.1) -> float:
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if values.size == 0:
        return 0.0
    shifted = (values - values.max()) / max(temperature, 1e-8)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return float(np.clip(probabilities.max(), 0.0, 1.0))


def _apply_occlusion(image: Image.Image, ratio: float, seed: int) -> Image.Image:
    """Hide a deterministic rectangular region with the crop background color."""
    result = image.convert("RGB").copy()
    if ratio <= 0.0:
        return result
    width, height = result.size
    target_area = max(1.0, ratio * width * height)
    rng = random.Random(seed)
    aspect = rng.uniform(0.55, 1.8)
    block_width = min(width, max(1, int(math.sqrt(target_area * aspect))))
    block_height = min(height, max(1, int(target_area / block_width)))
    left = rng.randint(0, max(0, width - block_width))
    top = rng.randint(0, max(0, height - block_height))
    draw = ImageDraw.Draw(result)
    draw.rectangle(
        [left, top, left + block_width - 1, top + block_height - 1],
        fill=(255, 255, 255),
    )
    return result


def _synthetic_flip(image: Image.Image) -> Image.Image:
    """Create a deterministic third image plane from an existing crop.

    This is a view-augmentation control, not a claim of a third PyBullet camera.
    """
    return ImageOps.mirror(image.convert("RGB"))


def _make_view_set(
    source_peg: Mapping[str, Image.Image],
    source_holes: Mapping[str, Mapping[str, Image.Image]],
    view_ids: Sequence[str],
    occlusion_ratio: float,
    seed: int,
) -> Tuple[Dict[str, Image.Image], Dict[str, Dict[str, Image.Image]]]:
    actual_views = [view for view in view_ids if view != "synthetic_flip"]
    peg_by_view: Dict[str, Image.Image] = {}
    holes_by_view: Dict[str, Dict[str, Image.Image]] = {}
    for view_index, view in enumerate(actual_views):
        peg_by_view[view] = _apply_occlusion(
            source_peg[view], occlusion_ratio, seed + 101 * (view_index + 1)
        )
        holes_by_view[view] = {}
        for candidate_index, (candidate_id, image) in enumerate(source_holes[view].items()):
            holes_by_view[view][candidate_id] = _apply_occlusion(
                image,
                occlusion_ratio,
                seed + 1009 * (view_index + 1) + candidate_index,
            )

    if "synthetic_flip" in view_ids:
        if "top" not in peg_by_view:
            raise ValueError("synthetic_flip requires the top view")
        peg_by_view["synthetic_flip"] = _synthetic_flip(peg_by_view["top"])
        holes_by_view["synthetic_flip"] = {
            candidate_id: _synthetic_flip(image)
            for candidate_id, image in holes_by_view["top"].items()
        }
    return peg_by_view, holes_by_view


def _score_views(
    peg_by_view: Mapping[str, Image.Image],
    holes_by_view: Mapping[str, Mapping[str, Image.Image]],
    view_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    per_view: Dict[str, Dict[str, float]] = {}
    for view in view_ids:
        per_view[view] = {
            candidate_id: -chamfer_distance(peg_by_view[view], image)
            for candidate_id, image in holes_by_view[view].items()
        }
    candidate_ids = list(per_view[view_ids[0]])
    fused = {
        candidate_id: float(np.mean([per_view[view][candidate_id] for view in view_ids]))
        for candidate_id in candidate_ids
    }
    return per_view, fused


def _stability_confidence(
    peg_by_view: Mapping[str, Image.Image],
    holes_by_view: Mapping[str, Mapping[str, Image.Image]],
    view_ids: Sequence[str],
    base_prediction: Optional[str],
    seed: int,
    repeats: int,
) -> float:
    if base_prediction is None:
        return 0.0
    if repeats <= 0:
        return 1.0
    stable = 0
    for repeat in range(repeats):
        perturbed_peg: Dict[str, Image.Image] = {}
        perturbed_holes: Dict[str, Dict[str, Image.Image]] = {}
        for view_index, view in enumerate(view_ids):
            perturbed_peg[view] = _apply_occlusion(
                peg_by_view[view], 0.08, seed + repeat * 10007 + view_index
            )
            perturbed_holes[view] = {
                candidate_id: _apply_occlusion(
                    image,
                    0.08,
                    seed + repeat * 10007 + view_index * 101 + candidate_index,
                )
                for candidate_index, (candidate_id, image) in enumerate(
                    holes_by_view[view].items()
                )
            }
        _, fused = _score_views(perturbed_peg, perturbed_holes, view_ids)
        prediction, _, _, _ = _rank(fused)
        stable += int(prediction == base_prediction)
    return float(stable / repeats)


def _build_candidate_pool(
    records: Sequence[EpisodeRecord], mode: str, views: Sequence[str]
) -> List[CandidateAsset]:
    pool: List[CandidateAsset] = []
    for record in records:
        inputs = load_observation(record, mode, views)
        if inputs is None:
            continue
        for candidate_id in record.ground_truth["candidate_hole_ids"]:
            pool.append(
                CandidateAsset(
                    source_episode_id=record.episode_id,
                    source_family=record.ground_truth["shape_family"],
                    candidate_id=candidate_id,
                    images_by_view={
                        view: inputs.holes_by_view[view][candidate_id].copy()
                        for view in views
                    },
                )
            )
    return pool


def _candidate_assets(
    record: EpisodeRecord,
    mode: str,
    views: Sequence[str],
    desired_count: int,
    distractor_type: str,
    pool: Sequence[CandidateAsset],
) -> Optional[Tuple[Dict[str, Image.Image], Dict[str, Dict[str, Image.Image]], str]]:
    inputs = load_observation(record, mode, views)
    if inputs is None:
        return None
    native_ids = list(record.ground_truth["candidate_hole_ids"])
    if desired_count < len(native_ids):
        native_ids = native_ids[:desired_count]
    if desired_count <= len(native_ids):
        return (
            {view: inputs.peg_by_view[view].copy() for view in views},
            {
                view: {candidate_id: inputs.holes_by_view[view][candidate_id].copy() for candidate_id in native_ids}
                for view in views
            },
            "native",
        )

    if distractor_type not in {"similar", "different"}:
        raise ValueError(f"Unsupported distractor_type={distractor_type!r} for {desired_count} candidates")
    desired_family = record.ground_truth["shape_family"]
    source_pool = [
        asset
        for asset in pool
        if asset.source_episode_id != record.episode_id
        and (
            (distractor_type == "similar" and asset.source_family == desired_family)
            or (distractor_type == "different" and asset.source_family != desired_family)
        )
    ]
    fallback = False
    if len(source_pool) < desired_count - len(native_ids):
        source_pool = [asset for asset in pool if asset.source_episode_id != record.episode_id]
        fallback = True
    if len(source_pool) < desired_count - len(native_ids):
        return None

    holes_by_view = {
        view: {candidate_id: inputs.holes_by_view[view][candidate_id].copy() for candidate_id in native_ids}
        for view in views
    }
    used = set(native_ids)
    for index, asset in enumerate(source_pool[: desired_count - len(native_ids)]):
        generated_id = f"distractor_{distractor_type}_{index:03d}"
        while generated_id in used:
            generated_id += "_x"
        used.add(generated_id)
        for view in views:
            holes_by_view[view][generated_id] = asset.images_by_view[view].copy()
    label = distractor_type + ("_fallback" if fallback else "")
    return (
        {view: inputs.peg_by_view[view].copy() for view in views},
        holes_by_view,
        label,
    )


def evaluate_sample(
    record: EpisodeRecord,
    mode: str,
    occlusion_name: str,
    occlusion_ratio: float,
    view_set_name: str,
    view_ids: Sequence[str],
    candidate_count: int,
    distractor_type: str,
    peg_by_view: Mapping[str, Image.Image],
    holes_by_view: Mapping[str, Mapping[str, Image.Image]],
    seed: int,
    stability_repeats: int,
) -> dict:
    varied_peg, varied_holes = _make_view_set(
        peg_by_view,
        holes_by_view,
        view_ids,
        occlusion_ratio,
        seed,
    )
    per_view, fused = _score_views(varied_peg, varied_holes, view_ids)
    predicted, ranked, top1, margin = _rank(fused)
    target = record.ground_truth["target_hole_id"]
    view_predictions = {view: _rank(per_view[view])[0] for view in view_ids}
    votes = Counter(prediction for prediction in view_predictions.values() if prediction is not None)
    consistency = max(votes.values()) / len(view_ids) if votes else 0.0
    confidence_values = {
        "margin": _margin_confidence(top1, top1 - margin),
        "multi_view_consistency": float(consistency),
        "stability": _stability_confidence(
            varied_peg,
            varied_holes,
            view_ids,
            predicted,
            seed + 7919,
            stability_repeats,
        ),
    }
    return {
        "episode_id": record.episode_id,
        "split": record.split,
        "seed": record.seed,
        "observation_mode": mode,
        "shape_family": record.ground_truth["shape_family"],
        "occlusion_level": occlusion_name,
        "occlusion_ratio": occlusion_ratio,
        "view_set": view_set_name,
        "views": list(view_ids),
        "evaluation_seed": seed,
        "view_note": "synthetic_flip is a deterministic augmentation, not a third PyBullet camera view"
        if "synthetic_flip" in view_ids
        else "native PyBullet crop views",
        "candidate_count": len(ranked),
        "requested_candidate_count": candidate_count,
        "distractor_type": distractor_type,
        "predicted_hole_id": predicted,
        "ground_truth_hole_id": target,
        "ranked_hole_ids": ranked,
        "top_1_correct": predicted == target,
        "top_3_correct": target in ranked[:3],
        "prediction_status": "uncertain" if predicted is None else "predicted",
        "uncertainty_reason": "score_tie" if predicted is None else None,
        "candidate_scores": fused,
        "scores_by_view": per_view,
        "view_predictions": view_predictions,
        "top_1_top_2_margin": float(margin),
        "confidence_values": confidence_values,
        "confidence_definitions": {
            "margin": "max(0, top1-top2)/(abs(top1)+abs(top2)); Chamfer score gap normalized to [0,1]",
            "multi_view_consistency": "fraction of views voting for the modal candidate",
            "stability": "fraction of deterministic 8% occlusion perturbations preserving the base prediction",
        },
        "ground_truth_yaw_deg": record.ground_truth.get("target_yaw_deg"),
        "predicted_yaw_deg": None,
        "yaw_error_deg": None,
        "yaw_defined": False,
    }


def _ece(rows: Sequence[dict], method: str, bins: int) -> dict:
    if not rows:
        return {"ece": None, "bins": bins, "count": 0}
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_rows = []
    weighted_error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = [
            row
            for row in rows
            if row["confidence_values"][method] >= lower
            and (
                row["confidence_values"][method] <= upper
                if index == bins - 1
                else row["confidence_values"][method] < upper
            )
        ]
        if not selected:
            continue
        confidence = float(np.mean([row["confidence_values"][method] for row in selected]))
        accuracy = float(np.mean([row["top_1_correct"] for row in selected]))
        gap = abs(confidence - accuracy)
        weighted_error += len(selected) / len(rows) * gap
        bin_rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": len(selected),
                "mean_confidence": confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return {"ece": float(weighted_error), "bins": bins, "count": len(rows), "bin_rows": bin_rows}


def _select_threshold(rows: Sequence[dict], method: str, target_coverage: float) -> dict:
    if not rows:
        return {"threshold": 1.0, "coverage": 0.0, "selective_accuracy": None, "abstained": 0}
    candidates = sorted({0.0, 1.0, *[float(row["confidence_values"][method]) for row in rows]})
    eligible = []
    for threshold in candidates:
        retained = [row for row in rows if row["confidence_values"][method] >= threshold]
        coverage = len(retained) / len(rows)
        accuracy = float(np.mean([row["top_1_correct"] for row in retained])) if retained else None
        payload = {
            "threshold": threshold,
            "coverage": coverage,
            "selective_accuracy": accuracy,
            "abstained": len(rows) - len(retained),
        }
        if coverage >= target_coverage:
            eligible.append(payload)
    if not eligible:
        eligible = [{"threshold": 1.0, "coverage": 0.0, "selective_accuracy": None, "abstained": len(rows)}]
    return max(
        eligible,
        key=lambda item: (
            item["selective_accuracy"] if item["selective_accuracy"] is not None else -1.0,
            item["coverage"],
            -item["threshold"],
        ),
    )


def _selective_summary(rows: Sequence[dict], method: str, threshold: float) -> dict:
    retained = [row for row in rows if row["confidence_values"][method] >= threshold]
    return {
        "threshold": float(threshold),
        "num_samples": len(rows),
        "retained": len(retained),
        "abstained": len(rows) - len(retained),
        "coverage": float(len(retained) / len(rows)) if rows else 0.0,
        "selective_accuracy": float(np.mean([row["top_1_correct"] for row in retained])) if retained else None,
        "top_3_accuracy": float(np.mean([row["top_3_correct"] for row in retained])) if retained else None,
    }


def _fixed_coverage_summary(rows: Sequence[dict], method: str, coverage: float) -> dict:
    if not rows:
        return {"target_coverage": coverage, "coverage": 0.0, "retained": 0, "selective_accuracy": None}
    retained_count = max(1, min(len(rows), int(round(coverage * len(rows)))))
    ranked = sorted(enumerate(rows), key=lambda item: (-item[1]["confidence_values"][method], item[0]))
    retained = [row for _, row in ranked[:retained_count]]
    return {
        "target_coverage": coverage,
        "coverage": retained_count / len(rows),
        "retained": retained_count,
        "abstained": len(rows) - retained_count,
        "selective_accuracy": float(np.mean([row["top_1_correct"] for row in retained])),
        "top_3_accuracy": float(np.mean([row["top_3_correct"] for row in retained])),
    }


def _random_fixed_coverage(rows: Sequence[dict], coverage: float, seed: int, repeats: int = 200) -> dict:
    if not rows:
        return {"target_coverage": coverage, "mean_selective_accuracy": None, "std_selective_accuracy": None}
    retained_count = max(1, min(len(rows), int(round(coverage * len(rows)))))
    values = []
    for repeat in range(repeats):
        rng = random.Random(seed + repeat)
        indices = rng.sample(range(len(rows)), retained_count)
        values.append(float(np.mean([rows[index]["top_1_correct"] for index in indices])))
    return {
        "target_coverage": coverage,
        "coverage": retained_count / len(rows),
        "retained": retained_count,
        "abstained": len(rows) - retained_count,
        "repeats": repeats,
        "mean_selective_accuracy": float(np.mean(values)),
        "std_selective_accuracy": float(np.std(values)),
    }


def _risk_coverage(rows: Sequence[dict], method: str) -> List[dict]:
    ranked = sorted(rows, key=lambda row: -row["confidence_values"][method])
    curve = [{"coverage": 0.0, "risk": 0.0, "selective_accuracy": None, "retained": 0}]
    for count in range(1, len(ranked) + 1):
        retained = ranked[:count]
        accuracy = float(np.mean([row["top_1_correct"] for row in retained]))
        curve.append(
            {
                "coverage": count / len(ranked),
                "risk": 1.0 - accuracy,
                "selective_accuracy": accuracy,
                "retained": count,
                "threshold": float(retained[-1]["confidence_values"][method]),
            }
        )
    return curve


def _group_summary(rows: Sequence[dict], method: str, threshold: float) -> List[dict]:
    groups: Dict[Tuple[str, str, str, int, str], List[dict]] = {}
    for row in rows:
        key = (
            row["observation_mode"],
            row["occlusion_level"],
            row["view_set"],
            row["requested_candidate_count"],
            row["distractor_type"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        selective = _selective_summary(group, method, threshold)
        output.append(
            {
                "observation_mode": key[0],
                "occlusion_level": key[1],
                "view_set": key[2],
                "requested_candidate_count": key[3],
                "distractor_type": key[4],
                "num_samples": len(group),
                "top_1_accuracy": float(np.mean([row["top_1_correct"] for row in group])),
                "top_3_accuracy": float(np.mean([row["top_3_correct"] for row in group])),
                "num_uncertain": sum(row.get("prediction_status") == "uncertain" for row in group),
                "uncertainty_rate": float(
                    np.mean([row.get("prediction_status") == "uncertain" for row in group])
                ),
                "episode_top_1_ci95": binary_summary(
                    row["top_1_correct"] for row in group
                )["accuracy_ci95"],
                "episode_top_3_ci95": binary_summary(
                    row["top_3_correct"] for row in group
                )["accuracy_ci95"],
                "yaw_mae_deg": None,
                "yaw_defined_count": 0,
                "confidence_mean": float(np.mean([row["confidence_values"][method] for row in group])),
                "selective": selective,
            }
        )
    return output


def _evaluate_split(
    records: Sequence[EpisodeRecord],
    mode: str,
    view_sets: Sequence[Mapping[str, object]],
    occlusion_levels: Mapping[str, float],
    candidate_counts: Sequence[int],
    pool: Sequence[CandidateAsset],
    seed: int,
    stability_repeats: int,
) -> List[dict]:
    rows = []
    native_views = tuple(
        dict.fromkeys(
            view
            for spec in view_sets
            for view in spec["views"]
            if view != "synthetic_flip"
        )
    )
    for record_index, record in enumerate(records):
        for candidate_count in candidate_counts:
            distractors = ["native"] if candidate_count <= 3 else ["similar", "different"]
            for distractor_type in distractors:
                assets = _candidate_assets(record, mode, native_views, candidate_count, distractor_type, pool)
                if assets is None:
                    continue
                source_peg, source_holes, effective_distractor = assets
                for occlusion_index, (occlusion_name, ratio) in enumerate(occlusion_levels.items()):
                    for view_index, view_spec in enumerate(view_sets):
                        view_ids = tuple(view_spec["views"])
                        row_seed = seed + record_index * 100000 + candidate_count * 1000 + occlusion_index * 100 + view_index
                        rows.append(
                            evaluate_sample(
                                record=record,
                                mode=mode,
                                occlusion_name=occlusion_name,
                                occlusion_ratio=float(ratio),
                                view_set_name=str(view_spec["name"]),
                                view_ids=view_ids,
                                candidate_count=candidate_count,
                                distractor_type=effective_distractor,
                                peg_by_view=source_peg,
                                holes_by_view=source_holes,
                                seed=row_seed,
                                stability_repeats=stability_repeats,
                            )
                        )
    return rows


def evaluate_phase6(
    dataset_root: Path,
    output_root: Path,
    observation_modes: Sequence[str],
    view_sets: Sequence[Mapping[str, object]],
    occlusion_levels: Mapping[str, float],
    candidate_counts: Sequence[int],
    validation_split: str = "val",
    test_split: str = "test",
    confidence_methods: Sequence[str] = ("margin", "multi_view_consistency", "stability"),
    target_coverage: float = 0.8,
    ece_bins: int = 10,
    seed: int = 17,
    stability_repeats: int = 4,
) -> Path:
    all_records = load_episodes(dataset_root, "all")
    validation_records = [record for record in all_records if record.split == validation_split]
    test_records = [record for record in all_records if record.split == test_split]
    if not validation_records or not test_records:
        raise ValueError("Phase 6 requires non-empty validation and test splits")

    all_rows: Dict[str, List[dict]] = {"val": [], "test": []}
    for mode in observation_modes:
        pool = _build_candidate_pool(all_records, mode, ["top", "oblique"])
        all_rows["val"].extend(
            _evaluate_split(validation_records, mode, view_sets, occlusion_levels, candidate_counts, pool, seed, stability_repeats)
        )
        all_rows["test"].extend(
            _evaluate_split(test_records, mode, view_sets, occlusion_levels, candidate_counts, pool, seed + 1000000, stability_repeats)
        )

    thresholds = {method: _select_threshold(all_rows["val"], method, target_coverage) for method in confidence_methods}
    calibration = {
        split: {method: _ece(all_rows[split], method, ece_bins) for method in confidence_methods}
        for split in ("val", "test")
    }
    test_unrejected = {
        "num_samples": len(all_rows["test"]),
        "episode_count": len(test_records),
        "top_1_accuracy": float(np.mean([row["top_1_correct"] for row in all_rows["test"]])) if all_rows["test"] else None,
        "top_3_accuracy": float(np.mean([row["top_3_correct"] for row in all_rows["test"]])) if all_rows["test"] else None,
        "yaw_mae_deg": None,
        "yaw_defined_count": 0,
    }
    selective = {method: _selective_summary(all_rows["test"], method, thresholds[method]["threshold"]) for method in confidence_methods}
    fixed_coverage = {method: _fixed_coverage_summary(all_rows["test"], method, target_coverage) for method in confidence_methods}
    fixed_coverage["random_rejection"] = _random_fixed_coverage(all_rows["test"], target_coverage, seed)
    groups = {method: _group_summary(all_rows["test"], method, thresholds[method]["threshold"]) for method in confidence_methods}
    risk_coverage = {method: _risk_coverage(all_rows["test"], method) for method in confidence_methods}
    aggregate = {
        "schema_version": "phase6.v1",
        "dataset_root": str(dataset_root),
        "observation_modes": list(observation_modes),
        "validation_split": validation_split,
        "test_split": test_split,
        "seed": seed,
        "condition_grid": {
            "occlusion_levels": dict(occlusion_levels),
            "view_sets": list(view_sets),
            "candidate_counts": list(candidate_counts),
            "candidate_note": "counts above 3 use deterministic candidate crops borrowed from other episodes; distractor family is recorded per row",
        },
        "confidence_definitions": {
            "margin": "normalized top-1/top-2 Chamfer score gap",
            "multi_view_consistency": "modal candidate vote fraction across views",
            "stability": "prediction agreement under repeated deterministic 8% occlusion perturbations",
        },
        "confidence_methods": list(confidence_methods),
        "ece": {
            "bin_count": ece_bins,
            "temperature_scaling": "not used; no temperature was fitted",
            "confidence_definition": "method-specific definitions above",
            "by_split": calibration,
        },
        "validation": {
            "num_samples": len(all_rows["val"]),
            "threshold_selection_objective": f"maximize validation selective accuracy subject to coverage >= {target_coverage}",
            "thresholds": thresholds,
        },
        "test": {
            "num_samples": len(all_rows["test"]),
            "unrejected": test_unrejected,
            "selective_by_method": selective,
            "fixed_coverage": fixed_coverage,
            "groups_by_method": groups,
            "risk_coverage": risk_coverage,
        },
        "notes": [
            "Thresholds are selected on validation rows only and applied once to the fixed test rows.",
            "Yaw is not estimated by the Chamfer baseline; yaw MAE is undefined and reported as null.",
            "The current dataset has two native views; synthetic_flip is an augmentation control for the three-view condition.",
            "Phase 6 evaluates the Chamfer baseline because the Phase 5 BLIP score is a forced-label proxy and is not calibrated.",
            "Confidence intervals in condition groups treat each episode as one Bernoulli observation; rows are never split into image crops.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for split in ("val", "test"):
            for row in all_rows[split]:
                handle.write(json.dumps({"split_group": split, **row}, sort_keys=True) + "\n")
    return output_root
