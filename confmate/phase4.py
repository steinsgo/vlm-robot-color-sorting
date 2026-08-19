"""Phase 4 peg-hole image-image matching baselines and evaluation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


REQUIRED_METHODS = ("random", "chamfer", "clip_single", "clip_multi")


@dataclass
class EpisodeRecord:
    episode_id: str
    split: str
    seed: int
    episode_dir: Path
    ground_truth: dict


@dataclass
class ObservationInputs:
    observation_mode: str
    view_ids: Tuple[str, ...]
    peg_by_view: Dict[str, Image.Image]
    holes_by_view: Dict[str, Dict[str, Image.Image]]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episodes(dataset_root: Path, split: str = "test") -> List[EpisodeRecord]:
    manifest = _read_json(dataset_root / "manifest.json")
    records = []
    for entry in manifest["episodes"]:
        if split != "all" and entry["split"] != split:
            continue
        relative = str(entry["ground_truth"]).replace("\\", "/")
        ground_truth_path = dataset_root / relative
        ground_truth = _read_json(ground_truth_path)
        records.append(
            EpisodeRecord(
                episode_id=entry["episode_id"],
                split=entry["split"],
                seed=int(ground_truth["seed"]),
                episode_dir=ground_truth_path.parent,
                ground_truth=ground_truth,
            )
        )
    if not records:
        raise ValueError(f"No episodes found for split={split!r} in {dataset_root}")
    return records


def load_observation(record: EpisodeRecord, mode: str, views: Sequence[str]) -> Optional[ObservationInputs]:
    peg_by_view: Dict[str, Image.Image] = {}
    holes_by_view: Dict[str, Dict[str, Image.Image]] = {}
    candidate_ids = list(record.ground_truth["candidate_hole_ids"])

    for view in views:
        mode_dir = record.episode_dir / "observations" / mode / view
        if mode == "rgb_only":
            # The Phase 3 rgb_only contract intentionally contains only rgb.png.
            # A detector/cropper is required before this mode can be evaluated.
            if not (mode_dir / "rgb.png").exists() or list(mode_dir.glob("candidate_*.png")) == []:
                return None
        peg_path = mode_dir / "peg.png"
        if not peg_path.exists():
            return None
        peg_by_view[view] = Image.open(peg_path).convert("RGB")
        candidates: Dict[str, Image.Image] = {}
        for index, candidate_id in enumerate(candidate_ids):
            candidate_path = mode_dir / f"candidate_{index:03d}.png"
            if not candidate_path.exists():
                return None
            candidates[candidate_id] = Image.open(candidate_path).convert("RGB")
        holes_by_view[view] = candidates
    return ObservationInputs(mode, tuple(views), peg_by_view, holes_by_view)


def _rank(scores: Dict[str, float]) -> Tuple[str, List[str], float, float]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    names = [name for name, _ in ranked]
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else top1
    return names[0], names, float(top1), float(top1 - top2)


def _softmax_confidence(scores: Dict[str, float], temperature: float = 0.1) -> float:
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if values.size == 0:
        return 0.0
    shifted = (values - values.max()) / max(temperature, 1e-8)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return float(probabilities.max())


def random_scores(candidate_ids: Sequence[str], seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    return {candidate_id: rng.random() for candidate_id in candidate_ids}


def _binary_shape(image: Image.Image, size: int = 64) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.NEAREST))
    # Crops have a white mask background. Keep dark and saturated object pixels.
    return gray < 245


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32)
    inner = np.zeros_like(mask)
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    return np.argwhere(mask & ~inner).astype(np.float32)


def chamfer_distance(first: Image.Image, second: Image.Image) -> float:
    first_points = _boundary(_binary_shape(first))
    second_points = _boundary(_binary_shape(second))
    if len(first_points) == 0 or len(second_points) == 0:
        return 1.0
    # The 64x64 boundary is small enough for an explicit pairwise distance.
    distances = np.sqrt(((first_points[:, None, :] - second_points[None, :, :]) ** 2).sum(axis=2))
    return float(0.5 * distances.min(axis=1).mean() + 0.5 * distances.min(axis=0).mean())


def chamfer_scores(peg: Image.Image, holes: Dict[str, Image.Image]) -> Dict[str, float]:
    # Higher scores are better for all methods, hence the negative distance.
    return {candidate_id: -chamfer_distance(peg, hole) for candidate_id, hole in holes.items()}


class CLIPImageImageMatcher:
    """Image-only CLIP encoder with explicit L2 normalization."""

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None, device: Optional[str] = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name, **kwargs)
        self.model = CLIPModel.from_pretrained(model_name, **kwargs).to(self.device)
        self.model.eval()

    def _embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().cpu().numpy()

    def score_view(self, peg: Image.Image, holes: Dict[str, Image.Image]) -> Dict[str, float]:
        candidate_ids = list(holes)
        embeddings = self._embed([peg] + [holes[candidate_id] for candidate_id in candidate_ids])
        peg_embedding = embeddings[0]
        hole_embeddings = embeddings[1:]
        scores = hole_embeddings @ peg_embedding
        return {candidate_id: float(score) for candidate_id, score in zip(candidate_ids, scores)}


def multi_view_clip_scores(
    matcher: CLIPImageImageMatcher,
    inputs: ObservationInputs,
) -> Dict[str, float]:
    per_view = [matcher.score_view(inputs.peg_by_view[view], inputs.holes_by_view[view]) for view in inputs.view_ids]
    candidate_ids = list(per_view[0])
    return {
        candidate_id: float(np.mean([scores[candidate_id] for scores in per_view]))
        for candidate_id in candidate_ids
    }


def yaw_error_deg(predicted_yaw: Optional[float], ground_truth: Optional[float], symmetry_order: Optional[int]) -> Optional[float]:
    if predicted_yaw is None or ground_truth is None:
        return None
    if symmetry_order is None:
        return None
    candidates = [ground_truth + 360.0 * index / symmetry_order for index in range(symmetry_order)]
    return float(min(abs(((predicted_yaw - candidate + 180.0) % 360.0) - 180.0) for candidate in candidates))


def make_result(
    record: EpisodeRecord,
    mode: str,
    method: str,
    scores: Dict[str, float],
    model_name: str,
    view_ids: Sequence[str],
    detail: Optional[dict] = None,
) -> dict:
    predicted, ranked, top_score, margin = _rank(scores)
    target = record.ground_truth["target_hole_id"]
    top3 = ranked[:3]
    return {
        "episode_id": record.episode_id,
        "split": record.split,
        "seed": record.seed,
        "observation_mode": mode,
        "method": method,
        "model": model_name,
        "views": list(view_ids),
        "score_definition": "L2-normalized image-image CLIP dot product" if method.startswith("clip") else method,
        "candidate_scores": {name: float(score) for name, score in scores.items()},
        "ranked_hole_ids": ranked,
        "predicted_hole_id": predicted,
        "ground_truth_hole_id": target,
        "matching_correct": predicted == target,
        "top_1_correct": predicted == target,
        "top_3_correct": target in top3,
        "top_1_top_2_margin": margin,
        "confidence": _softmax_confidence(scores),
        "confidence_definition": "max softmax over candidate scores, temperature=0.1; not calibrated",
        "predicted_yaw_deg": None,
        "ground_truth_yaw_deg": record.ground_truth.get("target_yaw_deg"),
        "yaw_error_deg": yaw_error_deg(
            None,
            record.ground_truth.get("target_yaw_deg"),
            record.ground_truth.get("yaw_symmetry_order"),
        ),
        "yaw_defined": False,
        "detail": detail or {},
    }


def make_skipped_result(record: EpisodeRecord, mode: str, method: str, reason: str, model_name: str) -> dict:
    return {
        "episode_id": record.episode_id,
        "split": record.split,
        "seed": record.seed,
        "observation_mode": mode,
        "method": method,
        "model": model_name,
        "status": "skipped",
        "skip_reason": reason,
        "candidate_scores": {},
        "predicted_hole_id": None,
        "ground_truth_hole_id": record.ground_truth["target_hole_id"],
        "matching_correct": None,
        "top_1_correct": None,
        "top_3_correct": None,
        "top_1_top_2_margin": None,
        "confidence": None,
        "predicted_yaw_deg": None,
        "ground_truth_yaw_deg": record.ground_truth.get("target_yaw_deg"),
        "yaw_error_deg": None,
        "yaw_defined": False,
    }


def aggregate_results(results: Sequence[dict]) -> dict:
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for result in results:
        if result.get("status", "ok") == "skipped":
            continue
        key = (result["split"], result["observation_mode"], result["method"])
        groups.setdefault(key, []).append(result)

    aggregate = []
    for (split, mode, method), group in sorted(groups.items()):
        yaw_values = [item["yaw_error_deg"] for item in group if item["yaw_error_deg"] is not None]
        aggregate.append(
            {
                "split": split,
                "observation_mode": mode,
                "method": method,
                "num_episodes": len(group),
                "matching_accuracy": float(np.mean([item["matching_correct"] for item in group])),
                "top_1_accuracy": float(np.mean([item["top_1_correct"] for item in group])),
                "top_3_accuracy": float(np.mean([item["top_3_correct"] for item in group])),
                "mean_top_1_top_2_margin": float(np.mean([item["top_1_top_2_margin"] for item in group])),
                "mean_confidence": float(np.mean([item["confidence"] for item in group])),
                "yaw_mae_deg": float(np.mean(yaw_values)) if yaw_values else None,
                "yaw_defined_count": len(yaw_values),
                "yaw_note": "Phase 4 matching baselines do not estimate yaw; yaw error is undefined.",
            }
        )
    return {"groups": aggregate, "num_scored": len([r for r in results if r.get("status", "ok") != "skipped"]), "num_skipped": len([r for r in results if r.get("status") == "skipped"])}


def evaluate_dataset(
    dataset_root: Path,
    output_root: Path,
    split: str,
    observation_modes: Sequence[str],
    methods: Sequence[str],
    model_name: str,
    cache_dir: Optional[Path],
    views: Sequence[str],
    seed: int,
) -> Path:
    records = load_episodes(dataset_root, split)
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    clip_matcher = None

    for mode in observation_modes:
        for record in records:
            inputs = load_observation(record, mode, views)
            for method in methods:
                if inputs is None:
                    results.append(make_skipped_result(record, mode, method, "rgb_only_has_no_candidate_crops", model_name))
                    continue
                if method == "random":
                    scores = random_scores(record.ground_truth["candidate_hole_ids"], seed + record.seed)
                    result = make_result(record, mode, method, scores, "random", views)
                elif method == "chamfer":
                    view = views[0]
                    scores = chamfer_scores(inputs.peg_by_view[view], inputs.holes_by_view[view])
                    result = make_result(record, mode, method, scores, "geometric_chamfer", [view])
                elif method == "clip_single":
                    if clip_matcher is None:
                        clip_matcher = CLIPImageImageMatcher(model_name, cache_dir=cache_dir)
                    view = views[0]
                    scores = clip_matcher.score_view(inputs.peg_by_view[view], inputs.holes_by_view[view])
                    result = make_result(record, mode, method, scores, model_name, [view])
                elif method == "clip_multi":
                    if clip_matcher is None:
                        clip_matcher = CLIPImageImageMatcher(model_name, cache_dir=cache_dir)
                    scores = multi_view_clip_scores(clip_matcher, inputs)
                    result = make_result(record, mode, method, scores, model_name, views)
                else:
                    raise ValueError(f"Unknown Phase 4 method: {method}")
                results.append(result)

    results_dir = output_root / "episodes"
    results_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        result_dir = results_dir / result["method"] / result["observation_mode"] / result["split"]
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / f"{result['episode_id']}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")

    aggregate = aggregate_results(results)
    aggregate.update(
        {
            "dataset_root": str(dataset_root),
            "split_requested": split,
            "observation_modes": list(observation_modes),
            "methods": list(methods),
            "model": model_name,
            "views": list(views),
            "seed": seed,
            "matching_definition": "image-image only for CLIP methods; image-text scores are not used",
        }
    )
    with (output_root / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    return output_root
