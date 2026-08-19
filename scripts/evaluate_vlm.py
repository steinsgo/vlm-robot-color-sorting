#!/usr/bin/env python3
"""Run Phase 5 mock or generative VLM matching on Phase 3 crops."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache-temp"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache-temp"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.phase4 import load_episodes, load_observation  # noqa: E402
from confmate.vlm import BlipVQAMatcher, MockVLMMatcher, VLMMatcher  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _make_skipped(record, mode: str, adapter: str, reason: str, model: str) -> dict:
    return {
        "episode_id": record.episode_id,
        "split": record.split,
        "seed": record.seed,
        "observation_mode": mode,
        "adapter": adapter,
        "model": model,
        "status": "skipped",
        "skip_reason": reason,
        "candidate_results": {},
        "predicted_hole_id": None,
        "ground_truth_hole_id": record.ground_truth["target_hole_id"],
        "matching_correct": None,
    }


def _evaluate_record(record, mode: str, matcher: VLMMatcher, prompt: str, views: List[str]) -> dict:
    inputs = load_observation(record, mode, views)
    if inputs is None:
        return _make_skipped(record, mode, matcher.model_name, "rgb_only_has_no_candidate_crops", matcher.model_name)

    candidate_results: Dict[str, dict] = {}
    for candidate_id in record.ground_truth["candidate_hole_ids"]:
        holes = {view: inputs.holes_by_view[view][candidate_id] for view in views}
        candidate_results[candidate_id] = matcher.match(inputs.peg_by_view, holes, prompt)

    ranked = sorted(
        candidate_results,
        key=lambda candidate_id: (-candidate_results[candidate_id]["score"], candidate_id),
    )
    predicted = ranked[0]
    predicted_result = candidate_results[predicted]
    return {
        "episode_id": record.episode_id,
        "split": record.split,
        "seed": record.seed,
        "observation_mode": mode,
        "adapter": predicted_result.get("adapter", matcher.model_name),
        "model": matcher.model_name,
        "prompt": prompt,
        "views": views,
        "candidate_results": candidate_results,
        "ranked_hole_ids": ranked,
        "predicted_hole_id": predicted,
        "ground_truth_hole_id": record.ground_truth["target_hole_id"],
        "matching_correct": predicted == record.ground_truth["target_hole_id"],
        "confidence": predicted_result["score"],
        "confidence_definition": "highest adapter YES/NO score; not calibrated",
    }


def _aggregate(results: List[dict]) -> dict:
    groups = {}
    for result in results:
        if result.get("status") == "skipped":
            continue
        key = (result["split"], result["observation_mode"], result["adapter"])
        groups.setdefault(key, []).append(result)
    rows = []
    for (split, mode, adapter), group in sorted(groups.items()):
        rows.append(
            {
                "split": split,
                "observation_mode": mode,
                "adapter": adapter,
                "num_episodes": len(group),
                "matching_accuracy": sum(item["matching_correct"] for item in group) / len(group),
                "mean_confidence": sum(item["confidence"] for item in group) / len(group),
            }
        )
    return {
        "groups": rows,
        "num_scored": sum(result.get("status") != "skipped" for result in results),
        "num_skipped": sum(result.get("status") == "skipped" for result in results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 VLM matching adapters.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter", choices=["mock", "blip"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    parser.add_argument("--observation-mode", action="append", dest="observation_modes", default=None)
    parser.add_argument("--prompt", default="Does the left peg fit the right hole? Answer only yes or no.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--view", action="append", dest="views", default=None, choices=["top", "oblique"])
    return parser


def main(argv=None) -> int:
    import yaml

    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    adapter = args.adapter or config.get("adapter", "mock")
    model_name = args.model_name or config.get("model_name", "Salesforce/blip-vqa-base")
    split = args.split or config.get("split", "test")
    dataset_value = args.dataset or Path(config.get("dataset", "datasets/peg_hole_v1"))
    configured_output = config.get("output_dir", f"evaluations/phase5/{adapter}")
    if args.output_dir is None and config.get("adapter", adapter) != adapter:
        configured_output = f"evaluations/phase5/{adapter}"
    output_value = args.output_dir or Path(configured_output)
    modes = args.observation_modes or config.get("observation_modes", ["oracle_crop", "render_mask_assisted", "rgb_only"])
    prompt = args.prompt if args.prompt != build_parser().get_default("prompt") else config.get("prompt", args.prompt)
    views = args.views or config.get("views", ["top", "oblique"])
    dataset_root = PROJECT_ROOT / dataset_value if not dataset_value.is_absolute() else dataset_value
    output_root = PROJECT_ROOT / output_value if not output_value.is_absolute() else output_value
    records = load_episodes(dataset_root, split)
    if args.max_episodes is not None:
        records = records[: args.max_episodes]

    if adapter == "mock":
        matcher: VLMMatcher = MockVLMMatcher()
    else:
        matcher = BlipVQAMatcher(model_name, cache_dir=PROJECT_ROOT / ".hf-cache-temp")

    results = []
    for mode in modes:
        for record in records:
            results.append(_evaluate_record(record, mode, matcher, prompt, views))

    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_root / "episodes"
    for result in results:
        result_dir = episodes_dir / result["adapter"] / result["observation_mode"] / result["split"]
        result_dir.mkdir(parents=True, exist_ok=True)
        _write_json(result_dir / f"{result['episode_id']}.json", result)
    _write_json(
        output_root / "aggregate.json",
        {
            **_aggregate(results),
            "adapter": adapter,
            "model": matcher.model_name,
            "prompt": prompt,
            "split": split,
            "observation_modes": modes,
            "views": views,
        },
    )
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"EVALUATION_DIR={output_root}")
    print("STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
