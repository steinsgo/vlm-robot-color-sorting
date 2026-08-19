#!/usr/bin/env python3
"""Run Phase 6 confidence, abstention, and partial-observation evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.phase6 import OCCLUSION_LEVELS, evaluate_phase6  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 confidence and abstention.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase6.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-coverage", type=float, default=None)
    parser.add_argument("--stability-repeats", type=int, default=None)
    return parser


def main(argv=None) -> int:
    import yaml

    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dataset_value = args.dataset or Path(config.get("dataset", "datasets/peg_hole_v1"))
    output_value = args.output_dir or Path(config.get("output_dir", "evaluations/phase6"))
    dataset_root = PROJECT_ROOT / dataset_value if not dataset_value.is_absolute() else dataset_value
    output_root = PROJECT_ROOT / output_value if not output_value.is_absolute() else output_value
    view_sets = config.get(
        "view_sets",
        [
            {"name": "single_view", "views": ["top"]},
            {"name": "two_view", "views": ["top", "oblique"]},
            {"name": "three_view_augmented", "views": ["top", "oblique", "synthetic_flip"]},
        ],
    )
    result = evaluate_phase6(
        dataset_root=dataset_root,
        output_root=output_root,
        observation_modes=config.get("observation_modes", ["oracle_crop", "render_mask_assisted"]),
        view_sets=view_sets,
        occlusion_levels=config.get("occlusion_levels", OCCLUSION_LEVELS),
        candidate_counts=config.get("candidate_counts", [3, 5, 8]),
        validation_split=config.get("validation_split", "val"),
        test_split=config.get("test_split", "test"),
        confidence_methods=config.get(
            "confidence_methods", ["margin", "multi_view_consistency", "stability"]
        ),
        target_coverage=(
            args.target_coverage
            if args.target_coverage is not None
            else float(config.get("target_coverage", 0.8))
        ),
        ece_bins=int(config.get("ece_bins", 10)),
        seed=int(config.get("seed", 17)),
        stability_repeats=(
            args.stability_repeats
            if args.stability_repeats is not None
            else int(config.get("stability_repeats", 4))
        ),
    )
    print(f"EVALUATION_DIR={result}")
    print("STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

