#!/usr/bin/env python3
"""Run Phase 8 PyBullet collision-proxy insertion validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.phase8 import InsertionConfig, evaluate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate collision-proxy peg-hole insertion.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase8.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    return parser


def main(argv=None) -> int:
    import yaml

    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dataset_value = args.dataset or Path(config.get("dataset", "datasets/peg_hole_v2"))
    output_value = args.output_dir or Path(config.get("output_dir", "evaluations/phase8"))
    split = args.split or config.get("split", "test")
    dataset_root = PROJECT_ROOT / dataset_value if not dataset_value.is_absolute() else dataset_value
    output_root = PROJECT_ROOT / output_value if not output_value.is_absolute() else output_value
    insertion_config = InsertionConfig(
        steps=args.steps if args.steps is not None else int(config.get("steps", 720)),
        time_step=float(config.get("time_step", 1.0 / 240.0)),
        start_height=float(config.get("start_height", 0.35)),
        floor_z=float(config.get("floor_z", 0.025)),
        floor_thickness=float(config.get("floor_thickness", 0.025)),
        wall_height=float(config.get("wall_height", 0.20)),
        wall_thickness=float(config.get("wall_thickness", 0.012)),
        position_tolerance=float(config.get("position_tolerance", 0.025)),
        height_tolerance=float(config.get("height_tolerance", 0.02)),
        velocity_tolerance=float(config.get("velocity_tolerance", 0.08)),
    )
    result = evaluate_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        split=split,
        config=insertion_config,
        max_episodes=(
            args.max_episodes
            if args.max_episodes is not None
            else config.get("max_episodes")
        ),
    )
    print(f"EVALUATION_DIR={result}")
    print("STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
