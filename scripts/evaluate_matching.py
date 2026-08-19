#!/usr/bin/env python3
"""Run Phase 4 peg-hole matching baselines."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache-temp"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache-temp"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.phase4 import REQUIRED_METHODS, evaluate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 peg-hole matching baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase4.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    parser.add_argument("--observation-mode", action="append", dest="observation_modes", default=None)
    parser.add_argument("--method", action="append", dest="methods", default=None, choices=REQUIRED_METHODS)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--view", action="append", dest="views", default=None, choices=["top", "oblique"])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import yaml

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dataset_value = args.dataset or Path(config.get("dataset", "datasets/peg_hole_v1"))
    output_value = args.output_dir or Path(config.get("output_dir", "evaluations/phase4"))
    split = args.split or config.get("split", "test")
    modes = args.observation_modes or config.get("observation_modes", ["oracle_crop", "render_mask_assisted", "rgb_only"])
    methods = args.methods or config.get("methods", list(REQUIRED_METHODS))
    model_name = args.model_name or config.get("model_name", "openai/clip-vit-base-patch32")
    seed = args.seed if args.seed is not None else config.get("seed", 17)
    views = args.views or config.get("views", ["top", "oblique"])
    cache_dir = PROJECT_ROOT / ".hf-cache-temp"
    output_dir = evaluate_dataset(
        dataset_root=(PROJECT_ROOT / dataset_value if not dataset_value.is_absolute() else dataset_value),
        output_root=(PROJECT_ROOT / output_value if not output_value.is_absolute() else output_value),
        split=split,
        observation_modes=modes,
        methods=methods,
        model_name=model_name,
        cache_dir=cache_dir,
        views=views,
        seed=seed,
    )
    print(f"EVALUATION_DIR={output_dir}")
    print("STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
