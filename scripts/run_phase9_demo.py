#!/usr/bin/env python3
"""Run the Phase 9 four-object confidence-aware Panda insertion mission."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache-temp"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache-temp"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.phase9 import Phase9Config, run_phase9  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 9 four-object Panda peg-hole mission.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase9.yaml"))
    parser.add_argument("--headless", action="store_true", help="Use PyBullet DIRECT mode.")
    parser.add_argument("--keep-open", action="store_true", help="Keep the GUI open after the episode.")
    parser.add_argument("--no-video", action="store_true", help="Do not save a GIF.")
    parser.add_argument("--no-sleep", action="store_true", help="Run without real-time motion delays.")
    parser.add_argument("--matcher", choices=["chamfer", "clip", "oracle"], default=None)
    parser.add_argument(
        "--shapes",
        default=None,
        help="Comma-separated four shapes (cylinder,sphere,square,cuboid).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    import yaml

    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = Phase9Config(
        **{
            key: value
            for key, value in payload.items()
            if key in Phase9Config.__dataclass_fields__
        }
    )
    if args.headless:
        config.gui = False
    if args.keep_open:
        config.keep_open = True
    if args.no_video:
        config.save_video = False
    if args.no_sleep:
        config.sleep = False
    if args.matcher is not None:
        config.matcher = args.matcher
    if args.shapes is not None:
        config.shape_families = tuple(item.strip() for item in args.shapes.split(",") if item.strip())
    if args.seed is not None:
        config.seed = args.seed
    if args.min_confidence is not None:
        config.min_confidence = args.min_confidence
    if args.min_margin is not None:
        config.min_margin = args.min_margin
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    result = run_phase9(config, project_root=PROJECT_ROOT)
    summary_path = result / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    print(f"RUN_DIR={result}")
    print("STATUS=ok")
    print(f"MISSION_PASS={str(bool(summary.get('mission_pass', False))).lower()}")
    mission = summary.get("mission", {})
    print(
        f"COMPLETED={mission.get('completed_count', 0)}/"
        f"{mission.get('object_count', config.num_objects)}"
    )
    print(
        "STEPS="
        + ",".join(
            f"{step.get('peg_id')}:{step.get('status')}"
            for step in summary.get("steps", [])
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
