"""Command-line entrypoint shared by Phase 2 scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BaselineConfig
from .runner import run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ConfMate Phase 2 baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-objects", type=int, default=None)
    parser.add_argument("--save-video", action="store_true", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = BaselineConfig.from_yaml(args.config)
    config.update_from_cli(
        headless=args.headless,
        seed=args.seed,
        num_objects=args.num_objects,
        save_video=args.save_video,
        output_dir=args.output_dir,
        prompt=args.prompt,
    )
    run_dir = run_baseline(config, project_root=Path.cwd())
    print(f"RUN_DIR={run_dir}")
    print("STATUS=ok")
    return 0
