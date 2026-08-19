#!/usr/bin/env python3
"""Generate the Phase 3 peg-hole scenes and observation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.peg_hole import PegHoleConfig, generate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the ConfMate Phase 3 peg-hole dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/peg_hole.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes-per-family", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = PegHoleConfig.from_yaml(args.config)
    config.update_from_cli(
        seed=args.seed,
        episodes_per_family=args.episodes_per_family,
        output_dir=args.output_dir,
    )
    output_dir = generate_dataset(config, project_root=Path.cwd())
    print(f"DATASET_DIR={output_dir}")
    print(f"EPISODES={len(list(output_dir.glob('*/*/ground_truth.json')))}")
    print("STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
