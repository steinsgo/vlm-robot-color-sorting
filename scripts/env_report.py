#!/usr/bin/env python3
"""Write a non-secret runtime and dependency snapshot for reproducibility."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECT_DEPENDENCIES = (
    "pybullet",
    "numpy",
    "torch",
    "transformers",
    "pillow",
    "matplotlib",
    "requests",
    "accelerate",
    "pyyaml",
    "tokenizers",
)


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def _package_versions() -> dict:
    versions = {}
    for package in DIRECT_DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _torch_runtime() -> dict:
    try:
        import torch

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as error:  # Environment reporting must not fail without Torch.
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def build_report() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "torch_runtime": _torch_runtime(),
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "remotes": _git_value("remote", "-v"),
            "working_tree_dirty": bool(_git_value("status", "--porcelain")),
        },
        "network": {
            "http_proxy_configured": bool(os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")),
            "https_proxy_configured": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")),
        },
        "security_note": "Proxy URLs, credentials, tokens, and cache contents are intentionally omitted.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write a reproducibility environment report.")
    parser.add_argument("--output", type=Path, default=Path("reports/env_report.json"))
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build_report(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"ENV_REPORT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
