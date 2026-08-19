#!/usr/bin/env python3
"""Generate the Phase 7 report figures, summary CSV, and demo GIF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.phase4 import load_episodes, load_observation  # noqa: E402
from confmate.phase6 import _make_view_set  # noqa: E402


OCCLUSION_ORDER = ["none", "light", "moderate", "heavy"]
SERIES_COLORS = {"margin": "#2563eb", "multi_view_consistency": "#d97706", "stability": "#059669"}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _find_record(records, episode_id: str):
    for record in records:
        if record.episode_id == episode_id:
            return record
    raise ValueError(f"Episode not found: {episode_id}")


def _show_image(ax, image: Image.Image, title: str) -> None:
    ax.imshow(image.convert("RGB"))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _plot_system_architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 3.7, 2.2, 1.25, "PyBullet\npeg-hole scenes", "#dbeafe"),
        (3.0, 3.7, 2.2, 1.25, "Multi-view RGB\ncrops/masks", "#dcfce7"),
        (5.7, 3.7, 2.2, 1.25, "Matching\nChamfer / CLIP / VLM", "#fef3c7"),
        (8.4, 3.7, 2.2, 1.25, "Confidence\nmargin / consistency / stability", "#fce7f3"),
        (4.2, 1.0, 3.7, 1.25, "Phase 6–7 evaluation\nval threshold → test report", "#ede9fe"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=color, edgecolor="#334155", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [((2.5, 4.32), (3.0, 4.32)), ((5.2, 4.32), (5.7, 4.32)), ((7.9, 4.32), (8.4, 4.32)), ((9.5, 3.7), (7.1, 2.25)), ((6.0, 3.7), (6.0, 2.25))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#475569"))
    ax.text(6.0, 5.55, "ConfMate confidence-aware VLM matching pipeline", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(6.0, 0.45, "Phase 7 artifacts: figures, risk-coverage, CSV summary, and reproducible demo", ha="center", va="center", fontsize=9, color="#475569")
    _save_figure(fig, output)


def _plot_scene(record, dataset_root: Path, output: Path) -> None:
    scene = Image.open(record.episode_dir / "views" / "top" / "rgb.png").convert("RGB")
    inputs = load_observation(record, "oracle_crop", ["top"])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    _show_image(axes[0], scene, f"Scene: {record.episode_id}\ntop RGB")
    _show_image(axes[1], inputs.peg_by_view["top"], "peg crop")
    for index, candidate_id in enumerate(record.ground_truth["candidate_hole_ids"][:2], start=2):
        _show_image(axes[index], inputs.holes_by_view["top"][candidate_id], f"{candidate_id} crop")
    fig.suptitle("Simulation scene and oracle crops (real Phase 3 data)", fontsize=13, weight="bold")
    fig.tight_layout()
    _save_figure(fig, output)


def _phase4_result(phase4_root: Path, method: str, mode: str, split: str, episode_id: str) -> dict:
    return _read_json(phase4_root / "episodes" / method / mode / split / f"{episode_id}.json")


def _plot_match_example(record, phase4_root: Path, output: Path) -> None:
    inputs = load_observation(record, "oracle_crop", ["top"])
    result = _phase4_result(phase4_root, "chamfer", "oracle_crop", "test", record.episode_id)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    _show_image(axes[0], inputs.peg_by_view["top"], "peg")
    ranked = result["ranked_hole_ids"]
    for index, candidate_id in enumerate(record.ground_truth["candidate_hole_ids"], start=1):
        score = result["candidate_scores"][candidate_id]
        marker = "TARGET / PREDICTED" if candidate_id == result["predicted_hole_id"] else "distractor"
        _show_image(axes[index], inputs.holes_by_view["top"][candidate_id], f"{candidate_id}\nscore={score:.3f} · {marker}")
    fig.suptitle("Successful matching example (Chamfer baseline)", fontsize=13, weight="bold")
    fig.text(0.5, 0.01, f"rank: {' > '.join(ranked)} · target={record.ground_truth['target_hole_id']}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_figure(fig, output)


def _plot_failure_rejection(record, output: Path, threshold: float) -> None:
    source = load_observation(record, "oracle_crop", ["top", "oblique"])
    seed = 1000017 + 100000 + 3000 + 100 + 1
    peg, holes = _make_view_set(
        source.peg_by_view,
        source.holes_by_view,
        ["top", "oblique"],
        0.15,
        seed,
    )
    score_rows = []
    for candidate_id, image in holes["top"].items():
        score_rows.append((candidate_id, -float(np.mean([0.0]))))
    # Use the Phase 6 JSON row for exact prediction/confidence; image display uses
    # the same deterministic occlusion parameters above.
    phase6_rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / "evaluations" / "phase6" / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    row = next(
        item
        for item in phase6_rows
        if item["split_group"] == "test"
        and item["episode_id"] == record.episode_id
        and item["observation_mode"] == "oracle_crop"
        and item["occlusion_level"] == "light"
        and item["view_set"] == "two_view"
        and item["requested_candidate_count"] == 3
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    _show_image(axes[0], peg["top"], "peg · light occlusion")
    for index, candidate_id in enumerate(record.ground_truth["candidate_hole_ids"], start=1):
        marker = "PREDICTED" if candidate_id == row["predicted_hole_id"] else "TARGET" if candidate_id == row["ground_truth_hole_id"] else "distractor"
        _show_image(axes[index], holes["top"][candidate_id], f"{candidate_id}\n{marker}")
    confidence = row["confidence_values"]["margin"]
    status = "ABSTAIN / uncertain" if confidence < threshold else "wrong but retained"
    fig.suptitle("Failure and abstention example", fontsize=13, weight="bold")
    fig.text(0.5, 0.01, f"predicted={row['predicted_hole_id']} · target={row['ground_truth_hole_id']} · margin confidence={confidence:.3f} < threshold={threshold:.3f} · {status}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save_figure(fig, output)


def _plot_occlusion_curve(aggregate: dict, output: Path) -> None:
    rows = aggregate["test"]["groups_by_method"]["margin"]
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["occlusion_level"]].append(row)
    top1 = [float(np.mean([item["top_1_accuracy"] for item in grouped[level]])) for level in OCCLUSION_ORDER]
    top3 = [float(np.mean([item["top_3_accuracy"] for item in grouped[level]])) for level in OCCLUSION_ORDER]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(OCCLUSION_ORDER))
    ax.plot(x, top1, marker="o", linewidth=2, color="#2563eb", label="Top-1")
    ax.plot(x, top3, marker="s", linewidth=2, color="#059669", label="Top-3")
    ax.set_xticks(x, [name.title() for name in OCCLUSION_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Occlusion condition")
    ax.set_ylabel("Accuracy")
    ax.set_title("Matching accuracy under increasing occlusion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output)


def _plot_risk_coverage(aggregate: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for method in ("margin", "multi_view_consistency", "stability"):
        curve = aggregate["test"]["risk_coverage"][method]
        ax.plot(
            [item["coverage"] for item in curve],
            [item["risk"] for item in curve],
            linewidth=2,
            color=SERIES_COLORS[method],
            label=method.replace("_", " "),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk = 1 - selective accuracy")
    ax.set_title("Risk-coverage curves on fixed test split")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output)


def _write_summary_csv(aggregate: dict, output: Path) -> int:
    rows = []
    for method, group_rows in aggregate["test"]["groups_by_method"].items():
        for row in group_rows:
            rows.append(
                {
                    "split": "test",
                    "row_type": "condition",
                    "confidence_method": method,
                    "observation_mode": row["observation_mode"],
                    "occlusion_level": row["occlusion_level"],
                    "view_set": row["view_set"],
                    "candidate_count": row["requested_candidate_count"],
                    "distractor_type": row["distractor_type"],
                    "num_samples": row["num_samples"],
                    "top_1_accuracy": row["top_1_accuracy"],
                    "top_3_accuracy": row["top_3_accuracy"],
                    "confidence_mean": row["confidence_mean"],
                    "coverage": row["selective"]["coverage"],
                    "selective_accuracy": row["selective"]["selective_accuracy"],
                    "abstained": row["selective"]["abstained"],
                }
            )
    for policy, row in aggregate["test"]["fixed_coverage"].items():
        if policy == "random_rejection":
            accuracy = row["mean_selective_accuracy"]
        else:
            accuracy = row["selective_accuracy"]
        rows.append(
            {
                "split": "test",
                "row_type": "fixed_coverage",
                "confidence_method": policy,
                "observation_mode": "all",
                "occlusion_level": "all",
                "view_set": "all",
                "candidate_count": "all",
                "distractor_type": "all",
                "num_samples": aggregate["test"]["num_samples"],
                "top_1_accuracy": "",
                "top_3_accuracy": "",
                "confidence_mean": "",
                "coverage": row["coverage"],
                "selective_accuracy": accuracy,
                "abstained": row["abstained"],
            }
        )
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _fit_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2))
    return canvas


def _demo_frame(stage: str, images: Sequence[Image.Image], labels: Sequence[str], progress: float, index: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", (720, 410), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((24, 18), "ConfMate Phase 7 demo", fill="#0f172a", font=font)
    draw.text((24, 42), stage, fill="#334155", font=font)
    panel_size = (210, 290)
    gap = 15
    for item_index, (image, label) in enumerate(zip(images, labels)):
        x = 24 + item_index * (panel_size[0] + gap)
        y = 76
        draw.rectangle((x, y, x + panel_size[0], y + panel_size[1]), outline="#cbd5e1", width=1)
        canvas.paste(_fit_image(image, (panel_size[0] - 10, panel_size[1] - 35)), (x + 5, y + 5))
        draw.text((x + 8, y + panel_size[1] - 25), label, fill="#0f172a", font=font)
    draw.rectangle((24, 382, 696, 390), fill="#e2e8f0")
    draw.rectangle((24, 382, 24 + int(672 * progress), 390), fill="#2563eb")
    draw.text((610, 18), f"{index + 1}/{total}", fill="#64748b", font=font)
    return canvas


def _create_demo_gif(records, output: Path, duration_seconds: float, fps: int) -> float:
    success_record = _find_record(records, "test_asymmetric_000")
    failure_record = _find_record(records, "test_asymmetric_001")
    success_inputs = load_observation(success_record, "oracle_crop", ["top"])
    failure_inputs = load_observation(failure_record, "oracle_crop", ["top"])
    scene = Image.open(success_record.episode_dir / "views" / "top" / "rgb.png").convert("RGB")
    success_images = [success_inputs.peg_by_view["top"]] + [success_inputs.holes_by_view["top"][cid] for cid in success_record.ground_truth["candidate_hole_ids"]]
    failure_images = [failure_inputs.peg_by_view["top"]] + [failure_inputs.holes_by_view["top"][cid] for cid in failure_record.ground_truth["candidate_hole_ids"]]
    total = max(1, int(round(duration_seconds * fps)))
    frames = []
    for index in range(total):
        phase = index / total
        if phase < 1 / 3:
            images, labels, stage = [scene], ["real PyBullet top RGB"], "1 · observe simulation scene"
        elif phase < 2 / 3:
            images, labels, stage = success_images, ["peg", "target hole", "distractor", "distractor"], "2 · successful matching"
        else:
            images, labels, stage = failure_images, ["occluded peg", "target", "predicted wrong", "distractor"], "3 · failure → confidence-aware abstention"
        frames.append(_demo_frame(stage, images, labels, (index + 1) / total, index, total))
    duration_ms = max(20, int(round(1000 / fps)))
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
    return total * duration_ms / 1000.0


def _write_results_readme(output: Path, csv_rows: int, gif_seconds: float) -> None:
    text = f"""# Phase 7 report artifacts

Generated from the fixed Phase 6 evaluation output.

## Reproduce

```powershell
python scripts/evaluate_confidence.py --config configs/phase6.yaml
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --dataset datasets/peg_hole_v1 --phase4-dir evaluations/phase4 --output-dir results
```

## Artifact status

- `system_architecture.png`: proposed ConfMate pipeline structure.
- `simulation_scene.png`: real Phase 3 PyBullet RGB scene and crops.
- `success_match.png`: real test-split Chamfer matching example.
- `failure_rejection.png`: deterministic light-occlusion failure selected for abstention illustration.
- `accuracy_vs_occlusion.png`: test accuracy curve from Phase 6 rows.
- `risk_coverage.png`: test risk-coverage curves from Phase 6 rows.
- `phase7_demo.gif`: {gif_seconds:.1f} seconds at the configured frame rate.
- `summary.csv`: {csv_rows} condition and fixed-coverage summary rows.

## Scope labels

The matching result is a preliminary Chamfer baseline, not a claim of calibrated BLIP performance. The three-view condition uses `synthetic_flip`, and 5/8-candidate conditions borrow crops from other episodes. Occlusion is crop-level rectangular corruption rather than a newly rendered occluding object. Yaw is not estimated.
"""
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase 7 report artifacts.")
    parser.add_argument("--phase6-json", type=Path, default=Path("evaluations/phase6/aggregate.json"))
    parser.add_argument("--phase6-results", type=Path, default=Path("evaluations/phase6/results.jsonl"))
    parser.add_argument("--phase4-dir", type=Path, default=Path("evaluations/phase4"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/peg_hole_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--duration-seconds", type=float, default=36.0)
    parser.add_argument("--fps", type=int, default=5)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = PROJECT_ROOT / args.dataset if not args.dataset.is_absolute() else args.dataset
    phase6_json = PROJECT_ROOT / args.phase6_json if not args.phase6_json.is_absolute() else args.phase6_json
    phase4_dir = PROJECT_ROOT / args.phase4_dir if not args.phase4_dir.is_absolute() else args.phase4_dir
    output = PROJECT_ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    aggregate = _read_json(phase6_json)
    records = load_episodes(dataset_root, "all")
    test_records = [record for record in records if record.split == "test"]
    threshold = aggregate["validation"]["thresholds"]["margin"]["threshold"]
    _plot_system_architecture(output / "system_architecture.png")
    _plot_scene(_find_record(test_records, "test_asymmetric_000"), dataset_root, output / "simulation_scene.png")
    _plot_match_example(_find_record(test_records, "test_asymmetric_000"), phase4_dir, output / "success_match.png")
    _plot_failure_rejection(_find_record(test_records, "test_asymmetric_001"), output / "failure_rejection.png", threshold)
    _plot_occlusion_curve(aggregate, output / "accuracy_vs_occlusion.png")
    _plot_risk_coverage(aggregate, output / "risk_coverage.png")
    csv_rows = _write_summary_csv(aggregate, output / "summary.csv")
    gif_seconds = _create_demo_gif(test_records, output / "phase7_demo.gif", args.duration_seconds, args.fps)
    _write_results_readme(output / "README.md", csv_rows, gif_seconds)
    try:
        source_phase6 = str(phase6_json.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        source_phase6 = str(phase6_json)
    manifest = {
        "phase": 7,
        "source_phase6": source_phase6,
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
        "gif_seconds": gif_seconds,
        "summary_csv_rows": csv_rows,
        "scope": {
            "baseline": "Chamfer matching on Phase 3 crops",
            "extension": "confidence-aware rejection and partial-observation evaluation",
            "preliminary": True,
            "known_limitations": ["synthetic_flip third view", "crop-level occlusion", "yaw undefined", "rgb_only not evaluated"],
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"REPORT_DIR={output}")
    print(f"GIF_SECONDS={gif_seconds:.1f}")
    print(f"SUMMARY_ROWS={csv_rows}")
    print("STATUS=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
