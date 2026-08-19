"""Run artifacts: JSON summaries, visual diagnostics, and optional video."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image


def create_run_dir(output_root: Path, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"baseline_{timestamp}_seed{seed}"
    suffix = 1
    candidate = run_dir
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{run_dir.name}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def save_visualization(
    image: Image.Image,
    bounding_boxes: Dict[str, tuple],
    scores: Dict[str, float],
    selected_target: Optional[str],
    confidence: float,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    fig, (image_axis, score_axis) = plt.subplots(1, 2, figsize=(16, 7))
    image_axis.imshow(image)
    image_axis.set_title("Camera frame: candidates and selected target")
    image_axis.axis("off")

    colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(bounding_boxes))))
    for index, (name, box) in enumerate(bounding_boxes.items()):
        x1, y1, x2, y2 = box
        selected = name == selected_target
        rectangle = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=4 if selected else 2,
            edgecolor="red" if selected else colors[index],
            facecolor="none",
        )
        image_axis.add_patch(rectangle)
        score = scores.get(name, 0.0)
        label = f"{'SELECTED ' if selected else ''}{name}\nconfidence={score:.4f}"
        image_axis.text(
            x1, max(0, y1 - 5), label, fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )

    names = list(scores)
    values = [scores[name] for name in names]
    bars = score_axis.bar(names, values, color=colors[:len(names)]) if names else []
    if selected_target in names:
        bars[names.index(selected_target)].set_color("red")
    score_axis.set_title(f"CLIP scores; selected confidence={confidence:.4f}")
    score_axis.set_ylabel("Normalized similarity")
    score_axis.set_ylim(0, max(1.0, max(values, default=0.0) * 1.2))
    score_axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_video(frames: Iterable[Image.Image], path: Path, fps: int) -> None:
    prepared = [frame.convert("RGB") for frame in frames]
    if not prepared:
        return
    first, *rest = prepared
    first.save(
        path,
        save_all=True,
        append_images=rest,
        duration=max(1, int(1000 / fps)),
        loop=0,
    )
