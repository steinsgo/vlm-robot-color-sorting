"""Phase 2 baseline orchestration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .artifacts import create_run_dir, save_video, save_visualization, write_json
from .config import BaselineConfig
from .control import PandaController
from .matching import CLIPMatcher
from .perception import ScenePerception
from .simulation import SimulationBackend


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_baseline(config: BaselineConfig, project_root: Optional[Path] = None) -> Path:
    """Run one reproducible perception-matching-control episode."""
    config.validate()
    project_root = project_root or Path.cwd()
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_dir = create_run_dir(output_root, config.seed)
    write_json(run_dir / "run_config.json", config.to_dict())

    seed_everything(config.seed)
    simulation = None
    frames = []
    summary = {
        "status": "error",
        "seed": config.seed,
        "headless": config.headless,
        "num_objects": config.num_objects,
        "model": config.model_name,
        "prompt": config.prompt,
        "candidate_scores": {},
        "candidate_count": 0,
        "selected_target": None,
        "confidence": 0.0,
        "action": None,
        "artifacts": {"run_config": "run_config.json"},
    }

    try:
        simulation = SimulationBackend(config)
        simulation.setup()
        perception = ScenePerception(simulation)
        matcher = CLIPMatcher(simulation)
        controller = PandaController(simulation)

        observation = perception.observe()
        frames.append(observation.image.copy())
        match = matcher.match(observation.crops, config.prompt)
        summary.update(
            {
                "candidate_scores": match.scores,
                "candidate_count": len(match.scores),
                "selected_target": match.selected_target,
                "confidence": match.confidence,
            }
        )

        if match.selected_target is not None:
            summary["action"] = controller.pick_and_place(match.selected_target)

        final_image = simulation.capture_frame()
        frames.append(final_image.copy())
        save_visualization(
            observation.image,
            observation.bounding_boxes,
            match.scores,
            match.selected_target,
            match.confidence,
            run_dir / "visualization.png",
        )
        summary["artifacts"]["visualization"] = "visualization.png"

        if config.save_video:
            save_video(frames, run_dir / "episode.gif", config.video_fps)
            summary["artifacts"]["video"] = "episode.gif"

        summary["status"] = "ok"
        write_json(run_dir / "summary.json", summary)
        return run_dir
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(run_dir / "summary.json", summary)
        raise
    finally:
        if simulation is not None:
            simulation.close()
