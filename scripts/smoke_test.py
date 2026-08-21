#!/usr/bin/env python3
"""Fast import/config/artifact smoke test for the Phase 2/3 architecture."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache-temp"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf-cache-temp"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.artifacts import save_visualization, write_json  # noqa: E402
from confmate.config import BaselineConfig  # noqa: E402
from confmate.control import PandaController  # noqa: E402
from confmate.matching import CLIPMatcher  # noqa: E402
from confmate.peg_hole import OBSERVATION_MODE_SPECS, SHAPE_FAMILIES, PegHoleConfig, shape_parts  # noqa: E402
from confmate.phase4 import REQUIRED_METHODS, chamfer_distance, random_scores  # noqa: E402
from confmate.phase6 import OCCLUSION_LEVELS  # noqa: E402
from confmate.phase9 import Phase9Config, decide_action  # noqa: E402
from confmate.vlm import MockVLMMatcher, VLMMatcher  # noqa: E402
from confmate.perception import SceneObservation, ScenePerception  # noqa: E402
from confmate.simulation import SimulationBackend  # noqa: E402


def main() -> int:
    config = BaselineConfig.from_yaml(PROJECT_ROOT / "configs" / "baseline.yaml")
    assert config.num_objects == 5
    assert config.seed == 17
    assert all((SceneObservation, ScenePerception, CLIPMatcher, PandaController, SimulationBackend))
    peg_hole_config = PegHoleConfig.from_yaml(PROJECT_ROOT / "configs" / "peg_hole.yaml")
    assert set(peg_hole_config.observation_modes) == set(OBSERVATION_MODE_SPECS)
    for family in SHAPE_FAMILIES:
        assert shape_parts(family, "peg")
        assert shape_parts(family, "hole", variant="distractor")
    assert set(REQUIRED_METHODS) == {"random", "chamfer", "clip_single", "clip_multi"}
    assert OCCLUSION_LEVELS == {"none": 0.0, "light": 0.15, "moderate": 0.35, "heavy": 0.55}
    assert Phase9Config().family == "rectangle"
    assert Phase9Config().matcher == "clip"
    assert decide_action({"hole_000": 0.9, "hole_001": 0.1}, 0.5, 0.01)["execute"]
    assert list(random_scores(["a", "b"], seed=17)) == ["a", "b"]
    assert chamfer_distance(Image.new("RGB", (16, 16), "black"), Image.new("RGB", (16, 16), "black")) == 0.0
    mock_result = MockVLMMatcher().match(
        {"top": Image.new("RGB", (16, 16), "black")},
        {"top": Image.new("RGB", (16, 16), "black")},
        "Answer yes or no.",
    )
    assert isinstance(MockVLMMatcher(), VLMMatcher)
    assert mock_result["label"] in {"YES", "NO"}
    assert set(("label", "score", "raw_text", "model", "view_scores")) <= set(mock_result)

    with tempfile.TemporaryDirectory(prefix="confmate_smoke_") as temp_dir:
        output = Path(temp_dir)
        write_json(output / "run_config.json", config.to_dict())
        save_visualization(
            Image.new("RGB", (120, 80), "white"),
            {"candidate": (10, 10, 50, 50)},
            {"candidate": 0.75},
            "candidate",
            0.75,
            output / "visualization.png",
        )
        assert (output / "run_config.json").exists()
        assert (output / "visualization.png").exists()

    print("SMOKE_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
