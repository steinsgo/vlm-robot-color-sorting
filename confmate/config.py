"""Configuration loading for reproducible baseline runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class BaselineConfig:
    """Runtime configuration shared by the Phase 2 modules."""

    model_name: str = "openai/clip-vit-base-patch32"
    seed: int = 17
    num_objects: int = 5
    headless: bool = False
    prompt: str = "a red object"
    output_dir: str = "runs"
    save_video: bool = False
    video_fps: int = 4
    settle_steps: int = 240

    @classmethod
    def from_yaml(cls, path: Path) -> "BaselineConfig":
        """Load a flat YAML config, accepting an optional ``baseline`` section."""
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if "baseline" in payload:
            payload = payload["baseline"] or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")

        valid_fields = set(cls.__dataclass_fields__)
        values = {key: value for key, value in payload.items() if key in valid_fields}
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_objects < 1 or self.num_objects > 5:
            raise ValueError("num_objects must be between 1 and 5")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")

    def update_from_cli(
        self,
        *,
        headless: Optional[bool] = None,
        seed: Optional[int] = None,
        num_objects: Optional[int] = None,
        save_video: Optional[bool] = None,
        output_dir: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> "BaselineConfig":
        """Apply only explicitly supplied CLI overrides."""
        updates: Dict[str, Any] = {
            "headless": headless,
            "seed": seed,
            "num_objects": num_objects,
            "save_video": save_video,
            "output_dir": output_dir,
            "prompt": prompt,
        }
        for key, value in updates.items():
            if value is not None:
                setattr(self, key, value)
        self.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
