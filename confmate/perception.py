"""Camera and candidate extraction module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import Image


@dataclass
class SceneObservation:
    image: Image.Image
    bounding_boxes: Dict[str, Tuple[int, int, int, int]]
    crops: Dict[str, Image.Image]

    @property
    def candidate_names(self):
        return list(self.crops.keys())


class ScenePerception:
    """Converts a simulator camera frame into candidate crops."""

    def __init__(self, simulation):
        self.simulation = simulation

    def observe(self) -> SceneObservation:
        image, boxes, crops = self.simulation.capture_and_process_scene()
        return SceneObservation(image=image, bounding_boxes=boxes, crops=crops)
