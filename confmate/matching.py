"""Candidate matching module backed by the existing CLIP implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MatchResult:
    scores: Dict[str, float]
    selected_target: Optional[str]
    confidence: float


class CLIPMatcher:
    """Keep matching independent from camera and robot control code."""

    def __init__(self, simulation):
        self.simulation = simulation

    def match(self, crops, prompt: str) -> MatchResult:
        scores = self.simulation.sim.compute_object_similarity(crops, prompt)
        selected, confidence = self.simulation.sim.select_best_object(scores)
        return MatchResult(
            scores={name: float(score) for name, score in scores.items()},
            selected_target=selected,
            confidence=float(confidence),
        )
