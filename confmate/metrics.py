"""Small, dependency-free metrics helpers used by evaluation reports."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a Wilson score interval for a Bernoulli proportion."""
    if trials <= 0:
        return (None, None)
    proportion = successes / float(trials)
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def binary_summary(values: Iterable[bool]) -> dict:
    """Summarize binary episode outcomes with a 95% Wilson interval."""
    normalized = [bool(value) for value in values]
    trials = len(normalized)
    successes = sum(normalized)
    interval = wilson_interval(successes, trials)
    return {
        "episode_count": trials,
        "accuracy": successes / float(trials) if trials else None,
        "accuracy_ci95": list(interval) if trials else [None, None],
    }
