"""Robot control module."""

from __future__ import annotations


class PandaController:
    """Executes the validated pick-and-place action for a selected candidate."""

    def __init__(self, simulation):
        self.simulation = simulation

    def pick_and_place(self, target: str):
        success = self.simulation.pick_and_place(target)
        return {
            "target": target,
            "target_zone": self.simulation.target_zone(target),
            "success": bool(success),
        }
