"""Simulation boundary for the modular Phase 2 pipeline."""

from __future__ import annotations

from typing import Any, Tuple

import pybullet as p

from panda_vision_simulation import VisionLanguagePandaSimulation


class SimulationBackend:
    """Owns PyBullet lifecycle and exposes scene/control primitives."""

    def __init__(self, config):
        self.config = config
        self.sim = VisionLanguagePandaSimulation(
            gui_mode=not config.headless,
            model_name=config.model_name,
            num_objects=config.num_objects,
            seed=config.seed,
        )

    @property
    def objects(self):
        return self.sim.objects

    def setup(self) -> None:
        self.sim.setup_simulation()
        for _ in range(self.config.settle_steps):
            p.stepSimulation()

    def capture_and_process_scene(self) -> Tuple[Any, dict, dict]:
        return self.sim.capture_and_process_scene()

    def capture_frame(self):
        return self.sim.capture_camera_image()

    def pick_and_place(self, object_name: str) -> bool:
        if not self.sim.pick_object(object_name):
            return False
        self.sim.place_object_in_sorting_zone(object_name)
        return True

    def target_zone(self, object_name: str) -> str:
        return self.sim._get_target_zone_for_object(object_name)

    def close(self) -> None:
        self.sim.cleanup()
