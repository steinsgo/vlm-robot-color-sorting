"""Regression tests for the Phase 8 collision-proxy insertion experiment."""

from __future__ import annotations

import unittest

from confmate.phase8 import InsertionConfig, _footprint_bounds, run_insertion_trial
from confmate.peg_hole import shape_parts


class InsertionValidationTests(unittest.TestCase):
    def test_footprint_bounds_are_positive(self):
        bounds = _footprint_bounds(shape_parts("rectangle", "hole", "target"))
        self.assertLess(bounds[0], bounds[1])
        self.assertLess(bounds[2], bounds[3])

    def test_target_rectangle_can_insert_into_collision_proxy(self):
        import pybullet as p

        connection = p.connect(p.DIRECT)
        try:
            result = run_insertion_trial(
                peg_family="rectangle",
                hole_family="rectangle",
                hole_variant="target",
                peg_yaw_deg=0.0,
                hole_yaw_deg=0.0,
                position=[0.0, 0.0],
                analytic_fit=True,
                config=InsertionConfig(steps=360),
            )
        finally:
            p.disconnect(connection)
        self.assertTrue(result["inserted"], result)
        self.assertFalse(result["wall_contact_seen"], result)

    def test_small_distractor_rejects_oversized_peg(self):
        import pybullet as p

        connection = p.connect(p.DIRECT)
        try:
            result = run_insertion_trial(
                peg_family="rectangle",
                hole_family="rectangle",
                hole_variant="distractor",
                peg_yaw_deg=0.0,
                hole_yaw_deg=0.0,
                position=[0.0, 0.0],
                analytic_fit=False,
                config=InsertionConfig(steps=360),
            )
        finally:
            p.disconnect(connection)
        self.assertFalse(result["inserted"], result)
        self.assertTrue(result["wall_contact_seen"], result)


if __name__ == "__main__":
    unittest.main()
