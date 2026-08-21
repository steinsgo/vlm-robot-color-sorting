"""Unit tests for Phase 9 confidence gating."""

from __future__ import annotations

import unittest

from confmate.phase9 import decide_action


class Phase9DecisionTests(unittest.TestCase):
    def test_tie_abstains_before_robot_motion(self):
        result = decide_action(
            {"hole_001": 0.5, "hole_000": 0.5},
            min_confidence=0.0,
            min_margin=0.0,
        )
        self.assertFalse(result["execute"])
        self.assertEqual(result["prediction_status"], "uncertain")
        self.assertEqual(result["uncertainty_reason"], "score_tie")

    def test_low_margin_abstains(self):
        result = decide_action(
            {"hole_000": 0.51, "hole_001": 0.50},
            min_confidence=0.0,
            min_margin=0.02,
        )
        self.assertFalse(result["execute"])
        self.assertEqual(result["uncertainty_reason"], "margin_below_threshold")

    def test_high_confidence_prediction_executes(self):
        result = decide_action(
            {"hole_000": 0.95, "hole_001": 0.05, "hole_002": 0.02},
            min_confidence=0.55,
            min_margin=0.02,
        )
        self.assertTrue(result["execute"])
        self.assertEqual(result["selected_hole_id"], "hole_000")
        self.assertEqual(result["prediction_status"], "predicted")


if __name__ == "__main__":
    unittest.main()

