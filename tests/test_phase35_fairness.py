"""Regression tests for the Phase 3.5 anti-leakage safeguards."""

from __future__ import annotations

import unittest

from confmate.peg_hole import PegHoleConfig, geometric_fit_check
from confmate.phase4 import _rank as phase4_rank
from confmate.phase6 import _rank as phase6_rank


class FairnessRegressionTests(unittest.TestCase):
    def test_equal_scores_abstain_in_phase4(self):
        prediction, ranked, _, margin = phase4_rank({"hole_001": 0.5, "hole_000": 0.5})
        self.assertIsNone(prediction)
        self.assertEqual(ranked, ["hole_000", "hole_001"])
        self.assertEqual(margin, 0.0)

    def test_equal_scores_abstain_in_phase6(self):
        prediction, _, _, margin = phase6_rank({"hole_001": 1.0, "hole_000": 1.0})
        self.assertIsNone(prediction)
        self.assertEqual(margin, 0.0)

    def test_analytic_fit_distinguishes_target_and_distractors(self):
        target = geometric_fit_check("asymmetric", "asymmetric", "target", 30.0, 30.0)
        clearance_distractor = geometric_fit_check("asymmetric", "asymmetric", "distractor", 30.0, 30.0)
        other_family = geometric_fit_check("asymmetric", "rectangle", "distractor", 30.0, 30.0)
        self.assertTrue(target["compatible"])
        self.assertGreater(target["fit_margin"], 0.0)
        self.assertFalse(clearance_distractor["compatible"])
        self.assertFalse(other_family["compatible"])

    def test_test_split_supports_at_least_100_independent_episodes(self):
        config = PegHoleConfig(episodes_per_split={"test": 100})
        config.validate()
        self.assertEqual(config.episodes_per_split["test"], 100)


if __name__ == "__main__":
    unittest.main()
