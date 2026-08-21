"""Regression tests for the Phase 9 four-object mission."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pybullet as p

from confmate.phase9 import MISSION_SHAPES, Phase9Config, decide_action, run_phase9
from confmate.phase9_mission import (
    HOLE_SLOTS,
    MultiObjectPegHoleScene,
    PandaInsertionController,
)


class Phase9MissionTests(unittest.TestCase):
    def test_default_mission_has_four_distinct_shapes(self):
        config = Phase9Config()
        self.assertEqual(config.num_objects, 4)
        self.assertEqual(config.shape_families, MISSION_SHAPES)
        self.assertEqual(len(set(config.shape_families)), 4)

    def test_single_remaining_candidate_is_not_rejected_by_margin_gate(self):
        result = decide_action(
            {"hole_003": 0.20},
            min_confidence=0.9,
            min_margin=0.5,
        )
        self.assertTrue(result["execute"])
        self.assertEqual(result["selected_hole_id"], "hole_003")
        self.assertEqual(result["margin"], 1.0)

    def test_pegs_have_one_dynamic_visual_body_each(self):
        connection = p.connect(p.DIRECT)
        try:
            scene = MultiObjectPegHoleScene(
                Phase9Config(gui=False, matcher="oracle", save_video=False, sleep=False)
            )
            scene.build()
            for peg in scene.pegs.values():
                visual_shapes = p.getVisualShapeData(peg["body_id"])
                self.assertTrue(visual_shapes, peg)
        finally:
            p.disconnect(connection)

    def test_circular_holes_are_low_rings_with_collision_proxies(self):
        connection = p.connect(p.DIRECT)
        try:
            scene = MultiObjectPegHoleScene(
                Phase9Config(gui=False, matcher="oracle", save_video=False, sleep=False)
            )
            scene.build()
            self.assertEqual(
                {tuple(hole["position"]) for hole in scene.holes.values()},
                {tuple(slot) for slot in HOLE_SLOTS},
            )
            for hole in scene.holes.values():
                self.assertTrue(hole["wall_ids"])
                self.assertIsInstance(hole["floor_id"], int)
                part = hole["parts"][0]
                if hole["shape_family"] in {"cylinder", "sphere"}:
                    self.assertEqual(part["kind"], "ring")
                    self.assertLess(part["height"], 0.04)
                    self.assertGreater(part["inner_radius"], 0.0)
                    self.assertGreater(part["outer_radius"], part["inner_radius"])
                else:
                    self.assertEqual(part["kind"], "box")
        finally:
            p.disconnect(connection)

    def test_release_geometry_rejects_a_peg_outside_the_opening(self):
        connection = p.connect(p.DIRECT)
        try:
            scene = MultiObjectPegHoleScene(
                Phase9Config(gui=False, matcher="oracle", save_video=False, sleep=False)
            )
            scene.build()
            peg_id = "peg_000"
            hole = scene.holes[scene.pegs[peg_id]["target_hole_id"]]
            controller = PandaInsertionController(scene, peg_id)
            orientation = p.getQuaternionFromEuler(
                [0.0, 0.0, 0.0174532925199433 * hole["yaw_deg"]]
            )
            self.assertTrue(
                controller._inside_hole_opening(
                    hole,
                    [hole["position"][0], hole["position"][1], hole["floor_top_z"]],
                    orientation,
                )
            )
            self.assertFalse(
                controller._inside_hole_opening(
                    hole,
                    [hole["position"][0] + 0.02, hole["position"][1], hole["floor_top_z"]],
                    orientation,
                )
            )
        finally:
            p.disconnect(connection)

    def test_oracle_mission_completes_all_four_objects(self):
        with tempfile.TemporaryDirectory(prefix="confmate_phase9_") as temp_dir:
            config = Phase9Config(
                matcher="oracle",
                gui=False,
                sleep=False,
                save_video=False,
                output_dir=temp_dir,
            )
            run_dir = run_phase9(config, project_root=Path.cwd())
            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        self.assertTrue(summary["mission_pass"], summary)
        self.assertEqual(summary["mission"]["completed_count"], 4)
        self.assertEqual(len(summary["steps"]), 4)
        self.assertTrue(all(step["status"] == "success" for step in summary["steps"]))
        self.assertTrue(
            all(step["action"]["wall_contact_count"] == 0 for step in summary["steps"])
        )
        self.assertTrue(
            all(
                set(step["action"]["grasp_finger_link_indices"]) == {9, 10}
                for step in summary["steps"]
            )
        )
        self.assertTrue(all(step["action"]["teleport_used"] is False for step in summary["steps"]))
        self.assertTrue(all(step["action"]["physics_release_used"] for step in summary["steps"]))
        self.assertTrue(all(step["action"]["inside_hole_opening"] for step in summary["steps"]))
        self.assertTrue(all(step["action"]["floor_supported"] for step in summary["steps"]))
        self.assertTrue(all(step["action"]["stable"] for step in summary["steps"]))
        self.assertTrue(any(step["action"]["air_rotation_steps"] > 1 for step in summary["steps"]))
        for step in summary["steps"]:
            action = step["action"]
            validation = action["grasp_validation"]
            criteria = validation["criteria"]
            self.assertTrue(action["grasped"])
            self.assertTrue(action["grasp_constraint_used"])
            self.assertGreater(action["grasp_lift"], 0.035)
            self.assertTrue(validation["valid"])
            self.assertTrue(criteria["both_finger_contacts"])
            self.assertTrue(criteria["side_contact_on_both_fingers"])
            self.assertTrue(criteria["opposing_contact_normals"])
            self.assertTrue(criteria["peg_between_fingers"])
            self.assertTrue(criteria["not_fully_closed"])
            self.assertGreaterEqual(validation["finger_gap"], 0.012)
            self.assertLessEqual(validation["finger_gap"], 0.080)
            self.assertLessEqual(validation["peg_centre_offset"], 0.012)
            hold_positions = action["grasp_hold_joint_positions"]
            transport_positions = action["transport_finger_joint_positions"]
            self.assertEqual(set(hold_positions), {"9", "10"})
            self.assertEqual(set(transport_positions), {"9", "10"})
            for link_index in ("9", "10"):
                self.assertAlmostEqual(
                    hold_positions[link_index],
                    transport_positions[link_index],
                    delta=0.005,
                )
            self.assertFalse(
                all(
                    transport_positions[link_index] <= 0.0045
                    for link_index in ("9", "10")
                )
            )


if __name__ == "__main__":
    unittest.main()
