# Phase 7 report artifacts

Generated from the fixed Phase 6 evaluation output.

## Reproduce

```powershell
python scripts/evaluate_confidence.py --config configs/phase6.yaml
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --dataset datasets/peg_hole_v2 --phase4-dir evaluations/phase4 --output-dir results
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml --output-dir datasets/peg_hole_v2_phase8
python scripts/evaluate_insertion.py --config configs/phase8.yaml
```

## Artifact status

- `system_architecture.png`: proposed ConfMate pipeline structure.
- `simulation_scene.png`: real Phase 3 PyBullet RGB scene and crops.
- `success_match.png`: real test-split Chamfer matching example.
- `failure_rejection.png`: deterministic light-occlusion failure selected for abstention illustration.
- `accuracy_vs_occlusion.png`: test accuracy curve from Phase 6 rows.
- `risk_coverage.png`: test risk-coverage curves from Phase 6 rows.
- `phase7_demo.gif`: 36.0 seconds at the configured frame rate.
- `summary.csv`: 364 condition and fixed-coverage summary rows.
- `phase8_aggregate.json`: tracked aggregate for the 100-episode collision-proxy insertion validation.

## Scope labels

The matching result is a preliminary Chamfer baseline, not a claim of calibrated BLIP performance. The v2 test split contains 100 independent episodes with randomized target IDs, candidate order, and candidate positions. The analytic fit label is footprint clearance, not insertion dynamics. The three-view condition uses `synthetic_flip`, and 5/8-candidate conditions borrow crops from other episodes. Occlusion is crop-level rectangular corruption rather than a newly rendered occluding object. Yaw is not estimated.

Phase 8 uses a perimeter-wall/floor collision proxy for physics-assisted fit validation; it is not a full insertion controller.
