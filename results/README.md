# Phase 7 report artifacts

Generated from the fixed Phase 6 evaluation output.

## Reproduce

```powershell
python scripts/evaluate_confidence.py --config configs/phase6.yaml
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --dataset datasets/peg_hole_v1 --phase4-dir evaluations/phase4 --output-dir results
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

## Scope labels

The matching result is a preliminary Chamfer baseline, not a claim of calibrated BLIP performance. The three-view condition uses `synthetic_flip`, and 5/8-candidate conditions borrow crops from other episodes. Occlusion is crop-level rectangular corruption rather than a newly rendered occluding object. Yaw is not estimated.
