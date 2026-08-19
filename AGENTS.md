# Repository working agreement

## Scope and claims

- Keep the ConfMate peg-hole task explicitly separate from the original color-sorting demo.
- Treat peg-hole results as a simulation prototype. The analytic fit label is not a claim of contact-rich insertion dynamics.
- Report `oracle_crop`, `render_mask_assisted`, and `rgb_only` separately. `rgb_only` is skipped until a detector/cropper exists.
- BLIP VQA output is a forced-label proxy unless token-level probabilities and calibration are implemented.

## Reproducible workflow

1. Record the current environment with `python scripts/env_report.py`.
2. Generate `datasets/peg_hole_v2` with `python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml`.
3. Run Phase 4, Phase 5 mock/BLIP, Phase 6, Phase 7, then Phase 8 insertion validation in that order.
4. Run `python -m unittest discover -s tests` and `python scripts/smoke_test.py` before committing.

## Repository hygiene

- Do not commit generated `datasets/`, `evaluations/`, model caches, or runtime `runs/`.
- Keep `results/` limited to report-ready, provenance-bearing artifacts.
- Update `upstream.lock` only when intentionally rebasing the ConfMate work on a new upstream commit.
- Treat Phase 8's perimeter-wall collision proxy as physics-assisted fit validation, not as a complete insertion controller.
