# Phase 9 Summary — Confidence-Gated Panda Peg–Hole Demo

Phase 9 connects the ConfMate peg–hole matching task to the Panda robot logic from the original PyBullet color-sorting demo.

## What was added

- `confmate/phase9.py`: GUI scene, two-view observation capture, Chamfer/CLIP/oracle scoring, tie-aware confidence gate, Panda IK, gripper control, grasp stabilization, insertion validation, JSON and GIF artifacts.
- `scripts/run_phase9_demo.py`: command-line GUI/headless entrypoint.
- `configs/phase9.yaml`: reproducible default configuration.
- `tests/test_phase9_demo.py`: confidence gate regression tests.

The original color-sorting demo remains unchanged. Phase 9 creates a separate table scene with a blue peg, candidate hole proxies, the Panda arm, score labels, and an explicit action gate. If the gate abstains, the robot does not move.

## Run the GUI

With the project environment active and the Hugging Face model available locally:

```bash
python scripts/run_phase9_demo.py --matcher clip --keep-open
```

For the deterministic controller-only check:

```bash
python scripts/run_phase9_demo.py --matcher oracle --keep-open
```

The oracle mode uses the analytic fit label and is a debug control, not a VLM result. `--matcher chamfer` runs the image-shape baseline.

## Validation on 2026-08-21

Using seed 17, `rectangle` peg, two rendered views, and the local `openai/clip-vit-base-patch32` model:

| Field | Result |
| --- | --- |
| PyBullet connection | `GUI` |
| matcher | CLIP image-image baseline |
| selected candidate | `hole_001` |
| target candidate | `hole_001` |
| confidence | `0.3831` |
| top-1/top-2 margin | `0.1059` |
| action gate | accepted at demo threshold `0.35` |
| grasped | `true` |
| inserted | `true` |
| perimeter-wall contacts | `0` |

The same action path also passed in `DIRECT` mode. The oracle control run produced `grasped=true`, `inserted=true`, and zero wall contacts.

## Scope and limitations

- The confidence threshold is a demo/action threshold, not calibrated token probability or a statistically validated policy.
- The hole is a collision-enabled perimeter-wall/floor proxy; this is not a full contact-rich insertion controller or a hardware claim.
- CLIP/Chamfer can abstain or select an incompatible candidate. The physical outcome is recorded rather than hidden.
- This single GUI episode is an integration demonstration, not a replacement for the Phase 4–7 multi-episode benchmark.
