# PyBullet Robotic Arm Simulation with Vision-Language Models

[![ConfMate CI](https://github.com/steinsgo/vlm-robot-color-sorting/actions/workflows/ci.yml/badge.svg?branch=confmate-baseline)](https://github.com/steinsgo/vlm-robot-color-sorting/actions/workflows/ci.yml)

A Python robotics repository containing the original PyBullet color-sorting demo and a reproducible ConfMate peg–hole matching prototype. The ConfMate results use simulation-assisted crops and synthetic corruption controls, and are reported as preliminary.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8edd7f96-cdcd-4495-b8e8-9ef2164e3d8a" alt="vlm (online-video-cutter" width="1000"/>
</p>

## Overview

The repository contains:

- **Physics-based robotic arm simulation** using PyBullet
- **Vision-guided object selection** using CLIP
- **Multi-view peg–hole candidate matching** with Chamfer, CLIP, mock VLM, and BLIP adapter baselines
- **Confidence-aware abstention** and partial-observation evaluation

The current implementation covers Phase 1–9, including a four-object confidence-gated Panda peg–hole mission. Phase 9 is an interactive simulation prototype; its default CLIP mode is explicitly separated from the deterministic oracle control mode and the CLIP/Chamfer benchmark results. For implementation status, environment notes, reproducibility commands, and known limitations, see [PHASE123_SUMMARY.md](PHASE123_SUMMARY.md), [PHASE9_SUMMARY.md](PHASE9_SUMMARY.md), and [history.md](history.md).

The robot captures images from its camera, analyzes them using CLIP, selects objects based on text descriptions, and performs pick-and-place operations, including color sorting and interactive placement.


## Implementation Videos
1) VLM interaction:


https://github.com/user-attachments/assets/17893bf4-5b80-47f8-9a8a-8a106cedf039



2) Automatic Sorting:


https://github.com/user-attachments/assets/93f907f5-6c96-4eb3-babd-b46292f15680


## Project Structure

```
├── panda_vision_simulation.py             # Vision-guided robot simulation Class
├── color_sorting_vlm.py                   # Color sorting and interactive demo with VLM
├── simple_pick_place_demo.py              # Simple pick and place demo
├── confmate/                              # Phase 2 simulation/perception/matching/control modules
├── configs/baseline.yaml                  # Reproducible baseline configuration
├── configs/peg_hole.yaml                  # Phase 3 scene generation configuration
├── configs/phase4.yaml                    # Phase 4 matching evaluation configuration
├── configs/phase5.yaml                    # Phase 5 VLM adapter configuration
├── configs/phase6.yaml                    # Phase 6 confidence evaluation configuration
├── configs/phase8.yaml                    # Phase 8 collision-proxy insertion configuration
├── configs/phase9.yaml                    # Phase 9 four-object mission configuration
├── scripts/generate_peg_hole_dataset.py   # Phase 3 dataset generator
├── scripts/evaluate_matching.py           # Phase 4 matching evaluator
├── scripts/evaluate_vlm.py                 # Phase 5 VLM evaluator
├── scripts/evaluate_confidence.py          # Phase 6 confidence/risk evaluator
├── scripts/evaluate_insertion.py           # Phase 8 physics-assisted fit evaluator
├── scripts/run_phase9_demo.py              # Phase 9 four-object Panda insertion mission
├── scripts/generate_report.py               # Phase 7 figures/CSV/GIF generator
├── results/                                # Phase 7 report artifacts
├── tests/                                  # Phase 3.5 fairness regression tests
├── reports/env_report.json                 # Reproducibility environment snapshot
├── PHASE9_SUMMARY.md                        # Phase 9 mission results and limitations
├── AGENTS.md                               # Scope and repository workflow
├── upstream.lock                            # Upstream/fork provenance lock
├── scripts/run_baseline.py                # Phase 2 CLI entrypoint
├── scripts/smoke_test.py                  # Fast architecture smoke test
├── requirements.txt                       # Python dependencies
└── README.md                              # This documentation
```

## Features

### Vision-Guided Robot Simulation
- **Camera image capture** from PyBullet simulation
- **Object detection** using 3D-to-2D projection and segmentation masks
- **Vision-language matching** with **CLIP** for object selection
- **Text-prompted pick and place** ("pick up the red cube")
- **Multiple objects** (red cube, red sphere, blue cube, green sphere, yellow cylinder)
- **Color sorting zones** with physical borders to prevent objects from falling
- **Interactive zone selection and throw action** for object placement

## Requirements

### Basic Requirements (for robot simulation)
- Python 3.9+
- PyBullet
- NumPy

### Full Requirements (for vision integration)
- All basic requirements plus:
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- Matplotlib
- Requests

## Installation
1. Clone or download this project:
   ```bash
   git clone https://github.com/steinsgo/vlm-robot-color-sorting.git
   ```

2. (Recommended) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Color Sorting and Interactive Demo

To run the color sorting and interactive demo:
```bash
python3 color_sorting_vlm.py
```

When you start the demo, you will be prompted to choose a mode:

1. **Automatic Color Sorting**: The robot will automatically sort all objects into their matching colored zones.
2. **Interactive Text Prompt**: You can enter text prompts to select which object the robot should pick up, and choose where to place it (including a "throw away" option).


Follow the on-screen instructions to interact with the robot and sorting zones.

### Phase 2 Modular Baseline

Run the architecture smoke test and the headless baseline from the repository root:

```bash
python scripts/smoke_test.py
python scripts/run_baseline.py --config configs/baseline.yaml --headless
```

The baseline accepts `--seed`, `--num-objects`, and `--save-video` overrides. Each run writes `run_config.json`, `summary.json`, and `visualization.png` under `runs/`; `--save-video` additionally writes `episode.gif`. The visualization contains the camera frame, candidate boxes, selected target, and CLIP confidence scores.

### Phase 3 Peg-Hole Dataset

Generate deterministic peg-hole scenes with shape-family-disjoint splits. The current v2 configuration creates 16 train, 20 validation, and 100 independent test episodes:

```bash
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml
```

Each episode contains `ground_truth.json`, top and oblique RGB views, randomized candidate IDs/order/positions, analytic fit labels, and the explicitly labeled `oracle_crop`, `render_mask_assisted`, and `rgb_only` observation inputs. The test split holds out the `asymmetric` shape family from train and validation. The analytic fit label is footprint clearance, not insertion dynamics.

The matching/evaluation commands are:

```bash
python scripts/evaluate_matching.py --config configs/phase4.yaml
python scripts/evaluate_vlm.py --config configs/phase5.yaml --adapter mock
python scripts/evaluate_confidence.py --config configs/phase6.yaml
```

BLIP VQA is an integration adapter. Its YES/NO score is a forced-label proxy (`YES=0.75`, `NO=0.25`), not a calibrated token probability. Ties abstain as `uncertain`.

### Phase 6 Confidence and Abstention

Run the validation-calibrated confidence and partial-observation evaluation:

```bash
python scripts/evaluate_confidence.py --config configs/phase6.yaml
```

The evaluator reports top-1/top-3 accuracy, ECE, risk-coverage, abstention, and fixed-coverage comparisons for margin, multi-view consistency, and perturbation stability. Thresholds are selected on `val` and applied once to `test`. The current dataset has two native views; the three-view condition uses a clearly labeled deterministic mirrored-view augmentation.

### Phase 7 Report and Demo

Generate the report figures, `summary.csv`, and a 36-second reproducible GIF:

```bash
python scripts/generate_report.py --phase6-json evaluations/phase6/aggregate.json --phase6-results evaluations/phase6/results.jsonl --dataset datasets/peg_hole_v2 --phase4-dir evaluations/phase4 --output-dir results --duration-seconds 36 --fps 5
```

The generated report distinguishes reproduced/preliminary Chamfer results from the proposed confidence-aware extension and records known synthetic-data limitations in [results/README.md](results/README.md).

### Phase 8 Collision-Proxy Insertion Validation

Run the physics-assisted fit check on the 100-episode test split:

```bash
python scripts/generate_peg_hole_dataset.py --config configs/peg_hole.yaml --output-dir datasets/peg_hole_v2_phase8
python scripts/evaluate_insertion.py --config configs/phase8.yaml
```

Phase 8 drops a collision-enabled peg into a perimeter-wall/floor proxy for each candidate and records wall contacts, settling height, lateral displacement, target insertion, distractor rejection, and agreement with the analytic fit label. The proxy is intentionally not presented as a full contact-rich robot insertion controller.

### Phase 9 Four-Object Confidence-Gated Panda Insertion Mission

Run the CLIP-driven GUI integration demo from the repository root:

```bash
python scripts/run_phase9_demo.py --matcher clip --keep-open
```

The mission creates four independent peg-hole pairs beside the Panda: cylinder, sphere, square, and cuboid. Seeded pseudo-random peg slots, yaws, candidate IDs, and execution order are shuffled, while hole positions remain fixed. Cylinder and sphere holes use low circular-ring collision proxies; square and cuboid holes use low box-wall proxies. A grasp is accepted only when both Panda finger links have side contacts with opposing normals, a nonzero reasonable finger gap, a peg-centered finger midpoint, a fixed grasp constraint, and an actual lift. After validation, the controller holds the measured finger joint openings instead of continuing toward a fully closed target, so the fingers do not visually pass through the constrained peg during transport. Finger/peg collision is intentionally disabled during the fixed-constraint carry to avoid contact-solver forces fighting the hand/peg constraint; real finger contacts are still required at grasp confirmation and collision is restored for release. The controller uses shape-specific pre-grasp heights, high safe waypoints on multi-object return paths, and an intermediate release descent so open fingers do not knock pending pegs or trigger an IK branch jump. The held peg is yaw-aligned in the air when needed, then released as a dynamic body above the rim; there is no pose teleport into the hole. A mission passes only when every released peg is inside its opening, supported by that hole's floor, stable, and free of final wall contact.

For a deterministic controller and collision-proxy check without downloading a model, use:

```bash
python scripts/run_phase9_demo.py --headless --matcher oracle --no-sleep --no-video
```

The configured per-side geometric clearance is `4 mm`; the cuboid hole has an explicit additional `3 mm` pre-scale tolerance for the narrow-side IK residual. Effective proxy clearances and shape-specific pre-grasp heights are recorded alongside it in `summary.json`. If the CLIP model is not cached, configure the same Hugging Face proxy used by the rest of the project before launching it.

The explicit oracle control mode is:

```bash
python scripts/run_phase9_demo.py --matcher oracle --keep-open
```

The `oracle` matcher is based on the analytic corresponding-pair label; it demonstrates the four-object robot/action path and must not be reported as VLM accuracy. Other target holes are still candidates for each active peg, so the selected hole is removed only after a successful one-to-one insertion.

To run the actual image matching baselines through the same GUI action gate:

```bash
python scripts/run_phase9_demo.py --matcher chamfer --keep-open
python scripts/run_phase9_demo.py --matcher clip --keep-open
```

The run writes `summary.json` and, by default, `phase9_mission.gif` under `runs/phase9/`. A Chamfer or CLIP prediction can abstain or select an incompatible candidate; the mission records that step as failed/abstained and continues evaluating the remaining objects. Only four successful physics releases produce `MISSION_PASS=true`; the CLI also prints `COMPLETED` and per-step statuses. The hole geometry remains a low-profile socket collision proxy, and confidence is not calibrated token probability. See [PHASE9_SUMMARY.md](PHASE9_SUMMARY.md) for the task specification and current oracle validation.

---





## Technical Details

### CLIP Vision-Language Matching

The robot uses CLIP to match images of objects to your text prompt:

```python
# For each object crop, CLIP returns a similarity score with the text prompt
similarity_scores = model.compute_object_similarity(crops, text_prompt)
selected_object, best_score = model.select_best_object(similarity_scores)
# Robot picks and places the selected object
```

CLIP compares each cropped object image to your prompt (like "red cube") and returns a score for each. The robot picks the object with the highest score and moves it as you choose.

### Inverse Kinematics

The robot uses PyBullet's built-in inverse kinematics solver to calculate joint angles needed to reach target positions:

```python
joint_positions = p.calculateInverseKinematics(
    self.robot_id,
    endEffectorLinkIndex=11,  # Panda end-effector link
    targetPosition=target_position,
    targetOrientation=target_orientation
)
```

### Robot Control

Joint control uses position control mode:
- **Position Control**: Joints move to target positions
- **Gripper Control**: Two-finger gripper with synchronized motion

## Future Enhancements

Possible extensions to this project:
- Integrate advanced object segmentation for more accurate identification.
- Detect and localize area zones visually, not just by preset coordinates.
- Enable multi-object reasoning for commands involving relationships (e.g., “Stack all blue cubes, then put the red sphere on top”).

## Resources

- [PyBullet Documentation](https://pybullet.org/)
- [Panda Robot Specifications](https://www.franka.de/technology)
- [Robotics and Physics Simulation Tutorial](https://pybullet.org/wordpress/)

## Contributing

Feel free to fork, modify, and submit PRs! Suggestions and improvements are welcome.
