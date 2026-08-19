"""Unified generative VLM matcher interface and adapters for Phase 5."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from PIL import Image

from .phase4 import chamfer_distance


def _normalise_views(peg_images, hole_images) -> Iterable[Tuple[str, Image.Image, Image.Image]]:
    if isinstance(peg_images, Mapping) and isinstance(hole_images, Mapping):
        view_ids = [view for view in peg_images if view in hole_images]
        for view in view_ids:
            yield view, peg_images[view], hole_images[view]
        return
    peg_list = list(peg_images)
    hole_list = list(hole_images)
    if len(peg_list) != len(hole_list):
        raise ValueError("peg_images and hole_images must have the same number of views")
    for index, (peg, hole) in enumerate(zip(peg_list, hole_list)):
        yield f"view_{index}", peg, hole


def _force_yes_no(raw_text: str) -> Tuple[str, str]:
    """Convert unconstrained model text into an evaluation-safe label."""
    lowered = raw_text.strip().lower()
    yes_match = re.search(r"\byes\b", lowered)
    no_match = re.search(r"\bno\b", lowered)
    if yes_match and (not no_match or yes_match.start() < no_match.start()):
        return "YES", "parsed_yes"
    if no_match:
        return "NO", "parsed_no"
    return "NO", "unrecognized_forced_no"


class VLMMatcher(ABC):
    """Stable interface for all generative VLM adapters."""

    model_name: str = "unknown"

    @abstractmethod
    def match(self, peg_images, hole_images, prompt) -> dict:
        """Return label, score, raw_text, model, and per-view scores."""


class MockVLMMatcher(VLMMatcher):
    """Deterministic offline adapter used to validate the Phase 5 pipeline."""

    model_name = "mock-vlm"

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def match(self, peg_images, hole_images, prompt) -> dict:
        del prompt
        view_scores = {}
        for view, peg, hole in _normalise_views(peg_images, hole_images):
            distance = chamfer_distance(peg, hole)
            view_scores[view] = float(math.exp(-distance / 8.0))
        score = float(sum(view_scores.values()) / max(len(view_scores), 1))
        label = "YES" if score >= self.threshold else "NO"
        return {
            "label": label,
            "score": score,
            "raw_text": label,
            "model": self.model_name,
            "view_scores": view_scores,
            "adapter": "mock",
            "parse_status": "deterministic_mock",
        }


class BlipVQAMatcher(VLMMatcher):
    """Optional actual VLM adapter using Salesforce BLIP VQA."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        cache_dir=None,
        device: Optional[str] = None,
    ):
        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.processor = BlipProcessor.from_pretrained(model_name, **kwargs)
        model_kwargs = dict(kwargs)
        model_kwargs["use_safetensors"] = False
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = BlipForQuestionAnswering.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pair_image(peg: Image.Image, hole: Image.Image) -> Image.Image:
        height = max(peg.height, hole.height)
        peg = peg.convert("RGB").resize((max(1, int(peg.width * height / peg.height)), height))
        hole = hole.convert("RGB").resize((max(1, int(hole.width * height / hole.height)), height))
        canvas = Image.new("RGB", (peg.width + hole.width + 8, height), "white")
        canvas.paste(peg, (0, 0))
        canvas.paste(hole, (peg.width + 8, 0))
        return canvas

    def _answer(self, image: Image.Image, prompt: str) -> Tuple[str, str]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=8)
        raw_text = self.processor.decode(generated[0], skip_special_tokens=True)
        return raw_text, "generated"

    def match(self, peg_images, hole_images, prompt) -> dict:
        view_scores: Dict[str, float] = {}
        raw_outputs = {}
        parse_status = {}
        for view, peg, hole in _normalise_views(peg_images, hole_images):
            raw_text, generation_status = self._answer(self._pair_image(peg, hole), prompt)
            label, parse_reason = _force_yes_no(raw_text)
            # BLIP VQA generation does not expose a calibrated YES/NO token
            # probability in this adapter. Keep the score explicitly as a
            # forced-label proxy rather than presenting it as calibrated.
            view_scores[view] = 0.75 if label == "YES" else 0.25
            raw_outputs[view] = raw_text
            parse_status[view] = f"{generation_status}:{parse_reason}"
        score = float(sum(view_scores.values()) / max(len(view_scores), 1))
        label = "YES" if score >= 0.5 else "NO"
        return {
            "label": label,
            "score": score,
            "raw_text": " | ".join(f"{view}: {text}" for view, text in raw_outputs.items()),
            "model": self.model_name,
            "view_scores": view_scores,
            "adapter": "blip_vqa",
            "parse_status": parse_status,
            "score_definition": "forced-label proxy: YES=0.75, NO=0.25; not calibrated",
        }
