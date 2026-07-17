"""
MONAI Aegis Vision-Language Detector — VLMTextDetector

Optional Florence-2 vision-language pass that finds burnt-in text regions
EasyOCR may miss (stylized overlays, low-contrast annotations, non-Latin
glyphs under a single-language OCR config). Detected regions are merged
into the pixel-redaction pipeline and classified by the same NER / safelist
logic as OCR hits.

Thread-safe: uses threading.local() for per-thread model instances.
Disabled by default — enable via ``ocr.vlm.enabled: true``.

Requires ``timm`` (Florence-2 dependency). Install with::

    pip install timm
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from monai_aegis.config.config_loader import as_bool

logger = logging.getLogger(__name__)

__all__ = ["VLMTextDetector"]


def _quad_to_easyocr_bbox(quad: List[float]) -> List[List[int]]:
    """Convert Florence-2 flat quad ``[x0,y0,x1,y1,x2,y2,x3,y3]`` to EasyOCR corners."""
    pts = [
        [int(round(quad[0])), int(round(quad[1]))],
        [int(round(quad[2])), int(round(quad[3]))],
        [int(round(quad[4])), int(round(quad[5]))],
        [int(round(quad[6])), int(round(quad[7]))],
    ]
    return pts


def _xyxy_to_easyocr_bbox(box: List[float]) -> List[List[int]]:
    """Convert ``[x0, y0, x1, y1]`` to EasyOCR 4-corner format."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bbox_iou_xywh(a: List[int], b: List[int]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _easyocr_bbox_xywh(bbox: Any) -> List[int]:
    xs = [float(pt[0]) for pt in bbox]
    ys = [float(pt[1]) for pt in bbox]
    x0, y0 = min(xs), min(ys)
    return [int(round(x0)), int(round(y0)),
            int(round(max(xs) - x0)), int(round(max(ys) - y0))]


class VLMTextDetector:
    """Florence-2 OCR_WITH_REGION (or custom prompt) text detector.

    Args:
        config: Full pipeline config; reads ``ocr.vlm`` and shares
            ``ner.device`` / ``ner.local_files_only`` when VLM-specific
            keys are omitted.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vlm = config.get("ocr", {}).get("vlm", {})
        ner = config.get("ner", {})
        self.enabled = as_bool(vlm.get("enabled"), default=False)
        self.model_name = vlm.get("model_name", "microsoft/Florence-2-base")
        self.model_revision = vlm.get("model_revision") or None
        self.local_files_only = as_bool(
            vlm.get("local_files_only", ner.get("local_files_only")),
            default=False,
        )
        self.device = vlm.get("device", ner.get("device", "cpu"))
        self.task = vlm.get("task", "<OCR_WITH_REGION>")
        self.text_input = vlm.get("text_input") or None
        self.overlap_iou_threshold = float(vlm.get("overlap_iou_threshold", 0.5))
        self.max_new_tokens = int(vlm.get("max_new_tokens", 1024))
        self._thread_local = threading.local()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_thread_local", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread_local = threading.local()

    def _ensure_models(self) -> Tuple[Any, Any]:
        if not hasattr(self._thread_local, "processor"):
            if self.local_files_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                import timm  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "ocr.vlm requires the 'timm' package. Install with: pip install timm"
                ) from e
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(
                "Loading VLM '%s' (revision: %s, offline: %s) on device '%s' "
                "(thread: %s)",
                self.model_name,
                self.model_revision or "default",
                self.local_files_only,
                self.device,
                threading.current_thread().name,
            )
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            model.to(self.device)
            model.eval()
            self._thread_local.processor = processor
            self._thread_local.model = model
        return self._thread_local.processor, self._thread_local.model

    def _parse_ocr_regions(
        self,
        parsed: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> List[Tuple[Any, str, float]]:
        """Normalize Florence-2 post-process output to EasyOCR-like triples."""
        # Florence-2 may key results by task prompt.
        payload = parsed.get(self.task, parsed)
        if not isinstance(payload, dict):
            return []

        labels = payload.get("labels") or payload.get("ocr_text") or []
        quads = payload.get("quad_boxes") or []
        boxes = payload.get("bboxes") or payload.get("boxes") or []

        results: List[Tuple[Any, str, float]] = []
        if quads and labels:
            for quad, label in zip(quads, labels):
                text = str(label).strip()
                if not text:
                    continue
                if len(quad) >= 8:
                    bbox = _quad_to_easyocr_bbox(list(quad)[:8])
                elif len(quad) >= 4:
                    bbox = _xyxy_to_easyocr_bbox(list(quad)[:4])
                else:
                    continue
                results.append((bbox, text, 0.9))
        elif boxes and labels:
            for box, label in zip(boxes, labels):
                text = str(label).strip()
                if not text:
                    continue
                bbox = _xyxy_to_easyocr_bbox(list(box)[:4])
                results.append((bbox, text, 0.9))
        return results

    def detect(self, pixel_array: np.ndarray) -> List[Tuple[Any, str, float]]:
        """Run the VLM OCR task on a channel-last uint8 image.

        Returns:
            List of ``(bbox, text, confidence)`` in EasyOCR bbox format.
        """
        if not self.enabled:
            return []

        from PIL import Image
        import torch

        processor, model = self._ensure_models()
        if pixel_array.ndim == 2:
            pil = Image.fromarray(pixel_array).convert("RGB")
        else:
            pil = Image.fromarray(pixel_array).convert("RGB")

        prompt = self.task
        if self.text_input:
            prompt = f"{self.task}{self.text_input}"

        inputs = processor(text=prompt, images=pil, return_tensors="pt")
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=self.task,
            image_size=(pil.width, pil.height),
        )
        return self._parse_ocr_regions(parsed, (pil.width, pil.height))

    def merge_novel_detections(
        self,
        existing: List[Tuple[Any, str, float, str]],
        vlm_results: List[Tuple[Any, str, float]],
    ) -> List[Tuple[Any, str, float, str]]:
        """Append VLM hits that do not substantially overlap existing boxes."""
        merged = list(existing)
        existing_xywh = [_easyocr_bbox_xywh(b) for (b, *_rest) in existing]
        for bbox, text, prob in vlm_results:
            xywh = _easyocr_bbox_xywh(bbox)
            if any(
                _bbox_iou_xywh(xywh, ex) >= self.overlap_iou_threshold
                for ex in existing_xywh
            ):
                continue
            merged.append((bbox, text, prob, "vlm"))
            existing_xywh.append(xywh)
        return merged
