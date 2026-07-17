"""
MONAI Aegis Handwriting Recognition — HandwritingRecognizer

Optional TrOCR-based re-recognition of burnt-in handwritten PHI.
EasyOCR detects text regions; when handwriting mode is enabled, low-
confidence (or all) crops are re-read with Microsoft's TrOCR handwritten
model, which recovers cursive / printed-handwriting that print OCR misses.

Thread-safe: uses threading.local() for per-thread model instances.
Disabled by default — enable via ``ocr.handwriting.enabled: true``.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from monai_aegis.config.config_loader import as_bool

logger = logging.getLogger(__name__)

__all__ = ["HandwritingRecognizer"]


def _crop_bbox(image: np.ndarray, bbox: Any) -> Optional[np.ndarray]:
    """Crop an EasyOCR 4-corner bbox from a channel-last image."""
    xs = [float(pt[0]) for pt in bbox]
    ys = [float(pt[1]) for pt in bbox]
    x0, x1 = int(max(0, min(xs))), int(min(image.shape[1], max(xs)))
    y0, y1 = int(max(0, min(ys))), int(min(image.shape[0], max(ys)))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


class HandwritingRecognizer:
    """TrOCR handwritten-text re-recognizer for OCR region crops.

    Args:
        config: Full pipeline config; reads ``ocr.handwriting`` and shares
            ``ner.device`` / ``ner.local_files_only`` when handwriting-specific
            keys are omitted.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        hw = config.get("ocr", {}).get("handwriting", {})
        ner = config.get("ner", {})
        self.enabled = as_bool(hw.get("enabled"), default=False)
        self.model_name = hw.get(
            "model_name", "microsoft/trocr-base-handwritten"
        )
        self.model_revision = hw.get("model_revision") or None
        self.local_files_only = as_bool(
            hw.get("local_files_only", ner.get("local_files_only")),
            default=False,
        )
        self.device = hw.get("device", ner.get("device", "cpu"))
        self.re_recognize_low_confidence = as_bool(
            hw.get("re_recognize_low_confidence"), default=True
        )
        self.re_recognize_all = as_bool(
            hw.get("re_recognize_all"), default=False
        )
        self.confidence_threshold = float(
            hw.get("confidence_threshold", 0.4)
        )
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
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            logger.info(
                "Loading handwriting model '%s' (revision: %s, offline: %s) "
                "on device '%s' (thread: %s)",
                self.model_name,
                self.model_revision or "default",
                self.local_files_only,
                self.device,
                threading.current_thread().name,
            )
            processor = TrOCRProcessor.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                local_files_only=self.local_files_only,
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                local_files_only=self.local_files_only,
            )
            model.to(self.device)
            model.eval()
            self._thread_local.processor = processor
            self._thread_local.model = model
        return self._thread_local.processor, self._thread_local.model

    def recognize_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """Recognize handwritten text in a single uint8 crop.

        Returns:
            ``(text, confidence)``. Confidence is 1.0 when the model returns
            non-empty text (TrOCR does not expose per-token scores in the
            default generate path); 0.0 for empty output.
        """
        from PIL import Image
        import torch

        processor, model = self._ensure_models()
        if crop.ndim == 2:
            pil = Image.fromarray(crop).convert("RGB")
        else:
            pil = Image.fromarray(crop).convert("RGB")

        pixel_values = processor(images=pil, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            generated = model.generate(pixel_values)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        text = (text or "").strip()
        return text, (1.0 if text else 0.0)

    def should_reread(self, prob: float, ocr_confidence_threshold: float) -> bool:
        """Whether this EasyOCR detection should be re-read with TrOCR."""
        if self.re_recognize_all:
            return True
        if self.re_recognize_low_confidence and prob < ocr_confidence_threshold:
            return True
        return False

    def enrich_results(
        self,
        pixel_array: np.ndarray,
        results: List[Tuple[Any, str, float]],
        ocr_confidence_threshold: float,
    ) -> List[Tuple[Any, str, float, str]]:
        """Re-recognize selected EasyOCR hits; tag each with a source label.

        Args:
            pixel_array: Channel-last uint8 image.
            results: EasyOCR ``(bbox, text, prob)`` triples.
            ocr_confidence_threshold: EasyOCR confidence gate from config.

        Returns:
            List of ``(bbox, text, prob, source)`` where ``source`` is
            ``'easyocr'`` or ``'handwriting'``.
        """
        enriched: List[Tuple[Any, str, float, str]] = []
        for bbox, text, prob in results:
            if not self.enabled or not self.should_reread(prob, ocr_confidence_threshold):
                enriched.append((bbox, text, prob, "easyocr"))
                continue
            crop = _crop_bbox(pixel_array, bbox)
            if crop is None:
                enriched.append((bbox, text, prob, "easyocr"))
                continue
            try:
                hw_text, hw_prob = self.recognize_crop(crop)
            except Exception as e:
                logger.warning("Handwriting re-recognition failed: %s", e)
                enriched.append((bbox, text, prob, "easyocr"))
                continue

            if hw_text and hw_prob >= self.confidence_threshold:
                # Prefer TrOCR when it recovers text from a weak EasyOCR hit,
                # or when re_recognize_all is on and TrOCR is confident.
                if prob < ocr_confidence_threshold or self.re_recognize_all:
                    enriched.append((bbox, hw_text, hw_prob, "handwriting"))
                    continue
            enriched.append((bbox, text, prob, "easyocr"))
        return enriched
