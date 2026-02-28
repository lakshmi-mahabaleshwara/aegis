"""
MONAI Aegis Pixel Transforms — RedactPixelPHI / RedactPixelPHId

OCR-based visual PHI detection and redaction for medical images.
Uses EasyOCR for text detection with two classification modes:
  1. Stanford NER (default) — semantic PHI classification via transformer model
  2. Regex Safelist (fallback) — pattern-based clinical marker preservation

Thread-safe: uses threading.local() for per-thread EasyOCR and NER instances.

Raises:
    PixelRedactionError: When OCR detection or redaction fails.
"""
import re
import threading
import numpy as np
import easyocr
import logging
import torch
from typing import Dict, Hashable, Mapping, Any, List, Optional, Tuple
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform, InvertibleTransform
from monai.data import MetaTensor

from transforms.exceptions import PixelRedactionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def detect_text(
    pixel_array: np.ndarray,
    ocr_reader: easyocr.Reader,
    config: Dict[str, Any],
    ner_classifier: Optional[Any] = None,
) -> Tuple[List[List], Dict[str, int]]:
    """
    Detect text in an image and return bounding boxes of PHI regions.

    Uses one of two classification strategies:
      - **NER mode** (when ``ner_classifier`` is provided): sends OCR text to
        the Stanford de-identifier NER model for semantic PHI classification.
      - **Safelist mode** (fallback): checks text against regex patterns from config.

    Args:
        pixel_array: Image data as numpy array, ``uint8`` preferred.
            Shape should be ``(H, W)`` for grayscale or ``(H, W, 3)`` for RGB.
        ocr_reader: Initialized EasyOCR Reader instance.
        config: Configuration dict with ``'ocr'`` and ``'safelist'`` sections.
        ner_classifier: Optional :py:class:`PHIClassifier` instance for
            NER-based detection. If ``None``, falls back to safelist mode.

    Returns:
        Tuple of ``(bboxes, stats)``:
          - ``bboxes``: List of EasyOCR-format bounding boxes for redaction.
          - ``stats``: Dict with ``total_detections``, ``low_confidence_count``,
            ``safelisted_count``, ``ner_classified_count``, ``redacted_count``.

    Raises:
        PixelRedactionError: If EasyOCR or NER classification raises an error.
    """
    bboxes = []
    stats = {
        'total_detections': 0,
        'low_confidence_count': 0,
        'safelisted_count': 0,
        'ner_classified_count': 0,
        'redacted_count': 0,
    }
    ocr_settings = config.get('ocr', {})

    decoder = ocr_settings.get('decoder', 'beamsearch')
    beam_width = ocr_settings.get('beam_width', 5)
    confidence_threshold = ocr_settings.get('confidence_threshold', 0.4)

    try:
        results = ocr_reader.readtext(
            pixel_array,
            decoder=decoder,
            beamWidth=beam_width
        )

        stats['total_detections'] = len(results)

        # Separate low-confidence detections first
        confident_results = []
        for (bbox, text, prob) in results:
            if prob < confidence_threshold:
                stats['low_confidence_count'] += 1
            else:
                confident_results.append((bbox, text, prob))

        if ner_classifier is not None:
            # --- NER Mode: semantic PHI classification ---
            texts = [text for (_, text, _) in confident_results]
            if texts:
                phi_flags = ner_classifier.classify_texts(texts)
                for (bbox, text, prob), is_phi in zip(confident_results, phi_flags):
                    stats['ner_classified_count'] += 1
                    if is_phi:
                        bboxes.append(bbox)
                        stats['redacted_count'] += 1
                    else:
                        stats['safelisted_count'] += 1
        else:
            # --- Safelist Mode: regex-based clinical marker preservation ---
            safelist_patterns = config.get('safelist', [])
            compiled_patterns = [re.compile(p) for p in safelist_patterns]

            for (bbox, text, prob) in confident_results:
                is_safe = any(pattern.search(text) for pattern in compiled_patterns)
                if is_safe:
                    stats['safelisted_count'] += 1
                else:
                    bboxes.append(bbox)
                    stats['redacted_count'] += 1

    except Exception as e:
        raise PixelRedactionError(
            f"OCR/redaction failed: {e}",
            transform="detect_text",
        ) from e

    return bboxes, stats


def apply_redaction(
    pixel_array: np.ndarray,
    bboxes: List[List[List[int]]],
) -> np.ndarray:
    """
    Applies black-box redaction to specified bounding boxes.
    Works on a copy to prevent side effects in the transform chain.

    Args:
        pixel_array: Image data as numpy array.
        bboxes: List of EasyOCR-format bounding boxes to redact.

    Returns:
        Redacted copy of the image.
    """
    output_array = pixel_array.copy()

    for bbox in bboxes:
        tl, tr, br, bl = bbox
        x_min = int(max(0, min(tl[0], bl[0])))
        x_max = int(min(output_array.shape[1], max(tr[0], br[0])))
        y_min = int(max(0, min(tl[1], tr[1])))
        y_max = int(min(output_array.shape[0], max(bl[1], br[1])))

        if output_array.ndim == 3:
            output_array[y_min:y_max, x_min:x_max, :] = 0
        else:
            output_array[y_min:y_max, x_min:x_max] = 0

    return output_array


# ---------------------------------------------------------------------------
# Array Transform
# ---------------------------------------------------------------------------

class RedactPixelPHI(Transform):
    """
    Array transform: Detect and redact burned-in PHI text from a medical image.

    Uses EasyOCR to detect text, then classifies each detection using either:
      - **Stanford NER model** (when ``ner.enabled: true`` in config) for
        semantic, context-aware PHI detection.
      - **Regex safelist** (fallback) for pattern-based clinical marker preservation.

    Thread-safe: uses ``threading.local()`` so each worker thread gets its
    own EasyOCR reader and NER classifier instances (loaded lazily on first use).
    This enables safe use with ``DataLoader(num_workers > 0)``.

    Args:
        config: Configuration dict with 'ocr', 'ner', and 'safelist' sections.

    Example::

        transform = RedactPixelPHI(config=config)
        redacted = transform(image_array)  # numpy array in, numpy array out
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._thread_local = threading.local()
        self._ner_enabled = config.get('ner', {}).get('enabled', False)

    @property
    def reader(self) -> easyocr.Reader:
        """Lazily create a per-thread EasyOCR reader."""
        if not hasattr(self._thread_local, 'reader'):
            self._thread_local.reader = easyocr.Reader(
                self.config['ocr'].get('languages', ['en']),
                gpu=self.config['ocr'].get('gpu_usage', False)
            )
        return self._thread_local.reader

    @property
    def ner_classifier(self) -> Optional[Any]:
        """Lazily create a per-thread NER classifier (if NER is enabled).

        Returns:
            A :py:class:`PHIClassifier` instance, or ``None`` if NER is disabled.
        """
        if not self._ner_enabled:
            return None
        if not hasattr(self._thread_local, 'ner_classifier'):
            from transforms.ner_classifier import PHIClassifier
            self._thread_local.ner_classifier = PHIClassifier(self.config)
            logger.info("NER classifier initialized for thread: %s",
                        threading.current_thread().name)
        return self._thread_local.ner_classifier

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: Image array in channel-first format (C, H, W).

        Returns:
            Redacted image array in same format and dtype.
        """
        orig_dtype = image.dtype
        pixel_array = image.copy()

        # Handle channel-first → channel-last for OCR
        is_grayscale = False
        is_rgb = False

        if pixel_array.ndim == 3:
            if pixel_array.shape[0] == 1:
                is_grayscale = True
                pixel_array = pixel_array.squeeze(0)
            elif pixel_array.shape[0] == 3:
                is_rgb = True
                pixel_array = np.transpose(pixel_array, (1, 2, 0))

        # Normalize to uint8 for EasyOCR
        if pixel_array.dtype != np.uint8:
            p_min, p_max = pixel_array.min(), pixel_array.max()
            norm_ocr = (255 * (pixel_array - p_min) / (p_max - p_min + 1e-5)).astype(np.uint8)
        else:
            norm_ocr = pixel_array

        # Detect and Redact (with NER if enabled)
        bboxes, self.last_stats = detect_text(
            norm_ocr, self.reader, self.config, self.ner_classifier
        )
        redacted = apply_redaction(pixel_array, bboxes)

        # Restore channel-first
        if is_grayscale:
            redacted = redacted[np.newaxis, ...]
        elif is_rgb:
            redacted = np.transpose(redacted, (2, 0, 1))

        return redacted.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Dictionary Transform (InvertibleTransform)
# ---------------------------------------------------------------------------

class RedactPixelPHId(MapTransform, InvertibleTransform):
    """
    Dictionary transform: Detect and redact burned-in PHI text from medical images.

    Dictionary-based wrapper of :py:class:`RedactPixelPHI`.
    Inherits from both ``MapTransform`` (for ``keys`` / ``key_iterator``) and
    ``InvertibleTransform`` (for ``push_transform`` / ``pop_transform``).
    This is the standard MONAI pattern for invertible dictionary transforms
    (e.g., ``Spacingd``, ``Orientationd``).

    Thread-safe via per-thread EasyOCR reader (``threading.local()``).
    Preserves MetaTensor metadata throughout the transform.

    The ``inverse()`` method cannot restore original pixel values (they are
    permanently zeroed), but it cleans up the redaction mask and stats from
    the data dictionary and pops the transform history entry.

    Args:
        keys: Keys of the data dictionary to process.
        config: Configuration dict with 'ocr' and 'safelist' sections.
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        transform = RedactPixelPHId(keys=["image"], config=config)
        data = transform({"image": meta_tensor})
    """

    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = RedactPixelPHI(config=config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Detect and redact PHI text for each keyed image in the data dict.

        Side-effects written per key:
            - ``{key}`` — redacted MetaTensor with updated pixels.
            - ``{key}_redaction_stats`` — dict of detection/redaction counts.
            - ``{key}_redaction_mask`` — binary ``uint8`` array ``(H, W)``
              where ``1`` = pixel was redacted.
            - Transform pushed to ``MetaTensor.applied_operations`` for
              MONAI invertibility tracking.

        Args:
            data: Pipeline data dictionary containing MetaTensors.

        Returns:
            Updated data dictionary with redacted images and metadata.

        Raises:
            PixelRedactionError: If OCR detection or redaction fails.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            # Extract filepath for error context
            meta = d.get(f"{key}_meta_dict", {})
            filepath = meta.get('filename_or_obj', '<unknown>')
            if isinstance(filepath, list):
                filepath = filepath[0]

            try:
                input_tensor = d[key]

                # Safety lock: warn if prior spatial transforms detected
                if isinstance(input_tensor, MetaTensor):
                    applied_ops = input_tensor.applied_operations if hasattr(input_tensor, 'applied_operations') else []
                    if len(applied_ops) > 0:
                        logger.warning(
                            "Aegis detected prior spatial transforms; "
                            "OCR accuracy may be compromised."
                        )

                # Extract numpy from MetaTensor/Tensor
                if hasattr(input_tensor, 'detach'):
                    pixel_array = input_tensor.detach().cpu().numpy()
                else:
                    pixel_array = np.array(input_tensor)

                # Apply array transform
                redacted = self.transform(pixel_array)

                # Store redaction stats for downstream routing
                stats = getattr(self.transform, 'last_stats', {})
                d[f"{key}_redaction_stats"] = stats

                # Generate binary redaction mask matched to spatial_shape
                spatial_shape = pixel_array.shape[1:]  # (H, W) from (C, H, W)
                redaction_mask = np.zeros(spatial_shape, dtype=np.uint8)
                bboxes = getattr(self.transform, 'last_stats', {}).get('_bboxes', [])

                # Recompute mask from the difference between original and redacted
                if pixel_array.ndim == 3:
                    diff = np.any(pixel_array != redacted, axis=0)
                else:
                    diff = pixel_array != redacted
                redaction_mask = diff.astype(np.uint8)
                d[f"{key}_redaction_mask"] = redaction_mask

                # Preserve MetaTensor
                if isinstance(input_tensor, MetaTensor):
                    d[key] = MetaTensor(torch.as_tensor(redacted), meta=input_tensor.meta)
                else:
                    d[key] = redacted

                # Push transform info for MONAI invertibility tracking
                self.push_transform(
                    d,
                    key,
                    extra_info={
                        'redaction_stats': stats,
                        'redaction_mask_shape': list(redaction_mask.shape),
                    }
                )
            except PixelRedactionError:
                raise
            except Exception as e:
                raise PixelRedactionError(
                    f"Pixel redaction failed: {e}",
                    filepath=str(filepath),
                    transform="RedactPixelPHId",
                ) from e

        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Undo the bookkeeping added by ``__call__``.

        .. note::
            Pixel values are **permanently zeroed** by redaction and cannot
            be restored. This method only cleans up the redaction mask,
            stats, and MONAI transform history.

        Args:
            data: Pipeline data dictionary.

        Returns:
            Data dictionary with redaction side-keys removed and
            transform history popped.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            # Pop the transform history entry
            self.pop_transform(d, key)

            # Clean up side-keys added by __call__
            d.pop(f"{key}_redaction_mask", None)
            d.pop(f"{key}_redaction_stats", None)

        return d
