"""
MONAI Aegis Pixel Transforms — RedactPixelPHI / RedactPixelPHId

OCR-based visual PHI detection and redaction for medical images.
Uses EasyOCR with configurable safelist to preserve clinical markers.

Thread-safe: uses threading.local() for per-thread EasyOCR reader instances.
"""
import re
import threading
import numpy as np
import easyocr
import logging
import torch
from typing import Dict, Hashable, Mapping, Any, List, Optional
from monai.transforms import Transform, MapTransform, InvertibleTransform
from monai.data import MetaTensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def detect_text(
    pixel_array: np.ndarray,
    ocr_reader: easyocr.Reader,
    config: Dict[str, Any]
) -> tuple:
    """
    Detects text in an image and returns bounding boxes for redaction,
    respecting the safelist of clinical markers.

    Args:
        pixel_array: Image data as numpy array (uint8 preferred).
        ocr_reader: Initialized EasyOCR Reader instance.
        config: Configuration dict with 'ocr' and 'safelist' sections.

    Returns:
        Tuple of (bboxes, stats) where stats is a dict with detection metrics.
    """
    bboxes = []
    stats = {
        'total_detections': 0,
        'low_confidence_count': 0,
        'safelisted_count': 0,
        'redacted_count': 0,
    }
    ocr_settings = config.get('ocr', {})
    safelist_patterns = config.get('safelist', [])

    decoder = ocr_settings.get('decoder', 'beamsearch')
    beam_width = ocr_settings.get('beam_width', 5)
    confidence_threshold = ocr_settings.get('confidence_threshold', 0.4)

    # Compile regex patterns
    compiled_patterns = [re.compile(p) for p in safelist_patterns]

    try:
        results = ocr_reader.readtext(
            pixel_array,
            decoder=decoder,
            beamWidth=beam_width
        )

        stats['total_detections'] = len(results)

        for (bbox, text, prob) in results:
            if prob < confidence_threshold:
                stats['low_confidence_count'] += 1
                continue

            # Check Safelist: skip clinical markers
            is_safe = any(pattern.search(text) for pattern in compiled_patterns)

            if is_safe:
                stats['safelisted_count'] += 1
            else:
                bboxes.append(bbox)
                stats['redacted_count'] += 1

    except Exception as e:
        logger.error(f"Error in detect_text: {e}")

    return bboxes, stats


def apply_redaction(pixel_array: np.ndarray, bboxes: List[list]) -> np.ndarray:
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

    Uses EasyOCR to detect text, checks against a safelist of clinical markers,
    and applies black-box redaction to non-safe text regions.

    Thread-safe: uses ``threading.local()`` so each worker thread gets its
    own EasyOCR reader instance (the model is loaded lazily on first use).
    This enables safe use with ``DataLoader(num_workers > 0)``.

    Args:
        config: Configuration dict with 'ocr' and 'safelist' sections.

    Example::

        transform = RedactPixelPHI(config=config)
        redacted = transform(image_array)  # numpy array in, numpy array out
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._thread_local = threading.local()

    @property
    def reader(self) -> easyocr.Reader:
        """Lazily create a per-thread EasyOCR reader."""
        if not hasattr(self._thread_local, 'reader'):
            self._thread_local.reader = easyocr.Reader(
                self.config['ocr'].get('languages', ['en']),
                gpu=self.config['ocr'].get('gpu_usage', False)
            )
        return self._thread_local.reader

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

        # Detect and Redact
        bboxes, self.last_stats = detect_text(norm_ocr, self.reader, self.config)
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
    Inherits from ``InvertibleTransform`` (which extends ``MapTransform``),
    enabling the MONAI ecosystem to track redacted regions through downstream
    spatial transforms (e.g., ``Spacingd``).

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

    def __init__(self, keys, config: Dict[str, Any], allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.transform = RedactPixelPHI(config=config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
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
            # Mask is 1 where pixels were zeroed (redacted), 0 elsewhere
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

        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            # Pop the transform history entry
            self.pop_transform(d, key)

            # Clean up side-keys added by __call__
            d.pop(f"{key}_redaction_mask", None)
            d.pop(f"{key}_redaction_stats", None)

        return d
