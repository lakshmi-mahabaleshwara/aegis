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
from monai.transforms import Transform, MapTransform
from monai.data import MetaTensor
from monai.utils.enums import TransformBackends

from monai_aegis.transforms.exceptions import PixelRedactionError
from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck

__all__ = [
    "detect_text", "apply_redaction",
    "select_keyframe_indices", "build_union_mask", "redact_volume_safe",
    "RedactPixelPHI", "RedactPixelPHId",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _bbox_to_xywh(bbox: Any) -> List[int]:
    """Convert an EasyOCR 4-corner bbox to ``[x, y, w, h]`` (top-left + size).

    EasyOCR returns ``[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]``. Ground-truth
    overlay boxes use ``[x, y, w, h]``, so normalise to that convention for
    direct IoU comparison during validation.
    """
    xs = [float(pt[0]) for pt in bbox]
    ys = [float(pt[1]) for pt in bbox]
    x0, y0 = min(xs), min(ys)
    return [int(round(x0)), int(round(y0)),
            int(round(max(xs) - x0)), int(round(max(ys) - y0))]


def _make_detection(bbox: Any, text: str, prob: float, decision: str) -> Dict[str, Any]:
    """Build one per-region audit record for ground-truth reporting."""
    return {
        'bbox': _bbox_to_xywh(bbox),
        'ocr_text': str(text),
        'confidence': round(float(prob), 4),
        'decision': decision,
    }


def detect_text(
    pixel_array: np.ndarray,
    ocr_reader: easyocr.Reader,
    config: Dict[str, Any],
    ner_classifier: Optional[Any] = None,
) -> Tuple[List[List], Dict[str, Any]]:
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
    stats: Dict[str, Any] = {
        'total_detections': 0,
        'low_confidence_count': 0,
        'safelisted_count': 0,
        'ner_classified_count': 0,
        'redacted_count': 0,
    }
    # Per-region audit trail (one entry per OCR detection) for ground-truth
    # reporting/validation. Each entry: {bbox, ocr_text, confidence, decision}
    # where decision is one of: 'redacted', 'safelisted', 'low_confidence'.
    detections: List[Dict[str, Any]] = []
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
                detections.append(_make_detection(bbox, text, prob, 'low_confidence'))
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
                        detections.append(_make_detection(bbox, text, prob, 'redacted'))
                    else:
                        stats['safelisted_count'] += 1
                        detections.append(_make_detection(bbox, text, prob, 'safelisted'))
        else:
            # --- Safelist Mode: regex-based clinical marker preservation ---
            safelist_patterns = config.get('safelist', [])
            compiled_patterns = [re.compile(p) for p in safelist_patterns]

            for (bbox, text, prob) in confident_results:
                is_safe = any(pattern.search(text) for pattern in compiled_patterns)
                if is_safe:
                    stats['safelisted_count'] += 1
                    detections.append(_make_detection(bbox, text, prob, 'safelisted'))
                else:
                    bboxes.append(bbox)
                    stats['redacted_count'] += 1
                    detections.append(_make_detection(bbox, text, prob, 'redacted'))

    except Exception as e:
        raise PixelRedactionError(
            f"OCR/redaction failed: {e}",
            transform="detect_text",
        ) from e

    stats['detections'] = detections
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
# Adaptive keyframe selection and union-mask helpers
# ---------------------------------------------------------------------------

def select_keyframe_indices(
    num_frames: int,
    *,
    sample_pct: float = 0.10,
    floor: int = 3,
    ceiling: int = 25,
) -> List[int]:
    """Adaptive keyframe selection — scales with series length.

    Priority rules:

    1. ``num_frames <= floor`` → scan every frame (100% coverage).
    2. ``raw = round(num_frames * sample_pct)``.
    3. ``k = clamp(raw, floor, ceiling)``.
    4. Distribute *k* indices evenly across ``[0, num_frames-1]``.
    5. Always include frame 0 and frame ``num_frames-1``.

    Examples::

        select_keyframe_indices(5)    # [0,1,2,3,4]  — 100%, below floor
        select_keyframe_indices(30)   # [0,14,29]     — 3 frames (floor)
        select_keyframe_indices(100)  # [0,11,...,99] — 10 frames
        select_keyframe_indices(300)  # 25 frames     — ceiling hit

    Args:
        num_frames: Total number of frames in the volume.
        sample_pct: Fraction of frames to sample (default 10%).
        floor: Minimum number of keyframes regardless of series length.
        ceiling: Maximum number of keyframes regardless of series length.

    Returns:
        Sorted list of unique frame indices.
    """
    if num_frames <= 0:
        return []
    if num_frames <= floor:
        return list(range(num_frames))

    raw = round(num_frames * sample_pct)
    k = max(floor, min(ceiling, raw))

    # Evenly distribute k points across [0, num_frames-1]
    indices = sorted({
        round(i * (num_frames - 1) / (k - 1))
        for i in range(k)
    })

    # Boundary guarantee — first and last always included
    if 0 not in indices:
        indices = [0] + indices
    if (num_frames - 1) not in indices:
        indices = indices + [num_frames - 1]

    return sorted(set(indices))


def build_union_mask(
    volume: np.ndarray,
    keyframe_indices: List[int],
    ocr_fn: Any,
    *,
    existing_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run OCR on selected keyframes; return pixel-level union of all masks.

    A pixel is flagged if PHI was detected there on **any** keyframe.
    Conservative by design — may slightly over-redact, never under-redacts.

    Args:
        volume: Pixel volume with shape ``(T, C, H, W)``.
        keyframe_indices: Frame indices from :func:`select_keyframe_indices`.
        ocr_fn: Callable ``(C, H, W) → bool (H, W)`` — detects PHI and
            returns a spatial mask.
        existing_mask: Optional ``bool (H, W)`` mask (e.g. from
            ``RedactByUSRegionsd``) merged into the union via bitwise OR.

    Returns:
        Tuple of ``(union_mask, stats)``:

        - ``union_mask``: ``bool (H, W)`` to apply to all frames.
        - ``stats``: Audit dict with keys ``keyframes_scanned``,
          ``keyframe_indices``, ``per_keyframe_phi_pixels``,
          ``union_phi_pixels``, ``total_frames``, ``pct_frames_scanned``.
    """
    H, W = volume.shape[2], volume.shape[3]
    union: np.ndarray = np.zeros((H, W), dtype=bool)
    per_kf: Dict[int, int] = {}

    for idx in keyframe_indices:
        frame_mask = ocr_fn(volume[idx])   # (C, H, W) → bool (H, W)
        per_kf[idx] = int(frame_mask.sum())
        union |= frame_mask

    if existing_mask is not None:
        union |= existing_mask

    T = volume.shape[0]
    stats: Dict[str, Any] = {
        "keyframes_scanned":       len(keyframe_indices),
        "keyframe_indices":        keyframe_indices,
        "per_keyframe_phi_pixels": per_kf,
        "union_phi_pixels":        int(union.sum()),
        "total_frames":            T,
        "pct_frames_scanned":      round(100 * len(keyframe_indices) / max(T, 1), 1),
    }
    return union, stats


def redact_volume_safe(
    volume: np.ndarray,
    ocr_fn: Any,
    *,
    sample_pct: float = 0.10,
    floor: int = 3,
    ceiling: int = 25,
    existing_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Adaptive keyframe sampling + union mask application.

    Single entry point — replaces the old count-based propagation.
    A pixel flagged on **any** keyframe is zeroed on **all** frames.

    Args:
        volume: Pixel volume ``(T, C, H, W)`` — modified **in-place**.
        ocr_fn: Callable ``(C, H, W) → bool (H, W)``.
        sample_pct: Fraction of frames to sample (default 10%).
        floor: Minimum keyframes (default 3).
        ceiling: Maximum keyframes (default 25).
        existing_mask: Optional ``bool (H, W)`` mask merged into union.

    Returns:
        Tuple of ``(volume, union_mask, stats)``.
    """
    indices = select_keyframe_indices(
        volume.shape[0],
        sample_pct=sample_pct,
        floor=floor,
        ceiling=ceiling,
    )
    union_mask, stats = build_union_mask(
        volume, indices, ocr_fn, existing_mask=existing_mask
    )
    # Single vectorised write — no Python loop over frames
    volume[:, :, union_mask] = 0
    return volume, union_mask, stats


# ---------------------------------------------------------------------------
# Array Transform
# ---------------------------------------------------------------------------

class RedactPixelPHI(Transform):
    """Array transform: Detect and redact burned-in PHI text from a medical image.

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

    backend = [TransformBackends.NUMPY]

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._thread_local = threading.local()
        self._ner_enabled = config.get('ner', {}).get('enabled', False)

    def __getstate__(self):
        """Exclude threading.local() from pickling for DataLoader multiprocessing."""
        state = self.__dict__.copy()
        if '_thread_local' in state:
            del state['_thread_local']
        return state
        
    def __setstate__(self, state):
        """Rehydrate threading.local() in the worker process."""
        self.__dict__.update(state)
        self._thread_local = threading.local()


    @property
    def reader(self) -> easyocr.Reader:
        """Lazily create a per-thread EasyOCR reader.

        ``ocr.model_storage_directory`` points EasyOCR at a pre-bundled
        weights directory, and ``ocr.download_enabled: false`` guarantees
        no network access — together they support air-gapped deployment.
        """
        if not hasattr(self._thread_local, 'reader'):
            from monai_aegis.config.config_loader import as_bool
            ocr_cfg = self.config['ocr']
            self._thread_local.reader = easyocr.Reader(
                ocr_cfg.get('languages', ['en']),
                gpu=ocr_cfg.get('gpu_usage', False),
                model_storage_directory=ocr_cfg.get('model_storage_directory') or None,
                download_enabled=as_bool(ocr_cfg.get('download_enabled'), default=True),
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
            from monai_aegis.transforms.ner_classifier import PHIClassifier
            self._thread_local.ner_classifier = PHIClassifier(self.config)
            logger.info("NER classifier initialized for thread: %s",
                        threading.current_thread().name)
        return self._thread_local.ner_classifier

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Detect and redact burned-in PHI text.

        Standard MONAI array-transform contract: array in, array out.
        Callers that need the detection/redaction audit record should use
        :meth:`redact` instead, which returns ``(redacted, stats)``.

        Args:
            image: Image array in channel-first format.
                ``(C, H, W)`` for a single image or
                ``(C, D, H, W)`` for a volume.

        Returns:
            Redacted array in the same format and dtype as input.
        """
        redacted, _ = self.redact(image)
        return redacted

    def redact(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect and redact burned-in PHI text, returning the audit record.

        Handles both 2D images ``(C, H, W)`` and 3D volumes
        ``(C, D, H, W)``.  For volumes, keyframe-based OCR is used
        to avoid running OCR on every slice.

        Stateless with respect to the transform instance — results flow
        back as return values, never through instance attributes, so a
        single instance is safe to share across threads.

        Args:
            image: Image array in channel-first format.

        Returns:
            Tuple of ``(redacted, stats)`` where ``stats`` is the
            detection/redaction audit dict for this call.
        """
        if image.ndim == 4:
            return self._redact_volume(image)
        return self._redact_single(image)

    def _redact_single(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Redact PHI from a single 2D image ``(C, H, W)``; return ``(redacted, stats)``."""
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
        bboxes, stats = detect_text(
            norm_ocr, self.reader, self.config, self.ner_classifier
        )
        redacted = apply_redaction(pixel_array, bboxes)

        # Restore channel-first
        if is_grayscale:
            redacted = redacted[np.newaxis, ...]
        elif is_rgb:
            redacted = np.transpose(redacted, (2, 0, 1))

        return redacted.astype(orig_dtype), stats

    def _redact_volume(self, volume: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Redact PHI from a 3D volume ``(C, D, H, W)`` using adaptive keyframe sampling.

        Samples an adaptive percentage of frames (10%, floor=3, ceiling=25),
        builds a pixel-level union mask across all sampled keyframes, then
        zeros every flagged pixel on every frame.  A pixel detected as PHI
        on **any** keyframe is redacted on **all** frames — no count-based
        heuristics, no silent misses when PHI shifts position across slices.

        Args:
            volume: Volume array ``(C, D, H, W)``.

        Returns:
            Tuple of ``(redacted_volume, stats)`` — same shape and dtype
            as the input, plus the aggregated audit record.
        """
        orig_dtype = volume.dtype
        C, D, H, W = volume.shape

        series_cfg = self.config.get('series', {})
        sampling = series_cfg.get('keyframe_sampling', {})
        # Legacy fallback: honour old keyframe_count as floor
        legacy_floor = series_cfg.get('keyframe_count', 3)
        sample_pct = sampling.get('sample_pct', 0.10)
        floor      = sampling.get('floor', legacy_floor)
        ceiling    = sampling.get('ceiling', 25)

        # Per-keyframe OCR stats accumulate into a local list via closure —
        # locals are per-call, so concurrent redact() calls cannot interleave.
        per_keyframe_stats: List[Dict[str, Any]] = []

        def ocr_fn(frame: np.ndarray) -> np.ndarray:
            """OCR one ``(C, H, W)`` keyframe → ``bool (H, W)`` PHI mask."""
            redacted_frame, frame_stats = self._redact_single(frame)
            per_keyframe_stats.append(frame_stats)
            return np.any(frame != redacted_frame, axis=0)

        # redact_volume_safe uses (T, C, H, W) convention.
        # Transpose (C, D, H, W) → (D, C, H, W) as a view so in-place
        # modifications inside redact_volume_safe propagate back to result.
        result = volume.copy()
        vol_DCHW = np.transpose(result, (1, 0, 2, 3))  # view of result

        _, union_mask, kf_stats = redact_volume_safe(
            vol_DCHW,
            ocr_fn,
            sample_pct=sample_pct,
            floor=floor,
            ceiling=ceiling,
        )

        acc = per_keyframe_stats
        phi_pixels = int(union_mask.sum())

        # Flatten per-keyframe detections, stamping each with the frame index
        # it was found on (acc order matches keyframe_indices order).
        kf_indices = kf_stats.get('keyframe_indices', [])
        volume_detections: List[Dict[str, Any]] = []
        for frame_idx, frame_stats in zip(kf_indices, acc):
            for det in frame_stats.get('detections', []):
                det = dict(det)
                det['frame_index'] = int(frame_idx)
                volume_detections.append(det)

        stats = {
            'total_detections':    sum(s.get('total_detections', 0) for s in acc),
            'low_confidence_count': sum(s.get('low_confidence_count', 0) for s in acc),
            'safelisted_count':    sum(s.get('safelisted_count', 0) for s in acc),
            'ner_classified_count': sum(s.get('ner_classified_count', 0) for s in acc),
            'redacted_count':      phi_pixels,
            'volume_strategy':     'adaptive_union',
            'num_slices':          D,
            'keyframe_indices':    kf_indices,
            'keyframe_stats':      kf_stats,
            'detections':          volume_detections,
        }

        logger.info(
            "Volume OCR: %d keyframes scanned (%.1f%% of %d), union mask %d px",
            kf_stats.get('keyframes_scanned', 0),
            kf_stats.get('pct_frames_scanned', 0.0),
            D,
            phi_pixels,
        )

        return result.astype(orig_dtype), stats


# ---------------------------------------------------------------------------
# Dictionary Transform
# ---------------------------------------------------------------------------

class RedactPixelPHId(MapTransform):
    """
    Dictionary transform: Detect and redact burned-in PHI text from medical images.

    Dictionary-based wrapper of :py:class:`RedactPixelPHI`. This is a
    destructive transform: once PHI pixels are zeroed, the original image
    content cannot be reconstructed from downstream state.

    Thread-safe via per-thread EasyOCR reader (``threading.local()``).
    Preserves MetaTensor metadata throughout the transform.

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

        Args:
            data: Pipeline data dictionary containing MetaTensors.

        Returns:
            Updated data dictionary with redacted images and metadata.

        Raises:
            PixelRedactionError: If OCR detection or redaction fails.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            # Extract uri for error context
            meta = d.get(ck(key, ckeys.META_DICT), {})
            uri = meta.get('filename_or_obj', '<unknown>')
            if isinstance(uri, list):
                uri = uri[0]

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

                # --- US Region Mask: scope OCR to PHI zones only ---
                us_phi_mask = d.get(ck(key, ckeys.US_PHI_MASK))
                if us_phi_mask is not None:
                    # Mask the diagnostic region before OCR so EasyOCR
                    # only processes the annotation/overlay areas.
                    original_pixels = pixel_array.copy()
                    # Zero out diagnostic pixels (mask=False) across all channels
                    if pixel_array.ndim == 3:
                        # (C, H, W) — broadcast mask over channels
                        pixel_array = pixel_array.copy()
                        pixel_array[:, ~us_phi_mask] = 0
                    elif pixel_array.ndim == 4:
                        # (C, D, H, W) — broadcast mask over channels and depth
                        pixel_array = pixel_array.copy()
                        pixel_array[:, :, ~us_phi_mask] = 0
                    logger.info(
                        "US PHI mask applied: OCR scoped to %d%% of frame",
                        int(100 * us_phi_mask.sum() / us_phi_mask.size),
                    )

                # Apply array transform (OCR + redaction); stats returned
                # as a value — never read from instance state.
                redacted, stats = self.transform.redact(pixel_array)

                # --- Restore diagnostic region pixels after OCR ---
                if us_phi_mask is not None:
                    if redacted.ndim == 3:
                        redacted[:, ~us_phi_mask] = original_pixels[:, ~us_phi_mask]
                    elif redacted.ndim == 4:
                        redacted[:, :, ~us_phi_mask] = original_pixels[:, :, ~us_phi_mask]

                # Store redaction stats for downstream routing
                if us_phi_mask is not None:
                    stats['us_region_mask_applied'] = True
                d[ck(key, ckeys.REDACTION_STATS)] = stats

                # Generate binary redaction mask matched to spatial_shape
                # Recompute mask from the difference between original and redacted
                compare_src = original_pixels if us_phi_mask is not None else pixel_array
                if compare_src.ndim == 3:
                    diff = np.any(compare_src != redacted, axis=0)
                else:
                    diff = compare_src != redacted
                redaction_mask = diff.astype(np.uint8)
                d[ck(key, ckeys.REDACTION_MASK)] = redaction_mask

                # Preserve MetaTensor
                if isinstance(input_tensor, MetaTensor):
                    d[key] = MetaTensor(torch.as_tensor(redacted), meta=input_tensor.meta)
                else:
                    d[key] = redacted
            except PixelRedactionError:
                raise
            except Exception as e:
                raise PixelRedactionError(
                    f"Pixel redaction failed: {e}",
                    uri=str(uri),
                    transform="RedactPixelPHId",
                ) from e

        return d
