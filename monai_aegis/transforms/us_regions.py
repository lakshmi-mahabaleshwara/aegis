"""
MONAI Aegis US Region Transforms — RedactByUSRegions / RedactByUSRegionsd

Ultrasound fan-geometry aware pre-processing for scoped OCR redaction.
Reads SequenceOfUltrasoundRegions (0018,6011) from the cached pydicom.Dataset,
builds a binary PHI-zone mask marking pixels *outside* the diagnostic region(s),
and writes it to the data dict for downstream consumption by RedactPixelPHId.

Thread-safe: stateless, no disk I/O. Operates on the in-memory cached Dataset.
"""
import numpy as np
import pydicom
import logging
from typing import Dict, Hashable, Mapping, Any, Optional, Tuple

from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform

__all__ = ["RedactByUSRegions", "RedactByUSRegionsd"]

logger = logging.getLogger(__name__)

# DICOM tag for SequenceOfUltrasoundRegions
_US_REGIONS_TAG = (0x0018, 0x6011)


def _extract_region_bounds(region_item: pydicom.Dataset) -> Optional[Tuple[int, int, int, int]]:
    """Extract pixel bounds from a single US region sequence item.

    Args:
        region_item: A single item from SequenceOfUltrasoundRegions.

    Returns:
        Tuple of (x0, y0, x1, y1) pixel bounds, or None if bounds are missing.
    """
    try:
        x0 = int(region_item.RegionLocationMinX0)
        y0 = int(region_item.RegionLocationMinY0)
        x1 = int(region_item.RegionLocationMaxX1)
        y1 = int(region_item.RegionLocationMaxY1)
        return (x0, y0, x1, y1)
    except AttributeError:
        logger.warning(
            "US region item missing one or more location attributes, skipping"
        )
        return None


def build_us_phi_mask(
    dataset: pydicom.Dataset,
    image_height: int,
    image_width: int,
) -> Optional[np.ndarray]:
    """Build a binary PHI-zone mask from SequenceOfUltrasoundRegions.

    The mask is True for pixels *outside* all diagnostic regions (i.e. the
    annotation/overlay areas where burnt-in PHI typically resides) and False
    for pixels *inside* the diagnostic fan — which should not be OCR'd.

    Args:
        dataset: Cached pydicom.Dataset with potential (0018,6011) tag.
        image_height: Height of the pixel array.
        image_width: Width of the pixel array.

    Returns:
        Boolean ndarray of shape (H, W), or None if the tag is absent or
        the modality is not US.
    """
    modality = getattr(dataset, "Modality", "")
    if modality != "US":
        return None

    if _US_REGIONS_TAG not in dataset:
        logger.debug("US modality but no SequenceOfUltrasoundRegions tag found")
        return None

    regions_seq = dataset[_US_REGIONS_TAG].value
    if not regions_seq:
        return None

    # Start with all-True mask (everything is PHI zone)
    mask = np.ones((image_height, image_width), dtype=bool)

    region_count = 0
    for item in regions_seq:
        bounds = _extract_region_bounds(item)
        if bounds is None:
            continue

        x0, y0, x1, y1 = bounds

        # Clamp to image boundaries
        x0 = max(0, min(x0, image_width))
        y0 = max(0, min(y0, image_height))
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))

        if x1 <= x0 or y1 <= y0:
            logger.warning(
                "US region has zero or negative area after clamping: "
                "(%d, %d, %d, %d), skipping", x0, y0, x1, y1
            )
            continue

        # Mark the diagnostic region as False (NOT a PHI zone)
        mask[y0:y1, x0:x1] = False
        region_count += 1

    if region_count == 0:
        logger.warning("SequenceOfUltrasoundRegions present but no valid regions extracted")
        return None

    logger.info(
        "Built US PHI mask from %d region(s): diagnostic area = %d%% of frame",
        region_count,
        int(100 * (1 - mask.sum() / mask.size)),
    )
    return mask


# ---------------------------------------------------------------------------
# Array Transform
# ---------------------------------------------------------------------------

class RedactByUSRegions(Transform):
    """Array transform: Build a PHI-zone mask from DICOM US region metadata.

    For US-modality DICOMs with a SequenceOfUltrasoundRegions (0018,6011)
    tag, this transform uses the scanner-reported pixel boundaries of the
    diagnostic region(s) to build a binary mask. Pixels outside the
    diagnostic fan are flagged as potential PHI zones.

    For non-US modalities or when the tag is absent, this is a no-op.

    Args:
        config: Configuration dict with optional 'us_regions' section.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        us_cfg = config.get("us_regions", {})
        self.enabled = us_cfg.get("enabled", True)

    def __call__(
        self,
        dataset: pydicom.Dataset,
        image_height: int,
        image_width: int,
    ) -> Optional[np.ndarray]:
        """Build the US PHI-zone mask.

        Args:
            dataset: Cached pydicom.Dataset.
            image_height: Height of the image in pixels.
            image_width: Width of the image in pixels.

        Returns:
            Boolean ndarray (H, W) where True = PHI zone, or None if N/A.
        """
        if not self.enabled:
            return None
        return build_us_phi_mask(dataset, image_height, image_width)


# ---------------------------------------------------------------------------
# Dictionary Transform
# ---------------------------------------------------------------------------

class RedactByUSRegionsd(MapTransform):
    """Dictionary transform: Extract US fan geometry and write PHI-zone mask.

    Reads the cached ``{key}_dicom_dataset`` from the data dict (written by
    :py:class:`LoadDicomRawd`), extracts SequenceOfUltrasoundRegions bounds,
    and writes ``{key}_us_phi_mask`` for downstream use by
    :py:class:`RedactPixelPHId`.

    For non-US modalities or when ``(0018,6011)`` is absent, the transform
    is a **no-op** — no mask is written, and ``RedactPixelPHId`` falls back
    to full-frame OCR.

    Thread-safe, stateless, zero disk I/O.

    Args:
        keys: Keys of the data dictionary to process.
        config: Configuration dict with optional 'us_regions' section.
        allow_missing_keys: If True, skip missing keys instead of raising.
    """

    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = RedactByUSRegions(config=config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Extract US region mask for each keyed image in the data dict.

        Side-effects written per key (US modality with (0018,6011) only):
            - ``{key}_us_phi_mask`` — boolean ``(H, W)`` array where True =
              outside diagnostic region (PHI zone).

        Args:
            data: Pipeline data dictionary containing cached Datasets.

        Returns:
            Updated data dictionary (unchanged for non-US inputs).
        """
        d = dict(data)
        for key in self.key_iterator(d):
            # Read cached Dataset (written by LoadDicomRawd / LoadDicomSeriesd)
            dataset = d.get(f"{key}_dicom_dataset")
            if dataset is None:
                # Series mode: check the list of datasets
                datasets_list = d.get(f"{key}_dicom_datasets")
                if datasets_list and isinstance(datasets_list, list):
                    dataset = datasets_list[0]

            if dataset is None:
                continue

            # Get image dimensions from the MetaTensor
            tensor = d.get(key)
            if tensor is None:
                continue

            if hasattr(tensor, 'shape'):
                shape = tensor.shape
                if len(shape) == 3:
                    # (C, H, W)
                    image_height, image_width = int(shape[1]), int(shape[2])
                elif len(shape) == 4:
                    # (C, D, H, W) — series/volume
                    image_height, image_width = int(shape[2]), int(shape[3])
                else:
                    continue
            else:
                continue

            mask = self.transform(dataset, image_height, image_width)
            if mask is not None:
                d[f"{key}_us_phi_mask"] = mask
                logger.info(
                    "US PHI mask written to data dict for key '%s' (%dx%d)",
                    key, image_width, image_height,
                )

        return d
