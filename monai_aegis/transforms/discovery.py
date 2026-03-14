"""
MONAI Aegis DICOM Discovery — discover, group, validate, sort

Pure functions for scanning a dataset folder, grouping DICOM files
into studies and series, validating slice geometry, and sorting slices
for correct volume assembly.

These are **not** MONAI Transforms — they are utility functions called
by ``run_pipeline.py`` before the pipeline is invoked.

Raises:
    SeriesLoadError: When a series has no valid imaging slices or
        critical metadata is missing.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from monai_aegis.config.storage import AegisFileSystem

import pydicom

from monai_aegis.transforms.exceptions import SeriesLoadError

__all__ = [
    "DicomSliceInfo", "discover_dicoms", "group_into_series",
    "validate_series", "sort_slices", "discover_images",
]

logger = logging.getLogger(__name__)

# Modalities that contain pixel data suitable for de-identification
ACCEPTED_MODALITIES = frozenset([
    'CT', 'MR', 'US', 'CR', 'DX', 'MG', 'PT', 'XA', 'RF', 'OT',
])

# Series grouping key: (StudyInstanceUID, SeriesInstanceUID)
SeriesKey = Tuple[str, str]


@dataclass
class DicomSliceInfo:
    """Lightweight metadata for a single DICOM slice.

    Extracted with ``stop_before_pixels=True`` during discovery
    so pixel data is never loaded at this stage.

    Attributes:
        uri: Absolute path to the ``.dcm`` file.
        sop_instance_uid: Unique identifier for this slice.
        study_instance_uid: Study-level grouping identifier.
        series_instance_uid: Series-level grouping identifier.
        instance_number: Slice ordering number (may be ``None``).
        image_position_patient: ``[x, y, z]`` patient coordinates
            (may be ``None`` for non-cross-sectional modalities).
        modality: DICOM modality code (e.g. ``'CT'``, ``'MR'``).
        rows: Pixel rows in this slice.
        columns: Pixel columns in this slice.
        pixel_spacing: ``[row_spacing, col_spacing]`` in mm.
        slice_thickness: Thickness in mm (may be ``None``).
        image_orientation_patient: Direction cosines (6-element list).
        number_of_frames: If > 1, this is a multi-frame DICOM.
    """
    uri: str
    sop_instance_uid: str
    study_instance_uid: str
    series_instance_uid: str
    instance_number: Optional[int]
    image_position_patient: Optional[List[float]]
    modality: str
    rows: int
    columns: int
    pixel_spacing: Optional[List[float]] = None
    slice_thickness: Optional[float] = None
    image_orientation_patient: Optional[List[float]] = None
    number_of_frames: Optional[int] = None


# -----------------------------------------------------------------------
# Step 1 — Discover all DICOM files
# -----------------------------------------------------------------------

def discover_dicoms(
    folder: str,
    accepted_modalities: Optional[frozenset] = None,
    fs: Optional['AegisFileSystem'] = None,
) -> List[DicomSliceInfo]:
    """Recursively scan *folder* for valid imaging DICOM files.

    Reads each ``.dcm`` file with ``stop_before_pixels=True`` to
    avoid loading pixel data.  Non-imaging modalities (SR, RTSTRUCT,
    etc.) are skipped.

    Args:
        folder: Root directory to scan.
        accepted_modalities: Set of modality codes to accept.
            Defaults to :py:data:`ACCEPTED_MODALITIES`.

    Returns:
        List of :py:class:`DicomSliceInfo` for every valid slice found.

    Raises:
        SeriesLoadError: If *folder* does not exist or is not a directory.
    """
    if fs is not None:
        if not fs.isdir(folder):
            raise SeriesLoadError(
                f"Input folder does not exist: {folder}",
                uri=folder,
                transform="discover_dicoms",
            )
    else:
        if not os.path.isdir(folder):
            raise SeriesLoadError(
                f"Input folder does not exist: {folder}",
                uri=folder,
                transform="discover_dicoms",
            )

    if accepted_modalities is None:
        accepted_modalities = ACCEPTED_MODALITIES

    slices: List[DicomSliceInfo] = []

    walker = fs.walk(folder) if fs is not None else os.walk(folder)
    for root, _dirs, files in walker:
        for fname in files:
            if not fname.lower().endswith('.dcm'):
                continue
            _fs = fs if fs is not None else AegisFileSystem()
            fpath = _fs.join(root, fname)
            try:
                with _fs.open_read(fpath) as f:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception as e:
                logger.warning("Skipping unreadable DICOM %s: %s", fpath, e)
                continue

            modality = getattr(ds, 'Modality', '').upper()
            if modality not in accepted_modalities:
                logger.debug("Skipping non-imaging modality %s: %s", modality, fpath)
                continue

            # Extract optional positioning metadata
            ipp = None
            if hasattr(ds, 'ImagePositionPatient'):
                try:
                    ipp = [float(v) for v in ds.ImagePositionPatient]
                except (TypeError, ValueError):
                    pass

            iop = None
            if hasattr(ds, 'ImageOrientationPatient'):
                try:
                    iop = [float(v) for v in ds.ImageOrientationPatient]
                except (TypeError, ValueError):
                    pass

            ps = None
            if hasattr(ds, 'PixelSpacing'):
                try:
                    ps = [float(v) for v in ds.PixelSpacing]
                except (TypeError, ValueError):
                    pass

            slices.append(DicomSliceInfo(
                uri=fpath,
                sop_instance_uid=getattr(ds, 'SOPInstanceUID', ''),
                study_instance_uid=getattr(ds, 'StudyInstanceUID', ''),
                series_instance_uid=getattr(ds, 'SeriesInstanceUID', ''),
                instance_number=getattr(ds, 'InstanceNumber', None),
                image_position_patient=ipp,
                modality=modality,
                rows=getattr(ds, 'Rows', 0),
                columns=getattr(ds, 'Columns', 0),
                pixel_spacing=ps,
                slice_thickness=getattr(ds, 'SliceThickness', None),
                image_orientation_patient=iop,
                number_of_frames=getattr(ds, 'NumberOfFrames', None),
            ))

    logger.info("Discovered %d imaging DICOM slices in %s", len(slices), folder)
    return slices


def discover_images(
    folder: str,
    accepted_extensions: Optional[frozenset] = None,
    fs: Optional['AegisFileSystem'] = None,
) -> List[str]:
    """Recursively scan *folder* for standard imaging files (JPEGs/PNGs).

    Args:
        folder: Root directory to scan.
        accepted_extensions: Set of extensions to accept (e.g., {'.jpg', '.png'}).
            Defaults to {'.jpg', '.jpeg', '.png'}.

    Returns:
        Sorted list of absolute file paths found.

    Raises:
        SeriesLoadError: If *folder* does not exist or is not a directory.
    """
    if fs is not None:
        if not fs.isdir(folder):
            raise SeriesLoadError(
                f"Input folder does not exist: {folder}",
                uri=folder,
                transform="discover_images",
            )
    else:
        if not os.path.isdir(folder):
            raise SeriesLoadError(
                f"Input folder does not exist: {folder}",
                uri=folder,
                transform="discover_images",
            )

    if accepted_extensions is None:
        accepted_extensions = frozenset(['.jpg', '.jpeg', '.png'])

    images: List[str] = []

    walker = fs.walk(folder) if fs is not None else os.walk(folder)
    for root, _dirs, files in walker:
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in accepted_extensions:
                continue
                
            if fs is not None:
                fpath = fs.join(root, fname)
            else:
                fpath = os.path.join(root, fname)
            
            images.append(fpath)

    # Sort alphabetically by path as there is no spatial metadata.
    images.sort()

    logger.info("Discovered %d standard images in %s", len(images), folder)
    return images



# -----------------------------------------------------------------------
# Step 2 — Group into series
# -----------------------------------------------------------------------

def group_into_series(
    slices: List[DicomSliceInfo],
) -> Dict[SeriesKey, List[DicomSliceInfo]]:
    """Group slices by ``(StudyInstanceUID, SeriesInstanceUID)``.

    Using both UIDs prevents collisions when a PACS export
    reuses SeriesInstanceUID across different studies.

    Args:
        slices: Flat list of discovered slices.

    Returns:
        Dict mapping ``SeriesKey`` → list of slices for that series.
    """
    groups: Dict[SeriesKey, List[DicomSliceInfo]] = {}
    for s in slices:
        key: SeriesKey = (s.study_instance_uid, s.series_instance_uid)
        groups.setdefault(key, []).append(s)

    logger.info("Grouped into %d series", len(groups))
    return groups


# -----------------------------------------------------------------------
# Step 3 — Validate series geometry
# -----------------------------------------------------------------------

def _geometry_signature(s: DicomSliceInfo) -> tuple:
    """Return a hashable geometry tuple for sub-series splitting."""
    return (
        s.rows,
        s.columns,
        tuple(s.pixel_spacing) if s.pixel_spacing else None,
        s.slice_thickness,
        tuple(s.image_orientation_patient) if s.image_orientation_patient else None,
    )


def validate_series(
    series: List[DicomSliceInfo],
) -> List[List[DicomSliceInfo]]:
    """Validate geometry consistency and split if necessary.

    Slices with mismatched ``Rows``, ``Columns``, ``PixelSpacing``,
    ``SliceThickness``, or ``ImageOrientationPatient`` are split into
    separate sub-series (e.g., localizer vs. diagnostic slices).

    Args:
        series: All slices for a single series.

    Returns:
        List of sub-series (often just one), each geometrically consistent.
    """
    if not series:
        return []

    sub_groups: Dict[tuple, List[DicomSliceInfo]] = {}
    for s in series:
        sig = _geometry_signature(s)
        sub_groups.setdefault(sig, []).append(s)

    if len(sub_groups) > 1:
        logger.warning(
            "Series %s has %d geometry groups — splitting into sub-series",
            series[0].series_instance_uid,
            len(sub_groups),
        )

    return list(sub_groups.values())


# -----------------------------------------------------------------------
# Step 4 — Sort slices
# -----------------------------------------------------------------------

def sort_slices(series: List[DicomSliceInfo]) -> List[DicomSliceInfo]:
    """Sort slices in anatomically correct order.

    Priority:
      1. ``ImagePositionPatient[2]`` (z-coordinate) — the gold standard
         for cross-sectional imaging (CT, MR).
      2. ``InstanceNumber`` — fallback for modalities without spatial
         positioning (US, CR).

    Args:
        series: Slices belonging to one (sub-)series.

    Returns:
        Sorted copy of the input list.
    """
    # Prefer ImagePositionPatient if available on all slices
    if all(s.image_position_patient is not None for s in series):
        return sorted(series, key=lambda s: s.image_position_patient[2])

    # Fallback to InstanceNumber
    if all(s.instance_number is not None for s in series):
        return sorted(series, key=lambda s: s.instance_number)

    # Last resort — filename order
    logger.warning(
        "Series %s lacks positioning and instance numbers — using filename order",
        series[0].series_instance_uid if series else '?',
    )
    return sorted(series, key=lambda s: s.uri)
