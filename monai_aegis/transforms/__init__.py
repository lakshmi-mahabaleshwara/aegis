"""
MONAI Aegis — Medical Image De-identification Transforms

Public API for the monai_aegis package.

Example::

    from monai_aegis.transforms import RedactPixelPHId, ScrubDicomMetadatad, build_pipeline

    # Use individual transforms:
    pipeline = Compose([
        LoadDicomRawd(keys=["image"]),
        RedactPixelPHId(keys=["image"], config=config),
        ScrubDicomMetadatad(keys=["image"], config=config),
        SaveDicomd(keys=["image"], output_dir="./output"),
    ])

    # Or use the convenience builder:
    pipeline = build_pipeline(config_path="config/config.yaml", output_dir="./output")

    # Series-aware pipeline:
    from monai_aegis.transforms import build_series_pipeline, discover_dicoms
    series_pipeline = build_series_pipeline(config_path="config/config.yaml", output_dir="./output")
"""
from transforms.io import LoadDicomRaw, LoadDicomRawd, SaveDicom, SaveDicomd
from transforms.series_io import LoadDicomSeries, LoadDicomSeriesd, SaveDicomSeries, SaveDicomSeriesd
from transforms.pixel import RedactPixelPHI, RedactPixelPHId
from transforms.metadata import ScrubDicomMetadata, ScrubDicomMetadatad
from transforms.utility import AegisIdentityManager
from transforms.ner_classifier import PHIClassifier
from transforms.pipeline import build_pipeline, build_series_pipeline
from transforms.discovery import (
    DicomSliceInfo,
    discover_dicoms,
    group_into_series,
    validate_series,
    sort_slices,
    ACCEPTED_MODALITIES,
)
from transforms.exceptions import (
    AegisTransformError,
    DicomLoadError,
    ImageLoadError,
    PixelRedactionError,
    MetadataScrubError,
    DicomSaveError,
    SeriesLoadError,
    SeriesSaveError,
)

__all__ = [
    # I/O (single-file)
    "LoadDicomRaw",
    "LoadDicomRawd",
    "SaveDicom",
    "SaveDicomd",
    # I/O (series)
    "LoadDicomSeries",
    "LoadDicomSeriesd",
    "SaveDicomSeries",
    "SaveDicomSeriesd",
    # Pixel Redaction
    "RedactPixelPHI",
    "RedactPixelPHId",
    # Metadata Scrubbing
    "ScrubDicomMetadata",
    "ScrubDicomMetadatad",
    # NER
    "PHIClassifier",
    # Utilities
    "AegisIdentityManager",
    # Pipeline
    "build_pipeline",
    "build_series_pipeline",
    # Discovery
    "DicomSliceInfo",
    "discover_dicoms",
    "group_into_series",
    "validate_series",
    "sort_slices",
    "ACCEPTED_MODALITIES",
    # Exceptions
    "AegisTransformError",
    "DicomLoadError",
    "ImageLoadError",
    "PixelRedactionError",
    "MetadataScrubError",
    "DicomSaveError",
    "SeriesLoadError",
    "SeriesSaveError",
]
