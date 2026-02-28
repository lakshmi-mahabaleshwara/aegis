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
"""
from transforms.io import LoadDicomRaw, LoadDicomRawd, SaveDicom, SaveDicomd
from transforms.pixel import RedactPixelPHI, RedactPixelPHId
from transforms.metadata import ScrubDicomMetadata, ScrubDicomMetadatad
from transforms.utility import AegisIdentityManager
from transforms.ner_classifier import PHIClassifier
from transforms.pipeline import build_pipeline
from transforms.exceptions import (
    AegisTransformError,
    DicomLoadError,
    ImageLoadError,
    PixelRedactionError,
    MetadataScrubError,
    DicomSaveError,
)

__all__ = [
    # I/O
    "LoadDicomRaw",
    "LoadDicomRawd",
    "SaveDicom",
    "SaveDicomd",
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
    # Exceptions
    "AegisTransformError",
    "DicomLoadError",
    "ImageLoadError",
    "PixelRedactionError",
    "MetadataScrubError",
    "DicomSaveError",
]
