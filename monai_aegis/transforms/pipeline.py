"""
MONAI Aegis Pipeline Builder

Composes the de-identification transforms into MONAI Compose pipelines.

- ``build_pipeline()`` — single-file mode (existing).
- ``build_series_pipeline()`` — series-aware volume mode (new).
"""
import os
import yaml
import torch
import logging
from typing import Optional
from monai.transforms import Compose

from transforms.io import LoadDicomRawd, SaveDicomd
from transforms.series_io import LoadDicomSeriesd, SaveDicomSeriesd
from transforms.pixel import RedactPixelPHId
from transforms.metadata import ScrubDicomMetadatad

logger = logging.getLogger(__name__)


def build_pipeline(config_path: str = '../config/config.yaml', output_dir: str = './output') -> Compose:
    """
    Build the Aegis de-identification pipeline.

    Loads configuration once and passes it to all transforms.

    Thread safety:
        - Steps 1-3 are thread-safe → can use ``DataLoader(num_workers > 0)``
        - Step 4 (SaveDicomd) is ``ThreadUnsafe`` → file I/O isolated at the end

    Args:
        config_path: Path to config.yaml.
        output_dir: Directory for de-identified output files.

    Returns:
        A MONAI Compose pipeline ready for ``pipeline({"image": filepath})``.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    # Load config ONCE
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device Detection
    is_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    logger.info(f"Device set to: {'GPU' if is_gpu else 'CPU'}")

    keys = ['image']

    return Compose([
        # ==========================================
        # INGESTION ZONE: Single Disk Read
        # ==========================================
        # Load file and cache pydicom.Dataset in-memory
        LoadDicomRawd(keys=keys),

        # ==========================================
        # LOGIC ZONE: Pure In-Memory Transforms
        # ==========================================
        # Visual Redaction (EasyOCR + safelist/NER)
        RedactPixelPHId(keys=keys, config=config),

        # Logical Redaction (DICOM metadata scrub)
        # Uses cached dataset, zero disk access
        ScrubDicomMetadatad(keys=keys, config=config),

        # ==========================================
        # PERSISTENCE ZONE: Single Disk Write
        # ==========================================
        # Save scrubbed DICOM to disk
        SaveDicomd(keys=keys, output_dir=output_dir),
    ])


def build_series_pipeline(
    config_path: str = '../config/config.yaml',
    output_dir: str = './output',
    input_dir: str = '',
) -> Compose:
    """Build the Aegis series-aware de-identification pipeline.

    Processes DICOM series as ``(C, D, H, W)`` volumes with keyframe OCR.

    Pipeline architecture::

        LoadDicomSeriesd   →  volume (C,D,H,W)
        RedactPixelPHId    →  keyframe OCR + pixel redaction
        ScrubDicomMetadatad →  per-slice metadata scrub
        SaveDicomSeriesd   →  output (preserves original paths)

    Args:
        config_path: Path to config.yaml.
        output_dir: Directory for de-identified output files.
        input_dir: Root input directory (for preserving folder structure).

    Returns:
        A MONAI Compose pipeline ready for
        ``pipeline({"image": [path1, path2, ...]})``.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    is_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    logger.info(f"Device set to: {'GPU' if is_gpu else 'CPU'}")

    keys = ['image']

    return Compose([
        # Ingestion: Load series as volume (C, D, H, W)
        LoadDicomSeriesd(keys=keys),

        # Logic: Keyframe OCR + pixel redaction
        RedactPixelPHId(keys=keys, config=config),

        # Logic: Per-slice metadata scrub with geometry preservation
        ScrubDicomMetadatad(keys=keys, config=config),

        # Persistence: Write de-identified series (preserving folder/filenames)
        SaveDicomSeriesd(keys=keys, output_dir=output_dir, input_dir=input_dir),
    ])
