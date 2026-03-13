"""
MONAI Aegis Pipeline Builder

Composes the de-identification transforms into MONAI Compose pipelines.

- ``build_pipeline()`` — single DICOM file mode.
- ``build_series_pipeline()`` — series-aware DICOM volume mode.
- ``build_image_pipeline()`` — standard image mode (JPEG/PNG).
- ``build_image_series_pipeline()`` — standard image volume mode (JPEG/PNG series).
"""
import os
import torch
import logging
from typing import Optional
from monai.transforms import Compose

from transforms.io import LoadDicomRawd, SaveDicomd, LoadImaged, SaveImaged
from transforms.series_io import LoadDicomSeriesd, SaveDicomSeriesd
from transforms.image_series_io import LoadImageSeriesd, SaveImageSeriesd
from config.config_loader import load_config
from config.storage import AegisFileSystem
from transforms.pixel import RedactPixelPHId
from transforms.metadata import ScrubDicomMetadatad

logger = logging.getLogger(__name__)


def build_pipeline(config_path: str = '../config/config.yaml', output_dir: str = './output', input_dir: str = '') -> Compose:
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
        A MONAI Compose pipeline ready for ``pipeline({"image": uri})``.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    # Load config ONCE (with env-var resolution)
    config = load_config(config_path)

    # Device Detection
    is_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    logger.info(f"Device set to: {'GPU' if is_gpu else 'CPU'}")

    # Storage backend (default: local filesystem)
    fs = AegisFileSystem.from_config(config)

    keys = ['image']

    return Compose([
        # ==========================================
        # INGESTION ZONE: Single Disk Read
        # ==========================================
        # Load file and cache pydicom.Dataset in-memory
        LoadDicomRawd(keys=keys, config=config, input_dir=input_dir, fs=fs),

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
        SaveDicomd(keys=keys, output_dir=output_dir, input_dir=input_dir, fs=fs),
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

    config = load_config(config_path)

    is_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    logger.info(f"Device set to: {'GPU' if is_gpu else 'CPU'}")

    # Storage backend
    fs = AegisFileSystem.from_config(config)

    keys = ['image']

    return Compose([
        # Ingestion: Load series as volume (C, D, H, W)
        LoadDicomSeriesd(keys=keys, config=config, fs=fs),

        # Logic: Keyframe OCR + pixel redaction
        RedactPixelPHId(keys=keys, config=config),

        # Logic: Per-slice metadata scrub with geometry preservation
        ScrubDicomMetadatad(keys=keys, config=config),

        # Persistence: Write de-identified series (preserving folder/filenames)
        SaveDicomSeriesd(keys=keys, output_dir=output_dir, input_dir=input_dir, fs=fs),
    ])


def build_image_pipeline(
    config_path: str = '../config/config.yaml',
    output_dir: str = './output',
    input_dir: str = '',
    output_ext: str = '.png',
) -> Compose:
    """Build the Aegis image-only de-identification pipeline.

    For standard images (JPEG/PNG) that need pixel redaction but have
    no DICOM metadata to scrub.

    Pipeline architecture::

        LoadImaged      →  pixel array (C, H, W)
        RedactPixelPHId →  OCR + pixel redaction (shared with DICOM)
        SaveImaged      →  output (PNG by default)

    Args:
        config_path: Path to config.yaml.
        output_dir: Directory for de-identified output files.
        output_ext: Extension for output images (default ``'.png'``).

    Returns:
        A MONAI Compose pipeline ready for ``pipeline({"image": uri})``.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    config = load_config(config_path)

    fs = AegisFileSystem.from_config(config)

    keys = ['image']

    return Compose([
        # Load: JPEG/PNG → channel-first MetaTensor
        LoadImaged(keys=keys, config=config, input_dir=input_dir, fs=fs),

        # Redact: shared pixel-level PHI detection (OCR + NER)
        RedactPixelPHId(keys=keys, config=config),

        # Save: channel-first → HWC → output image
        SaveImaged(keys=keys, output_dir=output_dir, input_dir=input_dir, output_ext=output_ext, fs=fs),
    ])


def build_image_series_pipeline(
    config_path: str = '../config/config.yaml',
    output_dir: str = './output',
    input_dir: str = '',
    output_ext: str = '.png',
) -> Compose:
    """Build the pipeline for a folder of standard images (JPEGs/PNGs).

    1. Load a folder of uniform images into a (C, D, H, W) volume.
    2. Detect/redact PHI via Keyframe Strategy.
    3. Save (D) slices to disk under the folder identity token.

    Args:
        config_path: Path to the YAML configuration file.
        output_dir: Base directory for de-identified output.
        output_ext: Extension for saved files.

    Returns:
        A composed MONAI transform pipeline.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    config = load_config(config_path)

    fs = AegisFileSystem.from_config(config)

    keys = ['image']

    return Compose([
        # Ingestion: Load a standard image folder into a volume (C, D, H, W)
        LoadImageSeriesd(keys=keys, config=config, input_dir=input_dir, fs=fs),

        # Logic: Keyframe OCR + pixel redaction
        RedactPixelPHId(keys=keys, config=config),

        # Output: Slice volume back into Token folder / Original Name
        SaveImageSeriesd(keys=keys, output_dir=output_dir, input_dir=input_dir, output_ext=output_ext, fs=fs),
    ])
