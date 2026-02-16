"""
MONAI Aegis Pipeline Builder

Composes the de-identification transforms into a single MONAI Compose pipeline.
"""
import os
import yaml
import torch
import logging
from typing import Optional
from monai.transforms import Compose

from transforms.io import LoadDicomRawd, SaveDicomd
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
        # 1. Load without spatial transforms → MetaTensor  (thread-safe)
        LoadDicomRawd(keys=keys),

        # 2. Visual Redaction (EasyOCR + safelist)          (thread-safe)
        RedactPixelPHId(keys=keys, config=config),

        # 3. Logical Redaction (DICOM metadata scrub)       (thread-safe)
        ScrubDicomMetadatad(keys=keys, config=config),

        # 4. Save scrubbed DICOM to disk                    (ThreadUnsafe — I/O)
        SaveDicomd(keys=keys, output_dir=output_dir),
    ])
