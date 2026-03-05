"""
Aegis Image De-identification Pipeline

Processes standard images (JPEG/PNG) with pixel-level PHI redaction.
No DICOM metadata scrubbing — images have no DICOM tags.

Usage::

    PYTHONPATH=monai_aegis python run_image_pipeline.py --config monai_aegis/config/config.yaml
"""
import os
import sys
import argparse
import shutil
import logging

from transforms.pipeline import build_image_pipeline
from config.config_loader import load_config
from config.storage import AegisFileSystem

logger = logging.getLogger(__name__)


def run_image_pipeline(config_path: str) -> None:
    """Process all JPEG/PNG images in ``input_dir``.

    Files with low OCR confidence are routed to ``not_processed_dir``.
    Output images are saved as PNG (lossless) by default.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    fs = AegisFileSystem.from_config(config)

    # Discover image files
    image_files = []
    walker = fs.walk(input_dir) if fs.protocol != 'file' else os.walk(input_dir)
    for root, _dirs, fnames in walker:
        for f in fnames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = fs.join(root, f) if fs.protocol != 'file' else os.path.join(root, f)
                image_files.append(fpath)
    image_files.sort()

    if not image_files:
        logger.warning("No image files found in %s", input_dir)
        return

    logger.info("Found %d image files in %s", len(image_files), input_dir)

    # Build pipeline
    pipeline = build_image_pipeline(config_path=config_path, output_dir=output_dir)
    logger.info("Image pipeline ready.")

    processed = 0
    not_processed = 0
    errors = 0

    for file_path in image_files:
        rel_path = os.path.relpath(file_path, input_dir)
        logger.info("Processing: %s", rel_path)

        try:
            result = pipeline({'image': file_path})

            stats = result.get('image_redaction_stats', {})
            low_conf = stats.get('low_confidence_count', 0)

            if low_conf > 0:
                dest = os.path.join(not_processed_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(file_path, dest)
                logger.warning("NOT PROCESSED: %s — low confidence → %s", rel_path, dest)
                not_processed += 1
                continue

            logger.info("Processed: %s", rel_path)
            processed += 1

        except Exception as e:
            logger.error("Error processing %s: %s", rel_path, e, exc_info=True)
            errors += 1

    logger.info(
        "Image processing complete. Processed: %d | Not processed: %d | Errors: %d",
        processed, not_processed, errors,
    )


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Aegis Image De-identification Pipeline",
    )
    parser.add_argument(
        "--config",
        default="monai_aegis/config/config.yaml",
        help="Path to config.yaml",
    )

    args = parser.parse_args()
    run_image_pipeline(args.config)
