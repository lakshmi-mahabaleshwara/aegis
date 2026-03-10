"""
Aegis Image De-identification Pipeline

Processes standard images (JPEG/PNG) with pixel-level PHI redaction.
No DICOM metadata scrubbing — images have no DICOM tags.

Usage::

    # Single-file image mode
    PYTHONPATH=monai_aegis python run_image_pipeline.py --config monai_aegis/config/config.yaml --mode single
    
    # Series-aware volume mode for identical standard images (recommended)
    PYTHONPATH=monai_aegis python run_image_pipeline.py --config monai_aegis/config/config.yaml --mode series
"""
import os
import sys
import argparse
import shutil
import logging
from datetime import datetime
from collections import defaultdict

from transforms.pipeline import build_image_pipeline, build_image_series_pipeline
from transforms.discovery import discover_images
from config.config_loader import load_config
from config.storage import AegisFileSystem

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Single-file Image mode
# -----------------------------------------------------------------------

def run_single(config_path: str) -> None:
    """Process all JPEG/PNG images in ``input_dir``.

    Files with low OCR confidence are routed to ``not_processed_dir``.
    Output images are saved as PNG (lossless) by default.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    base_output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')
    image_folder = paths.get('image_folder', 'image')

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(base_output_dir, image_folder, today_str)

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
        "Single-file image processing complete. Processed: %d | Not processed: %d | Errors: %d",
        processed, not_processed, errors,
    )


# -----------------------------------------------------------------------
# Series-aware Image mode
# -----------------------------------------------------------------------

def run_series(config_path: str) -> None:
    """Discover standard images, group by parent directory, and process as volumes.
    
    If images have different dimensions within the same folder, LoadImageSeriesd 
    will throw an error. In production, this can be caught to fallback to single-file processing.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    base_output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')
    image_folder = paths.get('image_folder', 'image')

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(base_output_dir, image_folder, today_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    fs = AegisFileSystem.from_config(config)

    # --- Discover ---
    logger.info("Discovering standard images in %s...", input_dir)
    image_files = discover_images(input_dir, fs=fs)

    if not image_files:
        logger.warning("No image files found. Falling back to single-file mode.")
        run_single(config_path)
        return

    # --- Group by immediate parent directory ---
    folder_groups = defaultdict(list)
    for f in image_files:
        if fs.protocol != 'file':
            parent_dir = fs.dirname(f)
        else:
            parent_dir = os.path.dirname(f)
        folder_groups[parent_dir].append(f)
        
    logger.info("Grouped into %d folder series", len(folder_groups))

    # --- Build pipeline ---
    logger.info("Building image series pipeline → %s", output_dir)
    pipeline = build_image_series_pipeline(
        config_path=config_path, output_dir=output_dir, output_ext='.png'
    )
    pipeline_single = build_image_pipeline(
        config_path=config_path, output_dir=output_dir, output_ext='.png'
    )
    logger.info("Image series pipelines ready.")

    series_count = 0
    total_slices = 0
    errors = 0

    for parent_dir, folder_images in folder_groups.items():
        # folder_images is already alphabetically sorted by discover_images
        label = os.path.basename(parent_dir) if fs.protocol == 'file' else fs.basename(parent_dir)
        
        logger.info(
            "Processing folder %s: %d images",
            label, len(folder_images),
        )

        if len(folder_images) == 1:
            try:
                logger.info("Routing singleton image to single-file mode.")
                result = pipeline_single({'image': folder_images[0]})
                
                stats = result.get('image_redaction_stats', {})
                low_conf = stats.get('low_confidence_count', 0)
                if low_conf > 0:
                    if fs.protocol != 'file':
                        rel_path = folder_images[0][len(input_dir):].lstrip('/')
                    else:
                        rel_path = os.path.relpath(folder_images[0], input_dir)
                    dest = os.path.join(not_processed_dir, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(folder_images[0], dest)
                    
                total_slices += 1
            except Exception as e:
                logger.error("Error processing single image %s: %s", label, e, exc_info=True)
                errors += 1
            continue

        try:
            result = pipeline({'image': folder_images})

            stats = result.get('image_redaction_stats', {})
            strategy = stats.get('volume_strategy', 'unknown')
            logger.info(
                "Folder %s complete: strategy=%s, redacted=%d",
                label, strategy, stats.get('redacted_count', 0),
            )

            series_count += 1
            total_slices += len(folder_images)

        except Exception as e:
            logger.error(
                "Error processing folder %s: %s", label, e, exc_info=True,
            )
            errors += 1

    logger.info(
        "Image series processing complete. "
        "Folders: %d | Images: %d | Errors: %d",
        series_count, total_slices, errors,
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
    parser.add_argument(
        "--mode",
        choices=["single", "series"],
        default="series",
        help="'single' (per-file) or 'series' (volume-aware, default).",
    )

    args = parser.parse_args()
    
    if args.mode == "series":
        run_series(args.config)
    else:
        run_single(args.config)
