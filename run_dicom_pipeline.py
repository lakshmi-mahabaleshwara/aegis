"""
Aegis DICOM De-identification Pipeline

Processes DICOM files: single-file mode or series-aware volume mode.

Usage::

    # Single DICOM file mode
    PYTHONPATH=monai_aegis python run_dicom_pipeline.py --config monai_aegis/config/config.yaml

    # Series-aware volume mode (recommended)
    PYTHONPATH=monai_aegis python run_dicom_pipeline.py --config monai_aegis/config/config.yaml --mode series
"""
import os
import sys
import argparse
import shutil
import logging
from datetime import datetime

from transforms.pipeline import build_pipeline, build_series_pipeline
from transforms.discovery import (
    discover_dicoms, group_into_series, validate_series, sort_slices,
)
from config.config_loader import load_config
from config.storage import AegisFileSystem

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Single-file DICOM mode
# -----------------------------------------------------------------------

def run_single(config_path: str) -> None:
    """Process each DICOM file in ``input_dir`` independently.

    Files with low OCR confidence are routed to ``not_processed_dir``.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    base_output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')
    dicom_folder = paths.get('dicom_folder', 'dicom')

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(base_output_dir, dicom_folder, today_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    logger.info("Building single-file DICOM pipeline → %s", output_dir)
    pipeline = build_pipeline(config_path=config_path, output_dir=output_dir)
    logger.info("Pipeline ready.")

    processed = 0
    not_processed = 0
    errors = 0

    files = sorted(os.listdir(input_dir))
    for filename in files:
        if not filename.lower().endswith('.dcm'):
            continue

        file_path = os.path.join(input_dir, filename)
        logger.info("Processing: %s", filename)

        try:
            result = pipeline({'image': file_path})

            stats = result.get('image_redaction_stats', {})
            low_conf = stats.get('low_confidence_count', 0)

            if low_conf > 0:
                dest = os.path.join(not_processed_dir, filename)
                shutil.copy2(file_path, dest)
                logger.warning(
                    "NOT PROCESSED: %s — %d low-confidence regions → %s",
                    filename, low_conf, dest,
                )
                not_processed += 1
                continue

            out_path = os.path.join(output_dir, filename)
            if os.path.exists(out_path):
                logger.info("Saved DICOM → %s", out_path)
                processed += 1
            else:
                logger.warning("Output DICOM not found for %s", filename)

        except Exception as e:
            logger.error("Error processing %s: %s", filename, e, exc_info=True)
            errors += 1

    logger.info(
        "Single-file DICOM complete. Processed: %d | Not processed: %d | Errors: %d",
        processed, not_processed, errors,
    )


# -----------------------------------------------------------------------
# Series-aware DICOM mode
# -----------------------------------------------------------------------

def run_series(config_path: str) -> None:
    """Discover, group, validate, sort, and process DICOM series as volumes.

    Falls back to single-file mode if no series are found.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    base_output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')
    dicom_folder = paths.get('dicom_folder', 'dicom')

    today_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(base_output_dir, dicom_folder, today_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    fs = AegisFileSystem.from_config(config)

    # --- Discover ---
    logger.info("Discovering DICOM files in %s...", input_dir)
    slices = discover_dicoms(input_dir, fs=fs)

    if not slices:
        logger.warning("No DICOM files found. Falling back to single-file mode.")
        run_single(config_path)
        return

    # --- Group ---
    series_groups = group_into_series(slices)

    # --- Build pipeline ---
    logger.info("Building series pipeline → %s", output_dir)
    pipeline = build_series_pipeline(
        config_path=config_path, output_dir=output_dir, input_dir=input_dir,
    )
    pipeline_single = build_pipeline(
        config_path=config_path, output_dir=output_dir,
    )
    logger.info("Series pipelines ready.")

    series_count = 0
    total_slices = 0
    errors = 0

    for (study_uid, series_uid), series_slices in series_groups.items():
        sub_series_list = validate_series(series_slices)

        for sub_idx, sub_series in enumerate(sub_series_list):
            sorted_series = sort_slices(sub_series)
            uris = [s.uri for s in sorted_series]

            label = series_uid
            if len(sub_series_list) > 1:
                label += f" (sub-{sub_idx})"

            logger.info(
                "Processing series %s: %d slices (Study: %s)",
                label, len(uris), study_uid[:8],
            )

            if len(uris) == 1:
                filename = os.path.basename(uris[0])
                logger.info("Routing singleton DICOM %s to single-file mode.", filename)
                try:
                    result = pipeline_single({'image': uris[0]})
                    stats = result.get('image_redaction_stats', {})
                    if stats.get('low_confidence_count', 0) > 0:
                        dest = os.path.join(not_processed_dir, filename)
                        shutil.copy2(uris[0], dest)
                    total_slices += 1
                except Exception as e:
                    logger.error("Error processing series %s: %s", label, e, exc_info=True)
                    errors += 1
                continue

            try:
                result = pipeline({'image': uris})

                stats = result.get('image_redaction_stats', {})
                strategy = stats.get('volume_strategy', 'unknown')
                target_token = result.get('image_target_token')
                
                out_msg = f"Token: {target_token}" if target_token else "Original structure"
                
                logger.info(
                    "Series %s complete: strategy=%s, redacted=%d | Output: %s",
                    label, strategy, stats.get('redacted_count', 0), out_msg
                )

                series_count += 1
                total_slices += len(uris)

            except Exception as e:
                logger.error(
                    "Error processing series %s: %s", label, e, exc_info=True,
                )
                errors += 1

    logger.info(
        "DICOM series processing complete. "
        "Series: %d | Slices: %d | Errors: %d",
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
        description="Aegis DICOM De-identification Pipeline",
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
