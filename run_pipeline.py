"""
Aegis De-identification Pipeline — CLI Entry Point

Usage::

    # Single-file mode (default — existing behavior)
    PYTHONPATH=monai_aegis python run_pipeline.py --config monai_aegis/config/config.yaml

    # Series-aware mode (volume processing)
    PYTHONPATH=monai_aegis python run_pipeline.py --config monai_aegis/config/config.yaml --mode series
"""
import os
import sys
import argparse
import shutil
import logging

from transforms.pipeline import build_pipeline, build_series_pipeline
from transforms.discovery import discover_dicoms, group_into_series, validate_series, sort_slices
import pydicom
from PIL import Image
import numpy as np
from config.config_loader import load_config
from config.storage import AegisFileSystem

logger = logging.getLogger(__name__)





# -----------------------------------------------------------------------
# Single-file mode (existing)
# -----------------------------------------------------------------------

def run_single_pipeline(config_path: str) -> None:
    """Run the single-file de-identification pipeline.

    Processes each file in ``input_dir`` independently.
    Images with low OCR confidence are routed to ``not_processed_dir``.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    logger.info("Building single-file pipeline with output_dir=%s...", output_dir)
    pipeline = build_pipeline(config_path=config_path, output_dir=output_dir)
    logger.info("Pipeline ready.")

    processed_count = 0
    not_processed_count = 0
    errors = 0

    files = sorted(os.listdir(input_dir))
    for filename in files:
        file_path = os.path.join(input_dir, filename)

        if not (filename.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png'))):
            continue

        logger.info("Processing: %s", filename)

        try:
            data = {'image': file_path}
            result = pipeline(data)

            # Check redaction stats — route low-confidence images
            stats = result.get('image_redaction_stats', {})
            total = stats.get('total_detections', 0)
            low_conf = stats.get('low_confidence_count', 0)

            if low_conf > 0:
                not_processed_path = os.path.join(not_processed_dir, filename)
                shutil.copy2(file_path, not_processed_path)
                logger.warning(
                    "NOT PROCESSED: %s — %d/%d text regions below confidence threshold. "
                    "Copied to %s", filename, low_conf, total, not_processed_path
                )
                not_processed_count += 1
                continue

            # DICOMs handled by SaveDicomd
            if filename.lower().endswith('.dcm'):
                out_path = os.path.join(output_dir, filename)
                if os.path.exists(out_path):
                    logger.info("Saved DICOM to %s", out_path)
                    processed_count += 1
                else:
                    logger.warning("Output DICOM not found for %s", filename)
            else:
                # Images need manual saving
                img_array = result['image']

                if hasattr(img_array, 'cpu'):
                    img_array = img_array.cpu().numpy()
                elif not isinstance(img_array, np.ndarray):
                    img_array = np.array(img_array)

                if img_array.ndim == 3:
                    if img_array.shape[0] == 1:
                        img_array = img_array.squeeze(0)
                    elif img_array.shape[0] == 3:
                        img_array = np.moveaxis(img_array, 0, -1)

                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    if img_array.max() <= 1.1:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                elif img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)

                img = Image.fromarray(img_array)
                out_path = os.path.join(output_dir, filename)
                img.save(out_path)
                logger.info("Saved Image to %s", out_path)
                processed_count += 1

        except Exception as e:
            logger.error("Error processing %s: %s", filename, e, exc_info=True)
            errors += 1

    logger.info(
        "Processing complete. Processed: %d | Not processed: %d | Errors: %d",
        processed_count, not_processed_count, errors,
    )
    if not_processed_count > 0:
        logger.warning(
            "%d file(s) in %s/ need manual review.",
            not_processed_count, not_processed_dir,
        )


# -----------------------------------------------------------------------
# Series mode (new)
# -----------------------------------------------------------------------

def run_series_pipeline(config_path: str) -> None:
    """Run the series-aware de-identification pipeline.

    Discovers DICOM files, groups into studies/series, validates
    geometry, sorts slices, and processes each series as a
    ``(C, D, H, W)`` volume.

    Non-DICOM files in ``input_dir`` are processed with the
    single-file pipeline as a fallback.
    """
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    # Storage backend
    fs = AegisFileSystem.from_config(config)

    # --- Step 1: Discover DICOM files ---
    logger.info("Discovering DICOM files in %s...", input_dir)
    slices = discover_dicoms(input_dir, fs=fs)

    if not slices:
        logger.warning("No DICOM files found. Falling back to single-file mode.")
        run_single_pipeline(config_path)
        return

    # --- Step 2: Group into series ---
    series_groups = group_into_series(slices)

    # --- Step 3: Build series pipeline ---
    logger.info("Building series pipeline with output_dir=%s...", output_dir)
    pipeline = build_series_pipeline(config_path=config_path, output_dir=output_dir, input_dir=input_dir)
    logger.info("Series pipeline ready.")

    series_processed = 0
    total_slices = 0
    errors = 0

    for (study_uid, series_uid), series_slices in series_groups.items():
        # --- Step 4: Validate geometry and split if needed ---
        sub_series_list = validate_series(series_slices)

        for sub_idx, sub_series in enumerate(sub_series_list):
            # --- Step 5: Sort slices ---
            sorted_series = sort_slices(sub_series)
            filepaths = [s.filepath for s in sorted_series]

            series_label = f"{series_uid}"
            if len(sub_series_list) > 1:
                series_label += f" (sub-{sub_idx})"

            logger.info(
                "Processing series %s: %d slices (Study: %s)",
                series_label, len(filepaths), study_uid[:8],
            )

            try:
                data = {'image': filepaths}
                result = pipeline(data)

                # Check redaction stats
                stats = result.get('image_redaction_stats', {})
                strategy = stats.get('volume_strategy', 'unknown')
                logger.info(
                    "Series %s complete: strategy=%s, redacted=%d",
                    series_label, strategy, stats.get('redacted_count', 0),
                )

                series_processed += 1
                total_slices += len(filepaths)

            except Exception as e:
                logger.error(
                    "Error processing series %s: %s", series_label, e,
                    exc_info=True,
                )
                errors += 1

    # --- Fallback: process non-DICOM files with single-file pipeline ---
    non_dicom_files = []
    for root, _dirs, fnames in os.walk(input_dir):
        for f in fnames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                non_dicom_files.append(os.path.join(root, f))
    non_dicom_files.sort()

    if non_dicom_files:
        logger.info(
            "Processing %d non-DICOM files with single-file pipeline...",
            len(non_dicom_files),
        )
        single_pipeline = build_pipeline(config_path=config_path, output_dir=output_dir)
        for file_path in non_dicom_files:
            # Preserve subdirectory structure in the output
            rel_path = os.path.relpath(file_path, input_dir)
            filename = os.path.basename(file_path)
            try:
                result = single_pipeline({'image': file_path})

                stats = result.get('image_redaction_stats', {})
                low_conf = stats.get('low_confidence_count', 0)
                if low_conf > 0:
                    np_dest = os.path.join(not_processed_dir, rel_path)
                    os.makedirs(os.path.dirname(np_dest), exist_ok=True)
                    shutil.copy2(file_path, np_dest)
                    logger.warning("NOT PROCESSED: %s — low confidence", rel_path)
                    continue

                img_array = result['image']
                if hasattr(img_array, 'cpu'):
                    img_array = img_array.cpu().numpy()
                elif not isinstance(img_array, np.ndarray):
                    img_array = np.array(img_array)

                if img_array.ndim == 3:
                    if img_array.shape[0] == 1:
                        img_array = img_array.squeeze(0)
                    elif img_array.shape[0] == 3:
                        img_array = np.moveaxis(img_array, 0, -1)

                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.1:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)

                out_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(img_array).save(out_path)
                logger.info("Saved Image to %s", rel_path)
                series_processed += 1

            except Exception as e:
                logger.error("Error processing %s: %s", rel_path, e, exc_info=True)
                errors += 1

    logger.info(
        "Series processing complete. "
        "Series: %d | Total slices: %d | Errors: %d",
        series_processed, total_slices, errors,
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

    parser = argparse.ArgumentParser(description="Run Aegis De-identification Pipeline")
    parser.add_argument(
        "--config",
        default="monai_aegis/config/config.yaml",
        help="Path to config.yaml (default: monai_aegis/config/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "series"],
        default="single",
        help="Pipeline mode: 'single' (per-file) or 'series' (volume-aware). Default: single.",
    )

    args = parser.parse_args()

    if args.mode == "series":
        run_series_pipeline(args.config)
    else:
        run_single_pipeline(args.config)
