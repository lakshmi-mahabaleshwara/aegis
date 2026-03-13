import torch
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
import traceback

def simple_collate(batch):
    return batch

from datetime import datetime

import monai
from monai.data import Dataset, DataLoader, list_data_collate

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

import traceback

def simple_collate(batch):
    return batch


class SafeTransformPipeline:
    """Module-level transform wrapper for PyTorch multiprocessing pickling."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def __call__(self, data):
        try:
            return self.pipeline(data)
        except Exception as e:
            logger.error("Error in transform pipeline: %s", getattr(e, 'filepath', data.get('image')), exc_info=True)
            data['error'] = str(e)
            data['error_trace'] = traceback.format_exc()
            return data

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

    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = os.path.join(base_output_dir, dicom_folder, timestamp_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)
    fs = AegisFileSystem.from_config(config)

    pipeline_input_dir = os.path.join(input_dir, dicom_folder)

    logger.info("Building single-file DICOM pipeline → %s", output_dir)
    pipeline = build_pipeline(config_path=config_path, output_dir=output_dir, input_dir=pipeline_input_dir)
    logger.info("Pipeline ready.")

    processed = 0
    not_processed = 0
    errors = 0

    from transforms.discovery import discover_dicoms
    slices = discover_dicoms(pipeline_input_dir, fs=fs)
    data_list = [{'image': s.uri} for s in slices]

    if not data_list:
        logger.warning("No DICOM files found in %s", input_dir)
        return

    logger.info("Created dataset with %d items", len(data_list))

    # Wrap the pipeline in a Dataset
    # We use a custom wrapper to gracefully catch exceptions during DataLoader worker execution


    dataset = Dataset(data=data_list, transform=SafeTransformPipeline(pipeline))
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=0, 
        collate_fn=simple_collate
    )

    for batch in dataloader:
        for data_dict in batch:
            if 'error' in data_dict:
                errors += 1
                file_path = data_dict.get('file_path', data_dict.get('image', 'unknown'))
                if isinstance(file_path, str) and file_path != 'unknown':
                    rel_path = os.path.relpath(file_path, pipeline_input_dir)
                    dest = os.path.join(not_processed_dir, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    try:
                        shutil.copy2(file_path, dest)
                        logger.warning("NOT PROCESSED (ERROR): %s → %s", rel_path, dest)
                    except Exception:
                        pass
                continue

            file_path = data_dict.get('file_path', data_dict.get('image', 'unknown'))
            if not isinstance(file_path, str) and hasattr(file_path, 'meta'):
                file_path = file_path.meta.get('filename_or_obj', 'unknown')
            filename = os.path.basename(file_path)
            rel_path = os.path.relpath(file_path, pipeline_input_dir)

            stats_dict = data_dict.get('image_redaction_stats', {})
            low_conf = stats_dict.get('low_confidence_count', 0)

            if low_conf > 0:
                dest = os.path.join(not_processed_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(file_path, dest)
                logger.warning(
                    "NOT PROCESSED: %s — %d low-confidence regions → %s",
                    rel_path, low_conf, dest,
                )
                not_processed += 1
                
                # Manual file deletion fallback for saved output 
                out_path = data_dict.get('image_saved_path', os.path.join(output_dir, filename))
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            out_path = data_dict.get('image_saved_path', os.path.join(output_dir, filename))
            if os.path.exists(out_path):
                logger.info("Saved DICOM → %s", out_path)
                processed += 1
            else:
                logger.warning("Output DICOM not found for %s", filename)

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

    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = os.path.join(base_output_dir, dicom_folder, timestamp_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    fs = AegisFileSystem.from_config(config)
    pipeline_input_dir = os.path.join(input_dir, dicom_folder)

    # --- Discover ---
    logger.info("Discovering DICOM files in %s...", pipeline_input_dir)
    slices = discover_dicoms(pipeline_input_dir, fs=fs)

    if not slices:
        logger.warning("No DICOM files found. Falling back to single-file mode.")
        run_single(config_path)
        return

    # --- Group ---
    series_groups = group_into_series(slices)

    # --- Build pipeline ---
    logger.info("Building series pipeline → %s", output_dir)
    pipeline = build_series_pipeline(
        config_path=config_path, output_dir=output_dir, input_dir=pipeline_input_dir,
    )
    pipeline_single = build_pipeline(
        config_path=config_path, output_dir=output_dir, input_dir=pipeline_input_dir
    )
    logger.info("Series pipelines ready.")

    singletons_data = []
    series_data = []

    for (study_uid, series_uid), series_slices in series_groups.items():
        sub_series_list = validate_series(series_slices)

        for sub_idx, sub_series in enumerate(sub_series_list):
            sorted_series = sort_slices(sub_series)
            uris = [s.uri for s in sorted_series]

            label = series_uid
            if len(sub_series_list) > 1:
                label += f" (sub-{sub_idx})"

            if len(uris) == 1:
                singletons_data.append({
                    'image': uris[0], 
                    'label': label, 
                    'filename': os.path.basename(uris[0])
                })
            else:
                series_data.append({
                    'image': uris, 
                    'label': label, 
                    'num_slices': len(uris),
                    'study_uid': study_uid[:8]
                })

    series_count = 0
    total_slices = 0
    errors = 0



    # 1. Process Singletons DataList via PipelineSingle in a DataLoader
    if singletons_data:
        logger.info("Processing %d singletons discovered in series groupings via DataLoader", len(singletons_data))
        ds_single = Dataset(data=singletons_data, transform=SafeTransformPipeline(pipeline_single))
        dl_single = DataLoader(ds_single, batch_size=1, num_workers=min(4, os.cpu_count() or 1), collate_fn=simple_collate)
        
        for batch in dl_single:
            for data_dict in batch:
                if 'error' in data_dict:
                    errors += 1
                    file_path = data_dict.get('image', 'unknown')
                    if isinstance(file_path, str) and file_path != 'unknown':
                        rel_path = os.path.relpath(file_path, pipeline_input_dir)
                        dest = os.path.join(not_processed_dir, rel_path)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        try:
                            shutil.copy2(file_path, dest)
                            logger.warning("NOT PROCESSED (ERROR): %s → %s", rel_path, dest)
                        except Exception:
                            pass
                    continue

                file_path = data_dict.get('image', 'unknown')
                if not isinstance(file_path, str) and hasattr(file_path, 'meta'):
                    file_path = file_path.meta.get('filename_or_obj', 'unknown')
                filename = data_dict.get('filename', 'unknown')
                rel_path = os.path.relpath(file_path, pipeline_input_dir)

                stats_dict = data_dict.get('image_redaction_stats', {})
                low_conf = stats_dict.get('low_confidence_count', 0)

                if low_conf > 0:
                    dest = os.path.join(not_processed_dir, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(file_path, dest)
                    logger.warning("NOT PROCESSED: %s — %d low-confidence regions → %s", rel_path, low_conf, dest)
                    # Manual file deletion fallback for saved output 
                    out_path = data_dict.get('image_saved_path', os.path.join(output_dir, rel_path))
                    if os.path.exists(out_path): os.remove(out_path)
                else:
                    total_slices += 1

    # 2. Process Volume Series DataList via Pipeline in a DataLoader
    if series_data:
        logger.info("Processing %d series volumes via DataLoader", len(series_data))
        ds_series = Dataset(data=series_data, transform=SafeTransformPipeline(pipeline))
        dl_series = DataLoader(ds_series, batch_size=1, num_workers=min(4, os.cpu_count() or 1), collate_fn=simple_collate)
        
        for batch in dl_series:
            for data_dict in batch:
                label = data_dict.get('label', 'unknown')
                if 'error' in data_dict:
                    logger.error("Error processing series %s", label)
                    errors += 1
                    image_list = data_dict.get('image', [])
                    if isinstance(image_list, list):
                        for file_path in image_list:
                            if isinstance(file_path, str):
                                rel_path = os.path.relpath(file_path, pipeline_input_dir)
                                dest = os.path.join(not_processed_dir, rel_path)
                                os.makedirs(os.path.dirname(dest), exist_ok=True)
                                try:
                                    shutil.copy2(file_path, dest)
                                except Exception:
                                    pass
                        logger.warning("NOT PROCESSED (ERROR): Series %s → %s", label, not_processed_dir)
                    continue
                
                stats_dict = data_dict.get('image_redaction_stats', {})
                strategy = stats_dict.get('volume_strategy', 'unknown')
                target_token = data_dict.get('image_target_token')
                out_msg = f"Token: {target_token}" if target_token else "Original structure"
                num_slices = data_dict.get('num_slices', 0)
                
                logger.info(
                    "Series %s complete: strategy=%s, redacted=%d | Output: %s",
                    label, strategy, stats_dict.get('redacted_count', 0), out_msg
                )

                series_count += 1
                total_slices += num_slices


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
