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
import argparse
import shutil
import logging
import traceback

from datetime import datetime
from collections import defaultdict

from monai.data import Dataset, DataLoader

from monai_aegis.transforms.pipeline import build_image_pipeline, build_image_series_pipeline
from monai_aegis.transforms.discovery import discover_images
from monai_aegis.config.config_loader import load_config
from monai_aegis.config.storage import AegisFileSystem

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Single-file Image mode
# -----------------------------------------------------------------------

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

    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = os.path.join(base_output_dir, image_folder, timestamp_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    fs = AegisFileSystem.from_config(config)

    pipeline_input_dir = os.path.join(input_dir, image_folder)

    # Discover image files
    image_files = []
    walker = fs.walk(pipeline_input_dir) if fs.protocol != 'file' else os.walk(pipeline_input_dir)
    for root, _dirs, fnames in walker:
        for f in fnames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = fs.join(root, f) if fs.protocol != 'file' else os.path.join(root, f)
                image_files.append(fpath)
    image_files.sort()

    if not image_files:
        logger.warning("No image files found in %s", pipeline_input_dir)
        return

    logger.info("Found %d image files in %s", len(image_files), pipeline_input_dir)

    pipeline = build_image_pipeline(config_path=config_path, output_dir=output_dir, input_dir=pipeline_input_dir)
    logger.info("Image pipeline ready.")

    processed = 0
    not_processed = 0
    errors = 0

    data_list = []
    for file_path in image_files:
        rel_path = os.path.relpath(file_path, input_dir)
        data_list.append({'image': file_path, 'file_path': file_path, 'rel_path': rel_path})

    logger.info("Created dataset with %d items", len(data_list))



    dataset = Dataset(data=data_list, transform=SafeTransformPipeline(pipeline))
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=min(4, os.cpu_count() or 1), 
        collate_fn=simple_collate
    )

    for batch in dataloader:
        for data_dict in batch:
            if 'error' in data_dict:
                errors += 1
                rel_path = data_dict.get('rel_path', 'unknown')
                file_path = data_dict.get('file_path', rel_path)
                if isinstance(file_path, str) and file_path != 'unknown':
                    dest = os.path.join(not_processed_dir, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    try:
                        shutil.copy2(file_path, dest)
                        logger.warning("NOT PROCESSED (ERROR): %s → %s", rel_path, dest)
                    except Exception:
                        pass
                continue

            rel_path = data_dict.get('rel_path', 'unknown')
            file_path = data_dict.get('file_path', rel_path) # Added fallback

            stats_dict = data_dict.get('image_redaction_stats', {})
            low_conf = stats_dict.get('low_confidence_count', 0)

            if low_conf > 0:
                dest = os.path.join(not_processed_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(file_path, dest)
                logger.warning("NOT PROCESSED: %s — %d low-confidence regions → %s", rel_path, low_conf, dest)
                not_processed += 1
                
                # Cleanup saved output from pipeline out
                out_path = data_dict.get('image_saved_path')
                if not out_path:
                    filename = os.path.basename(rel_path)
                    out_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.png')
                if os.path.exists(out_path): os.remove(out_path)
                continue

            logger.info("Processed: %s", rel_path)
            processed += 1

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

    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = os.path.join(base_output_dir, image_folder, timestamp_str)

    config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    fs = AegisFileSystem.from_config(config)

    pipeline_input_dir = os.path.join(input_dir, image_folder)

    # --- Discover ---
    logger.info("Discovering standard images in %s...", pipeline_input_dir)
    image_files = discover_images(pipeline_input_dir, fs=fs)

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
        config_path=config_path, output_dir=output_dir, input_dir=pipeline_input_dir, output_ext='.png'
    )
    pipeline_single = build_image_pipeline(
        config_path=config_path, output_dir=output_dir, input_dir=pipeline_input_dir, output_ext='.png'
    )
    logger.info("Image series pipelines ready.")

    series_count = 0
    total_slices = 0
    errors = 0

    singletons_data = []
    series_data = []

    for parent_dir, folder_images in folder_groups.items():
        label = os.path.basename(parent_dir) if fs.protocol == 'file' else fs.basename(parent_dir)
        
        if len(folder_images) == 1:
            if fs.protocol != 'file':
                rel_path = folder_images[0][len(pipeline_input_dir):].lstrip('/')
            else:
                rel_path = os.path.relpath(folder_images[0], pipeline_input_dir)
            singletons_data.append({
                'image': folder_images[0], 
                'file_path': folder_images[0], 
                'label': label, 
                'rel_path': rel_path
            })
        else:
            series_data.append({
                'image': folder_images, 
                'label': label, 
                'num_slices': len(folder_images)
            })

    series_count = 0
    total_slices = 0
    errors = 0



    if singletons_data:
        logger.info("Processing %d singletons discovered in folders via DataLoader", len(singletons_data))
        ds_single = Dataset(data=singletons_data, transform=SafeTransformPipeline(pipeline_single))
        dl_single = DataLoader(ds_single, batch_size=1, num_workers=min(4, os.cpu_count() or 1), collate_fn=simple_collate)
        
        for batch in dl_single:
            for data_dict in batch:
                if 'error' in data_dict:
                    errors += 1
                    file_path = data_dict.get('file_path', data_dict.get('image', 'unknown'))
                    rel_path = data_dict.get('rel_path', 'unknown')
                    if isinstance(file_path, str) and file_path != 'unknown':
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
                rel_path = data_dict.get('rel_path', 'unknown')

                stats_dict = data_dict.get('image_redaction_stats', {})
                low_conf = stats_dict.get('low_confidence_count', 0)

                if low_conf > 0:
                    dest = os.path.join(not_processed_dir, rel_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(file_path, dest)
                    logger.warning("NOT PROCESSED: %s — %d low-confidence regions → %s", rel_path, low_conf, dest)
                    
                    out_path = data_dict.get('image_saved_path')
                    if not out_path:
                        filename = os.path.basename(rel_path)
                        out_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.png')
                    if os.path.exists(out_path): os.remove(out_path)
                else:
                    total_slices += 1

    if series_data:
        logger.info("Processing %d folder sequences via DataLoader", len(series_data))
        ds_series = Dataset(data=series_data, transform=SafeTransformPipeline(pipeline))
        dl_series = DataLoader(ds_series, batch_size=1, num_workers=min(4, os.cpu_count() or 1), collate_fn=simple_collate)
        
        for batch in dl_series:
            for data_dict in batch:
                label = data_dict.get('label', 'unknown')
                if 'error' in data_dict:
                    logger.error("Error processing folder %s", label)
                    errors += 1
                    image_list = data_dict.get('image', [])
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
                num_slices = data_dict.get('num_slices', 0)
                
                logger.info(
                    "Folder %s complete: strategy=%s, redacted=%d",
                    label, strategy, stats_dict.get('redacted_count', 0)
                )

                series_count += 1
                total_slices += num_slices

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
