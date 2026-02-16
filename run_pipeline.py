import os
import sys
import argparse
import shutil
import logging
import yaml

from transforms.pipeline import build_pipeline
import pydicom
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(config_path):
    """
    Runs the Aegis pipeline using paths from config.yaml.

    Images where OCR detects text but confidence is too low
    are routed to staging_not_processed/ for manual review.
    """
    # Load config
    config = load_config(config_path)
    paths = config.get('paths', {})
    input_dir = paths.get('input_dir', 'staging_input')
    output_dir = paths.get('output_dir', 'staging_output')
    not_processed_dir = paths.get('not_processed_dir', 'staging_not_processed')

    # Resolve config to absolute path
    config_path = os.path.abspath(config_path)

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)

    # Initialize pipeline
    logger.info("Building pipeline with output_dir=%s...", output_dir)

    pipeline = build_pipeline(config_path=config_path, output_dir=output_dir)

    logger.info("Pipeline ready.")

    processed_count = 0
    not_processed_count = 0
    errors = 0

    files = sorted(os.listdir(input_dir))
    for filename in files:
        file_path = os.path.join(input_dir, filename)

        # Simple filter
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
            redacted = stats.get('redacted_count', 0)

            # If OCR detected text that fell below confidence threshold,
            # the image may contain unredacted PHI → route for manual review
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

                # Check for MetaTensor and convert
                if hasattr(img_array, 'cpu'):
                    img_array = img_array.cpu().numpy()
                elif not isinstance(img_array, np.ndarray):
                    img_array = np.array(img_array)

                # Handle dimensions: (C, H, W) -> (H, W, C)
                if img_array.ndim == 3:
                    if img_array.shape[0] == 1:
                        img_array = img_array.squeeze(0)
                    elif img_array.shape[0] == 3:
                        img_array = np.moveaxis(img_array, 0, -1)

                # Scale float32 (0-1) to uint8 (0-255) range
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

    logger.info("Processing complete. Processed: %d | Not processed: %d | Errors: %d",
                processed_count, not_processed_count, errors)

    if not_processed_count > 0:
        logger.warning("%d file(s) in %s/ need manual review.",
                       not_processed_count, not_processed_dir)


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
        help="Path to config.yaml (default: monai_aegis/config/config.yaml)"
    )

    args = parser.parse_args()

    run_pipeline(args.config)
