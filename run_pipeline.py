"""
Aegis De-identification Pipeline — Unified Orchestrator

Routes files to the appropriate pipeline by format:

- ``.dcm`` → DICOM pipeline (single or series mode)
- ``.jpg``, ``.jpeg``, ``.png`` → Image pipeline

Can also run each pipeline independently::

    PYTHONPATH=monai_aegis python run_dicom_pipeline.py --config config.yaml --mode series
    PYTHONPATH=monai_aegis python run_image_pipeline.py --config config.yaml

Or run both via this orchestrator::

    PYTHONPATH=monai_aegis python run_pipeline.py --config config.yaml --mode auto
    PYTHONPATH=monai_aegis python run_pipeline.py --config config.yaml --mode dicom
    PYTHONPATH=monai_aegis python run_pipeline.py --config config.yaml --mode image
"""
import argparse
import logging

from run_dicom_pipeline import run_single as run_dicom_single
from run_dicom_pipeline import run_series as run_dicom_series
from run_image_pipeline import run_image_pipeline

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Aegis De-identification Pipeline — Unified Orchestrator",
    )
    parser.add_argument(
        "--config",
        default="monai_aegis/config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "dicom", "dicom-single", "image"],
        default="auto",
        help=(
            "Pipeline mode: "
            "'auto' (DICOM series + images, default), "
            "'dicom' (DICOM series only), "
            "'dicom-single' (DICOM per-file), "
            "'image' (JPEG/PNG only)."
        ),
    )

    args = parser.parse_args()

    if args.mode == "auto":
        logger.info("=== Running DICOM series pipeline ===")
        run_dicom_series(args.config)
        logger.info("=== Running Image pipeline ===")
        run_image_pipeline(args.config)

    elif args.mode == "dicom":
        run_dicom_series(args.config)

    elif args.mode == "dicom-single":
        run_dicom_single(args.config)

    elif args.mode == "image":
        run_image_pipeline(args.config)


if __name__ == "__main__":
    main()
