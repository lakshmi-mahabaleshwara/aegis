"""
Package-local unified CLI entrypoint.
"""
import argparse
import logging

from monai_aegis.dicom_runner import run_series as run_dicom_series
from monai_aegis.dicom_runner import run_single as run_dicom_single
from monai_aegis.image_runner import run_series as run_image_series

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Aegis De-identification Pipeline")
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
            "'image' (JPEG/PNG series only)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "auto":
        logger.info("=== Running DICOM series pipeline ===")
        run_dicom_series(args.config)
        logger.info("=== Running image pipeline ===")
        run_image_series(args.config)
    elif args.mode == "dicom":
        run_dicom_series(args.config)
    elif args.mode == "dicom-single":
        run_dicom_single(args.config)
    else:
        run_image_series(args.config)


__all__ = ["main"]


if __name__ == "__main__":
    main()
