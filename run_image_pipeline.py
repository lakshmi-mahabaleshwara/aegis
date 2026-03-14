"""Compatibility wrapper for the package-local image runner."""

from monai_aegis.image_runner import main, run_series, run_single


if __name__ == "__main__":
    main()
