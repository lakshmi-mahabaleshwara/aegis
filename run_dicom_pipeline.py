"""Compatibility wrapper for the package-local DICOM runner."""

from monai_aegis.dicom_runner import main, run_series, run_single


if __name__ == "__main__":
    main()
