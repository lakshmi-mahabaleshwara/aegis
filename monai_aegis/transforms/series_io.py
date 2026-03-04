"""
MONAI Aegis Series I/O — LoadDicomSeries/d / SaveDicomSeries/d

Load a DICOM series as a ``(C, D, H, W)`` volume and write
de-identified series back to disk, one slice per file.

All loads are thread-safe; saves are ``ThreadUnsafe``.

Raises:
    SeriesLoadError: When a series cannot be assembled into a volume.
    SeriesSaveError: When a de-identified series cannot be written.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Hashable, List, Mapping, Optional

import numpy as np
import pydicom
import pydicom.uid
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, ThreadUnsafe, Transform
from monai.utils.enums import TransformBackends

from transforms.exceptions import SeriesLoadError, SeriesSaveError

__all__ = ["LoadDicomSeries", "LoadDicomSeriesd", "SaveDicomSeries", "SaveDicomSeriesd"]

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Array transforms
# -----------------------------------------------------------------------

class LoadDicomSeries(Transform):
    """Array transform: Load a DICOM series as a ``(C, D, H, W)`` volume.

    Handles two storage formats:
      - **Multi-file series** — one ``.dcm`` per slice, stacked into a volume.
      - **Multi-frame DICOM** — single file with ``NumberOfFrames > 1``.

    Args:
        None — stateless, all inputs passed to ``__call__``.

    Returns:
        MetaTensor with shape ``(C, D, H, W)`` and enriched ``.meta`` dict.

    Raises:
        SeriesLoadError: If the series cannot be loaded or assembled.
    """

    backend = [TransformBackends.NUMPY]

    def __call__(
        self,
        filepaths: List[str],
        datasets: Optional[List[pydicom.Dataset]] = None,
    ) -> MetaTensor:
        """Load a series of DICOM files into a single volume tensor.

        Args:
            filepaths: Sorted list of ``.dcm`` file paths for one series.
            datasets: Optional pre-loaded pydicom Datasets (one per filepath).
                If ``None``, each file is read from disk.

        Returns:
            MetaTensor with shape ``(1, D, H, W)`` for grayscale or
            ``(3, D, H, W)`` for RGB, plus enriched ``.meta`` dict.

        Raises:
            SeriesLoadError: If files cannot be read, pixels are missing,
                or slice dimensions are inconsistent.
        """
        if not filepaths:
            raise SeriesLoadError(
                "Empty file list — nothing to load",
                transform="LoadDicomSeries",
            )

        # Load datasets if not provided
        if datasets is None:
            datasets = []
            for fp in filepaths:
                try:
                    datasets.append(pydicom.dcmread(fp))
                except Exception as e:
                    raise SeriesLoadError(
                        f"Failed to read DICOM: {e}",
                        filepath=fp,
                        transform="LoadDicomSeries",
                    ) from e

        # ---- Multi-frame DICOM (single file, NumberOfFrames > 1) ----
        if len(datasets) == 1 and hasattr(datasets[0], 'NumberOfFrames'):
            nf = int(datasets[0].NumberOfFrames)
            if nf > 1:
                return self._load_multiframe(datasets[0], filepaths[0])

        # ---- Multi-file series ----
        return self._load_multifile(datasets, filepaths)

    def _load_multiframe(
        self,
        ds: pydicom.Dataset,
        filepath: str,
    ) -> MetaTensor:
        """Load a single multi-frame DICOM as a volume."""
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise SeriesLoadError(
                f"Failed to read pixel data from multi-frame DICOM: {e}",
                filepath=filepath,
                transform="LoadDicomSeries",
            ) from e

        # pixel_array shape: (D, H, W) or (D, H, W, 3)
        if pixel_array.ndim == 3:
            # Grayscale: (D, H, W) → (1, D, H, W)
            pixel_array = pixel_array[np.newaxis, ...]
        elif pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
            # RGB: (D, H, W, 3) → (3, D, H, W)
            pixel_array = np.transpose(pixel_array, (3, 0, 1, 2))
        else:
            # Unexpected — wrap with channel dim
            pixel_array = pixel_array[np.newaxis, ...]

        meta = {
            'filename_or_obj': filepath,
            'spatial_shape': pixel_array.shape[1:],   # (D, H, W)
            'original_channel_dim': 0,
            'modality': getattr(ds, 'Modality', ''),
            'patient_id': getattr(ds, 'PatientID', ''),
            'study_date': getattr(ds, 'StudyDate', ''),
            'study_instance_uid': getattr(ds, 'StudyInstanceUID', ''),
            'series_instance_uid': getattr(ds, 'SeriesInstanceUID', ''),
            'slice_filepaths': [filepath],
            'is_multiframe': True,
            'num_slices': int(ds.NumberOfFrames),
        }
        return MetaTensor(torch.as_tensor(pixel_array), meta=meta)

    def _load_multifile(
        self,
        datasets: List[pydicom.Dataset],
        filepaths: List[str],
    ) -> MetaTensor:
        """Stack multiple single-frame DICOMs into a volume."""
        slice_arrays = []
        for i, (ds, fp) in enumerate(zip(datasets, filepaths)):
            try:
                arr = ds.pixel_array.astype(np.float32)
            except Exception as e:
                raise SeriesLoadError(
                    f"Failed to read pixel data from slice {i}: {e}",
                    filepath=fp,
                    transform="LoadDicomSeries",
                ) from e

            # Normalize to (H, W) or (H, W, 3)
            if arr.ndim == 2:
                pass  # grayscale — fine
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                pass  # RGB — fine
            else:
                logger.warning("Unexpected pixel shape %s in %s", arr.shape, fp)

            slice_arrays.append(arr)

        # Stack → (D, H, W) or (D, H, W, 3)
        try:
            volume = np.stack(slice_arrays, axis=0)
        except ValueError as e:
            raise SeriesLoadError(
                f"Cannot stack slices — inconsistent dimensions: {e}",
                filepath=filepaths[0],
                transform="LoadDicomSeries",
            ) from e

        # Add channel dimension
        if volume.ndim == 3:
            # Grayscale: (D, H, W) → (1, D, H, W)
            volume = volume[np.newaxis, ...]
        elif volume.ndim == 4 and volume.shape[-1] == 3:
            # RGB: (D, H, W, 3) → (3, D, H, W)
            volume = np.transpose(volume, (3, 0, 1, 2))

        ds0 = datasets[0]
        meta = {
            'filename_or_obj': filepaths[0],
            'spatial_shape': volume.shape[1:],   # (D, H, W)
            'original_channel_dim': 0,
            'modality': getattr(ds0, 'Modality', ''),
            'patient_id': getattr(ds0, 'PatientID', ''),
            'study_date': getattr(ds0, 'StudyDate', ''),
            'study_instance_uid': getattr(ds0, 'StudyInstanceUID', ''),
            'series_instance_uid': getattr(ds0, 'SeriesInstanceUID', ''),
            'slice_filepaths': list(filepaths),
            'is_multiframe': False,
            'num_slices': len(datasets),
        }
        return MetaTensor(torch.as_tensor(volume), meta=meta)


# -----------------------------------------------------------------------
# Dictionary transforms
# -----------------------------------------------------------------------

class LoadDicomSeriesd(MapTransform):
    """Dictionary transform: Load a DICOM series as a ``(C, D, H, W)`` volume.

    Expects ``{key}`` to contain a **list of file paths** for one series
    (output of :py:func:`discover_dicoms` + :py:func:`sort_slices`).

    Thread-safe — stateless file reading only.

    Args:
        keys: Keys whose values are lists of sorted file paths.
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        transform = LoadDicomSeriesd(keys=["image"])
        data = transform({"image": ["/path/slice1.dcm", "/path/slice2.dcm"]})
        # data["image"].shape → (1, 2, 512, 512)
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadDicomSeries()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Load each keyed list of file paths into a volume MetaTensor.

        Side-effects written to the data dict per key:
            - ``{key}`` — MetaTensor ``(C, D, H, W)``.
            - ``{key}_meta_dict`` — live reference to ``MetaTensor.meta``.
            - ``{key}_dicom_datasets`` — list of cached ``pydicom.Dataset``.

        Args:
            data: Dictionary mapping keys to lists of DICOM file paths.

        Returns:
            Updated dictionary with volume tensors and metadata.

        Raises:
            SeriesLoadError: If any series cannot be loaded.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            filepaths = d[key]
            if isinstance(filepaths, str):
                filepaths = [filepaths]
            filepaths = [str(fp) for fp in filepaths]

            try:
                # Load all datasets and cache them
                datasets: List[pydicom.Dataset] = []
                for fp in filepaths:
                    datasets.append(pydicom.dcmread(fp))
                d[f"{key}_dicom_datasets"] = datasets

                meta_tensor = self.transform(filepaths, datasets=datasets)
                d[key] = meta_tensor
                d[f"{key}_meta_dict"] = meta_tensor.meta
            except SeriesLoadError:
                raise
            except Exception as e:
                raise SeriesLoadError(
                    f"Unexpected error loading series: {e}",
                    filepath=filepaths[0] if filepaths else '',
                    transform="LoadDicomSeriesd",
                ) from e

        return d


class SaveDicomSeries(Transform):
    """Array transform: Write a de-identified series to disk.

    Preserves the original folder structure and filenames relative to
    ``input_dir``.  If ``input_dir`` is not provided, files are saved
    with their original basenames directly into ``output_dir``.

    Regenerates ``SOPInstanceUID`` per slice to ensure uniqueness.

    Marked as ``ThreadUnsafe`` because it performs file I/O.

    Args:
        output_dir: Root directory for de-identified output.
        input_dir: Root input directory (used to compute relative paths).
    """

    def __init__(self, output_dir: str, input_dir: str = '') -> None:
        super().__init__()
        self.output_dir = output_dir
        self.input_dir = input_dir

    def __call__(
        self,
        datasets: List[pydicom.Dataset],
        original_filepaths: List[str],
    ) -> List[str]:
        """Save a list of scrubbed datasets as individual DICOM files.

        Output paths mirror the original input structure::

            input_dir/sub/slice.dcm  →  output_dir/sub/slice.dcm

        Args:
            datasets: Scrubbed pydicom Datasets (one per slice).
            original_filepaths: Original input file paths (one per slice).

        Returns:
            List of output file paths.

        Raises:
            SeriesSaveError: If any slice cannot be written.
        """
        output_paths: List[str] = []
        for i, (ds, orig_fp) in enumerate(zip(datasets, original_filepaths)):
            # Regenerate SOPInstanceUID for uniqueness
            ds.SOPInstanceUID = pydicom.uid.generate_uid()

            # Preserve original folder + filename
            if self.input_dir:
                rel_path = os.path.relpath(orig_fp, self.input_dir)
            else:
                rel_path = os.path.basename(orig_fp)

            out_path = os.path.join(self.output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                ds.save_as(out_path)
            except Exception as e:
                raise SeriesSaveError(
                    f"Failed to save slice {i}: {e}",
                    filepath=out_path,
                    transform="SaveDicomSeries",
                ) from e

            output_paths.append(out_path)

        logger.info(
            "Saved %d slices to %s", len(datasets), self.output_dir
        )
        return output_paths


class SaveDicomSeriesd(MapTransform, ThreadUnsafe):
    """Dictionary transform: Write de-identified DICOM series to disk.

    Picks up ``{key}_scrubbed_datasets`` (list of scrubbed Datasets)
    from the data dict and writes them to the output directory,
    preserving the original folder structure and filenames.

    Marked as ``ThreadUnsafe`` because it performs file I/O.

    Args:
        keys: Keys of the data dictionary to process.
        output_dir: Root directory for de-identified output.
        input_dir: Root input directory (for computing relative paths).
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        pipeline = Compose([
            LoadDicomSeriesd(keys=["image"]),
            RedactPixelPHId(keys=["image"], config=config),
            ScrubDicomMetadatad(keys=["image"], config=config),
            SaveDicomSeriesd(keys=["image"], output_dir="./output", input_dir="./input"),
        ])
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        input_dir: str = '',
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.saver = SaveDicomSeries(output_dir=output_dir, input_dir=input_dir)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Save scrubbed DICOM series found in the data dict.

        Uses ``slice_filepaths`` from the metadata to preserve the
        original folder structure and filenames.

        Args:
            data: Pipeline data dictionary.

        Returns:
            The data dictionary (unmodified — saving is a side-effect).

        Raises:
            SeriesSaveError: If any slice cannot be written.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            scrubbed_list = d.get(f"{key}_scrubbed_datasets")
            if scrubbed_list is None:
                continue

            meta = d.get(f"{key}_meta_dict", {})
            original_filepaths = meta.get('slice_filepaths', [])

            try:
                self.saver(
                    datasets=scrubbed_list,
                    original_filepaths=original_filepaths,
                )
            except SeriesSaveError:
                raise
            except Exception as e:
                raise SeriesSaveError(
                    f"Unexpected error saving series: {e}",
                    filepath=self.saver.output_dir,
                    transform="SaveDicomSeriesd",
                ) from e

        return d

