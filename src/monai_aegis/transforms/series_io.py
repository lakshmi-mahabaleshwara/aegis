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
from typing import Any, Dict, Hashable, List, Mapping, Optional, TYPE_CHECKING

from monai_aegis.config.storage import AegisFileSystem

import numpy as np
import pydicom
import pydicom.uid
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, ThreadUnsafe, Transform
from monai.utils.enums import TransformBackends

from monai_aegis.transforms.exceptions import SeriesLoadError, SeriesSaveError
from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck

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
        uris: List[str],
        datasets: Optional[List[pydicom.Dataset]] = None,
        fs: Optional['AegisFileSystem'] = None,
    ) -> MetaTensor:
        """Load a series of DICOM files into a single volume tensor.

        Args:
            uris: Sorted list of ``.dcm`` file paths for one series.
            datasets: Optional pre-loaded pydicom Datasets (one per uri).
                If ``None``, each file is read from disk.

        Returns:
            MetaTensor with shape ``(1, D, H, W)`` for grayscale or
            ``(3, D, H, W)`` for RGB, plus enriched ``.meta`` dict.

        Raises:
            SeriesLoadError: If files cannot be read, pixels are missing,
                or slice dimensions are inconsistent.
        """
        if not uris:
            raise SeriesLoadError(
                "Empty file list — nothing to load",
                transform="LoadDicomSeries",
            )

        # Load datasets if not provided
        if datasets is None:
            datasets = []
            for fp in uris:
                try:
                    if fs is not None:
                        with fs.open_read(fp) as f:
                            datasets.append(pydicom.dcmread(f))
                    else:
                        datasets.append(pydicom.dcmread(fp))
                except Exception as e:
                    raise SeriesLoadError(
                        f"Failed to read DICOM: {e}",
                        uri=fp,
                        transform="LoadDicomSeries",
                    ) from e

        # ---- Multi-frame DICOM (single file, NumberOfFrames > 1) ----
        if len(datasets) == 1 and hasattr(datasets[0], 'NumberOfFrames'):
            nf = int(datasets[0].NumberOfFrames)
            if nf > 1:
                return self._load_multiframe(datasets[0], uris[0])

        # ---- Multi-file series ----
        return self._load_multifile(datasets, uris)

    def _load_multiframe(
        self,
        ds: pydicom.Dataset,
        uri: str,
    ) -> MetaTensor:
        """Load a single multi-frame DICOM as a volume."""
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise SeriesLoadError(
                f"Failed to read pixel data from multi-frame DICOM: {e}",
                uri=uri,
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
            'filename_or_obj': uri,
            'spatial_shape': pixel_array.shape[1:],   # (D, H, W)
            'original_channel_dim': 0,
            'modality': getattr(ds, 'Modality', ''),
            'patient_id': getattr(ds, 'PatientID', ''),
            'study_date': getattr(ds, 'StudyDate', ''),
            'study_instance_uid': getattr(ds, 'StudyInstanceUID', ''),
            'series_instance_uid': getattr(ds, 'SeriesInstanceUID', ''),
            'slice_uris': [uri],
            'is_multiframe': True,
            'num_slices': int(ds.NumberOfFrames),
        }
        return MetaTensor(torch.as_tensor(pixel_array), meta=meta)

    def _load_multifile(
        self,
        datasets: List[pydicom.Dataset],
        uris: List[str],
    ) -> MetaTensor:
        """Stack multiple single-frame DICOMs into a volume."""
        slice_arrays = []
        for i, (ds, fp) in enumerate(zip(datasets, uris)):
            try:
                arr = ds.pixel_array.astype(np.float32)
            except Exception as e:
                raise SeriesLoadError(
                    f"Failed to read pixel data from slice {i}: {e}",
                    uri=fp,
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
                uri=uris[0],
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
            'filename_or_obj': uris[0],
            'spatial_shape': volume.shape[1:],   # (D, H, W)
            'original_channel_dim': 0,
            'modality': getattr(ds0, 'Modality', ''),
            'patient_id': getattr(ds0, 'PatientID', ''),
            'study_date': getattr(ds0, 'StudyDate', ''),
            'study_instance_uid': getattr(ds0, 'StudyInstanceUID', ''),
            'series_instance_uid': getattr(ds0, 'SeriesInstanceUID', ''),
            'slice_uris': list(uris),
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
    Also handles Identity Tokenization for the series.

    Args:
        keys: Keys whose values are lists of sorted file paths.
        config: Configuration dict (used for tokenization salt).
        allow_missing_keys: If True, skip missing keys instead of raising.
        fs: Optional :py:class:`AegisFileSystem` for cloud I/O.

    Example::

        transform = LoadDicomSeriesd(keys=["image"], config=config)
        data = transform({"image": ["/path/slice1.dcm", "/path/slice2.dcm"]})
        # data["image"].shape → (1, 2, 512, 512)
    """

    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any],
        input_dir: str = "",
        allow_missing_keys: bool = False,
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadDicomSeries()
        self.fs = fs
        self.input_dir = input_dir
        from monai_aegis.transforms.utility import AegisIdentityManager
        self.identity_manager = AegisIdentityManager.from_config(config)

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
            uris = d[key]
            if isinstance(uris, str):
                uris = [uris]
            uris = [str(fp) for fp in uris]

            try:
                # Load all datasets and cache them
                datasets: List[pydicom.Dataset] = []
                for fp in uris:
                    if self.fs is not None:
                        with self.fs.open_read(fp) as f:
                            datasets.append(pydicom.dcmread(f))
                    else:
                        datasets.append(pydicom.dcmread(fp))
                d[ck(key, ckeys.DICOM_DATASETS)] = datasets

                meta_tensor = self.transform(uris, datasets=datasets, fs=self.fs)
                d[key] = meta_tensor
                d[ck(key, ckeys.META_DICT)] = meta_tensor.meta

                from monai_aegis.transforms.utility import resolve_target_token
                target_token = resolve_target_token(
                    uri=uris[0],
                    identity_manager=self.identity_manager,
                    input_dir=self.input_dir,
                    fs=self.fs,
                )

                d[ck(key, ckeys.TARGET_TOKEN)] = target_token
                
            except SeriesLoadError:
                raise
            except Exception as e:
                raise SeriesLoadError(
                    f"Unexpected error loading series: {e}",
                    uri=uris[0] if uris else '',
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

    def __init__(self, output_dir: str, input_dir: str = '', fs: Optional['AegisFileSystem'] = None) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.fs = fs

    def __call__(
        self,
        datasets: List[pydicom.Dataset],
        original_uris: List[str],
        target_token: Optional[str] = None,
    ) -> List[str]:
        """Save a list of scrubbed datasets as individual DICOM files.

        Output paths mirror the original input structure unless a
        ``target_token`` is provided, which forces the output folder to be
        just the token (ignoring original folder structure).

        Args:
            datasets: Scrubbed pydicom Datasets (one per slice).
            original_uris: Original input file paths (one per slice).
            target_token: Optional enforced output subdirectory name across all slices.

        Returns:
            List of output file paths.

        Raises:
            SeriesSaveError: If any slice cannot be written.
        """
        output_paths: List[str] = []
        for i, (ds, orig_fp) in enumerate(zip(datasets, original_uris)):
            # SOPInstanceUID is regenerated deterministically upstream during
            # metadata scrubbing (ScrubDicomMetadata.remap_uid). Persist it
            # as-is and only keep the file-meta pointer in sync, so the output
            # is a valid DICOM Part 10 object. Re-randomising here previously
            # broke both file-meta consistency and cross-run determinism.
            if getattr(ds, 'file_meta', None) is not None:
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

            from monai_aegis.transforms.utility import build_output_path
            out_path = build_output_path(
                uri=orig_fp,
                output_dir=self.output_dir,
                input_dir=self.input_dir,
                target_token=target_token,
                is_cloud=self.fs is not None and self.fs.protocol != 'file'
            )

            if self.fs is not None:
                self.fs.makedirs(self.fs.dirname(out_path))
            else:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                if self.fs is not None:
                    with self.fs.open_write(out_path) as f:
                        ds.save_as(f)
                else:
                    ds.save_as(out_path)
            except Exception as e:
                raise SeriesSaveError(
                    f"Failed to save slice {i}: {e}",
                    uri=out_path,
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
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.saver = SaveDicomSeries(output_dir=output_dir, input_dir=input_dir, fs=fs)
        self.fs = fs

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Save scrubbed DICOM series found in the data dict.

        Uses ``slice_uris`` from the metadata to preserve the
        original folder structure and filenames.

        A loaded series that was never scrubbed raises — un-de-identified
        data must never be silently dropped.

        Args:
            data: Pipeline data dictionary.

        Returns:
            The data dictionary (unmodified — saving is a side-effect).

        Raises:
            SeriesSaveError: If the scrub stage did not run for a loaded
                series, or any slice cannot be written.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            meta = d.get(ck(key, ckeys.META_DICT), {})
            if meta.get("is_multiframe"):
                scrubbed_ds = d.get(ck(key, ckeys.SCRUBBED_DS))
                if scrubbed_ds is None:
                    raise SeriesSaveError(
                        "No scrubbed dataset in pipeline data for a loaded "
                        "multi-frame DICOM — was ScrubDicomMetadatad run "
                        "before SaveDicomSeriesd?",
                        uri=str(meta.get("filename_or_obj", "")),
                        transform="SaveDicomSeriesd",
                    )

                original_uri = meta.get("filename_or_obj")
                if isinstance(original_uri, list):
                    original_uri = original_uri[0]
                if not original_uri:
                    continue

                from monai_aegis.transforms.io import SaveDicom
                try:
                    saver = SaveDicom(
                        output_dir=self.saver.output_dir,
                        input_dir=self.saver.input_dir,
                        fs=self.fs,
                    )
                    out_path = saver(
                        dataset=scrubbed_ds,
                        uri=str(original_uri),
                        target_token=d.get(ck(key, ckeys.TARGET_TOKEN)),
                    )
                    d[ck(key, ckeys.SAVED_PATH)] = out_path
                except SeriesSaveError:
                    raise
                except Exception as e:
                    raise SeriesSaveError(
                        f"Unexpected error saving multi-frame series: {e}",
                        uri=str(original_uri),
                        transform="SaveDicomSeriesd",
                    ) from e
                continue

            scrubbed_list = d.get(ck(key, ckeys.SCRUBBED_DATASETS))
            if scrubbed_list is None:
                if ck(key, ckeys.DICOM_DATASETS) in d:
                    raise SeriesSaveError(
                        "No scrubbed datasets in pipeline data for a loaded "
                        "series — was ScrubDicomMetadatad run before "
                        "SaveDicomSeriesd?",
                        uri=str(meta.get("filename_or_obj", "")),
                        transform="SaveDicomSeriesd",
                    )
                continue  # nothing loaded for this key
            original_uris = meta.get('slice_uris', [])

            try:
                out_paths = self.saver(
                    datasets=scrubbed_list,
                    original_uris=original_uris,
                    target_token=d.get(ck(key, ckeys.TARGET_TOKEN)),
                )
                d[ck(key, ckeys.SAVED_PATHS)] = out_paths
            except SeriesSaveError:
                raise
            except Exception as e:
                raise SeriesSaveError(
                    f"Unexpected error saving series: {e}",
                    uri=self.saver.output_dir,
                    transform="SaveDicomSeriesd",
                ) from e

        return d
