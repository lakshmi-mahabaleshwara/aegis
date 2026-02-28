"""
MONAI Aegis I/O Transforms — LoadDicomRaw / LoadDicomRawd / SaveDicom / SaveDicomd

Load DICOM/JPEG without spatial transforms and save scrubbed DICOMs.
"""
import os
import numpy as np
import pydicom
import torch
import logging
from typing import Dict, Hashable, Mapping, Any, Optional
from PIL import Image
from monai.transforms import Transform, MapTransform, ThreadUnsafe
from monai.data import MetaTensor

from transforms.exceptions import (
    DicomLoadError, ImageLoadError, DicomSaveError
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load Transforms (Thread-safe)
# ---------------------------------------------------------------------------

class LoadDicomRaw(Transform):
    """
    Array transform: Load a DICOM or JPEG file without spatial transforms.

    Preserves original pixel orientation and returns a MetaTensor with metadata.
    Unlike MONAI's ``LoadImage``, this avoids affine transforms that can
    rotate burned-in text away from OCR detection.

    Returns:
        MetaTensor in channel-first format (C, H, W).
    """

    def __call__(self, filepath: str, dataset: Optional[pydicom.Dataset] = None) -> MetaTensor:
        filepath = str(filepath)

        if filepath.lower().endswith('.dcm'):
            try:
                ds = dataset if dataset is not None else pydicom.dcmread(filepath)
                pixel_array = ds.pixel_array.astype(np.float32)
            except Exception as e:
                raise DicomLoadError(
                    f"Failed to read DICOM: {e}",
                    filepath=filepath,
                    transform="LoadDicomRaw",
                ) from e

            if pixel_array.ndim == 2:
                pixel_array = pixel_array[np.newaxis, ...]  # (1, H, W)
            elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
                pixel_array = np.transpose(pixel_array, (2, 0, 1))  # (3, H, W)

            meta = {
                'filename_or_obj': filepath,
                'spatial_shape': pixel_array.shape[1:],
                'original_channel_dim': 0,
                'modality': getattr(ds, 'Modality', ''),
                'patient_id': getattr(ds, 'PatientID', ''),
                'study_date': getattr(ds, 'StudyDate', ''),
            }
            return MetaTensor(torch.as_tensor(pixel_array), meta=meta)

        else:
            try:
                img = Image.open(filepath)
                pixel_array = np.array(img).astype(np.float32)
            except Exception as e:
                raise ImageLoadError(
                    f"Failed to read image: {e}",
                    filepath=filepath,
                    transform="LoadDicomRaw",
                ) from e

            if pixel_array.ndim == 2:
                pixel_array = pixel_array[np.newaxis, ...]
            elif pixel_array.ndim == 3 and pixel_array.shape[-1] in [3, 4]:
                pixel_array = np.transpose(pixel_array[:, :, :3], (2, 0, 1))

            meta = {
                'filename_or_obj': filepath,
                'spatial_shape': pixel_array.shape[1:],
                'original_channel_dim': 0,
            }
            return MetaTensor(torch.as_tensor(pixel_array), meta=meta)


class LoadDicomRawd(MapTransform):
    """
    Dictionary transform: Load DICOM/JPEG without spatial transforms.

    Dictionary-based wrapper of :py:class:`LoadDicomRaw`.
    Thread-safe — stateless file reading only.

    Args:
        keys: Keys whose values are file paths to load.
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        transform = LoadDicomRawd(keys=["image"])
        data = transform({"image": "/path/to/file.dcm"})
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadDicomRaw()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            filepath = str(d[key])
            try:
                ds = None
                
                # Read dataset once to keep it in memory
                if filepath.lower().endswith('.dcm'):
                    ds = pydicom.dcmread(filepath)
                    d[f"{key}_dicom_dataset"] = ds
                    
                meta_tensor = self.transform(filepath, dataset=ds)
                d[key] = meta_tensor

                # Alias MetaTensor.meta → {key}_meta_dict for backward compatibility.
                d[f"{key}_meta_dict"] = meta_tensor.meta
            except (DicomLoadError, ImageLoadError):
                raise
            except Exception as e:
                raise DicomLoadError(
                    f"Unexpected error loading file: {e}",
                    filepath=filepath,
                    transform="LoadDicomRawd",
                ) from e
        return d


# ---------------------------------------------------------------------------
# Save Transforms (ThreadUnsafe — file I/O)
# ---------------------------------------------------------------------------

class SaveDicom(Transform):
    """
    Array transform: Save a scrubbed pydicom Dataset to disk.

    This is the dedicated I/O transform for writing de-identified DICOMs.
    Separated from metadata scrubbing to keep transforms pure and thread-safe.

    Marked as ``ThreadUnsafe`` because it performs file I/O.

    Args:
        output_dir: Directory to write scrubbed DICOM files.

    Example::

        saver = SaveDicom(output_dir="./output")
        saver(dataset=scrubbed_ds, filepath="/path/to/original.dcm")
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, dataset: pydicom.Dataset, filepath: str) -> str:
        """
        Args:
            dataset: Scrubbed pydicom Dataset to save.
            filepath: Original filepath (used to derive output filename).

        Returns:
            Path to the saved file.
        """
        out_path = os.path.join(self.output_dir, os.path.basename(filepath))
        try:
            dataset.save_as(out_path)
        except Exception as e:
            raise DicomSaveError(
                f"Failed to save scrubbed DICOM: {e}",
                filepath=out_path,
                transform="SaveDicom",
            ) from e
        logger.info(f"Saved scrubbed DICOM to {out_path}")
        return out_path


class SaveDicomd(MapTransform, ThreadUnsafe):
    """
    Dictionary transform: Save scrubbed DICOM datasets to disk.

    Picks up ``{key}_scrubbed_ds`` from the data dict (written by
    :py:class:`ScrubDicomMetadatad`) and saves it to ``output_dir``.

    Marked as ``ThreadUnsafe`` because it performs file I/O.
    This is the ONLY transform in the pipeline that writes to disk,
    following MONAI's pattern of separating I/O from computation.

    Args:
        keys: Keys of the data dictionary to process.
        output_dir: Directory to write scrubbed DICOM files.
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        pipeline = Compose([
            LoadDicomRawd(keys=["image"]),
            RedactPixelPHId(keys=["image"], config=config),
            ScrubDicomMetadatad(keys=["image"], config=config),
            SaveDicomd(keys=["image"], output_dir="./output"),  # I/O last
        ])
    """

    def __init__(self, keys, output_dir: str, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.saver = SaveDicom(output_dir=output_dir)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            scrubbed_ds = d.get(f"{key}_scrubbed_ds")
            if scrubbed_ds is None:
                continue

            # Get original filepath for output filename
            meta = d.get(f"{key}_meta_dict", {})
            fpath = meta.get('filename_or_obj')
            if isinstance(fpath, list):
                fpath = fpath[0]
            if not fpath:
                continue

            try:
                self.saver(dataset=scrubbed_ds, filepath=fpath)
            except DicomSaveError:
                raise
            except Exception as e:
                raise DicomSaveError(
                    f"Unexpected error saving DICOM: {e}",
                    filepath=str(fpath),
                    transform="SaveDicomd",
                ) from e

        return d
