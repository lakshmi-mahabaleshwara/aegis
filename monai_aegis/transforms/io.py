"""
MONAI Aegis I/O Transforms — LoadDicomRaw / LoadDicomRawd / SaveDicom / SaveDicomd

Load DICOM/JPEG without spatial transforms and save scrubbed DICOMs.

This module contains the **Ingestion** and **Persistence** zones of the
pipeline.  All loads are thread-safe; all saves are marked ``ThreadUnsafe``
so MONAI routes them to the main thread in multi-worker DataLoaders.

Raises:
    DicomLoadError: When a DICOM file cannot be read or parsed.
    ImageLoadError: When a standard image (JPEG/PNG) cannot be opened.
    DicomSaveError: When a scrubbed dataset cannot be written to disk.
"""
import io as _io
import os
import numpy as np
import pydicom
import torch
import logging
from typing import Dict, Hashable, Mapping, Any, Optional, Sequence, TYPE_CHECKING, Union

from monai_aegis.config.storage import AegisFileSystem
from PIL import Image
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform, ThreadUnsafe
from monai.data import MetaTensor
from monai.utils.enums import TransformBackends

from monai_aegis.transforms.exceptions import (
    DicomLoadError, ImageLoadError, DicomSaveError
)

__all__ = [
    "LoadDicomRaw", "LoadDicomRawd", "SaveDicom", "SaveDicomd",
    "LoadImage", "LoadImaged", "SaveImage", "SaveImaged",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load Transforms (Thread-safe)
# ---------------------------------------------------------------------------

class LoadDicomRaw(Transform):
    """
    Array transform: Load a DICOM file without spatial transforms.

    Preserves original pixel orientation and returns a MetaTensor with metadata.
    Unlike MONAI's ``LoadImage``, this avoids affine transforms that can
    rotate burned-in text away from OCR detection.

    .. note::
        This transform handles **DICOM files only** (``.dcm``).
        For JPEG/PNG images, use :py:class:`LoadImage` instead.

    Returns:
        MetaTensor in channel-first format ``(C, H, W)``.

    Raises:
        DicomLoadError: If the DICOM file is corrupted, unreadable,
            or missing pixel data.
    """

    backend = [TransformBackends.NUMPY]

    def __call__(
        self,
        uri: str,
        dataset: Optional[pydicom.Dataset] = None,
        fs: Optional['AegisFileSystem'] = None,
    ) -> MetaTensor:
        """Load a single file and return a channel-first MetaTensor.

        Args:
            uri: Absolute or relative path to the ``.dcm`` file.
            dataset: Pre-loaded pydicom Dataset (skips ``dcmread`` if provided).
            fs: Optional :py:class:`AegisFileSystem` for cloud-native I/O.

        Returns:
            MetaTensor with shape ``(C, H, W)`` and enriched ``.meta`` dict.

        Raises:
            DicomLoadError: If the file cannot be parsed.
        """
        uri = str(uri)

        if uri.lower().endswith('.dcm'):
            try:
                if dataset is not None:
                    ds = dataset
                else:
                    _fs = fs if fs is not None else AegisFileSystem()
                    with _fs.open_read(uri) as f:
                        ds = pydicom.dcmread(f)
                pixel_array = ds.pixel_array.astype(np.float32)
            except Exception as e:
                raise DicomLoadError(
                    f"Failed to read DICOM: {e}",
                    uri=uri,
                    transform="LoadDicomRaw",
                ) from e

            if pixel_array.ndim == 2:
                pixel_array = pixel_array[np.newaxis, ...]  # (1, H, W)
            elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
                pixel_array = np.transpose(pixel_array, (2, 0, 1))  # (3, H, W)

            meta = {
                'filename_or_obj': uri,
                'spatial_shape': pixel_array.shape[1:],
                'original_channel_dim': 0,
                'modality': getattr(ds, 'Modality', ''),
                'patient_id': getattr(ds, 'PatientID', ''),
                'study_date': getattr(ds, 'StudyDate', ''),
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

    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any] = None,
        input_dir: str = "",
        allow_missing_keys: bool = False,
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadDicomRaw()
        self.fs = fs
        self.input_dir = input_dir
        from monai_aegis.transforms.utility import AegisIdentityManager
        self.identity_manager = AegisIdentityManager.from_config(config) if config else None

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Load each keyed file path into a MetaTensor.

        Side-effects written to the data dict per key:
            - ``{key}`` — the loaded MetaTensor ``(C, H, W)``.
            - ``{key}_meta_dict`` — live reference to ``MetaTensor.meta``.
            - ``{key}_dicom_dataset`` — cached ``pydicom.Dataset`` (DICOM only).

        Args:
            data: Input dictionary mapping keys to file paths.

        Returns:
            Updated dictionary with loaded tensors and metadata.

        Raises:
            DicomLoadError: If any DICOM file cannot be read.
            ImageLoadError: If any standard image file cannot be read.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            uri = str(d[key])
            try:
                ds = None
                
                # Read dataset once to keep it in memory
                if uri.lower().endswith('.dcm'):
                    if self.fs is not None:
                        with self.fs.open_read(uri) as f:
                            ds = pydicom.dcmread(f)
                    else:
                        ds = pydicom.dcmread(uri)
                    d[f"{key}_dicom_dataset"] = ds
                    
                import os
                target_token = None
                if self.identity_manager:
                    rel_path = uri
                    if hasattr(self, 'input_dir') and self.input_dir:
                        if self.fs is not None:
                            rel_path = self.fs.relpath(uri, self.input_dir)
                        else:
                            rel_path = os.path.relpath(uri, self.input_dir)
                    
                    sep = '/' if self.fs is not None else os.sep
                    parts = rel_path.split(sep)
                    folder_name = parts[0] if len(parts) > 1 else "default"
                    
                    target_token = self.identity_manager.get_token(folder_name)
                    # Keep original folder for singleton items in root or shallow nesting (len <= 1)
                    if len(parts) <= 1:
                        target_token = None

                meta_tensor = self.transform(uri, dataset=ds, fs=self.fs)
                d[key] = meta_tensor
                d[f"{key}_target_token"] = target_token

                # Alias MetaTensor.meta → {key}_meta_dict for backward compatibility.
                d[f"{key}_meta_dict"] = meta_tensor.meta
            except (DicomLoadError, ImageLoadError):
                raise
            except Exception as e:
                raise DicomLoadError(
                    f"Unexpected error loading file: {e}",
                    uri=uri,
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
        saver(dataset=scrubbed_ds, uri="/path/to/original.dcm")
    """

    def __init__(self, output_dir: str, input_dir: str = "", fs: Optional['AegisFileSystem'] = None):
        super().__init__()
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.fs = fs
        if fs is not None:
            fs.makedirs(output_dir)
        else:
            os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, dataset: pydicom.Dataset, uri: str, target_token: Optional[str] = None) -> str:
        """
        Args:
            dataset: Scrubbed pydicom Dataset to save.
            uri: Original uri (used to derive output filename).
            target_token: Optional identity token to replace the parent folder.

        Returns:
            Path to the saved file.
        """
        from monai_aegis.transforms.utility import build_output_path
        out_path = build_output_path(
            uri=uri,
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
                    dataset.save_as(f)
            else:
                dataset.save_as(out_path)
        except Exception as e:
            raise DicomSaveError(
                f"Failed to save scrubbed DICOM: {e}",
                uri=out_path,
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

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        input_dir: str = "",
        allow_missing_keys: bool = False,
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.saver = SaveDicom(output_dir=output_dir, input_dir=input_dir, fs=fs)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Save scrubbed DICOM datasets found in the data dict.

        Looks for ``{key}_scrubbed_ds`` entries written by
        :py:class:`ScrubDicomMetadatad`. Non-DICOM keys are silently skipped.

        Args:
            data: Pipeline data dictionary.

        Returns:
            The data dictionary (unmodified — saving is a side-effect).

        Raises:
            DicomSaveError: If the output file cannot be written.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            scrubbed_ds = d.get(f"{key}_scrubbed_ds")
            if scrubbed_ds is None:
                continue

            # Get original uri for output filename
            meta = d.get(f"{key}_meta_dict", {})
            fpath = meta.get('filename_or_obj')
            if isinstance(fpath, list):
                fpath = fpath[0]
            if not fpath:
                continue

            try:
                out_path = self.saver(dataset=scrubbed_ds, uri=fpath, target_token=d.get(f"{key}_target_token"))
                d[f"{key}_saved_path"] = out_path
            except DicomSaveError:
                raise
            except Exception as e:
                raise DicomSaveError(
                    f"Unexpected error saving DICOM: {e}",
                    uri=str(fpath),
                    transform="SaveDicomd",
                ) from e

        return d


# ---------------------------------------------------------------------------
# Image-only Load / Save Transforms
# ---------------------------------------------------------------------------

class LoadImage(Transform):
    """Array transform: Load a standard image (JPEG/PNG) without spatial transforms.

    Returns a MetaTensor in channel-first ``(C, H, W)`` format.

    .. note::
        This transform handles **standard images only** (``.jpg``, ``.jpeg``,
        ``.png``).  For DICOM files, use :py:class:`LoadDicomRaw` instead.

    Raises:
        ImageLoadError: If the file cannot be opened or decoded.
    """

    backend = [TransformBackends.NUMPY]

    def __call__(
        self,
        uri: str,
        fs: Optional['AegisFileSystem'] = None,
    ) -> MetaTensor:
        """Load a single image file and return a channel-first MetaTensor.

        Args:
            uri: Path to ``.jpg``, ``.jpeg``, or ``.png`` file.
            fs: Optional :py:class:`AegisFileSystem` for cloud I/O.

        Returns:
            MetaTensor with shape ``(C, H, W)`` and metadata.

        Raises:
            ImageLoadError: If the image cannot be read.
        """
        uri = str(uri)
        try:
            if fs is not None:
                with fs.open_read(uri) as f:
                    img = Image.open(f)
                    img.load()
            else:
                img = Image.open(uri)
            pixel_array = np.array(img).astype(np.float32)
        except Exception as e:
            raise ImageLoadError(
                f"Failed to read image: {e}",
                uri=uri,
                transform="LoadImage",
            ) from e

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]       # (1, H, W)
        elif pixel_array.ndim == 3 and pixel_array.shape[-1] in [3, 4]:
            pixel_array = np.transpose(pixel_array[:, :, :3], (2, 0, 1))  # (3, H, W)

        meta = {
            'filename_or_obj': uri,
            'spatial_shape': pixel_array.shape[1:],
            'original_channel_dim': 0,
        }
        return MetaTensor(torch.as_tensor(pixel_array), meta=meta)


class LoadImaged(MapTransform):
    """Dictionary transform: Load standard images (JPEG/PNG).

    Dictionary-based wrapper of :py:class:`LoadImage`.
    Thread-safe — stateless file reading only.
    Also handles Identity Tokenization for standard files.

    Args:
        keys: Keys whose values are file paths to load.
        config: Configuration dict (used for tokenization salt).
        allow_missing_keys: If True, skip missing keys instead of raising.
        fs: Optional :py:class:`AegisFileSystem` for cloud I/O.

    Example::

        transform = LoadImaged(keys=["image"], config=config)
        data = transform({"image": "/path/to/file.jpg"})
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
        self.transform = LoadImage()
        self.input_dir = input_dir
        self.fs = fs
        from monai_aegis.transforms.utility import AegisIdentityManager
        self.identity_manager = AegisIdentityManager.from_config(config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Load each keyed file path into a MetaTensor.

        Args:
            data: Input dictionary mapping keys to image file paths.

        Returns:
            Updated dictionary with loaded tensors and metadata.

        Raises:
            ImageLoadError: If any image file cannot be read.
        """
        import os
        d = dict(data)
        for key in self.key_iterator(d):
            uri = str(d[key])
            try:
                # 1. Target Token Generation using Top-Level Folder
                rel_path = uri
                if hasattr(self, 'input_dir') and self.input_dir:
                    if self.fs is not None:
                        rel_path = self.fs.relpath(uri, self.input_dir)
                    else:
                        rel_path = os.path.relpath(uri, self.input_dir)
                
                sep = '/' if self.fs is not None else os.sep
                parts = rel_path.split(sep)
                folder_name = parts[0] if len(parts) > 1 else "default"
                
                target_token = self.identity_manager.get_token(folder_name)
                # Keep original folder for singleton items in root or shallow nesting (len <= 1)
                if len(parts) <= 1:
                    target_token = None
                    
                # 2. Proceed with normal loading
                meta_tensor = self.transform(uri, fs=self.fs)
                d[key] = meta_tensor
                d[f"{key}_meta_dict"] = meta_tensor.meta
                d[f"{key}_target_token"] = target_token
            except ImageLoadError:
                raise
            except Exception as e:
                raise ImageLoadError(
                    f"Unexpected error loading image: {e}",
                    uri=uri,
                    transform="LoadImaged",
                ) from e
        return d


class SaveImage(Transform):
    """Array transform: Save a redacted image (JPEG/PNG) to disk.

    Converts channel-first ``(C, H, W)`` MetaTensor back to HWC format
    and writes as PNG (lossless) or JPEG.

    Marked as ``ThreadUnsafe`` because it performs file I/O.

    Args:
        output_dir: Directory to write output images.
        output_ext: Extension for output files (default ``'.png'``).
        fs: Optional :py:class:`AegisFileSystem` for cloud I/O.
    """

    def __init__(
        self,
        output_dir: str,
        input_dir: str = "",
        output_ext: str = '.png',
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.output_ext = output_ext
        self.fs = fs

    def __call__(self, tensor: MetaTensor, uri: str, target_token: Optional[str] = None) -> str:
        """Save a redacted image.

        Args:
            tensor: Channel-first MetaTensor ``(C, H, W)`` to save.
            uri: Original uri (used to derive output filename).
            target_token: Optional identity token to replace the parent folder.

        Returns:
            Path to the saved file.
        """
        img_array = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else np.array(tensor)

        # Channel-first → HWC for PIL
        if img_array.ndim == 3:
            if img_array.shape[0] == 1:
                img_array = img_array.squeeze(0)
            elif img_array.shape[0] == 3:
                img_array = np.moveaxis(img_array, 0, -1)

        # Normalize to uint8
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.1:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        # Build output path (use target_token if passed, replacing top-level directory)
            from monai_aegis.transforms.utility import build_output_path
        out_path = build_output_path(
            uri=uri,
            output_dir=self.output_dir,
            input_dir=self.input_dir,
            target_token=target_token,
            is_cloud=self.fs is not None and self.fs.protocol != 'file',
            output_ext=self.output_ext
        )

        if self.fs is not None:
            self.fs.makedirs(self.fs.dirname(out_path))
            pil_img = Image.fromarray(img_array)
            with self.fs.open_write(out_path) as f:
                pil_img.save(f, format=self.output_ext.lstrip('.').upper())
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(img_array).save(out_path)

        logger.info("Saved redacted image to %s", out_path)
        return out_path


class SaveImaged(MapTransform, ThreadUnsafe):
    """Dictionary transform: Save redacted images (JPEG/PNG) to disk.

    Looks up the MetaTensor at each key, converts to HWC, and saves.

    Args:
        keys: Keys of the data dictionary to save.
        output_dir: Directory to write output images.
        output_ext: Extension for output files (default ``'.png'``).
        allow_missing_keys: If True, skip missing keys instead of raising.
        fs: Optional :py:class:`AegisFileSystem` for cloud I/O.

    Example::

        pipeline = Compose([
            LoadImaged(keys=["image"]),
            RedactPixelPHId(keys=["image"], config=config),
            SaveImaged(keys=["image"], output_dir="./output"),
        ])
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        input_dir: str = "",
        output_ext: str = '.png',
        allow_missing_keys: bool = False,
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.saver = SaveImage(output_dir=output_dir, input_dir=input_dir, output_ext=output_ext, fs=fs)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Save redacted images found in the data dict.

        Args:
            data: Pipeline data dictionary.

        Returns:
            The data dictionary with ``{key}_saved_path`` added.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            tensor = d[key]
            if not isinstance(tensor, (MetaTensor, torch.Tensor)):
                continue

            meta = d.get(f"{key}_meta_dict", {})
            fpath = meta.get('filename_or_obj', '')
            if isinstance(fpath, list):
                fpath = fpath[0]
            if not fpath:
                continue

            out_path = self.saver(tensor=tensor, uri=fpath, target_token=d.get(f"{key}_target_token"))
            d[f"{key}_saved_path"] = out_path

        return d
