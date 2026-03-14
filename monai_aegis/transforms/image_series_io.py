"""
MONAI Aegis Series I/O — LoadImageSeries/d / SaveImageSeries/d

Load a directory of standard images (JPEGs/PNGs) as a ``(C, D, H, W)`` volume and
write them back to disk.

All loads are thread-safe; saves are ``ThreadUnsafe``.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Hashable, List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from monai_aegis.config.storage import AegisFileSystem

import numpy as np
import torch
from PIL import Image
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform, ThreadUnsafe, Transform
from monai.utils.enums import TransformBackends

from monai_aegis.transforms.exceptions import SeriesLoadError, SeriesSaveError

__all__ = ["LoadImageSeries", "LoadImageSeriesd", "SaveImageSeries", "SaveImageSeriesd"]

logger = logging.getLogger(__name__)

class LoadImageSeries(Transform):
    backend = [TransformBackends.NUMPY]

    def __call__(
        self,
        uris: List[str],
        fs: Optional['AegisFileSystem'] = None,
    ) -> MetaTensor:
        if not uris:
            raise SeriesLoadError("Empty file list — nothing to load", transform="LoadImageSeries")

        slice_arrays = []
        expected_shape = None

        for i, fp in enumerate(uris):
            try:
                if fs is not None:
                    with fs.open_read(fp) as f:
                        img = Image.open(f).convert("RGB")
                        arr = np.array(img, dtype=np.float32)
                else:
                    img = Image.open(fp).convert("RGB")
                    arr = np.array(img, dtype=np.float32)
            except Exception as e:
                raise SeriesLoadError(f"Failed to read image slice {i}: {e}", uri=fp, transform="LoadImageSeries") from e

            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise SeriesLoadError(f"Dimension mismatch across images: {expected_shape} != {arr.shape}", uri=fp, transform="LoadImageSeries")
            
            slice_arrays.append(arr)

        try:
            # Stack into (D, H, W, 3)
            volume = np.stack(slice_arrays, axis=0)
        except ValueError as e:
            raise SeriesLoadError(f"Cannot stack slices: {e}", uri=uris[0], transform="LoadImageSeries") from e

        # Convert to (3, D, H, W)
        volume = np.transpose(volume, (3, 0, 1, 2))

        meta = {
            'filename_or_obj': uris[0],
            'spatial_shape': volume.shape[1:],   # (D, H, W)
            'original_channel_dim': 0,
            'slice_uris': list(uris),
            'num_slices': len(uris),
        }
        return MetaTensor(torch.as_tensor(volume), meta=meta)

class LoadImageSeriesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any],
        input_dir: str = "",
        allow_missing_keys: bool = False,
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadImageSeries()
        self.fs = fs
        self.input_dir = input_dir
        from monai_aegis.transforms.utility import AegisIdentityManager
        self.identity_manager = AegisIdentityManager.from_config(config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            uris = d[key]
            if isinstance(uris, str):
                uris = [uris]
            uris = [str(fp) for fp in uris]

            try:
                meta_tensor = self.transform(uris, fs=self.fs)
                d[key] = meta_tensor
                d[f"{key}_meta_dict"] = meta_tensor.meta
                
                from monai_aegis.transforms.utility import resolve_target_token
                target_token = resolve_target_token(
                    uri=uris[0],
                    identity_manager=self.identity_manager,
                    input_dir=self.input_dir,
                    fs=self.fs,
                )

                d[f"{key}_target_token"] = target_token
                
            except SeriesLoadError:
                raise
            except Exception as e:
                raise SeriesLoadError(f"Unexpected error loading image series: {e}", uri=uris[0] if uris else '', transform="LoadImageSeriesd") from e

        return d

class SaveImageSeries(Transform):
    def __init__(self, output_dir: str, input_dir: str = "", output_ext: str = ".png", fs: Optional['AegisFileSystem'] = None) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.output_ext = output_ext
        self.fs = fs

    def __call__(
        self,
        tensor: torch.Tensor | np.ndarray,
        original_uris: List[str],
        target_token: Optional[str] = None,
    ) -> List[str]:
        output_paths: List[str] = []
        
        # Determine format
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = tensor.copy()
            
        # Ensure array is channels last: (3, D, H, W) -> (D, H, W, 3)
        if arr.ndim == 4 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 3, 0))
            
        num_slices = arr.shape[0]
        if num_slices != len(original_uris):
            raise SeriesSaveError(f"Tensor Depth ({num_slices}) does not match supplied original uris ({len(original_uris)})", uri="", transform="SaveImageSeries")

        for i in range(num_slices):
            slice_arr = arr[i]
            orig_fp = original_uris[i]
            
            # Convert float32 back to uint8 for saving via PIL
            if slice_arr.dtype != np.uint8:
                slice_arr = np.clip(slice_arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(slice_arr, mode="RGB")

            from monai_aegis.transforms.utility import build_output_path
            out_path = build_output_path(
                uri=orig_fp,
                output_dir=self.output_dir,
                input_dir=self.input_dir,
                target_token=target_token,
                is_cloud=self.fs is not None and self.fs.protocol != 'file',
                output_ext=self.output_ext
            )

            if self.fs is not None:
                self.fs.makedirs(self.fs.dirname(out_path))
            else:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                if self.fs is not None:
                    with self.fs.open_write(out_path) as f:
                        img.save(f)
                else:
                    img.save(out_path)
            except Exception as e:
                raise SeriesSaveError(f"Failed to save image slice {i}: {e}", uri=out_path, transform="SaveImageSeries") from e

            output_paths.append(out_path)

        logger.info("Saved %d standard image slices to %s", num_slices, self.output_dir)
        return output_paths

class SaveImageSeriesd(MapTransform, ThreadUnsafe):
    def __init__(
        self,
        keys: KeysCollection,
        output_dir: str,
        input_dir: str = "",
        output_ext: str = ".png",
        allow_missing_keys: bool = False,
        fs: Optional['AegisFileSystem'] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.saver = SaveImageSeries(output_dir=output_dir, input_dir=input_dir, output_ext=output_ext, fs=fs)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            tensor = d[key]
            meta = d.get(f"{key}_meta_dict", {})
            original_uris = meta.get('slice_uris', [])

            try:
                out_paths = self.saver(
                    tensor=tensor,
                    original_uris=original_uris,
                    target_token=d.get(f"{key}_target_token"),
                )
                d[f"{key}_saved_paths"] = out_paths
            except SeriesSaveError:
                raise
            except Exception as e:
                raise SeriesSaveError(f"Unexpected error saving image series: {e}", uri=self.saver.output_dir, transform="SaveImageSeriesd") from e

        return d
