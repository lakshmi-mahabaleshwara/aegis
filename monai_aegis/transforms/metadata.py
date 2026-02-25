"""
MONAI Aegis Metadata Transforms — ScrubDicomMetadata / ScrubDicomMetadatad

DICOM metadata de-identification with configurable PII actions
(REMOVE, ZERO, DUMMY) and pixel data injection.

Thread-safe: no file I/O in __call__() — saving is handled by SaveDicomd.
"""
import numpy as np
import pydicom
import copy
import torch
import logging
from typing import Dict, Hashable, Mapping, Any, Optional
from monai.transforms import Transform, MapTransform
from monai.data import MetaTensor

from transforms.utility import AegisIdentityManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Array Transform
# ---------------------------------------------------------------------------

class ScrubDicomMetadata(Transform):
    """
    Array transform: Scrub PII from a DICOM dataset and inject redacted pixels.

    Applies configurable PII actions (REMOVE, ZERO, DUMMY) to DICOM tags,
    removes private tags, and injects redacted pixel data with correct bit-depth.

    This transform is **pure** — it operates in-memory and does NOT write to disk.
    Use :py:class:`SaveDicom` / :py:class:`SaveDicomd` for persistence.

    Args:
        config: Configuration dict with 'pii_mapping' section.

    Example::

        transform = ScrubDicomMetadata(config=config)
        scrubbed_ds = transform(filepath="/path/to/file.dcm", pixel_data=redacted_array)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.identity_manager = AegisIdentityManager()

    def __call__(
        self,
        filepath: str,
        pixel_data: Optional[np.ndarray] = None,
        dataset: Optional[pydicom.Dataset] = None
    ) -> pydicom.Dataset:
        """
        Args:
            filepath: Path to the DICOM file (fallback if dataset is None).
            pixel_data: Optional redacted pixel array to inject.
            dataset: Optional in-memory dataset to process.

        Returns:
            Scrubbed pydicom Dataset (in-memory, not saved to disk).
        """
        if dataset is not None:
            ds = copy.deepcopy(dataset)
        else:
            ds = pydicom.dcmread(filepath)

        pii_mapping = self.config.get('pii_mapping', {})

        # 1. Metadata Scrubbing
        for tag_str, action in pii_mapping.items():
            try:
                clean_tag_str = tag_str.strip('() ').replace(' ', '')
                parts = clean_tag_str.split(',')
                tag = pydicom.tag.Tag(int(parts[0], 16), int(parts[1], 16))
            except Exception as e:
                logger.error(f"Error parsing tag {tag_str}: {e}")
                continue

            if tag in ds:
                action = action.upper()
                if action == 'REMOVE':
                    del ds[tag]
                elif action == 'ZERO':
                    vr = ds[tag].VR
                    ds[tag].value = b'' if vr in ['OB', 'OW', 'UN'] else ''
                elif action == 'DUMMY':
                    token = self.identity_manager.get_token(str(ds[tag].value))
                    ds[tag].value = token

        ds.remove_private_tags()

        # 2. Pixel Data Injection (in-memory only)
        if pixel_data is not None:
            if hasattr(ds, 'file_meta'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            contiguous_pixels = np.ascontiguousarray(pixel_data)

            # Synchronize Bit-Depth
            if contiguous_pixels.dtype == np.uint8:
                ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
                ds.PixelRepresentation = 0
            else:
                ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
                ds.PixelRepresentation = 1 if np.issubdtype(contiguous_pixels.dtype, np.signedinteger) else 0

            ds.SamplesPerPixel = 1 if contiguous_pixels.ndim == 2 else contiguous_pixels.shape[-1]
            ds.PhotometricInterpretation = "MONOCHROME2" if ds.SamplesPerPixel == 1 else "RGB"

            ds.Rows, ds.Columns = contiguous_pixels.shape[:2]
            ds.PixelData = contiguous_pixels.tobytes()

            # Fix Windowing
            p_min, p_max = float(contiguous_pixels.min()), float(contiguous_pixels.max())
            ds.WindowCenter = str((p_max + p_min) / 2)
            ds.WindowWidth = str(p_max - p_min)

        return ds


# ---------------------------------------------------------------------------
# Dictionary Transform
# ---------------------------------------------------------------------------

class ScrubDicomMetadatad(MapTransform):
    """
    Dictionary transform: Scrub PII from DICOM metadata and inject redacted pixels.

    Dictionary-based wrapper of :py:class:`ScrubDicomMetadata`.
    Only processes DICOM files (skips JPEG/PNG).

    **Thread-safe**: operates in-memory only. Use :py:class:`SaveDicomd`
    for file persistence as a separate pipeline step.

    Args:
        keys: Keys of the data dictionary to process.
        config: Configuration dict with 'pii_mapping' section.
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        transform = ScrubDicomMetadatad(keys=["image"], config=config)
        data = transform({"image": meta_tensor, "image_meta_dict": {...}})
    """

    def __init__(self, keys, config: Dict[str, Any], allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.transform = ScrubDicomMetadata(config=config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            try:
                # 1. Retrieve file path from metadata
                meta = d.get(f"{key}_meta_dict", {})
                fpath = meta.get('filename_or_obj')
                if isinstance(fpath, list):
                    fpath = fpath[0]
                if not fpath:
                    continue

                # Only handle DICOM files
                if not str(fpath).lower().endswith('.dcm'):
                    continue

                # 2. Extract pixels from MetaTensor
                tensor = d[key]
                pix = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.array(tensor)

                # Convert channel-first → DICOM format
                if pix.ndim == 3:
                    if pix.shape[0] == 1:
                        pix = pix.squeeze(0)
                    elif pix.shape[0] == 3:
                        pix = np.transpose(pix, (1, 2, 0))

                # Handle normalized float arrays using cached dataset if available
                ds_orig = d.get(f"{key}_dicom_dataset")
                if ds_orig is None:
                    ds_orig = pydicom.dcmread(fpath)
                    
                if pix.max() <= 1.1:
                    orig_max = ds_orig.get("LargestImagePixelValue", 4095)
                    pix = (pix * orig_max)

                # 3. Delegate to array transform (in-memory scrub, no save)
                scrubbed_ds = self.transform(
                    filepath=fpath,
                    pixel_data=pix.astype(ds_orig.pixel_array.dtype),
                    dataset=ds_orig
                )

                # 4. Store scrubbed dataset in dict for downstream SaveDicomd
                d[f"{key}_scrubbed_ds"] = scrubbed_ds

                # 5. Update MetaTensor with scrubbed pixels (full propagation)
                if hasattr(scrubbed_ds, 'pixel_array'):
                    scrubbed_pix = scrubbed_ds.pixel_array.astype(np.float32)
                    # Restore channel-first format
                    if scrubbed_pix.ndim == 2:
                        scrubbed_pix = scrubbed_pix[np.newaxis, ...]
                    elif scrubbed_pix.ndim == 3 and scrubbed_pix.shape[-1] == 3:
                        scrubbed_pix = np.transpose(scrubbed_pix, (2, 0, 1))

                    original = d[key]
                    if isinstance(original, MetaTensor):
                        d[key] = MetaTensor(torch.as_tensor(scrubbed_pix), meta=original.meta)
                    else:
                        d[key] = scrubbed_pix

            except Exception as e:
                logger.error(f"Error in ScrubDicomMetadatad for file {fpath}: {e}")
                raise

        return d
