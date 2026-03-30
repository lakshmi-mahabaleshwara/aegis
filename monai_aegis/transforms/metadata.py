"""
MONAI Aegis Metadata Transforms — ScrubDicomMetadata / ScrubDicomMetadatad

DICOM metadata de-identification with configurable PII actions
(REMOVE, ZERO, DUMMY) and pixel data injection.

Thread-safe: no file I/O in ``__call__()`` — saving is handled by
:py:class:`SaveDicomd`.

Raises:
    MetadataScrubError: When tag parsing, pixel injection, or dataset
        manipulation fails during scrubbing.
"""
import numpy as np
import pydicom
import pydicom.uid
import copy
import torch
import logging
from typing import Dict, Hashable, Mapping, Any, Optional
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform
from monai.data import MetaTensor
from monai.utils.enums import TransformBackends

from monai_aegis.transforms.utility import AegisIdentityManager
from monai_aegis.transforms.exceptions import MetadataScrubError

__all__ = ["ScrubDicomMetadata", "ScrubDicomMetadatad"]

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
        scrubbed_ds = transform(uri="/path/to/file.dcm", pixel_data=redacted_array)
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.identity_manager = AegisIdentityManager.from_config(config)
        self._pii_actions = self._build_pii_actions(config.get('pii_mapping', {}))

    @staticmethod
    def _parse_hex_part(s: str) -> int:
        """Parse one DICOM tag group/element string as a hex integer.

        Accepts both plain hex (``0010``) and explicit ``0x``-prefixed hex
        (``0x0010``).  Plain digits are always interpreted as base-16 via
        ``int(s, 16)`` — never as decimal or octal — so leading zeros are
        harmless and no ``SyntaxError`` can arise from the string value.
        """
        s = s.strip()
        if s.lower().startswith('0x'):
            return int(s, 0)   # 0x-prefix: let Python auto-detect (always 16)
        return int(s, 16)      # plain hex digits: explicit base-16

    @staticmethod
    def _parse_tag_key(tag_str: str) -> tuple:
        """Parse a DICOM tag string key into a ``(group, element)`` int tuple.

        Accepts any of these formats from ``pii_mapping`` in ``config.yaml``:

        * ``'(0010,0010)'``   — preferred, standard DICOM notation (no space)
        * ``'(0010, 0010)'``  — with space after comma (legacy, still accepted)
        * ``'(0x0010,0x0010)'`` — explicit hex prefix (also accepted)

        Args:
            tag_str: DICOM tag as a string key from ``pii_mapping``.

        Returns:
            ``(group, element)`` as a tuple of ints.

        Raises:
            ValueError: If ``tag_str`` does not contain exactly two
                comma-separated parts.
        """
        inner = tag_str.strip().strip('()')
        parts = inner.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid DICOM tag format: {tag_str!r}")
        return (
            ScrubDicomMetadata._parse_hex_part(parts[0]),
            ScrubDicomMetadata._parse_hex_part(parts[1]),
        )

    @staticmethod
    def _build_pii_actions(pii_mapping: Dict[str, Any]) -> Dict[pydicom.tag.BaseTag, str]:
        """Parse configured tag actions once so recursive scrubbing can reuse them."""
        actions: Dict[pydicom.tag.BaseTag, str] = {}
        for tag_str, action in pii_mapping.items():
            try:
                group, element = ScrubDicomMetadata._parse_tag_key(tag_str)
                tag = pydicom.tag.Tag(group, element)
            except Exception as e:
                logger.error("Error parsing tag %s: %s", tag_str, e)
                continue
            actions[tag] = str(action).upper()
        return actions

    def _apply_action(self, ds: pydicom.Dataset, tag: pydicom.tag.BaseTag, action: str) -> None:
        """Apply one configured action to a single element on a dataset."""
        if tag not in ds:
            return
        if action == 'REMOVE':
            del ds[tag]
        elif action == 'ZERO':
            vr = ds[tag].VR
            ds[tag].value = b'' if vr in ['OB', 'OW', 'UN'] else ''
        elif action == 'DUMMY':
            token = self.identity_manager.get_token(str(ds[tag].value))
            ds[tag].value = token

    def _scrub_dataset_recursive(self, ds: pydicom.Dataset) -> None:
        """Apply configured actions to this dataset and every nested sequence item."""
        for elem in list(ds):
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        self._scrub_dataset_recursive(item)
            action = self._pii_actions.get(elem.tag)
            if action is not None:
                self._apply_action(ds, elem.tag, action)

    def _remove_private_tags_recursive(self, ds: pydicom.Dataset) -> None:
        """Remove private tags from this dataset and nested sequence items."""
        ds.remove_private_tags()
        for elem in ds:
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        self._remove_private_tags_recursive(item)

    def __call__(
        self,
        uri: str,
        pixel_data: Optional[np.ndarray] = None,
        dataset: Optional[pydicom.Dataset] = None
    ) -> pydicom.Dataset:
        """
        Args:
            uri: Path to the DICOM file (fallback if dataset is None).
            pixel_data: Optional redacted pixel array to inject.
            dataset: Optional in-memory dataset to process.

        Returns:
            Scrubbed pydicom Dataset (in-memory, not saved to disk).

        Raises:
            MetadataScrubError: If tag parsing or pixel injection fails
                (raised by the calling dictionary transform).
        """
        if dataset is not None:
            ds = copy.deepcopy(dataset)
        else:
            ds = pydicom.dcmread(uri)

        # 1. Metadata Scrubbing
        self._scrub_dataset_recursive(ds)
        self._remove_private_tags_recursive(ds)

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

            if contiguous_pixels.ndim == 2:
                ds.SamplesPerPixel = 1
                ds.Rows, ds.Columns = contiguous_pixels.shape
                if hasattr(ds, "NumberOfFrames"):
                    del ds.NumberOfFrames
            elif contiguous_pixels.ndim == 3:
                if contiguous_pixels.shape[-1] in (3, 4):
                    ds.SamplesPerPixel = 3
                    ds.Rows, ds.Columns = contiguous_pixels.shape[:2]
                    contiguous_pixels = contiguous_pixels[..., :3]
                    ds.PlanarConfiguration = 0
                    if hasattr(ds, "NumberOfFrames"):
                        del ds.NumberOfFrames
                else:
                    ds.NumberOfFrames = str(contiguous_pixels.shape[0])
                    ds.SamplesPerPixel = 1
                    ds.Rows, ds.Columns = contiguous_pixels.shape[1:]
            elif contiguous_pixels.ndim == 4 and contiguous_pixels.shape[-1] in (3, 4):
                ds.NumberOfFrames = str(contiguous_pixels.shape[0])
                ds.SamplesPerPixel = 3
                ds.Rows, ds.Columns = contiguous_pixels.shape[1:3]
                contiguous_pixels = contiguous_pixels[..., :3]
                ds.PlanarConfiguration = 0
            else:
                raise MetadataScrubError(
                    f"Unsupported pixel array shape for DICOM injection: {contiguous_pixels.shape}",
                    transform="ScrubDicomMetadata",
                )

            ds.PhotometricInterpretation = "MONOCHROME2" if ds.SamplesPerPixel == 1 else "RGB"
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

    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = ScrubDicomMetadata(config=config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Scrub PII from DICOM datasets in the data dict.

        Supports two modes:
          - **Single-file** — reads ``{key}_dicom_dataset`` (single Dataset).
          - **Series** — reads ``{key}_dicom_datasets`` (list of Datasets),
            scrubs each with consistent tokenization, preserves geometry tags,
            and regenerates ``SOPInstanceUID`` per slice.

        Side-effects written per key:
            - ``{key}_scrubbed_ds`` — single scrubbed Dataset (single mode).
            - ``{key}_scrubbed_datasets`` — list of scrubbed Datasets (series mode).
            - ``{key}`` — MetaTensor updated with scrubbed pixel data.

        Args:
            data: Pipeline data dictionary containing MetaTensors and
                cached dataset(s).

        Returns:
            Updated data dictionary with scrubbed datasets.

        Raises:
            MetadataScrubError: If tag parsing, pixel injection, or
                dataset manipulation fails.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            try:
                meta = d.get(f"{key}_meta_dict", {})
                fpath = meta.get('filename_or_obj')
                if isinstance(fpath, list):
                    fpath = fpath[0]
                if not fpath:
                    continue

                # Only handle files that were loaded as DICOM (content-driven)
                has_dataset = (f"{key}_dicom_dataset" in d or
                               f"{key}_dicom_datasets" in d)
                if not has_dataset:
                    continue

                # ---- Series mode: list of datasets ----
                datasets_list = d.get(f"{key}_dicom_datasets")
                if datasets_list is not None and isinstance(datasets_list, list):
                    if meta.get("is_multiframe"):
                        d = self._scrub_multiframe(d, key, datasets_list[0])
                        continue
                    d = self._scrub_series(d, key, datasets_list)
                    continue

                # ---- Single-file mode (existing path) ----
                d = self._scrub_single(d, key, fpath)

            except MetadataScrubError:
                raise
            except Exception as e:
                raise MetadataScrubError(
                    f"Metadata scrubbing failed: {e}",
                    uri=str(fpath),
                    transform="ScrubDicomMetadatad",
                ) from e

        return d

    def _scrub_multiframe(
        self,
        d: Dict[Hashable, Any],
        key: Hashable,
        dataset: pydicom.Dataset,
    ) -> Dict[Hashable, Any]:
        """Scrub a single multi-frame DICOM while preserving frame structure."""
        tensor = d[key]
        volume = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.array(tensor)

        if volume.ndim != 4:
            raise MetadataScrubError(
                f"Expected 4D volume for multi-frame scrub, got {volume.ndim}D",
                transform="ScrubDicomMetadatad",
            )

        if volume.shape[0] == 1:
            pixel_data = volume.squeeze(0)
        elif volume.shape[0] == 3:
            pixel_data = np.transpose(volume, (1, 2, 3, 0))
        else:
            raise MetadataScrubError(
                f"Unsupported channel layout for multi-frame scrub: {volume.shape}",
                transform="ScrubDicomMetadatad",
            )

        if pixel_data.max() <= 1.1:
            orig_max = dataset.get("LargestImagePixelValue", 4095)
            pixel_data = pixel_data * orig_max

        fpath = d.get(f"{key}_meta_dict", {}).get("filename_or_obj", "")
        scrubbed_ds = self.transform(
            uri=str(fpath),
            pixel_data=pixel_data.astype(dataset.pixel_array.dtype),
            dataset=dataset,
        )
        scrubbed_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        d[f"{key}_scrubbed_ds"] = scrubbed_ds

        scrubbed_pix = scrubbed_ds.pixel_array.astype(np.float32)
        if scrubbed_pix.ndim == 3:
            scrubbed_pix = scrubbed_pix[np.newaxis, ...]
        elif scrubbed_pix.ndim == 4 and scrubbed_pix.shape[-1] == 3:
            scrubbed_pix = np.transpose(scrubbed_pix, (3, 0, 1, 2))

        original = d[key]
        if isinstance(original, MetaTensor):
            d[key] = MetaTensor(torch.as_tensor(scrubbed_pix), meta=original.meta)
        else:
            d[key] = scrubbed_pix

        return d

    def _scrub_single(
        self,
        d: Dict[Hashable, Any],
        key: Hashable,
        fpath: str,
    ) -> Dict[Hashable, Any]:
        """Scrub a single DICOM dataset (existing pipeline path)."""
        tensor = d[key]
        pix = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.array(tensor)

        # Convert channel-first → DICOM format
        if pix.ndim == 3:
            if pix.shape[0] == 1:
                pix = pix.squeeze(0)
            elif pix.shape[0] == 3:
                pix = np.transpose(pix, (1, 2, 0))

        # Handle normalized float arrays using cached dataset
        ds_orig = d.get(f"{key}_dicom_dataset")
        if ds_orig is None:
            ds_orig = pydicom.dcmread(fpath)

        if pix.max() <= 1.1:
            orig_max = ds_orig.get("LargestImagePixelValue", 4095)
            pix = (pix * orig_max)

        scrubbed_ds = self.transform(
            uri=fpath,
            pixel_data=pix.astype(ds_orig.pixel_array.dtype),
            dataset=ds_orig
        )

        d[f"{key}_scrubbed_ds"] = scrubbed_ds

        # Update MetaTensor with scrubbed pixels
        if hasattr(scrubbed_ds, 'pixel_array'):
            scrubbed_pix = scrubbed_ds.pixel_array.astype(np.float32)
            if scrubbed_pix.ndim == 2:
                scrubbed_pix = scrubbed_pix[np.newaxis, ...]
            elif scrubbed_pix.ndim == 3 and scrubbed_pix.shape[-1] == 3:
                scrubbed_pix = np.transpose(scrubbed_pix, (2, 0, 1))

            original = d[key]
            if isinstance(original, MetaTensor):
                d[key] = MetaTensor(torch.as_tensor(scrubbed_pix), meta=original.meta)
            else:
                d[key] = scrubbed_pix

        return d

    # Geometry tags that must NEVER be modified during series scrubbing
    _GEOMETRY_TAGS = frozenset([
        'PixelSpacing', 'SliceThickness', 'ImageOrientationPatient',
        'ImagePositionPatient', 'SpacingBetweenSlices',
        'ImagePositionPatient',
    ])

    def _scrub_series(
        self,
        d: Dict[Hashable, Any],
        key: Hashable,
        datasets: list,
    ) -> Dict[Hashable, Any]:
        """Scrub a list of DICOM datasets (series mode).

        For each slice:
          1. Extract 2D pixel data from the volume tensor.
          2. Scrub metadata with consistent tokenization.
          3. Preserve geometry tags unchanged.
          4. Regenerate ``SOPInstanceUID``.
        """
        tensor = d[key]
        volume = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.array(tensor)

        # volume shape: (C, D, H, W) from LoadDicomSeriesd
        if volume.ndim != 4:
            logger.warning("Expected 4D volume for series scrub, got %dD", volume.ndim)
            # Fallback: treat as single
            fpath = d.get(f"{key}_meta_dict", {}).get('filename_or_obj', '')
            return self._scrub_single(d, key, str(fpath))

        C, D, H, W = volume.shape
        scrubbed_datasets = []

        for i, ds_orig in enumerate(datasets):
            # Extract 2D slice pixels: (C, H, W) → DICOM format
            slice_pix = volume[:, i, :, :]
            if slice_pix.shape[0] == 1:
                slice_pix = slice_pix.squeeze(0)
            elif slice_pix.shape[0] == 3:
                slice_pix = np.transpose(slice_pix, (1, 2, 0))

            # Handle normalized floats
            if slice_pix.max() <= 1.1:
                orig_max = ds_orig.get("LargestImagePixelValue", 4095)
                slice_pix = (slice_pix * orig_max)

            fpath_i = d.get(f"{key}_meta_dict", {}).get(
                'slice_uris', ['']
            )
            fp = fpath_i[i] if i < len(fpath_i) else fpath_i[0]

            # Save geometry before scrubbing
            saved_geometry = {}
            for tag_name in self._GEOMETRY_TAGS:
                if hasattr(ds_orig, tag_name):
                    saved_geometry[tag_name] = getattr(ds_orig, tag_name)

            # Scrub metadata + inject redacted pixels
            scrubbed_ds = self.transform(
                uri=fp,
                pixel_data=slice_pix.astype(ds_orig.pixel_array.dtype),
                dataset=ds_orig
            )

            # Restore geometry tags
            for tag_name, val in saved_geometry.items():
                setattr(scrubbed_ds, tag_name, val)

            # Regenerate SOPInstanceUID for uniqueness
            scrubbed_ds.SOPInstanceUID = pydicom.uid.generate_uid()

            scrubbed_datasets.append(scrubbed_ds)

        # Store scrubbed list for downstream SaveDicomSeriesd
        d[f"{key}_scrubbed_datasets"] = scrubbed_datasets

        # Update volume tensor with scrubbed pixels
        scrubbed_volume = volume.copy()
        for i, sds in enumerate(scrubbed_datasets):
            if hasattr(sds, 'pixel_array'):
                sp = sds.pixel_array.astype(np.float32)
                if sp.ndim == 2:
                    scrubbed_volume[0, i, :, :] = sp
                elif sp.ndim == 3 and sp.shape[-1] == 3:
                    scrubbed_volume[:, i, :, :] = np.transpose(sp, (2, 0, 1))

        original = d[key]
        if isinstance(original, MetaTensor):
            d[key] = MetaTensor(torch.as_tensor(scrubbed_volume), meta=original.meta)
        else:
            d[key] = scrubbed_volume

        logger.info("Scrubbed %d slices in series", D)
        return d
