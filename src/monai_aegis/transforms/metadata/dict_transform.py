"""
``ScrubDicomMetadatad`` — dictionary wrapper around :class:`ScrubDicomMetadata`.

Bridges the MONAI data dict (channel-first tensors, ``{key}_meta_dict``,
cached pydicom datasets from the loaders) and the array transform. Three
modes, selected from the cached-dataset keys and ``is_multiframe``:

==================  ==============================  ==================================
Mode                Reads                           Writes
==================  ==============================  ==================================
single              ``{key}_dicom_dataset``         ``{key}_scrubbed_ds``,
                    (or ``dcmread`` fallback);      ``{key}_tag_actions``
                    ``(C, H, W)`` or, for a
                    multi-frame file,
                    ``(C, D, H, W)``
series              ``{key}_dicom_datasets`` list   ``{key}_scrubbed_datasets``,
                                                    ``{key}_tag_actions_per_slice``
multi-frame         ``{key}_dicom_datasets[0]``     ``{key}_scrubbed_ds``,
                    + ``is_multiframe``             ``{key}_tag_actions``
==================  ==============================  ==================================

In every mode ``{key}`` itself is replaced by the scrubbed pixels as a
float32 channel-first array (a ``MetaTensor`` carrying the original
``.meta`` when the input was one, otherwise a plain ``ndarray``).

**Thread-safe**: operates in-memory only. Use :py:class:`SaveDicomd` /
:py:class:`SaveDicomSeriesd` for file persistence as a separate step.
"""
import logging
from typing import Any, Dict, Hashable, List, Mapping

import numpy as np
import pydicom
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck
from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms.metadata.constants import (
    CHANNELS_MONO,
    CHANNELS_RGB,
    DEFAULT_LARGEST_PIXEL_VALUE,
    DICT_TRANSFORM_NAME,
    GEOMETRY_KEYWORDS,
    KW_LARGEST_IMAGE_PIXEL_VALUE,
    META_FILENAME_KEY,
    META_IS_MULTIFRAME_KEY,
    META_SLICE_URIS_KEY,
    NORMALIZED_MAX,
    VOLUME_RANK,
)
from monai_aegis.transforms.metadata.scrubber import ScrubDicomMetadata

__all__ = ["ScrubDicomMetadatad"]

# Same logger name as before the package split, so existing log routing is unchanged.
logger = logging.getLogger(__package__)


# ---------------------------------------------------------------------------
# Helpers shared by the three modes
# ---------------------------------------------------------------------------

def _as_numpy(tensor: Any) -> np.ndarray:
    """Detach a torch tensor / MetaTensor to numpy; pass arrays through."""
    return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.array(tensor)


def _rescale_if_normalized(pix: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Undo [0, 1] normalisation using the source dataset's pixel range."""
    if pix.max() <= NORMALIZED_MAX:
        orig_max = ds.get(KW_LARGEST_IMAGE_PIXEL_VALUE, DEFAULT_LARGEST_PIXEL_VALUE)
        pix = pix * orig_max
    return pix


def _wrap_like(original: Any, array: np.ndarray) -> Any:
    """Return *array* as a MetaTensor (preserving ``original.meta``) or ndarray, matching *original*."""
    if isinstance(original, MetaTensor):
        return MetaTensor(torch.as_tensor(array), meta=original.meta)
    return array


# ---------------------------------------------------------------------------
# Dictionary Transform
# ---------------------------------------------------------------------------

class ScrubDicomMetadatad(MapTransform):
    """
    Dictionary transform: Scrub PII from DICOM metadata and inject redacted pixels.

    Dictionary-based wrapper of :py:class:`ScrubDicomMetadata`.
    Only processes DICOM files (skips JPEG/PNG).

    Args:
        keys: Keys of the data dictionary to process.
        config: Configuration dict with 'pii_mapping' section.
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example::

        transform = ScrubDicomMetadatad(keys=["image"], config=config)
        data = transform({"image": meta_tensor, "image_meta_dict": {...}})
    """

    # Historical name; the rule lives in ``constants``.
    _GEOMETRY_TAGS = GEOMETRY_KEYWORDS

    def __init__(
        self,
        keys: KeysCollection,
        config: Dict[str, Any],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = ScrubDicomMetadata(config=config)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """Scrub PII from the DICOM dataset(s) cached for each key.

        Keys without a file path or without a cached DICOM dataset
        (JPEG/PNG inputs) are skipped. See the module docstring for what
        each mode reads and writes.

        Raises:
            MetadataScrubError: If scrubbing fails; errors raised by the
                array transform pass through unchanged, anything else is
                wrapped with the file path.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            try:
                meta = d.get(ck(key, ckeys.META_DICT), {})
                fpath = meta.get(META_FILENAME_KEY)
                if isinstance(fpath, list):
                    fpath = fpath[0]
                if not fpath:
                    continue

                has_dataset = (ck(key, ckeys.DICOM_DATASET) in d or
                               ck(key, ckeys.DICOM_DATASETS) in d)
                if not has_dataset:
                    continue

                datasets_list = d.get(ck(key, ckeys.DICOM_DATASETS))
                if datasets_list is not None and isinstance(datasets_list, list):
                    if meta.get(META_IS_MULTIFRAME_KEY):
                        d = self._scrub_multiframe(d, key, datasets_list[0])
                        continue
                    d = self._scrub_series(d, key, datasets_list)
                    continue

                d = self._scrub_single(d, key, fpath)

            except MetadataScrubError:
                raise
            except Exception as e:
                raise MetadataScrubError(
                    f"Metadata scrubbing failed: {e}",
                    uri=str(fpath),
                    transform=DICT_TRANSFORM_NAME,
                ) from e

        return d

    # ------------------------------------------------------------------
    # Single-file mode
    # ------------------------------------------------------------------

    def _scrub_single(
        self,
        d: Dict[Hashable, Any],
        key: Hashable,
        fpath: str,
    ) -> Dict[Hashable, Any]:
        """Scrub one dataset: ``{key}_scrubbed_ds`` + ``{key}_tag_actions`` out.

        The tensor is ``(C, H, W)`` for a single frame or ``(C, D, H, W)``
        for a multi-frame file loaded by ``LoadDicomRaw`` (e.g. an
        ultrasound cine loop); either way it comes back in the same layout.
        """
        pix = _as_numpy(d[key])
        is_volume = pix.ndim == VOLUME_RANK

        # channel-first -> DICOM layout
        if pix.ndim == 3:
            if pix.shape[0] == CHANNELS_MONO:
                pix = pix.squeeze(0)
            elif pix.shape[0] == CHANNELS_RGB:
                pix = np.transpose(pix, (1, 2, 0))
        elif is_volume:
            if pix.shape[0] == CHANNELS_MONO:
                pix = pix.squeeze(0)                        # (D, H, W)
            elif pix.shape[0] == CHANNELS_RGB:
                pix = np.transpose(pix, (1, 2, 3, 0))       # (D, H, W, 3)
            else:
                raise MetadataScrubError(
                    f"Unsupported channel layout for multi-frame scrub: {pix.shape}",
                    transform=DICT_TRANSFORM_NAME,
                )

        ds_orig = d.get(ck(key, ckeys.DICOM_DATASET))
        if ds_orig is None:
            ds_orig = pydicom.dcmread(fpath)

        pix = _rescale_if_normalized(pix, ds_orig)

        scrubbed_ds, tag_actions = self.transform.scrub(
            uri=fpath,
            pixel_data=pix.astype(ds_orig.pixel_array.dtype),
            dataset=ds_orig
        )

        d[ck(key, ckeys.SCRUBBED_DS)] = scrubbed_ds
        d[ck(key, ckeys.TAG_ACTIONS)] = tag_actions

        # DICOM layout -> channel-first
        if hasattr(scrubbed_ds, 'pixel_array'):
            scrubbed_pix = scrubbed_ds.pixel_array.astype(np.float32)
            if is_volume:
                if scrubbed_pix.ndim == 3:                  # (D, H, W)
                    scrubbed_pix = scrubbed_pix[np.newaxis, ...]
                elif scrubbed_pix.ndim == 4 and scrubbed_pix.shape[-1] == CHANNELS_RGB:
                    scrubbed_pix = np.transpose(scrubbed_pix, (3, 0, 1, 2))
            elif scrubbed_pix.ndim == 2:
                scrubbed_pix = scrubbed_pix[np.newaxis, ...]
            elif scrubbed_pix.ndim == 3 and scrubbed_pix.shape[-1] == CHANNELS_RGB:
                scrubbed_pix = np.transpose(scrubbed_pix, (2, 0, 1))

            d[key] = _wrap_like(d[key], scrubbed_pix)

        return d

    # ------------------------------------------------------------------
    # Multi-frame mode
    # ------------------------------------------------------------------

    def _scrub_multiframe(
        self,
        d: Dict[Hashable, Any],
        key: Hashable,
        dataset: pydicom.Dataset,
    ) -> Dict[Hashable, Any]:
        """Scrub one multi-frame dataset: ``(C, D, H, W)`` in, ``{key}_scrubbed_ds`` + ``{key}_tag_actions`` out."""
        volume = _as_numpy(d[key])

        if volume.ndim != VOLUME_RANK:
            raise MetadataScrubError(
                f"Expected 4D volume for multi-frame scrub, got {volume.ndim}D",
                transform=DICT_TRANSFORM_NAME,
            )

        # channel-first -> DICOM layout
        if volume.shape[0] == CHANNELS_MONO:
            pixel_data = volume.squeeze(0)
        elif volume.shape[0] == CHANNELS_RGB:
            pixel_data = np.transpose(volume, (1, 2, 3, 0))
        else:
            raise MetadataScrubError(
                f"Unsupported channel layout for multi-frame scrub: {volume.shape}",
                transform=DICT_TRANSFORM_NAME,
            )

        pixel_data = _rescale_if_normalized(pixel_data, dataset)

        fpath = d.get(ck(key, ckeys.META_DICT), {}).get(META_FILENAME_KEY, "")
        scrubbed_ds, tag_actions = self.transform.scrub(
            uri=str(fpath),
            pixel_data=pixel_data.astype(dataset.pixel_array.dtype),
            dataset=dataset,
        )
        d[ck(key, ckeys.SCRUBBED_DS)] = scrubbed_ds
        d[ck(key, ckeys.TAG_ACTIONS)] = tag_actions

        # DICOM layout -> channel-first
        scrubbed_pix = scrubbed_ds.pixel_array.astype(np.float32)
        if scrubbed_pix.ndim == 3:
            scrubbed_pix = scrubbed_pix[np.newaxis, ...]
        elif scrubbed_pix.ndim == 4 and scrubbed_pix.shape[-1] == CHANNELS_RGB:
            scrubbed_pix = np.transpose(scrubbed_pix, (3, 0, 1, 2))

        d[key] = _wrap_like(d[key], scrubbed_pix)
        return d

    # ------------------------------------------------------------------
    # Series mode
    # ------------------------------------------------------------------

    def _scrub_series(
        self,
        d: Dict[Hashable, Any],
        key: Hashable,
        datasets: List[pydicom.Dataset],
    ) -> Dict[Hashable, Any]:
        """Scrub a slice list: ``(C, D, H, W)`` in, ``{key}_scrubbed_datasets`` + ``{key}_tag_actions_per_slice`` out.

        Per slice: extract the 2-D pixels, scrub with consistent tokenization
        and UID remapping, then restore the geometry tags unchanged.
        """
        volume = _as_numpy(d[key])

        if volume.ndim != VOLUME_RANK:
            logger.warning("Expected 4D volume for series scrub, got %dD", volume.ndim)
            fpath = d.get(ck(key, ckeys.META_DICT), {}).get(META_FILENAME_KEY, '')
            return self._scrub_single(d, key, str(fpath))

        C, D, H, W = volume.shape
        scrubbed_datasets = []
        tag_actions_per_slice = []

        for i, ds_orig in enumerate(datasets):
            # channel-first -> DICOM layout
            slice_pix = volume[:, i, :, :]
            if slice_pix.shape[0] == CHANNELS_MONO:
                slice_pix = slice_pix.squeeze(0)
            elif slice_pix.shape[0] == CHANNELS_RGB:
                slice_pix = np.transpose(slice_pix, (1, 2, 0))

            slice_pix = _rescale_if_normalized(slice_pix, ds_orig)

            fpath_i = d.get(ck(key, ckeys.META_DICT), {}).get(META_SLICE_URIS_KEY, [''])
            fp = fpath_i[i] if i < len(fpath_i) else fpath_i[0]

            saved_geometry = {}
            for tag_name in self._GEOMETRY_TAGS:
                if hasattr(ds_orig, tag_name):
                    saved_geometry[tag_name] = getattr(ds_orig, tag_name)

            scrubbed_ds, slice_tag_actions = self.transform.scrub(
                uri=fp,
                pixel_data=slice_pix.astype(ds_orig.pixel_array.dtype),
                dataset=ds_orig
            )

            for tag_name, val in saved_geometry.items():
                setattr(scrubbed_ds, tag_name, val)

            scrubbed_datasets.append(scrubbed_ds)
            tag_actions_per_slice.append(slice_tag_actions)

        d[ck(key, ckeys.SCRUBBED_DATASETS)] = scrubbed_datasets
        d[ck(key, ckeys.TAG_ACTIONS_PER_SLICE)] = tag_actions_per_slice

        # DICOM layout -> channel-first, written back into a copy of the volume
        scrubbed_volume = volume.copy()
        for i, sds in enumerate(scrubbed_datasets):
            if hasattr(sds, 'pixel_array'):
                sp = sds.pixel_array.astype(np.float32)
                if sp.ndim == 2:
                    scrubbed_volume[0, i, :, :] = sp
                elif sp.ndim == 3 and sp.shape[-1] == CHANNELS_RGB:
                    scrubbed_volume[:, i, :, :] = np.transpose(sp, (2, 0, 1))

        d[key] = _wrap_like(d[key], scrubbed_volume)

        logger.info("Scrubbed %d slices in series", D)
        return d
