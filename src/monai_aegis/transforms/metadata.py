"""
MONAI Aegis Metadata Transforms — ScrubDicomMetadata / ScrubDicomMetadatad

DICOM metadata de-identification with configurable PII actions
(REMOVE, ZERO, DUMMY, KEEP) and pixel data injection. Every output
dataset is stamped with the PS3.15 de-identification attestation
attributes (PatientIdentityRemoved, DeidentificationMethod[CodeSequence],
BurnedInAnnotation).

Thread-safe: no file I/O in ``__call__()`` — saving is handled by
:py:class:`SaveDicomd`.

Raises:
    MetadataScrubError: When tag parsing, pixel injection, or dataset
        manipulation fails during scrubbing.
"""
import numpy as np
import pydicom
import pydicom.uid
import pydicom.tag
import pydicom.datadict
import copy
import torch
import logging
from typing import Dict, Hashable, Mapping, Any, Optional
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform
from monai.data import MetaTensor
from monai.utils.enums import TransformBackends

from monai_aegis import __version__ as _AEGIS_VERSION
from monai_aegis.transforms.utility import AegisIdentityManager
from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck

__all__ = ["ScrubDicomMetadata", "ScrubDicomMetadatad"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Array Transform
# ---------------------------------------------------------------------------

class ScrubDicomMetadata(Transform):
    """
    Array transform: Scrub PII from a DICOM dataset and inject redacted pixels.

    Applies configurable PII actions (REMOVE, ZERO, DUMMY, KEEP) to DICOM
    tags, removes private tags, injects redacted pixel data with correct
    bit-depth, and stamps the PS3.15 attestation attributes on the output.

    This transform is **pure** — it operates in-memory and does NOT write to disk.
    Use :py:class:`SaveDicom` / :py:class:`SaveDicomd` for persistence.

    Args:
        config: Configuration dict with 'pii_mapping' section.

    Example::

        transform = ScrubDicomMetadata(config=config)
        scrubbed_ds = transform(uri="/path/to/file.dcm", pixel_data=redacted_array)
    """

    backend = [TransformBackends.NUMPY]

    # DICOM-standard / well-known UID root.  UIDs under this root identify
    # object *types* (SOP Class, Transfer Syntax, coding schemes, ...) — never
    # patients — so they are preserved verbatim.  Anything under an org root is
    # an instance/study/series/frame identifier and gets re-mapped.
    _DICOM_STANDARD_UID_ROOT = "1.2.840.10008"

    # UID keywords that name an object *class* or the implementation rather
    # than a specific instance, preserved even when they use a non-standard
    # root (otherwise the output would no longer be a valid DICOM object).
    _PRESERVE_UID_KEYWORDS = frozenset([
        "SOPClassUID",
        "MediaStorageSOPClassUID",
        "TransferSyntaxUID",
        "ImplementationClassUID",
        "ReferencedSOPClassUID",
    ])

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.identity_manager = AegisIdentityManager.from_config(config)
        self._pii_actions = self._build_pii_actions(config.get('pii_mapping', {}))
        # Deterministic UID remapping is keyed off the same secret salt as
        # identity tokenization so the mapping is reproducible across slices,
        # files, and runs (required to keep Study/Series links consistent).
        self._uid_salt = self.identity_manager.salt
        # Shared across threads by design (consistent UID remapping across
        # slices); dict get/set are atomic under the GIL and a lost race
        # only costs a recomputed hash, never an inconsistent mapping.
        self._uid_cache: Dict[str, str] = {}

    def remap_uid(self, original_uid: str, fallback_entropy: str = "") -> str:
        """Deterministically remap a DICOM UID from the tokenization salt.

        Same original UID + same salt → same de-identified UID, across
        slices, files, and runs — the re-linkage contract that identity
        tokens already honor, extended to UIDs. When the source carries no
        UID, *fallback_entropy* (e.g. the file path) keys the mapping; with
        neither, a random UID preserves uniqueness at the cost of
        repeat-run stability for that degenerate input.
        """
        source = str(original_uid or "").strip() or str(fallback_entropy or "").strip()
        if not source:
            return pydicom.uid.generate_uid()
        cached = self._uid_cache.get(source)
        if cached is None:
            cached = pydicom.uid.generate_uid(entropy_srcs=[self._uid_salt or "", source])
            self._uid_cache[source] = cached
        return cached

    def _remap_uid_value(self, value: Any, fallback_entropy: str = "") -> Any:
        """Remap one UID value, preserving class- and standard-defined UIDs.

        UIDs under the DICOM standard root identify object *types* (SOP
        Class, Transfer Syntax, coding schemes) — never patients — so they
        pass through verbatim; rewriting them would make the object invalid.
        Empty values are left untouched. Everything else is a
        patient-linkable instance/study/series/frame identifier and is
        remapped deterministically.
        """
        source = str(value or "").strip()
        if not source or source.startswith(self._DICOM_STANDARD_UID_ROOT):
            return value
        return self.remap_uid(source, fallback_entropy=fallback_entropy)

    def _remap_uids_recursive(
        self,
        ds: pydicom.Dataset,
        tag_actions: list,
        fallback_entropy: str = "",
    ) -> None:
        """Remap every patient-linkable UID on a dataset and nested items.

        Walks all ``UI``-valued elements (recursing into sequences) and
        remaps them via :meth:`_remap_uid_value`, except those whose keyword
        names an object class or the implementation
        (:attr:`_PRESERVE_UID_KEYWORDS`). Because :meth:`remap_uid` is
        deterministic and cached, a UID that appears in several places — a
        SeriesInstanceUID shared across slices, or a ReferencedSOPInstanceUID
        pointing at another instance in the same object — is rewritten to the
        *same* replacement everywhere, so cross-references stay intact.
        Each rewrite is recorded in the per-call audit trail.
        """
        for elem in ds:
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        self._remap_uids_recursive(item, tag_actions, fallback_entropy)
                continue
            if elem.VR != "UI" or elem.value is None:
                continue
            keyword = pydicom.datadict.keyword_for_tag(elem.tag) or ''
            if keyword in self._PRESERVE_UID_KEYWORDS:
                continue
            # A UID element is either a single value (pydicom.uid.UID, a str
            # subclass) or a MultiValue of them; handle both uniformly.
            if isinstance(elem.value, str):
                new_value = self._remap_uid_value(elem.value, fallback_entropy)
                changed = str(new_value) != str(elem.value)
            else:
                originals = list(elem.value)
                new_value = [self._remap_uid_value(v, fallback_entropy) for v in originals]
                changed = any(str(a) != str(b) for a, b in zip(new_value, originals))
            if not changed:
                continue
            elem.value = new_value
            tag_actions.append({
                'tag': '({:04X},{:04X})'.format(elem.tag.group, elem.tag.element),
                'keyword': keyword,
                'action': 'REMAP',
                'redacted': True,
            })

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

    # The complete action vocabulary (KEEP = deliberate retention, audited
    # with redacted=False). Validated at construction time — a typo'd
    # action must fail the run before any file is touched, never silently
    # leave PHI in place at scrub time.
    _VALID_ACTIONS = frozenset({'REMOVE', 'ZERO', 'DUMMY', 'KEEP'})

    @staticmethod
    def _build_pii_actions(pii_mapping: Dict[str, Any]) -> Dict[pydicom.tag.BaseTag, str]:
        """Parse configured tag actions once so recursive scrubbing can reuse them.

        Raises:
            ValueError: If a tag key cannot be parsed or an action is not
                one of REMOVE / ZERO / DUMMY. Misconfiguration fails the
                pipeline at construction rather than silently skipping the
                rule (which would pass PHI through unscrubbed).
        """
        actions: Dict[pydicom.tag.BaseTag, str] = {}
        for tag_str, action in pii_mapping.items():
            try:
                group, element = ScrubDicomMetadata._parse_tag_key(tag_str)
                tag = pydicom.tag.Tag(group, element)
            except Exception as e:
                raise ValueError(
                    f"Invalid DICOM tag {tag_str!r} in pii_mapping: {e}"
                ) from e
            action_upper = str(action).upper()
            if action_upper not in ScrubDicomMetadata._VALID_ACTIONS:
                raise ValueError(
                    f"Unknown pii_mapping action {action!r} for tag {tag_str} — "
                    f"valid actions: {sorted(ScrubDicomMetadata._VALID_ACTIONS)}. "
                    "Refusing to run: a mistyped action would leave this tag's "
                    "PHI in place while the audit reported it as redacted."
                )
            actions[tag] = action_upper
        return actions

    def _apply_action(
        self,
        ds: pydicom.Dataset,
        tag: pydicom.tag.BaseTag,
        action: str,
        tag_actions: list,
    ) -> None:
        """Apply one configured action to a single element on a dataset.

        Appends the audit record to *tag_actions* (owned by the current
        :meth:`scrub` call) rather than instance state, so concurrent
        calls cannot interleave their audit trails.
        """
        if tag not in ds:
            return
        keyword = pydicom.datadict.keyword_for_tag(tag) or ''
        if action == 'REMOVE':
            del ds[tag]
        elif action == 'ZERO':
            vr = ds[tag].VR
            ds[tag].value = b'' if vr in ['OB', 'OW', 'UN'] else ''
        elif action == 'DUMMY':
            token = self.identity_manager.get_token(str(ds[tag].value))
            ds[tag].value = token
        elif action == 'KEEP':
            pass  # deliberate retention — value untouched, recorded below
        else:
            # Unreachable when actions come from _build_pii_actions (which
            # validates), but guards any future caller: an unknown action
            # must never fall through and record a redaction that did not
            # happen.
            raise MetadataScrubError(
                f"Unhandled pii_mapping action {action!r} for tag {tag}",
                transform="ScrubDicomMetadata",
            )
        # Record the applied action for ground-truth reporting. REMOVE /
        # ZERO / DUMMY de-identify the element; KEEP retains it by design.
        tag_actions.append({
            'tag': '({:04X},{:04X})'.format(tag.group, tag.element),
            'keyword': keyword,
            'action': action,
            'redacted': action != 'KEEP',
        })

    def _stamp_attestation(
        self,
        ds: pydicom.Dataset,
        pixel_redacted: bool,
        tag_actions: list,
    ) -> None:
        """Stamp the PS3.15 de-identification attestation attributes.

        Writes the machine-checkable "this object was de-identified"
        markers that receivers (archives, PACS, downstream de-id audits)
        key on:

          - (0012,0062) PatientIdentityRemoved = YES
          - (0012,0063) DeidentificationMethod (multi-valued LO)
          - (0012,0064) DeidentificationMethodCodeSequence — DCM 113100
            (Basic Application Confidentiality Profile), plus DCM 113101
            (Clean Pixel Data Option) when pixel redaction ran
          - (0028,0301) BurnedInAnnotation = NO — only when pixel
            redaction actually ran on this instance

        Called after the scrub pass so the configured tag actions can
        never strip the attestation itself.
        """
        ds.PatientIdentityRemoved = 'YES'
        # Multi-valued LO: each value must stay within the 64-char VR limit.
        method = [f'MONAI Aegis {_AEGIS_VERSION}', 'PS3.15 header scrub']
        codes = [('113100', 'Basic Application Confidentiality Profile')]
        if pixel_redacted:
            method.append('Pixel PHI redaction (OCR/NER)')
            codes.append(('113101', 'Clean Pixel Data Option'))
        ds.DeidentificationMethod = method
        seq = []
        for value, meaning in codes:
            item = pydicom.Dataset()
            item.CodeValue = value
            item.CodingSchemeDesignator = 'DCM'
            item.CodeMeaning = meaning
            seq.append(item)
        ds.DeidentificationMethodCodeSequence = seq

        stamped = [
            ('(0012,0062)', 'PatientIdentityRemoved'),
            ('(0012,0063)', 'DeidentificationMethod'),
            ('(0012,0064)', 'DeidentificationMethodCodeSequence'),
        ]
        if pixel_redacted:
            ds.BurnedInAnnotation = 'NO'
            stamped.append(('(0028,0301)', 'BurnedInAnnotation'))
        for tag, keyword in stamped:
            tag_actions.append({
                'tag': tag,
                'keyword': keyword,
                'action': 'ATTEST',
                'redacted': False,
            })

    def _scrub_dataset_recursive(self, ds: pydicom.Dataset, tag_actions: list) -> None:
        """Apply configured actions to this dataset and every nested sequence item."""
        for elem in list(ds):
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        self._scrub_dataset_recursive(item, tag_actions)
            action = self._pii_actions.get(elem.tag)
            if action is not None:
                self._apply_action(ds, elem.tag, action, tag_actions)

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
        """Scrub a dataset, discarding the audit record.

        Standard array-transform contract. Callers that need the per-tag
        audit trail should use :meth:`scrub`, which returns
        ``(dataset, tag_actions)``.
        """
        ds, _ = self.scrub(uri, pixel_data=pixel_data, dataset=dataset)
        return ds

    def scrub(
        self,
        uri: str,
        pixel_data: Optional[np.ndarray] = None,
        dataset: Optional[pydicom.Dataset] = None
    ) -> tuple:
        """Scrub PII and return the dataset together with its audit trail.

        Stateless with respect to the transform instance — the audit
        record flows back as a return value, never through instance
        attributes, so a single instance is safe to share across threads.

        Args:
            uri: Path to the DICOM file (fallback if dataset is None).
            pixel_data: Optional redacted pixel array to inject.
            dataset: Optional in-memory dataset to process.

        Returns:
            Tuple of ``(scrubbed_dataset, tag_actions)`` where
            ``tag_actions`` is a list of per-tag audit dicts
            (``{'tag', 'keyword', 'action', 'redacted'}``). Includes
            ``action='ATTEST'`` rows (``redacted=False``) for the PS3.15
            attestation attributes stamped on the output.

        Raises:
            MetadataScrubError: If tag parsing or pixel injection fails
                (raised by the calling dictionary transform).
        """
        if dataset is not None:
            ds = copy.deepcopy(dataset)
        else:
            ds = pydicom.dcmread(uri)

        # Captured before scrubbing: keys the deterministic UID remap.
        original_sop_uid = str(getattr(ds, 'SOPInstanceUID', ''))

        # Per-call audit trail, owned by this invocation.
        tag_actions: list = []

        # 1. Metadata Scrubbing
        self._scrub_dataset_recursive(ds, tag_actions)
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

        # 3. Remap every patient-linkable UID (PS3.15 action U): SOP / Study /
        #    Series / FrameOfReference instance UIDs and any UID referenced in
        #    nested sequences, all replaced with new, internally consistent
        #    values. Class- and implementation-identifying UIDs (SOP Class,
        #    Transfer Syntax, ...) and anything under the DICOM standard root
        #    are preserved so the output stays a valid DICOM object. The remap
        #    is deterministic, so every slice sharing a Study/Series UID
        #    receives the same replacement — Study/Series links survive while
        #    the original institutional UIDs do not.
        self._remap_uids_recursive(ds, tag_actions, fallback_entropy=uri)
        # A valid Part 10 object must carry a SOP Instance UID even when the
        # source omitted one; mint it deterministically from the file path.
        if 'SOPInstanceUID' not in ds or not str(getattr(ds, 'SOPInstanceUID', '')).strip():
            ds.SOPInstanceUID = self.remap_uid(original_sop_uid, fallback_entropy=uri)
        # Keep file_meta in sync so the object stays internally consistent.
        if hasattr(ds, 'file_meta') and ds.file_meta is not None and 'SOPInstanceUID' in ds:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        # 4. De-identification attestation (PS3.15) — stamped last so the
        #    scrub pass can never remove the attestation attributes.
        self._stamp_attestation(ds, pixel_redacted=pixel_data is not None,
                                tag_actions=tag_actions)

        return ds, tag_actions


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
                meta = d.get(ck(key, ckeys.META_DICT), {})
                fpath = meta.get('filename_or_obj')
                if isinstance(fpath, list):
                    fpath = fpath[0]
                if not fpath:
                    continue

                # Only handle files that were loaded as DICOM (content-driven)
                has_dataset = (ck(key, ckeys.DICOM_DATASET) in d or
                               ck(key, ckeys.DICOM_DATASETS) in d)
                if not has_dataset:
                    continue

                # ---- Series mode: list of datasets ----
                datasets_list = d.get(ck(key, ckeys.DICOM_DATASETS))
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

        fpath = d.get(ck(key, ckeys.META_DICT), {}).get("filename_or_obj", "")
        scrubbed_ds, tag_actions = self.transform.scrub(
            uri=str(fpath),
            pixel_data=pixel_data.astype(dataset.pixel_array.dtype),
            dataset=dataset,
        )
        d[ck(key, ckeys.SCRUBBED_DS)] = scrubbed_ds
        d[ck(key, ckeys.TAG_ACTIONS)] = tag_actions

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
        ds_orig = d.get(ck(key, ckeys.DICOM_DATASET))
        if ds_orig is None:
            ds_orig = pydicom.dcmread(fpath)

        if pix.max() <= 1.1:
            orig_max = ds_orig.get("LargestImagePixelValue", 4095)
            pix = (pix * orig_max)

        scrubbed_ds, tag_actions = self.transform.scrub(
            uri=fpath,
            pixel_data=pix.astype(ds_orig.pixel_array.dtype),
            dataset=ds_orig
        )

        d[ck(key, ckeys.SCRUBBED_DS)] = scrubbed_ds
        d[ck(key, ckeys.TAG_ACTIONS)] = tag_actions

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
            fpath = d.get(ck(key, ckeys.META_DICT), {}).get('filename_or_obj', '')
            return self._scrub_single(d, key, str(fpath))

        C, D, H, W = volume.shape
        scrubbed_datasets = []
        tag_actions_per_slice = []

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

            fpath_i = d.get(ck(key, ckeys.META_DICT), {}).get(
                'slice_uris', ['']
            )
            fp = fpath_i[i] if i < len(fpath_i) else fpath_i[0]

            # Save geometry before scrubbing
            saved_geometry = {}
            for tag_name in self._GEOMETRY_TAGS:
                if hasattr(ds_orig, tag_name):
                    saved_geometry[tag_name] = getattr(ds_orig, tag_name)

            # Scrub metadata + inject redacted pixels
            scrubbed_ds, slice_tag_actions = self.transform.scrub(
                uri=fp,
                pixel_data=slice_pix.astype(ds_orig.pixel_array.dtype),
                dataset=ds_orig
            )

            # Restore geometry tags
            for tag_name, val in saved_geometry.items():
                setattr(scrubbed_ds, tag_name, val)

            scrubbed_datasets.append(scrubbed_ds)
            tag_actions_per_slice.append(slice_tag_actions)

        # Store scrubbed list for downstream SaveDicomSeriesd
        d[ck(key, ckeys.SCRUBBED_DATASETS)] = scrubbed_datasets
        d[ck(key, ckeys.TAG_ACTIONS_PER_SLICE)] = tag_actions_per_slice

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
