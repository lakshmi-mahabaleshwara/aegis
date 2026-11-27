"""
MONAI Aegis Metadata Transforms — ScrubDicomMetadata / ScrubDicomMetadatad

DICOM metadata de-identification with configurable PII actions
(REMOVE, ZERO, DUMMY, KEEP), deterministic UID-graph remapping, pixel data
injection, and PS3.15 attestation stamping.

Thread-safe: no file I/O in ``__call__()`` — saving is handled by
:py:class:`SaveDicomd`.

Package layout (one responsibility per module)::

    constants        every literal the package uses (actions, VRs, keywords, roots, codes)
    audit            TagAuditTrail — the {'tag','keyword','action','redacted'} row contract
    tag_actions      pii_mapping parsing/validation + REMOVE/ZERO/DUMMY/KEEP, recursive
    uid_remap        deterministic UID-graph remap (PS3.15 action U)
    pixel_injection  write a redacted array back into the dataset (pixel-module sync)
    attestation      PS3.15 de-identification stamps + ATTEST audit rows
    scrubber         ScrubDicomMetadata — sequences the steps above
    dict_transform   ScrubDicomMetadatad — MONAI data-dict wrapper (single / series / multi-frame)

Raises:
    MetadataScrubError: When pixel injection or dataset manipulation fails
        during scrubbing.
"""
# Re-exported on purpose: callers (and tests) patch
# ``monai_aegis.transforms.metadata.pydicom.dcmread``.
import pydicom  # noqa: F401

from monai_aegis.transforms.metadata.dict_transform import ScrubDicomMetadatad
from monai_aegis.transforms.metadata.scrubber import ScrubDicomMetadata

__all__ = ["ScrubDicomMetadata", "ScrubDicomMetadatad"]
