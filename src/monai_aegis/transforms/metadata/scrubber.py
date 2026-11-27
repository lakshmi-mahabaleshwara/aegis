"""
``ScrubDicomMetadata`` — the array transform that orchestrates one scrub.

The transform owns no scrubbing logic of its own; it sequences the
single-responsibility modules of this package. The order is part of the
observable contract — later steps override earlier ones, and the audit
rows are emitted in this sequence::

    1. header scrub       configured REMOVE/ZERO/DUMMY/KEEP actions, recursively  (tag_actions)
    2. private tags       removed recursively, not audited                         (tag_actions)
    3. pixel injection    optional; PixelData + pixel-module tags                  (pixel_injection)
    4. UID graph remap    PS3.15 action U; SOP UID guaranteed; file_meta synced    (uid_remap)
    5. attestation        PS3.15 stamps, always last so nothing can strip them     (attestation)

Thread-safe: the only state on the instance is immutable-after-construction
(action table, identity manager) or a shared UID cache that is safe by
design; the audit trail is created per call and returned, never stored.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
from monai.transforms import Transform
from monai.utils.enums import TransformBackends

from monai_aegis.transforms.metadata.attestation import stamp_attestation
from monai_aegis.transforms.metadata.audit import TagActionRow, TagAuditTrail
from monai_aegis.transforms.metadata.constants import (
    CONFIG_PII_MAPPING_KEY,
    DICOM_STANDARD_UID_ROOT,
    KW_SOP_INSTANCE_UID,
    PRESERVE_UID_KEYWORDS,
    VALID_ACTIONS,
)
from monai_aegis.transforms.metadata.pixel_injection import inject_pixel_data
from monai_aegis.transforms.metadata.tag_actions import TagActionScrubber, build_pii_actions
from monai_aegis.transforms.metadata.uid_remap import UidRemapper
from monai_aegis.transforms.utility import AegisIdentityManager

__all__ = ["ScrubDicomMetadata"]


class ScrubDicomMetadata(Transform):
    """
    Array transform: Scrub PII from a DICOM dataset and inject redacted pixels.

    Applies configurable PII actions (REMOVE, ZERO, DUMMY, KEEP) to DICOM
    tags, removes private tags, injects redacted pixel data with correct
    bit-depth, remaps every patient-linkable UID, and stamps the PS3.15
    attestation attributes on the output.

    This transform is **pure** — it operates in-memory and does NOT write to disk.
    Use :py:class:`SaveDicom` / :py:class:`SaveDicomd` for persistence.

    Args:
        config: Configuration dict with 'pii_mapping' section.

    Raises:
        ValueError: At construction, if ``pii_mapping`` contains a malformed
            tag or an unknown action.

    Example::

        transform = ScrubDicomMetadata(config=config)
        scrubbed_ds = transform(uri="/path/to/file.dcm", pixel_data=redacted_array)
    """

    backend = [TransformBackends.NUMPY]

    # Historical names for the rules, which now live in ``constants``.
    _DICOM_STANDARD_UID_ROOT = DICOM_STANDARD_UID_ROOT
    _PRESERVE_UID_KEYWORDS = PRESERVE_UID_KEYWORDS
    _VALID_ACTIONS = VALID_ACTIONS

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.identity_manager = AegisIdentityManager.from_config(config)
        self._header = TagActionScrubber(
            build_pii_actions(config.get(CONFIG_PII_MAPPING_KEY, {})),
            self.identity_manager,
        )
        # UID remapping shares the tokenization salt so it is reproducible
        # across slices, files, and runs (keeps Study/Series links consistent).
        self._uids = UidRemapper(self.identity_manager.salt)

    def remap_uid(self, original_uid: str, fallback_entropy: str = "") -> str:
        """Deterministically remap a DICOM UID (see :meth:`UidRemapper.remap_uid`)."""
        return self._uids.remap_uid(original_uid, fallback_entropy=fallback_entropy)

    def __call__(
        self,
        uri: str,
        pixel_data: Optional[np.ndarray] = None,
        dataset: Optional[pydicom.Dataset] = None
    ) -> pydicom.Dataset:
        """Scrub a dataset, discarding the audit record.

        Standard array-transform contract. Callers that need the per-tag
        audit trail should use :meth:`scrub`.
        """
        ds, _ = self.scrub(uri, pixel_data=pixel_data, dataset=dataset)
        return ds

    def scrub(
        self,
        uri: str,
        pixel_data: Optional[np.ndarray] = None,
        dataset: Optional[pydicom.Dataset] = None
    ) -> Tuple[pydicom.Dataset, List[TagActionRow]]:
        """Scrub PII and return the dataset together with its audit trail.

        Args:
            uri: Path to the DICOM file (read only if *dataset* is None);
                also the fallback entropy for minting a SOP Instance UID.
            pixel_data: Optional redacted pixel array to inject.
            dataset: Optional in-memory dataset; it is deep-copied, never mutated.

        Returns:
            ``(scrubbed_dataset, tag_actions)`` — the audit rows
            (``{'tag', 'keyword', 'action', 'redacted'}``) in the order the
            steps ran, ending with the ``ATTEST`` rows.

        Raises:
            MetadataScrubError: If pixel injection is given an unsupported
                array shape.
        """
        ds = self._working_copy(uri, dataset)
        original_sop_uid = str(getattr(ds, KW_SOP_INSTANCE_UID, ""))
        trail = TagAuditTrail()

        self._header.scrub_recursive(ds, trail)
        self._header.remove_private_tags_recursive(ds)

        if pixel_data is not None:
            inject_pixel_data(ds, pixel_data)

        self._uids.remap_dataset_recursive(ds, trail, fallback_entropy=uri)
        self._uids.ensure_sop_instance_uid(ds, original_sop_uid, fallback_entropy=uri)

        stamp_attestation(ds, pixel_redacted=pixel_data is not None, trail=trail)

        return ds, trail.to_list()

    @staticmethod
    def _working_copy(uri: str, dataset: Optional[pydicom.Dataset]) -> pydicom.Dataset:
        """The dataset to scrub: a deep copy of *dataset*, or the file at *uri*."""
        if dataset is not None:
            return copy.deepcopy(dataset)
        return pydicom.dcmread(uri)
