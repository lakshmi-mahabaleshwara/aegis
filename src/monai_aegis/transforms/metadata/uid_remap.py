"""
Deterministic UID-graph remapping (PS3.15 action U).

Every patient-linkable UID — SOP / Study / Series / FrameOfReference
instance UIDs and any UID referenced in nested sequences — is replaced with
a new value derived from ``(tokenization salt, original UID)``. Because the
derivation is deterministic and cached, a UID reused across the object (or
across slices, files and runs with the same salt) always lands on the same
replacement, so intra-object references and Study/Series links survive.

Preserved verbatim:

* anything under the DICOM standard root (``DICOM_STANDARD_UID_ROOT``)
  — SOP Class, Transfer Syntax, coding schemes — these identify object
  *types*, never patients;
* elements whose keyword names an object class or the implementation
  (``PRESERVE_UID_KEYWORDS``), even under an organisational root.

``file_meta`` is *not* walked; only ``MediaStorageSOPInstanceUID`` is
re-synchronised with the (remapped) ``SOPInstanceUID`` afterwards
(:meth:`UidRemapper.ensure_sop_instance_uid`).

Every element whose value changes is recorded on the audit trail as an
``action='REMAP'`` row (``redacted=True``), one row per element even when
the element is multi-valued.
"""
from typing import Any, Dict, Optional

import pydicom
import pydicom.uid

from monai_aegis.transforms.metadata.audit import TagAuditTrail, keyword_for
from monai_aegis.transforms.metadata.constants import (
    DICOM_STANDARD_UID_ROOT,
    FILE_META_ATTR,
    KW_MEDIA_STORAGE_SOP_INSTANCE_UID,
    KW_SOP_INSTANCE_UID,
    PRESERVE_UID_KEYWORDS,
    VR_SEQUENCE,
    VR_UID,
)

__all__ = ["UidRemapper"]


class UidRemapper:
    """Salt-keyed, cached UID remapper shared by every scrub call.

    Args:
        salt: The tokenization salt. Keyed off the same secret as identity
            tokenization so the mapping is reproducible across slices, files,
            and runs (required to keep Study/Series links consistent).

    The cache is shared across threads by design (consistent UID remapping
    across slices); dict get/set are atomic under the GIL and a lost race
    only costs a recomputed hash, never an inconsistent mapping.
    """

    def __init__(self, salt: Optional[str]) -> None:
        self._salt = salt
        self._cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Deterministic UID generation
    # ------------------------------------------------------------------

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
        cached = self._cache.get(source)
        if cached is None:
            cached = pydicom.uid.generate_uid(entropy_srcs=[self._salt or "", source])
            self._cache[source] = cached
        return cached

    # ------------------------------------------------------------------
    # Preservation rules
    # ------------------------------------------------------------------

    @staticmethod
    def is_preserved_keyword(keyword: str) -> bool:
        """True if *keyword* names an object class / implementation UID that must not be remapped."""
        return keyword in PRESERVE_UID_KEYWORDS

    @staticmethod
    def is_standard_uid(value: Any) -> bool:
        """True if *value* is empty or lies under the DICOM standard root (never remapped)."""
        source = str(value or "").strip()
        return not source or source.startswith(DICOM_STANDARD_UID_ROOT)

    def remap_value(self, value: Any, fallback_entropy: str = "") -> Any:
        """Remap one UID value, preserving class- and standard-defined UIDs.

        UIDs under the DICOM standard root identify object *types* (SOP
        Class, Transfer Syntax, coding schemes) — never patients — so they
        pass through verbatim; rewriting them would make the object invalid.
        Empty values are left untouched. Everything else is a
        patient-linkable instance/study/series/frame identifier and is
        remapped deterministically.
        """
        if self.is_standard_uid(value):
            return value
        return self.remap_uid(str(value).strip(), fallback_entropy=fallback_entropy)

    # ------------------------------------------------------------------
    # Recursive traversal + audit
    # ------------------------------------------------------------------

    def remap_dataset_recursive(
        self,
        ds: pydicom.Dataset,
        trail: TagAuditTrail,
        fallback_entropy: str = "",
    ) -> None:
        """Remap every patient-linkable UID on a dataset and nested items.

        Walks all ``UI``-valued elements (recursing into sequences) and
        remaps them via :meth:`remap_value`, except those whose keyword
        names an object class or the implementation
        (:meth:`is_preserved_keyword`). Because :meth:`remap_uid` is
        deterministic and cached, a UID that appears in several places — a
        SeriesInstanceUID shared across slices, or a ReferencedSOPInstanceUID
        pointing at another instance in the same object — is rewritten to the
        *same* replacement everywhere, so cross-references stay intact.
        Each rewrite is recorded on *trail* as a ``REMAP`` row.
        """
        for elem in ds:
            if elem.VR == VR_SEQUENCE:
                for item in elem.value:
                    if isinstance(item, pydicom.Dataset):
                        self.remap_dataset_recursive(item, trail, fallback_entropy)
                continue
            if elem.VR != VR_UID or elem.value is None:
                continue
            keyword = keyword_for(elem.tag)
            if self.is_preserved_keyword(keyword):
                continue
            # A UID element is either a single value (pydicom.uid.UID, a str
            # subclass) or a MultiValue of them; handle both uniformly.
            if isinstance(elem.value, str):
                new_value = self.remap_value(elem.value, fallback_entropy)
                changed = str(new_value) != str(elem.value)
            else:
                originals = list(elem.value)
                new_value = [self.remap_value(v, fallback_entropy) for v in originals]
                changed = any(str(a) != str(b) for a, b in zip(new_value, originals))
            if not changed:
                continue
            elem.value = new_value
            trail.record_remap(elem.tag, keyword)

    # ------------------------------------------------------------------
    # SOP Instance UID guarantee + file_meta consistency
    # ------------------------------------------------------------------

    def ensure_sop_instance_uid(
        self,
        ds: pydicom.Dataset,
        original_sop_uid: str,
        fallback_entropy: str = "",
    ) -> None:
        """Guarantee a SOP Instance UID and keep ``file_meta`` in sync.

        A valid Part 10 object must carry a SOP Instance UID even when the
        source omitted one; mint it deterministically from
        *original_sop_uid* (captured before scrubbing) or, failing that,
        from *fallback_entropy* (the file path). ``MediaStorageSOPInstanceUID``
        is then re-pointed at the dataset value so the object stays
        internally consistent. Not audited — the SOP UID remap itself is
        already recorded by :meth:`remap_dataset_recursive` when present.
        """
        current = str(getattr(ds, KW_SOP_INSTANCE_UID, "")).strip()
        if KW_SOP_INSTANCE_UID not in ds or not current:
            setattr(ds, KW_SOP_INSTANCE_UID, self.remap_uid(original_sop_uid, fallback_entropy=fallback_entropy))
        file_meta = getattr(ds, FILE_META_ATTR, None)
        if file_meta is not None and KW_SOP_INSTANCE_UID in ds:
            setattr(file_meta, KW_MEDIA_STORAGE_SOP_INSTANCE_UID, getattr(ds, KW_SOP_INSTANCE_UID))
