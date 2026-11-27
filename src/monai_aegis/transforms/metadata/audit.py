"""
Audit trail for DICOM header de-identification.

Every step of the metadata scrub (configured tag actions, UID remapping,
attestation stamping) reports what it did as a *tag-action row*::

    {'tag': '(GGGG,EEEE)', 'keyword': str, 'action': str, 'redacted': bool}

:class:`TagAuditTrail` collects those rows, in order, for one
:meth:`ScrubDicomMetadata.scrub` call, and :meth:`TagAuditTrail.to_list`
hands them back as the plain ``list[dict]`` that downstream code expects
(``monai_aegis.reporting`` writes them to ``aegis_tag_actions.csv``;
``monai_aegis.envelope`` counts ``redacted`` rows). This module is the single
definition of that shape — no other module builds a row dict by hand.

The trail is created per call and returned, never stored on a transform
instance, so one transform can be shared across threads.
"""
from typing import Any, Dict, Iterator, List

import pydicom.datadict
import pydicom.tag

from monai_aegis.transforms.metadata.constants import (
    ACTION_ATTEST,
    ACTION_KEEP,
    ACTION_REMAP,
    ROW_KEY_ACTION,
    ROW_KEY_KEYWORD,
    ROW_KEY_REDACTED,
    ROW_KEY_TAG,
    TAG_FORMAT,
)

__all__ = ["TagActionRow", "TagAuditTrail", "format_tag", "keyword_for", "make_row"]

# One audit row.
TagActionRow = Dict[str, Any]


def format_tag(tag: pydicom.tag.BaseTag) -> str:
    """Render a tag as ``'(GGGG,EEEE)'`` with upper-case hex."""
    return TAG_FORMAT.format(tag.group, tag.element)


def keyword_for(tag: pydicom.tag.BaseTag) -> str:
    """DICOM keyword for *tag*, or ``''`` when the dictionary has none."""
    return pydicom.datadict.keyword_for_tag(tag) or ""


def make_row(tag_str: str, keyword: str, action: str, redacted: bool) -> TagActionRow:
    """Build one audit row from an already-formatted tag string.

    Key order is part of the contract (CSV column order downstream reads
    the dict in insertion order).
    """
    return {
        ROW_KEY_TAG: tag_str,
        ROW_KEY_KEYWORD: keyword,
        ROW_KEY_ACTION: action,
        ROW_KEY_REDACTED: redacted,
    }


class TagAuditTrail:
    """Ordered collector of tag-action rows for a single scrub call.

    Example::

        trail = TagAuditTrail()
        trail.record_action(tag, "PatientName", "DUMMY")
        trail.record_remap(tag, "SOPInstanceUID")
        trail.record_attest("(0012,0062)", "PatientIdentityRemoved")
        rows = trail.to_list()   # -> list[dict], in the order recorded
    """

    def __init__(self) -> None:
        self._rows: List[TagActionRow] = []

    # ------------------------------------------------------------------
    # Dedicated record methods — one per kind of scrub step
    # ------------------------------------------------------------------

    def record_action(self, tag: pydicom.tag.BaseTag, keyword: str, action: str) -> None:
        """Record a configured ``pii_mapping`` action (REMOVE / ZERO / DUMMY / KEEP).

        REMOVE / ZERO / DUMMY de-identify the element (``redacted=True``);
        KEEP retains it by design (``redacted=False``).
        """
        self._rows.append(make_row(format_tag(tag), keyword, action, redacted=action != ACTION_KEEP))

    def record_remap(self, tag: pydicom.tag.BaseTag, keyword: str) -> None:
        """Record a UID-graph remap of one element (``action=REMAP``, ``redacted=True``)."""
        self._rows.append(make_row(format_tag(tag), keyword, ACTION_REMAP, redacted=True))

    def record_attest(self, tag_str: str, keyword: str) -> None:
        """Record a stamped PS3.15 attestation attribute (``action=ATTEST``, ``redacted=False``).

        Takes the tag already formatted (``'(0012,0062)'``) because the
        attestation step names its attributes statically.
        """
        self._rows.append(make_row(tag_str, keyword, ACTION_ATTEST, redacted=False))

    # ------------------------------------------------------------------
    # Conversion back to the list[dict] contract
    # ------------------------------------------------------------------

    def to_list(self) -> List[TagActionRow]:
        """The rows recorded so far, in order, as the downstream ``list[dict]`` contract."""
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[TagActionRow]:
        return iter(self._rows)
