"""
PS3.15 de-identification attestation stamping.

:func:`stamp_attestation` writes the machine-checkable "this object was
de-identified" markers that receivers (archives, PACS, downstream de-id
audits, ``aegis-verify``) key on, and records one ``ATTEST`` audit row
(``redacted=False``) per attribute written:

  - (0012,0062) PatientIdentityRemoved = YES
  - (0012,0063) DeidentificationMethod (multi-valued LO): tool + version,
    "PS3.15 header scrub", plus the pixel-redaction entry when pixel
    redaction ran
  - (0012,0064) DeidentificationMethodCodeSequence — DCM 113100
    (Basic Application Confidentiality Profile), plus DCM 113101
    (Clean Pixel Data Option) when pixel redaction ran
  - (0028,0301) BurnedInAnnotation = NO — only when pixel redaction
    actually ran on this instance; a header-only scrub leaves whatever the
    source carried untouched

Existing values of these attributes on the source are **replaced**, not
extended. Must be called *after* the scrub pass so the configured tag
actions can never strip the attestation itself.
"""
from typing import List, Tuple

import pydicom

from monai_aegis import __version__ as _AEGIS_VERSION
from monai_aegis.transforms.metadata.audit import TagAuditTrail
from monai_aegis.transforms.metadata.constants import (
    BURNED_IN_ANNOTATION_VALUE,
    CODE_BASIC_PROFILE,
    CODE_CLEAN_PIXEL,
    CODING_SCHEME_DCM,
    DEID_METHOD_HEADER_SCRUB,
    DEID_METHOD_PIXEL_REDACTION,
    DEID_METHOD_TOOL_TEMPLATE,
    KW_BURNED_IN_ANNOTATION,
    KW_DEIDENTIFICATION_METHOD,
    KW_DEIDENTIFICATION_METHOD_CODE_SEQUENCE,
    KW_PATIENT_IDENTITY_REMOVED,
    PATIENT_IDENTITY_REMOVED_VALUE,
    TAG_BURNED_IN_ANNOTATION,
    TAG_DEIDENTIFICATION_METHOD,
    TAG_DEIDENTIFICATION_METHOD_CODE_SEQUENCE,
    TAG_PATIENT_IDENTITY_REMOVED,
)

__all__ = ["deidentification_codes", "deidentification_methods", "stamp_attestation"]


def deidentification_methods(pixel_redacted: bool) -> List[str]:
    """Values for (0012,0063) DeidentificationMethod, in order."""
    method = [DEID_METHOD_TOOL_TEMPLATE.format(version=_AEGIS_VERSION), DEID_METHOD_HEADER_SCRUB]
    if pixel_redacted:
        method.append(DEID_METHOD_PIXEL_REDACTION)
    return method


def deidentification_codes(pixel_redacted: bool) -> List[Tuple[str, str]]:
    """``(CodeValue, CodeMeaning)`` pairs for (0012,0064), in order."""
    codes = [CODE_BASIC_PROFILE]
    if pixel_redacted:
        codes.append(CODE_CLEAN_PIXEL)
    return codes


def _code_sequence(codes: List[Tuple[str, str]]) -> List[pydicom.Dataset]:
    """Build DeidentificationMethodCodeSequence items under the DCM scheme."""
    seq = []
    for value, meaning in codes:
        item = pydicom.Dataset()
        item.CodeValue = value
        item.CodingSchemeDesignator = CODING_SCHEME_DCM
        item.CodeMeaning = meaning
        seq.append(item)
    return seq


def stamp_attestation(ds: pydicom.Dataset, pixel_redacted: bool, trail: TagAuditTrail) -> None:
    """Stamp the PS3.15 attestation attributes on *ds* and audit them as ``ATTEST`` rows.

    Args:
        ds: The dataset to stamp (modified in place).
        pixel_redacted: True when pixel redaction ran on this instance —
            adds the Clean Pixel Data Option entries and ``BurnedInAnnotation=NO``.
        trail: Per-call audit trail; receives one row per attribute written,
            in tag order, after any rows already recorded.
    """
    setattr(ds, KW_PATIENT_IDENTITY_REMOVED, PATIENT_IDENTITY_REMOVED_VALUE)
    setattr(ds, KW_DEIDENTIFICATION_METHOD, deidentification_methods(pixel_redacted))
    setattr(ds, KW_DEIDENTIFICATION_METHOD_CODE_SEQUENCE, _code_sequence(deidentification_codes(pixel_redacted)))

    stamped = [
        (TAG_PATIENT_IDENTITY_REMOVED, KW_PATIENT_IDENTITY_REMOVED),
        (TAG_DEIDENTIFICATION_METHOD, KW_DEIDENTIFICATION_METHOD),
        (TAG_DEIDENTIFICATION_METHOD_CODE_SEQUENCE, KW_DEIDENTIFICATION_METHOD_CODE_SEQUENCE),
    ]
    if pixel_redacted:
        setattr(ds, KW_BURNED_IN_ANNOTATION, BURNED_IN_ANNOTATION_VALUE)
        stamped.append((TAG_BURNED_IN_ANNOTATION, KW_BURNED_IN_ANNOTATION))
    for tag, keyword in stamped:
        trail.record_attest(tag, keyword)
