"""
MONAI Aegis Inter-Transform Contract — data-dict key registry.

Aegis transforms coordinate through side-channel entries in the MONAI
pipeline data dict, keyed as ``{key}_{suffix}`` (matching MONAI's own
``{key}_meta_dict`` convention). Every suffix the pipeline reads or
writes is defined here — transforms and runners must build these keys
via :func:`ck` instead of inline f-strings, so a typo fails at import
time rather than silently disabling a de-identification step.

Contract (producer → consumer):

================== ========================= =================================
Suffix              Producer                  Consumer(s)
================== ========================= =================================
META_DICT           loaders                   all transforms, runners, reporting
DICOM_DATASET       LoadDicomRawd             RedactByUSRegionsd,
                                              ScrubDicomMetadatad, reporting
DICOM_DATASETS      LoadDicomSeriesd          RedactByUSRegionsd,
                                              ScrubDicomMetadatad, reporting
TARGET_TOKEN        loaders                   savers
US_PHI_MASK         RedactByUSRegionsd        RedactPixelPHId
REDACTION_STATS     RedactPixelPHId           runners, reporting
REDACTION_MASK      RedactPixelPHId           external consumers (validation)
SCRUBBED_DS         ScrubDicomMetadatad       SaveDicomd, SaveDicomSeriesd
                                              (multi-frame), reporting
SCRUBBED_DATASETS   ScrubDicomMetadatad       SaveDicomSeriesd, reporting
TAG_ACTIONS         ScrubDicomMetadatad       reporting
TAG_ACTIONS_PER_    ScrubDicomMetadatad       reporting
SLICE               (series mode)
SAVED_PATH          SaveDicomd, SaveImaged,   runners
                    SaveDicomSeriesd
                    (multi-frame)
SAVED_PATHS         SaveDicomSeriesd,         runners
                    SaveImageSeriesd
================== ========================= =================================
"""
from typing import Hashable

__all__ = [
    "ck",
    "META_DICT",
    "DICOM_DATASET",
    "DICOM_DATASETS",
    "TARGET_TOKEN",
    "US_PHI_MASK",
    "REDACTION_STATS",
    "REDACTION_MASK",
    "SCRUBBED_DS",
    "SCRUBBED_DATASETS",
    "TAG_ACTIONS",
    "TAG_ACTIONS_PER_SLICE",
    "SAVED_PATH",
    "SAVED_PATHS",
]

# Loaders
META_DICT = "meta_dict"
DICOM_DATASET = "dicom_dataset"
DICOM_DATASETS = "dicom_datasets"
TARGET_TOKEN = "target_token"

# US region masking
US_PHI_MASK = "us_phi_mask"

# Pixel redaction
REDACTION_STATS = "redaction_stats"
REDACTION_MASK = "redaction_mask"

# Metadata scrubbing
SCRUBBED_DS = "scrubbed_ds"
SCRUBBED_DATASETS = "scrubbed_datasets"
TAG_ACTIONS = "tag_actions"
TAG_ACTIONS_PER_SLICE = "tag_actions_per_slice"

# Savers
SAVED_PATH = "saved_path"
SAVED_PATHS = "saved_paths"


def ck(key: Hashable, suffix: str) -> str:
    """Build the data-dict key for a side-channel value.

    Example::

        d[ck("image", SCRUBBED_DS)]   # → d["image_scrubbed_ds"]
    """
    return f"{key}_{suffix}"
