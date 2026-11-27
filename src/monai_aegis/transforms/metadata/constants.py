"""
Constants shared by the ``monai_aegis.transforms.metadata`` package.

Every literal that shapes de-identification output — action names, audit
row keys, VR codes, DICOM keywords, UID roots — is defined here and only
here. Modules in this package import from this file instead of repeating
string literals, so a rename or a rule change happens in exactly one place.

NOTE: ``monai_aegis.verify`` carries a deliberate copy of
:data:`DICOM_STANDARD_UID_ROOT` and :data:`PRESERVE_UID_KEYWORDS` so the
verifier can run without the transform stack — keep them in sync.
"""
import pydicom.uid

__all__ = [
    # Transform identity / configuration
    "ARRAY_TRANSFORM_NAME",
    "DICT_TRANSFORM_NAME",
    "CONFIG_PII_MAPPING_KEY",
    # Data-dict contract read by the dictionary transform
    "META_FILENAME_KEY",
    "META_SLICE_URIS_KEY",
    "META_IS_MULTIFRAME_KEY",
    "KW_LARGEST_IMAGE_PIXEL_VALUE",
    "DEFAULT_LARGEST_PIXEL_VALUE",
    "NORMALIZED_MAX",
    "CHANNELS_MONO",
    "CHANNELS_RGB",
    "VOLUME_RANK",
    "GEOMETRY_KEYWORDS",
    # Configured pii_mapping actions
    "ACTION_REMOVE",
    "ACTION_ZERO",
    "ACTION_DUMMY",
    "ACTION_KEEP",
    "VALID_ACTIONS",
    # Actions emitted by other scrub steps
    "ACTION_REMAP",
    "ACTION_ATTEST",
    # Audit row shape
    "ROW_KEY_TAG",
    "ROW_KEY_KEYWORD",
    "ROW_KEY_ACTION",
    "ROW_KEY_REDACTED",
    "TAG_FORMAT",
    # pii_mapping key syntax
    "TAG_KEY_BRACKETS",
    "TAG_KEY_SEPARATOR",
    "HEX_PREFIX",
    "HEX_BASE",
    "AUTO_BASE",
    # Value representations
    "VR_SEQUENCE",
    "VR_UID",
    "BINARY_VRS",
    "VR_MAX_LEN",
    "ZERO_BINARY_VALUE",
    "ZERO_TEXT_VALUE",
    # DICOM keywords / attributes
    "KW_SOP_INSTANCE_UID",
    "KW_MEDIA_STORAGE_SOP_INSTANCE_UID",
    "KW_NUMBER_OF_FRAMES",
    "FILE_META_ATTR",
    # Pixel injection
    "INJECTED_TRANSFER_SYNTAX",
    "BIT_DEPTH_8",
    "BIT_DEPTH_16",
    "PIXEL_REPRESENTATION_UNSIGNED",
    "PIXEL_REPRESENTATION_SIGNED",
    "SAMPLES_PER_PIXEL_MONO",
    "SAMPLES_PER_PIXEL_RGB",
    "COLOR_CHANNEL_COUNTS",
    "RGB_CHANNELS",
    "RANK_SINGLE_FRAME_MONO",
    "RANK_SINGLE_FRAME_COLOR_OR_MULTI_FRAME_MONO",
    "RANK_MULTI_FRAME_COLOR",
    "FRAME_AXIS",
    "PLANAR_CONFIGURATION_INTERLEAVED",
    "PHOTOMETRIC_MONOCHROME2",
    "PHOTOMETRIC_RGB",
    # UID preservation rules
    "DICOM_STANDARD_UID_ROOT",
    "PRESERVE_UID_KEYWORDS",
    # PS3.15 attestation
    "KW_PATIENT_IDENTITY_REMOVED",
    "KW_DEIDENTIFICATION_METHOD",
    "KW_DEIDENTIFICATION_METHOD_CODE_SEQUENCE",
    "KW_BURNED_IN_ANNOTATION",
    "TAG_PATIENT_IDENTITY_REMOVED",
    "TAG_DEIDENTIFICATION_METHOD",
    "TAG_DEIDENTIFICATION_METHOD_CODE_SEQUENCE",
    "TAG_BURNED_IN_ANNOTATION",
    "PATIENT_IDENTITY_REMOVED_VALUE",
    "BURNED_IN_ANNOTATION_VALUE",
    "DEID_METHOD_TOOL_TEMPLATE",
    "DEID_METHOD_HEADER_SCRUB",
    "DEID_METHOD_PIXEL_REDACTION",
    "CODING_SCHEME_DCM",
    "CODE_BASIC_PROFILE",
    "CODE_CLEAN_PIXEL",
]

# ---------------------------------------------------------------------------
# Transform identity (used in MetadataScrubError.transform) / configuration
# ---------------------------------------------------------------------------
ARRAY_TRANSFORM_NAME = "ScrubDicomMetadata"
DICT_TRANSFORM_NAME = "ScrubDicomMetadatad"
# config.yaml section holding the {tag: action} table.
CONFIG_PII_MAPPING_KEY = "pii_mapping"

# ---------------------------------------------------------------------------
# Data-dict contract read by ScrubDicomMetadatad (keys written by the loaders
# in transforms/io.py and transforms/series_io.py into ``{key}_meta_dict``)
# ---------------------------------------------------------------------------
META_FILENAME_KEY = "filename_or_obj"
META_SLICE_URIS_KEY = "slice_uris"
META_IS_MULTIFRAME_KEY = "is_multiframe"
# Loaders may hand over pixels normalised to [0, 1]; anything whose maximum
# is <= NORMALIZED_MAX is scaled back by LargestImagePixelValue (or the
# 12-bit default when the source does not carry it).
KW_LARGEST_IMAGE_PIXEL_VALUE = "LargestImagePixelValue"
DEFAULT_LARGEST_PIXEL_VALUE = 4095
NORMALIZED_MAX = 1.1
# Channel-first layouts the dictionary transform converts to DICOM layout.
CHANNELS_MONO = 1
CHANNELS_RGB = 3
# (C, D, H, W) volume produced by the series / multi-frame loaders.
VOLUME_RANK = 4
# Geometry tags that must NEVER be modified during series scrubbing: saved
# before and restored after each slice scrub.
GEOMETRY_KEYWORDS = frozenset([
    "PixelSpacing", "SliceThickness", "ImageOrientationPatient",
    "ImagePositionPatient", "SpacingBetweenSlices",
])

# ---------------------------------------------------------------------------
# Configured pii_mapping actions
# ---------------------------------------------------------------------------
ACTION_REMOVE = "REMOVE"
ACTION_ZERO = "ZERO"
ACTION_DUMMY = "DUMMY"
ACTION_KEEP = "KEEP"

# The complete action vocabulary (KEEP = deliberate retention, audited
# with redacted=False). Validated at construction time — a typo'd
# action must fail the run before any file is touched, never silently
# leave PHI in place at scrub time.
VALID_ACTIONS = frozenset({ACTION_REMOVE, ACTION_ZERO, ACTION_DUMMY, ACTION_KEEP})

# ---------------------------------------------------------------------------
# Actions emitted by scrub steps other than pii_mapping
# ---------------------------------------------------------------------------
# Audit action recorded for every rewritten UID element.
ACTION_REMAP = "REMAP"
# Audit action recorded for every PS3.15 attestation attribute stamped.
ACTION_ATTEST = "ATTEST"

# ---------------------------------------------------------------------------
# Audit row shape: {'tag': '(GGGG,EEEE)', 'keyword': str, 'action': str, 'redacted': bool}
# ---------------------------------------------------------------------------
ROW_KEY_TAG = "tag"
ROW_KEY_KEYWORD = "keyword"
ROW_KEY_ACTION = "action"
ROW_KEY_REDACTED = "redacted"
# str.format template for a tag: upper-case hex group/element in parentheses.
TAG_FORMAT = "({:04X},{:04X})"

# ---------------------------------------------------------------------------
# pii_mapping tag-key syntax: '(0010,0010)', '(0010, 0010)', '(0x0010,0x0010)'
# ---------------------------------------------------------------------------
TAG_KEY_BRACKETS = "()"
TAG_KEY_SEPARATOR = ","
HEX_PREFIX = "0x"
HEX_BASE = 16   # plain digits are always hex, never decimal/octal
AUTO_BASE = 0   # for '0x'-prefixed input: int() auto-detects (always 16)

# ---------------------------------------------------------------------------
# Value representations
# ---------------------------------------------------------------------------
VR_SEQUENCE = "SQ"
VR_UID = "UI"
# VRs whose ZERO action writes empty bytes rather than an empty string.
BINARY_VRS = ("OB", "OW", "UN")
ZERO_BINARY_VALUE = b""
ZERO_TEXT_VALUE = ""

# Maximum value length (in characters) for the fixed-length string VRs that a
# DUMMY token may be written into. A token is ``TOKEN_`` + 16 hex chars = 22
# characters, which overflows the 16-char VRs (SH, AE, CS). Over-length values
# are non-conformant and rejected by strict DICOM readers, so a token bound for
# one of these is truncated to fit. Truncation is deterministic (same source
# value -> same prefix) and leaves >=10 hex chars (>=40 bits) of hash entropy —
# ample for a pseudonym. VRs without a short fixed cap (LO=64, PN=64, and the
# text VRs) comfortably hold the full token and are intentionally omitted.
VR_MAX_LEN = {"SH": 16, "AE": 16, "CS": 16}

# ---------------------------------------------------------------------------
# DICOM keywords / dataset attributes
# ---------------------------------------------------------------------------
KW_SOP_INSTANCE_UID = "SOPInstanceUID"
KW_MEDIA_STORAGE_SOP_INSTANCE_UID = "MediaStorageSOPInstanceUID"
KW_NUMBER_OF_FRAMES = "NumberOfFrames"
FILE_META_ATTR = "file_meta"

# ---------------------------------------------------------------------------
# Pixel injection: values written to the Image Pixel module when a redacted
# array is injected (see pixel_injection.py)
# ---------------------------------------------------------------------------
# Injected arrays are written uncompressed, so the object is re-labelled
# Explicit VR Little Endian regardless of the source transfer syntax.
INJECTED_TRANSFER_SYNTAX = pydicom.uid.ExplicitVRLittleEndian
# (BitsAllocated, BitsStored, HighBit) for uint8 arrays and for everything else.
BIT_DEPTH_8 = (8, 8, 7)
BIT_DEPTH_16 = (16, 16, 15)
PIXEL_REPRESENTATION_UNSIGNED = 0
PIXEL_REPRESENTATION_SIGNED = 1
SAMPLES_PER_PIXEL_MONO = 1
SAMPLES_PER_PIXEL_RGB = 3
# A trailing axis of this size marks a colour array (RGB or RGBA).
COLOR_CHANNEL_COUNTS = (3, 4)
# Colour arrays are stored as RGB: an alpha channel is dropped.
RGB_CHANNELS = 3
# Array ranks of the accepted layouts: (H, W) / (H, W, C) or (D, H, W) / (D, H, W, C).
RANK_SINGLE_FRAME_MONO = 2
RANK_SINGLE_FRAME_COLOR_OR_MULTI_FRAME_MONO = 3
RANK_MULTI_FRAME_COLOR = 4
# Leading axis of a multi-frame array.
FRAME_AXIS = 0
# Colour-by-pixel (interleaved) layout, matching ``tobytes()`` of an (..., 3) array.
PLANAR_CONFIGURATION_INTERLEAVED = 0
PHOTOMETRIC_MONOCHROME2 = "MONOCHROME2"
PHOTOMETRIC_RGB = "RGB"

# ---------------------------------------------------------------------------
# UID preservation rules (PS3.15 action U)
# ---------------------------------------------------------------------------
# DICOM-standard / well-known UID root.  UIDs under this root identify
# object *types* (SOP Class, Transfer Syntax, coding schemes, ...) — never
# patients — so they are preserved verbatim.  Anything under an org root is
# an instance/study/series/frame identifier and gets re-mapped.
DICOM_STANDARD_UID_ROOT = "1.2.840.10008"

# UID keywords that name an object *class* or the implementation rather
# than a specific instance, preserved even when they use a non-standard
# root (otherwise the output would no longer be a valid DICOM object).
PRESERVE_UID_KEYWORDS = frozenset([
    "SOPClassUID",
    "MediaStorageSOPClassUID",
    "TransferSyntaxUID",
    "ImplementationClassUID",
    "ReferencedSOPClassUID",
])

# ---------------------------------------------------------------------------
# PS3.15 de-identification attestation (see attestation.py)
# ---------------------------------------------------------------------------
KW_PATIENT_IDENTITY_REMOVED = "PatientIdentityRemoved"
KW_DEIDENTIFICATION_METHOD = "DeidentificationMethod"
KW_DEIDENTIFICATION_METHOD_CODE_SEQUENCE = "DeidentificationMethodCodeSequence"
KW_BURNED_IN_ANNOTATION = "BurnedInAnnotation"
TAG_PATIENT_IDENTITY_REMOVED = "(0012,0062)"
TAG_DEIDENTIFICATION_METHOD = "(0012,0063)"
TAG_DEIDENTIFICATION_METHOD_CODE_SEQUENCE = "(0012,0064)"
TAG_BURNED_IN_ANNOTATION = "(0028,0301)"
PATIENT_IDENTITY_REMOVED_VALUE = "YES"
# Stamped only when pixel redaction actually ran on the instance.
BURNED_IN_ANNOTATION_VALUE = "NO"
# DeidentificationMethod is a multi-valued LO: each value must stay within
# the 64-char VR limit.
DEID_METHOD_TOOL_TEMPLATE = "MONAI Aegis {version}"
DEID_METHOD_HEADER_SCRUB = "PS3.15 header scrub"
DEID_METHOD_PIXEL_REDACTION = "Pixel PHI redaction (OCR/NER)"
# DeidentificationMethodCodeSequence items: (CodeValue, CodeMeaning) under
# the DCM coding scheme.
CODING_SCHEME_DCM = "DCM"
CODE_BASIC_PROFILE = ("113100", "Basic Application Confidentiality Profile")
CODE_CLEAN_PIXEL = ("113101", "Clean Pixel Data Option")
