"""
MONAI Aegis Custom Exceptions

Structured exception hierarchy for contextual error handling across
the de-identification pipeline. Each exception carries the uri
and originating transform name for diagnostic logging.

Hierarchy::

    AegisTransformError          (base — catch-all for pipeline errors)
    ├── DicomLoadError           (corrupted / unreadable DICOM)
    ├── ImageLoadError           (unreadable JPEG / PNG)
    ├── PixelRedactionError      (OCR / NER failure during redaction)
    ├── MetadataScrubError       (tag parsing / pixel injection failure)
    └── DicomSaveError           (write failure to output directory)
"""


class AegisTransformError(Exception):
    """Base exception for all Aegis pipeline transform errors.

    Args:
        message: Human-readable description of what went wrong.
        uri: Path to the file being processed when the error occurred.
        transform: Name of the transform class where the error originated.
    """

    def __init__(self, message: str, uri: str = "", transform: str = ""):
        self.uri = uri
        self.transform = transform
        full_msg = f"[{transform}] {message}" if transform else message
        if uri:
            full_msg += f" (file: {uri})"
        super().__init__(full_msg)


class DicomLoadError(AegisTransformError):
    """Raised when a DICOM file cannot be read or parsed.

    Common causes: corrupted file, missing pixel data, unsupported
    transfer syntax, file not found.
    """
    pass


class ImageLoadError(AegisTransformError):
    """Raised when a standard image file (JPEG, PNG) cannot be opened.

    Common causes: corrupted file, unsupported format, file not found,
    permission denied.
    """
    pass


class PixelRedactionError(AegisTransformError):
    """Raised when OCR detection or pixel redaction fails.

    Common causes: EasyOCR initialization failure, NER model error,
    malformed image array, out-of-memory during inference.
    """
    pass


class MetadataScrubError(AegisTransformError):
    """Raised when DICOM metadata scrubbing or pixel injection fails.

    Common causes: invalid tag format in config, dataset missing expected
    attributes, pixel data shape mismatch.
    """
    pass


class DicomSaveError(AegisTransformError):
    """Raised when a scrubbed DICOM dataset cannot be written to disk.

    Common causes: output directory not writable, disk full,
    permission denied, invalid dataset state.
    """
    pass


class SeriesLoadError(AegisTransformError):
    """Raised when a DICOM series cannot be loaded as a volume.

    Common causes: inconsistent slice geometry within a series,
    missing slices, incompatible pixel formats across slices,
    no valid imaging files found in the series.
    """
    pass


class SeriesSaveError(AegisTransformError):
    """Raised when a de-identified DICOM series cannot be written to disk.

    Common causes: output directory not writable, invalid dataset state
    after scrubbing, slice count mismatch between volume and metadata.
    """
    pass
