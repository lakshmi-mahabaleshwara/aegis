"""
Write a redacted pixel array back into a DICOM dataset.

:func:`inject_pixel_data` replaces ``PixelData`` and re-derives every Image
Pixel module attribute that must agree with the array, in this order:

1. transfer syntax    → Explicit VR Little Endian (array is written uncompressed)
2. contiguity         → ``np.ascontiguousarray`` so ``tobytes()`` is row-major
3. bit depth / sign   → ``BitsAllocated`` / ``BitsStored`` / ``HighBit`` / ``PixelRepresentation``
4. dimensions         → ``Rows`` / ``Columns`` / ``SamplesPerPixel`` / ``NumberOfFrames`` /
                         ``PlanarConfiguration``, from the array layout (below)
5. photometric        → ``MONOCHROME2`` or ``RGB``
6. pixel data         → ``PixelData = array.tobytes()``
7. windowing          → ``WindowCenter`` / ``WindowWidth`` recomputed from the array range

Accepted layouts (``H`` rows, ``W`` columns, ``D`` frames)::

    (H, W)            monochrome, single frame
    (H, W, 3|4)       RGB / RGBA (alpha dropped), single frame
    (D, H, W)         monochrome, multi-frame
    (D, H, W, 3|4)    RGB / RGBA (alpha dropped), multi-frame

A 3-D array whose last axis is 3 or 4 is always read as colour; any other
shape raises :class:`MetadataScrubError`. ``NumberOfFrames`` is removed for
single-frame layouts and written (as a string) for multi-frame ones.
``uint8`` maps to 8-bit storage, every other dtype to 16-bit (signed iff the
dtype is a signed integer).
"""
import numpy as np
import pydicom

from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms.metadata.constants import (
    ARRAY_TRANSFORM_NAME,
    BIT_DEPTH_16,
    BIT_DEPTH_8,
    COLOR_CHANNEL_COUNTS,
    FILE_META_ATTR,
    FRAME_AXIS,
    INJECTED_TRANSFER_SYNTAX,
    KW_NUMBER_OF_FRAMES,
    PHOTOMETRIC_MONOCHROME2,
    PHOTOMETRIC_RGB,
    PIXEL_REPRESENTATION_SIGNED,
    PIXEL_REPRESENTATION_UNSIGNED,
    PLANAR_CONFIGURATION_INTERLEAVED,
    RANK_MULTI_FRAME_COLOR,
    RANK_SINGLE_FRAME_COLOR_OR_MULTI_FRAME_MONO,
    RANK_SINGLE_FRAME_MONO,
    RGB_CHANNELS,
    SAMPLES_PER_PIXEL_MONO,
    SAMPLES_PER_PIXEL_RGB,
)

__all__ = ["inject_pixel_data"]


def _set_transfer_syntax(ds: pydicom.Dataset) -> None:
    """Label the object as uncompressed Explicit VR Little Endian (when it has file meta)."""
    if hasattr(ds, FILE_META_ATTR):
        getattr(ds, FILE_META_ATTR).TransferSyntaxUID = INJECTED_TRANSFER_SYNTAX


def _sync_bit_depth(ds: pydicom.Dataset, pixels: np.ndarray) -> None:
    """Derive BitsAllocated / BitsStored / HighBit / PixelRepresentation from the dtype."""
    if pixels.dtype == np.uint8:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = BIT_DEPTH_8
        ds.PixelRepresentation = PIXEL_REPRESENTATION_UNSIGNED
    else:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = BIT_DEPTH_16
        ds.PixelRepresentation = (
            PIXEL_REPRESENTATION_SIGNED if np.issubdtype(pixels.dtype, np.signedinteger)
            else PIXEL_REPRESENTATION_UNSIGNED
        )


def _drop_number_of_frames(ds: pydicom.Dataset) -> None:
    """Single-frame layouts must not carry a stale NumberOfFrames."""
    if hasattr(ds, KW_NUMBER_OF_FRAMES):
        delattr(ds, KW_NUMBER_OF_FRAMES)


def _sync_dimensions(ds: pydicom.Dataset, pixels: np.ndarray) -> np.ndarray:
    """Derive Rows / Columns / SamplesPerPixel / NumberOfFrames / PlanarConfiguration.

    Returns the array to store — the input, or its first three channels
    when an alpha channel is dropped.

    Raises:
        MetadataScrubError: If the array shape is not one of the accepted layouts.
    """
    if pixels.ndim == RANK_SINGLE_FRAME_MONO:
        ds.SamplesPerPixel = SAMPLES_PER_PIXEL_MONO
        ds.Rows, ds.Columns = pixels.shape
        _drop_number_of_frames(ds)
    elif pixels.ndim == RANK_SINGLE_FRAME_COLOR_OR_MULTI_FRAME_MONO:
        if pixels.shape[-1] in COLOR_CHANNEL_COUNTS:
            ds.SamplesPerPixel = SAMPLES_PER_PIXEL_RGB
            ds.Rows, ds.Columns = pixels.shape[:2]
            pixels = pixels[..., :RGB_CHANNELS]
            ds.PlanarConfiguration = PLANAR_CONFIGURATION_INTERLEAVED
            _drop_number_of_frames(ds)
        else:
            setattr(ds, KW_NUMBER_OF_FRAMES, str(pixels.shape[FRAME_AXIS]))
            ds.SamplesPerPixel = SAMPLES_PER_PIXEL_MONO
            ds.Rows, ds.Columns = pixels.shape[1:]
    elif pixels.ndim == RANK_MULTI_FRAME_COLOR and pixels.shape[-1] in COLOR_CHANNEL_COUNTS:
        setattr(ds, KW_NUMBER_OF_FRAMES, str(pixels.shape[FRAME_AXIS]))
        ds.SamplesPerPixel = SAMPLES_PER_PIXEL_RGB
        ds.Rows, ds.Columns = pixels.shape[1:3]
        pixels = pixels[..., :RGB_CHANNELS]
        ds.PlanarConfiguration = PLANAR_CONFIGURATION_INTERLEAVED
    else:
        raise MetadataScrubError(
            f"Unsupported pixel array shape for DICOM injection: {pixels.shape}",
            transform=ARRAY_TRANSFORM_NAME,
        )
    return pixels


def _sync_windowing(ds: pydicom.Dataset, pixels: np.ndarray) -> None:
    """Recompute WindowCenter / WindowWidth from the stored array's value range."""
    p_min, p_max = float(pixels.min()), float(pixels.max())
    ds.WindowCenter = str((p_max + p_min) / 2)
    ds.WindowWidth = str(p_max - p_min)


def inject_pixel_data(ds: pydicom.Dataset, pixel_data: np.ndarray) -> None:
    """Inject *pixel_data* into *ds* in place, synchronising the pixel module tags.

    Args:
        ds: The dataset to update (modified in place).
        pixel_data: Redacted pixels in one of the accepted layouts; the
            dtype decides the stored bit depth.

    Raises:
        MetadataScrubError: If the array shape is not one of the accepted layouts.
    """
    _set_transfer_syntax(ds)
    contiguous_pixels = np.ascontiguousarray(pixel_data)
    _sync_bit_depth(ds, contiguous_pixels)
    contiguous_pixels = _sync_dimensions(ds, contiguous_pixels)
    ds.PhotometricInterpretation = (
        PHOTOMETRIC_MONOCHROME2 if ds.SamplesPerPixel == SAMPLES_PER_PIXEL_MONO else PHOTOMETRIC_RGB
    )
    ds.PixelData = contiguous_pixels.tobytes()
    _sync_windowing(ds, contiguous_pixels)
