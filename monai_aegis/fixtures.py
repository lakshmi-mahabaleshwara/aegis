"""Synthetic test-fixture generator for Aegis.

Produces small, deliberately fake medical images — DICOM or PNG/JPEG —
carrying burnt-in text on the pixels and synthetic identifiers in the
headers, so any harness (this repo's tests, an external skill catalog, a
demo) can exercise the full de-identification path without ever touching
real data.

Everything about a fixture is parameterized (size, burnt-in lines, header
values, modality), and generation is deterministic: the same arguments
produce the same bytes, so fixtures can be regenerated instead of
committed. Default identifier values are chosen to be unmistakably
synthetic and to avoid tripping PII linters (no realistic long numeric
IDs, no realistic UIDs).

Also installed as the ``aegis-fixture`` console command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SIZE = (256, 192)  # (width, height)
DEFAULT_TEXT = ("SYNTHETIC PATIENT", "ID SYN-0001")
DEFAULT_TEXT_SIZE = 22

# Fixed, obviously synthetic defaults — deterministic across generations.
DEFAULT_PATIENT_NAME = "SYNTHETIC^PATIENT"
DEFAULT_PATIENT_ID = "SYN-0001"
DEFAULT_BIRTH_DATE = "19000101"
DEFAULT_ACCESSION = "SYN-ACC-1"
DEFAULT_STUDY_DATE = "20200101"
DEFAULT_MODALITY = "OT"
_UID_ROOT = "1.2.826.0.1.3680043.8.498"
DEFAULT_SOP_UID = _UID_ROOT + ".9001"
DEFAULT_STUDY_UID = _UID_ROOT + ".9002"
DEFAULT_SERIES_UID = _UID_ROOT + ".9003"


def _font(text_size: int):
    try:
        return ImageFont.load_default(size=text_size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font only
        return ImageFont.load_default()


def render_text_pixels(
    size: Tuple[int, int] = DEFAULT_SIZE,
    lines: Sequence[str] = DEFAULT_TEXT,
    text_size: int = DEFAULT_TEXT_SIZE,
) -> np.ndarray:
    """Render white text lines on a black canvas; returns uint8 (H, W)."""
    canvas = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(canvas)
    font = _font(text_size)
    y = 8
    for line in lines:
        draw.text((8, y), str(line), fill=255, font=font)
        y += text_size + 8
    return np.asarray(canvas, dtype=np.uint8)


def make_synthetic_dicom(
    path: str,
    size: Tuple[int, int] = DEFAULT_SIZE,
    burned_in_text: Sequence[str] = DEFAULT_TEXT,
    text_size: int = DEFAULT_TEXT_SIZE,
    patient_name: str = DEFAULT_PATIENT_NAME,
    patient_id: str = DEFAULT_PATIENT_ID,
    birth_date: str = DEFAULT_BIRTH_DATE,
    accession: str = DEFAULT_ACCESSION,
    study_date: str = DEFAULT_STUDY_DATE,
    modality: str = DEFAULT_MODALITY,
    sop_uid: str = DEFAULT_SOP_UID,
    with_private_tag: bool = False,
) -> str:
    """Write a synthetic single-frame DICOM with burnt-in text and header PHI.

    Returns the written path. ``with_private_tag`` adds one vendor-private
    element so private-tag removal can be exercised.
    """
    import pydicom

    pixels = render_text_pixels(size, burned_in_text, text_size)

    ds = pydicom.Dataset()
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = birth_date
    ds.AccessionNumber = accession
    ds.StudyDate = study_date
    ds.Modality = modality
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = DEFAULT_STUDY_UID
    ds.SeriesInstanceUID = DEFAULT_SERIES_UID
    if with_private_tag:
        ds.add_new(pydicom.tag.Tag(0x0009, 0x0010), "LO", "SYNTHETIC VENDOR")

    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.ImplementationClassUID = _UID_ROOT + ".9999"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta = file_meta
    ds.preamble = b"\0" * 128

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return str(path)


def make_synthetic_image(
    path: str,
    size: Tuple[int, int] = DEFAULT_SIZE,
    burned_in_text: Sequence[str] = DEFAULT_TEXT,
    text_size: int = DEFAULT_TEXT_SIZE,
) -> str:
    """Write a synthetic PNG/JPEG (by extension) with burnt-in text lines."""
    pixels = render_text_pixels(size, burned_in_text, text_size)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").convert("RGB").save(str(path))
    return str(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aegis-fixture",
        description=(
            "Generate a synthetic medical-image fixture (fake burnt-in text "
            "and, for DICOM, fake header identifiers) for exercising the "
            "de-identification pipeline without real data."
        ),
    )
    parser.add_argument("output", help="Output path (.dcm, .png, .jpg — kind by extension).")
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        metavar="LINE",
        help="Burnt-in text line (repeatable; default: two synthetic lines).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--text-size", type=int, default=DEFAULT_TEXT_SIZE)
    parser.add_argument(
        "--with-private-tag",
        action="store_true",
        help="DICOM only: include one vendor-private element.",
    )
    args = parser.parse_args(argv)

    lines = args.text or list(DEFAULT_TEXT)
    size = (args.width, args.height)
    if Path(args.output).suffix.lower() in (".jpg", ".jpeg", ".png"):
        written = make_synthetic_image(args.output, size, lines, args.text_size)
    else:
        written = make_synthetic_dicom(
            args.output,
            size,
            lines,
            args.text_size,
            with_private_tag=args.with_private_tag,
        )
    print(written)
    return 0


__all__ = [
    "render_text_pixels",
    "make_synthetic_dicom",
    "make_synthetic_image",
    "main",
    "DEFAULT_SIZE",
    "DEFAULT_TEXT",
    "DEFAULT_PATIENT_NAME",
    "DEFAULT_PATIENT_ID",
]


if __name__ == "__main__":
    sys.exit(main())
