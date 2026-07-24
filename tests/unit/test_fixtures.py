"""Unit tests for the synthetic fixture generator (monai_aegis/fixtures.py).

Pins the generator contract: valid readable DICOM with the declared
synthetic header identifiers and non-blank burnt-in pixels, valid PNG/JPEG
output, parameterization, determinism (same arguments → same bytes), and
the aegis-fixture CLI kind-by-extension dispatch.
"""

import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monai_aegis import fixtures


def test_dicom_fixture_valid_and_synthetic(tmp_path):
    path = fixtures.make_synthetic_dicom(str(tmp_path / "f.dcm"))
    ds = pydicom.dcmread(path)
    assert str(ds.PatientName) == fixtures.DEFAULT_PATIENT_NAME
    assert str(ds.PatientID) == fixtures.DEFAULT_PATIENT_ID
    assert str(ds.file_meta.MediaStorageSOPInstanceUID) == str(ds.SOPInstanceUID)
    assert ds.pixel_array.shape == (fixtures.DEFAULT_SIZE[1], fixtures.DEFAULT_SIZE[0])
    assert ds.pixel_array.max() > 0  # burnt-in text rendered


def test_dicom_fixture_parameterized(tmp_path):
    path = fixtures.make_synthetic_dicom(
        str(tmp_path / "f.dcm"),
        size=(128, 64),
        burned_in_text=["ONLY LINE"],
        patient_name="OTHER^SYNTH",
        modality="US",
        with_private_tag=True,
    )
    ds = pydicom.dcmread(path)
    assert str(ds.PatientName) == "OTHER^SYNTH"
    assert ds.Modality == "US"
    assert ds.pixel_array.shape == (64, 128)
    assert any(elem.tag.is_private for elem in ds.iterall())


def test_dicom_fixture_no_private_tags_by_default(tmp_path):
    ds = pydicom.dcmread(fixtures.make_synthetic_dicom(str(tmp_path / "f.dcm")))
    assert not any(elem.tag.is_private for elem in ds.iterall())


def test_image_fixture_png_and_jpeg(tmp_path):
    for name in ("f.png", "f.jpg"):
        path = fixtures.make_synthetic_image(str(tmp_path / name))
        with Image.open(path) as img:
            assert img.size == fixtures.DEFAULT_SIZE
            assert np.asarray(img.convert("L")).max() > 0


def test_generation_deterministic(tmp_path):
    a = Path(fixtures.make_synthetic_dicom(str(tmp_path / "a.dcm"))).read_bytes()
    b = Path(fixtures.make_synthetic_dicom(str(tmp_path / "b.dcm"))).read_bytes()
    assert a == b
    pa = Path(fixtures.make_synthetic_image(str(tmp_path / "a.png"))).read_bytes()
    pb = Path(fixtures.make_synthetic_image(str(tmp_path / "b.png"))).read_bytes()
    assert pa == pb


def test_render_text_pixels_shape_and_content():
    pixels = fixtures.render_text_pixels(size=(100, 50), lines=["HI"])
    assert pixels.shape == (50, 100)
    assert pixels.dtype == np.uint8
    assert pixels.max() == 255 and pixels.min() == 0


def test_cli_dispatch_by_extension(tmp_path, capsys):
    assert fixtures.main([str(tmp_path / "x.dcm"), "--text", "A", "--text", "B"]) == 0
    assert fixtures.main([str(tmp_path / "x.png")]) == 0
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines == [str(tmp_path / "x.dcm"), str(tmp_path / "x.png")]
    pydicom.dcmread(str(tmp_path / "x.dcm"))
    with Image.open(str(tmp_path / "x.png")):
        pass


def test_fixture_fails_default_verification(tmp_path):
    """The raw fixture is deliberately dirty — verification must reject it."""
    from monai_aegis import verify

    run = tmp_path / "run"
    run.mkdir()
    fixtures.make_synthetic_dicom(str(run / "raw.dcm"), with_private_tag=True)
    report = verify.verify_run(str(run))
    assert report["status"] == "fail"
    failed = {f["check"] for f in report["findings"]}
    assert "patient-name-tokenized" in failed
    assert "no-private-tags" in failed
