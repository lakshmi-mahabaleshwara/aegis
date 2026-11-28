"""Multi-frame DICOM (e.g. ultrasound cine loops) through the single-file path.

Regression tests for a fail-open bug: ``LoadDicomRaw`` left pydicom's
``(F, H, W, 3)`` / ``(F, H, W)`` arrays unconverted, so ``RedactPixelPHI``
read the image rows as frames, OCR never saw a real frame, and burnt-in
PHI passed through unredacted (RGB) or crashed EasyOCR (grayscale).

OCR is replaced by a deterministic fake reader that "reads" any bright
block as text; it also records every array it is handed, so the tests can
assert OCR only ever sees real frames.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import torch
from monai.data import MetaTensor
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from monai_aegis import dicom_runner, image_runner
from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck
from monai_aegis.transforms.exceptions import DicomLoadError, PixelRedactionError
from monai_aegis.transforms.io import LoadDicomRaw, _to_channel_first
from monai_aegis.transforms.pipeline import build_pipeline
from monai_aegis.transforms.pixel import RedactPixelPHI

H, W = 48, 96
TEXT = (slice(4, 16), slice(4, 60))      # burnt-in "PHI" block, same on every frame
TEXT_VALUE = 255
# Blanked area. ocr.redaction_padding only grows a box over adjacent pixels of
# its text colour; beside the block there is only background, so it stays as is.
REDACTED = TEXT


def _frames(n: int, rgb: bool, cols: int = W) -> np.ndarray:
    rng = np.random.default_rng(n)
    fr = rng.integers(40, 160, size=(n, H, cols), dtype=np.uint8)
    fr[:, TEXT[0], TEXT[1]] = TEXT_VALUE
    return np.repeat(fr[..., None], 3, axis=-1) if rgb else fr


def _write_dicom(path: str, pixels: np.ndarray, rgb: bool) -> None:
    multi = pixels.ndim == (4 if rgb else 3)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1" if multi else "1.2.840.10008.5.1.4.1.1.6.1"
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=fm, preamble=b"\x00" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = fm.MediaStorageSOPClassUID, fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientName, ds.PatientID, ds.Modality = "Doe^Jane", "MRN123", "US"
    if rgb:
        ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    else:
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    if multi:
        ds.NumberOfFrames = pixels.shape[0]
    ds.Rows, ds.Columns = pixels.shape[-3:-1] if rgb else pixels.shape[-2:]
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()
    ds.save_as(path, enforce_file_format=True)


class FakeReader:
    """Stands in for easyocr.Reader: one detection covering any bright block."""

    def __init__(self, confidence: float = 0.99):
        self.confidence = confidence
        self.shapes = []

    def readtext(self, image, **_):
        self.shapes.append(tuple(image.shape))
        gray = image if image.ndim == 2 else image.max(axis=-1)
        ys, xs = np.where(gray > 200)
        if not len(ys):
            return []
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return [([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "SMITH JOHN", self.confidence)]


def _overlay(tmp: str) -> str:
    path = os.path.join(tmp, "overlay.yaml")
    Path(path).write_text("ner:\n  enabled: false\nus_regions:\n  enabled: false\n")
    return path


class TestLoaderLayouts(unittest.TestCase):
    """pydicom layout -> channel-first, decided by SamplesPerPixel / NumberOfFrames."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, pixels, rgb):
        path = os.path.join(self.tmp, "x.dcm")
        _write_dicom(path, pixels, rgb)
        return np.asarray(LoadDicomRaw()(path))

    def test_single_frame_mono(self):
        out = self._load(_frames(1, rgb=False)[0], rgb=False)
        self.assertEqual(out.shape, (1, H, W))

    def test_single_frame_rgb(self):
        out = self._load(_frames(1, rgb=True)[0], rgb=True)
        self.assertEqual(out.shape, (3, H, W))

    def test_multi_frame_mono_is_volume(self):
        px = _frames(5, rgb=False)
        out = self._load(px, rgb=False)
        self.assertEqual(out.shape, (1, 5, H, W))
        np.testing.assert_array_equal(out[0, 3], px[3])

    def test_multi_frame_rgb_is_volume(self):
        px = _frames(5, rgb=True)
        px[2, 10, 70] = (1, 2, 3)   # distinct per-channel value at a known frame/row/col
        out = self._load(px, rgb=True)
        self.assertEqual(out.shape, (3, 5, H, W))
        self.assertEqual(tuple(out[:, 2, 10, 70]), (1.0, 2.0, 3.0))

    def test_multi_frame_mono_with_three_columns_is_not_read_as_rgb(self):
        # (F, H, 3) is ambiguous by shape; SamplesPerPixel == 1 must win.
        px = _frames(4, rgb=False, cols=3)
        out = self._load(px, rgb=False)
        self.assertEqual(out.shape, (1, 4, H, 3))

    def test_unsupported_layout_fails_closed(self):
        ds = pydicom.Dataset()
        ds.SamplesPerPixel = 3
        with self.assertRaises(DicomLoadError):
            _to_channel_first(np.zeros((2, 4, 4, 4)), ds, "x.dcm")   # 4-D but not colour-last
        with self.assertRaises(DicomLoadError):
            _to_channel_first(np.zeros((2, 2, 2, 2, 3)), ds, "x.dcm")


class TestVolumeGuard(unittest.TestCase):

    def test_frames_on_wrong_axis_raise_instead_of_passing_through(self):
        t = RedactPixelPHI(config={"ocr": {}, "ner": {"enabled": False}, "safelist": []})
        with self.assertRaises(PixelRedactionError):
            t.redact(np.zeros((6, H, W, 3), dtype=np.float32))   # raw (F, H, W, 3)


class TestSingleFilePipelineMultiFrame(unittest.TestCase):
    """build_pipeline (runner singletons, aegis-deidentify, MCP) on cine loops."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inp = os.path.join(self.tmp, "in")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.inp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, rgb: bool, frames: int = 7):
        src = os.path.join(self.inp, "cine.dcm")
        px = _frames(frames, rgb=rgb)
        _write_dicom(src, px, rgb)
        reader = FakeReader()
        with patch("monai_aegis.transforms.pixel.easyocr.Reader", return_value=reader):
            pipeline = build_pipeline(output_dir=self.out, input_dir=self.inp, overlay_path=_overlay(self.tmp))
            result = pipeline({"image": src})
        return px, result, reader

    def _check(self, rgb: bool):
        px, result, reader = self._run(rgb)
        frames = px.shape[0]

        # OCR only ever saw whole frames.
        self.assertTrue(reader.shapes)
        for shape in reader.shapes:
            self.assertEqual(shape[:2], (H, W))
        stats = result[ck("image", ckeys.REDACTION_STATS)]
        self.assertEqual(stats["num_slices"], frames)
        self.assertGreater(stats["redacted_count"], 0)

        out = pydicom.dcmread(result[ck("image", ckeys.SAVED_PATH)])
        self.assertEqual(int(out.NumberOfFrames), frames)
        self.assertEqual(out.SamplesPerPixel, 3 if rgb else 1)
        self.assertEqual(out.PhotometricInterpretation, "RGB" if rgb else "MONOCHROME2")
        arr = out.pixel_array
        self.assertEqual(arr.shape, px.shape)

        # PHI block (plus side padding) removed on every frame, everything else untouched.
        self.assertTrue((arr[:, REDACTED[0], REDACTED[1]] == 0).all())
        outside = np.ones((H, W), bool)
        outside[REDACTED[0], REDACTED[1]] = False
        np.testing.assert_array_equal(arr[:, outside], px[:, outside])

        # The in-memory tensor keeps the channel-first volume layout.
        self.assertEqual(tuple(result["image"].shape), (3 if rgb else 1, frames, H, W))

    def test_rgb_cine_is_redacted_on_every_frame(self):
        self._check(rgb=True)

    def test_grayscale_cine_is_redacted_on_every_frame(self):
        self._check(rgb=False)

    def test_header_still_scrubbed(self):
        _, result, _ = self._run(rgb=True)
        out = pydicom.dcmread(result[ck("image", ckeys.SAVED_PATH)])
        self.assertNotEqual(str(out.PatientName), "Doe^Jane")
        self.assertEqual(out.PatientIdentityRemoved, "YES")


class TestSeriesLowConfidenceGoesToReview(unittest.TestCase):
    """Multi-file series / image folders follow the single-file review rule."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inp = os.path.join(self.tmp, "in")
        self.out = os.path.join(self.tmp, "out")
        self.review = os.path.join(self.tmp, "review")
        for d in (self.inp, self.out, self.review):
            os.makedirs(d)
        self.sources, self.outputs = [], []
        for i in range(3):
            s = os.path.join(self.inp, "series1", f"{i}.dcm")
            o = os.path.join(self.out, "TOKEN_x", "series1", f"{i}.dcm")
            for p in (s, o):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                Path(p).write_bytes(b"x")
            self.sources.append(s)
            self.outputs.append(o)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _data(self, low_conf: int):
        return {
            "image": MetaTensor(torch.zeros(1, 3, 2, 2)),
            ck("image", ckeys.META_DICT): {"slice_uris": list(self.sources)},
            ck("image", ckeys.REDACTION_STATS): {"low_confidence_count": low_conf},
            ck("image", ckeys.SAVED_PATHS): list(self.outputs),
        }

    def _paths(self, module):
        return module.RunnerPaths(
            config_path="", input_dir=self.inp, output_dir=self.out,
            not_processed_dir=self.review, pipeline_input_dir=self.inp,
            dataloader_num_workers=0,
        )

    def _check(self, module):
        summary = module.RunSummary()
        self.assertFalse(module._handle_low_confidence_series(self._data(0), self._paths(module), summary, "s1"))
        self.assertTrue(all(os.path.exists(o) for o in self.outputs))

        self.assertTrue(module._handle_low_confidence_series(self._data(1), self._paths(module), summary, "s1"))
        self.assertFalse(any(os.path.exists(o) for o in self.outputs))
        for i in range(3):
            self.assertTrue(os.path.exists(os.path.join(self.review, "series1", f"{i}.dcm")))
        self.assertEqual(summary.not_processed, 3)
        self.assertEqual(summary.series, 0)

    def test_dicom_series(self):
        self._check(dicom_runner)

    def test_image_folder(self):
        self._check(image_runner)


if __name__ == "__main__":
    unittest.main()
