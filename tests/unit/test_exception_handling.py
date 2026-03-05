import unittest
import tempfile
import os
import numpy as np

from transforms.exceptions import (
    AegisTransformError,
    DicomLoadError,
    ImageLoadError,
    PixelRedactionError,
    MetadataScrubError,
    DicomSaveError,
)
from transforms.io import LoadDicomRawd, LoadImaged, SaveDicom


class TestExceptionHierarchy(unittest.TestCase):
    """Test that all custom exceptions inherit from AegisTransformError."""

    def test_hierarchy(self):
        for exc_cls in [DicomLoadError, ImageLoadError, PixelRedactionError,
                        MetadataScrubError, DicomSaveError]:
            self.assertTrue(issubclass(exc_cls, AegisTransformError))

    def test_exception_carries_context(self):
        exc = DicomLoadError("bad file", filepath="/tmp/test.dcm", transform="LoadDicomRaw")
        self.assertEqual(exc.filepath, "/tmp/test.dcm")
        self.assertEqual(exc.transform, "LoadDicomRaw")
        self.assertIn("LoadDicomRaw", str(exc))
        self.assertIn("/tmp/test.dcm", str(exc))


class TestDicomLoadError(unittest.TestCase):
    """Test that corrupted or missing DICOM files raise DicomLoadError."""

    def test_missing_dicom_file(self):
        loader = LoadDicomRawd(keys=['image'])
        with self.assertRaises(DicomLoadError) as ctx:
            loader({'image': '/nonexistent/path/missing.dcm'})
        self.assertIn('missing.dcm', str(ctx.exception))

    def test_corrupted_dicom_file(self):
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as f:
            f.write(b'NOT A DICOM FILE - CORRUPTED DATA')
            bad_dcm = f.name
        try:
            loader = LoadDicomRawd(keys=['image'])
            with self.assertRaises(DicomLoadError) as ctx:
                loader({'image': bad_dcm})
            self.assertIn(bad_dcm, str(ctx.exception))
        finally:
            os.unlink(bad_dcm)


class TestImageLoadError(unittest.TestCase):
    """Test that unreadable image files raise ImageLoadError."""

    def test_missing_image_file(self):
        loader = LoadImaged(keys=['image'])
        with self.assertRaises(ImageLoadError) as ctx:
            loader({'image': '/nonexistent/path/missing.jpg'})
        self.assertIn('missing.jpg', str(ctx.exception))

    def test_corrupted_image_file(self):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'NOT A JPEG FILE')
            bad_jpg = f.name
        try:
            loader = LoadImaged(keys=['image'])
            with self.assertRaises(ImageLoadError) as ctx:
                loader({'image': bad_jpg})
            self.assertIn(bad_jpg, str(ctx.exception))
        finally:
            os.unlink(bad_jpg)


class TestDicomSaveError(unittest.TestCase):
    """Test that write failures raise DicomSaveError."""

    def test_save_to_invalid_directory(self):
        from unittest.mock import MagicMock
        saver = SaveDicom.__new__(SaveDicom)
        saver.output_dir = '/nonexistent/readonly/dir'
        saver.fs = None

        mock_ds = MagicMock()
        mock_ds.save_as.side_effect = OSError("Permission denied")

        with self.assertRaises(DicomSaveError) as ctx:
            saver(dataset=mock_ds, filepath='/input/test.dcm')
        self.assertIn('test.dcm', str(ctx.exception))
        self.assertIn('SaveDicom', str(ctx.exception))


class TestPixelRedactionError(unittest.TestCase):
    """Test that PixelRedactionError carries context."""

    def test_error_attributes(self):
        exc = PixelRedactionError(
            "OCR crashed",
            filepath="/tmp/image.jpg",
            transform="RedactPixelPHId"
        )
        self.assertEqual(exc.filepath, "/tmp/image.jpg")
        self.assertEqual(exc.transform, "RedactPixelPHId")
        self.assertIn("OCR crashed", str(exc))


class TestMetadataScrubError(unittest.TestCase):
    """Test that MetadataScrubError carries context."""

    def test_error_attributes(self):
        exc = MetadataScrubError(
            "Invalid tag format",
            filepath="/tmp/scan.dcm",
            transform="ScrubDicomMetadatad"
        )
        self.assertEqual(exc.filepath, "/tmp/scan.dcm")
        self.assertEqual(exc.transform, "ScrubDicomMetadatad")
        self.assertIn("Invalid tag format", str(exc))


if __name__ == '__main__':
    unittest.main()
