import unittest
import numpy as np
from PIL import Image
import io
import os
import tempfile

from monai_aegis.transforms.io import LoadDicomRawd, LoadImaged


class TestLoadDicomRawd(unittest.TestCase):
    """Unit tests for LoadDicomRawd transform (DICOM only)"""

    def setUp(self):
        self.loader = LoadDicomRawd(keys=['image'])


class TestLoadImaged(unittest.TestCase):
    """Unit tests for LoadImaged transform (JPEG/PNG)"""

    def setUp(self):
        self.loader = LoadImaged(keys=['image'], config={})

    def test_jpeg_loading(self):
        """Test loading JPEG images"""
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            jpg_path = f.name
            img.save(jpg_path)

        try:
            data = {'image': jpg_path}
            result = self.loader(data)

            # Verify shape is (3, H, W) - channel first
            self.assertEqual(result['image'].ndim, 3)
            self.assertEqual(result['image'].shape[0], 3)

            # Verify metadata dict exists and is synced with MetaTensor.meta
            self.assertIn('image_meta_dict', result)
            self.assertEqual(result['image_meta_dict']['filename_or_obj'], jpg_path)

            meta = result['image_meta_dict']
            self.assertIn('spatial_shape', meta)
            self.assertIn('original_channel_dim', meta)
            self.assertEqual(meta['original_channel_dim'], 0)

            # Verify meta_dict is a reference to MetaTensor.meta
            self.assertIs(result['image_meta_dict'], result['image'].meta)

            # Image files should NOT have a cached DICOM dataset
            self.assertNotIn('image_dicom_dataset', result)
        finally:
            import os
            os.unlink(jpg_path)

    def test_grayscale_jpeg_loading(self):
        """Test loading grayscale JPEG images"""
        img = Image.new('L', (10, 10), color=128)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            jpg_path = f.name
            img.save(jpg_path)

        try:
            data = {'image': jpg_path}
            result = self.loader(data)

            # Verify shape is (1, H, W) - channel first for grayscale
            self.assertEqual(result['image'].ndim, 3)
            self.assertEqual(result['image'].shape[0], 1)
        finally:
            os.unlink(jpg_path)

    def test_target_token_uses_top_level_relative_folder(self):
        """Nested image inputs should tokenize by the first path segment under input_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            nested_dir = os.path.join(input_dir, "patient_a", "series_1")
            os.makedirs(nested_dir)
            image_path = os.path.join(nested_dir, "img.jpg")
            Image.new('RGB', (10, 10), color='red').save(image_path)

            loader = LoadImaged(
                keys=['image'],
                config={'tokenization': {'salt': 'test-salt'}},
                input_dir=input_dir,
            )
            result = loader({'image': image_path})

            self.assertIsNotNone(result['image_target_token'])

    def test_target_token_is_none_for_root_level_image(self):
        """Files directly under input_dir should keep their original top-level layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            os.makedirs(input_dir)
            image_path = os.path.join(input_dir, "img.jpg")
            Image.new('RGB', (10, 10), color='red').save(image_path)

            loader = LoadImaged(
                keys=['image'],
                config={'tokenization': {'salt': 'test-salt'}},
                input_dir=input_dir,
            )
            result = loader({'image': image_path})

            self.assertIsNone(result['image_target_token'])


class TestLoadDicomRawdDicom(unittest.TestCase):
    """Unit tests for LoadDicomRawd transform (DICOM files only)"""

    def setUp(self):
        self.loader = LoadDicomRawd(keys=['image'])

    def test_dicom_loading_caches_dataset(self):
        """Test that DICOM loading caches pydicom.Dataset and enriches metadata"""
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import tempfile
        
        # Create a minimal DICOM file
        suffix = '.dcm'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            dcm_path = f.name
        
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.Rows = 10
        ds.Columns = 10
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelData = np.zeros((10, 10), dtype=np.uint16).tobytes()
        ds.Modality = 'US'
        ds.PatientID = 'TEST123'
        ds.StudyDate = '20260101'
        ds.save_as(dcm_path)
        
        try:
            data = {'image': dcm_path}
            result = self.loader(data)
            
            # Verify cached pydicom.Dataset
            self.assertIn('image_dicom_dataset', result)
            self.assertIsInstance(result['image_dicom_dataset'], pydicom.Dataset)
            
            # Verify enriched DICOM metadata in MetaTensor.meta
            meta = result['image'].meta
            self.assertEqual(meta['modality'], 'US')
            self.assertEqual(meta['patient_id'], 'TEST123')
            self.assertEqual(meta['study_date'], '20260101')
            self.assertEqual(meta['original_channel_dim'], 0)
            
            # Verify meta_dict is a reference (not a detached copy)
            self.assertIs(result['image_meta_dict'], result['image'].meta)
        finally:
            os.unlink(dcm_path)

    def test_target_token_uses_top_level_relative_folder_for_dicom(self):
        """Nested DICOM inputs should tokenize by the first path segment under input_dir."""
        import pydicom
        from pydicom.dataset import FileDataset
        from pydicom.uid import ExplicitVRLittleEndian

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            nested_dir = os.path.join(input_dir, "patient_a", "series_1")
            os.makedirs(nested_dir)
            dcm_path = os.path.join(nested_dir, "img.dcm")

            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
            ds.Rows = 10
            ds.Columns = 10
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.PixelData = np.zeros((10, 10), dtype=np.uint16).tobytes()
            ds.Modality = 'US'
            ds.save_as(dcm_path)

            loader = LoadDicomRawd(
                keys=['image'],
                config={'tokenization': {'salt': 'test-salt'}},
                input_dir=input_dir,
            )
            result = loader({'image': dcm_path})

            self.assertIsNotNone(result['image_target_token'])

    def test_target_token_is_none_for_root_level_dicom(self):
        """Root-level DICOM files should not be tokenized."""
        import pydicom
        from pydicom.dataset import FileDataset
        from pydicom.uid import ExplicitVRLittleEndian

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            os.makedirs(input_dir)
            dcm_path = os.path.join(input_dir, "img.dcm")

            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
            ds.Rows = 10
            ds.Columns = 10
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.PixelData = np.zeros((10, 10), dtype=np.uint16).tobytes()
            ds.Modality = 'US'
            ds.save_as(dcm_path)

            loader = LoadDicomRawd(
                keys=['image'],
                config={'tokenization': {'salt': 'test-salt'}},
                input_dir=input_dir,
            )
            result = loader({'image': dcm_path})

            self.assertIsNone(result['image_target_token'])


if __name__ == '__main__':
    unittest.main()
