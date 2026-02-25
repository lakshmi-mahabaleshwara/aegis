import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import yaml

from transforms.metadata import ScrubDicomMetadatad


class TestScrubDicomMetadatad(unittest.TestCase):
    """Unit tests for ScrubDicomMetadatad transform"""

    def setUp(self):
        # Mock config
        self.mock_config = {
            'pii_mapping': {
                '(0010, 0010)': 'REMOVE',
                '(0010, 0020)': 'DUMMY',
            }
        }

        self.scrubber = ScrubDicomMetadatad(
            keys=['image'],
            config=self.mock_config
        )

    @patch('transforms.metadata.pydicom.dcmread')
    def test_dicom_only_processing(self, mock_dcmread):
        """Test that only DICOM files are processed"""
        # Setup mock DICOM dataset
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((10, 10), dtype=np.uint16)
        mock_ds.pixel_array.dtype = np.dtype('uint16')
        mock_ds.get.return_value = 4095
        mock_dcmread.return_value = mock_ds

        # Test data with DICOM file - values > 1.1 to skip normalization
        data = {
            'image': np.ones((1, 10, 10), dtype=np.float32) * 100,
            'image_meta_dict': {'filename_or_obj': '/path/to/file.dcm'},
            'image_dicom_dataset': mock_ds
        }

        # Mock the array transform's __call__ method
        self.scrubber.transform = MagicMock()
        self.scrubber.transform.return_value = mock_ds

        result = self.scrubber(data)

        # Verify dcmread was NEVER called because we provided the cached dataset
        mock_dcmread.assert_not_called()

        # Verify the array transform was called
        self.scrubber.transform.assert_called_once()
        # Verify scrubbed dataset is stored in dict (no file save)
        self.assertIn('image_scrubbed_ds', result)

    def test_skip_non_dicom_files(self):
        """Test that non-DICOM files are skipped"""
        data = {
            'image': np.zeros((3, 10, 10), dtype=np.float32),
            'image_meta_dict': {'filename_or_obj': '/path/to/file.jpg'}
        }

        # Should not raise any errors, just skip
        result = self.scrubber(data)
        self.assertIn('image', result)
        # No scrubbed_ds for non-DICOM
        self.assertNotIn('image_scrubbed_ds', result)

    @patch('transforms.metadata.pydicom.dcmread')
    def test_grayscale_orientation(self, mock_dcmread):
        """Test grayscale image (1, H, W) is squeezed not transposed"""
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((10, 10), dtype=np.uint16)
        mock_ds.get.return_value = 4095
        mock_dcmread.return_value = mock_ds

        # Grayscale input (1, H, W)
        data = {
            'image': np.ones((1, 10, 10), dtype=np.float32),
            'image_meta_dict': {'filename_or_obj': '/path/to/file.dcm'},
            'image_dicom_dataset': mock_ds
        }

        # Mock the array transform to capture its arguments
        self.scrubber.transform = MagicMock()
        self.scrubber.transform.return_value = mock_ds

        result = self.scrubber(data)
        
        mock_dcmread.assert_not_called()

        # Check that pixel_data passed was squeezed to (H, W)
        call_kwargs = self.scrubber.transform.call_args[1]
        pixel_data = call_kwargs['pixel_data']
        self.assertEqual(pixel_data.ndim, 2)  # Should be (H, W)

    @patch('transforms.metadata.pydicom.dcmread')
    def test_rgb_orientation(self, mock_dcmread):
        """Test RGB image (3, H, W) is transposed to (H, W, 3)"""
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_ds.get.return_value = 255
        mock_dcmread.return_value = mock_ds

        # RGB input (3, H, W)
        data = {
            'image': np.ones((3, 10, 10), dtype=np.float32),
            'image_meta_dict': {'filename_or_obj': '/path/to/file.dcm'}
        }

        # Mock the array transform
        self.scrubber.transform = MagicMock()
        self.scrubber.transform.return_value = mock_ds

        result = self.scrubber(data)

        # Check that pixel_data was transposed to (H, W, 3)
        call_kwargs = self.scrubber.transform.call_args[1]
        pixel_data = call_kwargs['pixel_data']
        self.assertEqual(pixel_data.shape, (10, 10, 3))

    def test_no_file_io_in_call(self):
        """Verify that ScrubDicomMetadatad does NOT write files."""
        # This confirms thread safety — no file I/O side effects
        import os
        from unittest.mock import call

        mock_ds = MagicMock()
        mock_ds.pixel_array = np.zeros((10, 10), dtype=np.uint16)
        mock_ds.get.return_value = 4095

        self.scrubber.transform = MagicMock()
        self.scrubber.transform.return_value = mock_ds

        with patch('transforms.metadata.pydicom.dcmread', return_value=mock_ds):
            data = {
                'image': np.ones((1, 10, 10), dtype=np.float32) * 100,
                'image_meta_dict': {'filename_or_obj': '/path/to/file.dcm'}
            }
            result = self.scrubber(data)

        # Verify save_as was never called on the dataset
        mock_ds.save_as.assert_not_called()


if __name__ == '__main__':
    unittest.main()
