import unittest
from unittest.mock import patch
import numpy as np
import torch

from monai_aegis.transforms.pixel import RedactPixelPHId
from monai.data import MetaTensor
from monai.transforms import MapTransform


class TestRedactPixelPHIdCompliance(unittest.TestCase):
    """Tests for MONAI API base class compliance"""

    def test_inherits_from_monai_bases(self):
        """Verify RedactPixelPHId inherits from MapTransform only."""
        self.assertTrue(issubclass(RedactPixelPHId, MapTransform))

    def test_cooperative_init(self):
        """Verify super().__init__() correctly initializes keys via MapTransform"""
        config = {'ocr': {'languages': ['en']}, 'safelist': []}
        transform = RedactPixelPHId(keys=['image', 'mask'], config=config)
        # MapTransform stores keys as KeysCollection
        self.assertEqual(len(transform.keys), 2)
        self.assertFalse(transform.allow_missing_keys)

    def test_no_inverse_contract(self):
        """Verify RedactPixelPHId does not advertise fake invertibility."""
        config = {'ocr': {'languages': ['en']}, 'safelist': []}
        transform = RedactPixelPHId(keys=['image'], config=config)
        self.assertFalse(hasattr(transform, 'inverse'))


class TestRedactPixelPHId(unittest.TestCase):
    """Unit tests for RedactPixelPHId transform"""

    def setUp(self):
        self.config = {
            'ocr': {'languages': ['en'], 'confidence_threshold': 0.4},
            'safelist': ['^m$', '^R$']
        }

        self.scrubber = RedactPixelPHId(
            keys=['image'],
            config=self.config
        )

    @patch('monai_aegis.transforms.pixel.easyocr.Reader')
    @patch('monai_aegis.transforms.pixel.detect_text')
    @patch('monai_aegis.transforms.pixel.apply_redaction')
    def test_pixel_scrubbing_flow(self, mock_apply, mock_detect, mock_reader):
        """Test the basic pixel scrubbing flow"""
        test_data = np.random.rand(1, 10, 10).astype(np.float32)

        empty_stats = {
            'total_detections': 0, 'low_confidence_count': 0,
            'safelisted_count': 0, 'redacted_count': 0,
        }
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': test_data}
        result = self.scrubber(data)

        self.assertIn('image', result)
        mock_detect.assert_called_once()

    @patch('monai_aegis.transforms.pixel.easyocr.Reader')
    @patch('monai_aegis.transforms.pixel.detect_text')
    @patch('monai_aegis.transforms.pixel.apply_redaction')
    def test_grayscale_shape_preservation(self, mock_apply, mock_detect, mock_reader):
        """Test that grayscale images maintain shape (1, H, W)"""
        test_data = np.ones((1, 10, 10), dtype=np.float32)

        empty_stats = {
            'total_detections': 0, 'low_confidence_count': 0,
            'safelisted_count': 0, 'redacted_count': 0,
        }
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': test_data}
        result = self.scrubber(data)

        # Should restore to (1, H, W) after processing
        self.assertEqual(result['image'].shape, (1, 10, 10))

    @patch('monai_aegis.transforms.pixel.easyocr.Reader')
    @patch('monai_aegis.transforms.pixel.detect_text')
    @patch('monai_aegis.transforms.pixel.apply_redaction')
    def test_redaction_mask_is_exposed_without_transform_history(
        self, mock_apply, mock_detect, mock_reader
    ):
        """Test that redaction side outputs exist without fake invertibility."""
        test_data = np.ones((1, 10, 10), dtype=np.float32)
        meta_tensor = MetaTensor(
            torch.as_tensor(test_data), meta={'filename_or_obj': 'test.dcm'}
        )

        empty_stats = {
            'total_detections': 0, 'low_confidence_count': 0,
            'safelisted_count': 0, 'redacted_count': 0,
        }
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': meta_tensor}
        result = self.scrubber(data)

        output_tensor = result['image']
        self.assertIsInstance(output_tensor, MetaTensor)
        self.assertEqual(len(output_tensor.applied_operations), 0)

        self.assertIn('image_redaction_mask', result)
        self.assertEqual(result['image_redaction_mask'].shape, (10, 10))

    @patch('monai_aegis.transforms.pixel.easyocr.Reader')
    @patch('monai_aegis.transforms.pixel.detect_text')
    @patch('monai_aegis.transforms.pixel.apply_redaction')
    def test_spatial_transform_warning(self, mock_apply, mock_detect, mock_reader):
        """Test that a warning is logged when prior spatial transforms exist"""
        test_data = np.ones((1, 10, 10), dtype=np.float32)
        meta_tensor = MetaTensor(
            torch.as_tensor(test_data), meta={'filename_or_obj': 'test.dcm'}
        )

        meta_tensor.applied_operations.append({
            'class': 'Spacingd',
            'extra_info': {'pixdim': [1.0, 1.0]},
        })

        empty_stats = {
            'total_detections': 0, 'low_confidence_count': 0,
            'safelisted_count': 0, 'redacted_count': 0,
        }
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': meta_tensor}

        with self.assertLogs('monai_aegis.transforms.pixel', level='WARNING') as cm:
            self.scrubber(data)

        self.assertTrue(
            any('prior spatial transforms' in msg for msg in cm.output),
            f"Expected spatial transform warning, got: {cm.output}"
        )


if __name__ == '__main__':
    unittest.main()
