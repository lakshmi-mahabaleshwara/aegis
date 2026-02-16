import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import os
import yaml

from transforms.pixel import RedactPixelPHId


class TestRedactPixelPHId(unittest.TestCase):
    """Unit tests for RedactPixelPHId transform"""

    def setUp(self):
        self.config = {
            'ocr': {'languages': ['en'], 'confidence_threshold': 0.4},
            'safelist': ['^m$', '^R$']
        }

        # No need to mock easyocr.Reader — it's lazily initialized via threading.local()
        self.scrubber = RedactPixelPHId(
            keys=['image'],
            config=self.config
        )

    @patch('transforms.pixel.detect_text')
    @patch('transforms.pixel.apply_redaction')
    def test_pixel_scrubbing_flow(self, mock_apply, mock_detect):
        """Test the basic pixel scrubbing flow"""
        # Create test data (channel-first format)
        test_data = np.random.rand(1, 10, 10).astype(np.float32)

        # Mock OCR to detect nothing
        empty_stats = {'total_detections': 0, 'low_confidence_count': 0, 'safelisted_count': 0, 'redacted_count': 0}
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': test_data}
        result = self.scrubber(data)

        self.assertIn('image', result)

        # Verify detect_text was called
        mock_detect.assert_called_once()

    @patch('transforms.pixel.detect_text')
    @patch('transforms.pixel.apply_redaction')
    def test_grayscale_shape_preservation(self, mock_apply, mock_detect):
        """Test that grayscale images maintain shape (1, H, W)"""
        # Grayscale input (1, H, W)
        test_data = np.ones((1, 10, 10), dtype=np.float32)

        empty_stats = {'total_detections': 0, 'low_confidence_count': 0, 'safelisted_count': 0, 'redacted_count': 0}
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': test_data}
        result = self.scrubber(data)

        # Should restore to (1, H, W) after processing
        self.assertEqual(result['image'].shape, (1, 10, 10))


if __name__ == '__main__':
    unittest.main()
