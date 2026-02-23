import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import os
import yaml
import logging
import torch

from transforms.pixel import RedactPixelPHId
from monai.data import MetaTensor


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

    @patch('transforms.pixel.detect_text')
    @patch('transforms.pixel.apply_redaction')
    def test_push_transform_stores_info(self, mock_apply, mock_detect):
        """Test that push_transform stores redaction info in MetaTensor history"""
        test_data = np.ones((1, 10, 10), dtype=np.float32)
        meta_tensor = MetaTensor(torch.as_tensor(test_data), meta={'filename_or_obj': 'test.dcm'})

        empty_stats = {'total_detections': 0, 'low_confidence_count': 0, 'safelisted_count': 0, 'redacted_count': 0}
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': meta_tensor}
        result = self.scrubber(data)

        # Verify transform was pushed to applied_operations
        output_tensor = result['image']
        self.assertIsInstance(output_tensor, MetaTensor)
        self.assertTrue(len(output_tensor.applied_operations) > 0)

        # Verify extra_info contains redaction metadata
        last_op = output_tensor.applied_operations[-1]
        self.assertIn('extra_info', last_op)
        self.assertIn('redaction_stats', last_op['extra_info'])
        self.assertIn('redaction_mask_shape', last_op['extra_info'])
        self.assertEqual(last_op['extra_info']['redaction_mask_shape'], [10, 10])

        # Verify redaction mask is present and matches spatial_shape
        self.assertIn('image_redaction_mask', result)
        self.assertEqual(result['image_redaction_mask'].shape, (10, 10))

    @patch('transforms.pixel.detect_text')
    @patch('transforms.pixel.apply_redaction')
    def test_inverse_cleans_up(self, mock_apply, mock_detect):
        """Test that inverse() removes side-keys and pops transform history"""
        test_data = np.ones((1, 10, 10), dtype=np.float32)
        meta_tensor = MetaTensor(torch.as_tensor(test_data), meta={'filename_or_obj': 'test.dcm'})

        empty_stats = {'total_detections': 0, 'low_confidence_count': 0, 'safelisted_count': 0, 'redacted_count': 0}
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        # Forward pass
        data = {'image': meta_tensor}
        result = self.scrubber(data)
        self.assertIn('image_redaction_mask', result)
        self.assertIn('image_redaction_stats', result)

        # Inverse pass
        inverted = self.scrubber.inverse(result)
        self.assertNotIn('image_redaction_mask', inverted)
        self.assertNotIn('image_redaction_stats', inverted)

    @patch('transforms.pixel.detect_text')
    @patch('transforms.pixel.apply_redaction')
    def test_spatial_transform_warning(self, mock_apply, mock_detect):
        """Test that a warning is logged when prior spatial transforms exist"""
        test_data = np.ones((1, 10, 10), dtype=np.float32)
        meta_tensor = MetaTensor(torch.as_tensor(test_data), meta={'filename_or_obj': 'test.dcm'})

        # Simulate prior spatial transform in history
        meta_tensor.applied_operations.append({
            'class': 'Spacingd',
            'extra_info': {'pixdim': [1.0, 1.0]},
        })

        empty_stats = {'total_detections': 0, 'low_confidence_count': 0, 'safelisted_count': 0, 'redacted_count': 0}
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = test_data.squeeze(0)

        data = {'image': meta_tensor}

        with self.assertLogs('transforms.pixel', level='WARNING') as cm:
            self.scrubber(data)

        self.assertTrue(
            any('prior spatial transforms' in msg for msg in cm.output),
            f"Expected spatial transform warning, got: {cm.output}"
        )


if __name__ == '__main__':
    unittest.main()

