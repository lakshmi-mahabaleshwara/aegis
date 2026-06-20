import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from monai_aegis.transforms.pixel import RedactPixelPHI


class TestVLMRedaction(unittest.TestCase):
    def setUp(self):
        self.config = {
            'ocr': {
                'engine': 'vlm',
            },
            'vlm': {
                'enabled': True,
                'model_name': 'microsoft/Florence-2-large',
                'device': 'cpu'
            }
        }
        
    @patch('monai_aegis.transforms.vlm_classifier.VLMClassifier')
    def test_vlm_redaction(self, MockVLMClassifier):
        # Setup mock VLM classifier
        mock_instance = MagicMock()
        
        # Mock detect_phi_boxes to return a single bounding box
        # EasyOCR format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        mock_instance.detect_phi_boxes.return_value = (
            [
                [[10, 10], [20, 10], [20, 20], [10, 20]]
            ], 
            {'total_detections': 1, 'redacted_count': 1}
        )
        MockVLMClassifier.return_value = mock_instance
        
        transform = RedactPixelPHI(config=self.config)
        # Manually set the vlm_classifier to our mock to avoid lazy loading
        transform._thread_local.vlm_classifier = mock_instance
        
        # Create a dummy image (C, H, W)
        image = np.ones((3, 50, 50), dtype=np.float32)
        
        # Apply transform
        redacted = transform(image)
        
        # Verify the mock was called
        mock_instance.detect_phi_boxes.assert_called_once()
        
        # Check that the region was redacted (set to 0)
        # Region: y:10-20, x:10-20
        self.assertEqual(np.sum(redacted[:, 10:20, 10:20]), 0)
        
        # Check that other regions are preserved
        self.assertEqual(redacted[0, 5, 5], 1.0)
        
    @patch('monai_aegis.transforms.vlm_classifier.VLMClassifier')
    def test_vlm_grayscale_conversion(self, MockVLMClassifier):
        # Setup mock VLM classifier
        mock_instance = MagicMock()
        mock_instance.detect_phi_boxes.return_value = ([], {})
        MockVLMClassifier.return_value = mock_instance
        
        transform = RedactPixelPHI(config=self.config)
        transform._thread_local.vlm_classifier = mock_instance
        
        # Create a dummy grayscale image (1, H, W)
        image = np.ones((1, 50, 50), dtype=np.uint8)
        
        # Apply transform
        redacted = transform(image)
        
        # Check that detect_phi_boxes was called with RGB array
        called_args, _ = mock_instance.detect_phi_boxes.call_args
        passed_array = called_args[0]
        
        self.assertEqual(passed_array.ndim, 3)
        self.assertEqual(passed_array.shape[-1], 3)
        
if __name__ == '__main__':
    unittest.main()
