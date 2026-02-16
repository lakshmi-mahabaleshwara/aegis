
import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

from transforms.pixel import detect_text, apply_redaction, RedactPixelPHI

class TestPixelScrubber(unittest.TestCase):

    def setUp(self):
        self.mock_reader = MagicMock()
        self.config = {
            'ocr': {
                'confidence_threshold': 0.5
            },
            'safelist': ['^Patient$']
        }
        # Dummy image (100x100 RGB)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8) 
        # White background
        self.image[:] = 255

    def test_redact_pixels_match(self):
        # Mock OCR output: bbox, text, confidence
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, 'SECRET', 0.9)]
        
        bboxes, stats = detect_text(self.image, self.mock_reader, self.config)
        redacted_img = apply_redaction(self.image, bboxes)
        
        # Check area is black (0)
        self.assertTrue(np.all(redacted_img[10:20, 10:50, :] == 0))
        # Check outside area is still white (255)
        self.assertTrue(np.all(redacted_img[0:5, 0:5, :] == 255))

    def test_redact_pixels_safelist(self):
        # Text "Patient" matching safelist
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, 'Patient', 0.9)]
        
        bboxes, stats = detect_text(self.image, self.mock_reader, self.config)
        redacted_img = apply_redaction(self.image, bboxes)
        
        # Check area is NOT redacted (still white 255)
        self.assertTrue(np.all(redacted_img[10:20, 10:50, :] == 255))

    def test_low_confidence_ignored(self):
        # Text "MAYBE" with low confidence
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, 'MAYBE', 0.1)]
        
        bboxes, stats = detect_text(self.image, self.mock_reader, self.config)
        redacted_img = apply_redaction(self.image, bboxes)
        
        # Check area is NOT redacted
        self.assertTrue(np.all(redacted_img[10:20, 10:50, :] == 255))

    def test_empty_image(self):
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        self.mock_reader.readtext.return_value = []
        
        bboxes, stats = detect_text(empty_img, self.mock_reader, self.config)
        redacted_img = apply_redaction(empty_img, bboxes)
        self.assertEqual(redacted_img.size, 0)

    def test_no_text_detected(self):
        noise_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.mock_reader.readtext.return_value = []
        
        bboxes, stats = detect_text(noise_img, self.mock_reader, self.config)
        redacted_img = apply_redaction(noise_img, bboxes)
        
        # Should be identical to input
        self.assertTrue(np.array_equal(noise_img, redacted_img))

if __name__ == '__main__':
    unittest.main()
