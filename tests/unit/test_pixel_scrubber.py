
import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

from monai_aegis.transforms.pixel import detect_text, apply_redaction, RedactPixelPHI

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

    # --- NER Integration Tests ---

    def test_ner_phi_detected(self):
        """Test that NER-classified PHI text is redacted."""
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, 'John Doe', 0.9)]

        mock_ner = MagicMock()
        mock_ner.classify_texts.return_value = [True]  # PHI

        bboxes, stats = detect_text(self.image, self.mock_reader, self.config, ner_classifier=mock_ner)
        redacted_img = apply_redaction(self.image, bboxes)

        # PHI text should be redacted (black)
        self.assertTrue(np.all(redacted_img[10:20, 10:50, :] == 0))
        self.assertEqual(stats['ner_classified_count'], 1)
        self.assertEqual(stats['redacted_count'], 1)

    def test_ner_clinical_preserved(self):
        """Test that NER-classified non-PHI clinical text is preserved."""
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, 'Depth 13.0', 0.9)]

        mock_ner = MagicMock()
        mock_ner.classify_texts.return_value = [False]  # Not PHI

        bboxes, stats = detect_text(self.image, self.mock_reader, self.config, ner_classifier=mock_ner)
        redacted_img = apply_redaction(self.image, bboxes)

        # Non-PHI text should be preserved (still white)
        self.assertTrue(np.all(redacted_img[10:20, 10:50, :] == 255))
        self.assertEqual(stats['safelisted_count'], 1)
        self.assertEqual(stats['redacted_count'], 0)

    def test_ner_fallback_to_safelist(self):
        """Test that when ner_classifier is None, regex safelist is used."""
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, 'Patient', 0.9)]

        # No NER classifier — should fall back to safelist
        bboxes, stats = detect_text(self.image, self.mock_reader, self.config, ner_classifier=None)
        redacted_img = apply_redaction(self.image, bboxes)

        # "Patient" matches safelist regex — should be preserved
        self.assertTrue(np.all(redacted_img[10:20, 10:50, :] == 255))
        self.assertEqual(stats['safelisted_count'], 1)
        self.assertEqual(stats['ner_classified_count'], 0)

if __name__ == '__main__':
    unittest.main()

