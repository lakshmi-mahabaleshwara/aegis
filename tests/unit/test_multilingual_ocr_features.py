"""Unit tests for multilingual OCR, handwriting, and VLM stages."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from monai_aegis.transforms.language_presets import (
    LANGUAGE_PRESETS,
    resolve_ocr_languages,
)
from monai_aegis.transforms.handwriting import HandwritingRecognizer
from monai_aegis.transforms.vlm_detector import (
    VLMTextDetector,
    _bbox_iou_xywh,
    _quad_to_easyocr_bbox,
)
from monai_aegis.transforms.pixel import detect_text, RedactPixelPHI


class TestLanguagePresets(unittest.TestCase):

    def test_explicit_languages_win(self):
        langs = resolve_ocr_languages({"languages": ["en", "es"], "language_preset": "cjk"})
        self.assertEqual(langs, ["en", "es"])

    def test_preset_expands(self):
        langs = resolve_ocr_languages({"languages": [], "language_preset": "cjk"})
        self.assertEqual(langs, LANGUAGE_PRESETS["cjk"])
        self.assertIn("ja", langs)
        self.assertIn("en", langs)

    def test_default_english(self):
        self.assertEqual(resolve_ocr_languages({}), ["en"])
        self.assertEqual(resolve_ocr_languages({"languages": [], "language_preset": ""}), ["en"])

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            resolve_ocr_languages({"language_preset": "klingon"})

    def test_medical_eu_preset(self):
        langs = resolve_ocr_languages({"language_preset": "medical_eu"})
        self.assertIn("de", langs)
        self.assertIn("fr", langs)


class TestHandwritingRecognizer(unittest.TestCase):

    def test_disabled_by_default(self):
        hw = HandwritingRecognizer({"ocr": {}})
        self.assertFalse(hw.enabled)

    def test_should_reread_low_confidence(self):
        hw = HandwritingRecognizer({
            "ocr": {
                "handwriting": {
                    "enabled": True,
                    "re_recognize_low_confidence": True,
                    "re_recognize_all": False,
                }
            }
        })
        self.assertTrue(hw.should_reread(0.2, 0.4))
        self.assertFalse(hw.should_reread(0.9, 0.4))

    def test_enrich_results_replaces_low_conf(self):
        hw = HandwritingRecognizer({
            "ocr": {
                "handwriting": {
                    "enabled": True,
                    "re_recognize_low_confidence": True,
                    "confidence_threshold": 0.4,
                }
            }
        })
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        bbox = [[5, 5], [60, 5], [60, 30], [5, 30]]
        results = [(bbox, "???", 0.1)]

        with patch.object(hw, "recognize_crop", return_value=("Jane Doe", 1.0)):
            enriched = hw.enrich_results(image, results, ocr_confidence_threshold=0.4)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0][1], "Jane Doe")
        self.assertEqual(enriched[0][3], "handwriting")


class TestVlmDetector(unittest.TestCase):

    def test_disabled_by_default(self):
        vlm = VLMTextDetector({"ocr": {}})
        self.assertFalse(vlm.enabled)

    def test_quad_conversion(self):
        bbox = _quad_to_easyocr_bbox([1, 2, 10, 2, 10, 8, 1, 8])
        self.assertEqual(bbox[0], [1, 2])
        self.assertEqual(bbox[2], [10, 8])

    def test_iou(self):
        self.assertGreater(_bbox_iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]), 0.1)
        self.assertEqual(_bbox_iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)

    def test_merge_skips_overlapping(self):
        vlm = VLMTextDetector({
            "ocr": {"vlm": {"enabled": True, "overlap_iou_threshold": 0.5}}
        })
        existing_bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]
        existing = [(existing_bbox, "A", 0.9, "easyocr")]
        # Near-identical box from VLM — should be skipped
        vlm_hits = [(existing_bbox, "A-vlm", 0.9)]
        merged = vlm.merge_novel_detections(existing, vlm_hits)
        self.assertEqual(len(merged), 1)

        # Distant box — should be kept
        novel = [[[50, 50], [80, 50], [80, 70], [50, 70]], "B", 0.9]
        merged = vlm.merge_novel_detections(existing, [novel])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1][3], "vlm")


class TestDetectTextIntegration(unittest.TestCase):

    def setUp(self):
        self.image = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.config = {"ocr": {"confidence_threshold": 0.5}, "safelist": []}
        self.mock_reader = MagicMock()

    def test_handwriting_rescue_then_redact(self):
        bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]
        self.mock_reader.readtext.return_value = [(bbox, "???", 0.1)]

        mock_hw = MagicMock()
        mock_hw.enabled = True
        mock_hw.enrich_results.return_value = [(bbox, "SECRET", 0.95, "handwriting")]

        bboxes, stats = detect_text(
            self.image,
            self.mock_reader,
            self.config,
            handwriting_recognizer=mock_hw,
        )
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(stats["handwriting_count"], 1)
        self.assertEqual(stats["detections"][0]["source"], "handwriting")
        self.assertEqual(stats["detections"][0]["decision"], "redacted")

    def test_vlm_adds_novel_region(self):
        self.mock_reader.readtext.return_value = []
        novel_bbox = [[10, 10], [50, 10], [50, 20], [10, 20]]

        mock_vlm = MagicMock()
        mock_vlm.enabled = True
        mock_vlm.detect.return_value = [(novel_bbox, "PHI NAME", 0.9)]
        mock_vlm.merge_novel_detections.side_effect = (
            lambda existing, hits: existing + [(b, t, p, "vlm") for b, t, p in hits]
        )

        bboxes, stats = detect_text(
            self.image,
            self.mock_reader,
            self.config,
            vlm_detector=mock_vlm,
        )
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(stats["vlm_count"], 1)
        self.assertEqual(stats["detections"][0]["source"], "vlm")

    def test_reader_uses_resolved_languages(self):
        config = {
            "ocr": {
                "languages": [],
                "language_preset": "medical_eu",
                "gpu_usage": False,
                "model_storage_directory": "",
                "download_enabled": True,
            },
            "ner": {"enabled": False},
        }
        transform = RedactPixelPHI(config=config)
        with patch("monai_aegis.transforms.pixel.easyocr.Reader", return_value=MagicMock()) as mock_reader:
            _ = transform.reader
        self.assertEqual(mock_reader.call_args[0][0], LANGUAGE_PRESETS["medical_eu"])


if __name__ == "__main__":
    unittest.main()
