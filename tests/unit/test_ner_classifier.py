"""
Unit tests for PHIClassifier (NER-based PHI detection).
"""
import unittest
from unittest.mock import MagicMock

from monai_aegis.transforms.ner_classifier import PHIClassifier


class TestPHIClassifier(unittest.TestCase):
    """Tests for the NER-based PHIClassifier."""

    def setUp(self):
        self.config = {
            'ner': {
                'enabled': True,
                'model_name': 'StanfordAIMI/stanford-deidentifier-base',
                'device': 'cpu',
                'phi_labels': [
                    'PATIENT', 'DOCTOR', 'USERNAME', 'IDNUM', 'MEDICALRECORD',
                    'HOSPITAL', 'DATE', 'AGE', 'PHONE', 'FAX', 'EMAIL', 'URL',
                    'SSN', 'ACCOUNT', 'LICENSE', 'STREET', 'CITY', 'STATE',
                    'ZIP', 'COUNTRY', 'ORGANIZATION', 'PROFESSION',
                ],
                'clinical_allowlist': ['mindray', 'ge', 'adult abd n', 'b'],
                'clinical_patterns': [
                    r'^(F\s*H?\d|D\s+\d|G\s+\d|FR\s+\d|DR\s+\d)',
                ],
                'phi_heuristic_patterns': [
                    r'(?i)(hospital|clinic|centre|center|diagnostic)',
                    r'^\d{8}[-/]\d{6}[-/]?\w*$',
                ],
            }
        }

    def _make_classifier_with_mock(self, side_effect):
        """Helper: create a PHIClassifier with a mocked pipeline."""
        classifier = PHIClassifier(self.config)
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = side_effect
        # Directly inject into thread-local storage (bypasses lazy loading)
        classifier._thread_local.pipeline = mock_pipeline
        return classifier

    def test_phi_detected(self):
        """Test that PHI text is correctly classified as PHI."""
        classifier = self._make_classifier_with_mock([
            [{'entity_group': 'PATIENT', 'score': 0.99, 'word': 'John', 'start': 0, 'end': 4}],
        ])
        results = classifier.classify_texts(["John Doe"])
        self.assertEqual(results, [True])

    def test_clinical_text_preserved(self):
        """Test that clinical/biomarker text is classified as non-PHI."""
        classifier = self._make_classifier_with_mock([
            [],  # "Depth 13.0" — no PHI entities detected
        ])
        results = classifier.classify_texts(["Depth 13.0"])
        self.assertEqual(results, [False])

    def test_empty_text_not_phi(self):
        """Test that empty strings are classified as non-PHI."""
        classifier = PHIClassifier(self.config)
        results = classifier.classify_texts(["", "  "])
        self.assertEqual(results, [False, False])

    def test_mixed_texts(self):
        """Test classification of mixed PHI and non-PHI texts."""
        classifier = self._make_classifier_with_mock([
            [{'entity_group': 'PATIENT', 'score': 0.99, 'word': 'John', 'start': 0, 'end': 4}],
            [],  # "B" — no PHI
            [{'entity_group': 'HOSPITAL', 'score': 0.95, 'word': 'Mayo', 'start': 0, 'end': 4}],
            [],  # "FH5.0" — no PHI
        ])
        results = classifier.classify_texts(["John Doe", "B", "Mayo Clinic", "FH5.0"])
        self.assertEqual(results, [True, False, True, False])

    def test_phi_labels_from_config(self):
        """Test that PHI labels are loaded from monai_aegis.config."""
        classifier = PHIClassifier(self.config)
        required = {'PATIENT', 'DOCTOR', 'HOSPITAL', 'DATE', 'IDNUM', 'PHONE', 'EMAIL'}
        self.assertTrue(required.issubset(classifier.phi_labels))

    def test_error_defaults_to_phi(self):
        """Test that classification errors default to treating text as PHI (safe default)."""
        classifier = self._make_classifier_with_mock(
            RuntimeError("Model error")
        )
        results = classifier.classify_texts(["Unknown text"])
        # Should default to PHI=True on error for safety
        self.assertEqual(results, [True])


_LOGGER = "monai_aegis.transforms.ner_classifier"


class TestPHIClassifierLoggingIsPhiFree(unittest.TestCase):
    """OCR fragments are burned-in PHI; log lines must never echo them raw."""

    def setUp(self):
        # An explicit salt makes the logged token deterministic and avoids
        # the default-salt warning noise.
        self.config = {
            'tokenization': {'salt': 'ner-log-test-salt'},
            'ner': {
                'enabled': True,
                'phi_labels': ['PATIENT', 'HOSPITAL'],
                'clinical_allowlist': ['depth 13.0'],
                'clinical_patterns': [],
                'phi_heuristic_patterns': [r'(?i)hospital'],
            },
        }

    def _classifier(self, side_effect=None):
        classifier = PHIClassifier(self.config)
        if side_effect is not None:
            mock_pipeline = MagicMock()
            mock_pipeline.side_effect = side_effect
            classifier._thread_local.pipeline = mock_pipeline
        return classifier

    def _assert_phi_free(self, records, secret):
        blob = "\n".join(r.getMessage() for r in records)
        self.assertNotIn(secret, blob, f"raw text leaked into logs: {blob!r}")
        self.assertIn("TOKEN_", blob, "expected a tokenized reference in the log")

    def test_heuristic_layer_logs_token_not_text(self):
        secret = "SUNNYVALE HOSPITAL"
        classifier = self._classifier()
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            self.assertEqual(classifier.classify_texts([secret]), [True])
        self._assert_phi_free(cm.records, secret)

    def test_clinical_layer_logs_token_not_text(self):
        secret = "Depth 13.0"
        classifier = self._classifier()
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            self.assertEqual(classifier.classify_texts([secret]), [False])
        self._assert_phi_free(cm.records, secret)

    def test_ner_layer_logs_token_not_text(self):
        secret = "Zorquil Blaxton"  # reaches the NER model (no earlier match)
        classifier = self._classifier(side_effect=[
            [{'entity_group': 'PATIENT', 'score': 0.99}],
        ])
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            self.assertEqual(classifier.classify_texts([secret]), [True])
        self._assert_phi_free(cm.records, secret)

    def test_error_path_logs_neither_text_nor_exception_message(self):
        secret = "Nevaeh Qwixby"
        classifier = self._classifier(
            side_effect=RuntimeError(f"model choked on {secret}")
        )
        with self.assertLogs(_LOGGER, level="ERROR") as cm:
            self.assertEqual(classifier.classify_texts([secret]), [True])
        blob = "\n".join(r.getMessage() for r in cm.records)
        # Neither the OCR text nor the exception message (which embeds it)
        # may appear — only the token and the exception type name.
        self.assertNotIn(secret, blob)
        self.assertIn("TOKEN_", blob)
        self.assertIn("RuntimeError", blob)


if __name__ == '__main__':
    unittest.main()
