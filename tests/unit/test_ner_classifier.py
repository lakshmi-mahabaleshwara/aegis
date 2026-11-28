"""
Unit tests for PHIClassifier (NER-based PHI detection).
"""
import unittest
from unittest.mock import MagicMock, patch

from monai_aegis.transforms.ner_classifier import PHIClassifier


class TestPHIClassifier(unittest.TestCase):
    """Tests for the NER-based PHIClassifier."""

    def setUp(self):
        self.config = {
            'ner': {
                'enabled': True,
                'model_name': 'StanfordAIMI/stanford-deidentifier-base',
                'device': 'cpu',
                'phi_labels': ['PATIENT', 'HCW', 'HOSPITAL', 'ID', 'DATE', 'PHONE'],
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
        required = {'PATIENT', 'HCW', 'HOSPITAL', 'DATE', 'ID', 'PHONE'}
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


# Entity labels emitted by the pinned StanfordAIMI/stanford-deidentifier-base
# revision (its config.id2label, minus 'O').
STANFORD_LABELS = {'PATIENT', 'HCW', 'HOSPITAL', 'ID', 'DATE', 'PHONE', 'VENDOR'}


def _packaged_config():
    from monai_aegis.cli import DEFAULT_CONFIG
    from monai_aegis.config.config_loader import load_config
    return load_config(DEFAULT_CONFIG)


def _rules_only_classifier(config):
    """Classifier whose NER layer never flags anything, so only the
    rule layers (age / clinical / heuristic) decide."""
    classifier = PHIClassifier(config)
    classifier._thread_local.pipeline = MagicMock(return_value=[])
    return classifier


def _fake_pipe(labels):
    pipe = MagicMock()
    pipe.model.config.id2label = {i: label for i, label in enumerate(labels)}
    return pipe


class TestPhiLabelsMatchModel(unittest.TestCase):
    """phi_labels must name labels the model actually emits — a mismatch
    (e.g. DOCTOR/IDNUM vs the model's HCW/ID) silently keeps PHI visible."""

    def test_packaged_phi_labels_are_model_labels(self):
        labels = set(_packaged_config()['ner']['phi_labels'])
        self.assertTrue(labels, "phi_labels must not be empty")
        self.assertEqual(labels - STANFORD_LABELS, set(),
                         "phi_labels lists labels the model never emits")

    def test_packaged_phi_labels_cover_every_phi_label(self):
        labels = set(_packaged_config()['ner']['phi_labels'])
        self.assertEqual(labels, STANFORD_LABELS - {'VENDOR'})

    def test_check_raises_when_no_label_overlaps(self):
        classifier = PHIClassifier({'ner': {'phi_labels': ['DOCTOR', 'IDNUM']}})
        with self.assertRaisesRegex(ValueError, "shares no label"):
            classifier.check_phi_labels(_fake_pipe(['O', 'B-PATIENT', 'I-PATIENT', 'B-HCW']))

    def test_check_warns_on_labels_the_model_never_emits(self):
        classifier = PHIClassifier({'ner': {'phi_labels': ['PATIENT', 'DOCTOR']}})
        with self.assertLogs("monai_aegis.transforms.ner_classifier", level="WARNING") as cm:
            classifier.check_phi_labels(_fake_pipe(['O', 'B-PATIENT', 'I-PATIENT']))
        self.assertIn("DOCTOR", "\n".join(r.getMessage() for r in cm.records))

    def test_check_ignores_models_without_label_map(self):
        classifier = PHIClassifier({'ner': {'phi_labels': ['PATIENT']}})
        classifier.check_phi_labels(object())  # no .model.config.id2label → no-op

    def test_label_mismatch_fails_closed_without_reloading_model(self):
        classifier = PHIClassifier({'ner': {'phi_labels': ['DOCTOR']}})
        with patch('transformers.pipeline', return_value=_fake_pipe(['O', 'PATIENT'])) as mock_hf:
            # NER errors are treated as PHI, so a mismatched config redacts
            # rather than silently preserving names...
            self.assertEqual(classifier.classify_texts(["Zorquil Blaxton", "Nevaeh Qwixby"]), [True, True])
        # ...and the model is loaded once, not once per text.
        self.assertEqual(mock_hf.call_count, 1)


class TestAgePatterns(unittest.TestCase):
    """The NER model has no AGE label; ages are caught by ner.age_patterns,
    which run before the clinical rules."""

    @classmethod
    def setUpClass(cls):
        cls.config = _packaged_config()

    def test_ages_are_redacted(self):
        ages = ["Age 57Y", "AGE: 57", "Pt age 57", "57Y", "57 Y", "57 yrs", "57yo",
                "57 y/o", "57 years old", "057Y", "006M", "003W", "010D",
                "M/57", "57/F", "6 months old", "3 days old"]
        classifier = _rules_only_classifier(self.config)
        self.assertEqual(dict(zip(ages, classifier.classify_texts(ages))),
                         {a: True for a in ages})

    def test_sex_age_overlays_beat_clinical_patterns(self):
        # "F 57Y" matches the ultrasound-frequency clinical pattern and
        # "F57Y" the probe-ID pattern; the age layer must win.
        classifier = _rules_only_classifier(self.config)
        self.assertTrue(classifier._is_clinical("F 57Y"))
        self.assertEqual(classifier.classify_texts(["F 57Y", "F57Y"]), [True, True])

    def test_clinical_text_is_not_mistaken_for_age(self):
        clinical = ["F H5.0", "FH5.0", "D 13.0", "G 25", "FR 25", "DR 130", "MI 1.14",
                    "TIS 0.3", "AP 81.28%", "Z 1.10", "SC5-1N", "C5-2", "5.7 cm",
                    "3.5 MHz", "Adult ABD", "B", "Stage 2", "Image 3", "Average 57",
                    "12w3d", "iClear 5", "Consona N9"]
        classifier = _rules_only_classifier(self.config)
        self.assertEqual(dict(zip(clinical, classifier.classify_texts(clinical))),
                         {c: False for c in clinical})


class TestNerTokenizerCasing(unittest.TestCase):
    """Regression guard for the transformers 5 breakage: the pinned model is
    uncased, and transformers 5 loads its tokenizer with do_lower_case=False,
    turning every capitalized word into [UNK] — NER then never sees names.
    Runs only when the model is already in the local HuggingFace cache."""

    def test_capitalized_names_are_not_unknown_tokens(self):
        ner = _packaged_config()['ner']
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                ner['model_name'], revision=ner['model_revision'], local_files_only=True)
        except Exception as exc:  # model not cached / transformers missing
            self.skipTest(f"NER tokenizer not available offline: {type(exc).__name__}")
        for text in ("John Smith", "MARIA GARCIA", "St Mary Hospital"):
            self.assertNotIn("[UNK]", tokenizer.tokenize(text), text)


if __name__ == '__main__':
    unittest.main()
