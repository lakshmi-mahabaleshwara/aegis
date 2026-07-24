"""Tests for offline / pinned-model support (air-gapped deployment).

Covers:
  - ``as_bool`` coercion (env-interpolated config values arrive as strings)
  - NER model revision pinning + local-files-only mode
  - EasyOCR configurable weights directory + download kill-switch
"""
import os
import unittest
from unittest.mock import MagicMock, patch

from monai_aegis.config.config_loader import as_bool
from monai_aegis.transforms.ner_classifier import PHIClassifier
from monai_aegis.transforms.pixel import RedactPixelPHI


class TestAsBool(unittest.TestCase):

    def test_real_booleans_pass_through(self):
        self.assertTrue(as_bool(True))
        self.assertFalse(as_bool(False, default=True))

    def test_env_style_strings(self):
        # ${VAR:default} interpolation produces strings — 'false' must not be truthy
        for truthy in ('true', 'True', '1', 'yes', 'on'):
            self.assertTrue(as_bool(truthy))
        for falsy in ('false', 'False', '0', 'no', 'off'):
            self.assertFalse(as_bool(falsy, default=True))

    def test_empty_and_none_use_default(self):
        self.assertTrue(as_bool('', default=True))
        self.assertFalse(as_bool(None, default=False))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            as_bool('maybe')


class TestNerModelPinning(unittest.TestCase):

    def _config(self, **ner_overrides):
        ner = {
            'enabled': True,
            'model_name': 'StanfordAIMI/stanford-deidentifier-base',
            'model_revision': 'abc123def456',
            'local_files_only': 'false',
            'device': 'cpu',
            'phi_labels': ['PATIENT'],
        }
        ner.update(ner_overrides)
        return {'ner': ner}

    def test_revision_passed_to_hf_pipeline(self):
        classifier = PHIClassifier(self._config())
        with patch('transformers.pipeline', return_value=MagicMock()) as mock_hf:
            _ = classifier.pipeline
        kwargs = mock_hf.call_args[1]
        self.assertEqual(kwargs['revision'], 'abc123def456')
        # local_files_only must NOT be forwarded via model_kwargs — transformers
        # >= 5 sets it itself and a duplicate raises "got multiple values for
        # keyword argument 'local_files_only'". Offline is driven by env vars.
        self.assertNotIn('local_files_only', kwargs.get('model_kwargs', {}))

    def test_empty_revision_floats_on_default_branch(self):
        classifier = PHIClassifier(self._config(model_revision=''))
        self.assertIsNone(classifier.model_revision)

    def test_shipped_config_pins_the_revision(self):
        # The packaged config.yaml must carry an explicit revision pin so
        # deployments never float on the model repo's default branch.
        from monai_aegis.api import default_config_path
        from monai_aegis.config.config_loader import load_config
        classifier = PHIClassifier(load_config(default_config_path()))
        self.assertIsNotNone(classifier.model_revision)
        self.assertNotIn('${', classifier.model_revision)

    def test_local_files_only_sets_offline_env(self):
        classifier = PHIClassifier(self._config(local_files_only='true'))
        self.assertTrue(classifier.local_files_only)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            with patch('transformers.pipeline', return_value=MagicMock()) as mock_hf:
                _ = classifier.pipeline
            # Offline is enforced through the env vars, not a model_kwargs flag.
            self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '1')
            self.assertEqual(os.environ.get('TRANSFORMERS_OFFLINE'), '1')
            self.assertNotIn(
                'local_files_only', mock_hf.call_args[1].get('model_kwargs', {})
            )


class TestEasyOcrOfflineConfig(unittest.TestCase):

    def test_model_dir_and_download_flag_passed_to_reader(self):
        config = {
            'ocr': {
                'languages': ['en'],
                'gpu_usage': False,
                'model_storage_directory': '/models/easyocr',
                'download_enabled': 'false',
            },
            'ner': {'enabled': False},
        }
        transform = RedactPixelPHI(config=config)
        with patch('monai_aegis.transforms.pixel.easyocr.Reader', return_value=MagicMock()) as mock_reader:
            _ = transform.reader
        kwargs = mock_reader.call_args[1]
        self.assertEqual(kwargs['model_storage_directory'], '/models/easyocr')
        self.assertFalse(kwargs['download_enabled'])

    def test_empty_model_dir_falls_back_to_easyocr_default(self):
        config = {
            'ocr': {'languages': ['en'], 'model_storage_directory': ''},
            'ner': {'enabled': False},
        }
        transform = RedactPixelPHI(config=config)
        with patch('monai_aegis.transforms.pixel.easyocr.Reader', return_value=MagicMock()) as mock_reader:
            _ = transform.reader
        kwargs = mock_reader.call_args[1]
        self.assertIsNone(kwargs['model_storage_directory'])
        self.assertTrue(kwargs['download_enabled'])  # default stays permissive


if __name__ == '__main__':
    unittest.main()
