"""Concurrency tests: audit records must never cross-contaminate.

Regression tests for the shared-mutable-state bug where transforms
parked audit results (``last_stats`` / ``last_tag_actions``) on the
shared transform instance. Under threaded execution (ThreadDataLoader,
or a service embedding the pipeline), two in-flight files could swap
or destroy each other's audit records — corrupting the ground-truth
trail that compliance validation depends on.

The fix routes all audit data through return values
(:meth:`RedactPixelPHI.redact`, :meth:`ScrubDicomMetadata.scrub`),
so a single shared instance is safe. These tests hammer one instance
from multiple threads and assert every result matches its own input.
"""
import threading
import time
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset

from monai_aegis.transforms.metadata import ScrubDicomMetadata
from monai_aegis.transforms.pixel import RedactPixelPHI

N_ITERATIONS = 30


def _make_dataset(patient_name=None, patient_id=None) -> Dataset:
    """Build a minimal in-memory DICOM dataset with selected PII tags."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = 'US'
    if patient_name is not None:
        ds.PatientName = patient_name
    if patient_id is not None:
        ds.PatientID = patient_id
    return ds


class TestConcurrentScrubAuditIsolation(unittest.TestCase):
    """Two threads scrubbing different datasets through ONE shared
    ScrubDicomMetadata instance must each get their own tag actions."""

    def setUp(self):
        config = {
            'pii_mapping': {
                '(0010,0010)': 'DUMMY',   # PatientName
                '(0010,0020)': 'DUMMY',   # PatientID
            },
            'tokenization': {'salt': 'test-salt'},
        }
        self.transform = ScrubDicomMetadata(config=config)

    def test_tag_actions_do_not_cross_contaminate(self):
        # ds_a carries ONLY PatientName; ds_b carries ONLY PatientID —
        # so each thread's expected audit trail is distinguishable.
        ds_a = _make_dataset(patient_name='Alice Anderson')
        ds_b = _make_dataset(patient_id='MRN-B-12345')

        results = {'a': [], 'b': []}
        errors = []
        barrier = threading.Barrier(2)

        def worker(name, ds):
            try:
                barrier.wait()
                for _ in range(N_ITERATIONS):
                    _, actions = self.transform.scrub(uri=f'{name}.dcm', dataset=ds)
                    results[name].append([act['keyword'] for act in actions])
                    time.sleep(0)  # encourage interleaving
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=('a', ds_a)),
            threading.Thread(target=worker, args=('b', ds_b)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results['a']), N_ITERATIONS)
        self.assertEqual(len(results['b']), N_ITERATIONS)
        # Every audit trail for 'a' mentions exactly its own tag, never b's.
        for keywords in results['a']:
            self.assertEqual(keywords, ['PatientName'])
        for keywords in results['b']:
            self.assertEqual(keywords, ['PatientID'])

    def test_no_audit_state_left_on_instance(self):
        ds = _make_dataset(patient_name='Alice Anderson')
        self.transform.scrub(uri='a.dcm', dataset=ds)
        self.assertFalse(hasattr(self.transform, 'last_tag_actions'))


class TestConcurrentPixelAuditIsolation(unittest.TestCase):
    """Two threads redacting different images through ONE shared
    RedactPixelPHI instance must each get their own stats."""

    def setUp(self):
        config = {
            'ocr': {'languages': ['en'], 'confidence_threshold': 0.4},
            'ner': {'enabled': False},
            'safelist': [],
        }
        self.transform = RedactPixelPHI(config=config)

    def test_stats_match_own_image(self):
        # detect_text is mocked to report a detection count derived from
        # the image content, so each thread can verify it got back the
        # stats for the image it submitted.
        def fake_detect_text(pixel_array, reader, config, ner_classifier=None):
            marker = int(pixel_array.max())  # 10 for image A, 20 for image B
            time.sleep(0.001)  # widen the race window
            stats = {
                'total_detections': marker,
                'low_confidence_count': 0,
                'safelisted_count': 0,
                'ner_classified_count': 0,
                'redacted_count': 0,
                'detections': [],
            }
            return [], stats

        image_a = np.full((1, 16, 16), 10, dtype=np.uint8)
        image_b = np.full((1, 16, 16), 20, dtype=np.uint8)

        results = {'a': [], 'b': []}
        errors = []
        barrier = threading.Barrier(2)

        def worker(name, image, marker):
            try:
                barrier.wait()
                for _ in range(N_ITERATIONS):
                    _, stats = self.transform.redact(image)
                    results[name].append((stats['total_detections'], marker))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        with patch('monai_aegis.transforms.pixel.detect_text', side_effect=fake_detect_text), \
             patch.object(RedactPixelPHI, 'reader', new_callable=PropertyMock, return_value=None):
            threads = [
                threading.Thread(target=worker, args=('a', image_a, 10)),
                threading.Thread(target=worker, args=('b', image_b, 20)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(errors, [])
        for name in ('a', 'b'):
            self.assertEqual(len(results[name]), N_ITERATIONS)
            for got, expected in results[name]:
                self.assertEqual(
                    got, expected,
                    f"thread {name!r} received another thread's redaction stats",
                )


if __name__ == '__main__':
    unittest.main()
