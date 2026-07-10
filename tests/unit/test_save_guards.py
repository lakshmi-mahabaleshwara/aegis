"""Savers must fail loudly when the scrub stage did not run.

A DICOM that was loaded but never scrubbed must never be silently
skipped by the save stage — silently dropping un-de-identified data
hides a broken pipeline from the operator and the audit trail.
"""
import tempfile
import unittest

from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck
from monai_aegis.transforms.exceptions import DicomSaveError, SeriesSaveError
from monai_aegis.transforms.io import SaveDicomd
from monai_aegis.transforms.series_io import SaveDicomSeriesd

from tests.unit.test_concurrent_audit import _make_dataset


class TestSaveDicomdGuard(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saver = SaveDicomd(keys=['image'], output_dir=self.tmpdir)

    def test_loaded_but_unscrubbed_dicom_raises(self):
        data = {
            'image': object(),
            ck('image', ckeys.DICOM_DATASET): _make_dataset(patient_name='Alice'),
            ck('image', ckeys.META_DICT): {'filename_or_obj': 'a.dcm'},
            # no SCRUBBED_DS — ScrubDicomMetadatad never ran
        }
        with self.assertRaises(DicomSaveError):
            self.saver(data)

    def test_non_dicom_item_is_skipped(self):
        data = {
            'image': object(),
            ck('image', ckeys.META_DICT): {'filename_or_obj': 'a.png'},
            # no cached DICOM dataset → image pipeline item, nothing to save
        }
        result = self.saver(data)  # must not raise
        self.assertNotIn(ck('image', ckeys.SAVED_PATH), result)


class TestSaveDicomSeriesdGuard(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saver = SaveDicomSeriesd(keys=['image'], output_dir=self.tmpdir)

    def test_loaded_but_unscrubbed_series_raises(self):
        data = {
            'image': object(),
            ck('image', ckeys.DICOM_DATASETS): [_make_dataset(patient_name='Alice')],
            ck('image', ckeys.META_DICT): {'filename_or_obj': 'a.dcm', 'slice_uris': ['a.dcm']},
            # no SCRUBBED_DATASETS — ScrubDicomMetadatad never ran
        }
        with self.assertRaises(SeriesSaveError):
            self.saver(data)

    def test_unscrubbed_multiframe_raises(self):
        data = {
            'image': object(),
            ck('image', ckeys.DICOM_DATASETS): [_make_dataset(patient_name='Alice')],
            ck('image', ckeys.META_DICT): {
                'filename_or_obj': 'a.dcm',
                'slice_uris': ['a.dcm'],
                'is_multiframe': True,
            },
            # no SCRUBBED_DS
        }
        with self.assertRaises(SeriesSaveError):
            self.saver(data)


if __name__ == '__main__':
    unittest.main()
