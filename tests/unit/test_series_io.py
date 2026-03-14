"""Unit tests for series I/O transforms."""
import unittest
import tempfile
import os

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

from monai_aegis.transforms.series_io import LoadDicomSeries, LoadDicomSeriesd, SaveDicomSeries
from monai_aegis.transforms.exceptions import SeriesLoadError, SeriesSaveError


def _make_dcm_dataset(rows=64, cols=64, instance_num=1, modality='CT',
                      series_uid='1.2.3', study_uid='1.2.4'):
    """Create a minimal pydicom Dataset with pixel data (in-memory)."""
    ds = pydicom.Dataset()
    ds.Modality = modality
    ds.Rows = rows
    ds.Columns = cols
    ds.InstanceNumber = instance_num
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.PatientID = 'TEST_PATIENT'
    ds.StudyDate = '20260101'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelData = np.random.randint(0, 4096, (rows, cols), dtype=np.uint16).tobytes()

    # file_meta for saving
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    return ds


def _save_dcm(ds, path):
    """Save a dataset to file with preamble."""
    ds.save_as(path, write_like_original=False)


def _make_multiframe_dataset(rows=64, cols=64, frames=3, modality='CT',
                             series_uid='1.2.3', study_uid='1.2.4'):
    """Create a minimal multi-frame DICOM dataset."""
    ds = _make_dcm_dataset(
        rows=rows,
        cols=cols,
        instance_num=1,
        modality=modality,
        series_uid=series_uid,
        study_uid=study_uid,
    )
    ds.NumberOfFrames = str(frames)
    ds.PixelData = np.random.randint(
        0, 4096, (frames, rows, cols), dtype=np.uint16
    ).tobytes()
    return ds


class TestLoadDicomSeries(unittest.TestCase):
    """Test the LoadDicomSeries array transform."""

    def test_loads_multifile_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            datasets = []
            for i in range(3):
                fp = os.path.join(tmpdir, f'slice_{i}.dcm')
                ds = _make_dcm_dataset(instance_num=i + 1)
                _save_dcm(ds, fp)
                paths.append(fp)
                datasets.append(ds)

            loader = LoadDicomSeries()
            result = loader(paths, datasets=datasets)

            # Should be (1, 3, 64, 64) — 1 channel, 3 slices
            self.assertEqual(result.shape, (1, 3, 64, 64))
            self.assertIn('series_instance_uid', result.meta)
            self.assertEqual(result.meta['num_slices'], 3)
            self.assertFalse(result.meta['is_multiframe'])

    def test_empty_uris_raises(self):
        loader = LoadDicomSeries()
        with self.assertRaises(SeriesLoadError):
            loader([])

    def test_loads_multiframe_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, 'multiframe.dcm')
            ds = _make_multiframe_dataset(frames=4)
            _save_dcm(ds, fp)

            loader = LoadDicomSeries()
            result = loader([fp], datasets=[ds])

            self.assertEqual(result.shape, (1, 4, 64, 64))
            self.assertTrue(result.meta['is_multiframe'])
            self.assertEqual(result.meta['num_slices'], 4)


class TestLoadDicomSeriesd(unittest.TestCase):
    """Test the LoadDicomSeriesd dictionary transform."""

    def test_loads_series_from_file_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(2):
                fp = os.path.join(tmpdir, f'slice_{i}.dcm')
                ds = _make_dcm_dataset(instance_num=i + 1)
                _save_dcm(ds, fp)
                paths.append(fp)

            loader = LoadDicomSeriesd(keys=['image'], config={})
            data = loader({'image': paths})

            self.assertEqual(data['image'].shape, (1, 2, 64, 64))
            self.assertIn('image_dicom_datasets', data)
            self.assertEqual(len(data['image_dicom_datasets']), 2)
            self.assertIn('image_meta_dict', data)


class TestSaveDicomSeries(unittest.TestCase):
    """Test the SaveDicomSeries array transform."""

    def test_saves_series_with_regenerated_uids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake input directory with original files
            input_dir = os.path.join(tmpdir, 'input', 'sub')
            os.makedirs(input_dir)
            original_paths = []
            datasets = []
            for i in range(3):
                ds = _make_dcm_dataset(instance_num=i + 1)
                fp = os.path.join(input_dir, f'slice_{i}.dcm')
                _save_dcm(ds, fp)
                original_paths.append(fp)
                datasets.append(ds)

            original_uids = [ds.SOPInstanceUID for ds in datasets]
            output_dir = os.path.join(tmpdir, 'output')

            saver = SaveDicomSeries(
                output_dir=output_dir,
                input_dir=os.path.join(tmpdir, 'input'),
            )
            paths = saver(
                datasets=datasets,
                original_uris=original_paths,
            )

            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(os.path.exists(p))
                # Verify folder structure preserved: output/sub/slice_X.dcm
                self.assertIn('sub', p)

            # SOPInstanceUIDs should have been regenerated
            for ds, orig_uid in zip(datasets, original_uids):
                self.assertNotEqual(ds.SOPInstanceUID, orig_uid)


if __name__ == '__main__':
    unittest.main()
