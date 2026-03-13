"""Unit tests for DICOM discovery, grouping, validation, and sorting."""
import unittest
import tempfile
import os

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np

from transforms.discovery import (
    DicomSliceInfo,
    discover_dicoms,
    group_into_series,
    validate_series,
    sort_slices,
    ACCEPTED_MODALITIES,
)
from transforms.exceptions import SeriesLoadError


def _make_dcm(path, modality='CT', rows=64, cols=64, instance_num=1,
              series_uid='1.2.3', study_uid='1.2.4', sop_uid=None,
              ipp=None, pixel_spacing=None, slice_thickness=1.0):
    """Create a minimal valid DICOM file for testing."""
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = sop_uid or pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b'\x00' * 128)
    ds.Modality = modality
    ds.Rows = rows
    ds.Columns = cols
    ds.InstanceNumber = instance_num
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()

    if ipp:
        ds.ImagePositionPatient = ipp
    if pixel_spacing:
        ds.PixelSpacing = pixel_spacing
    if slice_thickness:
        ds.SliceThickness = slice_thickness

    ds.save_as(path)
    return ds


class TestDiscoverDicoms(unittest.TestCase):
    """Test discover_dicoms scanning and filtering."""

    def test_discovers_valid_dicoms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_dcm(os.path.join(tmpdir, 'slice1.dcm'))
            _make_dcm(os.path.join(tmpdir, 'slice2.dcm'), instance_num=2)
            slices = discover_dicoms(tmpdir)
            self.assertEqual(len(slices), 2)

    def test_skips_non_imaging_modality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_dcm(os.path.join(tmpdir, 'report.dcm'), modality='SR')
            slices = discover_dicoms(tmpdir)
            self.assertEqual(len(slices), 0)

    def test_raises_on_invalid_folder(self):
        with self.assertRaises(SeriesLoadError):
            discover_dicoms('/nonexistent/folder')

    def test_recursive_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, 'sub')
            os.makedirs(subdir)
            _make_dcm(os.path.join(subdir, 's1.dcm'))
            slices = discover_dicoms(tmpdir)
            self.assertEqual(len(slices), 1)


class TestGroupIntoSeries(unittest.TestCase):

    def test_groups_by_study_and_series(self):
        s1 = DicomSliceInfo('a.dcm', 'sop1', 'study1', 'series1', 1, None, 'CT', 64, 64)
        s2 = DicomSliceInfo('b.dcm', 'sop2', 'study1', 'series1', 2, None, 'CT', 64, 64)
        s3 = DicomSliceInfo('c.dcm', 'sop3', 'study1', 'series2', 1, None, 'CT', 64, 64)
        groups = group_into_series([s1, s2, s3])
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[('study1', 'series1')]), 2)
        self.assertEqual(len(groups[('study1', 'series2')]), 1)


class TestValidateSeries(unittest.TestCase):

    def test_consistent_geometry_single_group(self):
        s1 = DicomSliceInfo('a.dcm', 'sop1', 'study1', 'series1', 1, None, 'CT', 512, 512, [0.5, 0.5], 1.0)
        s2 = DicomSliceInfo('b.dcm', 'sop2', 'study1', 'series1', 2, None, 'CT', 512, 512, [0.5, 0.5], 1.0)
        result = validate_series([s1, s2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_mixed_geometry_splits(self):
        s1 = DicomSliceInfo('a.dcm', 'sop1', 'study1', 'series1', 1, None, 'CT', 512, 512, [0.5, 0.5], 1.0)
        s2 = DicomSliceInfo('b.dcm', 'sop2', 'study1', 'series1', 2, None, 'CT', 256, 256, [1.0, 1.0], 5.0)
        result = validate_series([s1, s2])
        self.assertEqual(len(result), 2)  # Split into two sub-series


class TestSortSlices(unittest.TestCase):

    def test_sorts_by_ipp(self):
        s1 = DicomSliceInfo('a.dcm', 'sop1', 'st', 'se', 1, [0, 0, 30.0], 'CT', 64, 64)
        s2 = DicomSliceInfo('b.dcm', 'sop2', 'st', 'se', 2, [0, 0, 10.0], 'CT', 64, 64)
        s3 = DicomSliceInfo('c.dcm', 'sop3', 'st', 'se', 3, [0, 0, 20.0], 'CT', 64, 64)
        result = sort_slices([s1, s2, s3])
        self.assertEqual([s.uri for s in result], ['b.dcm', 'c.dcm', 'a.dcm'])

    def test_falls_back_to_instance_number(self):
        s1 = DicomSliceInfo('a.dcm', 'sop1', 'st', 'se', 3, None, 'CT', 64, 64)
        s2 = DicomSliceInfo('b.dcm', 'sop2', 'st', 'se', 1, None, 'CT', 64, 64)
        result = sort_slices([s1, s2])
        self.assertEqual(result[0].instance_number, 1)

    def test_filename_fallback(self):
        s1 = DicomSliceInfo('z.dcm', 'sop1', 'st', 'se', None, None, 'CT', 64, 64)
        s2 = DicomSliceInfo('a.dcm', 'sop2', 'st', 'se', None, None, 'CT', 64, 64)
        result = sort_slices([s1, s2])
        self.assertEqual(result[0].uri, 'a.dcm')


if __name__ == '__main__':
    unittest.main()
