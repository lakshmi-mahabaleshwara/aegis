"""
Unit tests for config.storage — AegisFileSystem (fsspec abstraction layer).

Uses fsspec's ``memory://`` protocol for isolated, fast tests with no disk I/O.
"""
import os
import tempfile
import unittest

import pydicom
import pydicom.uid
import numpy as np


from monai_aegis.config.storage import AegisFileSystem


class TestAegisFileSystemLocal(unittest.TestCase):
    """Test AegisFileSystem with local filesystem (protocol='file')."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fs = AegisFileSystem(protocol='file')

    def test_default_protocol_is_file(self):
        fs = AegisFileSystem()
        self.assertEqual(fs.protocol, 'file')

    def test_open_write_and_read(self):
        path = os.path.join(self.tmpdir, 'test.txt')
        with self.fs.open_write(path) as f:
            f.write(b'hello aegis')
        with self.fs.open_read(path) as f:
            content = f.read()
        self.assertEqual(content, b'hello aegis')

    def test_makedirs_creates_nested(self):
        path = os.path.join(self.tmpdir, 'a', 'b', 'c')
        self.fs.makedirs(path)
        self.assertTrue(os.path.isdir(path))

    def test_exists(self):
        path = os.path.join(self.tmpdir, 'exists.txt')
        self.assertFalse(self.fs.exists(path))
        with open(path, 'w') as f:
            f.write('test')
        self.assertTrue(self.fs.exists(path))

    def test_isdir(self):
        self.assertTrue(self.fs.isdir(self.tmpdir))
        path = os.path.join(self.tmpdir, 'file.txt')
        with open(path, 'w') as f:
            f.write('test')
        self.assertFalse(self.fs.isdir(path))

    def test_walk(self):
        # Create nested structure
        sub = os.path.join(self.tmpdir, 'sub')
        os.makedirs(sub)
        with open(os.path.join(self.tmpdir, 'a.txt'), 'w') as f:
            f.write('a')
        with open(os.path.join(sub, 'b.txt'), 'w') as f:
            f.write('b')

        walked = list(self.fs.walk(self.tmpdir))
        # At least root and sub
        self.assertGreaterEqual(len(walked), 2)
        root_entry = walked[0]
        self.assertIn('a.txt', root_entry[2])

    def test_relpath(self):
        result = self.fs.relpath('/a/b/c/file.txt', '/a/b')
        self.assertEqual(result, 'c/file.txt')

    def test_join(self):
        result = self.fs.join('/base', 'sub', 'file.txt')
        self.assertEqual(result, os.path.join('/base', 'sub', 'file.txt'))

    def test_basename(self):
        self.assertEqual(self.fs.basename('/a/b/c/file.dcm'), 'file.dcm')

    def test_dirname(self):
        self.assertEqual(self.fs.dirname('/a/b/c/file.dcm'), '/a/b/c')

    def test_repr(self):
        self.assertEqual(repr(self.fs), "AegisFileSystem(protocol='file')")


class TestAegisFileSystemFromConfig(unittest.TestCase):
    """Test from_config factory method."""

    def test_default_when_no_storage_section(self):
        config = {'paths': {'input_dir': '/input'}}
        fs = AegisFileSystem.from_config(config)
        self.assertEqual(fs.protocol, 'file')

    def test_explicit_protocol(self):
        config = {'storage': {'protocol': 'memory', 'options': {}}}
        fs = AegisFileSystem.from_config(config)
        self.assertEqual(fs.protocol, 'memory')

    def test_options_passed_through(self):
        config = {
            'storage': {
                'protocol': 'file',
                'options': {'auto_mkdir': True},
            }
        }
        fs = AegisFileSystem.from_config(config)
        self.assertEqual(fs.protocol, 'file')


class TestExperimentalStorageWarning(unittest.TestCase):
    """Non-file protocols announce their experimental status once per process."""

    _LOGGER = "monai_aegis.config.storage"

    def setUp(self):
        from monai_aegis.config import storage as storage_mod
        self._storage_mod = storage_mod
        storage_mod._experimental_warned.clear()

    def tearDown(self):
        self._storage_mod._experimental_warned.clear()

    def test_remote_protocol_warns_and_only_once(self):
        with self.assertLogs(self._LOGGER, level="WARNING") as cm:
            AegisFileSystem(protocol="memory")
        blob = "\n".join(cm.output)
        self.assertIn("EXPERIMENTAL", blob)
        self.assertIn("memory", blob)
        # Same protocol again in the same process must not re-warn.
        with self.assertNoLogs(self._LOGGER, level="WARNING"):
            AegisFileSystem(protocol="memory")

    def test_file_protocol_is_silent(self):
        with self.assertNoLogs(self._LOGGER, level="WARNING"):
            AegisFileSystem(protocol="file")


class TestAegisFileSystemMemory(unittest.TestCase):
    """Test AegisFileSystem with in-memory filesystem (no disk I/O)."""

    def setUp(self):
        self.fs = AegisFileSystem(protocol='memory')

    def test_write_and_read_memory(self):
        with self.fs.open_write('/mem/test.bin') as f:
            f.write(b'\x00\x01\x02\x03')
        with self.fs.open_read('/mem/test.bin') as f:
            data = f.read()
        self.assertEqual(data, b'\x00\x01\x02\x03')

    def test_exists_memory(self):
        self.assertFalse(self.fs.exists('/mem/nope.txt'))
        with self.fs.open_write('/mem/yes.txt') as f:
            f.write(b'data')
        self.assertTrue(self.fs.exists('/mem/yes.txt'))


class TestDicomByteStreamRoundTrip(unittest.TestCase):
    """Test that pydicom works with byte streams via AegisFileSystem."""

    def test_pydicom_read_write_via_fs(self):
        """Verify pydicom.dcmread / save_as work with file-like objects."""
        fs = AegisFileSystem(protocol='memory')

        # Create a minimal DICOM dataset
        ds = pydicom.Dataset()
        ds.PatientName = 'Test^Patient'
        ds.PatientID = '12345'
        ds.Modality = 'CT'
        ds.Rows = 4
        ds.Columns = 4
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelRepresentation = 0
        ds.PixelData = np.zeros((4, 4), dtype=np.uint8).tobytes()
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()

        ds.is_little_endian = True
        ds.is_implicit_VR = True

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds.file_meta = file_meta
        ds.preamble = b'\x00' * 128

        # Write via fs
        with fs.open_write('/dicom/test.dcm') as f:
            pydicom.dcmwrite(f, ds)

        # Read back via fs
        with fs.open_read('/dicom/test.dcm') as f:
            loaded = pydicom.dcmread(f)

        self.assertEqual(loaded.PatientName, 'Test^Patient')
        self.assertEqual(loaded.PatientID, '12345')
        self.assertEqual(loaded.Modality, 'CT')


if __name__ == '__main__':
    unittest.main()
