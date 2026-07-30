import unittest
import os
import shutil
import tempfile
import numpy as np
import pydicom
from PIL import Image

from monai_aegis.transforms.pipeline import build_pipeline, build_image_pipeline

class TestAegisPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = 'tests/integration/temp_data'
        self.output_dir = 'tests/integration/temp_output'
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create Dummy DICOM
        self.dcm_path = os.path.join(self.test_dir, 'test.dcm')
        ds = pydicom.Dataset()
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        
        # File Meta
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageSOPInstanceUID = '1.2.3'
        file_meta.ImplementationClassUID = '1.2.3.4'
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds.file_meta = file_meta
        
        # Preamble is critical for dcmread without force=True
        ds.preamble = b"\0" * 128
        
        # Create pixel data (10x10)
        pixel_array = np.zeros((10, 10), dtype=np.uint8)
        ds.PixelData = pixel_array.tobytes()
        ds.Rows = 10
        ds.Columns = 10
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        ds.save_as(self.dcm_path, write_like_original=False)
        
        # Create Dummy JPEG
        self.jpg_path = os.path.join(self.test_dir, 'test.jpg')
        img = Image.new('RGB', (10, 10), color='white')
        img.save(self.jpg_path)

        # Build Pipelines
        self.pipeline = build_pipeline()
        self.image_pipeline = build_image_pipeline()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_pipeline_execution(self):
        # Build pipeline with output_dir
        self.pipeline = build_pipeline(output_dir=self.output_dir)

        # 1. Run on DICOM
        data = {'image': self.dcm_path}
        
        # Capture modification time before run
        mtime_before = os.path.getmtime(self.dcm_path)
        
        result = self.pipeline(data)
        
        # Check result keys
        self.assertIn('image', result)
        self.assertTrue(hasattr(result['image'], 'shape'))
        
        # Verify Non-Destructive behavior on INPUT file
        mtime_after = os.path.getmtime(self.dcm_path)
        self.assertEqual(mtime_before, mtime_after, "Input DICOM should not be modified!")
        
        ds_input = pydicom.dcmread(self.dcm_path)
        self.assertIn('PatientName', ds_input, "Input DICOM should still have PatientName!")

        # Verify Redacted Output in OUTPUT Dir
        expected_output_path = os.path.join(self.output_dir, 'test.dcm')
        self.assertTrue(os.path.exists(expected_output_path), "Output DICOM not found in output_dir!")
        
        ds_output = pydicom.dcmread(expected_output_path)
        # PatientName should be tokenized (DUMMY action), not removed
        self.assertIn('PatientName', ds_output, "PatientName should still exist (tokenized)")
        patient_name = str(ds_output.PatientName)
        self.assertTrue(patient_name.startswith('TOKEN_'), f"PatientName should be tokenized, got: {patient_name}")

        # 2. Run on JPEG via the image pipeline
        self.image_pipeline = build_image_pipeline(output_dir=self.output_dir)
        data = {'image': self.jpg_path}
        result = self.image_pipeline(data)
        self.assertIn('image', result)
        self.assertTrue(hasattr(result['image'], 'shape'))


    def test_missing_file(self):
        # Path to non-existent file
        data = {'image': os.path.join(self.test_dir, 'ghost.dcm')}
        
        with self.assertRaises(Exception):
            self.pipeline(data)

    def test_minimal_config(self):
        min_config_path = os.path.join(self.test_dir, 'min_config.yaml')
        with open(min_config_path, 'w') as f:
            f.write("ocr:\n  languages: ['en']\npii_mapping: {}\n")
            
        pipeline = build_image_pipeline(config_path=os.path.abspath(min_config_path))

        # Run on JPEG
        data = {'image': self.jpg_path}
        result = pipeline(data)
        self.assertIn('image', result)


class TestConfigPathResolution(unittest.TestCase):
    """Config-path resolution in the pipeline builders."""

    def test_default_config_path_is_absolute_and_exists(self):
        from monai_aegis.transforms.pipeline import DEFAULT_CONFIG_PATH
        self.assertTrue(os.path.isabs(DEFAULT_CONFIG_PATH))
        self.assertTrue(os.path.isfile(DEFAULT_CONFIG_PATH))

    def test_relative_config_path_resolves_against_cwd(self):
        from unittest.mock import patch
        from monai_aegis.transforms import pipeline as P

        captured = {}

        def fake_load(config_path, overlay_path=None):
            captured['config_path'] = config_path
            return {}

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with patch.object(P, 'load_config', side_effect=fake_load):
                    P._resolve_builder_context('sub/my.yaml')
            finally:
                os.chdir(cwd)

        # Resolved against the working directory, not the transforms/ package dir.
        self.assertEqual(captured['config_path'], os.path.join(os.path.realpath(tmp), 'sub', 'my.yaml'))
        self.assertNotIn('transforms', captured['config_path'])

    def test_absolute_config_path_passed_through_unchanged(self):
        from unittest.mock import patch
        from monai_aegis.transforms import pipeline as P

        captured = {}

        def fake_load(config_path, overlay_path=None):
            captured['config_path'] = config_path
            return {}

        abs_path = os.path.abspath('/tmp/aegis/explicit.yaml')
        with patch.object(P, 'load_config', side_effect=fake_load):
            P._resolve_builder_context(abs_path)
        self.assertEqual(captured['config_path'], abs_path)


if __name__ == '__main__':
    unittest.main()
