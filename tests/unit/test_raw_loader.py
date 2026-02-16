import unittest
import numpy as np
from PIL import Image
import io

from transforms.io import LoadDicomRawd


class TestLoadDicomRawd(unittest.TestCase):
    """Unit tests for LoadDicomRawd transform"""

    def setUp(self):
        self.loader = LoadDicomRawd(keys=['image'])

    def test_jpeg_loading(self):
        """Test loading JPEG images"""
        # Create a dummy JPEG in memory
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            jpg_path = f.name
            img.save(jpg_path)
        
        try:
            data = {'image': jpg_path}
            result = self.loader(data)
            
            # Verify shape is (3, H, W) - channel first
            self.assertEqual(result['image'].ndim, 3)
            self.assertEqual(result['image'].shape[0], 3)
            
            # Verify metadata dict exists
            self.assertIn('image_meta_dict', result)
            self.assertEqual(result['image_meta_dict']['filename_or_obj'], jpg_path)
        finally:
            import os
            os.unlink(jpg_path)

    def test_grayscale_jpeg_loading(self):
        """Test loading grayscale JPEG images"""
        img = Image.new('L', (10, 10), color=128)
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            jpg_path = f.name
            img.save(jpg_path)
        
        try:
            data = {'image': jpg_path}
            result = self.loader(data)
            
            # Verify shape is (1, H, W) - channel first for grayscale
            self.assertEqual(result['image'].ndim, 3)
            self.assertEqual(result['image'].shape[0], 1)
        finally:
            import os
            os.unlink(jpg_path)


if __name__ == '__main__':
    unittest.main()
