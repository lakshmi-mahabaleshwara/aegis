"""Unit tests for volume redaction in RedactPixelPHI."""
import unittest
import numpy as np

from transforms.pixel import RedactPixelPHI


class TestVolumeRedaction(unittest.TestCase):
    """Test that RedactPixelPHI handles 4D volumes correctly."""

    def _make_config(self):
        return {
            'ocr': {
                'decoder': 'beamsearch',
                'beam_width': 5,
                'confidence_threshold': 0.4,
                'languages': ['en'],
                'gpu_usage': False,
            },
            'ner': {'enabled': False},
            'safelist': [],
            'series': {'keyframe_count': 3},
        }

    def test_volume_shape_preserved(self):
        """Test that 4D volume shape (C,D,H,W) is preserved after redaction."""
        config = self._make_config()
        transform = RedactPixelPHI(config=config)

        # Create a blank volume — no text should be detected
        volume = np.zeros((1, 5, 64, 64), dtype=np.float32)
        result = transform(volume)

        self.assertEqual(result.shape, (1, 5, 64, 64))
        self.assertEqual(result.dtype, np.float32)

    def test_2d_still_works(self):
        """Test that existing 3D (C,H,W) images still work."""
        config = self._make_config()
        transform = RedactPixelPHI(config=config)

        image = np.zeros((1, 64, 64), dtype=np.float32)
        result = transform(image)

        self.assertEqual(result.shape, (1, 64, 64))

    def test_volume_stats_contain_strategy(self):
        """Test that last_stats contains volume-specific fields."""
        config = self._make_config()
        transform = RedactPixelPHI(config=config)

        volume = np.zeros((1, 3, 32, 32), dtype=np.float32)
        transform(volume)

        self.assertIn('volume_strategy', transform.last_stats)
        self.assertIn('num_slices', transform.last_stats)
        self.assertIn('keyframe_indices', transform.last_stats)
        self.assertEqual(transform.last_stats['num_slices'], 3)

    def test_small_volume_all_slices_as_keyframes(self):
        """If D <= keyframe_count, all slices should be keyframes."""
        config = self._make_config()
        config['series']['keyframe_count'] = 5
        transform = RedactPixelPHI(config=config)

        volume = np.zeros((1, 2, 32, 32), dtype=np.float32)
        transform(volume)

        # With 2 slices and keyframe_count=5, all 2 slices are keyframes
        self.assertEqual(transform.last_stats['keyframe_indices'], [0, 1])


if __name__ == '__main__':
    unittest.main()
