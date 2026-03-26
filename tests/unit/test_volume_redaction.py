"""Unit tests for volume redaction in RedactPixelPHI."""
import unittest
import numpy as np

from monai_aegis.transforms.pixel import (
    RedactPixelPHI,
    select_keyframe_indices,
    build_union_mask,
    redact_volume_safe,
)


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


class TestSelectKeyframeIndices(unittest.TestCase):
    """Unit tests for select_keyframe_indices."""

    def test_below_floor_returns_all(self):
        # T=2, floor=3 → scan all 2 frames
        self.assertEqual(select_keyframe_indices(2, floor=3), [0, 1])

    def test_at_floor_returns_all(self):
        self.assertEqual(select_keyframe_indices(3, floor=3), [0, 1, 2])

    def test_always_includes_first_and_last(self):
        indices = select_keyframe_indices(100, sample_pct=0.10, floor=3, ceiling=25)
        self.assertIn(0, indices)
        self.assertIn(99, indices)

    def test_10pct_of_100_frames(self):
        indices = select_keyframe_indices(100, sample_pct=0.10, floor=3, ceiling=25)
        self.assertEqual(len(indices), 10)

    def test_ceiling_caps_large_series(self):
        indices = select_keyframe_indices(2000, sample_pct=0.10, floor=3, ceiling=25)
        self.assertLessEqual(len(indices), 25)

    def test_floor_applies_to_sparse_series(self):
        # 10% of 15 = 1.5 → rounds to 2, but floor=3
        indices = select_keyframe_indices(15, sample_pct=0.10, floor=3, ceiling=25)
        self.assertGreaterEqual(len(indices), 3)

    def test_indices_are_sorted_and_unique(self):
        indices = select_keyframe_indices(50)
        self.assertEqual(indices, sorted(set(indices)))

    def test_empty_volume(self):
        self.assertEqual(select_keyframe_indices(0), [])


class TestBuildUnionMask(unittest.TestCase):
    """Unit tests for build_union_mask."""

    def test_union_covers_all_keyframe_phi(self):
        # Two keyframes each have PHI in different pixel locations
        volume = np.zeros((5, 1, 4, 4), dtype=np.uint8)

        call_count = [0]
        def mock_ocr(frame):
            mask = np.zeros((4, 4), dtype=bool)
            if call_count[0] == 0:
                mask[0, 0] = True   # frame 0: PHI top-left
            else:
                mask[3, 3] = True   # other frames: PHI bottom-right
            call_count[0] += 1
            return mask

        indices = [0, 2, 4]
        union_mask, _ = build_union_mask(volume, indices, mock_ocr)
        self.assertTrue(union_mask[0, 0])   # from first keyframe
        self.assertTrue(union_mask[3, 3])   # from later keyframes
        self.assertFalse(union_mask[1, 1])  # clean pixel

    def test_existing_mask_merged_into_union(self):
        volume = np.zeros((5, 1, 4, 4), dtype=np.uint8)
        existing = np.zeros((4, 4), dtype=bool)
        existing[2, 2] = True  # simulates RedactByUSRegionsd output

        def zero_ocr(_frame):
            return np.zeros((4, 4), dtype=bool)  # OCR finds nothing

        union_mask, _ = build_union_mask(
            volume, [0, 2, 4], zero_ocr, existing_mask=existing
        )
        # US region mask pixel must survive even when OCR finds nothing
        self.assertTrue(union_mask[2, 2])

    def test_stats_dict_keys(self):
        volume = np.zeros((10, 1, 4, 4), dtype=np.uint8)
        _, stats = build_union_mask(
            volume, [0, 5, 9], lambda f: np.zeros((4, 4), dtype=bool)
        )
        for key in (
            "keyframes_scanned", "keyframe_indices",
            "per_keyframe_phi_pixels", "union_phi_pixels",
            "total_frames", "pct_frames_scanned",
        ):
            self.assertIn(key, stats)


class TestRedactVolumeSafe(unittest.TestCase):
    """Unit tests for redact_volume_safe."""

    def test_phi_on_one_keyframe_zeroed_on_all_frames(self):
        """Regression: PHI at pixel (0,0) on frame 0 must be zeroed on frame 9."""
        volume = np.ones((10, 1, 4, 4), dtype=np.uint8) * 200

        call_count = [0]
        def mock_ocr(frame):
            mask = np.zeros((4, 4), dtype=bool)
            if call_count[0] == 0:   # first keyframe only
                mask[0, 0] = True
            call_count[0] += 1
            return mask

        _, union_mask, _ = redact_volume_safe(
            volume, mock_ocr, sample_pct=0.10, floor=3, ceiling=25
        )
        # union guarantee: pixel (0,0) zeroed on every frame
        self.assertEqual(int(volume[9, 0, 0, 0]), 0)

    def test_clean_pixels_preserved(self):
        volume = np.ones((10, 1, 4, 4), dtype=np.uint8) * 200

        def zero_ocr(_frame):
            return np.zeros((4, 4), dtype=bool)

        redact_volume_safe(volume, zero_ocr)
        # No PHI found → no pixels zeroed → all values intact
        self.assertTrue((volume == 200).all())

    def test_ceiling_respected(self):
        volume = np.zeros((2000, 1, 4, 4), dtype=np.uint8)
        scanned = []

        def counting_ocr(_frame):
            scanned.append(1)
            return np.zeros((4, 4), dtype=bool)

        redact_volume_safe(volume, counting_ocr, ceiling=25)
        self.assertLessEqual(len(scanned), 25)


if __name__ == '__main__':
    unittest.main()
