"""
Unit tests for RedactByUSRegions / RedactByUSRegionsd transforms.

Tests US fan-geometry mask building, edge cases, and integration
with the downstream RedactPixelPHId transform.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pydicom
import torch
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from monai.data import MetaTensor

from monai_aegis.transforms.us_regions import (
    RedactByUSRegions,
    RedactByUSRegionsd,
    build_us_phi_mask,
    _extract_region_bounds,
)


def _make_us_dataset(modality="US", regions=None):
    """Create a minimal pydicom Dataset with optional US region sequence.

    Args:
        modality: DICOM Modality string.
        regions: List of (x0, y0, x1, y1) tuples for region bounds.

    Returns:
        pydicom.Dataset with populated attributes.
    """
    ds = Dataset()
    ds.Modality = modality

    if regions is not None:
        seq_items = []
        for (x0, y0, x1, y1) in regions:
            item = Dataset()
            item.RegionSpatialFormat = 1
            item.RegionDataType = 1
            item.RegionFlags = 1
            item.RegionLocationMinX0 = x0
            item.RegionLocationMinY0 = y0
            item.RegionLocationMaxX1 = x1
            item.RegionLocationMaxY1 = y1
            seq_items.append(item)
        ds.add_new((0x0018, 0x6011), "SQ", Sequence(seq_items))

    return ds


class TestExtractRegionBounds(unittest.TestCase):
    """Tests for the _extract_region_bounds helper."""

    def test_valid_bounds(self):
        item = Dataset()
        item.RegionLocationMinX0 = 10
        item.RegionLocationMinY0 = 20
        item.RegionLocationMaxX1 = 300
        item.RegionLocationMaxY1 = 400
        self.assertEqual(_extract_region_bounds(item), (10, 20, 300, 400))

    def test_missing_attribute(self):
        item = Dataset()
        item.RegionLocationMinX0 = 10
        # Missing Y0, X1, Y1
        self.assertIsNone(_extract_region_bounds(item))


class TestBuildUSPhiMask(unittest.TestCase):
    """Tests for the build_us_phi_mask function."""

    def test_non_us_modality_returns_none(self):
        ds = _make_us_dataset(modality="CT", regions=[(10, 10, 100, 100)])
        mask = build_us_phi_mask(ds, 200, 200)
        self.assertIsNone(mask)

    def test_us_no_sequence_tag_returns_none(self):
        ds = _make_us_dataset(modality="US", regions=None)
        mask = build_us_phi_mask(ds, 200, 200)
        self.assertIsNone(mask)

    def test_single_region_mask(self):
        """One region => mask is False inside region, True outside."""
        ds = _make_us_dataset(modality="US", regions=[(50, 50, 150, 150)])
        mask = build_us_phi_mask(ds, 200, 200)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (200, 200))
        # Inside region: NOT a PHI zone
        self.assertFalse(mask[75, 100])
        self.assertFalse(mask[50, 50])
        # Outside region: PHI zone
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[199, 199])
        self.assertTrue(mask[10, 10])

    def test_multiple_regions_union(self):
        """Two non-overlapping regions => union non-PHI area."""
        ds = _make_us_dataset(modality="US", regions=[
            (0, 0, 50, 50),
            (100, 100, 200, 200),
        ])
        mask = build_us_phi_mask(ds, 200, 200)

        self.assertIsNotNone(mask)
        # Inside region 1: not PHI
        self.assertFalse(mask[25, 25])
        # Inside region 2: not PHI
        self.assertFalse(mask[150, 150])
        # Between regions: PHI zone
        self.assertTrue(mask[75, 75])

    def test_region_clamped_to_image(self):
        """Regions with coords exceeding image dims are clamped."""
        ds = _make_us_dataset(modality="US", regions=[(-10, -10, 500, 500)])
        mask = build_us_phi_mask(ds, 100, 100)

        self.assertIsNotNone(mask)
        # Entire image should be diagnostic (non-PHI)
        self.assertFalse(mask.any())

    def test_zero_area_region_skipped(self):
        """Region with zero area after clamping is skipped."""
        ds = _make_us_dataset(modality="US", regions=[(50, 50, 50, 50)])
        mask = build_us_phi_mask(ds, 100, 100)
        # Zero-area region should be skipped, returns None (no valid regions)
        self.assertIsNone(mask)


class TestRedactByUSRegions(unittest.TestCase):
    """Tests for the array transform."""

    def test_enabled_us_returns_mask(self):
        config = {"us_regions": {"enabled": True}}
        transform = RedactByUSRegions(config=config)
        ds = _make_us_dataset(modality="US", regions=[(10, 10, 90, 90)])
        mask = transform(ds, 100, 100)
        self.assertIsNotNone(mask)

    def test_disabled_returns_none(self):
        config = {"us_regions": {"enabled": False}}
        transform = RedactByUSRegions(config=config)
        ds = _make_us_dataset(modality="US", regions=[(10, 10, 90, 90)])
        mask = transform(ds, 100, 100)
        self.assertIsNone(mask)

    def test_default_enabled(self):
        """When us_regions section is missing, it defaults to enabled."""
        config = {}
        transform = RedactByUSRegions(config=config)
        ds = _make_us_dataset(modality="US", regions=[(10, 10, 90, 90)])
        mask = transform(ds, 100, 100)
        self.assertIsNotNone(mask)


class TestRedactByUSRegionsd(unittest.TestCase):
    """Tests for the dictionary transform."""

    def setUp(self):
        self.config = {"us_regions": {"enabled": True}}

    def test_writes_mask_to_data_dict(self):
        """Pipeline integration: mask is written under {key}_us_phi_mask."""
        ds = _make_us_dataset(modality="US", regions=[(20, 20, 80, 80)])
        tensor = MetaTensor(
            torch.zeros(1, 100, 100),
            meta={"filename_or_obj": "test.dcm"},
        )
        data = {
            "image": tensor,
            "image_dicom_dataset": ds,
            "image_meta_dict": {"filename_or_obj": "test.dcm"},
        }

        transform = RedactByUSRegionsd(keys=["image"], config=self.config)
        result = transform(data)

        self.assertIn("image_us_phi_mask", result)
        mask = result["image_us_phi_mask"]
        self.assertEqual(mask.shape, (100, 100))
        # Inside diagnostic region
        self.assertFalse(mask[50, 50])
        # Outside diagnostic region
        self.assertTrue(mask[0, 0])

    def test_non_us_no_mask(self):
        """Non-US modality should not produce a mask."""
        ds = _make_us_dataset(modality="MR", regions=[(20, 20, 80, 80)])
        tensor = MetaTensor(
            torch.zeros(1, 100, 100),
            meta={"filename_or_obj": "test.dcm"},
        )
        data = {
            "image": tensor,
            "image_dicom_dataset": ds,
            "image_meta_dict": {"filename_or_obj": "test.dcm"},
        }

        transform = RedactByUSRegionsd(keys=["image"], config=self.config)
        result = transform(data)

        self.assertNotIn("image_us_phi_mask", result)

    def test_series_mode_datasets_list(self):
        """Series mode: reads from {key}_dicom_datasets list."""
        ds = _make_us_dataset(modality="US", regions=[(10, 10, 90, 90)])
        tensor = MetaTensor(
            torch.zeros(1, 5, 100, 100),  # (C, D, H, W)
            meta={"filename_or_obj": "test.dcm"},
        )
        data = {
            "image": tensor,
            "image_dicom_datasets": [ds],
            "image_meta_dict": {"filename_or_obj": "test.dcm"},
        }

        transform = RedactByUSRegionsd(keys=["image"], config=self.config)
        result = transform(data)

        self.assertIn("image_us_phi_mask", result)
        self.assertEqual(result["image_us_phi_mask"].shape, (100, 100))

    def test_no_dataset_passthrough(self):
        """No cached dataset → no mask, passthrough."""
        tensor = MetaTensor(
            torch.zeros(1, 100, 100),
            meta={"filename_or_obj": "test.jpg"},
        )
        data = {
            "image": tensor,
            "image_meta_dict": {"filename_or_obj": "test.jpg"},
        }

        transform = RedactByUSRegionsd(keys=["image"], config=self.config)
        result = transform(data)

        self.assertNotIn("image_us_phi_mask", result)


class TestPixelRedactionWithUSMask(unittest.TestCase):
    """Integration: verify RedactPixelPHId respects the US PHI mask."""

    @patch('monai_aegis.transforms.pixel.easyocr.Reader')
    @patch('monai_aegis.transforms.pixel.detect_text')
    @patch('monai_aegis.transforms.pixel.apply_redaction')
    def test_ocr_receives_masked_pixels(self, mock_apply, mock_detect, mock_reader):
        """When US mask is present, diagnostic pixels should be zeroed before OCR."""
        from monai_aegis.transforms.pixel import RedactPixelPHId

        config = {
            'ocr': {'languages': ['en'], 'confidence_threshold': 0.4},
            'safelist': [],
        }
        scrubber = RedactPixelPHId(keys=['image'], config=config)

        # Create image: all 128 (gray)
        test_data = np.full((1, 100, 100), 128.0, dtype=np.float32)

        # Mask: top 50 rows are diagnostic (False), bottom 50 are PHI zone (True)
        us_mask = np.zeros((100, 100), dtype=bool)
        us_mask[50:, :] = True

        empty_stats = {
            'total_detections': 0, 'low_confidence_count': 0,
            'safelisted_count': 0, 'redacted_count': 0,
        }
        mock_detect.return_value = ([], empty_stats)
        mock_apply.return_value = np.full((100, 100), 128.0, dtype=np.float32)

        data = {
            'image': test_data,
            'image_meta_dict': {'filename_or_obj': 'test.dcm'},
            'image_us_phi_mask': us_mask,
        }
        result = scrubber(data)

        # Verify detect_text was called
        mock_detect.assert_called_once()

        # The pixels passed to detect_text should have the diagnostic
        # region (top 50 rows) zeroed out
        call_args = mock_detect.call_args
        ocr_input = call_args[0][0]  # first positional arg: pixel_array
        # Top 50 rows (diagnostic) should be 0
        self.assertTrue(np.all(ocr_input[:50, :] == 0),
                        "Diagnostic pixels should be zeroed before OCR")

        # Stats should include us_region_mask_applied flag
        self.assertTrue(result['image_redaction_stats'].get('us_region_mask_applied'))

    @patch('monai_aegis.transforms.pixel.easyocr.Reader')
    @patch('monai_aegis.transforms.pixel.detect_text')
    @patch('monai_aegis.transforms.pixel.apply_redaction')
    def test_diagnostic_pixels_restored_after_ocr(self, mock_apply, mock_detect, mock_reader):
        """After OCR, diagnostic region pixels must be restored from original."""
        from monai_aegis.transforms.pixel import RedactPixelPHId

        config = {
            'ocr': {'languages': ['en'], 'confidence_threshold': 0.4},
            'safelist': [],
        }
        scrubber = RedactPixelPHId(keys=['image'], config=config)

        # Original: all 200
        test_data = np.full((1, 100, 100), 200.0, dtype=np.float32)

        # Mask: left half diagnostic, right half PHI zone
        us_mask = np.zeros((100, 100), dtype=bool)
        us_mask[:, 50:] = True

        empty_stats = {
            'total_detections': 0, 'low_confidence_count': 0,
            'safelisted_count': 0, 'redacted_count': 0,
        }
        mock_detect.return_value = ([], empty_stats)
        # Simulate redaction returning zeros where PHI was found
        mock_apply.return_value = np.zeros((100, 100), dtype=np.float32)

        data = {
            'image': test_data,
            'image_meta_dict': {'filename_or_obj': 'test.dcm'},
            'image_us_phi_mask': us_mask,
        }
        result = scrubber(data)

        output = result['image']
        if hasattr(output, 'numpy'):
            output = output.numpy()

        # Diagnostic region (left half) should be restored to 200
        self.assertTrue(
            np.all(output[0, :, :50] == 200.0),
            "Diagnostic pixels should be restored after OCR"
        )


class TestSeriesModeUnion(unittest.TestCase):
    """Tests for RedactByUSRegionsd series-mode union across all datasets."""

    def _make_transform(self):
        config = {'us_regions': {'enabled': True}}
        return RedactByUSRegionsd(keys=['image'], config=config)

    def _make_data(self, datasets_list, h=64, w=64):
        tensor = np.zeros((1, len(datasets_list), h, w), dtype=np.float32)
        return {
            'image': tensor,
            'image_dicom_datasets': datasets_list,
        }

    def test_series_mode_unions_masks_across_all_datasets(self):
        """Mask must cover regions from all datasets, not just datasets[0]."""
        # Three datasets with non-overlapping diagnostic regions
        ds0 = _make_us_dataset(regions=[(0, 0, 10, 10)])    # top-left
        ds1 = _make_us_dataset(regions=[(20, 20, 30, 30)])  # middle
        ds2 = _make_us_dataset(regions=[(50, 50, 60, 60)])  # bottom-right

        result = self._make_transform()(self._make_data([ds0, ds1, ds2]))
        mask = result['image_us_phi_mask']

        # Each dataset's diagnostic region must be False (not-PHI) in the union
        self.assertFalse(mask[5, 5],   "ds0 region should be non-PHI")
        self.assertFalse(mask[25, 25], "ds1 region should be non-PHI")
        self.assertFalse(mask[55, 55], "ds2 region should be non-PHI")
        # Pixel outside all regions is PHI
        self.assertTrue(mask[63, 63])

    def test_series_mode_first_dataset_missing_tag_still_gets_mask(self):
        """If datasets[0] has no US-regions tag, later datasets must not be skipped."""
        ds_no_tag = _make_us_dataset(regions=None)   # valid US but no tag
        ds_with_tag = _make_us_dataset(regions=[(10, 10, 50, 50)])

        result = self._make_transform()(self._make_data([ds_no_tag, ds_with_tag]))
        self.assertIn('image_us_phi_mask', result,
                      "Mask should be written even when first dataset has no tag")
        mask = result['image_us_phi_mask']
        self.assertFalse(mask[30, 30], "Region from ds1 should be non-PHI")

    def test_single_image_mode_unchanged(self):
        """Single-dataset path must still write the correct mask."""
        ds = _make_us_dataset(regions=[(5, 5, 30, 30)])
        tensor = np.zeros((1, 64, 64), dtype=np.float32)
        data = {'image': tensor, 'image_dicom_dataset': ds}

        result = self._make_transform()(data)
        self.assertIn('image_us_phi_mask', result)
        mask = result['image_us_phi_mask']
        self.assertFalse(mask[10, 10], "Diagnostic pixel should be non-PHI")
        self.assertTrue(mask[63, 63],  "Outside pixel should be PHI")


if __name__ == '__main__':
    unittest.main()
