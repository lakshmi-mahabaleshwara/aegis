"""Unit tests for ``monai_aegis.transforms.metadata.pixel_injection`` in isolation.

``inject_pixel_data`` is exercised directly on minimal datasets so every
responsibility is pinned on its own: contiguity, bit depth / sign, Rows /
Columns / NumberOfFrames / SamplesPerPixel / PlanarConfiguration,
PhotometricInterpretation, the exact PixelData bytes, windowing, transfer
syntax, and the unsupported-shape error. Shape families are covered end to
end through the transform by ``test_metadata_characterization.py``.
"""
import unittest

import numpy as np
import pydicom
import pydicom.uid
from pydicom.dataset import Dataset, FileMetaDataset

from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms.metadata import constants
from monai_aegis.transforms.metadata.pixel_injection import inject_pixel_data


def _ds(with_file_meta=True, number_of_frames=None, planar=None):
    ds = Dataset()
    if with_file_meta:
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.JPEGBaseline8Bit
    if number_of_frames is not None:
        ds.NumberOfFrames = number_of_frames
    if planar is not None:
        ds.PlanarConfiguration = planar
    ds.WindowCenter = "999"
    ds.WindowWidth = "999"
    return ds


def _pixel_tags(ds):
    keys = ("BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "SamplesPerPixel",
            "Rows", "Columns", "PhotometricInterpretation", "PlanarConfiguration", "NumberOfFrames")
    return {k: (ds[k].value if k in ds else None) for k in keys}


class TestContiguity(unittest.TestCase):

    def test_fortran_order_input_is_written_row_major(self):
        c_order = np.arange(24, dtype=np.uint16).reshape(4, 6)
        f_order = np.asfortranarray(c_order)
        self.assertFalse(f_order.flags.c_contiguous)
        ds_c, ds_f = _ds(), _ds()
        inject_pixel_data(ds_c, c_order)
        inject_pixel_data(ds_f, f_order)
        self.assertEqual(ds_f.PixelData, ds_c.PixelData)
        self.assertEqual(ds_f.PixelData, c_order.tobytes())
        np.testing.assert_array_equal(ds_f.pixel_array, c_order)

    def test_transposed_and_strided_views_are_written_as_seen(self):
        base = np.arange(48, dtype=np.uint8).reshape(6, 8)
        view = base.T[::2, ::3]          # non-contiguous in both axes
        ds = _ds()
        inject_pixel_data(ds, view)
        self.assertEqual((ds.Rows, ds.Columns), view.shape)
        self.assertEqual(ds.PixelData, np.ascontiguousarray(view).tobytes())
        np.testing.assert_array_equal(ds.pixel_array, view)

    def test_input_array_is_not_modified(self):
        arr = np.arange(96, dtype=np.uint8).reshape(4, 6, 4)
        before = arr.copy()
        inject_pixel_data(_ds(), arr)
        np.testing.assert_array_equal(arr, before)


class TestBitDepth(unittest.TestCase):

    def test_dtype_table(self):
        cases = {
            np.uint8: (constants.BIT_DEPTH_8, constants.PIXEL_REPRESENTATION_UNSIGNED),
            np.uint16: (constants.BIT_DEPTH_16, constants.PIXEL_REPRESENTATION_UNSIGNED),
            np.int16: (constants.BIT_DEPTH_16, constants.PIXEL_REPRESENTATION_SIGNED),
            # quirk: any non-uint8 dtype is labelled 16-bit, even when the
            # sample size differs (int8 / 32-bit). The dict transform always
            # casts to the source dataset's dtype, so these are only reachable
            # through the array API.
            np.int8: (constants.BIT_DEPTH_16, constants.PIXEL_REPRESENTATION_SIGNED),
            np.int32: (constants.BIT_DEPTH_16, constants.PIXEL_REPRESENTATION_SIGNED),
            np.uint32: (constants.BIT_DEPTH_16, constants.PIXEL_REPRESENTATION_UNSIGNED),
            np.float32: (constants.BIT_DEPTH_16, constants.PIXEL_REPRESENTATION_UNSIGNED),
        }
        for dtype, ((alloc, stored, high), repr_) in cases.items():
            ds = _ds()
            inject_pixel_data(ds, np.zeros((2, 2), dtype=dtype))
            self.assertEqual((ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation),
                             (alloc, stored, high, repr_), dtype)

    def test_bytes_follow_the_array_dtype_not_the_label(self):
        ds = _ds()
        inject_pixel_data(ds, np.zeros((2, 3), dtype=np.int16))
        self.assertEqual(len(ds.PixelData), 2 * 3 * 2)
        ds = _ds()
        inject_pixel_data(ds, np.zeros((2, 3), dtype=np.uint8))
        self.assertEqual(len(ds.PixelData), 2 * 3)


class TestDimensions(unittest.TestCase):

    def test_2d(self):
        ds = _ds(number_of_frames="7", planar=1)
        arr = np.arange(24, dtype=np.uint16).reshape(4, 6)
        inject_pixel_data(ds, arr)
        self.assertEqual(_pixel_tags(ds), {
            "BitsAllocated": 16, "BitsStored": 16, "HighBit": 15, "PixelRepresentation": 0,
            "SamplesPerPixel": 1, "Rows": 4, "Columns": 6,
            "PhotometricInterpretation": "MONOCHROME2",
            "PlanarConfiguration": 1,      # untouched for monochrome (stale value survives)
            "NumberOfFrames": None,        # removed
        })
        self.assertEqual(ds.PixelData, arr.tobytes())

    def test_3d_grayscale_frames(self):
        ds = _ds(number_of_frames="7")
        arr = np.arange(120, dtype=np.uint16).reshape(5, 4, 6)
        inject_pixel_data(ds, arr)
        tags = _pixel_tags(ds)
        self.assertEqual((tags["NumberOfFrames"], tags["SamplesPerPixel"], tags["Rows"], tags["Columns"]),
                         (5, 1, 4, 6))
        self.assertEqual(ds["NumberOfFrames"].VR, "IS")
        self.assertEqual(tags["PhotometricInterpretation"], "MONOCHROME2")
        self.assertIsNone(tags["PlanarConfiguration"])
        self.assertEqual(ds.PixelData, arr.tobytes())
        self.assertEqual(ds.pixel_array.shape, (5, 4, 6))

    def test_3d_rgb(self):
        ds = _ds(number_of_frames="7", planar=1)
        arr = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        inject_pixel_data(ds, arr)
        self.assertEqual(_pixel_tags(ds), {
            "BitsAllocated": 8, "BitsStored": 8, "HighBit": 7, "PixelRepresentation": 0,
            "SamplesPerPixel": 3, "Rows": 4, "Columns": 6,
            "PhotometricInterpretation": "RGB",
            "PlanarConfiguration": 0,      # source value 1 overwritten: data is interleaved
            "NumberOfFrames": None,
        })
        self.assertEqual(ds.PixelData, arr.tobytes())

    def test_3d_rgba_drops_alpha_from_bytes(self):
        ds = _ds()
        arr = np.arange(96, dtype=np.uint8).reshape(4, 6, 4)
        inject_pixel_data(ds, arr)
        self.assertEqual(ds.SamplesPerPixel, 3)
        self.assertEqual(ds.PixelData, np.ascontiguousarray(arr[..., :3]).tobytes())
        np.testing.assert_array_equal(ds.pixel_array, arr[..., :3])

    def test_3d_last_axis_3_or_4_is_colour_even_for_16bit(self):
        for ch in (3, 4):
            ds = _ds()
            inject_pixel_data(ds, np.zeros((4, 6, ch), dtype=np.uint16))
            self.assertEqual((ds.SamplesPerPixel, ds.BitsAllocated, ds.PhotometricInterpretation),
                             (3, 16, "RGB"))
            self.assertEqual(len(ds.PixelData), 4 * 6 * 3 * 2)

    def test_3d_ambiguous_small_width_is_frames_not_colour(self):
        # (3, H, W) with W not in {3, 4} → 3 frames; (H, W, 3) → colour.
        ds = _ds()
        inject_pixel_data(ds, np.zeros((3, 4, 6), dtype=np.uint8))
        self.assertEqual((ds.NumberOfFrames, ds.SamplesPerPixel), (3, 1))
        ds = _ds()
        inject_pixel_data(ds, np.zeros((4, 6, 3), dtype=np.uint8))
        self.assertEqual(("NumberOfFrames" in ds, ds.SamplesPerPixel), (False, 3))

    def test_4d_rgb_and_rgba(self):
        for ch in (3, 4):
            ds = _ds(number_of_frames="1", planar=1)
            arr = (np.arange(5 * 4 * 6 * ch) % 251).astype(np.uint8).reshape(5, 4, 6, ch)
            inject_pixel_data(ds, arr)
            tags = _pixel_tags(ds)
            self.assertEqual((tags["NumberOfFrames"], tags["SamplesPerPixel"], tags["Rows"],
                              tags["Columns"], tags["PlanarConfiguration"], tags["PhotometricInterpretation"]),
                             (5, 3, 4, 6, 0, "RGB"), ch)
            self.assertEqual(ds.PixelData, np.ascontiguousarray(arr[..., :3]).tobytes())
            self.assertEqual(ds.pixel_array.shape, (5, 4, 6, 3))

    def test_missing_number_of_frames_on_single_frame_is_fine(self):
        ds = _ds(number_of_frames=None)
        inject_pixel_data(ds, np.zeros((2, 2), dtype=np.uint8))
        self.assertNotIn("NumberOfFrames", ds)


class TestUnsupportedShapes(unittest.TestCase):

    def test_rejected_shapes(self):
        for shape in ((6,), (5, 4, 6, 1), (5, 4, 6, 2), (5, 4, 6, 5), (2, 5, 4, 6, 3)):
            ds = _ds()
            with self.assertRaises(MetadataScrubError, msg=shape) as ctx:
                inject_pixel_data(ds, np.zeros(shape, dtype=np.uint8))
            self.assertEqual(ctx.exception.transform, constants.ARRAY_TRANSFORM_NAME)
            self.assertEqual(ctx.exception.uri, "")
            self.assertIn(f"Unsupported pixel array shape for DICOM injection: {shape}", str(ctx.exception))
            self.assertNotIn("PixelData", ds)

    def test_side_effects_before_the_shape_check_are_applied(self):
        # quirk: transfer syntax and bit depth are written before the shape is
        # validated. Harmless in the pipeline (the caller works on a deep copy
        # that is discarded on error) but observable on the dataset passed in.
        ds = _ds()
        with self.assertRaises(MetadataScrubError):
            inject_pixel_data(ds, np.zeros((6,), dtype=np.int16))
        self.assertEqual(ds.file_meta.TransferSyntaxUID, constants.INJECTED_TRANSFER_SYNTAX)
        self.assertEqual((ds.BitsAllocated, ds.PixelRepresentation), (16, 1))
        self.assertEqual((ds.WindowCenter, ds.WindowWidth), ("999", "999"))


class TestPhotometricPixelDataWindowing(unittest.TestCase):

    def test_transfer_syntax_forced_only_when_file_meta_exists(self):
        ds = _ds()
        inject_pixel_data(ds, np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(ds.file_meta.TransferSyntaxUID, pydicom.uid.ExplicitVRLittleEndian)
        ds = _ds(with_file_meta=False)
        inject_pixel_data(ds, np.zeros((2, 2), dtype=np.uint8))
        self.assertFalse(hasattr(ds, "file_meta"))
        self.assertEqual(len(ds.PixelData), 4)

    def test_windowing_values_and_string_format(self):
        ds = _ds()
        inject_pixel_data(ds, np.array([[-12, 0], [5, 11]], dtype=np.int16))
        self.assertEqual((ds.WindowCenter, ds.WindowWidth), ("-0.5", "23.0"))
        self.assertEqual(ds["WindowCenter"].VR, "DS")
        ds = _ds()
        inject_pixel_data(ds, np.full((2, 2), 7, dtype=np.uint8))
        self.assertEqual((ds.WindowCenter, ds.WindowWidth), ("7.0", "0.0"))
        ds = _ds()
        inject_pixel_data(ds, np.array([[0, 4095]], dtype=np.uint16))
        self.assertEqual((ds.WindowCenter, ds.WindowWidth), ("2047.5", "4095.0"))

    def test_windowing_ignores_a_dropped_alpha_channel(self):
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[..., 3] = 255                 # opaque alpha only
        ds = _ds()
        inject_pixel_data(ds, arr)
        self.assertEqual((ds.WindowCenter, ds.WindowWidth), ("0.0", "0.0"))

    def test_windowing_for_rgb_uses_all_retained_channels(self):
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[0, 0, 2] = 200                # single blue sample
        ds = _ds()
        inject_pixel_data(ds, arr)
        self.assertEqual((ds.WindowCenter, ds.WindowWidth), ("100.0", "200.0"))

    def test_photometric_follows_samples_per_pixel(self):
        ds = _ds()
        inject_pixel_data(ds, np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(ds.PhotometricInterpretation, constants.PHOTOMETRIC_MONOCHROME2)
        ds = _ds()
        inject_pixel_data(ds, np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(ds.PhotometricInterpretation, constants.PHOTOMETRIC_RGB)

    def test_injection_is_idempotent(self):
        arr = np.arange(24, dtype=np.uint16).reshape(4, 6)
        ds = _ds(number_of_frames="7")
        inject_pixel_data(ds, arr)
        once = (_pixel_tags(ds), ds.PixelData, ds.WindowCenter, ds.WindowWidth)
        inject_pixel_data(ds, arr)
        self.assertEqual((_pixel_tags(ds), ds.PixelData, ds.WindowCenter, ds.WindowWidth), once)

    def test_round_trips_through_pydicom_decoder_for_every_layout(self):
        layouts = {
            (4, 6): np.uint16, (4, 6, 3): np.uint8, (4, 6, 4): np.uint8,
            (5, 4, 6): np.int16, (5, 4, 6, 3): np.uint8, (5, 4, 6, 4): np.uint8,
        }
        for shape, dtype in layouts.items():
            arr = (np.arange(int(np.prod(shape))) % 127).astype(dtype).reshape(shape)
            ds = _ds()
            inject_pixel_data(ds, arr)
            expected = arr[..., :3] if shape[-1] in constants.COLOR_CHANNEL_COUNTS and len(shape) > 2 else arr
            np.testing.assert_array_equal(ds.pixel_array, expected, err_msg=str(shape))


if __name__ == "__main__":
    unittest.main()
