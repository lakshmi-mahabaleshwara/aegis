"""Characterization tests for ``monai_aegis.transforms.metadata``.

These tests pin the *currently observed* behavior of ``ScrubDicomMetadata``
and ``ScrubDicomMetadatad`` in the areas the existing unit tests do not
reach (pixel-injection shape/dtype branches, series mode, multi-valued UID
elements, audit-row ordering, error wrapping). They are the regression gate
for the planned split of ``metadata.py`` into smaller modules: every test
here must pass unchanged before *and* after the refactor.

Scope note: a characterization test documents what the code does today. A
few behaviors pinned here are quirks rather than design goals (called out
inline with ``# quirk:``). Changing them is a deliberate behavior change
that belongs in its own commit with its own test update — never in the
refactor.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pydicom
import pydicom.uid
import torch
from monai.data import MetaTensor
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms.metadata import ScrubDicomMetadata, ScrubDicomMetadatad

_STD_ROOT = "1.2.840.10008"
_PYDICOM_ROOT = "1.2.826.0.1.3680043.8.498."
_FOREIGN = "1.2.840.113619.2.55."

_ROW_KEYS = {"tag", "keyword", "action", "redacted"}


def _cfg(mapping=None, salt="char-salt"):
    return {"pii_mapping": dict(mapping or {}), "tokenization": {"salt": salt}}


def _ds_with_pixels(i=0, rows=4, cols=6, value=1000, largest=4000):
    """Minimal uint16 MONOCHROME2 slice with geometry + UIDs."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = _STD_ROOT + ".5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = _FOREIGN + f"sop.{i}"
    ds.SOPClassUID = _STD_ROOT + ".5.1.4.1.1.7"
    ds.SOPInstanceUID = _FOREIGN + f"sop.{i}"
    ds.SeriesInstanceUID = _FOREIGN + "series"
    ds.StudyInstanceUID = _FOREIGN + "study"
    ds.PatientName = "Doe^Jane"
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.full((rows, cols), value + i, dtype=np.uint16).tobytes()
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 2.0
    ds.SpacingBetweenSlices = 2.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(i)]
    if largest is not None:
        ds.LargestImagePixelValue = largest
    return ds


# ---------------------------------------------------------------------------
# 1. Recursive tag scrubbing
# ---------------------------------------------------------------------------

class TestRecursiveScrubCharacterization(unittest.TestCase):

    def _nested_ds(self):
        ds = Dataset()
        ds.SOPInstanceUID = _FOREIGN + "1"
        ds.PatientName = "Top^Name"
        ds.PatientID = "TOP"
        ds.StudyID = "S"
        inner = Dataset()
        inner.PatientID = "INNER"
        ib = inner.private_block(0x0011, "INNER PRIV", 0x10)
        ib.add_new(0x01, "LO", "inner secret")
        mid = Dataset()
        mid.PatientName = "Mid^Name"
        mid.add_new((0x0008, 0x1140), "SQ", Sequence([inner]))  # ReferencedImageSequence
        ds.add_new((0x0040, 0x0275), "SQ", Sequence([mid]))     # RequestAttributesSequence
        pb = ds.private_block(0x0009, "TOP PRIV", 0x10)
        pb.add_new(0x01, "LO", "top secret")
        return ds

    def test_actions_apply_at_every_nesting_depth(self):
        t = ScrubDicomMetadata(_cfg({"(0010,0010)": "REMOVE", "(0010,0020)": "DUMMY"}))
        out, _ = t.scrub("f.dcm", dataset=self._nested_ds())
        mid = out[(0x0040, 0x0275)].value[0]
        inner = mid[(0x0008, 0x1140)].value[0]
        self.assertNotIn("PatientName", out)
        self.assertNotIn("PatientName", mid)
        self.assertTrue(str(out.PatientID).startswith("TOKEN_"))
        self.assertTrue(str(inner.PatientID).startswith("TOKEN_"))

    def test_private_tags_removed_at_every_depth_without_audit_rows(self):
        t = ScrubDicomMetadata(_cfg())
        out, rows = t.scrub("f.dcm", dataset=self._nested_ds())
        self.assertFalse(any(e.tag.is_private for e in out.iterall()))
        # quirk: private-tag removal is not recorded in the audit trail.
        self.assertFalse(any(r["tag"].startswith("(0009") or r["tag"].startswith("(0011") for r in rows))

    def test_audit_rows_are_depth_first_at_the_sequence_tag_position(self):
        # Top-level elements are visited in tag order; when an SQ element is
        # reached its items are scrubbed (recursively) *before* the next
        # top-level sibling, so nested rows are interleaved at that position.
        t = ScrubDicomMetadata(_cfg({"(0010,0010)": "REMOVE", "(0010,0020)": "DUMMY", "(0020,0010)": "DUMMY"}))
        _, rows = t.scrub("f.dcm", dataset=self._nested_ds())
        scrub_rows = [r for r in rows if r["action"] in ("REMOVE", "DUMMY")]
        self.assertEqual(
            [r["keyword"] for r in scrub_rows],
            ["PatientName", "PatientID", "StudyID",   # top level (0010,0010) (0010,0020) (0020,0010)
             "PatientID",                              # depth 2 (inside (0008,1140) inside (0040,0275))
             "PatientName"],                           # depth 1 (mid item), visited after its own SQ child
        )

    def test_nested_rows_have_same_shape_as_top_level_rows(self):
        # quirk: no depth / path information — a nested PatientID row is
        # indistinguishable from a top-level one.
        t = ScrubDicomMetadata(_cfg({"(0010,0020)": "DUMMY"}))
        _, rows = t.scrub("f.dcm", dataset=self._nested_ds())
        pid_rows = [r for r in rows if r["keyword"] == "PatientID"]
        self.assertEqual(len(pid_rows), 2)
        self.assertEqual(pid_rows[0], pid_rows[1])
        self.assertEqual(set(pid_rows[0]), _ROW_KEYS)

    def test_action_on_sequence_tag_itself_scrubs_children_then_removes_sequence(self):
        t = ScrubDicomMetadata(_cfg({"(0040,0275)": "REMOVE", "(0010,0020)": "DUMMY"}))
        out, rows = t.scrub("f.dcm", dataset=self._nested_ds())
        self.assertNotIn((0x0040, 0x0275), out)
        # Children were scrubbed (and audited) before the parent was dropped.
        self.assertEqual([r["keyword"] for r in rows if r["action"] == "DUMMY"], ["PatientID", "PatientID"])
        self.assertIn("RequestAttributesSequence", [r["keyword"] for r in rows if r["action"] == "REMOVE"])

    def test_configured_but_absent_tags_produce_no_rows(self):
        t = ScrubDicomMetadata(_cfg({"(0010,0030)": "REMOVE"}))  # PatientBirthDate, absent
        ds = Dataset()
        ds.SOPInstanceUID = _FOREIGN + "1"
        _, rows = t.scrub("f.dcm", dataset=ds)
        self.assertEqual([r["action"] for r in rows], ["REMAP", "ATTEST", "ATTEST", "ATTEST"])

    def test_zero_on_binary_vr_writes_empty_bytes_and_on_text_vr_empty_str(self):
        t = ScrubDicomMetadata(_cfg({"(0042,0011)": "ZERO", "(0008,0020)": "ZERO"}))
        ds = Dataset()
        ds.add_new((0x0042, 0x0011), "OB", b"\x01\x02")  # EncapsulatedDocument
        ds.StudyDate = "20240101"
        out, _ = t.scrub("f.dcm", dataset=ds)
        self.assertEqual(out[(0x0042, 0x0011)].value, b"")
        self.assertEqual(out.StudyDate, "")

    def test_explicit_action_on_private_tag_is_audited_then_tag_removed_anyway(self):
        # Private removal runs after the action pass, so a private tag named
        # in pii_mapping is audited under its action and still ends up gone.
        t = ScrubDicomMetadata(_cfg({"(0029,1010)": "ZERO"}))
        ds = Dataset()
        ds.add_new((0x0029, 0x1010), "OB", b"\x01\x02")
        out, rows = t.scrub("f.dcm", dataset=ds)
        self.assertNotIn((0x0029, 0x1010), out)
        self.assertIn("(0029,1010)", [r["tag"] for r in rows if r["action"] == "ZERO"])

    def test_dummy_on_empty_value_stays_empty(self):
        t = ScrubDicomMetadata(_cfg({"(0010,0020)": "DUMMY"}))
        ds = Dataset()
        ds.PatientID = ""
        out, rows = t.scrub("f.dcm", dataset=ds)
        self.assertEqual(out.PatientID, "")
        self.assertTrue(any(r["keyword"] == "PatientID" and r["redacted"] for r in rows))

    def test_input_dataset_is_never_mutated(self):
        t = ScrubDicomMetadata(_cfg({"(0010,0010)": "REMOVE"}))
        src = self._nested_ds()
        t.scrub("f.dcm", dataset=src)
        self.assertIn("PatientName", src)
        self.assertEqual(str(src.SOPInstanceUID), _FOREIGN + "1")
        self.assertTrue(any(e.tag.is_private for e in src.iterall()))

    def test_tag_key_accepts_0x_prefixed_hex(self):
        t = ScrubDicomMetadata(_cfg({"(0x0010,0x0010)": "REMOVE"}))
        ds = Dataset()
        ds.PatientName = "X"
        out, _ = t.scrub("f.dcm", dataset=ds)
        self.assertNotIn("PatientName", out)

    def test_dcmread_is_used_when_no_dataset_is_supplied(self):
        t = ScrubDicomMetadata(_cfg())
        ds = Dataset()
        ds.SOPInstanceUID = _FOREIGN + "9"
        # Patch target used by the existing dict-transform tests — must keep working.
        with patch("monai_aegis.transforms.metadata.pydicom.dcmread", return_value=ds) as m:
            out = t(uri="/some/file.dcm")
        m.assert_called_once_with("/some/file.dcm")
        self.assertEqual(out.PatientIdentityRemoved, "YES")


# ---------------------------------------------------------------------------
# 2. UID graph remapping
# ---------------------------------------------------------------------------

class TestUidRemapCharacterization(unittest.TestCase):

    def test_multi_valued_ui_element_is_remapped_per_value(self):
        t = ScrubDicomMetadata(_cfg())
        ds = Dataset()
        ds.StudyInstanceUID = _FOREIGN + "study"
        ds.add_new((0x0008, 0x1155), "UI", [_FOREIGN + "study", _STD_ROOT + ".1.2"])
        out, rows = t.scrub("f.dcm", dataset=ds)
        vals = [str(v) for v in out[(0x0008, 0x1155)].value]
        self.assertEqual(vals[0], str(out.StudyInstanceUID))   # same source -> same replacement
        self.assertEqual(vals[1], _STD_ROOT + ".1.2")          # standard root kept in place
        self.assertEqual(sum(1 for r in rows if r["keyword"] == "ReferencedSOPInstanceUID"), 1)

    def test_standard_root_uid_under_non_class_keyword_is_preserved_without_row(self):
        t = ScrubDicomMetadata(_cfg())
        ds = Dataset()
        ds.add_new((0x0008, 0x010C), "UI", _STD_ROOT + ".2.16.4")  # CodingSchemeUID
        out, rows = t.scrub("f.dcm", dataset=ds)
        self.assertEqual(str(out[(0x0008, 0x010C)].value), _STD_ROOT + ".2.16.4")
        self.assertFalse(any(r["action"] == "REMAP" for r in rows))

    def test_file_meta_is_not_walked_only_media_storage_instance_uid_synced(self):
        t = ScrubDicomMetadata(_cfg())
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.ImplementationClassUID = _FOREIGN + "impl"
        ds.file_meta.MediaStorageSOPInstanceUID = _FOREIGN + "sop"
        ds.SOPInstanceUID = _FOREIGN + "sop"
        out, _ = t.scrub("f.dcm", dataset=ds)
        self.assertEqual(str(out.file_meta.ImplementationClassUID), _FOREIGN + "impl")
        self.assertEqual(str(out.file_meta.MediaStorageSOPInstanceUID), str(out.SOPInstanceUID))
        self.assertTrue(str(out.SOPInstanceUID).startswith(_PYDICOM_ROOT))

    def test_remap_uid_fallback_entropy_is_deterministic_and_random_without_any_source(self):
        t = ScrubDicomMetadata(_cfg())
        a = t.remap_uid("", fallback_entropy="/path/x.dcm")
        b = t.remap_uid("", fallback_entropy="/path/x.dcm")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith(_PYDICOM_ROOT))
        r1, r2 = t.remap_uid(""), t.remap_uid("")
        self.assertNotEqual(r1, r2)

    def test_remap_cache_is_shared_across_calls_on_one_instance(self):
        t = ScrubDicomMetadata(_cfg())
        ds1 = Dataset(); ds1.SeriesInstanceUID = _FOREIGN + "s"
        ds2 = Dataset(); ds2.SeriesInstanceUID = _FOREIGN + "s"
        o1, _ = t.scrub("a.dcm", dataset=ds1)
        o2, _ = t.scrub("b.dcm", dataset=ds2)
        self.assertEqual(str(o1.SeriesInstanceUID), str(o2.SeriesInstanceUID))

    def test_sop_uid_minted_from_uri_when_source_has_none(self):
        t = ScrubDicomMetadata(_cfg())
        o1, _ = t.scrub("/in/a.dcm", dataset=Dataset())
        o2, _ = t.scrub("/in/a.dcm", dataset=Dataset())
        o3, _ = t.scrub("/in/b.dcm", dataset=Dataset())
        self.assertEqual(str(o1.SOPInstanceUID), str(o2.SOPInstanceUID))
        self.assertNotEqual(str(o1.SOPInstanceUID), str(o3.SOPInstanceUID))

    def test_remap_rows_follow_scrub_rows_and_precede_attest_rows(self):
        t = ScrubDicomMetadata(_cfg({"(0010,0010)": "REMOVE"}))
        ds = Dataset()
        ds.PatientName = "X"
        ds.SOPInstanceUID = _FOREIGN + "1"
        _, rows = t.scrub("f.dcm", dataset=ds)
        self.assertEqual([r["action"] for r in rows], ["REMOVE", "REMAP", "ATTEST", "ATTEST", "ATTEST"])
        for r in rows:
            self.assertEqual(set(r), _ROW_KEYS)
            self.assertRegex(r["tag"], r"^\([0-9A-F]{4},[0-9A-F]{4}\)$")


# ---------------------------------------------------------------------------
# 3. Pixel data injection
# ---------------------------------------------------------------------------

class TestPixelInjectionCharacterization(unittest.TestCase):

    def setUp(self):
        self.t = ScrubDicomMetadata(_cfg())

    def _base(self):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds.SOPInstanceUID = _FOREIGN + "1"
        ds.NumberOfFrames = "7"
        ds.WindowCenter = "999"
        ds.WindowWidth = "999"
        ds.BurnedInAnnotation = "YES"
        return ds

    def _inject(self, arr):
        out, rows = self.t.scrub("f.dcm", pixel_data=arr, dataset=self._base())
        return out, rows

    def test_2d_uint16(self):
        out, _ = self._inject(np.arange(24, dtype=np.uint16).reshape(4, 6) * 100)
        self.assertEqual((out.BitsAllocated, out.BitsStored, out.HighBit, out.PixelRepresentation), (16, 16, 15, 0))
        self.assertEqual((out.Rows, out.Columns, out.SamplesPerPixel), (4, 6, 1))
        self.assertEqual(out.PhotometricInterpretation, "MONOCHROME2")
        self.assertNotIn("NumberOfFrames", out)
        self.assertNotIn("PlanarConfiguration", out)
        self.assertEqual(out.file_meta.TransferSyntaxUID, pydicom.uid.ExplicitVRLittleEndian)
        self.assertEqual((str(out.WindowCenter), str(out.WindowWidth)), ("1150.0", "2300.0"))
        np.testing.assert_array_equal(out.pixel_array, np.arange(24, dtype=np.uint16).reshape(4, 6) * 100)

    def test_2d_int16_sets_signed_pixel_representation(self):
        out, _ = self._inject(np.arange(24, dtype=np.int16).reshape(4, 6) - 12)
        self.assertEqual(out.PixelRepresentation, 1)
        self.assertEqual(out.BitsAllocated, 16)
        self.assertEqual(str(out.WindowCenter), "-0.5")

    def test_2d_uint8(self):
        out, _ = self._inject(np.zeros((4, 6), dtype=np.uint8))
        self.assertEqual((out.BitsAllocated, out.BitsStored, out.HighBit, out.PixelRepresentation), (8, 8, 7, 0))
        self.assertEqual((str(out.WindowCenter), str(out.WindowWidth)), ("0.0", "0.0"))

    def test_rgb_hw3(self):
        out, _ = self._inject(np.zeros((4, 6, 3), dtype=np.uint8))
        self.assertEqual((out.SamplesPerPixel, out.PhotometricInterpretation, out.PlanarConfiguration), (3, "RGB", 0))
        self.assertEqual((out.Rows, out.Columns), (4, 6))
        self.assertNotIn("NumberOfFrames", out)
        self.assertEqual(len(out.PixelData), 4 * 6 * 3)

    def test_rgba_hw4_drops_alpha(self):
        out, _ = self._inject(np.zeros((4, 6, 4), dtype=np.uint8))
        self.assertEqual(out.SamplesPerPixel, 3)
        self.assertEqual(len(out.PixelData), 4 * 6 * 3)
        self.assertEqual(out.pixel_array.shape, (4, 6, 3))

    def test_frames_dhw(self):
        out, _ = self._inject(np.zeros((5, 4, 6), dtype=np.uint16))
        self.assertEqual(out.NumberOfFrames, 5)
        self.assertIsInstance(out["NumberOfFrames"].value, int)  # IS VR coerces the str
        self.assertEqual((out.Rows, out.Columns, out.SamplesPerPixel), (4, 6, 1))
        self.assertEqual(out.PhotometricInterpretation, "MONOCHROME2")
        self.assertEqual(out.pixel_array.shape, (5, 4, 6))

    def test_3d_with_last_dim_3_is_always_rgb_not_frames(self):
        # quirk: (3, H, W) with W not in (3,4) is 3 frames; (H, W, 3) is RGB
        # even if 3 frames of width 3 were intended.
        out, _ = self._inject(np.zeros((3, 4, 6), dtype=np.uint8))
        self.assertEqual((out.NumberOfFrames, out.SamplesPerPixel), (3, 1))
        out, _ = self._inject(np.zeros((4, 6, 3), dtype=np.uint16))
        self.assertEqual((out.SamplesPerPixel, out.BitsAllocated), (3, 16))

    def test_frames_rgb_dhw3_and_dhw4(self):
        for ch in (3, 4):
            out, _ = self._inject(np.zeros((5, 4, 6, ch), dtype=np.uint8))
            self.assertEqual((out.NumberOfFrames, out.SamplesPerPixel, out.PlanarConfiguration), (5, 3, 0))
            self.assertEqual((out.Rows, out.Columns), (4, 6))
            self.assertEqual(len(out.PixelData), 5 * 4 * 6 * 3)

    def test_unsupported_shapes_raise_metadata_scrub_error(self):
        for shape in ((5,), (5, 4, 6, 1), (2, 5, 4, 6, 3)):
            with self.assertRaises(MetadataScrubError) as ctx:
                self._inject(np.zeros(shape, dtype=np.uint8))
            self.assertEqual(ctx.exception.transform, "ScrubDicomMetadata")
            self.assertIn("Unsupported pixel array shape", str(ctx.exception))

    def test_header_only_leaves_pixel_related_tags_untouched(self):
        out, rows = self.t.scrub("f.dcm", dataset=self._base())
        self.assertEqual(out.file_meta.TransferSyntaxUID, pydicom.uid.ImplicitVRLittleEndian)
        self.assertEqual(out.NumberOfFrames, 7)
        self.assertEqual((str(out.WindowCenter), str(out.WindowWidth)), ("999", "999"))
        self.assertEqual(out.BurnedInAnnotation, "YES")  # not overridden without pixel cleaning
        self.assertNotIn("BurnedInAnnotation", [r["keyword"] for r in rows])

    def test_injection_without_file_meta_does_not_raise(self):
        ds = Dataset()
        ds.SOPInstanceUID = _FOREIGN + "1"
        out, _ = self.t.scrub("f.dcm", pixel_data=np.zeros((2, 2), dtype=np.uint8), dataset=ds)
        self.assertEqual(len(out.PixelData), 4)

    def test_injection_runs_after_tag_scrub_so_injected_tags_win(self):
        # Ordering contract: scrub -> private removal -> pixel injection ->
        # UID remap -> attestation. A mapping that REMOVEs WindowCenter is
        # overridden by the injection step.
        t = ScrubDicomMetadata(_cfg({"(0028,1050)": "REMOVE"}))
        out, rows = t.scrub("f.dcm", pixel_data=np.zeros((2, 2), dtype=np.uint8), dataset=self._base())
        self.assertIn("WindowCenter", out)
        self.assertIn("WindowCenter", [r["keyword"] for r in rows if r["action"] == "REMOVE"])


# ---------------------------------------------------------------------------
# 4. PS3.15 attestation stamping
# ---------------------------------------------------------------------------

class TestAttestationCharacterization(unittest.TestCase):

    def test_exact_method_strings_and_code_items(self):
        from monai_aegis import __version__
        t = ScrubDicomMetadata(_cfg())
        out, _ = t.scrub("f.dcm", dataset=Dataset())
        self.assertEqual(list(out.DeidentificationMethod), [f"MONAI Aegis {__version__}", "PS3.15 header scrub"])
        seq = out.DeidentificationMethodCodeSequence
        self.assertEqual(len(seq), 1)
        self.assertEqual((seq[0].CodeValue, seq[0].CodingSchemeDesignator, seq[0].CodeMeaning),
                         ("113100", "DCM", "Basic Application Confidentiality Profile"))

    def test_pixel_path_appends_clean_pixel_method_and_code(self):
        t = ScrubDicomMetadata(_cfg())
        out, _ = t.scrub("f.dcm", pixel_data=np.zeros((2, 2), dtype=np.uint8), dataset=Dataset())
        self.assertEqual(list(out.DeidentificationMethod)[2], "Pixel PHI redaction (OCR/NER)")
        seq = out.DeidentificationMethodCodeSequence
        self.assertEqual([(i.CodeValue, i.CodeMeaning) for i in seq],
                         [("113100", "Basic Application Confidentiality Profile"),
                          ("113101", "Clean Pixel Data Option")])
        self.assertEqual(out.BurnedInAnnotation, "NO")

    def test_existing_attestation_on_source_is_replaced_not_appended(self):
        t = ScrubDicomMetadata(_cfg())
        ds = Dataset()
        ds.DeidentificationMethod = ["Some other tool"]
        item = Dataset(); item.CodeValue = "999999"; item.CodingSchemeDesignator = "DCM"; item.CodeMeaning = "x"
        ds.DeidentificationMethodCodeSequence = [item]
        out, _ = t.scrub("f.dcm", dataset=ds)
        self.assertNotIn("Some other tool", list(out.DeidentificationMethod))
        self.assertEqual([i.CodeValue for i in out.DeidentificationMethodCodeSequence], ["113100"])

    def test_attest_rows_are_always_the_trailing_rows(self):
        t = ScrubDicomMetadata(_cfg({"(0010,0010)": "DUMMY"}))
        ds = Dataset(); ds.PatientName = "X"; ds.SOPInstanceUID = _FOREIGN + "1"
        _, rows = t.scrub("f.dcm", pixel_data=np.zeros((2, 2), dtype=np.uint8), dataset=ds)
        self.assertEqual([r["tag"] for r in rows[-4:]],
                         ["(0012,0062)", "(0012,0063)", "(0012,0064)", "(0028,0301)"])
        self.assertTrue(all(r["redacted"] is False for r in rows[-4:]))


# ---------------------------------------------------------------------------
# 5. Dictionary transform — real inner transform, no mocks
# ---------------------------------------------------------------------------

class TestDictTransformSingleMode(unittest.TestCase):

    def setUp(self):
        self.tr = ScrubDicomMetadatad(keys=["image"], config=_cfg({"(0010,0010)": "DUMMY"}))

    def _data(self, tensor, ds, fpath="/in/a.dcm"):
        return {"image": tensor, "image_meta_dict": {"filename_or_obj": fpath}, "image_dicom_dataset": ds}

    def test_outputs_and_metatensor_roundtrip(self):
        img = MetaTensor(torch.full((1, 4, 6), 777.0), meta={"filename_or_obj": "/in/a.dcm"})
        d = self._data(img, _ds_with_pixels())
        out = self.tr(d)
        self.assertEqual(sorted(k for k in out if k not in d), ["image_scrubbed_ds", "image_tag_actions"])
        self.assertNotIn("image_scrubbed_ds", d)  # input dict untouched
        self.assertIsInstance(out["image"], MetaTensor)
        self.assertEqual(tuple(out["image"].shape), (1, 4, 6))
        self.assertEqual(out["image"].dtype, torch.float32)
        self.assertEqual(out["image"].meta["filename_or_obj"], "/in/a.dcm")
        self.assertEqual(int(out["image_scrubbed_ds"].pixel_array[0, 0]), 777)
        self.assertEqual(float(out["image"][0, 0, 0]), 777.0)
        self.assertTrue(str(out["image_scrubbed_ds"].PatientName).startswith("TOKEN_"))
        self.assertTrue(any(r["action"] == "DUMMY" for r in out["image_tag_actions"]))

    def test_normalized_input_is_rescaled_by_largest_image_pixel_value(self):
        out = self.tr(self._data(MetaTensor(torch.full((1, 4, 6), 0.5)), _ds_with_pixels(largest=4000)))
        self.assertEqual(int(out["image_scrubbed_ds"].pixel_array[0, 0]), 2000)

    def test_normalized_input_defaults_to_4095_when_largest_absent(self):
        out = self.tr(self._data(MetaTensor(torch.full((1, 4, 6), 1.0)), _ds_with_pixels(largest=None)))
        self.assertEqual(int(out["image_scrubbed_ds"].pixel_array[0, 0]), 4095)

    def test_plain_ndarray_input_yields_plain_ndarray_output(self):
        out = self.tr(self._data(np.full((1, 4, 6), 500.0, dtype=np.float32), _ds_with_pixels()))
        self.assertIsInstance(out["image"], np.ndarray)
        self.assertEqual(out["image"].shape, (1, 4, 6))

    def test_rgb_single_roundtrip(self):
        ds = _ds_with_pixels()
        ds.SamplesPerPixel = 3; ds.PhotometricInterpretation = "RGB"; ds.PlanarConfiguration = 0
        ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7
        ds.PixelData = np.zeros((4, 6, 3), dtype=np.uint8).tobytes()
        out = self.tr(self._data(MetaTensor(torch.full((3, 4, 6), 200.0)), ds))
        self.assertEqual(out["image_scrubbed_ds"].SamplesPerPixel, 3)
        self.assertEqual(tuple(out["image"].shape), (3, 4, 6))

    def test_2d_input_is_passed_through_without_channel_handling(self):
        out = self.tr(self._data(np.full((4, 6), 500.0, dtype=np.float32), _ds_with_pixels()))
        self.assertEqual(out["image_scrubbed_ds"].pixel_array.shape, (4, 6))
        self.assertEqual(out["image"].shape, (1, 4, 6))  # re-expanded to channel-first on the way back

    def test_missing_cached_dataset_falls_back_to_dcmread(self):
        ds = _ds_with_pixels()
        d = {"image": MetaTensor(torch.full((1, 4, 6), 500.0)), "image_meta_dict": {"filename_or_obj": "/in/a.dcm"},
             "image_dicom_datasets": None}  # key present -> treated as DICOM; no single dataset cached
        with patch("monai_aegis.transforms.metadata.pydicom.dcmread", return_value=ds) as m:
            out = self.tr(d)
        m.assert_called_once_with("/in/a.dcm")
        self.assertIn("image_scrubbed_ds", out)

    def test_non_dicom_and_missing_filename_are_skipped(self):
        out = self.tr({"image": MetaTensor(torch.zeros((3, 4, 6))), "image_meta_dict": {"filename_or_obj": "/in/a.jpg"}})
        self.assertEqual(sorted(out), ["image", "image_meta_dict"])
        out = self.tr({"image": MetaTensor(torch.zeros((1, 4, 6))), "image_meta_dict": {}, "image_dicom_dataset": _ds_with_pixels()})
        self.assertNotIn("image_scrubbed_ds", out)

    def test_filename_list_uses_first_entry(self):
        d = {"image": MetaTensor(torch.full((1, 4, 6), 500.0)),
             "image_meta_dict": {"filename_or_obj": ["/in/a.dcm", "/in/b.dcm"]},
             "image_dicom_dataset": _ds_with_pixels()}
        self.assertIn("image_scrubbed_ds", self.tr(d))

    def test_generic_failure_is_wrapped_with_uri_and_transform(self):
        d = self._data(MetaTensor(torch.full((1, 4, 6), 500.0)), _ds_with_pixels(), fpath="/in/broken.dcm")
        with patch.object(self.tr.transform, "scrub", side_effect=RuntimeError("boom")):
            with self.assertRaises(MetadataScrubError) as ctx:
                self.tr(d)
        self.assertEqual(ctx.exception.transform, "ScrubDicomMetadatad")
        self.assertEqual(ctx.exception.uri, "/in/broken.dcm")
        self.assertIn("boom", str(ctx.exception))

    def test_inner_metadata_scrub_error_is_reraised_unwrapped(self):
        d = self._data(MetaTensor(torch.full((1, 4, 6), 500.0)), _ds_with_pixels())
        inner = MetadataScrubError("inner", transform="ScrubDicomMetadata")
        with patch.object(self.tr.transform, "scrub", side_effect=inner):
            with self.assertRaises(MetadataScrubError) as ctx:
                self.tr(d)
        self.assertIs(ctx.exception, inner)

    def test_missing_key_raises_keyerror_from_monai_not_wrapped(self):
        with self.assertRaises(KeyError):
            self.tr({"other": 1})


class TestDictTransformSeriesMode(unittest.TestCase):

    def _series(self, n=3, mapping=None):
        mapping = mapping if mapping is not None else {"(0010,0010)": "DUMMY"}
        tr = ScrubDicomMetadatad(keys=["image"], config=_cfg(mapping))
        dss = [_ds_with_pixels(i) for i in range(n)]
        vol = MetaTensor(torch.full((1, n, 4, 6), 0.25), meta={"filename_or_obj": "/in/s0.dcm"})
        d = {"image": vol,
             "image_meta_dict": {"filename_or_obj": "/in/s0.dcm", "is_multiframe": False,
                                 "slice_uris": [f"/in/s{i}.dcm" for i in range(n)]},
             "image_dicom_datasets": dss}
        return tr, d

    def test_outputs_keys_and_uid_linkage(self):
        tr, d = self._series()
        out = tr(d)
        self.assertEqual(sorted(k for k in out if k not in d), ["image_scrubbed_datasets", "image_tag_actions_per_slice"])
        self.assertNotIn("image_scrubbed_ds", out)
        sds = out["image_scrubbed_datasets"]
        self.assertEqual(len(sds), 3)
        self.assertEqual(len({str(s.SOPInstanceUID) for s in sds}), 3)
        self.assertEqual(len({str(s.SeriesInstanceUID) for s in sds}), 1)
        self.assertTrue(str(sds[0].SeriesInstanceUID).startswith(_PYDICOM_ROOT))
        self.assertEqual(len(out["image_tag_actions_per_slice"]), 3)
        for s in sds:
            self.assertEqual(str(s.file_meta.MediaStorageSOPInstanceUID), str(s.SOPInstanceUID))

    def test_volume_is_rewritten_from_scrubbed_pixels_and_stays_metatensor(self):
        tr, d = self._series()
        out = tr(d)
        self.assertIsInstance(out["image"], MetaTensor)
        self.assertEqual(tuple(out["image"].shape), (1, 3, 4, 6))
        self.assertEqual(float(out["image"][0, 1, 0, 0]), 0.25 * 4000)
        self.assertEqual(out["image"].meta["filename_or_obj"], "/in/s0.dcm")
        self.assertEqual(int(out["image_scrubbed_datasets"][2].pixel_array[0, 0]), 1000)

    def test_geometry_tags_are_restored_even_when_mapping_targets_them(self):
        # quirk: series mode restores geometry after scrubbing; single mode
        # does not. Pinning the divergence so the refactor preserves it.
        tr, d = self._series(mapping={"(0028,0030)": "REMOVE"})  # PixelSpacing
        out = tr(d)
        for s in out["image_scrubbed_datasets"]:
            self.assertEqual([float(v) for v in s.PixelSpacing], [0.5, 0.5])
        single = ScrubDicomMetadatad(keys=["image"], config=_cfg({"(0028,0030)": "REMOVE"}))
        o = single({"image": MetaTensor(torch.full((1, 4, 6), 500.0)),
                    "image_meta_dict": {"filename_or_obj": "/in/a.dcm"}, "image_dicom_dataset": _ds_with_pixels()})
        self.assertNotIn("PixelSpacing", o["image_scrubbed_ds"])

    def test_per_slice_rows_have_row_schema_and_sections(self):
        tr, d = self._series()
        out = tr(d)
        for rows in out["image_tag_actions_per_slice"]:
            self.assertEqual([r["action"] for r in rows],
                             ["DUMMY", "REMAP", "REMAP", "REMAP", "ATTEST", "ATTEST", "ATTEST", "ATTEST"])
            self.assertTrue(all(set(r) == _ROW_KEYS for r in rows))

    def test_slice_uri_fallback_when_fewer_uris_than_slices(self):
        tr, d = self._series()
        d["image_meta_dict"]["slice_uris"] = ["/in/only.dcm"]
        out = tr(d)
        self.assertEqual(len(out["image_scrubbed_datasets"]), 3)

    def test_non_4d_volume_falls_back_to_single_mode_path(self):
        tr, d = self._series(n=1)
        d["image"] = MetaTensor(torch.full((1, 4, 6), 500.0))
        d["image_dicom_dataset"] = d["image_dicom_datasets"][0]  # what the single path looks up
        out = tr(d)
        self.assertIn("image_scrubbed_ds", out)
        self.assertNotIn("image_scrubbed_datasets", out)

    def test_raw_scale_slices_are_not_rescaled(self):
        tr, d = self._series()
        d["image"] = MetaTensor(torch.full((1, 3, 4, 6), 777.0))
        out = tr(d)
        self.assertEqual(int(out["image_scrubbed_datasets"][0].pixel_array[0, 0]), 777)

    def test_rgb_series_roundtrip(self):
        tr, d = self._series()
        for ds in d["image_dicom_datasets"]:
            ds.SamplesPerPixel = 3; ds.PhotometricInterpretation = "RGB"; ds.PlanarConfiguration = 0
            ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7
            ds.PixelData = np.zeros((4, 6, 3), dtype=np.uint8).tobytes()
        d["image"] = MetaTensor(torch.full((3, 3, 4, 6), 200.0))
        out = tr(d)
        self.assertEqual(out["image_scrubbed_datasets"][0].SamplesPerPixel, 3)
        self.assertEqual(tuple(out["image"].shape), (3, 3, 4, 6))
        self.assertEqual(float(out["image"][2, 1, 0, 0]), 200.0)

    def test_slices_without_geometry_tags_are_fine_and_plain_ndarray_output(self):
        tr, d = self._series()
        for ds in d["image_dicom_datasets"]:
            for kw in ("PixelSpacing", "SliceThickness", "ImageOrientationPatient",
                       "ImagePositionPatient", "SpacingBetweenSlices"):
                delattr(ds, kw)
        d["image"] = np.full((1, 3, 4, 6), 500.0, dtype=np.float32)
        out = tr(d)
        self.assertIsInstance(out["image"], np.ndarray)
        self.assertEqual(len(out["image_scrubbed_datasets"]), 3)
        self.assertNotIn("PixelSpacing", out["image_scrubbed_datasets"][0])


class TestDictTransformMultiframeMode(unittest.TestCase):

    def _mf(self, channels=1):
        ds = _ds_with_pixels()
        ds.NumberOfFrames = "3"
        if channels == 3:
            ds.SamplesPerPixel = 3; ds.PhotometricInterpretation = "RGB"; ds.PlanarConfiguration = 0
            ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7
            ds.PixelData = np.zeros((3, 4, 6, 3), dtype=np.uint8).tobytes()
        else:
            ds.PixelData = np.full((3, 4, 6), 900, dtype=np.uint16).tobytes()
        tr = ScrubDicomMetadatad(keys=["image"], config=_cfg())
        d = {"image": MetaTensor(torch.full((channels, 3, 4, 6), 200.0)),
             "image_meta_dict": {"filename_or_obj": "/in/mf.dcm", "is_multiframe": True, "slice_uris": ["/in/mf.dcm"]},
             "image_dicom_datasets": [ds]}
        return tr, d

    def test_multiframe_mono_roundtrip(self):
        tr, d = self._mf()
        out = tr(d)
        self.assertEqual(sorted(k for k in out if k not in d), ["image_scrubbed_ds", "image_tag_actions"])
        self.assertEqual(out["image_scrubbed_ds"].NumberOfFrames, 3)
        self.assertEqual(tuple(out["image"].shape), (1, 3, 4, 6))
        self.assertEqual(int(out["image_scrubbed_ds"].pixel_array[1, 0, 0]), 200)

    def test_multiframe_rgb_roundtrip(self):
        tr, d = self._mf(channels=3)
        out = tr(d)
        self.assertEqual(out["image_scrubbed_ds"].SamplesPerPixel, 3)
        self.assertEqual(tuple(out["image"].shape), (3, 3, 4, 6))

    def test_multiframe_normalized_input_is_rescaled_and_ndarray_output(self):
        tr, d = self._mf()
        d["image"] = np.full((1, 3, 4, 6), 0.5, dtype=np.float32)
        out = tr(d)
        self.assertIsInstance(out["image"], np.ndarray)
        self.assertEqual(int(out["image_scrubbed_ds"].pixel_array[0, 0, 0]), 2000)  # 0.5 * LargestImagePixelValue

    def test_multiframe_rejects_non_4d_and_odd_channel_layout(self):
        tr, d = self._mf()
        d["image"] = MetaTensor(torch.zeros((1, 4, 6)))
        with self.assertRaises(MetadataScrubError) as ctx:
            tr(d)
        self.assertEqual(ctx.exception.transform, "ScrubDicomMetadatad")
        tr, d = self._mf()
        d["image"] = MetaTensor(torch.zeros((2, 3, 4, 6)))
        with self.assertRaises(MetadataScrubError):
            tr(d)


if __name__ == "__main__":
    unittest.main()
