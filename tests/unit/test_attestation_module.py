"""Unit tests for ``monai_aegis.transforms.metadata.attestation`` in isolation.

``stamp_attestation`` is exercised directly on minimal datasets so each
responsibility is pinned on its own: the four attributes it writes, their
VRs and values, the ATTEST rows it records, and that the result satisfies
the verifier's attestation checks. The transform-level contract is covered
by ``test_attestation.py`` and ``test_metadata_characterization.py``.
"""
import unittest

import pydicom.tag
from pydicom.dataset import Dataset

from monai_aegis import __version__, verify
from monai_aegis.transforms.metadata import constants
from monai_aegis.transforms.metadata.attestation import (
    deidentification_codes,
    deidentification_methods,
    stamp_attestation,
)
from monai_aegis.transforms.metadata.audit import TagAuditTrail

_LO_MAX_LEN = 64


class TestValueHelpers(unittest.TestCase):

    def test_methods_header_only_and_pixel(self):
        self.assertEqual(deidentification_methods(False),
                         [f"MONAI Aegis {__version__}", "PS3.15 header scrub"])
        self.assertEqual(deidentification_methods(True),
                         [f"MONAI Aegis {__version__}", "PS3.15 header scrub", "Pixel PHI redaction (OCR/NER)"])

    def test_codes_header_only_and_pixel(self):
        self.assertEqual(deidentification_codes(False), [("113100", "Basic Application Confidentiality Profile")])
        self.assertEqual(deidentification_codes(True), [("113100", "Basic Application Confidentiality Profile"),
                                                        ("113101", "Clean Pixel Data Option")])

    def test_every_method_value_fits_the_lo_vr(self):
        for value in deidentification_methods(True):
            self.assertLessEqual(len(value), _LO_MAX_LEN, value)

    def test_helpers_return_fresh_lists(self):
        a, b = deidentification_methods(True), deidentification_methods(True)
        self.assertIsNot(a, b)
        a.append("x")
        self.assertNotIn("x", deidentification_methods(True))
        self.assertNotIn("x", deidentification_codes(True))


class TestStampAttestation(unittest.TestCase):

    def setUp(self):
        self.trail = TagAuditTrail()

    def test_header_only_attributes_values_and_vrs(self):
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=False, trail=self.trail)
        self.assertEqual(ds.PatientIdentityRemoved, "YES")
        self.assertEqual(ds["PatientIdentityRemoved"].VR, "CS")
        self.assertEqual(list(ds.DeidentificationMethod), deidentification_methods(False))
        self.assertEqual(ds["DeidentificationMethod"].VR, "LO")
        self.assertEqual(ds["DeidentificationMethodCodeSequence"].VR, "SQ")
        self.assertNotIn("BurnedInAnnotation", ds)
        self.assertEqual(sorted(e.keyword for e in ds),
                         ["DeidentificationMethod", "DeidentificationMethodCodeSequence", "PatientIdentityRemoved"])

    def test_code_sequence_item_structure(self):
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        seq = ds.DeidentificationMethodCodeSequence
        self.assertEqual(len(seq), 2)
        for item, (value, meaning) in zip(seq, deidentification_codes(True)):
            self.assertEqual([e.keyword for e in item], ["CodeValue", "CodingSchemeDesignator", "CodeMeaning"])
            self.assertEqual((item.CodeValue, item.CodingSchemeDesignator, item.CodeMeaning),
                             (value, constants.CODING_SCHEME_DCM, meaning))
            self.assertEqual((item["CodeValue"].VR, item["CodingSchemeDesignator"].VR, item["CodeMeaning"].VR),
                             ("SH", "SH", "LO"))

    def test_pixel_path_adds_burned_in_annotation(self):
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        self.assertEqual(ds.BurnedInAnnotation, "NO")
        self.assertEqual(ds["BurnedInAnnotation"].VR, "CS")
        self.assertEqual(list(ds.DeidentificationMethod), deidentification_methods(True))

    def test_header_only_leaves_source_burned_in_annotation_untouched(self):
        ds = Dataset()
        ds.BurnedInAnnotation = "YES"
        stamp_attestation(ds, pixel_redacted=False, trail=self.trail)
        self.assertEqual(ds.BurnedInAnnotation, "YES")
        self.assertNotIn("BurnedInAnnotation", [r["keyword"] for r in self.trail])

    def test_source_attestation_values_are_replaced(self):
        ds = Dataset()
        ds.PatientIdentityRemoved = "NO"
        ds.DeidentificationMethod = ["Other tool", "Manual review"]
        item = Dataset(); item.CodeValue = "999"; item.CodingSchemeDesignator = "99X"; item.CodeMeaning = "x"
        ds.DeidentificationMethodCodeSequence = [item]
        ds.BurnedInAnnotation = "YES"
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        self.assertEqual(ds.PatientIdentityRemoved, "YES")
        self.assertEqual(list(ds.DeidentificationMethod), deidentification_methods(True))
        self.assertEqual([i.CodeValue for i in ds.DeidentificationMethodCodeSequence], ["113100", "113101"])
        self.assertEqual(ds.BurnedInAnnotation, "NO")

    def test_other_attributes_are_untouched(self):
        ds = Dataset()
        ds.PatientName = "Doe^J"
        ds.Modality = "CT"
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        self.assertEqual((str(ds.PatientName), ds.Modality), ("Doe^J", "CT"))
        self.assertEqual(len(ds), 2 + 4)

    def test_attest_rows_header_only_and_pixel(self):
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=False, trail=self.trail)
        self.assertEqual(self.trail.to_list(), [
            {"tag": "(0012,0062)", "keyword": "PatientIdentityRemoved", "action": "ATTEST", "redacted": False},
            {"tag": "(0012,0063)", "keyword": "DeidentificationMethod", "action": "ATTEST", "redacted": False},
            {"tag": "(0012,0064)", "keyword": "DeidentificationMethodCodeSequence", "action": "ATTEST",
             "redacted": False},
        ])
        trail = TagAuditTrail()
        stamp_attestation(Dataset(), pixel_redacted=True, trail=trail)
        self.assertEqual([r["tag"] for r in trail], ["(0012,0062)", "(0012,0063)", "(0012,0064)", "(0028,0301)"])
        self.assertTrue(all(r["action"] == constants.ACTION_ATTEST and r["redacted"] is False for r in trail))

    def test_row_tags_match_the_real_element_tags(self):
        # The tag strings are static constants; make sure they point at the
        # elements actually written.
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        for row in self.trail:
            g, e = (int(x, 16) for x in row["tag"].strip("()").split(","))
            self.assertIn(pydicom.tag.Tag(g, e), ds, row)
            self.assertEqual(ds[pydicom.tag.Tag(g, e)].keyword, row["keyword"])

    def test_rows_are_appended_after_existing_trail_rows(self):
        self.trail.record_remap(pydicom.tag.Tag(0x0008, 0x0018), "SOPInstanceUID")
        stamp_attestation(Dataset(), pixel_redacted=False, trail=self.trail)
        self.assertEqual([r["action"] for r in self.trail], ["REMAP", "ATTEST", "ATTEST", "ATTEST"])

    def test_stamping_is_in_place_and_idempotent(self):
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        snapshot = str(ds)
        stamp_attestation(ds, pixel_redacted=True, trail=TagAuditTrail())
        self.assertEqual(str(ds), snapshot)
        self.assertEqual(len(ds.DeidentificationMethodCodeSequence), 2)

    def test_stamped_dataset_passes_the_verifier_attestation_checks(self):
        # Closes the loop with aegis-verify: the packaged checklist's
        # attestation checks must accept exactly what this module writes.
        ds = Dataset()
        stamp_attestation(ds, pixel_redacted=True, trail=self.trail)
        checklist = verify.load_checklist(verify.default_checklist_path())
        attestation_checks = [c for c in checklist["dicom_checks"]
                              if c["id"].startswith("attestation-") or c["id"] == "burned-in-annotation-cleared"]
        self.assertEqual(len(attestation_checks), 3)
        for check in attestation_checks:
            ok, detail = verify._check_dataset(ds, check)
            self.assertTrue(ok, f"{check['id']}: {detail}")


if __name__ == "__main__":
    unittest.main()
