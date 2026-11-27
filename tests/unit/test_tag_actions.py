"""Unit tests for ``monai_aegis.transforms.metadata.tag_actions`` in isolation.

Covers the pieces the transform-level tests cannot reach (the unknown-action
guard, the absent-tag no-op, the parsed-table accessor) and pins the parsing
/ VR-fit helpers on their own. Transform-level behavior of the same code is
covered by ``test_attestation.py`` (validation), ``test_metadata_scrubber.py``
(VR fit, nested scrub) and ``test_metadata_characterization.py``.
"""
import unittest

import pydicom.tag
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from monai_aegis.transforms.exceptions import MetadataScrubError
from monai_aegis.transforms.metadata import constants
from monai_aegis.transforms.metadata.audit import TagAuditTrail
from monai_aegis.transforms.metadata.tag_actions import (
    TagActionScrubber,
    build_pii_actions,
    fit_token_to_vr,
    parse_hex_part,
    parse_tag_key,
)
from monai_aegis.transforms.utility import AegisIdentityManager


class TestParsing(unittest.TestCase):

    def test_parse_hex_part_forms(self):
        self.assertEqual(parse_hex_part("0010"), 0x10)
        self.assertEqual(parse_hex_part(" 0x0010 "), 0x10)
        self.assertEqual(parse_hex_part("0X7FE0"), 0x7FE0)
        self.assertEqual(parse_hex_part("0020"), 0x20)   # never octal / decimal

    def test_parse_hex_part_rejects_non_hex(self):
        with self.assertRaises(ValueError):
            parse_hex_part("zz")

    def test_parse_tag_key_forms(self):
        for key in ("(0010,0010)", "(0010, 0010)", "(0x0010,0x0010)", " 0010,0010 ", "(0010 ,0010)"):
            self.assertEqual(parse_tag_key(key), (0x10, 0x10), key)

    def test_parse_tag_key_rejects_wrong_arity(self):
        for key in ("(0010)", "(0010,0010,0010)", ""):
            with self.assertRaises(ValueError):
                parse_tag_key(key)

    def test_build_pii_actions_normalises_and_validates(self):
        table = build_pii_actions({"(0010,0010)": "remove", "(0010, 0020)": "Dummy", "(0x0008,0x0020)": "ZERO"})
        self.assertEqual(table, {pydicom.tag.Tag(0x0010, 0x0010): "REMOVE",
                                 pydicom.tag.Tag(0x0010, 0x0020): "DUMMY",
                                 pydicom.tag.Tag(0x0008, 0x0020): "ZERO"})
        self.assertEqual(build_pii_actions({}), {})

    def test_build_pii_actions_error_messages(self):
        with self.assertRaises(ValueError) as ctx:
            build_pii_actions({"(0010)": "REMOVE"})
        self.assertIn("Invalid DICOM tag '(0010)' in pii_mapping", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            build_pii_actions({"(0010,0010)": "DELETE"})
        self.assertIn("Unknown pii_mapping action 'DELETE'", str(ctx.exception))
        self.assertIn(str(sorted(constants.VALID_ACTIONS)), str(ctx.exception))

    def test_fit_token_to_vr_uses_constants_table(self):
        token = "TOKEN_" + "a" * 16
        for vr, limit in constants.VR_MAX_LEN.items():
            self.assertEqual(fit_token_to_vr(token, vr), token[:limit])
        for vr in ("LO", "PN", "LT", "UT", "DA"):
            self.assertEqual(fit_token_to_vr(token, vr), token)
        self.assertEqual(fit_token_to_vr("short", "SH"), "short")


class TestTagActionScrubber(unittest.TestCase):

    def setUp(self):
        self.idm = AegisIdentityManager(salt="unit")
        self.trail = TagAuditTrail()

    def _scrubber(self, mapping):
        return TagActionScrubber(build_pii_actions(mapping), self.idm)

    def test_pii_actions_property_exposes_parsed_table(self):
        s = self._scrubber({"(0010,0010)": "REMOVE"})
        self.assertEqual(s.pii_actions, {pydicom.tag.Tag(0x0010, 0x0010): "REMOVE"})

    def test_apply_action_on_absent_tag_is_a_silent_no_op(self):
        s = self._scrubber({})
        ds = Dataset()
        s.apply_action(ds, pydicom.tag.Tag(0x0010, 0x0010), constants.ACTION_REMOVE, self.trail)
        self.assertEqual(len(self.trail), 0)
        self.assertEqual(len(ds), 0)

    def test_apply_action_unknown_action_raises_and_records_nothing(self):
        s = self._scrubber({})
        ds = Dataset()
        ds.PatientName = "X"
        with self.assertRaises(MetadataScrubError) as ctx:
            s.apply_action(ds, pydicom.tag.Tag(0x0010, 0x0010), "SHRED", self.trail)
        self.assertEqual(ctx.exception.transform, constants.ARRAY_TRANSFORM_NAME)
        self.assertEqual(len(self.trail), 0)
        self.assertEqual(str(ds.PatientName), "X")  # value untouched

    def test_each_action_value_effect(self):
        s = self._scrubber({})
        ds = Dataset()
        ds.PatientName = "Doe^J"; ds.PatientID = "MRN"; ds.StudyDate = "20240101"
        ds.add_new((0x0042, 0x0011), "OB", b"\x01")
        ds.StudyID = "788962607"
        s.apply_action(ds, pydicom.tag.Tag(0x0010, 0x0010), constants.ACTION_REMOVE, self.trail)
        s.apply_action(ds, pydicom.tag.Tag(0x0010, 0x0020), constants.ACTION_KEEP, self.trail)
        s.apply_action(ds, pydicom.tag.Tag(0x0008, 0x0020), constants.ACTION_ZERO, self.trail)
        s.apply_action(ds, pydicom.tag.Tag(0x0042, 0x0011), constants.ACTION_ZERO, self.trail)
        s.apply_action(ds, pydicom.tag.Tag(0x0020, 0x0010), constants.ACTION_DUMMY, self.trail)
        self.assertNotIn("PatientName", ds)
        self.assertEqual(ds.PatientID, "MRN")
        self.assertEqual(ds.StudyDate, constants.ZERO_TEXT_VALUE)
        self.assertEqual(ds[(0x0042, 0x0011)].value, constants.ZERO_BINARY_VALUE)
        self.assertEqual(ds.StudyID, fit_token_to_vr(self.idm.get_token("788962607"), "SH"))
        self.assertEqual([(r["action"], r["redacted"]) for r in self.trail],
                         [("REMOVE", True), ("KEEP", False), ("ZERO", True), ("ZERO", True), ("DUMMY", True)])

    def test_scrub_recursive_children_before_parent_action(self):
        s = self._scrubber({"(0040,0275)": "REMOVE", "(0010,0020)": "DUMMY"})
        inner = Dataset(); inner.PatientID = "INNER"
        ds = Dataset(); ds.PatientID = "TOP"
        ds.add_new((0x0040, 0x0275), "SQ", Sequence([inner]))
        s.scrub_recursive(ds, self.trail)
        self.assertNotIn((0x0040, 0x0275), ds)
        self.assertEqual([r["keyword"] for r in self.trail], ["PatientID", "PatientID", "RequestAttributesSequence"])

    def test_remove_private_tags_recursive_is_static_and_deep(self):
        inner = Dataset()
        inner.private_block(0x0011, "INNER", 0x10).add_new(0x01, "LO", "x")
        ds = Dataset()
        ds.private_block(0x0009, "TOP", 0x10).add_new(0x01, "LO", "y")
        ds.add_new((0x0040, 0x0275), "SQ", Sequence([inner]))
        TagActionScrubber.remove_private_tags_recursive(ds)
        self.assertFalse(any(e.tag.is_private for e in ds.iterall()))
        self.assertIn((0x0040, 0x0275), ds)


if __name__ == "__main__":
    unittest.main()
