"""Unit tests for ``monai_aegis.transforms.metadata.audit`` in isolation.

Pins the tag-action row contract consumed by ``monai_aegis.reporting`` and
``monai_aegis.envelope``: key set and order, tag formatting, and the
``redacted`` flag each record method assigns.
"""
import unittest

import pydicom.tag

from monai_aegis.transforms.metadata import constants
from monai_aegis.transforms.metadata.audit import TagAuditTrail, format_tag, keyword_for, make_row


class TestRowHelpers(unittest.TestCase):

    def test_format_tag_is_upper_hex_in_parentheses(self):
        self.assertEqual(format_tag(pydicom.tag.Tag(0x0010, 0x0010)), "(0010,0010)")
        self.assertEqual(format_tag(pydicom.tag.Tag(0x0020, 0x000D)), "(0020,000D)")
        self.assertEqual(format_tag(pydicom.tag.Tag(0x7FE0, 0x0010)), "(7FE0,0010)")

    def test_keyword_for_known_and_private_tags(self):
        self.assertEqual(keyword_for(pydicom.tag.Tag(0x0010, 0x0010)), "PatientName")
        self.assertEqual(keyword_for(pydicom.tag.Tag(0x0009, 0x1001)), "")

    def test_make_row_key_order_and_values(self):
        row = make_row("(0010,0010)", "PatientName", constants.ACTION_DUMMY, True)
        self.assertEqual(list(row), [constants.ROW_KEY_TAG, constants.ROW_KEY_KEYWORD,
                                     constants.ROW_KEY_ACTION, constants.ROW_KEY_REDACTED])
        self.assertEqual(row, {"tag": "(0010,0010)", "keyword": "PatientName", "action": "DUMMY", "redacted": True})


class TestTagAuditTrail(unittest.TestCase):

    def setUp(self):
        self.trail = TagAuditTrail()
        self.tag = pydicom.tag.Tag(0x0010, 0x0020)

    def test_starts_empty(self):
        self.assertEqual(len(self.trail), 0)
        self.assertEqual(list(self.trail), [])
        self.assertEqual(self.trail.to_list(), [])

    def test_record_action_redacted_flag_per_action(self):
        for action in (constants.ACTION_REMOVE, constants.ACTION_ZERO, constants.ACTION_DUMMY):
            self.trail.record_action(self.tag, "PatientID", action)
        self.trail.record_action(self.tag, "PatientID", constants.ACTION_KEEP)
        rows = self.trail.to_list()
        self.assertEqual([r["action"] for r in rows], ["REMOVE", "ZERO", "DUMMY", "KEEP"])
        self.assertEqual([r["redacted"] for r in rows], [True, True, True, False])
        self.assertTrue(all(r["tag"] == "(0010,0020)" and r["keyword"] == "PatientID" for r in rows))

    def test_record_remap(self):
        self.trail.record_remap(pydicom.tag.Tag(0x0008, 0x0018), "SOPInstanceUID")
        self.assertEqual(self.trail.to_list(),
                         [{"tag": "(0008,0018)", "keyword": "SOPInstanceUID", "action": "REMAP", "redacted": True}])

    def test_record_attest_takes_preformatted_tag(self):
        self.trail.record_attest("(0012,0062)", "PatientIdentityRemoved")
        self.assertEqual(self.trail.to_list(),
                         [{"tag": "(0012,0062)", "keyword": "PatientIdentityRemoved",
                           "action": "ATTEST", "redacted": False}])

    def test_order_is_insertion_order_and_len_iter_agree(self):
        self.trail.record_attest("(0012,0062)", "PatientIdentityRemoved")
        self.trail.record_action(self.tag, "PatientID", constants.ACTION_DUMMY)
        self.trail.record_remap(pydicom.tag.Tag(0x0008, 0x0018), "SOPInstanceUID")
        self.assertEqual([r["action"] for r in self.trail], ["ATTEST", "DUMMY", "REMAP"])
        self.assertEqual(len(self.trail), 3)
        self.assertEqual(list(self.trail), self.trail.to_list())

    def test_to_list_is_a_plain_list_of_plain_dicts(self):
        self.trail.record_remap(self.tag, "PatientID")
        rows = self.trail.to_list()
        self.assertIs(type(rows), list)
        self.assertIs(type(rows[0]), dict)

    def test_trails_are_independent(self):
        other = TagAuditTrail()
        self.trail.record_remap(self.tag, "PatientID")
        self.assertEqual(len(other), 0)


if __name__ == "__main__":
    unittest.main()
