"""Unit tests for ``monai_aegis.transforms.metadata.uid_remap`` in isolation.

``UidRemapper`` is exercised directly — no ``ScrubDicomMetadata`` in the
loop — so each rule (deterministic generation, preservation, recursive
traversal, REMAP audit rows, SOP-UID guarantee) is pinned on its own.
The transform-level contract is covered by ``test_metadata_scrubber.py``
and ``test_metadata_characterization.py``.
"""
import unittest

import pydicom
import pydicom.uid
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

from monai_aegis import verify
from monai_aegis.transforms.metadata import constants
from monai_aegis.transforms.metadata.audit import TagAuditTrail
from monai_aegis.transforms.metadata.uid_remap import UidRemapper

_STD = constants.DICOM_STANDARD_UID_ROOT
_FOREIGN = "1.2.840.113619.2.55."
_PYDICOM_ROOT = "1.2.826.0.1.3680043.8.498."


class TestDeterministicGeneration(unittest.TestCase):

    def test_same_salt_same_uid_across_instances(self):
        a = UidRemapper("s").remap_uid(_FOREIGN + "1")
        b = UidRemapper("s").remap_uid(_FOREIGN + "1")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith(_PYDICOM_ROOT))

    def test_different_salt_different_uid(self):
        self.assertNotEqual(UidRemapper("a").remap_uid(_FOREIGN + "1"),
                            UidRemapper("b").remap_uid(_FOREIGN + "1"))

    def test_matches_pydicom_generate_uid_derivation(self):
        # The contract downstream tooling relies on: generate_uid(entropy_srcs=[salt, uid]).
        expected = pydicom.uid.generate_uid(entropy_srcs=["s", _FOREIGN + "1"])
        self.assertEqual(UidRemapper("s").remap_uid(_FOREIGN + "1"), expected)

    def test_none_salt_is_treated_as_empty_string(self):
        expected = pydicom.uid.generate_uid(entropy_srcs=["", _FOREIGN + "1"])
        self.assertEqual(UidRemapper(None).remap_uid(_FOREIGN + "1"), expected)

    def test_source_is_stripped_before_lookup(self):
        r = UidRemapper("s")
        self.assertEqual(r.remap_uid("  " + _FOREIGN + "1  "), r.remap_uid(_FOREIGN + "1"))

    def test_fallback_entropy_used_only_when_uid_is_empty(self):
        r = UidRemapper("s")
        self.assertEqual(r.remap_uid("", fallback_entropy="/p/x.dcm"), r.remap_uid("/p/x.dcm"))
        self.assertEqual(r.remap_uid(None, fallback_entropy="/p/x.dcm"), r.remap_uid("/p/x.dcm"))
        self.assertNotEqual(r.remap_uid(_FOREIGN + "1", fallback_entropy="/p/x.dcm"), r.remap_uid("/p/x.dcm"))

    def test_no_source_at_all_yields_a_fresh_valid_uid_each_time(self):
        r = UidRemapper("s")
        u1, u2 = r.remap_uid(""), r.remap_uid("", fallback_entropy="   ")
        self.assertNotEqual(u1, u2)
        for u in (u1, u2):
            self.assertTrue(pydicom.uid.UID(u).is_valid)

    def test_cache_returns_identical_object_and_is_per_instance(self):
        r = UidRemapper("s")
        first = r.remap_uid(_FOREIGN + "1")
        self.assertIs(r.remap_uid(_FOREIGN + "1"), first)
        self.assertEqual(len(r._cache), 1)
        self.assertEqual(len(UidRemapper("s")._cache), 0)


class TestPreservationRules(unittest.TestCase):

    def setUp(self):
        self.r = UidRemapper("s")

    def test_preserved_keywords_come_from_constants(self):
        for kw in constants.PRESERVE_UID_KEYWORDS:
            self.assertTrue(UidRemapper.is_preserved_keyword(kw))
        for kw in ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
                   "FrameOfReferenceUID", "ReferencedSOPInstanceUID", ""):
            self.assertFalse(UidRemapper.is_preserved_keyword(kw))

    def test_standard_root_and_empty_values_are_standard(self):
        self.assertTrue(UidRemapper.is_standard_uid(_STD + ".1.2"))
        self.assertTrue(UidRemapper.is_standard_uid("  " + _STD + ".5.1.4.1.1.7  "))
        self.assertTrue(UidRemapper.is_standard_uid(""))
        self.assertTrue(UidRemapper.is_standard_uid(None))
        self.assertFalse(UidRemapper.is_standard_uid(_FOREIGN + "1"))
        # quirk (pre-existing, mirrored in monai_aegis.verify): the check is a
        # plain prefix match without a trailing '.', so a foreign root that
        # merely shares the leading digits is also treated as standard.
        # Tightening this is a behavior change for a separate commit.
        self.assertTrue(UidRemapper.is_standard_uid("1.2.840.100081.1"))

    def test_remap_value_passes_standard_and_empty_through_unchanged(self):
        std = pydicom.uid.UID(_STD + ".1.2")
        self.assertIs(self.r.remap_value(std), std)
        self.assertEqual(self.r.remap_value(""), "")
        self.assertIsNone(self.r.remap_value(None))

    def test_remap_value_remaps_foreign_uids_deterministically(self):
        self.assertEqual(self.r.remap_value(_FOREIGN + "1"), self.r.remap_uid(_FOREIGN + "1"))

    def test_rules_stay_in_sync_with_the_verifier_copy(self):
        # monai_aegis.verify duplicates these on purpose (no torch dependency);
        # drift between the two would make aegis-verify disagree with the scrubber.
        self.assertEqual(verify._PRESERVE_UID_KEYWORDS, constants.PRESERVE_UID_KEYWORDS)
        self.assertEqual(verify._DICOM_STANDARD_UID_ROOT, constants.DICOM_STANDARD_UID_ROOT)


class TestRecursiveTraversal(unittest.TestCase):

    def setUp(self):
        self.r = UidRemapper("s")
        self.trail = TagAuditTrail()

    def _ds(self):
        ds = Dataset()
        ds.SOPClassUID = _STD + ".5.1.4.1.1.7"
        ds.SOPInstanceUID = _FOREIGN + "sop"
        ds.StudyInstanceUID = _FOREIGN + "study"
        ds.PatientName = "Not^A^UID"
        inner = Dataset()
        inner.ReferencedSOPClassUID = _STD + ".5.1.4.1.1.7"
        inner.ReferencedSOPInstanceUID = _FOREIGN + "study"
        mid = Dataset()
        mid.FrameOfReferenceUID = _FOREIGN + "for"
        mid.add_new((0x0008, 0x1140), "SQ", Sequence([inner]))
        ds.add_new((0x0040, 0x0275), "SQ", Sequence([mid]))
        return ds

    def test_remaps_at_every_depth_and_keeps_shared_values_consistent(self):
        ds = self._ds()
        self.r.remap_dataset_recursive(ds, self.trail)
        mid = ds[(0x0040, 0x0275)].value[0]
        inner = mid[(0x0008, 0x1140)].value[0]
        self.assertTrue(str(ds.SOPInstanceUID).startswith(_PYDICOM_ROOT))
        self.assertTrue(str(mid.FrameOfReferenceUID).startswith(_PYDICOM_ROOT))
        self.assertEqual(str(inner.ReferencedSOPInstanceUID), str(ds.StudyInstanceUID))
        self.assertEqual(str(ds.SOPClassUID), _STD + ".5.1.4.1.1.7")
        self.assertEqual(str(inner.ReferencedSOPClassUID), _STD + ".5.1.4.1.1.7")
        self.assertEqual(str(ds.PatientName), "Not^A^UID")

    def test_remap_rows_are_depth_first_in_tag_order(self):
        self.r.remap_dataset_recursive(self._ds(), self.trail)
        rows = self.trail.to_list()
        self.assertEqual(
            [(row["tag"], row["keyword"]) for row in rows],
            [("(0008,0018)", "SOPInstanceUID"),
             ("(0020,000D)", "StudyInstanceUID"),
             ("(0008,1155)", "ReferencedSOPInstanceUID"),   # depth 2, reached via (0040,0275)
             ("(0020,0052)", "FrameOfReferenceUID")],       # depth 1, visited after its own SQ child
        )
        self.assertTrue(all(row["action"] == constants.ACTION_REMAP and row["redacted"] is True for row in rows))
        self.assertTrue(all(list(row) == ["tag", "keyword", "action", "redacted"] for row in rows))

    def test_multi_valued_element_remapped_per_value_with_single_row(self):
        ds = Dataset()
        ds.add_new((0x0008, 0x1155), "UI", [_FOREIGN + "a", _STD + ".1.2", _FOREIGN + "a"])
        self.r.remap_dataset_recursive(ds, self.trail)
        values = [str(v) for v in ds[(0x0008, 0x1155)].value]
        self.assertEqual(values[0], self.r.remap_uid(_FOREIGN + "a"))
        self.assertEqual(values[1], _STD + ".1.2")
        self.assertEqual(values[0], values[2])
        self.assertEqual(len(self.trail), 1)

    def test_unchanged_elements_produce_no_rows(self):
        ds = Dataset()
        ds.SOPClassUID = _STD + ".5.1.4.1.1.7"                       # preserved keyword
        ds.add_new((0x0008, 0x010C), "UI", _STD + ".2.16.4")          # standard root, non-class keyword
        ds.add_new((0x0008, 0x1155), "UI", [_STD + ".1.2", _STD + ".1.2.1"])
        ds.add_new((0x0020, 0x000D), "UI", "")                        # empty
        ds.add_new((0x0020, 0x000E), "UI", None)                      # None value
        ds.PatientName = _FOREIGN + "looks.like.a.uid.but.is.PN"
        self.r.remap_dataset_recursive(ds, self.trail)
        self.assertEqual(len(self.trail), 0)
        self.assertEqual(str(ds.PatientName), _FOREIGN + "looks.like.a.uid.but.is.PN")

    def test_file_meta_is_not_walked(self):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPInstanceUID = _FOREIGN + "sop"
        ds.file_meta.ImplementationClassUID = _FOREIGN + "impl"
        self.r.remap_dataset_recursive(ds, self.trail)
        self.assertEqual(str(ds.file_meta.MediaStorageSOPInstanceUID), _FOREIGN + "sop")
        self.assertEqual(str(ds.file_meta.ImplementationClassUID), _FOREIGN + "impl")
        self.assertEqual(len(self.trail), 0)

    def test_fallback_entropy_is_threaded_to_every_element(self):
        # Only matters for blank values (which are skipped), so it must be a
        # no-op for the mapping of real UIDs: same output with and without it.
        ds1, ds2 = Dataset(), Dataset()
        ds1.SOPInstanceUID = ds2.SOPInstanceUID = _FOREIGN + "sop"
        self.r.remap_dataset_recursive(ds1, TagAuditTrail(), fallback_entropy="/p/x.dcm")
        self.r.remap_dataset_recursive(ds2, TagAuditTrail())
        self.assertEqual(str(ds1.SOPInstanceUID), str(ds2.SOPInstanceUID))


class TestEnsureSopInstanceUid(unittest.TestCase):

    def setUp(self):
        self.r = UidRemapper("s")

    def test_existing_uid_is_left_alone_and_file_meta_synced(self):
        ds = Dataset()
        ds.SOPInstanceUID = _FOREIGN + "already"
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPInstanceUID = _FOREIGN + "stale"
        self.r.ensure_sop_instance_uid(ds, original_sop_uid=_FOREIGN + "orig", fallback_entropy="/p/x.dcm")
        self.assertEqual(str(ds.SOPInstanceUID), _FOREIGN + "already")
        self.assertEqual(str(ds.file_meta.MediaStorageSOPInstanceUID), _FOREIGN + "already")

    def test_missing_uid_minted_from_original_then_from_entropy(self):
        ds = Dataset()
        self.r.ensure_sop_instance_uid(ds, original_sop_uid=_FOREIGN + "orig", fallback_entropy="/p/x.dcm")
        self.assertEqual(str(ds.SOPInstanceUID), self.r.remap_uid(_FOREIGN + "orig"))
        ds = Dataset()
        self.r.ensure_sop_instance_uid(ds, original_sop_uid="", fallback_entropy="/p/x.dcm")
        self.assertEqual(str(ds.SOPInstanceUID), self.r.remap_uid("/p/x.dcm"))

    def test_blank_uid_counts_as_missing(self):
        ds = Dataset()
        ds.SOPInstanceUID = "   "
        self.r.ensure_sop_instance_uid(ds, original_sop_uid="", fallback_entropy="/p/x.dcm")
        self.assertEqual(str(ds.SOPInstanceUID), self.r.remap_uid("/p/x.dcm"))

    def test_no_file_meta_and_no_sources_is_tolerated(self):
        ds = Dataset()
        self.r.ensure_sop_instance_uid(ds, original_sop_uid="", fallback_entropy="")
        self.assertTrue(pydicom.uid.UID(str(ds.SOPInstanceUID)).is_valid)
        self.assertFalse(hasattr(ds, "file_meta"))

    def test_file_meta_set_to_none_is_skipped(self):
        ds = Dataset()
        ds.SOPInstanceUID = _FOREIGN + "x"
        ds.file_meta = None
        self.r.ensure_sop_instance_uid(ds, original_sop_uid="", fallback_entropy="")
        self.assertIsNone(ds.file_meta)


if __name__ == "__main__":
    unittest.main()
