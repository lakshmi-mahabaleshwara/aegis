
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

from monai_aegis.transforms.metadata import ScrubDicomMetadata
from monai_aegis.transforms.utility import AegisIdentityManager

class TestScrubDicomMetadata(unittest.TestCase):

    def setUp(self):
        # Create a dummy dataset
        self.ds = Dataset()
        self.ds.PatientName = "Test^Patient"
        self.ds.PatientID = "123456"
        self.ds.StudyDate = "20230101"
        self.ds.AccessionNumber = "ACC001"
        self.ds.Modality = "MR"
        
        # Window Center/Width (Visual Integrity)
        self.ds.WindowCenter = "400"
        self.ds.WindowWidth = "2000"
        
        # Add private tag
        block = self.ds.private_block(0x0009, "Private Creator", 0x10)
        block.add_new(0x01, "LO", "Private Secret")
        
        # Config
        self.config = {
            'pii_mapping': {
                '(0010, 0010)': 'REMOVE', # PatientName
                '(0010, 0020)': 'DUMMY',  # PatientID
                '(0008, 0020)': 'ZERO',   # StudyDate
                '(0008, 0050)': 'KEEP'    # AccessionNumber
            }
        }

        # Create the array transform (no output_dir for in-memory testing)
        self.transform = ScrubDicomMetadata(config=self.config)

    def test_scrub_metadata_actions(self):
        # Need to test internal logic directly since ScrubDicomMetadata reads from file
        # Test through the underlying logic
        pii_mapping = self.config.get('pii_mapping', {})
        identity_manager = AegisIdentityManager()

        for tag_str, action in pii_mapping.items():
            clean_tag_str = tag_str.strip('() ').replace(' ', '')
            parts = clean_tag_str.split(',')
            tag = pydicom.tag.Tag(int(parts[0], 16), int(parts[1], 16))

            if tag in self.ds:
                action = action.upper()
                if action == 'REMOVE':
                    del self.ds[tag]
                elif action == 'ZERO':
                    vr = self.ds[tag].VR
                    self.ds[tag].value = b'' if vr in ['OB', 'OW', 'UN'] else ''
                elif action == 'DUMMY':
                    token = identity_manager.get_token(str(self.ds[tag].value))
                    self.ds[tag].value = token

        self.ds.remove_private_tags()

        # Check REMOVE
        self.assertNotIn('PatientName', self.ds)
        
        # Check DUMMY (Tokenization)
        self.assertTrue(str(self.ds.PatientID).startswith('TOKEN_'))
        
        # Check ZERO
        self.assertEqual(self.ds.StudyDate, '')
        
        # Check KEEP
        self.assertEqual(self.ds.AccessionNumber, 'ACC001')

    def test_visual_integrity_preservation(self):
        # Window Center/Width should be preserved if not in PII mapping
        pii_mapping = self.config.get('pii_mapping', {})
        identity_manager = AegisIdentityManager()
        
        for tag_str, action in pii_mapping.items():
            clean_tag_str = tag_str.strip('() ').replace(' ', '')
            parts = clean_tag_str.split(',')
            tag = pydicom.tag.Tag(int(parts[0], 16), int(parts[1], 16))
            if tag in self.ds:
                action = action.upper()
                if action == 'REMOVE':
                    del self.ds[tag]

        self.assertEqual(self.ds.WindowCenter, "400")
        self.assertEqual(self.ds.WindowWidth, "2000")

    def test_private_tag_removal(self):
        self.ds.remove_private_tags()
        
        found = False
        for elem in self.ds:
            if elem.tag.group == 0x0009:
                found = True
                break
        self.assertFalse(found, "Private tags should be removed")

    def test_missing_tags_handled_gracefully(self):
        self.config['pii_mapping']['(0010, 0030)'] = 'REMOVE'  # PatientBirthDate (missing)
        # Should not raise
        pii_mapping = self.config['pii_mapping']
        for tag_str, action in pii_mapping.items():
            clean_tag_str = tag_str.strip('() ').replace(' ', '')
            parts = clean_tag_str.split(',')
            tag = pydicom.tag.Tag(int(parts[0], 16), int(parts[1], 16))
            if tag in self.ds and action.upper() == 'REMOVE':
                del self.ds[tag]
        self.assertNotIn('PatientBirthDate', self.ds)

    def test_recursive_sequence_scrubbing(self):
        nested = Dataset()
        nested.PatientName = "Nested^Patient"
        nested.PatientID = "NESTED123"
        nested.StudyDate = "20240101"
        nested.AccessionNumber = "ACC-NESTED"
        block = nested.private_block(0x0009, "Nested Private Creator", 0x10)
        block.add_new(0x01, "LO", "Nested Secret")
        self.ds.add_new((0x0040, 0x0275), 'SQ', Sequence([nested]))

        scrubbed = self.transform(uri="unused.dcm", dataset=self.ds)
        nested_scrubbed = scrubbed[(0x0040, 0x0275)].value[0]

        self.assertNotIn('PatientName', nested_scrubbed)
        self.assertTrue(str(nested_scrubbed.PatientID).startswith('TOKEN_'))
        self.assertEqual(nested_scrubbed.StudyDate, '')
        self.assertEqual(nested_scrubbed.AccessionNumber, 'ACC-NESTED')

        for elem in nested_scrubbed:
            self.assertFalse(elem.tag.is_private, "Nested private tags should be removed")


# Roots used to tell apart "original, institution-issued" UIDs from the ones
# the scrubber generates. Foreign root = a value that must be replaced;
# pydicom root = where remap_uid lands; standard root = preserved verbatim.
_FOREIGN_ROOT = "1.2.840.113619.2.55.3."      # e.g. a GE-issued UID root
_PYDICOM_ROOT = "1.2.826.0.1.3680043.8.498."  # generate_uid() default root
_STD_ROOT = "1.2.840.10008"                   # DICOM standard (class) root


class TestUidGraphRemap(unittest.TestCase):
    """The scrubber replaces every patient-linkable UID (PS3.15 action U)."""

    def setUp(self):
        # These tests drive the salt through config; AEGIS_TOKEN_SALT would
        # override it (AegisIdentityManager gives env priority), so neutralize
        # any ambient value and restore it afterwards. Keeps the suite immune
        # to a leaked env salt regardless of test order.
        self._saved_salt = os.environ.pop("AEGIS_TOKEN_SALT", None)

    def tearDown(self):
        if self._saved_salt is not None:
            os.environ["AEGIS_TOKEN_SALT"] = self._saved_salt
        else:
            os.environ.pop("AEGIS_TOKEN_SALT", None)

    def _make_ds(self, sop="1", study="100", series="200", frame="300", salt="s1"):
        ds = Dataset()
        ds.PatientName = "Test^Patient"
        ds.SOPClassUID = _STD_ROOT + ".5.1.4.1.1.7"
        ds.SOPInstanceUID = _FOREIGN_ROOT + sop
        ds.StudyInstanceUID = _FOREIGN_ROOT + study
        ds.SeriesInstanceUID = _FOREIGN_ROOT + series
        ds.FrameOfReferenceUID = _FOREIGN_ROOT + frame
        ref = Dataset()
        ref.ReferencedSOPClassUID = _STD_ROOT + ".5.1.4.1.1.7"
        # Points at the study UID value on purpose — used to prove that a UID
        # reused across the object is rewritten to the same replacement.
        ref.ReferencedSOPInstanceUID = _FOREIGN_ROOT + study
        ds.add_new((0x0008, 0x1140), "SQ", Sequence([ref]))  # ReferencedImageSequence
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = ds.SOPClassUID
        fm.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        fm.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta = fm
        return ds

    def _scrub(self, ds, salt="s1"):
        cfg = {"pii_mapping": {}, "tokenization": {"salt": salt}}
        return ScrubDicomMetadata(config=cfg).scrub("f.dcm", dataset=ds)

    def test_all_patient_linkable_uids_are_remapped(self):
        out, _ = self._scrub(self._make_ds())
        for keyword in ("SOPInstanceUID", "StudyInstanceUID",
                        "SeriesInstanceUID", "FrameOfReferenceUID"):
            value = str(getattr(out, keyword))
            self.assertTrue(value.startswith(_PYDICOM_ROOT),
                            f"{keyword} not remapped: {value}")
            self.assertFalse(value.startswith(_FOREIGN_ROOT),
                             f"{keyword} kept its original value")

    def test_class_and_standard_root_uids_are_preserved(self):
        out, _ = self._scrub(self._make_ds())
        self.assertEqual(str(out.SOPClassUID), _STD_ROOT + ".5.1.4.1.1.7")
        self.assertEqual(str(out.file_meta.MediaStorageSOPClassUID),
                         _STD_ROOT + ".5.1.4.1.1.7")
        ref = out[(0x0008, 0x1140)].value[0]
        self.assertEqual(str(ref.ReferencedSOPClassUID), _STD_ROOT + ".5.1.4.1.1.7")

    def test_file_meta_stays_consistent(self):
        out, _ = self._scrub(self._make_ds())
        self.assertEqual(str(out.file_meta.MediaStorageSOPInstanceUID),
                         str(out.SOPInstanceUID))

    def test_shared_uid_rewritten_consistently_preserving_references(self):
        # ReferencedSOPInstanceUID held the same value as StudyInstanceUID in
        # the input; both must land on the identical replacement so the
        # cross-reference still resolves after de-identification.
        out, _ = self._scrub(self._make_ds())
        ref = out[(0x0008, 0x1140)].value[0]
        self.assertEqual(str(ref.ReferencedSOPInstanceUID),
                         str(out.StudyInstanceUID))

    def test_remap_is_deterministic_across_instances(self):
        out1, _ = self._scrub(self._make_ds(), salt="same")
        out2, _ = self._scrub(self._make_ds(), salt="same")
        self.assertEqual(str(out1.StudyInstanceUID), str(out2.StudyInstanceUID))
        self.assertEqual(str(out1.SeriesInstanceUID), str(out2.SeriesInstanceUID))
        self.assertEqual(str(out1.SOPInstanceUID), str(out2.SOPInstanceUID))

    def test_salt_changes_the_mapping(self):
        out1, _ = self._scrub(self._make_ds(), salt="salt-a")
        out2, _ = self._scrub(self._make_ds(), salt="salt-b")
        self.assertNotEqual(str(out1.StudyInstanceUID), str(out2.StudyInstanceUID))

    def test_two_slices_share_series_uid_but_differ_by_instance(self):
        # Same series, different instances → same remapped SeriesInstanceUID,
        # distinct remapped SOPInstanceUIDs. This is the series-linkage contract.
        out1, _ = self._scrub(self._make_ds(sop="1"), salt="k")
        out2, _ = self._scrub(self._make_ds(sop="2"), salt="k")
        self.assertEqual(str(out1.SeriesInstanceUID), str(out2.SeriesInstanceUID))
        self.assertNotEqual(str(out1.SOPInstanceUID), str(out2.SOPInstanceUID))

    def test_remap_actions_are_audited(self):
        _, actions = self._scrub(self._make_ds())
        remapped = {a["keyword"] for a in actions if a["action"] == "REMAP"}
        self.assertEqual(
            remapped,
            {"SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
             "FrameOfReferenceUID", "ReferencedSOPInstanceUID"},
        )
        self.assertTrue(all(a["redacted"] for a in actions if a["action"] == "REMAP"))

    def test_missing_sop_uid_is_minted(self):
        ds = self._make_ds()
        del ds.SOPInstanceUID
        del ds.file_meta.MediaStorageSOPInstanceUID
        out, _ = self._scrub(ds)
        self.assertTrue(str(out.SOPInstanceUID).startswith(_PYDICOM_ROOT))
        self.assertEqual(str(out.file_meta.MediaStorageSOPInstanceUID),
                         str(out.SOPInstanceUID))


if __name__ == '__main__':
    unittest.main()
