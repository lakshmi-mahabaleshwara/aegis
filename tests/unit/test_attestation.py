"""Tests for PS3.15 de-identification attestation stamping and
pii_mapping action validation.

Attestation: every scrubbed dataset must carry the machine-checkable
"this object was de-identified" markers — (0012,0062/0063/0064), plus
BurnedInAnnotation=NO when pixel redaction ran.

Validation: a typo'd action in pii_mapping must fail the run at
construction time, never silently leave PHI in place while the audit
reports it as redacted.
"""
import unittest

import numpy as np

from monai_aegis.transforms.metadata import ScrubDicomMetadata
from tests.unit.test_concurrent_audit import _make_dataset

_CONFIG = {
    'pii_mapping': {
        '(0010,0010)': 'DUMMY',   # PatientName
    },
    'tokenization': {'salt': 'test-salt'},
}


class TestAttestationStamping(unittest.TestCase):

    def setUp(self):
        self.transform = ScrubDicomMetadata(config=_CONFIG)

    def test_header_only_scrub_stamps_basic_profile(self):
        ds = _make_dataset(patient_name='Alice Anderson')
        scrubbed, tag_actions = self.transform.scrub(uri='a.dcm', dataset=ds)

        self.assertEqual(scrubbed.PatientIdentityRemoved, 'YES')
        method = scrubbed.DeidentificationMethod
        self.assertTrue(any('MONAI Aegis' in str(v) for v in method))
        codes = [item.CodeValue for item in scrubbed.DeidentificationMethodCodeSequence]
        self.assertIn('113100', codes)   # Basic Application Confidentiality Profile
        self.assertNotIn('113101', codes)  # no pixel redaction on this call
        # BurnedInAnnotation is a claim about pixels — never stamped
        # unless pixel redaction actually ran.
        self.assertNotIn('BurnedInAnnotation', scrubbed)

        attest_rows = [a for a in tag_actions if a['action'] == 'ATTEST']
        self.assertEqual(
            [a['keyword'] for a in attest_rows],
            ['PatientIdentityRemoved', 'DeidentificationMethod',
             'DeidentificationMethodCodeSequence'],
        )
        self.assertTrue(all(a['redacted'] is False for a in attest_rows))

    def test_pixel_redaction_adds_clean_pixel_attestation(self):
        ds = _make_dataset(patient_name='Alice Anderson')
        pixels = np.zeros((16, 16), dtype=np.uint8)
        scrubbed, tag_actions = self.transform.scrub(
            uri='a.dcm', pixel_data=pixels, dataset=ds)

        codes = [item.CodeValue for item in scrubbed.DeidentificationMethodCodeSequence]
        self.assertIn('113100', codes)
        self.assertIn('113101', codes)   # Clean Pixel Data Option
        self.assertEqual(scrubbed.BurnedInAnnotation, 'NO')
        for item in scrubbed.DeidentificationMethodCodeSequence:
            self.assertEqual(item.CodingSchemeDesignator, 'DCM')
        self.assertIn(
            'BurnedInAnnotation',
            [a['keyword'] for a in tag_actions if a['action'] == 'ATTEST'],
        )

    def test_attestation_survives_a_mapping_that_targets_it(self):
        # Even if someone configures an action on (0012,0062), the stamp
        # runs after the scrub pass and must win.
        config = {
            'pii_mapping': {'(0012,0062)': 'REMOVE'},
            'tokenization': {'salt': 'test-salt'},
        }
        transform = ScrubDicomMetadata(config=config)
        ds = _make_dataset(patient_name='Alice Anderson')
        ds.PatientIdentityRemoved = 'NO'
        scrubbed, _ = transform.scrub(uri='a.dcm', dataset=ds)
        self.assertEqual(scrubbed.PatientIdentityRemoved, 'YES')


class TestPiiActionValidation(unittest.TestCase):

    def test_unknown_action_fails_at_construction(self):
        config = {'pii_mapping': {'(0010,0010)': 'DELETE'}}  # typo for REMOVE
        with self.assertRaises(ValueError) as ctx:
            ScrubDicomMetadata(config=config)
        self.assertIn('DELETE', str(ctx.exception))

    def test_invalid_tag_fails_at_construction(self):
        config = {'pii_mapping': {'(0010)': 'REMOVE'}}
        with self.assertRaises(ValueError):
            ScrubDicomMetadata(config=config)

    def test_keep_action_retains_value_and_audits_it(self):
        config = {
            'pii_mapping': {
                '(0010,0010)': 'DUMMY',
                '(0010,0020)': 'KEEP',
            },
            'tokenization': {'salt': 'test-salt'},
        }
        transform = ScrubDicomMetadata(config=config)
        ds = _make_dataset(patient_name='Alice Anderson', patient_id='MRN-1')
        scrubbed, tag_actions = transform.scrub(uri='a.dcm', dataset=ds)

        self.assertEqual(scrubbed.PatientID, 'MRN-1')  # retained verbatim
        kept = [a for a in tag_actions if a['action'] == 'KEEP']
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['keyword'], 'PatientID')
        self.assertFalse(kept[0]['redacted'])

    def test_actions_are_case_insensitive(self):
        config = {'pii_mapping': {'(0010,0010)': 'remove'}}
        transform = ScrubDicomMetadata(config=config)
        ds = _make_dataset(patient_name='Alice Anderson')
        scrubbed, _ = transform.scrub(uri='a.dcm', dataset=ds)
        self.assertNotIn('PatientName', scrubbed)


if __name__ == '__main__':
    unittest.main()
