"""Line context and redaction padding for split OCR lines.

OCR returns "Robinson, Brandy S." as "Robinson," + "Brandy" and misses
"S." entirely; it returns "08/01/2021 23:16" as two fragments. Judged one
by one, "Brandy" and "23:16" were kept and "S." was never seen. Line
context re-judges a split line as a whole; padding covers tokens OCR drops.
"""
import unittest
from unittest.mock import MagicMock

import numpy as np

from monai_aegis.transforms.ner_classifier import PHIClassifier
from monai_aegis.transforms.pixel import (
    _apply_line_context, apply_redaction, detect_text, group_text_lines, pad_redaction_boxes,
)


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


NER_CFG = {
    'phi_labels': ['PATIENT', 'HCW', 'HOSPITAL', 'ID', 'DATE', 'PHONE'],
    'clinical_allowlist': ['lung'],
    'clinical_patterns': [r'^(F\s*H?\d|D\s+\d|G\s+\d)', r'\d+\.?\d*\s*(cm|mm|MHz)$'],
    'phi_heuristic_patterns': [r'(?i)(hospital|clinic|centre)', r'\b\d{2}/\d{2}/\d{4}\b'],
    'age_patterns': [r'(?i)\b\d{1,3}\s*Y\b'],
}


def fake_ner(entities_by_text):
    """NER stand-in: returns PATIENT spans for listed names, wherever they occur."""
    def run(text):
        out = []
        for needle, label in entities_by_text.items():
            start = text.find(needle)
            if start >= 0:
                out.append({'entity_group': label, 'score': 0.99, 'word': needle,
                            'start': start, 'end': start + len(needle)})
        return out
    return run


def classifier(entities=None, side_effect=None):
    clf = PHIClassifier({'ner': dict(NER_CFG)})
    pipe = MagicMock()
    pipe.side_effect = side_effect or fake_ner(entities or {})
    clf._thread_local.pipeline = pipe
    return clf


class TestGroupTextLines(unittest.TestCase):

    def test_adjacent_fragments_on_one_row_form_a_line(self):
        b = [box(110, 7, 160, 25), box(1, 0, 92, 26)]           # out of order on purpose
        self.assertEqual(group_text_lines(b), [[1, 0]])

    def test_far_column_and_next_row_stay_separate(self):
        b = [box(1, 0, 92, 26), box(101, 7, 161, 25),           # name line
             box(431, 7, 473, 23),                               # right column, same row
             box(3, 30, 113, 48)]                                # next row
        lines = sorted(group_text_lines(b))
        self.assertEqual(lines, [[0, 1], [2], [3]])


class TestClassifyLine(unittest.TestCase):

    def test_name_span_covers_both_fragments(self):
        clf = classifier({'Robinson, Brandy': 'PATIENT'})
        self.assertEqual(clf.classify_line(["Robinson,", "Brandy"]), [True, True])

    def test_only_fragments_inside_the_span_are_flagged(self):
        clf = classifier({'Robinson': 'PATIENT'})
        self.assertEqual(clf.classify_line(["Robinson,", "ABDOMEN"]), [True, False])

    def test_heuristic_date_span(self):
        clf = classifier()
        self.assertEqual(clf.classify_line(["08/01/2021", "23:16"]), [True, False])
        clf = classifier({'08/01/2021 23:16': 'DATE'})
        self.assertEqual(clf.classify_line(["08/01/2021", "23:16"]), [True, True])

    def test_clinical_fragment_never_flagged_by_context(self):
        # NER (wrongly) spans the clinical fragment too; it must still be kept.
        clf = classifier({'Robinson LUNG': 'PATIENT', 'Robinson 3.5 MHz': 'PATIENT'})
        self.assertEqual(clf.classify_line(["Robinson", "LUNG"]), [True, False])
        self.assertEqual(clf.classify_line(["Robinson", "3.5 MHz"]), [True, False])

    def test_non_phi_entities_ignored(self):
        clf = classifier({'SYNTHETIC': 'VENDOR'})
        self.assertEqual(clf.classify_line(["Manufacturer:", "SYNTHETIC"]), [False, False])

    def test_ner_error_fails_closed_except_clinical(self):
        clf = classifier(side_effect=RuntimeError("boom"))
        self.assertEqual(clf.classify_line(["Brandy", "S.", "LUNG"]), [True, True, False])

    def test_offsets_are_relative_to_the_line_not_the_context_prefix(self):
        # Short lines are wrapped as "The report mentions <line>." before NER.
        clf = classifier({'Brandy': 'PATIENT'})
        self.assertEqual(clf.classify_line(["Doe", "Brandy"]), [False, True])
        clf._thread_local.pipeline.assert_called_with("The report mentions Doe Brandy.")


class TestApplyLineContext(unittest.TestCase):

    RESULTS = [(box(1, 0, 92, 26), "Robinson,", 1.0), (box(101, 7, 161, 25), "Brandy", 1.0),
               (box(431, 7, 473, 23), "LUNG", 1.0)]

    def test_adds_redaction_never_removes(self):
        clf = classifier({'Robinson, Brandy': 'PATIENT'})
        flags = _apply_line_context(self.RESULTS, [True, False, False], clf, {'ner': {}})
        self.assertEqual(flags, [True, True, False])
        clf = classifier({})   # line context finds nothing: fragment flags stand
        flags = _apply_line_context(self.RESULTS, [True, False, False], clf, {'ner': {}})
        self.assertEqual(flags, [True, False, False])

    def test_can_be_switched_off(self):
        clf = classifier({'Robinson, Brandy': 'PATIENT'})
        flags = _apply_line_context(self.RESULTS, [True, False, False], clf, {'ner': {'line_context': False}})
        self.assertEqual(flags, [True, False, False])
        clf._thread_local.pipeline.assert_not_called()


def text_image():
    """Black image, white "Brandy" (x 101-160) and an undetected "S." (x 169-183)."""
    img = np.zeros((40, 300, 3), dtype=np.uint8)
    for x in range(103, 159, 6):
        img[9:23, x:x + 3] = 250          # strokes inside the OCR box
    img[10:24, 169:172] = img[10:24, 175:177] = 250
    img[20:24, 181:184] = 250             # "S."
    return img


class TestPadRedactionBoxes(unittest.TestCase):

    BRANDY = box(101, 7, 161, 25)          # h = 18 -> allowance 27 px a side

    def test_without_image_uses_full_allowance(self):
        (out,) = pad_redaction_boxes([box(100, 10, 160, 28)], [], 1.5)
        self.assertEqual(out, box(73, 10, 187, 28))

    def test_stops_at_kept_text_on_the_same_line_only(self):
        keep = [box(170, 12, 200, 26),    # same line, right
                box(60, 40, 90, 58)]      # next line, left: ignored
        (out,) = pad_redaction_boxes([box(100, 10, 160, 28)], keep, 1.5)
        self.assertEqual(out, box(73, 10, 170, 28))

    def test_grows_over_adjacent_ink_and_stops_after_it(self):
        (out,) = pad_redaction_boxes([self.BRANDY], [], 1.5, image=text_image())
        self.assertEqual(out, box(101, 7, 184 + 2, 25))   # last ink col 183, +2 margin; left: no ink

    def test_does_not_grow_into_other_content(self):
        img = text_image()
        img[:, 162:200] = 0
        img[5:30, 165:200] = 100           # grey image content (e.g. an ultrasound fan)
        (out,) = pad_redaction_boxes([self.BRANDY], [], 1.5, image=img)
        self.assertEqual(out, self.BRANDY)

    def test_bright_speckle_in_image_content_is_not_text(self):
        # Ultrasound speckle can hit the text colour; a column that is mostly
        # neither text nor background colour is image content and stops growth.
        img = text_image()
        img[:, 162:200] = 0
        rng = np.random.default_rng(0)
        img[5:30, 162:200] = rng.integers(30, 120, size=(25, 38, 1), dtype=np.uint8)
        img[[9, 14, 20], 163:200:3] = 252
        (out,) = pad_redaction_boxes([self.BRANDY], [], 1.5, image=img)
        self.assertEqual(out, self.BRANDY)

    def test_stops_at_a_gap_wider_than_a_word_space(self):
        img = text_image()
        img[:, 162:300] = 0
        img[10:24, 180:183] = 250          # 19 blank columns, then ink: beyond 0.8 x 18
        (out,) = pad_redaction_boxes([self.BRANDY], [], 1.5, image=img)
        self.assertEqual(out, self.BRANDY)

    def test_never_grows_into_kept_text(self):
        # Kept box starts at x 175: ink up to 171 is covered (+2 margin), 175 never is.
        (out,) = pad_redaction_boxes([self.BRANDY], [box(175, 8, 200, 24)], 1.5, image=text_image())
        self.assertEqual(out, box(101, 7, 174, 25))
        (out,) = pad_redaction_boxes([self.BRANDY], [box(172, 8, 200, 24)], 1.5, image=text_image())
        self.assertEqual(out, box(101, 7, 172, 25))

    def test_zero_ratio_is_a_no_op(self):
        b = [box(100, 10, 160, 28)]
        self.assertEqual(pad_redaction_boxes(b, [], 0, image=text_image()), b)


class TestDetectTextSplitName(unittest.TestCase):
    """End to end through detect_text: split name, split date, dropped initial."""

    def test_whole_name_and_initial_removed_clinical_label_and_image_kept(self):
        img = np.zeros((80, 500, 3), dtype=np.uint8)
        for x0, x1, y0, y1 in [(1, 92, 7, 25), (101, 161, 7, 25),       # "Robinson," "Brandy"
                               (5, 103, 43, 61), (109, 161, 43, 61),     # "08/01/2021" "23:16"
                               (431, 473, 7, 23)]:                       # "LUNG"
            img[y0 + 2:y1 - 2, x0 + 2:x1 - 2:4] = 250                    # text strokes
        img[10:24, 169:184:3] = 250                       # "S." - OCR does not detect it
        img[40:70, 166:220] = 90                          # image content right of "23:16"
        reader = MagicMock()
        reader.readtext.return_value = [
            (box(1, 7, 92, 25), "Robinson,", 1.0), (box(101, 7, 161, 25), "Brandy", 1.0),
            (box(5, 43, 103, 61), "08/01/2021", 0.95), (box(109, 43, 161, 61), "23:16", 0.99),
            (box(431, 7, 473, 23), "LUNG", 1.0),
        ]
        clf = classifier({'Robinson': 'PATIENT', 'Robinson, Brandy': 'PATIENT',
                          '08/01/2021 23:16': 'DATE'})
        cfg = {'ocr': {'redaction_padding': 1.5}, 'ner': {}}
        bboxes, stats = detect_text(img, reader, cfg, ner_classifier=clf)
        out = apply_redaction(img, bboxes)

        self.assertEqual(stats['redacted_count'], 4)
        self.assertEqual(stats['safelisted_count'], 1)
        self.assertFalse(out[:, :166].any(), "name, initial, date or time still visible")
        np.testing.assert_array_equal(out[40:70, 166:220], img[40:70, 166:220], "image content blanked")
        np.testing.assert_array_equal(out[7:23, 431:473], img[7:23, 431:473], "clinical label blanked")
        # Report rows keep the OCR box, not the padded one.
        self.assertIn([101, 7, 60, 18], [d['bbox'] for d in stats['detections']])


if __name__ == "__main__":
    unittest.main()
