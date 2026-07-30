"""Unit tests for ground-truth reporting (monai_aegis.reporting)."""
import csv
import os

import pydicom

from monai_aegis import reporting


def _ds(sop_uid):
    ds = pydicom.Dataset()
    ds.SOPInstanceUID = sop_uid
    return ds


def test_is_enabled_default_true():
    assert reporting.is_enabled({}) is True
    assert reporting.is_enabled({"reporting": {}}) is True
    assert reporting.is_enabled({"reporting": {"save_ground_truth": False}}) is False


def test_extract_single_file_pixel_and_tag_rows():
    data = {
        "image_meta_dict": {"filename_or_obj": "/in/ED01_z07.dcm"},
        "image_dicom_dataset": _ds("1.2.ORIG"),
        "image_scrubbed_ds": _ds("1.2.DEID"),
        "image_redaction_stats": {
            "detections": [
                {"bbox": [5, 5, 183, 15], "ocr_text": "Robinson, Brandy S.",
                 "confidence": 0.97, "decision": "redacted"},
                {"bbox": [5, 495, 356, 12], "ocr_text": "SYNTHETIC-DEID",
                 "confidence": 0.88, "decision": "safelisted"},
            ],
        },
        "image_tag_actions": [
            {"tag": "(0010,0010)", "keyword": "PatientName",
             "action": "DUMMY", "redacted": True},
        ],
    }

    pixel_rows, tag_rows = reporting.extract_records(data)

    assert len(pixel_rows) == 2
    first = pixel_rows[0]
    assert first["original_sop_uid"] == "1.2.ORIG"
    assert first["deid_sop_uid"] == "1.2.DEID"
    assert (first["bbox_x"], first["bbox_y"], first["bbox_w"], first["bbox_h"]) == (5, 5, 183, 15)
    assert first["decision"] == "redacted"
    assert {r["decision"] for r in pixel_rows} == {"redacted", "safelisted"}

    assert len(tag_rows) == 1
    assert tag_rows[0]["tag"] == "(0010,0010)"
    assert tag_rows[0]["original_sop_uid"] == "1.2.ORIG"


def test_extract_series_maps_frame_index_to_slice_sop():
    data = {
        "image_meta_dict": {"filename_or_obj": "/in/series"},
        "image_dicom_datasets": [_ds("1.2.O0"), _ds("1.2.O1")],
        "image_scrubbed_datasets": [_ds("1.2.D0"), _ds("1.2.D1")],
        "image_redaction_stats": {
            "detections": [
                {"bbox": [1, 1, 2, 2], "ocr_text": "X", "confidence": 0.9,
                 "decision": "redacted", "frame_index": 1},
            ],
        },
        "image_tag_actions_per_slice": [
            [{"tag": "(0010,0020)", "keyword": "PatientID", "action": "DUMMY", "redacted": True}],
            [{"tag": "(0010,0020)", "keyword": "PatientID", "action": "DUMMY", "redacted": True}],
        ],
    }

    pixel_rows, tag_rows = reporting.extract_records(data)

    assert pixel_rows[0]["original_sop_uid"] == "1.2.O1"
    assert pixel_rows[0]["deid_sop_uid"] == "1.2.D1"
    assert pixel_rows[0]["frame_index"] == 1
    assert [r["original_sop_uid"] for r in tag_rows] == ["1.2.O0", "1.2.O1"]


def test_extract_series_attributes_source_path_per_slice():
    # Multi-file series: each row's source_path must point at the slice it
    # came from, not always the first slice.
    data = {
        "image_meta_dict": {
            "filename_or_obj": "/in/series/slice_0.dcm",
            "slice_uris": ["/in/series/slice_0.dcm", "/in/series/slice_1.dcm"],
        },
        "image_dicom_datasets": [_ds("1.2.O0"), _ds("1.2.O1")],
        "image_scrubbed_datasets": [_ds("1.2.D0"), _ds("1.2.D1")],
        "image_redaction_stats": {
            "detections": [
                {"bbox": [1, 1, 2, 2], "ocr_text": "X", "confidence": 0.9,
                 "decision": "redacted", "frame_index": 1},
            ],
        },
        "image_tag_actions_per_slice": [
            [{"tag": "(0010,0020)", "keyword": "PatientID", "action": "DUMMY", "redacted": True}],
            [{"tag": "(0010,0020)", "keyword": "PatientID", "action": "DUMMY", "redacted": True}],
        ],
    }

    pixel_rows, tag_rows = reporting.extract_records(data)

    # Pixel detection on frame 1 attributes to slice_1, keyed to slice_1's UIDs.
    assert pixel_rows[0]["source_path"] == "/in/series/slice_1.dcm"
    assert pixel_rows[0]["original_sop_uid"] == "1.2.O1"
    # Tag rows attribute to their own slice, index-aligned.
    assert [r["source_path"] for r in tag_rows] == [
        "/in/series/slice_0.dcm", "/in/series/slice_1.dcm",
    ]


def test_extract_series_without_slice_uris_falls_back_to_single_path():
    # Back-compat: absent slice_uris (e.g. multi-frame single file), every row
    # resolves to the one real source path.
    data = {
        "image_meta_dict": {"filename_or_obj": "/in/multiframe.dcm"},
        "image_dicom_datasets": [_ds("1.2.O0"), _ds("1.2.O1")],
        "image_scrubbed_datasets": [_ds("1.2.D0"), _ds("1.2.D1")],
        "image_redaction_stats": {
            "detections": [
                {"bbox": [1, 1, 2, 2], "ocr_text": "X", "confidence": 0.9,
                 "decision": "redacted", "frame_index": 1},
            ],
        },
        "image_tag_actions_per_slice": [
            [{"tag": "(0010,0020)", "keyword": "PatientID", "action": "DUMMY", "redacted": True}],
            [{"tag": "(0010,0020)", "keyword": "PatientID", "action": "DUMMY", "redacted": True}],
        ],
    }

    pixel_rows, tag_rows = reporting.extract_records(data)

    assert pixel_rows[0]["source_path"] == "/in/multiframe.dcm"
    assert {r["source_path"] for r in tag_rows} == {"/in/multiframe.dcm"}


def test_extract_empty_dict_is_safe():
    pixel_rows, tag_rows = reporting.extract_records({})
    assert pixel_rows == []
    assert tag_rows == []


def test_write_reports_creates_both_files(tmp_path):
    pixel_rows = [{
        "source_path": "/in/a.dcm", "original_sop_uid": "O", "deid_sop_uid": "D",
        "frame_index": "", "bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4,
        "text_token": "TOKEN_abc", "text_len": 2,
        "confidence": 0.5, "decision": "redacted",
    }]
    paths = reporting.write_reports(pixel_rows, [], str(tmp_path))

    assert len(paths) == 2
    pixel_file = os.path.join(str(tmp_path), reporting.PIXEL_DETECTIONS_FILE)
    tag_file = os.path.join(str(tmp_path), reporting.TAG_ACTIONS_FILE)
    assert os.path.exists(pixel_file) and os.path.exists(tag_file)

    with open(pixel_file, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["decision"] == "redacted"
    assert rows[0]["text_token"] == "TOKEN_abc"
    assert "ocr_text" not in rows[0]


def test_write_reports_skips_tag_file_when_not_applicable(tmp_path):
    # tag_rows=None → image-pipeline mode: no DICOM tags exist, so the
    # tag-actions CSV must not be created at all.
    paths = reporting.write_reports([], None, str(tmp_path))

    assert len(paths) == 1
    assert os.path.exists(os.path.join(str(tmp_path), reporting.PIXEL_DETECTIONS_FILE))
    assert not os.path.exists(os.path.join(str(tmp_path), reporting.TAG_ACTIONS_FILE))


def test_sanitize_pixel_rows_tokenizes_text():
    from monai_aegis.transforms.utility import AegisIdentityManager
    tokenizer = AegisIdentityManager(salt="test-salt")
    rows = [
        {"ocr_text": "Robinson, Brandy S.", "decision": "redacted"},
        {"ocr_text": "Robinson, Brandy S.", "decision": "redacted"},
        {"ocr_text": "", "decision": "safelisted"},
    ]

    safe = reporting.sanitize_pixel_rows(rows, tokenizer)

    assert all("ocr_text" not in r for r in safe)
    assert safe[0]["text_token"].startswith("TOKEN_")
    assert safe[0]["text_len"] == len("Robinson, Brandy S.")
    # Same text → same token, so repeats remain correlatable without PHI.
    assert safe[0]["text_token"] == safe[1]["text_token"]
    assert safe[2]["text_token"] == ""
    # And the join property: tokenizing the ground-truth side matches.
    assert safe[0]["text_token"] == tokenizer.get_token("Robinson, Brandy S.")


def _phi_config(**reporting_overrides):
    rep = {"save_ground_truth": True}
    rep.update(reporting_overrides)
    return {"reporting": rep, "tokenization": {"salt": "test-salt"}}


_DETECTION_DATA = {
    "image_meta_dict": {"filename_or_obj": "/in/scan.dcm"},
    "image_redaction_stats": {
        "detections": [
            {"bbox": [5, 5, 183, 15], "ocr_text": "Robinson, Brandy S.",
             "confidence": 0.97, "decision": "redacted"},
        ],
    },
}


def test_accumulator_default_csv_is_phi_free(tmp_path):
    report = reporting.GroundTruthAccumulator(_phi_config())
    report.collect(_DETECTION_DATA)
    report.flush(str(tmp_path))

    with open(os.path.join(str(tmp_path), reporting.PIXEL_DETECTIONS_FILE), newline="") as f:
        content = f.read()
    assert "Robinson" not in content
    rows = list(csv.DictReader(content.splitlines()))
    assert rows[0]["text_token"].startswith("TOKEN_")
    assert rows[0]["text_len"] == str(len("Robinson, Brandy S."))
    assert not os.path.exists(
        os.path.join(str(tmp_path), reporting.PIXEL_DETECTIONS_PHI_FILE))


def test_accumulator_phi_text_requires_phi_dir():
    import pytest
    with pytest.raises(ValueError, match="phi_report_dir"):
        reporting.GroundTruthAccumulator(_phi_config(include_phi_text=True))


def test_accumulator_phi_dir_must_differ_from_output(tmp_path):
    import pytest
    report = reporting.GroundTruthAccumulator(
        _phi_config(include_phi_text=True, phi_report_dir=str(tmp_path)))
    report.collect(_DETECTION_DATA)
    with pytest.raises(ValueError, match="must differ"):
        report.flush(str(tmp_path))


def test_accumulator_opt_in_writes_verbatim_phi_file(tmp_path):
    out_dir = tmp_path / "deid_output"
    phi_dir = tmp_path / "phi_quarantine"
    report = reporting.GroundTruthAccumulator(
        _phi_config(include_phi_text=True, phi_report_dir=str(phi_dir)))
    report.collect(_DETECTION_DATA)
    report.flush(str(out_dir))

    # Default file in the output dir stays tokenized...
    with open(os.path.join(str(out_dir), reporting.PIXEL_DETECTIONS_FILE)) as f:
        assert "Robinson" not in f.read()
    # ...while the verbatim report lands only in the quarantine dir.
    phi_file = os.path.join(str(phi_dir), reporting.PIXEL_DETECTIONS_PHI_FILE)
    with open(phi_file, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["ocr_text"] == "Robinson, Brandy S."
    assert not os.path.exists(
        os.path.join(str(out_dir), reporting.PIXEL_DETECTIONS_PHI_FILE))


def test_accumulator_image_mode_omits_tag_actions(tmp_path):
    report = reporting.GroundTruthAccumulator(
        {"reporting": {"save_ground_truth": True}}, include_tag_actions=False
    )
    report.collect({
        "image_meta_dict": {"filename_or_obj": "/in/scan.png"},
        "image_redaction_stats": {
            "detections": [
                {"bbox": [5, 5, 183, 15], "ocr_text": "Robinson, Brandy S.",
                 "confidence": 0.97, "decision": "redacted"},
            ],
        },
    })
    report.flush(str(tmp_path))

    assert os.path.exists(os.path.join(str(tmp_path), reporting.PIXEL_DETECTIONS_FILE))
    assert not os.path.exists(os.path.join(str(tmp_path), reporting.TAG_ACTIONS_FILE))
