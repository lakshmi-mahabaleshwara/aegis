"""Unit tests for the declarative verification engine (monai_aegis/verify.py).

Covers the check vocabulary (positive and negative paths for every type),
checklist load-time validation (fail-fast on unknown types, bad params,
bad tags, bad patterns), run-level report checks, status derivation,
PHI-freedom of findings, and report schema conformance. Uses hand-built
pydicom datasets and CSVs — no models, no pipeline.
"""

import csv
import json
import sys
from pathlib import Path

import pydicom
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monai_aegis import verify
from monai_aegis.api import InputError

PHI_NAME = "Doe^John"
PHI_ID = "MRN-8675309"


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _base_dataset(**overrides):
    ds = pydicom.Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = "1.2.826.0.1.3680043.8.498.77"
    ds.Modality = "OT"
    for key, value in overrides.items():
        setattr(ds, key, value)
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.8.498.9999"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta = file_meta
    ds.preamble = b"\0" * 128
    return ds


def _clean_dataset():
    """A dataset shaped like real Aegis output — passes the default checklist."""
    ds = _base_dataset(
        PatientName="TOKEN_a1b2c3d4e5f6",
        PatientID="TOKEN_0f9e8d7c6b5a",
        StudyDate="00000000",
        PatientIdentityRemoved="YES",
        BurnedInAnnotation="NO",
    )
    method_item = pydicom.Dataset()
    method_item.CodeValue = "113100"
    ds.DeidentificationMethodCodeSequence = [method_item]
    return ds


def _dirty_dataset():
    """A dataset shaped like raw input — fails the default checklist."""
    ds = _base_dataset(
        PatientName=PHI_NAME,
        PatientID=PHI_ID,
        PatientBirthDate="19850304",
        AccessionNumber="ACC-123",
        StudyDate="20240101",
    )
    ds.add_new(pydicom.tag.Tag(0x0009, 0x0010), "LO", "VENDOR SECRET")
    return ds


def _write(ds, path: Path):
    ds.save_as(str(path), write_like_original=False)
    return path


def _write_reports(run_dir: Path, deid_uids=(), pixel_columns=None, tag_columns=None):
    pixel_columns = pixel_columns or [
        "source_path",
        "original_sop_uid",
        "deid_sop_uid",
        "frame_index",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "text_token",
        "text_len",
        "confidence",
        "decision",
    ]
    with open(run_dir / "aegis_pixel_detections.csv", "w", newline="") as f:
        csv.writer(f).writerow(pixel_columns)
    tag_columns = tag_columns or [
        "source_path",
        "original_sop_uid",
        "deid_sop_uid",
        "tag",
        "keyword",
        "action",
        "redacted",
    ]
    with open(run_dir / "aegis_tag_actions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(tag_columns)
        for uid in deid_uids:
            writer.writerow(["/in/x.dcm", "1.2.3", uid, "(0010,0010)", "PatientName", "DUMMY", "True"])


def _clean_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    ds = _clean_dataset()
    _write(ds, run / "scan.dcm")
    _write_reports(run, deid_uids=[str(ds.SOPInstanceUID)])
    return run


# ---------------------------------------------------------------------------
# Checklist loading — fail fast on invalid data
# ---------------------------------------------------------------------------


def test_default_checklist_loads():
    checklist = verify.load_checklist()
    assert checklist["name"] == "ps315-deidentification"
    assert checklist["dicom_checks"] and checklist["run_checks"]


def _load_from_yaml(tmp_path, text):
    path = tmp_path / "checklist.yaml"
    path.write_text(text)
    return verify.load_checklist(str(path))


def test_unknown_check_type_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown check type"):
        _load_from_yaml(tmp_path, "name: x\ndicom_checks:\n  - type: tag_exists\n    tag: '(0010,0010)'\n")


def test_missing_parameter_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing parameters"):
        _load_from_yaml(tmp_path, "name: x\ndicom_checks:\n  - type: tag_equals\n    tag: '(0010,0010)'\n")


def test_bad_tag_rejected(tmp_path):
    with pytest.raises(ValueError, match="unresolvable"):
        _load_from_yaml(tmp_path, "name: x\ndicom_checks:\n  - type: tag_absent\n    tag: 'NotAKeyword'\n")


def test_keyword_tag_accepted(tmp_path):
    checklist = _load_from_yaml(tmp_path, "name: x\ndicom_checks:\n  - type: tag_absent\n    tag: 'PatientName'\n")
    assert checklist["dicom_checks"][0]["_tag"] == pydicom.tag.Tag(0x0010, 0x0010)


def test_bad_pattern_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid pattern"):
        _load_from_yaml(
            tmp_path,
            "name: x\ndicom_checks:\n  - type: tag_matches\n    tag: '(0010,0010)'\n    pattern: '('\n",
        )


def test_bad_severity_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid severity"):
        _load_from_yaml(
            tmp_path,
            "name: x\ndicom_checks:\n  - type: no_private_tags\n    severity: fatal\n",
        )


def test_empty_checklist_rejected(tmp_path):
    with pytest.raises(ValueError, match="defines no checks"):
        _load_from_yaml(tmp_path, "name: x\n")


# ---------------------------------------------------------------------------
# Full-run verification: clean output passes, raw input fails
# ---------------------------------------------------------------------------


def test_clean_run_passes(tmp_path):
    report = verify.verify_run(str(_clean_run(tmp_path)))
    assert report["status"] == "pass"
    assert report["totals"]["files_checked"] == 1
    assert report["totals"]["failures"] == 0
    assert [f for f in report["findings"] if f["severity"] == "error"] == []


def test_dirty_run_fails_with_expected_checks(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write(_dirty_dataset(), run / "raw.dcm")
    report = verify.verify_run(str(run))
    assert report["status"] == "fail"
    failed = {f["check"] for f in report["findings"]}
    assert {
        "attestation-patient-identity-removed",
        "patient-name-tokenized",
        "patient-id-tokenized",
        "patient-birth-date-removed",
        "accession-number-removed",
        "study-date-zeroed",
        "no-private-tags",
        "pixel-report-present",
    } <= failed


def test_findings_never_echo_phi_values(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write(_dirty_dataset(), run / "raw.dcm")
    report_text = json.dumps(verify.verify_run(str(run)))
    assert PHI_NAME not in report_text
    assert PHI_ID not in report_text
    assert "VENDOR SECRET" not in report_text
    assert "19850304" not in report_text


def test_missing_run_dir_raises_input_error(tmp_path):
    with pytest.raises(InputError):
        verify.verify_run(str(tmp_path / "nope"))


def test_report_schema_conformance(tmp_path):
    pytest.importorskip("jsonschema")
    for build in (_clean_run,):
        report = verify.verify_run(str(build(tmp_path)))
        verify.validate_report(json.loads(json.dumps(report)))
    run = tmp_path / "dirty"
    run.mkdir()
    _write(_dirty_dataset(), run / "raw.dcm")
    verify.validate_report(json.loads(json.dumps(verify.verify_run(str(run)))))


# ---------------------------------------------------------------------------
# Individual check semantics
# ---------------------------------------------------------------------------


def _one_check_run(tmp_path, ds, yaml_text):
    run = tmp_path / "one"
    run.mkdir(exist_ok=True)
    _write(ds, run / "f.dcm")
    path = tmp_path / "one.yaml"
    path.write_text(yaml_text)
    return verify.verify_run(str(run), checklist=str(path))


def test_file_meta_mismatch_detected_in_memory():
    # Checked in memory: pydicom's save path self-heals file meta on write,
    # so a persisted mismatch can only come from other producers.
    ds = _base_dataset()
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.999"
    check = verify.load_checklist()["dicom_checks"]
    consistent = next(c for c in check if c["type"] == "file_meta_consistent")
    passed, detail = verify._check_dataset(ds, consistent)
    assert not passed
    assert "MediaStorageSOPInstanceUID" in detail


def test_tag_equals_allow_absent(tmp_path):
    yaml_text = (
        "name: x\ndicom_checks:\n"
        "  - type: tag_equals\n    tag: '(0012,0062)'\n    value: 'YES'\n    allow_absent: true\n"
    )
    report = _one_check_run(tmp_path, _base_dataset(), yaml_text)
    assert report["status"] == "pass"


def test_tag_matches_fullmatch_not_substring(tmp_path):
    yaml_text = (
        "name: x\ndicom_checks:\n" "  - type: tag_matches\n    tag: '(0010,0010)'\n    pattern: 'TOKEN_[0-9a-f]+'\n"
    )
    report = _one_check_run(tmp_path, _base_dataset(PatientName="prefix TOKEN_abc123 suffix"), yaml_text)
    assert report["status"] == "fail"  # substring match must not pass


def test_warning_severity_does_not_fail_run(tmp_path):
    yaml_text = "name: x\ndicom_checks:\n" "  - type: tag_absent\n    tag: '(0010,0010)'\n    severity: warning\n"
    report = _one_check_run(tmp_path, _base_dataset(PatientName="X"), yaml_text)
    assert report["status"] == "pass"
    assert report["totals"]["warnings"] == 1
    assert report["findings"][0]["severity"] == "warning"


def test_outputs_accounted_detects_missing_row(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ds = _clean_dataset()
    _write(ds, run / "scan.dcm")
    _write_reports(run, deid_uids=["1.2.826.0.1.3680043.8.498.42"])  # wrong uid
    report = verify.verify_run(str(run))
    assert report["status"] == "fail"
    assert any(f["check"] == "outputs-accounted-in-tag-report" for f in report["findings"])


def test_outputs_accounted_vacuous_without_dicom_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_reports(run)
    report = verify.verify_run(str(run))
    assert not any(f["check"] == "outputs-accounted-in-tag-report" for f in report["findings"])


def test_forbidden_report_column_detected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ds = _clean_dataset()
    _write(ds, run / "scan.dcm")
    _write_reports(
        run,
        deid_uids=[str(ds.SOPInstanceUID)],
        pixel_columns=["source_path", "ocr_text", "decision"],
    )
    report = verify.verify_run(str(run))
    failed = {f["check"] for f in report["findings"]}
    assert "pixel-report-phi-free" in failed
    assert "pixel-report-complete" in failed


def test_image_only_run_tag_report_warning_only(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with open(run / "aegis_pixel_detections.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "source_path",
                "original_sop_uid",
                "deid_sop_uid",
                "frame_index",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "text_token",
                "text_len",
                "confidence",
                "decision",
            ]
        )
    report = verify.verify_run(str(run))
    assert report["status"] == "pass"
    assert {f["check"] for f in report["findings"]} == {"tag-report-present"}


# ---------------------------------------------------------------------------
# instance_uids_remapped — patient-linkable UIDs must not survive de-id
# ---------------------------------------------------------------------------

_REMAP_YAML = (
    "name: x\ndicom_checks:\n"
    "  - type: instance_uids_remapped\n"
    "    root: '1.2.826.0.1.3680043.8.498.'\n"
)


def test_instance_uids_remapped_passes_when_all_under_root(tmp_path):
    ds = _base_dataset(
        StudyInstanceUID="1.2.826.0.1.3680043.8.498.100",
        SeriesInstanceUID="1.2.826.0.1.3680043.8.498.200",
    )
    report = _one_check_run(tmp_path, ds, _REMAP_YAML)
    assert report["status"] == "pass"


def test_instance_uids_remapped_fails_on_foreign_root(tmp_path):
    ds = _base_dataset(
        StudyInstanceUID="1.2.840.113619.2.55.3.100",   # original GE-issued UID
        SeriesInstanceUID="1.2.840.113619.2.55.3.200",  # survived de-identification
    )
    report = _one_check_run(tmp_path, ds, _REMAP_YAML)
    assert report["status"] == "fail"
    finding = next(f for f in report["findings"] if f["check"] == "instance_uids_remapped")
    assert "2 patient-linkable UID" in finding["detail"]
    # the offending UID value is never echoed into the report
    assert "1.2.840.113619" not in json.dumps(report)


def test_instance_uids_remapped_ignores_class_uids(tmp_path):
    # SOPClassUID lives under the DICOM standard root and must not be flagged
    # even though it is not under the remap root.
    ds = _base_dataset(StudyInstanceUID="1.2.826.0.1.3680043.8.498.100")
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    report = _one_check_run(tmp_path, ds, _REMAP_YAML)
    assert report["status"] == "pass"


def test_instance_uids_remapped_in_default_checklist(tmp_path):
    # The packaged checklist ships the rule and a real Aegis-shaped output
    # (all UIDs under the pydicom root) satisfies it.
    report = verify.verify_run(str(_clean_run(tmp_path)))
    assert report["status"] == "pass"
    ids = {c["id"] for c in verify.load_checklist()["dicom_checks"]}
    assert "instance-uids-remapped" in ids
