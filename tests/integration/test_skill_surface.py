"""Integration tests for the skill surface (api.deidentify + aegis-deidentify).

Runs the real pipeline (EasyOCR + NER models load — slow on first run) on
tiny synthetic inputs and pins the externally observable skill contract:

  - envelope conforms to the packaged JSON Schema
  - artifacts and ground-truth reports land flat in the caller's directory
  - outputs carry the PS3.15 attestation and tokenized identifiers
  - determinism: identical input + config + salt ⇒ byte-identical DICOM
    artifacts, identical reports, identical envelopes (modulo timing/paths)
  - the CLI emits exactly one JSON document on stdout

Run with pytest (module-scoped fixtures share the expensive runs):
    python -m pytest tests/integration/test_skill_surface.py -q
"""

import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pydicom
import pytest
from PIL import Image


from monai_aegis import api, envelope, skill_cli

SALT = "integration-test-salt"
# Original, institution-issued UIDs (a foreign root) — de-identification must
# replace every one of them with an Aegis-generated UID under the pydicom root.
_FOREIGN_ROOT = "1.2.840.113619.2.55.3."
_PYDICOM_ROOT = "1.2.826.0.1.3680043.8.498."
ORIGINAL_SOP_UID = _FOREIGN_ROOT + "1000"
ORIGINAL_STUDY_UID = _FOREIGN_ROOT + "2000"
ORIGINAL_SERIES_UID = _FOREIGN_ROOT + "3000"


def _make_dicom(path: Path) -> None:
    ds = pydicom.Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "SYNTH-0001"
    ds.Modality = "OT"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = ORIGINAL_SOP_UID
    ds.StudyInstanceUID = ORIGINAL_STUDY_UID
    ds.SeriesInstanceUID = ORIGINAL_SERIES_UID

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta = file_meta
    ds.preamble = b"\0" * 128

    pixels = np.zeros((16, 16), dtype=np.uint8)
    ds.PixelData = pixels.tobytes()
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.save_as(str(path), write_like_original=False)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One input set, de-identified twice with the same salt.

    Sets AEGIS_TOKEN_SALT through a MonkeyPatch context so it is restored on
    teardown — a bare ``os.environ[...] =`` here would leak the salt into
    later tests that expect a clean environment (e.g. default-salt resolution).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AEGIS_TOKEN_SALT", SALT)
        in_dir = tmp_path_factory.mktemp("skill_in")
        _make_dicom(in_dir / "scan.dcm")
        Image.new("RGB", (16, 16), color="white").save(in_dir / "photo.jpg")

        results = {}
        for run in ("run1", "run2"):
            out_dir = tmp_path_factory.mktemp(f"skill_out_{run}")
            results[run] = (out_dir, api.deidentify(str(in_dir), str(out_dir)))
        yield in_dir, results


# ---------------------------------------------------------------------------
# Envelope contract
# ---------------------------------------------------------------------------


def test_envelope_success_and_schema_valid(runs):
    pytest.importorskip("jsonschema")
    _in_dir, results = runs
    _out_dir, env = results["run1"]
    assert env["status"] == "success"
    assert env["totals"]["files_processed"] == 2
    envelope.validate_envelope(json.loads(json.dumps(env)))


def test_envelope_is_phi_free(runs):
    _in_dir, results = runs
    _out_dir, env = results["run1"]
    text = json.dumps(env)
    for forbidden in ("ocr_text", "text_token", "Test^Patient", "SYNTH-0001"):
        assert forbidden not in text


def test_artifacts_and_reports_flat_in_output_dir(runs):
    _in_dir, results = runs
    out_dir, env = results["run1"]
    processed = {f["source_file"]: f for f in env["files"] if f["status"] == "processed"}
    assert set(processed) == {"scan.dcm", "photo.jpg"}
    for entry in processed.values():
        assert entry["artifacts"], f"no artifacts for {entry['source_file']}"
        for artifact in entry["artifacts"]:
            path = Path(artifact)
            assert path.is_file()
            assert path.parent == out_dir  # caller-owned flat layout
    assert (out_dir / "aegis_pixel_detections.csv").is_file()
    assert (out_dir / "aegis_tag_actions.csv").is_file()


# ---------------------------------------------------------------------------
# Output DICOM: attestation, tokenization, deterministic UIDs
# ---------------------------------------------------------------------------


def _output_dicom(results, run):
    out_dir, env = results[run]
    entry = next(f for f in env["files"] if f["source_file"] == "scan.dcm")
    return pydicom.dcmread(entry["artifacts"][0])


def test_output_dicom_attested_and_tokenized(runs):
    _in_dir, results = runs
    ds = _output_dicom(results, "run1")
    assert str(ds.PatientIdentityRemoved) == "YES"
    assert str(ds.BurnedInAnnotation) == "NO"
    assert str(ds.PatientName).startswith("TOKEN_")
    assert str(ds.PatientID).startswith("TOKEN_")
    assert str(ds.SOPInstanceUID) != ORIGINAL_SOP_UID
    assert str(ds.file_meta.MediaStorageSOPInstanceUID) == str(ds.SOPInstanceUID)


def test_output_dicom_all_uids_remapped_off_foreign_root(runs):
    # Every patient-linkable UID must be replaced with an Aegis-generated one;
    # none of the original institution-issued UIDs may survive.
    _in_dir, results = runs
    ds = _output_dicom(results, "run1")
    for keyword, original in (
        ("SOPInstanceUID", ORIGINAL_SOP_UID),
        ("StudyInstanceUID", ORIGINAL_STUDY_UID),
        ("SeriesInstanceUID", ORIGINAL_SERIES_UID),
    ):
        value = str(getattr(ds, keyword))
        assert value != original, f"{keyword} survived de-identification"
        assert value.startswith(_PYDICOM_ROOT), f"{keyword} not under remap root: {value}"
    # SOP Class UID (standard root) is preserved so the object stays valid.
    assert str(ds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.7"


def test_deid_study_series_uids_deterministic_across_runs(runs):
    _in_dir, results = runs
    ds1, ds2 = _output_dicom(results, "run1"), _output_dicom(results, "run2")
    assert str(ds1.StudyInstanceUID) == str(ds2.StudyInstanceUID)
    assert str(ds1.SeriesInstanceUID) == str(ds2.SeriesInstanceUID)


def test_deid_uid_deterministic_across_runs(runs):
    _in_dir, results = runs
    uid1 = str(_output_dicom(results, "run1").SOPInstanceUID)
    uid2 = str(_output_dicom(results, "run2").SOPInstanceUID)
    assert uid1 == uid2


# ---------------------------------------------------------------------------
# Determinism: identical input + config + salt ⇒ identical outputs
# ---------------------------------------------------------------------------


def _read_bytes_by_name(out_dir: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(out_dir.iterdir()) if p.is_file()}


def test_repeat_run_byte_identical_artifacts_and_reports(runs):
    _in_dir, results = runs
    files1 = _read_bytes_by_name(results["run1"][0])
    files2 = _read_bytes_by_name(results["run2"][0])
    assert files1.keys() == files2.keys()
    for name in files1:
        assert files1[name] == files2[name], f"{name} differs between identical runs"


def test_repeat_run_identical_envelopes_modulo_timing_and_paths(runs):
    _in_dir, results = runs

    def normalized(run):
        out_dir, env = results[run]
        env = copy.deepcopy(env)
        env.pop("elapsed_seconds", None)
        text = json.dumps(env, sort_keys=True)
        return text.replace(str(out_dir), "<out>")

    assert normalized("run1") == normalized("run2")


def test_pixel_report_tokens_not_verbatim(runs):
    _in_dir, results = runs
    out_dir, _env = results["run1"]
    with open(out_dir / "aegis_pixel_detections.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert "ocr_text" not in row
        assert row.get("text_token", "") == "" or row["text_token"].startswith("TOKEN_")


# ---------------------------------------------------------------------------
# CLI end-to-end (in-process): stdout carries exactly one JSON document
# ---------------------------------------------------------------------------


def test_cli_single_file_stdout_contract(runs, tmp_path, capsys):
    in_dir, _results = runs
    out_dir = tmp_path / "cli_out"
    code = skill_cli.main([str(in_dir / "scan.dcm"), "--output-dir", str(out_dir)])
    out = capsys.readouterr().out
    env = json.loads(out)  # whole stream is one JSON document
    assert code == skill_cli.EXIT_SUCCESS
    assert env["status"] == "success"
    assert env["totals"]["files_processed"] == 1
    assert (out_dir / "aegis_pixel_detections.csv").is_file()


def test_cli_missing_input_exit_two(tmp_path, capsys):
    code = skill_cli.main([str(tmp_path / "nope.dcm"), "--output-dir", str(tmp_path / "o")])
    env = json.loads(capsys.readouterr().out)
    assert code == skill_cli.EXIT_INVALID_INPUT
    assert env["status"] == "error"
