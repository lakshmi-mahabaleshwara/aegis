"""Integration tests for the trust loop: fixture → deidentify → verify.

Runs the real pipeline on generated synthetic fixtures and closes the
loop with the independent verification pass:

  - a raw fixture directory fails the default checklist
  - the de-identified output of that same directory passes it
  - burnt-in text rendered by the fixture generator is actually detected
    and redacted by the OCR path
  - the aegis-verify CLI end-to-end exit codes and stdout contract

Run with pytest (module-scoped fixtures share the expensive pipeline run):
    python -m pytest tests/integration/test_verify_surface.py -q
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monai_aegis import api, fixtures, verify, verify_cli

# Burnt-in strings that the deterministic heuristic layer (Layer 2) redacts
# without depending on the NER model's judgement of a specific name — a
# date-ID stamp and a facility name, mirroring real ultrasound annotations.
BURNED_IN_LINES = ("20240101-091825-6AAA", "HOSPITAL CENTER")
FIXTURE_SIZE = (400, 240)
FIXTURE_TEXT_SIZE = 26


@pytest.fixture(scope="module")
def trust_loop(tmp_path_factory):
    """Generate fixtures, de-identify them once, return (in_dir, out_dir, env).

    Sets AEGIS_TOKEN_SALT through a MonkeyPatch context so it is restored on
    teardown rather than leaking into later tests (env salt overrides config
    salt in AegisIdentityManager).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AEGIS_TOKEN_SALT", "verify-surface-salt")
        in_dir = tmp_path_factory.mktemp("verify_in")
        fixtures.make_synthetic_dicom(
            str(in_dir / "scan.dcm"),
            size=FIXTURE_SIZE,
            burned_in_text=BURNED_IN_LINES,
            text_size=FIXTURE_TEXT_SIZE,
            with_private_tag=True,
        )
        fixtures.make_synthetic_image(str(in_dir / "photo.png"))
        out_dir = tmp_path_factory.mktemp("verify_out")
        env = api.deidentify(str(in_dir), str(out_dir))
        yield in_dir, out_dir, env


def test_raw_fixture_dir_fails_checklist(trust_loop):
    in_dir, _out, _env = trust_loop
    report = verify.verify_run(str(in_dir))
    assert report["status"] == "fail"
    failed = {f["check"] for f in report["findings"]}
    assert "patient-name-tokenized" in failed
    assert "no-private-tags" in failed
    assert "attestation-patient-identity-removed" in failed


def test_deidentified_output_passes_checklist(trust_loop):
    _in, out_dir, env = trust_loop
    assert env["status"] == "success"
    report = verify.verify_run(str(out_dir))
    assert [f for f in report["findings"] if f["severity"] == "error"] == [], report["findings"]
    assert report["status"] == "pass"
    assert report["totals"]["files_checked"] == 1


def test_verification_report_schema_valid(trust_loop):
    pytest.importorskip("jsonschema")
    _in, out_dir, _env = trust_loop
    report = verify.verify_run(str(out_dir))
    verify.validate_report(json.loads(json.dumps(report)))


def test_burned_in_text_detected_and_redacted(trust_loop):
    _in, out_dir, env = trust_loop
    entry = next(f for f in env["files"] if f["source_file"] == "scan.dcm")
    assert entry["pixel_decisions"]["redacted"] >= 1, (
        "burnt-in PHI was not redacted — " f"decisions: {entry['pixel_decisions']}"
    )
    # The redacted regions are zeroed in the output pixels: the text rows at
    # the top of the frame must have lost brightness relative to the input.
    ds = pydicom.dcmread(entry["artifacts"][0])
    top_band = ds.pixel_array[:80]
    original = fixtures.render_text_pixels(FIXTURE_SIZE, BURNED_IN_LINES, FIXTURE_TEXT_SIZE)[:80]
    assert int(np.count_nonzero(top_band)) < int(np.count_nonzero(original))


def test_verify_cli_end_to_end(trust_loop, capsys):
    in_dir, out_dir, _env = trust_loop
    code = verify_cli.main([str(out_dir)])
    report = json.loads(capsys.readouterr().out)
    assert code == verify_cli.EXIT_PASS
    assert report["status"] == "pass"

    code = verify_cli.main([str(in_dir)])
    report = json.loads(capsys.readouterr().out)
    assert code == verify_cli.EXIT_FAIL
    assert report["status"] == "fail"

    assert (
        verify_cli.main(
            [
                str(in_dir / "nope"),
            ]
        )
        == verify_cli.EXIT_INVALID_INPUT
    )
    capsys.readouterr()
