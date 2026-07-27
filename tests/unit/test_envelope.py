"""Unit tests for the PHI-free result envelope (monai_aegis/envelope.py).

Covers the shared summarization used by every skill surface: decision
counting, the P2 guard (no OCR text or tokens in any envelope), artifact
collection, run-envelope status derivation and totals, and conformance of
built envelopes to the packaged JSON Schema.
"""

import json

import pytest


from monai_aegis import envelope

PHI_TEXT = "John Doe 1985-03-04"


def _fake_result(
    num_redacted=2, num_safelisted=1, num_low_conf=0, tags=3, source="/data/in/IMG_0042.dcm", saved_path=""
):
    """Pipeline-shaped result dict carrying raw PHI in the fields the
    envelope must never surface."""
    detections = []
    for _ in range(num_redacted):
        detections.append({"bbox": [1, 2, 3, 4], "ocr_text": PHI_TEXT, "confidence": 0.9, "decision": "redacted"})
    for _ in range(num_safelisted):
        detections.append({"bbox": [5, 6, 7, 8], "ocr_text": "MI 1.14", "confidence": 0.8, "decision": "safelisted"})
    for _ in range(num_low_conf):
        detections.append({"bbox": [9, 9, 9, 9], "ocr_text": PHI_TEXT, "confidence": 0.2, "decision": "low_confidence"})
    tag_actions = [
        {"tag": "(0010,0010)", "keyword": "PatientName", "action": "DUMMY", "redacted": True} for _ in range(tags)
    ]
    result = {
        "image_meta_dict": {"filename_or_obj": source},
        "image_redaction_stats": {"detections": detections},
        "image_tag_actions": tag_actions,
    }
    if saved_path:
        result["image_saved_path"] = saved_path
    return result


def _assert_phi_free(obj):
    text = json.dumps(obj)
    assert "ocr_text" not in text
    assert PHI_TEXT not in text
    assert "text_token" not in text


# ---------------------------------------------------------------------------
# summarize_result
# ---------------------------------------------------------------------------


def test_summarize_counts_and_phi_free():
    summary = envelope.summarize_result(_fake_result(2, 1, 1, tags=3))
    assert summary["pixel_regions_detected"] == 4
    assert summary["pixel_decisions"] == {"redacted": 2, "safelisted": 1, "low_confidence": 1}
    assert summary["header_tags_scrubbed"] == 3
    assert summary["needs_manual_review"] is True
    _assert_phi_free(summary)


def test_summarize_keep_tags_not_counted():
    result = _fake_result(tags=0)
    result["image_tag_actions"] = [
        {"tag": "(0008,0060)", "keyword": "Modality", "action": "KEEP", "redacted": False},
        {"tag": "(0010,0010)", "keyword": "PatientName", "action": "DUMMY", "redacted": True},
    ]
    assert envelope.summarize_result(result)["header_tags_scrubbed"] == 1


def test_summarize_no_review_when_confident():
    assert envelope.summarize_result(_fake_result(1, 1, 0))["needs_manual_review"] is False


def test_summarize_empty_result():
    summary = envelope.summarize_result({})
    assert summary["pixel_regions_detected"] == 0
    assert summary["header_tags_scrubbed"] == 0
    assert summary["needs_manual_review"] is False


# ---------------------------------------------------------------------------
# collect_artifacts
# ---------------------------------------------------------------------------


def test_collect_artifacts_single_path():
    result = _fake_result(saved_path="/out/IMG_0042.dcm")
    assert envelope.collect_artifacts(result) == ["/out/IMG_0042.dcm"]


def test_collect_artifacts_series_paths_sorted():
    result = _fake_result()
    result["image_saved_paths"] = ["/out/b.dcm", "/out/a.dcm"]
    assert envelope.collect_artifacts(result) == ["/out/a.dcm", "/out/b.dcm"]


def test_collect_artifacts_absent():
    assert envelope.collect_artifacts(_fake_result()) == []


# ---------------------------------------------------------------------------
# build_envelope — status derivation, totals, PHI guard
# ---------------------------------------------------------------------------


def _processed(name, kind="dicom", redacted=1, low_conf=0, tags=2, artifacts=()):
    decisions = {"redacted": redacted, "safelisted": 0, "low_confidence": low_conf}
    return envelope.file_entry(
        name,
        kind,
        {
            "pixel_regions_detected": redacted + low_conf,
            "pixel_decisions": decisions,
            "header_tags_scrubbed": tags,
            "needs_manual_review": low_conf > 0,
        },
        list(artifacts),
    )


def _build(files, **kwargs):
    return envelope.build_envelope(
        input_path="/data/in",
        output_dir="/data/out",
        mode="auto",
        files=files,
        **kwargs,
    )


def test_envelope_success_and_totals():
    env = _build(
        [_processed("a.dcm", redacted=2, tags=3), _processed("b.png", kind="image", redacted=1, tags=0)],
        reports=["/data/out/aegis_pixel_detections.csv"],
        elapsed_seconds=1.23,
    )
    assert env["status"] == "success"
    assert env["totals"]["files_discovered"] == 2
    assert env["totals"]["files_processed"] == 2
    assert env["totals"]["files_failed"] == 0
    assert env["totals"]["pixel_regions_detected"] == 3
    assert env["totals"]["pixel_decisions"]["redacted"] == 3
    assert env["totals"]["header_tags_scrubbed"] == 3
    assert env["needs_manual_review"] is False
    assert env["elapsed_seconds"] == 1.2
    assert env["errors"] == []
    _assert_phi_free(env)


def test_envelope_partial_when_some_fail():
    env = _build(
        [
            _processed("a.dcm"),
            envelope.failed_file_entry("bad.dcm", "dicom", "DicomLoadError: unreadable"),
        ]
    )
    assert env["status"] == "partial"
    assert env["totals"]["files_failed"] == 1
    assert env["errors"] == ["bad.dcm: DicomLoadError: unreadable"]


def test_envelope_error_when_all_fail():
    env = _build([envelope.failed_file_entry("bad.dcm", "dicom", "DicomLoadError: unreadable")])
    assert env["status"] == "error"


def test_envelope_review_flag_propagates():
    env = _build([_processed("a.dcm", low_conf=1)])
    assert env["needs_manual_review"] is True


def test_envelope_empty_discovery_is_success_with_message():
    env = _build([], message="No DICOM or image files found in /data/in.")
    assert env["status"] == "success"
    assert env["totals"]["files_discovered"] == 0
    assert "No DICOM" in env["message"]


def test_error_envelope():
    env = envelope.error_envelope("no such file: /x", "/x", "/out")
    assert env["status"] == "error"
    assert env["errors"] == ["no such file: /x"]
    assert env["envelope_version"] == envelope.ENVELOPE_VERSION


# ---------------------------------------------------------------------------
# safe_error — PHI-safe exception rendering (P4)
# ---------------------------------------------------------------------------


def test_safe_error_returns_type_name_only(monkeypatch):
    monkeypatch.delenv(envelope.VERBOSE_ERRORS_ENV, raising=False)
    exc = ValueError("bad date 1985-03-04 for John Doe")
    rendered = envelope.safe_error(exc)
    assert rendered == "ValueError"
    assert "John Doe" not in rendered
    assert "1985-03-04" not in rendered


def test_safe_error_verbose_opt_in_includes_message(monkeypatch):
    monkeypatch.setenv(envelope.VERBOSE_ERRORS_ENV, "1")
    exc = ValueError("context detail")
    assert envelope.safe_error(exc) == "ValueError: context detail"


def test_safe_error_verbose_falsey_values_stay_safe(monkeypatch):
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(envelope.VERBOSE_ERRORS_ENV, value)
        assert envelope.safe_error(RuntimeError("secret")) == "RuntimeError"


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


def test_schema_loads_and_pins_version():
    schema = envelope.schema()
    assert schema["properties"]["envelope_version"]["const"] == envelope.ENVELOPE_VERSION


def test_built_envelopes_validate_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    for env in (
        _build(
            [
                _processed("a.dcm", artifacts=["/data/out/a.dcm"]),
                envelope.failed_file_entry("bad.dcm", "dicom", "boom"),
            ],
            reports=["/data/out/aegis_pixel_detections.csv"],
            elapsed_seconds=0.5,
        ),
        _build([], message="No DICOM or image files found in /data/in."),
        envelope.error_envelope("no such file", "/x", "/out"),
    ):
        envelope.validate_envelope(env)  # must not raise
        # JSON round-trip stays valid (what a harness actually parses)
        jsonschema.validate(json.loads(json.dumps(env)), envelope.schema())


def test_schema_rejects_tampered_envelope():
    jsonschema = pytest.importorskip("jsonschema")
    env = _build([_processed("a.dcm")])
    env["status"] = "not-a-status"
    with pytest.raises(jsonschema.ValidationError):
        envelope.validate_envelope(env)


def test_schema_rejects_unknown_top_level_field():
    jsonschema = pytest.importorskip("jsonschema")
    env = _build([_processed("a.dcm")])
    env["ocr_text"] = "leaked"
    with pytest.raises(jsonschema.ValidationError):
        envelope.validate_envelope(env)
