"""Unit tests for the invocation facade (monai_aegis/api.py).

Pipeline execution is stubbed at the facade's seams (_build_pipeline and
the discovery helpers) — the real transforms are covered by their own
tests and by tests/integration/test_skill_surface.py. These tests pin the
facade contract: input classification, sorted discovery, per-file error
continuation, caller-owned flat output, report writing, and the
config/overlay resolution rules.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monai_aegis import api

PHI_TEXT = "John Doe 1985-03-04"


def _fake_result(source, saved_path, decision="redacted"):
    return {
        "image_meta_dict": {"filename_or_obj": source},
        "image_redaction_stats": {
            "detections": [
                {"bbox": [1, 2, 3, 4], "ocr_text": PHI_TEXT, "confidence": 0.9, "decision": decision},
            ]
        },
        "image_tag_actions": [
            {"tag": "(0010,0010)", "keyword": "PatientName", "action": "DUMMY", "redacted": True},
        ],
        "image_saved_path": saved_path,
    }


class _FakePipeline:
    """Records calls; returns a canned result or raises for marked paths."""

    def __init__(self, out_dir, fail_on=()):
        self.out_dir = out_dir
        self.fail_on = set(fail_on)
        self.calls = []

    def __call__(self, data):
        path = data["image"]
        self.calls.append(path)
        if Path(path).name in self.fail_on:
            raise RuntimeError(f"boom while reading {PHI_TEXT}")
        return _fake_result(path, str(Path(self.out_dir) / Path(path).name))


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Stub pipelines + discovery; return (input_dir, out_dir, pipelines)."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for name in ("b.dcm", "a.dcm", "c.png"):
        (in_dir / name).write_bytes(b"\x00" * 132)

    pipelines = {}

    def fake_build(kind, config_path, input_dir, output_dir, overlay_path=None):
        pipelines.setdefault(kind, _FakePipeline(output_dir))
        return pipelines[kind]

    monkeypatch.setattr(api, "_build_pipeline", fake_build)
    monkeypatch.setattr(
        api,
        "discover_dicom_files",
        lambda d: sorted(str(p) for p in Path(d).glob("*.dcm")),
    )
    monkeypatch.setattr(
        api,
        "discover_image_files",
        lambda d: sorted(str(p) for p in Path(d).glob("*.png")),
    )
    return in_dir, out_dir, pipelines


# ---------------------------------------------------------------------------
# Classification and discovery
# ---------------------------------------------------------------------------


def test_classify_input():
    assert api.classify_input("/x/scan.JPG") == "image"
    assert api.classify_input("/x/scan.png") == "image"
    assert api.classify_input("/x/scan.dcm") == "dicom"
    assert api.classify_input("/x/extensionless") == "dicom"


def test_discover_inputs_modes(stubbed):
    in_dir, _out, _p = stubbed
    auto = api.discover_inputs(str(in_dir), "auto")
    assert [(Path(p).name, k) for p, k in auto] == [
        ("a.dcm", "dicom"),
        ("b.dcm", "dicom"),
        ("c.png", "image"),
    ]
    assert all(k == "dicom" for _p, k in api.discover_inputs(str(in_dir), "dicom"))
    assert all(k == "image" for _p, k in api.discover_inputs(str(in_dir), "image"))


def test_discover_inputs_unknown_mode():
    with pytest.raises(api.InputError):
        api.discover_inputs("/tmp", "everything")


# ---------------------------------------------------------------------------
# Config / overlay resolution
# ---------------------------------------------------------------------------


def test_default_paths_exist():
    assert Path(api.default_config_path()).is_file()
    assert Path(api.skill_overlay_path()).is_file()


def test_overlay_defaults_to_skill_profile(monkeypatch):
    monkeypatch.delenv("AEGIS_CONFIG_OVERRIDE", raising=False)
    assert api.resolve_overlay_path(None) == api.skill_overlay_path()


def test_explicit_overlay_wins(monkeypatch):
    monkeypatch.delenv("AEGIS_CONFIG_OVERRIDE", raising=False)
    assert api.resolve_overlay_path("/my/overlay.yaml") == "/my/overlay.yaml"


def test_env_override_suppresses_skill_overlay(monkeypatch):
    monkeypatch.setenv("AEGIS_CONFIG_OVERRIDE", "/deploy/prod.yaml")
    assert api.resolve_overlay_path(None) is None  # loader honors the env var


def test_load_run_config_applies_skill_overlay(monkeypatch):
    monkeypatch.delenv("AEGIS_CONFIG_OVERRIDE", raising=False)
    config = api.load_run_config()
    assert config["runtime"]["dataloader_num_workers"] == 0
    assert config["reporting"]["save_ground_truth"] is True


# ---------------------------------------------------------------------------
# deidentify — directory runs
# ---------------------------------------------------------------------------


def test_directory_run_sorted_and_flat_output(stubbed):
    in_dir, out_dir, pipelines = stubbed
    env = api.deidentify(str(in_dir), str(out_dir))

    assert env["status"] == "success"
    assert [f["source_file"] for f in env["files"]] == ["a.dcm", "b.dcm", "c.png"]
    assert [Path(p).name for p in pipelines["dicom"].calls] == ["a.dcm", "b.dcm"]
    assert env["totals"]["files_processed"] == 3
    assert env["totals"]["header_tags_scrubbed"] == 3  # stub emits one DUMMY tag per file

    # Reports land directly in the caller's directory — no timestamped nesting.
    assert (out_dir / "aegis_pixel_detections.csv").is_file()
    assert (out_dir / "aegis_tag_actions.csv").is_file()
    assert sorted(Path(p).name for p in env["reports"]) == [
        "aegis_pixel_detections.csv",
        "aegis_tag_actions.csv",
    ]


def test_directory_run_per_file_error_continues(stubbed, monkeypatch):
    in_dir, out_dir, pipelines = stubbed

    def failing_build(kind, config_path, input_dir, output_dir, overlay_path=None):
        pipelines.setdefault(kind, _FakePipeline(output_dir, fail_on={"a.dcm"}))
        return pipelines[kind]

    monkeypatch.setattr(api, "_build_pipeline", failing_build)
    env = api.deidentify(str(in_dir), str(out_dir))

    assert env["status"] == "partial"
    assert env["totals"]["files_processed"] == 2
    assert env["totals"]["files_failed"] == 1
    failed = [f for f in env["files"] if f["status"] == "failed"]
    assert failed[0]["source_file"] == "a.dcm"
    # b.dcm and c.png still processed after the failure
    assert {f["source_file"] for f in env["files"] if f["status"] == "processed"} == {"b.dcm", "c.png"}


def test_envelope_error_field_is_phi_free(stubbed, monkeypatch):
    # The stub raises RuntimeError("boom while reading John Doe 1985-03-04").
    # The failing file's error field must carry only a PHI-safe code (the
    # exception type), never the raw message that embeds PHI (P4).
    in_dir, out_dir, pipelines = stubbed

    def failing_build(kind, config_path, input_dir, output_dir, overlay_path=None):
        pipelines.setdefault(kind, _FakePipeline(output_dir, fail_on={"a.dcm"}))
        return pipelines[kind]

    monkeypatch.setattr(api, "_build_pipeline", failing_build)
    env = api.deidentify(str(in_dir), str(out_dir))

    failed = next(f for f in env["files"] if f["status"] == "failed")
    assert failed["error"] == "RuntimeError"  # type name only, no message

    text = json.dumps(env)
    assert PHI_TEXT not in text  # the exception message is withheld
    assert "ocr_text" not in text
    assert "text_token" not in text


def test_verbose_errors_opt_in_restores_message(stubbed, monkeypatch):
    # Local-debugging escape hatch: AEGIS_VERBOSE_ERRORS re-enables the full
    # message for the caller who owns the data.
    in_dir, out_dir, pipelines = stubbed
    monkeypatch.setenv("AEGIS_VERBOSE_ERRORS", "1")

    def failing_build(kind, config_path, input_dir, output_dir, overlay_path=None):
        pipelines.setdefault(kind, _FakePipeline(output_dir, fail_on={"a.dcm"}))
        return pipelines[kind]

    monkeypatch.setattr(api, "_build_pipeline", failing_build)
    env = api.deidentify(str(in_dir), str(out_dir))
    failed = next(f for f in env["files"] if f["status"] == "failed")
    assert failed["error"].startswith("RuntimeError: ")


def test_empty_directory_success_with_message(stubbed, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    env = api.deidentify(str(empty), str(tmp_path / "out"))
    assert env["status"] == "success"
    assert env["totals"]["files_discovered"] == 0
    assert "No DICOM or image files found" in env["message"]


# ---------------------------------------------------------------------------
# deidentify — single file and input validation
# ---------------------------------------------------------------------------


def test_single_file_uses_own_kind(stubbed):
    in_dir, out_dir, pipelines = stubbed
    env = api.deidentify(str(in_dir / "c.png"), str(out_dir), mode="dicom")
    assert env["files"][0]["kind"] == "image"  # file's kind wins over mode
    assert env["status"] == "success"
    assert env["files"][0]["artifacts"] == [str(out_dir / "c.png")]


def test_missing_input_raises_input_error(tmp_path):
    with pytest.raises(api.InputError):
        api.deidentify(str(tmp_path / "nope.dcm"), str(tmp_path / "out"))


def test_bad_mode_raises_input_error(stubbed, tmp_path):
    in_dir, _out, _p = stubbed
    with pytest.raises(api.InputError):
        api.deidentify(str(in_dir), str(tmp_path / "out"), mode="everything")


def test_envelope_schema_conformance_from_facade(stubbed):
    pytest.importorskip("jsonschema")
    from monai_aegis import envelope as env_mod

    in_dir, out_dir, _p = stubbed
    env = api.deidentify(str(in_dir), str(out_dir))
    env_mod.validate_envelope(json.loads(json.dumps(env)))
