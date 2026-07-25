"""Unit tests for the aegis-mcp server (aegis_mcp_server.py).

Covers the spec §12 unit matrix: discovery fallback, job state transitions
with per-file error continuation, the P2 guard (no OCR text in any response
or job_summary artifact), the unknown-job_id envelope, and summarize_run
source priority / per-file filtering. The monai_aegis pipeline itself is
never invoked — pipeline execution is stubbed at the server's seams.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monai_aegis import mcp_server as server


PHI_TEXT = "John Doe 1985-03-04"


def _fake_result(num_redacted=2, num_safelisted=1, num_low_conf=0, tags=3,
                 source="/data/in/IMG_0042.dcm"):
    """Build a pipeline result dict shaped like the real transforms produce,
    with raw PHI in the fields the server must never surface."""
    detections = []
    for _ in range(num_redacted):
        detections.append({"bbox": [1, 2, 3, 4], "ocr_text": PHI_TEXT,
                           "confidence": 0.9, "decision": "redacted"})
    for _ in range(num_safelisted):
        detections.append({"bbox": [5, 6, 7, 8], "ocr_text": "MI 1.14",
                           "confidence": 0.8, "decision": "safelisted"})
    for _ in range(num_low_conf):
        detections.append({"bbox": [9, 9, 9, 9], "ocr_text": PHI_TEXT,
                           "confidence": 0.2, "decision": "low_confidence"})
    tag_actions = [
        {"tag": "(0010,0010)", "keyword": "PatientName", "action": "DUMMY", "redacted": True}
        for _ in range(tags)
    ]
    return {
        "image_meta_dict": {"filename_or_obj": source},
        "image_redaction_stats": {"detections": detections},
        "image_tag_actions": tag_actions,
    }


def _assert_phi_free(obj):
    text = json.dumps(obj)
    assert "ocr_text" not in text
    assert PHI_TEXT not in text
    assert "text_token" not in text


# ---------------------------------------------------------------------------
# Discovery fallback (DICM magic, extensionless files)
# ---------------------------------------------------------------------------

def _write_dicm(path: Path):
    path.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 16)


def test_fallback_discovery_finds_dicm_and_dcm(tmp_path):
    _write_dicm(tmp_path / "with_magic")           # extensionless, DICM magic
    (tmp_path / "nested").mkdir()
    _write_dicm(tmp_path / "nested" / "deep.ima")  # non-.dcm extension, DICM magic
    (tmp_path / "plain.dcm").write_bytes(b"no magic but .dcm extension")
    (tmp_path / "notes.txt").write_text("not a dicom")
    (tmp_path / "random.bin").write_bytes(b"\x00" * 200)

    found = {Path(p).name for p in server._fallback_discover(str(tmp_path))}
    assert found == {"with_magic", "deep.ima", "plain.dcm"}


def test_fallback_discovery_short_file_not_crash(tmp_path):
    (tmp_path / "tiny").write_bytes(b"xx")
    assert server._fallback_discover(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# P2 guard — no OCR text in any summarized output
# ---------------------------------------------------------------------------

def test_summarize_result_counts_and_phi_free():
    summary = server._summarize_result(_fake_result(2, 1, 1, tags=3))
    assert summary["pixel_regions_detected"] == 4
    assert summary["pixel_decisions"] == {"redacted": 2, "safelisted": 1, "low_confidence": 1}
    assert summary["header_tags_scrubbed"] == 3
    assert summary["needs_manual_review"] is True
    _assert_phi_free(summary)


def test_summarize_result_keep_tags_not_counted():
    result = _fake_result(tags=0)
    result["image_tag_actions"] = [
        {"tag": "(0008,0060)", "keyword": "Modality", "action": "KEEP", "redacted": False},
        {"tag": "(0010,0010)", "keyword": "PatientName", "action": "DUMMY", "redacted": True},
    ]
    assert server._summarize_result(result)["header_tags_scrubbed"] == 1


# ---------------------------------------------------------------------------
# Job state machine — per-file errors never fail the job
# ---------------------------------------------------------------------------

def _make_job(job_id, in_dir, out_dir):
    return {
        "job_id": job_id,
        "state": "queued",
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "total": 0,
        "processed": 0,
        "decisions": {name: 0 for name in server.PIXEL_DECISIONS},
        "header_tags_scrubbed": 0,
        "errors": [],
    }


def test_batch_job_continues_past_per_file_errors(tmp_path, monkeypatch):
    files = [str(tmp_path / n) for n in ("a.dcm", "b.dcm", "c.dcm")]
    monkeypatch.setattr(server, "_discover_dicom_files", lambda d: files)
    monkeypatch.setattr(server, "_discover_image_files", lambda d: [])
    monkeypatch.setattr(server, "_get_config", lambda: {"reporting": {"save_ground_truth": False}})

    def fake_run(fn, kind, path, in_dir, out_dir):
        if path.endswith("b.dcm"):
            raise RuntimeError("boom on b")
        return _fake_result(num_redacted=1, tags=2)

    monkeypatch.setattr(server, "_run_on_pipeline_thread", fake_run)

    job = _make_job("deadbeef", tmp_path, tmp_path / "out")
    server._run_batch_job(job)

    assert job["state"] == "completed"
    assert job["total"] == 3
    assert job["processed"] == 3
    assert len(job["errors"]) == 1
    assert "b.dcm" in job["errors"][0]
    assert job["decisions"]["redacted"] == 2
    assert job["header_tags_scrubbed"] == 4

    # Summary written, totals equal sum of per-file entries, PHI-free.
    summary_path = Path(job["summary_file"])
    assert summary_path.name == "job_summary_deadbeef.json"
    data = json.loads(summary_path.read_text())
    assert set(data["per_file"]) == {"a.dcm", "c.dcm"}
    per_file_tags = sum(v["header_tags_scrubbed"] for v in data["per_file"].values())
    assert data["totals"]["header_tags_scrubbed"] == per_file_tags
    assert data["totals"]["errors"] == 1
    _assert_phi_free(data)


def test_batch_job_discovery_failure_marks_failed(tmp_path, monkeypatch):
    def explode(_):
        raise RuntimeError("discovery exploded")

    monkeypatch.setattr(server, "_discover_dicom_files", explode)
    job = _make_job("cafef00d", tmp_path, tmp_path / "out")
    server._run_batch_job(job)
    assert job["state"] == "failed"
    # PHI-safe: the job message carries the exception type, not the raw text.
    assert job["message"] == "RuntimeError"
    assert "discovery exploded" not in job["message"]


def test_batch_job_empty_input_completes_with_message(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_discover_dicom_files", lambda d: [])
    monkeypatch.setattr(server, "_discover_image_files", lambda d: [])
    monkeypatch.setattr(server, "_get_config", lambda: {"reporting": {"save_ground_truth": False}})
    job = _make_job("00000000", tmp_path, tmp_path / "out")
    server._run_batch_job(job)
    assert job["state"] == "completed"
    assert job["total"] == 0
    assert "No DICOM or image files" in job["message"]


# ---------------------------------------------------------------------------
# Batch mode parameter — aligned with the unified CLI (auto | dicom | image)
# ---------------------------------------------------------------------------

def test_fallback_discover_images(tmp_path):
    (tmp_path / "a.PNG").write_bytes(b"x")
    (tmp_path / "b.jpeg").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.jpg").write_bytes(b"x")
    (tmp_path / "d.dcm").write_bytes(b"x")
    (tmp_path / "e.txt").write_text("x")
    found = {Path(p).name for p in server._fallback_discover_images(str(tmp_path))}
    assert found == {"a.PNG", "b.jpeg", "c.jpg"}


def test_discover_batch_files_mode_filtering(monkeypatch):
    monkeypatch.setattr(server, "_discover_dicom_files", lambda d: ["/x/a.dcm"])
    monkeypatch.setattr(server, "_discover_image_files", lambda d: ["/x/b.png"])
    assert server._discover_batch_files("/x", "auto") == [
        ("/x/a.dcm", "dicom"), ("/x/b.png", "image")]
    assert server._discover_batch_files("/x", "dicom") == [("/x/a.dcm", "dicom")]
    assert server._discover_batch_files("/x", "image") == [("/x/b.png", "image")]


def test_start_batch_job_invalid_mode(tmp_path):
    response = server.start_batch_job(str(tmp_path), mode="video")
    assert response["status"] == "error"
    assert "mode" in response["message"] and "auto" in response["message"]


def test_start_batch_job_mode_normalized(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_jobs", {})
    monkeypatch.setattr(server.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: None})())
    response = server.start_batch_job(str(tmp_path), mode=" DICOM ")
    assert response["status"] == "started"
    assert response["mode"] == "dicom"
    assert server._jobs[response["job_id"]]["mode"] == "dicom"


def test_batch_mode_routes_kinds_and_tag_csv_alignment(tmp_path, monkeypatch):
    """auto processes DICOM then images; image mode writes no tag-actions CSV,
    matching the image runner's convention."""
    monkeypatch.setattr(server, "_discover_dicom_files", lambda d: [str(tmp_path / "a.dcm")])
    monkeypatch.setattr(server, "_discover_image_files",
                        lambda d: [str(tmp_path / "x.png"), str(tmp_path / "y.jpg")])
    monkeypatch.setattr(server, "_get_config", lambda: {"reporting": {"save_ground_truth": True}})

    seen = []

    def fake_run(fn, kind, path, in_dir, out_dir):
        seen.append((kind, Path(path).name))
        result = _fake_result(num_redacted=1, tags=(2 if kind == "dicom" else 0), source=path)
        if kind == "image":
            del result["image_tag_actions"]
        return result

    monkeypatch.setattr(server, "_run_on_pipeline_thread", fake_run)

    job = _make_job("aaaa0001", tmp_path, tmp_path / "out_auto")
    job["mode"] = "auto"
    server._run_batch_job(job)
    assert job["state"] == "completed"
    assert seen == [("dicom", "a.dcm"), ("image", "x.png"), ("image", "y.jpg")]
    assert (tmp_path / "out_auto" / "aegis_tag_actions.csv").is_file()
    assert (tmp_path / "out_auto" / "aegis_pixel_detections.csv").is_file()

    seen.clear()
    job = _make_job("aaaa0002", tmp_path, tmp_path / "out_img")
    job["mode"] = "image"
    server._run_batch_job(job)
    assert [kind for kind, _ in seen] == ["image", "image"]
    assert (tmp_path / "out_img" / "aegis_pixel_detections.csv").is_file()
    assert not (tmp_path / "out_img" / "aegis_tag_actions.csv").exists()

    seen.clear()
    job = _make_job("aaaa0003", tmp_path, tmp_path / "out_dcm")
    job["mode"] = "dicom"
    server._run_batch_job(job)
    assert [kind for kind, _ in seen] == ["dicom"]

    summary = json.loads(Path(job["summary_file"]).read_text())
    assert summary["mode"] == "dicom"


# ---------------------------------------------------------------------------
# get_job_status
# ---------------------------------------------------------------------------

def test_unknown_job_id_envelope(monkeypatch):
    monkeypatch.setattr(server, "_jobs", {"abc12345": _make_job("abc12345", "/in", "/out")})
    response = server.get_job_status("nope1234")
    assert response["status"] == "error"
    assert "nope1234" in response["message"]
    assert response["known_jobs"] == ["abc12345"]


def test_job_status_snapshot_caps_errors(monkeypatch):
    job = _make_job("abc12345", "/in", "/out")
    job["state"] = "running"
    job["errors"] = [f"f{i}.dcm: boom" for i in range(25)]
    monkeypatch.setattr(server, "_jobs", {"abc12345": job})
    response = server.get_job_status("abc12345")
    assert response["status"] == "success"
    assert len(response["errors"]) == server.JOB_ERRORS_LIMIT
    assert response["error_count"] == 25


# ---------------------------------------------------------------------------
# summarize_run — source priority and per-file filtering
# ---------------------------------------------------------------------------

def _write_run_csvs(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "aegis_pixel_detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_path", "original_sop_uid", "deid_sop_uid", "frame_index",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "text_token", "text_len", "confidence", "decision",
        ])
        writer.writeheader()
        writer.writerow({"source_path": "/in/IMG_1.dcm", "text_token": "TOK_SECRET_A",
                         "text_len": 8, "decision": "redacted"})
        writer.writerow({"source_path": "/in/IMG_1.dcm", "text_token": "TOK_SECRET_B",
                         "text_len": 6, "decision": "safelisted"})
        writer.writerow({"source_path": "/in/IMG_2.dcm", "text_token": "TOK_SECRET_C",
                         "text_len": 4, "decision": "redacted"})
    with open(run_dir / "aegis_tag_actions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_path", "original_sop_uid", "deid_sop_uid",
            "tag", "keyword", "action", "redacted",
        ])
        writer.writeheader()
        writer.writerow({"source_path": "/in/IMG_1.dcm", "tag": "(0010,0010)",
                         "keyword": "PatientName", "action": "DUMMY", "redacted": "True"})
        writer.writerow({"source_path": "/in/IMG_1.dcm", "tag": "(0008,0080)",
                         "keyword": "InstitutionName", "action": "REMOVE", "redacted": "True"})
        writer.writerow({"source_path": "/in/IMG_2.dcm", "tag": "(0010,0010)",
                         "keyword": "PatientName", "action": "DUMMY", "redacted": "True"})


def test_summarize_run_prefers_csvs_over_job_summary(tmp_path):
    _write_run_csvs(tmp_path)
    (tmp_path / "job_summary_ffffffff.json").write_text(json.dumps(
        {"per_file": {"OTHER.dcm": {"pixel_decisions": {"redacted": 99}, "header_tags_scrubbed": 99}}}
    ))
    response = server.summarize_run(str(tmp_path))
    assert response["status"] == "success"
    assert response["source"] == "ground_truth_csv"
    assert response["files_with_detections"] == 2
    assert response["regions_by_decision"] == {"redacted": 2, "safelisted": 1}
    assert response["tags_by_action"] == {"DUMMY": 2, "REMOVE": 1}
    _assert_phi_free(response)
    assert "TOK_SECRET_A" not in json.dumps(response)


def test_summarize_run_per_file_filter(tmp_path):
    _write_run_csvs(tmp_path)
    response = server.summarize_run(str(tmp_path), source_filename="IMG_1.dcm")
    assert response["status"] == "success"
    assert response["pixel_regions_total"] == 2
    assert response["regions_by_decision"] == {"redacted": 1, "safelisted": 1}
    assert {(a["keyword"], a["action"]) for a in response["header_audit"]} == {
        ("PatientName", "DUMMY"), ("InstitutionName", "REMOVE"),
    }
    assert response["header_audit_truncated"] is False
    _assert_phi_free(response)


def test_summarize_run_per_file_no_match(tmp_path):
    _write_run_csvs(tmp_path)
    response = server.summarize_run(str(tmp_path), source_filename="MISSING.dcm")
    assert response["status"] == "error"


def test_summarize_run_falls_back_to_job_summary(tmp_path):
    (tmp_path / "job_summary_aa000000.json").write_text(json.dumps({
        "per_file": {
            "IMG_1.dcm": {"pixel_decisions": {"redacted": 2, "safelisted": 0, "low_confidence": 0},
                          "header_tags_scrubbed": 5},
            "IMG_2.dcm": {"pixel_decisions": {"redacted": 0, "safelisted": 0, "low_confidence": 0},
                          "header_tags_scrubbed": 5},
        },
    }))
    response = server.summarize_run(str(tmp_path))
    assert response["status"] == "success"
    assert response["source"] == "job_summary_aa000000.json"
    assert response["files_with_detections"] == 1
    assert response["regions_by_decision"]["redacted"] == 2
    assert response["header_tags_scrubbed"] == 10

    per_file = server.summarize_run(str(tmp_path), source_filename="IMG_1.dcm")
    assert per_file["matched_files"] == ["IMG_1.dcm"]
    assert per_file["header_tags_scrubbed"] == 5


def test_summarize_run_empty_dir_errors(tmp_path):
    response = server.summarize_run(str(tmp_path))
    assert response["status"] == "error"
    response = server.summarize_run(str(tmp_path / "missing"))
    assert response["status"] == "error"


# ---------------------------------------------------------------------------
# deidentify_file / start_batch_job argument validation
# ---------------------------------------------------------------------------

def test_deidentify_file_not_found():
    response = server.deidentify_file("/definitely/not/a/file.dcm")
    assert response["status"] == "error"
    assert "FileNotFoundError" in response["message"]


def test_start_batch_job_bad_dir():
    response = server.start_batch_job("/definitely/not/a/dir")
    assert response["status"] == "error"
    assert "NotADirectoryError" in response["message"]


def test_exc_message_withholds_raw_message(monkeypatch):
    # An MCP tool response leaves the trust boundary (agent / LLM context), so
    # a raw exception message that may embed PHI must never be returned.
    monkeypatch.delenv("AEGIS_VERBOSE_ERRORS", raising=False)
    try:
        raise RuntimeError("failed on patient John Doe 1985-03-04")
    except RuntimeError as exc:
        rendered = server._exc_message(exc)
    assert rendered == "RuntimeError"
    assert "John Doe" not in rendered


def test_exc_message_verbose_opt_in(monkeypatch):
    monkeypatch.setenv("AEGIS_VERBOSE_ERRORS", "1")
    try:
        raise RuntimeError("local debugging detail")
    except RuntimeError as exc:
        rendered = server._exc_message(exc)
    assert rendered == "RuntimeError: local debugging detail"


def test_start_batch_job_registers_and_returns_next_step(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_jobs", {})
    started = []
    monkeypatch.setattr(server.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())
    response = server.start_batch_job(str(tmp_path))
    assert response["status"] == "started"
    assert response["job_id"] in server._jobs
    assert response["job_id"] in response["next_step"]
    assert "get_job_status" in response["next_step"]
    assert len(started) == 1


# ---------------------------------------------------------------------------
# list_review_queue
# ---------------------------------------------------------------------------

def test_list_review_queue_names_only_and_truncation(tmp_path, monkeypatch):
    review = tmp_path / "review"
    (review / "sub").mkdir(parents=True)
    for i in range(60):
        (review / f"file_{i:03d}.dcm").write_bytes(b"x")
    (review / "sub" / "nested.dcm").write_bytes(b"x")
    (review / ".DS_Store").write_bytes(b"x")
    monkeypatch.setattr(server, "REVIEW_DIR", review)

    response = server.list_review_queue()
    assert response["status"] == "success"
    assert response["count"] == 61
    assert len(response["files"]) == server.REVIEW_QUEUE_LIMIT
    assert response["truncated"] is True
    assert all(isinstance(name, str) and "/" not in name for name in response["files"])


def test_list_review_queue_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REVIEW_DIR", tmp_path / "nope")
    response = server.list_review_queue()
    assert response["status"] == "success"
    assert response["count"] == 0
    assert response["files"] == []


# ---------------------------------------------------------------------------
# _merge_ground_truth — single-file calls accumulate across a shared dir
# ---------------------------------------------------------------------------

@pytest.fixture
def merge_config(monkeypatch):
    monkeypatch.setattr(server, "_get_config",
                        lambda: {"reporting": {"save_ground_truth": True}})


def _csv_rows(path):
    import csv as _csv
    with open(path, newline="") as f:
        return list(_csv.DictReader(f))


def test_merge_accumulates_across_files(tmp_path, merge_config):
    out = str(tmp_path)
    server._merge_ground_truth(_fake_result(num_redacted=1, tags=1, source="/in/a.dcm"), out)
    server._merge_ground_truth(_fake_result(num_redacted=2, tags=2, source="/in/b.dcm"), out)

    pixel = _csv_rows(tmp_path / "aegis_pixel_detections.csv")
    tags = _csv_rows(tmp_path / "aegis_tag_actions.csv")
    assert {r["source_path"] for r in pixel} == {"/in/a.dcm", "/in/b.dcm"}
    assert len(pixel) == 1 + 1 + 2 + 1  # 1 redacted + 1 safelisted per call, +1 redacted on b
    assert len(tags) == 3
    # PHI-free on disk: tokenized text only (P2/P5)
    blob = (tmp_path / "aegis_pixel_detections.csv").read_text()
    assert "ocr_text" not in blob and PHI_TEXT not in blob


def test_merge_rerun_replaces_own_rows_only(tmp_path, merge_config):
    out = str(tmp_path)
    server._merge_ground_truth(_fake_result(num_redacted=1, tags=1, source="/in/a.dcm"), out)
    server._merge_ground_truth(_fake_result(num_redacted=3, tags=2, source="/in/b.dcm"), out)
    # Re-run a.dcm with different results: its rows are replaced, b's kept.
    server._merge_ground_truth(_fake_result(num_redacted=2, num_safelisted=0, tags=4,
                                            source="/in/a.dcm"), out)

    pixel = _csv_rows(tmp_path / "aegis_pixel_detections.csv")
    tags = _csv_rows(tmp_path / "aegis_tag_actions.csv")
    a_pixel = [r for r in pixel if r["source_path"] == "/in/a.dcm"]
    b_pixel = [r for r in pixel if r["source_path"] == "/in/b.dcm"]
    assert len(a_pixel) == 2 and all(r["decision"] == "redacted" for r in a_pixel)
    assert len(b_pixel) == 4  # 3 redacted + 1 safelisted, untouched
    assert sum(1 for r in tags if r["source_path"] == "/in/a.dcm") == 4
    assert sum(1 for r in tags if r["source_path"] == "/in/b.dcm") == 2


def test_merge_rerun_with_zero_rows_clears_stale(tmp_path, merge_config):
    out = str(tmp_path)
    server._merge_ground_truth(_fake_result(num_redacted=2, tags=2, source="/in/a.dcm"), out)
    empty = _fake_result(0, 0, 0, tags=0, source="/in/a.dcm")
    empty["image_redaction_stats"] = {"detections": []}
    server._merge_ground_truth(empty, out)
    assert _csv_rows(tmp_path / "aegis_pixel_detections.csv") == []


def test_merge_image_result_creates_no_tag_csv(tmp_path, merge_config):
    image_result = _fake_result(num_redacted=1, tags=0, source="/in/scan.png")
    del image_result["image_tag_actions"]
    server._merge_ground_truth(image_result, str(tmp_path))
    assert (tmp_path / "aegis_pixel_detections.csv").is_file()
    assert not (tmp_path / "aegis_tag_actions.csv").exists()

    # A later DICOM merge creates the tag CSV; a further image merge keeps it.
    server._merge_ground_truth(_fake_result(tags=2, source="/in/a.dcm"), str(tmp_path))
    server._merge_ground_truth(image_result, str(tmp_path))
    assert len(_csv_rows(tmp_path / "aegis_tag_actions.csv")) == 2


def test_merge_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_get_config",
                        lambda: {"reporting": {"save_ground_truth": False}})
    server._merge_ground_truth(_fake_result(), str(tmp_path))
    assert list(tmp_path.iterdir()) == []
