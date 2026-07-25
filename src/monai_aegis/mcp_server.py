"""aegis-mcp — MCP server exposing the Aegis de-identification pipeline.

Exposes the ``monai_aegis`` medical-image de-identification pipeline as six
MCP tools so an orchestrating model (Claude Desktop, MCP Inspector, or a
local Ollama model behind mcpo/Open WebUI) can run, monitor, and audit
de-identification workflows through natural language — without any PHI ever
entering a model context window.

Privacy invariants (normative):
  P1  No tool response contains pixel data, image bytes, or base64 images.
  P2  No tool response contains OCR-extracted text. The ``ocr_text`` field
      of pipeline records and the ``ocr_text``/``text_token`` columns of the
      ground-truth CSVs are never read into a response.
  P3  Responses are limited to: file names (not contents), directory paths,
      counts, decision categories, bounding-box coordinates, DICOM tag
      ids/keywords/actions, timings, job states, and error messages.
  P4  Error messages never echo DICOM tag values or OCR text.
  P5  ``job_summary_*.json`` artifacts written here satisfy P1–P3.

Transport: stdio only. stdout is reserved exclusively for JSON-RPC frames;
``main()`` hardens this by dup'ing the real stdout for the protocol and
pointing fd 1 at stderr, so stray ``print()`` calls anywhere in the process
(including native libraries) cannot corrupt framing. All logging goes to
stderr.

Run directly (``python aegis_mcp_server.py``) or via an MCP client config;
see the README for Claude Desktop / MCP Inspector / mcpo examples.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from monai_aegis.config.mcp_server_config import (
    AEGIS_ROOT,
    BATCH_MODES,
    CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    HEADER_AUDIT_LIMIT,
    IMAGE_EXTENSIONS,
    JOB_ERRORS_LIMIT,
    PIXEL_DECISIONS,
    REVIEW_DIR,
    REVIEW_QUEUE_LIMIT,
    mcp,
)

logger = logging.getLogger("aegis-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline execution — single dedicated thread
#
# monai_aegis transforms hold their EasyOCR/NER models in threading.local(),
# so every thread that touches a pipeline loads its own model copies. Routing
# every pipeline call through one dedicated thread means one resident model
# set, warm_up warms the same instances every later call
# uses, and concurrent batch jobs are serialized per-file without relying on
# the transforms being concurrently reentrant.
# ---------------------------------------------------------------------------

_pipeline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-pipeline")
_pipeline_cache: Dict[Tuple[str, str, str], Any] = {}
_pipeline_lock = threading.Lock()

_config: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()


def _run_on_pipeline_thread(fn: Callable, *args: Any) -> Any:
    """Run *fn* on the dedicated pipeline thread and return its result."""
    return _pipeline_executor.submit(fn, *args).result()


def _get_config() -> Dict[str, Any]:
    global _config
    with _config_lock:
        if _config is None:
            from monai_aegis.config.config_loader import load_config

            _config = load_config(str(CONFIG_PATH))
        return _config


def _get_pipeline(kind: str, input_dir: str, output_dir: str) -> Any:
    """Build (or fetch cached) pipeline for a (kind, input, output) key.

    Must be called on the pipeline thread — building a pipeline may trigger
    model loading via the transforms' lazy per-thread initializers.
    """
    key = (kind, input_dir, output_dir)
    with _pipeline_lock:
        if key not in _pipeline_cache:
            from monai_aegis.transforms.pipeline import build_image_pipeline, build_pipeline

            if kind == "image":
                _pipeline_cache[key] = build_image_pipeline(
                    config_path=str(CONFIG_PATH), output_dir=output_dir, input_dir=input_dir
                )
            else:
                _pipeline_cache[key] = build_pipeline(
                    config_path=str(CONFIG_PATH), output_dir=output_dir, input_dir=input_dir
                )
        return _pipeline_cache[key]


def _process_one(kind: str, path: str, input_dir: str, output_dir: str) -> Dict[Any, Any]:
    """Run one file through the pipeline (pipeline thread only)."""
    pipeline = _get_pipeline(kind, input_dir, output_dir)
    return pipeline({"image": path})


def _warm_default_pipeline() -> None:
    """Build the default pipeline and force OCR + NER model loading.

    The transforms load their models lazily on first use, so merely building
    the Compose is not enough — touch the lazy properties of the pixel
    transform. Runs on the pipeline thread so the loaded instances are the
    ones every subsequent tool call uses.
    """
    pipeline = _get_pipeline("dicom", "", str(DEFAULT_OUTPUT_DIR))
    for transform in getattr(pipeline, "transforms", []):
        inner = getattr(transform, "transform", None)
        if inner is None:
            continue
        if hasattr(type(inner), "reader"):
            _ = inner.reader
        if hasattr(type(inner), "ner_classifier"):
            _ = inner.ner_classifier


# ---------------------------------------------------------------------------
# PHI-safe result summarization (P1–P3)
# ---------------------------------------------------------------------------

def _summarize_result(result: Dict[Any, Any]) -> Dict[str, Any]:
    """Reduce one pipeline result dict to PHI-free counts.

    Delegates to :mod:`monai_aegis.envelope` — the audited shared
    implementation of the P1–P4 invariants used by every Aegis surface.
    """
    from monai_aegis import envelope

    return envelope.summarize_result(result)


_ground_truth_lock = threading.Lock()


def _merge_ground_truth(result: Dict[Any, Any], out_dir: str) -> None:
    """Merge one file's ground-truth rows into *out_dir*'s CSV reports.

    The batch path flushes a GroundTruthAccumulator once into a fresh per-job
    directory, so plain overwrite is correct there. Single-file calls share an
    output directory across calls, so their rows are merged instead: existing
    rows for the same source file are replaced (a re-run refreshes that
    file's audit trail) and rows from other files are preserved. Pixel rows
    are tokenized before writing, exactly as the accumulator does. No-op when
    ``reporting.save_ground_truth`` is off.
    """
    from monai_aegis import reporting
    from monai_aegis.config.config_loader import as_bool
    from monai_aegis.transforms import context_keys as ckeys
    from monai_aegis.transforms.context_keys import ck
    from monai_aegis.transforms.utility import AegisIdentityManager

    config = _get_config()
    if not reporting.is_enabled(config):
        return

    pixel_rows, tag_rows = reporting.extract_records(result)
    safe_pixel_rows = reporting.sanitize_pixel_rows(
        pixel_rows, AegisIdentityManager.from_config(config)
    )
    # Replace by source file — derived from the result meta as well as the
    # rows, so a re-run that now yields zero rows still clears stale ones.
    meta = result.get(ck("image", ckeys.META_DICT), {}) or {}
    source = meta.get("filename_or_obj", "")
    source = source[0] if isinstance(source, (list, tuple)) and source else source
    sources = {str(source)} | {
        str(row.get("source_path")) for row in safe_pixel_rows + tag_rows
    }

    def merge(csv_path: Path, new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept = [r for r in _read_csv_rows(csv_path) if str(r.get("source_path")) not in sources]
        return kept + new_rows

    out = Path(out_dir)
    with _ground_truth_lock:
        merged_pixel = merge(out / reporting.PIXEL_DETECTIONS_FILE, safe_pixel_rows)
        # Image files have no DICOM tags: don't create the tag-actions CSV for
        # them, but preserve it once a DICOM run has produced one.
        tag_csv = out / reporting.TAG_ACTIONS_FILE
        merged_tags = merge(tag_csv, tag_rows) if (tag_rows or tag_csv.is_file()) else None
        reporting.write_reports(merged_pixel, merged_tags, out_dir)

        reporting_cfg = config.get("reporting", {}) or {}
        if as_bool(reporting_cfg.get("include_phi_text"), default=False):
            phi_dir = str(reporting_cfg.get("phi_report_dir") or "")
            existing_phi = (
                merge(Path(phi_dir) / reporting.PIXEL_DETECTIONS_PHI_FILE, [])
                if phi_dir else []
            )
            reporting.write_phi_report(existing_phi + pixel_rows, phi_dir, out_dir)


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

def _error(message: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "error", "message": message, **extra}


def _exc_message(exc: BaseException) -> str:
    # PHI-safe error code for tool responses: an MCP response leaves the
    # trust boundary (it enters an agent's context / an LLM provider), so a
    # raw exception message — which may carry tag values or OCR text — must
    # never be returned. Full detail stays in the server-side logs. Honors
    # the AEGIS_VERBOSE_ERRORS opt-in via envelope.safe_error.
    from monai_aegis import envelope

    logger.exception("tool error: %s", type(exc).__name__)
    return envelope.safe_error(exc)


def _resolve_dir(value: str, default: Path) -> Path:
    return (Path(value).expanduser() if value else default).resolve()


# ---------------------------------------------------------------------------
# Batch jobs
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _has_dicm_magic(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _fallback_discover(root: str) -> List[str]:
    """Content-driven DICOM discovery used if monai_aegis discovery is unavailable."""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            if name.lower().endswith(".dcm") or _has_dicm_magic(path):
                found.append(path)
    return sorted(found)


def _discover_dicom_files(input_dir: str) -> List[str]:
    try:
        from monai_aegis.api import discover_dicom_files
    except ImportError:
        logger.warning("monai_aegis discovery unavailable; using fallback scan")
        return _fallback_discover(input_dir)
    return discover_dicom_files(input_dir)


def _fallback_discover_images(root: str) -> List[str]:
    """Extension-driven image discovery, matching the image runner's scan."""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def _discover_image_files(input_dir: str) -> List[str]:
    try:
        from monai_aegis.api import discover_image_files
    except ImportError:
        logger.warning("monai_aegis discovery unavailable; using fallback scan")
        return _fallback_discover_images(input_dir)
    return discover_image_files(input_dir)


def _discover_batch_files(input_dir: str, mode: str) -> List[Tuple[str, str]]:
    """Discover files to process as ``(path, kind)`` pairs for a batch mode.

    Mirrors the unified CLI's mode semantics: 'dicom' is content-driven
    (DICM magic + imaging modality), 'image' is extension-driven
    (.jpg/.jpeg/.png), 'auto' is both — DICOM first, as the CLI runs them.
    """
    files: List[Tuple[str, str]] = []
    if mode in ("auto", "dicom"):
        files.extend((path, "dicom") for path in _discover_dicom_files(input_dir))
    if mode in ("auto", "image"):
        files.extend((path, "image") for path in _discover_image_files(input_dir))
    return files


def _write_job_summary(job: Dict[str, Any], per_file: Dict[str, Dict[str, Any]]) -> str:
    """Write the agent-readable job summary JSON (P1–P3 compliant: counts only)."""
    summary = {
        "job_id": job["job_id"],
        "mode": job.get("mode", "auto"),
        "input_dir": job["input_dir"],
        "output_dir": job["output_dir"],
        "totals": {
            "files": job["total"],
            "decisions": job["decisions"],
            "header_tags_scrubbed": job["header_tags_scrubbed"],
            "errors": len(job["errors"]),
        },
        "per_file": per_file,
    }
    os.makedirs(job["output_dir"], exist_ok=True)
    path = os.path.join(job["output_dir"], f"job_summary_{job['job_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def _run_batch_job(job: Dict[str, Any]) -> None:
    """Batch worker (daemon thread). Per-file errors never fail the job."""
    from monai_aegis import reporting

    job_id = job["job_id"]
    mode = job.get("mode", "auto")
    started = time.monotonic()
    try:
        job["state"] = "running"
        files = _discover_batch_files(job["input_dir"], mode)
        job["total"] = len(files)
        logger.info("[job %s] discovered %d files (mode=%s) in %s",
                    job_id, len(files), mode, job["input_dir"])

        # Match the runners' reporting convention: the image runner never
        # writes a tag-actions CSV (JPEG/PNG have no DICOM tags), the DICOM
        # runner always does.
        has_dicom = any(kind == "dicom" for _path, kind in files)
        accumulator = reporting.GroundTruthAccumulator(
            _get_config(), include_tag_actions=has_dicom
        )
        per_file: Dict[str, Dict[str, Any]] = {}

        for path, kind in files:
            name = os.path.basename(path)
            try:
                result = _run_on_pipeline_thread(
                    _process_one, kind, path, job["input_dir"], job["output_dir"]
                )
                accumulator.collect(result)
                summary = _summarize_result(result)
                for decision, count in summary["pixel_decisions"].items():
                    job["decisions"][decision] = job["decisions"].get(decision, 0) + count
                job["header_tags_scrubbed"] += summary["header_tags_scrubbed"]
                per_file[name] = {
                    "pixel_decisions": summary["pixel_decisions"],
                    "header_tags_scrubbed": summary["header_tags_scrubbed"],
                }
                logger.info("[job %s] processed %s", job_id, name)
            except Exception as exc:  # per-file: record and continue
                logger.exception("[job %s] failed on %s", job_id, name)
                job["errors"].append(f"{name}: {_exc_message(exc)}")
            finally:
                job["processed"] += 1

        if files:
            accumulator.flush(job["output_dir"])
        job["summary_file"] = _write_job_summary(job, per_file)
        job["elapsed_seconds"] = round(time.monotonic() - started, 1)
        if not files:
            wanted = {"auto": "DICOM or image", "dicom": "DICOM", "image": "image (.jpg/.jpeg/.png)"}
            job["message"] = f"No {wanted.get(mode, mode)} files found in input_dir."
        job["state"] = "completed"
        logger.info("[job %s] completed: %d/%d files, %d errors",
                    job_id, job["processed"], job["total"], len(job["errors"]))
    except Exception as exc:  # job-level failure (e.g. discovery)
        logger.exception("[job %s] job failed", job_id)
        job["elapsed_seconds"] = round(time.monotonic() - started, 1)
        job["message"] = _exc_message(exc)
        job["state"] = "failed"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def warm_up() -> dict:
    """Preload the Aegis OCR and NER models so later calls are fast.

    Call this once before de-identifying files. Takes seconds when model
    weights are already cached; the first-ever run may take minutes because
    the NER model (~500 MB) is downloaded. Safe to call repeatedly.

    Returns JSON: {"status": "success", "elapsed_seconds": <float>}.
    Never returns image data or extracted text.
    """
    started = time.monotonic()
    try:
        _run_on_pipeline_thread(_warm_default_pipeline)
    except Exception as exc:
        logger.exception("warm_up failed")
        return _error(_exc_message(exc))
    return {"status": "success", "elapsed_seconds": round(time.monotonic() - started, 1)}


@mcp.tool()
def deidentify_file(input_path: str, output_dir: str = "") -> dict:
    """De-identify a single medical image file (DICOM, JPEG, or PNG).

    Removes burned-in PHI text from pixels (OCR + NER) and scrubs DICOM
    header tags, writing the cleaned file to output_dir (default:
    staging_output/mcp). Synchronous — suitable for one file at a time; for
    whole directories or large multi-frame series use start_batch_job
    instead. Call warm_up first so models are loaded.

    Returns JSON counts only: pixel regions detected, decisions per category
    (redacted / safelisted / low_confidence), header tags scrubbed, timing,
    and needs_manual_review (true when low-confidence regions were found).
    Never returns image data, OCR text, or DICOM tag values.
    """
    try:
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            return _error(f"FileNotFoundError: no such file: {source}")
        out_dir = _resolve_dir(output_dir, DEFAULT_OUTPUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        kind = "image" if source.suffix.lower() in IMAGE_EXTENSIONS else "dicom"

        started = time.monotonic()
        result = _run_on_pipeline_thread(
            _process_one, kind, str(source), str(source.parent), str(out_dir)
        )
        elapsed = round(time.monotonic() - started, 1)

        # Ground-truth CSVs for auditability (G3). PHI-free by construction:
        # reporting tokenizes detected text before writing. Single-file calls
        # share an output directory, so rows are merged rather than the CSVs
        # overwritten — re-running a file replaces only that file's rows.
        _merge_ground_truth(result, str(out_dir))

        summary = _summarize_result(result)
        return {
            "status": "success",
            "source_file": source.name,
            "output_dir": str(out_dir),
            "elapsed_seconds": elapsed,
            **summary,
        }
    except Exception as exc:
        logger.exception("deidentify_file failed for %s", input_path)
        return _error(_exc_message(exc))


@mcp.tool()
def start_batch_job(input_dir: str, output_dir: str = "", mode: str = "auto") -> dict:
    """Start a background job that de-identifies every medical file in a directory.

    Scans input_dir recursively and processes each file without blocking this
    call — the tool returns immediately with a job_id. Poll progress with
    get_job_status(job_id). mode selects what to process: "auto" (default)
    handles both DICOM and standard images, "dicom" only DICOM files
    (content-driven detection), "image" only JPEG/PNG files. Output goes to
    output_dir (default: staging_output/mcp/<job_id>/), together with
    ground-truth CSVs and a job_summary JSON for auditing via summarize_run.
    Individual file failures do not stop the job.

    Returns JSON: {"status": "started", "job_id": ..., "output_dir": ...,
    "mode": ..., "next_step": ...}. Never returns image data or extracted text.
    """
    try:
        in_dir = Path(input_dir).expanduser().resolve()
        if not in_dir.is_dir():
            return _error(f"NotADirectoryError: no such directory: {in_dir}")
        mode = (mode or "auto").strip().lower()
        if mode not in BATCH_MODES:
            return _error(
                f"ValueError: unknown mode {mode!r}. Use one of: {', '.join(BATCH_MODES)}."
            )

        job_id = uuid.uuid4().hex[:8]
        out_dir = _resolve_dir(output_dir, DEFAULT_OUTPUT_DIR / job_id)
        job: Dict[str, Any] = {
            "job_id": job_id,
            "state": "queued",
            "mode": mode,
            "input_dir": str(in_dir),
            "output_dir": str(out_dir),
            "total": 0,
            "processed": 0,
            "decisions": {name: 0 for name in PIXEL_DECISIONS},
            "header_tags_scrubbed": 0,
            "errors": [],
        }
        with _jobs_lock:
            _jobs[job_id] = job
        threading.Thread(
            target=_run_batch_job, args=(job,), daemon=True, name=f"aegis-job-{job_id}"
        ).start()
        return {
            "status": "started",
            "job_id": job_id,
            "output_dir": str(out_dir),
            "mode": mode,
            "next_step": (
                f'The job is running in the background. Call get_job_status '
                f'with job_id="{job_id}" to check progress. When state is '
                f'"completed", call summarize_run with run_dir="{out_dir}" to audit results.'
            ),
        }
    except Exception as exc:
        logger.exception("start_batch_job failed for %s", input_dir)
        return _error(_exc_message(exc))


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Check the progress of a batch de-identification job.

    Returns a JSON snapshot: state (queued / running / completed / failed),
    total and processed file counts, pixel decision counts, header tags
    scrubbed, and up to 10 error messages. When completed, includes
    elapsed_seconds and the job summary file path. Never returns image data
    or extracted text.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        known = sorted(_jobs)
    if job is None:
        return _error(
            f"KeyError: unknown job_id {job_id!r}. Job ids are returned by start_batch_job.",
            known_jobs=known,
        )
    snapshot = {
        "status": "success",
        "job_id": job["job_id"],
        "state": job["state"],
        "mode": job.get("mode", "auto"),
        "input_dir": job["input_dir"],
        "output_dir": job["output_dir"],
        "total": job["total"],
        "processed": job["processed"],
        "decisions": dict(job["decisions"]),
        "header_tags_scrubbed": job["header_tags_scrubbed"],
        "errors": list(job["errors"][:JOB_ERRORS_LIMIT]),
        "error_count": len(job["errors"]),
    }
    for key in ("elapsed_seconds", "summary_file", "message"):
        if key in job:
            snapshot[key] = job[key]
    return snapshot


@mcp.tool()
def summarize_run(run_dir: str, source_filename: str = "") -> dict:
    """Audit a completed de-identification run directory.

    Reads the run's ground-truth reports — aegis_pixel_detections.csv and
    aegis_tag_actions.csv when present, else the newest job_summary JSON —
    and returns aggregate counts: files with detections, pixel regions by
    decision, and header tags by action. Pass source_filename to audit one
    original file (suffix match on its recorded source path); per-file mode
    additionally lists header keyword→action pairs (up to 30). Never returns
    OCR text, text tokens, or DICOM tag values.
    """
    try:
        run = Path(run_dir).expanduser().resolve()
        if not run.is_dir():
            return _error(f"NotADirectoryError: no such directory: {run}")

        pixel_csv = run / "aegis_pixel_detections.csv"
        tag_csv = run / "aegis_tag_actions.csv"
        if pixel_csv.is_file() or tag_csv.is_file():
            return _summarize_from_csvs(run, pixel_csv, tag_csv, source_filename)
        return _summarize_from_job_summary(run, source_filename)
    except Exception as exc:
        logger.exception("summarize_run failed for %s", run_dir)
        return _error(_exc_message(exc))


@mcp.tool()
def list_review_queue() -> dict:
    """List files quarantined for manual review (low-confidence detections).

    Returns the number of files in the review directory
    (staging_not_processed) and up to 50 file names — names only, never file
    contents or extracted text.
    """
    try:
        names: List[str] = []
        if REVIEW_DIR.is_dir():
            for dirpath, _dirs, files in os.walk(REVIEW_DIR):
                names.extend(n for n in files if not n.startswith("."))
        names.sort()
        return {
            "status": "success",
            "count": len(names),
            "files": names[:REVIEW_QUEUE_LIMIT],
            "truncated": len(names) > REVIEW_QUEUE_LIMIT,
            "review_dir": str(REVIEW_DIR),
        }
    except Exception as exc:
        logger.exception("list_review_queue failed")
        return _error(_exc_message(exc))


# ---------------------------------------------------------------------------
# summarize_run helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summarize_from_csvs(
    run: Path, pixel_csv: Path, tag_csv: Path, source_filename: str
) -> Dict[str, Any]:
    # P2: only source_path / decision / keyword / action / redacted columns
    # are ever read — never ocr_text or text_token.
    pixel_rows = _read_csv_rows(pixel_csv)
    tag_rows = _read_csv_rows(tag_csv)

    if source_filename:
        pixel_rows = [r for r in pixel_rows if str(r.get("source_path", "")).endswith(source_filename)]
        tag_rows = [r for r in tag_rows if str(r.get("source_path", "")).endswith(source_filename)]
        if not pixel_rows and not tag_rows:
            return _error(
                f"FileNotFoundError: no records for a source file matching "
                f"{source_filename!r} in {run}. Pass the original file name as recorded at "
                f"processing time."
            )

    decisions: Dict[str, int] = {}
    for row in pixel_rows:
        decision = str(row.get("decision") or "")
        if decision:
            decisions[decision] = decisions.get(decision, 0) + 1
    actions: Dict[str, int] = {}
    for row in tag_rows:
        action = str(row.get("action") or "")
        if action:
            actions[action] = actions.get(action, 0) + 1

    summary: Dict[str, Any] = {
        "status": "success",
        "run_dir": str(run),
        "source": "ground_truth_csv",
        "files_with_detections": len({r.get("source_path", "") for r in pixel_rows if r.get("source_path")}),
        "pixel_regions_total": len(pixel_rows),
        "regions_by_decision": decisions,
        "tags_by_action": actions,
    }
    if source_filename:
        audit = [
            {"keyword": str(r.get("keyword") or ""), "action": str(r.get("action") or "")}
            for r in tag_rows
        ]
        summary["source_filename"] = source_filename
        summary["header_audit"] = audit[:HEADER_AUDIT_LIMIT]
        summary["header_audit_truncated"] = len(audit) > HEADER_AUDIT_LIMIT
    return summary


def _summarize_from_job_summary(run: Path, source_filename: str) -> Dict[str, Any]:
    candidates = sorted(run.glob("job_summary_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return _error(
            f"FileNotFoundError: no ground-truth CSVs or job_summary JSON found in {run}. "
            f"Is this a completed Aegis run directory?"
        )
    newest = candidates[-1]
    with open(newest, encoding="utf-8") as f:
        data = json.load(f)

    per_file: Dict[str, Dict[str, Any]] = data.get("per_file", {}) or {}
    if source_filename:
        matches = {name: v for name, v in per_file.items() if name.endswith(source_filename)}
        if not matches:
            return _error(
                f"FileNotFoundError: no per-file entry matching {source_filename!r} "
                f"in {newest.name}."
            )
        per_file = matches

    decisions: Dict[str, int] = {}
    tags_scrubbed = 0
    files_with_detections = 0
    for entry in per_file.values():
        entry_decisions = entry.get("pixel_decisions", {}) or {}
        if sum(entry_decisions.values()) > 0:
            files_with_detections += 1
        for decision, count in entry_decisions.items():
            decisions[decision] = decisions.get(decision, 0) + int(count)
        tags_scrubbed += int(entry.get("header_tags_scrubbed", 0))

    summary: Dict[str, Any] = {
        "status": "success",
        "run_dir": str(run),
        "source": newest.name,
        "files_with_detections": files_with_detections,
        "pixel_regions_total": sum(decisions.values()),
        "regions_by_decision": decisions,
        "header_tags_scrubbed": tags_scrubbed,
    }
    if source_filename:
        summary["source_filename"] = source_filename
        summary["matched_files"] = sorted(per_file)
    return summary


# ---------------------------------------------------------------------------
# Entry point (stdout is protocol-only)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="[aegis-mcp] %(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting aegis-mcp | root=%s | config=%s", AEGIS_ROOT, CONFIG_PATH)
    if not CONFIG_PATH.is_file():
        logger.warning(
            "Config file not found: %s — tools will return structured errors "
            "until it exists (set AEGIS_ROOT to the Aegis repo root).",
            CONFIG_PATH,
        )

    # Reserve the real stdout for JSON-RPC, then point fd 1 at stderr so any
    # stray write to stdout — a print() in a dependency, a native library's
    # progress bar — lands in the log instead of corrupting protocol frames.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    protocol_binary = os.fdopen(protocol_fd, "wb")

    import anyio
    from mcp.server.stdio import stdio_server

    async def _serve() -> None:
        protocol_stream = anyio.wrap_file(io.TextIOWrapper(protocol_binary, encoding="utf-8"))
        async with stdio_server(stdout=protocol_stream) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    anyio.run(_serve)


if __name__ == "__main__":
    main()
