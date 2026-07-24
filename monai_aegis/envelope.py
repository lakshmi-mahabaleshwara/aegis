"""PHI-free result envelope for Aegis de-identification runs.

This module is the single place where a pipeline result dict is reduced to
the structured, machine-readable summary that every Aegis surface emits —
the ``aegis-deidentify`` CLI, the MCP server tools, and any third-party
skill wrapper. The envelope shape is versioned and validated by
``monai_aegis/schemas/envelope.schema.json`` (shipped with the package), so
external harnesses can gate on it without authoring their own schema.

Privacy invariants (normative — shared with the MCP server):
  P1  No envelope contains pixel data, image bytes, or base64 images.
  P2  No envelope contains OCR-extracted text. The ``ocr_text`` field of
      pipeline records and the ``ocr_text``/``text_token`` columns of the
      ground-truth CSVs are never read into an envelope.
  P3  Envelopes are limited to: file names and paths (not contents),
      counts, decision categories, timings, statuses, and error messages.
  P4  Error messages never echo DICOM tag values or OCR text.

Only ``decision`` is ever read from pixel rows and only ``redacted`` from
tag rows — the functions here are the audited implementation of those
invariants; surfaces must build summaries through this module rather than
reading pipeline records directly.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict, List, Optional

ENVELOPE_VERSION = "1.0"
TOOL_NAME = "monai-aegis"

# Canonical pixel-decision categories, in report order. Other modules
# (including the MCP server config) import this rather than redefining it.
PIXEL_DECISIONS = ("redacted", "safelisted", "low_confidence")

# Run statuses: every file succeeded / some failed / none succeeded (or the
# run itself failed before processing).
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_ERROR = "error"

SCHEMA_PACKAGE = "monai_aegis.schemas"
SCHEMA_FILENAME = "envelope.schema.json"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def summarize_result(result: Dict[Any, Any], key: str = "image") -> Dict[str, Any]:
    """Reduce one pipeline result dict to PHI-free counts.

    Only ``decision`` is read from pixel rows and only ``redacted`` from tag
    rows — never ``ocr_text`` (P2), never tag values (P4).
    """
    from monai_aegis import reporting

    pixel_rows, tag_rows = reporting.extract_records(result, key)
    decisions = {name: 0 for name in PIXEL_DECISIONS}
    for row in pixel_rows:
        decision = str(row.get("decision") or "")
        if decision:
            decisions[decision] = decisions.get(decision, 0) + 1
    tags_scrubbed = sum(1 for row in tag_rows if _is_true(row.get("redacted")))
    return {
        "pixel_regions_detected": len(pixel_rows),
        "pixel_decisions": decisions,
        "header_tags_scrubbed": tags_scrubbed,
        "needs_manual_review": decisions.get("low_confidence", 0) > 0,
    }


def collect_artifacts(result: Dict[Any, Any], key: str = "image") -> List[str]:
    """Return the output file paths recorded by the save transforms (P3: paths only)."""
    from monai_aegis.transforms import context_keys as ckeys
    from monai_aegis.transforms.context_keys import ck

    paths = result.get(ck(key, ckeys.SAVED_PATHS))
    if isinstance(paths, (list, tuple)):
        return sorted(str(p) for p in paths)
    path = result.get(ck(key, ckeys.SAVED_PATH))
    return [str(path)] if path else []


def file_entry(
    source_file: str,
    kind: str,
    summary: Dict[str, Any],
    artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the per-file envelope entry for a successfully processed file."""
    return {
        "source_file": source_file,
        "kind": kind,
        "status": "processed",
        "artifacts": list(artifacts or []),
        **summary,
    }


def failed_file_entry(source_file: str, kind: str, error: str) -> Dict[str, Any]:
    """Build the per-file envelope entry for a file that raised (P4: message only)."""
    return {
        "source_file": source_file,
        "kind": kind,
        "status": "failed",
        "error": error,
    }


def build_envelope(
    input_path: str,
    output_dir: str,
    mode: str,
    files: List[Dict[str, Any]],
    reports: Optional[List[str]] = None,
    elapsed_seconds: Optional[float] = None,
    message: str = "",
) -> Dict[str, Any]:
    """Assemble the versioned run envelope from per-file entries.

    Status derivation: ``success`` when every discovered file processed,
    ``partial`` when some failed, ``error`` when files were discovered but
    none processed. A run that discovered nothing is a ``success`` with a
    message — an empty directory is not a failure.
    """
    processed = [f for f in files if f.get("status") == "processed"]
    failed = [f for f in files if f.get("status") == "failed"]

    totals = {
        "files_discovered": len(files),
        "files_processed": len(processed),
        "files_failed": len(failed),
        "pixel_regions_detected": sum(f.get("pixel_regions_detected", 0) for f in processed),
        "pixel_decisions": {
            name: sum(f.get("pixel_decisions", {}).get(name, 0) for f in processed) for name in PIXEL_DECISIONS
        },
        "header_tags_scrubbed": sum(f.get("header_tags_scrubbed", 0) for f in processed),
    }

    if failed and not processed:
        status = STATUS_ERROR
    elif failed:
        status = STATUS_PARTIAL
    else:
        status = STATUS_SUCCESS

    envelope: Dict[str, Any] = {
        "envelope_version": ENVELOPE_VERSION,
        "tool": TOOL_NAME,
        "status": status,
        "mode": mode,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "needs_manual_review": any(f.get("needs_manual_review") for f in processed),
        "totals": totals,
        "files": files,
        "reports": sorted(reports or []),
        "errors": [f"{f['source_file']}: {f['error']}" for f in failed],
    }
    if elapsed_seconds is not None:
        envelope["elapsed_seconds"] = round(elapsed_seconds, 1)
    if message:
        envelope["message"] = message
    return envelope


def error_envelope(message: str, input_path: str = "", output_dir: str = "") -> Dict[str, Any]:
    """Build the envelope for a run that failed before any file was processed."""
    return {
        "envelope_version": ENVELOPE_VERSION,
        "tool": TOOL_NAME,
        "status": STATUS_ERROR,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "message": message,
        "errors": [message],
    }


def schema() -> Dict[str, Any]:
    """Load the packaged envelope JSON Schema."""
    text = resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILENAME).read_text(encoding="utf-8")
    return json.loads(text)


def validate_envelope(envelope: Dict[str, Any]) -> None:
    """Validate an envelope against the packaged schema.

    Requires the optional ``jsonschema`` package (a dev/test dependency);
    raises ``ImportError`` with guidance when it is missing, and
    ``jsonschema.ValidationError`` when the envelope does not conform.
    """
    try:
        import jsonschema
    except ImportError as exc:
        raise ImportError(
            "envelope validation requires the 'jsonschema' package — " "install with: pip install 'monai-aegis[dev]'"
        ) from exc
    jsonschema.validate(instance=envelope, schema=schema())


__all__ = [
    "ENVELOPE_VERSION",
    "TOOL_NAME",
    "PIXEL_DECISIONS",
    "STATUS_SUCCESS",
    "STATUS_PARTIAL",
    "STATUS_ERROR",
    "summarize_result",
    "collect_artifacts",
    "file_entry",
    "failed_file_entry",
    "build_envelope",
    "error_envelope",
    "schema",
    "validate_envelope",
]
