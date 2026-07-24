"""Declarative verification of de-identified Aegis output.

A generic assertion engine driven by a YAML checklist: the engine knows a
small vocabulary of check types (header-tag assertions, private-tag scans,
file-meta consistency, report-file assertions); *which* checks run — and
with what severity — is data, not code. The packaged default,
``monai_aegis/checklists/ps315.yaml``, encodes the guarantees of the
default ``pii_mapping``; deployments with different mappings supply their
own checklist file and never touch Python.

Checklists are validated at load time — an unknown check type, a missing
parameter, or an unresolvable tag fails immediately with a clear message,
mirroring the fail-fast contract of ``pii_mapping``.

The verification report is PHI-free by construction: finding details are
built from static text, tag ids/keywords, file names, and counts — never
from element values or report cell contents. Its shape is versioned and
validated by ``monai_aegis/schemas/verification.schema.json``.

This module depends only on pydicom and PyYAML, so verification can run in
environments where the OCR/NER models are not installed — auditing a run
does not require being able to produce one.
"""

from __future__ import annotations

import csv
import logging
import re
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pydicom
import yaml

from monai_aegis.api import InputError

logger = logging.getLogger(__name__)

REPORT_VERSION = "1.0"
TOOL_NAME = "monai-aegis"

CHECKLIST_PACKAGE = "monai_aegis.checklists"
DEFAULT_CHECKLIST_FILENAME = "ps315.yaml"

SEVERITIES = ("error", "warning")

STATUS_PASS = "pass"
STATUS_FAIL = "fail"

# Check vocabulary: type name → required parameter names. Optional
# parameters (severity, id, description, allow_absent) are shared.
DICOM_CHECK_TYPES: Dict[str, set] = {
    "tag_present": {"tag"},
    "tag_absent": {"tag"},
    "tag_equals": {"tag", "value"},
    "tag_matches": {"tag", "pattern"},
    "no_private_tags": set(),
    "file_meta_consistent": set(),
}
RUN_CHECK_TYPES: Dict[str, set] = {
    "report_present": {"file"},
    "report_columns_forbidden": {"file", "columns"},
    "report_columns_required": {"file", "columns"},
    "outputs_accounted_in_report": {"file", "column"},
}

_TAG_PATTERN = re.compile(r"\(?\s*([0-9a-fA-F]{4})\s*,\s*([0-9a-fA-F]{4})\s*\)?")


def default_checklist_path() -> str:
    """Path of the packaged default checklist."""
    return str(resources.files(CHECKLIST_PACKAGE).joinpath(DEFAULT_CHECKLIST_FILENAME))


def _resolve_tag(spec: str) -> pydicom.tag.BaseTag:
    """Resolve '(gggg,eeee)' or a DICOM keyword to a Tag; raise on neither."""
    match = _TAG_PATTERN.fullmatch(str(spec).strip())
    if match:
        return pydicom.tag.Tag(int(match.group(1), 16), int(match.group(2), 16))
    tag = pydicom.datadict.tag_for_keyword(str(spec).strip())
    if tag is None:
        raise ValueError(f"unresolvable DICOM tag or keyword: {spec!r}")
    return pydicom.tag.Tag(tag)


def _validate_check(check: Any, known_types: Dict[str, set], section: str) -> Dict[str, Any]:
    """Normalize and validate one checklist entry; raise ValueError if invalid."""
    if not isinstance(check, dict):
        raise ValueError(f"{section}: each check must be a mapping, got {type(check).__name__}")
    check_type = check.get("type")
    if check_type not in known_types:
        raise ValueError(
            f"{section}: unknown check type {check_type!r}. " f"Known types: {', '.join(sorted(known_types))}."
        )
    missing = known_types[check_type] - set(check)
    if missing:
        raise ValueError(f"{section}: check {check_type!r} missing parameters: {sorted(missing)}")

    normalized = dict(check)
    severity = normalized.setdefault("severity", "error")
    if severity not in SEVERITIES:
        raise ValueError(f"{section}: invalid severity {severity!r}. Use one of: {SEVERITIES}.")
    if "tag" in known_types[check_type]:
        normalized["_tag"] = _resolve_tag(normalized["tag"])
    if check_type == "tag_matches":
        try:
            normalized["_pattern"] = re.compile(str(normalized["pattern"]))
        except re.error as exc:
            raise ValueError(f"{section}: invalid pattern {normalized['pattern']!r}: {exc}") from exc
    if "columns" in known_types[check_type] and not isinstance(normalized["columns"], list):
        raise ValueError(f"{section}: 'columns' must be a list for check {check_type!r}")
    normalized.setdefault(
        "id",
        check_type
        + (
            f":{normalized.get('tag', normalized.get('file', ''))}"
            if normalized.get("tag") or normalized.get("file")
            else ""
        ),
    )
    return normalized


def load_checklist(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate a checklist YAML (default: the packaged PS3.15 one).

    Returns the normalized checklist dict. Raises ``ValueError`` on any
    structural problem — before any file is verified.
    """
    checklist_path = path or default_checklist_path()
    with open(checklist_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not data.get("name"):
        raise ValueError(f"checklist {checklist_path}: must be a mapping with a 'name'")
    dicom_checks = [_validate_check(c, DICOM_CHECK_TYPES, "dicom_checks") for c in (data.get("dicom_checks") or [])]
    run_checks = [_validate_check(c, RUN_CHECK_TYPES, "run_checks") for c in (data.get("run_checks") or [])]
    if not dicom_checks and not run_checks:
        raise ValueError(f"checklist {checklist_path}: defines no checks")
    return {
        "name": str(data["name"]),
        "description": str(data.get("description", "")),
        "dicom_checks": dicom_checks,
        "run_checks": run_checks,
    }


# ---------------------------------------------------------------------------
# Per-dataset checks (details never echo element values)
# ---------------------------------------------------------------------------


def _check_dataset(ds: pydicom.Dataset, check: Dict[str, Any]) -> Tuple[bool, str]:
    check_type = check["type"]
    if check_type == "no_private_tags":
        count = sum(1 for elem in ds.iterall() if elem.tag.is_private)
        return count == 0, f"{count} private tag(s) present"
    if check_type == "file_meta_consistent":
        meta = getattr(ds, "file_meta", None)
        if meta is None:
            return False, "file_meta is missing"
        if str(meta.get("MediaStorageSOPInstanceUID", "")) != str(getattr(ds, "SOPInstanceUID", "")):
            return False, "MediaStorageSOPInstanceUID does not match SOPInstanceUID"
        return True, ""

    tag = check["_tag"]
    label = f"tag {check['tag']}"
    present = tag in ds
    if check_type == "tag_present":
        return present, f"{label} is absent"
    if check_type == "tag_absent":
        return not present, f"{label} is present"
    if not present:
        allowed = bool(check.get("allow_absent"))
        return allowed, f"{label} is absent"
    value = str(ds[tag].value if ds[tag].value is not None else "")
    if check_type == "tag_equals":
        return value == str(check["value"]), f"{label} does not equal {check['value']!r}"
    # tag_matches — the value itself is never echoed into the detail.
    return (
        check["_pattern"].fullmatch(value) is not None,
        f"{label} value does not match pattern {check['pattern']!r}",
    )


# ---------------------------------------------------------------------------
# Run-level checks (report files)
# ---------------------------------------------------------------------------


def _csv_columns(path: Path) -> List[str]:
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    return list(header)


def _csv_column_values(path: Path, column: str) -> set:
    with open(path, newline="", encoding="utf-8") as f:
        return {str(row.get(column, "")) for row in csv.DictReader(f)}


def _check_run(run_dir: Path, check: Dict[str, Any], output_uids: List[str]) -> Tuple[bool, str]:
    check_type = check["type"]
    report = run_dir / str(check["file"])
    if check_type == "report_present":
        return report.is_file(), f"report {check['file']} is missing"
    if check_type == "outputs_accounted_in_report":
        if not output_uids:
            return True, ""
        if not report.is_file():
            return False, f"report {check['file']} is missing but DICOM outputs exist"
        recorded = _csv_column_values(report, str(check["column"]))
        unaccounted = sum(1 for uid in output_uids if uid not in recorded)
        return unaccounted == 0, (
            f"{unaccounted} output file(s) have no row in {check['file']} " f"(column {check['column']})"
        )
    if not report.is_file():
        # Column checks are vacuous when the presence check already covers
        # the missing file — don't double-report.
        return True, ""
    columns = _csv_columns(report)
    if check_type == "report_columns_forbidden":
        found = [c for c in check["columns"] if c in columns]
        return not found, f"report {check['file']} contains forbidden column(s): {found}"
    missing = [c for c in check["columns"] if c not in columns]
    return not missing, f"report {check['file']} is missing column(s): {missing}"


# ---------------------------------------------------------------------------
# Run verification
# ---------------------------------------------------------------------------


def _has_dicm_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def iter_output_dicoms(run_dir: Union[str, Path]) -> List[Path]:
    """DICOM files in a run directory (recursive, content-driven, sorted)."""
    root = Path(run_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and _has_dicm_magic(p))


def verify_run(
    run_dir: str,
    checklist: Union[None, str, Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify a de-identified run directory against a checklist.

    Args:
        run_dir: Directory containing the de-identified artifacts and the
            ground-truth reports.
        checklist: A loaded checklist dict, a path to a checklist YAML, or
            None for the packaged default.

    Returns:
        The PHI-free verification report (see
        ``monai_aegis/schemas/verification.schema.json``). ``status`` is
        ``fail`` when any error-severity check failed; warning-severity
        failures are reported as findings but do not fail the run.

    Raises:
        InputError: If *run_dir* is not a directory.
        ValueError: If the checklist is invalid.
    """
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"no such directory: {root}")
    if not isinstance(checklist, dict):
        checklist = load_checklist(checklist)

    findings: List[Dict[str, Any]] = []
    checks_evaluated = 0

    dicom_files = iter_output_dicoms(root)
    output_uids: List[str] = []
    for path in dicom_files:
        ds = pydicom.dcmread(str(path))
        output_uids.append(str(getattr(ds, "SOPInstanceUID", "")))
        for check in checklist["dicom_checks"]:
            checks_evaluated += 1
            passed, detail = _check_dataset(ds, check)
            if not passed:
                findings.append(
                    {
                        "check": check["id"],
                        "severity": check["severity"],
                        "file": path.name,
                        "detail": detail,
                    }
                )

    for check in checklist["run_checks"]:
        checks_evaluated += 1
        passed, detail = _check_run(root, check, output_uids)
        if not passed:
            findings.append(
                {
                    "check": check["id"],
                    "severity": check["severity"],
                    "file": str(check.get("file", "")),
                    "detail": detail,
                }
            )

    failures = sum(1 for f in findings if f["severity"] == "error")
    warnings = sum(1 for f in findings if f["severity"] == "warning")
    return {
        "report_version": REPORT_VERSION,
        "tool": TOOL_NAME,
        "checklist": checklist["name"],
        "run_dir": str(root),
        "status": STATUS_FAIL if failures else STATUS_PASS,
        "totals": {
            "files_checked": len(dicom_files),
            "checks_evaluated": checks_evaluated,
            "failures": failures,
            "warnings": warnings,
        },
        "findings": findings,
    }


def validate_report(report: Dict[str, Any]) -> None:
    """Validate a verification report against the packaged schema.

    Requires the optional ``jsonschema`` package (a dev/test dependency).
    """
    import json

    try:
        import jsonschema
    except ImportError as exc:
        raise ImportError(
            "report validation requires the 'jsonschema' package — " "install with: pip install 'monai-aegis[dev]'"
        ) from exc
    schema_text = (
        resources.files("monai_aegis.schemas").joinpath("verification.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(instance=report, schema=json.loads(schema_text))


__all__ = [
    "REPORT_VERSION",
    "TOOL_NAME",
    "STATUS_PASS",
    "STATUS_FAIL",
    "DICOM_CHECK_TYPES",
    "RUN_CHECK_TYPES",
    "default_checklist_path",
    "load_checklist",
    "iter_output_dicoms",
    "verify_run",
    "validate_report",
]
