"""Ground-truth reporting for Aegis de-identification runs.

When ``reporting.save_ground_truth`` is true (the default), the pipeline
runners use this module to flatten each processed file's redaction records
into CSV files written to the run output directory:

  - ``aegis_pixel_detections.csv`` — one row per OCR region (burnt-in PHI).
  - ``aegis_tag_actions.csv``      — one row per scrubbed DICOM header tag.

Both files carry the **original** ``SOPInstanceUID`` (preserved from the
cached source dataset) alongside the regenerated de-identified UID, so they
join directly to synthetic ground-truth CSVs that key on the original UID.

When the flag is false, nothing is written here — consume the same records
straight from the pipeline data dict (see :func:`extract_records`) and store
them in your own database.
"""
import csv
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

PIXEL_DETECTIONS_FILE = "aegis_pixel_detections.csv"
TAG_ACTIONS_FILE = "aegis_tag_actions.csv"

PIXEL_FIELDS = [
    "source_path", "original_sop_uid", "deid_sop_uid",
    "frame_index", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "ocr_text", "confidence", "decision",
]
TAG_FIELDS = [
    "source_path", "original_sop_uid", "deid_sop_uid",
    "tag", "keyword", "action", "redacted",
]


def is_enabled(config: Dict[str, Any]) -> bool:
    """Return whether ground-truth CSV reporting is enabled (default True)."""
    reporting = config.get("reporting", {}) or {}
    return bool(reporting.get("save_ground_truth", True))


def _sop_uid(ds: Any) -> str:
    return str(getattr(ds, "SOPInstanceUID", "")) if ds is not None else ""


def _source_path(data_dict: Dict[Any, Any], key: str) -> str:
    meta = data_dict.get(f"{key}_meta_dict", {}) or {}
    fpath = meta.get("filename_or_obj", "")
    if isinstance(fpath, (list, tuple)):
        fpath = fpath[0] if fpath else ""
    return str(fpath)


def _uid_pairs(data_dict: Dict[Any, Any], key: str) -> Tuple[List[Tuple[str, str]], bool]:
    """Return ``[(original_sop_uid, deid_sop_uid), ...]`` and an is_series flag.

    Single-file / multi-frame inputs yield a single pair; a multi-file series
    yields one pair per slice, index-aligned with the slice/frame index.
    """
    scrub_list = data_dict.get(f"{key}_scrubbed_datasets")
    if isinstance(scrub_list, list):
        orig_list = data_dict.get(f"{key}_dicom_datasets") or []
        pairs = []
        for i, sds in enumerate(scrub_list):
            ods = orig_list[i] if i < len(orig_list) else None
            pairs.append((_sop_uid(ods), _sop_uid(sds)))
        return pairs, True

    orig = data_dict.get(f"{key}_dicom_dataset")
    scrub = data_dict.get(f"{key}_scrubbed_ds")
    return [(_sop_uid(orig), _sop_uid(scrub))], False


def extract_records(
    data_dict: Dict[Any, Any],
    key: str = "image",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Flatten one processed pipeline result into (pixel_rows, tag_rows).

    Returns CSV-ready dict rows; callers store these or hand them to
    :func:`write_reports`. Safe to call on any result dict — missing
    sections simply yield no rows.
    """
    source_path = _source_path(data_dict, key)
    pairs, is_series = _uid_pairs(data_dict, key)
    default_orig, default_deid = pairs[0] if pairs else ("", "")

    # --- Pixel detections (burnt-in PHI) ---
    pixel_rows: List[Dict[str, Any]] = []
    stats = data_dict.get(f"{key}_redaction_stats", {}) or {}
    for det in stats.get("detections", []):
        frame_index = det.get("frame_index")
        orig_uid, deid_uid = default_orig, default_deid
        if is_series and isinstance(frame_index, int) and 0 <= frame_index < len(pairs):
            orig_uid, deid_uid = pairs[frame_index]
        bbox = det.get("bbox", [None, None, None, None])
        bx, by, bw, bh = (list(bbox) + [None, None, None, None])[:4]
        pixel_rows.append({
            "source_path": source_path,
            "original_sop_uid": orig_uid,
            "deid_sop_uid": deid_uid,
            "frame_index": frame_index if frame_index is not None else "",
            "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
            "ocr_text": det.get("ocr_text", ""),
            "confidence": det.get("confidence", ""),
            "decision": det.get("decision", ""),
        })

    # --- Header-tag actions ---
    tag_rows: List[Dict[str, Any]] = []
    per_slice = data_dict.get(f"{key}_tag_actions_per_slice")
    if isinstance(per_slice, list):
        for i, actions in enumerate(per_slice):
            orig_uid, deid_uid = pairs[i] if i < len(pairs) else (default_orig, default_deid)
            tag_rows.extend(_tag_rows(actions, source_path, orig_uid, deid_uid))
    else:
        tag_rows.extend(_tag_rows(
            data_dict.get(f"{key}_tag_actions", []),
            source_path, default_orig, default_deid,
        ))

    return pixel_rows, tag_rows


def _tag_rows(actions: Any, source_path: str, orig_uid: str, deid_uid: str) -> List[Dict[str, Any]]:
    rows = []
    for action in actions or []:
        rows.append({
            "source_path": source_path,
            "original_sop_uid": orig_uid,
            "deid_sop_uid": deid_uid,
            "tag": action.get("tag", ""),
            "keyword": action.get("keyword", ""),
            "action": action.get("action", ""),
            "redacted": action.get("redacted", ""),
        })
    return rows


def write_reports(
    pixel_rows: List[Dict[str, Any]],
    tag_rows: List[Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """Write the accumulated rows to CSV files in ``output_dir``.

    Always emits both files (with headers) even when empty, so downstream
    validation can rely on their presence. Returns the paths written.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for filename, fields, rows in (
        (PIXEL_DETECTIONS_FILE, PIXEL_FIELDS, pixel_rows),
        (TAG_ACTIONS_FILE, TAG_FIELDS, tag_rows),
    ):
        path = os.path.join(output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
        logger.info("Wrote ground-truth report: %s (%d rows)", path, len(rows))
    return written


class GroundTruthAccumulator:
    """Collect per-file redaction records across a run and write CSV reports.

    A no-op when ``reporting.save_ground_truth`` is false — in that case the
    caller is expected to read records from each result dict directly (e.g.
    via :func:`extract_records`) and persist them elsewhere. Shared by the
    DICOM and image runners.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = is_enabled(config)
        self.pixel_rows: List[Dict[str, Any]] = []
        self.tag_rows: List[Dict[str, Any]] = []

    def collect(self, data_dict: Dict[Any, Any], key: str = "image") -> None:
        if not self.enabled:
            return
        pixel_rows, tag_rows = extract_records(data_dict, key)
        self.pixel_rows.extend(pixel_rows)
        self.tag_rows.extend(tag_rows)

    def flush(self, output_dir: str) -> None:
        if not self.enabled:
            return
        write_reports(self.pixel_rows, self.tag_rows, output_dir)


__all__ = ["is_enabled", "extract_records", "write_reports", "GroundTruthAccumulator"]
