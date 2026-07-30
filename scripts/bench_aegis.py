#!/usr/bin/env python3
"""
AEGIS Benchmarking Script — bench_aegis.py

Measures per-study/per-file de-identification performance and generates
a CSV of metrics for performance analysis and reproducible reporting.

Metrics collected
-----------------
* **Metadata scrub**: number of tags removed, dummied, zeroed.
* **Pixel redaction**: OCR detections, NER classifications, redacted
  regions, low-confidence count.
* **Tokenization**: deterministic-token collision check across all
  processed files.
* **PHI detection quality** (when ``--ground-truth`` is provided):
  precision, recall, F1 computed per file.
* **Runtime**: wall-clock seconds per file/series (CPU and peak RSS).
* **Storage**: input vs. output file size.

Usage
-----
::

    python scripts/bench_aegis.py \\
        --input  staging_input/dicom \\
        --config src/monai_aegis/config/config.yaml \\
        --output tests/benchmark/results/ \\
        [--ground-truth tests/benchmark/public_dataset/ground_truth/annotations.csv] \\
        [--mode single|series]

"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom

# ---------------------------------------------------------------------------
# Optional dependency — psutil for memory profiling
# ---------------------------------------------------------------------------
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Aegis imports
# ---------------------------------------------------------------------------
from monai_aegis.config.config_loader import load_config
from monai_aegis.config.storage import AegisFileSystem
from monai_aegis.transforms.discovery import discover_dicoms, group_into_series, sort_slices, validate_series
from monai_aegis.transforms.pipeline import build_pipeline, build_series_pipeline

logger = logging.getLogger("bench_aegis")


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FileMetrics:
    """Per-file / per-series metrics row."""

    study_id: str = ""
    series_id: str = ""
    file_path: str = ""
    mode: str = "single"  # "single" | "series"
    num_slices: int = 1

    # --- Pixel redaction stats ---
    total_ocr_detections: int = 0
    low_confidence_count: int = 0
    safelisted_count: int = 0
    ner_classified_count: int = 0
    redacted_count: int = 0

    # --- Metadata stats ---
    num_tags_removed: int = 0
    num_tags_dummied: int = 0
    num_tags_zeroed: int = 0
    private_tags_removed: int = 0

    # --- Tokenization ---
    patient_id_token: str = ""
    patient_name_token: str = ""

    # --- PHI detection quality (requires ground truth) ---
    gt_phi_count: int = 0
    detected_phi_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # --- Performance ---
    runtime_seconds: float = 0.0
    peak_rss_mb: float = 0.0

    # --- Storage ---
    input_size_bytes: int = 0
    output_size_bytes: int = 0

    # --- Status ---
    status: str = "success"  # "success" | "error" | "low_confidence"
    error_message: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Ground-truth loader
# ═══════════════════════════════════════════════════════════════════════════

def load_ground_truth(gt_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load ground-truth annotations from CSV.

    Returns a dict keyed by normalised file path, each value is a list of
    annotation dicts with keys: text, x_min, y_min, x_max, y_max, is_phi,
    phi_type.
    """
    annotations: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isfile(gt_path):
        logger.warning("Ground-truth file not found: %s", gt_path)
        return annotations

    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = os.path.normpath(row["file_path"])
            entry = {
                "text": row.get("text", ""),
                "x_min": int(row.get("x_min", 0)),
                "y_min": int(row.get("y_min", 0)),
                "x_max": int(row.get("x_max", 0)),
                "y_max": int(row.get("y_max", 0)),
                "is_phi": row.get("is_phi", "false").lower() == "true",
                "phi_type": row.get("phi_type", ""),
            }
            annotations.setdefault(key, []).append(entry)
    logger.info("Loaded %d ground-truth files from %s", len(annotations), gt_path)
    return annotations


# ═══════════════════════════════════════════════════════════════════════════
# PHI quality scoring
# ═══════════════════════════════════════════════════════════════════════════

def compute_phi_quality(
    gt_annotations: List[Dict[str, Any]],
    redaction_stats: Dict[str, Any],
) -> Dict[str, float]:
    """Compare pipeline output against ground-truth annotations.

    Uses a simplified region-level matching: counts the number of
    ground-truth PHI regions vs. redacted regions.  For a full
    bounding-box IoU evaluation, extend this function.
    """
    gt_phi = [a for a in gt_annotations if a["is_phi"]]
    gt_non_phi = [a for a in gt_annotations if not a["is_phi"]]

    detected = redaction_stats.get("redacted_count", 0)
    safelisted = redaction_stats.get("safelisted_count", 0)

    # Simplified: assume all redacted regions are true positives up to
    # the ground-truth count, remainder are false positives.
    tp = min(detected, len(gt_phi))
    fp = max(0, detected - len(gt_phi))
    fn = max(0, len(gt_phi) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "gt_phi_count": len(gt_phi),
        "detected_phi_count": detected,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Metadata diff helper
# ═══════════════════════════════════════════════════════════════════════════

def count_metadata_changes(
    original_ds: pydicom.Dataset,
    scrubbed_ds: pydicom.Dataset,
) -> Dict[str, int]:
    """Compare original vs. scrubbed DICOM datasets at the tag level."""
    removed = 0
    dummied = 0
    zeroed = 0
    private_removed = 0

    orig_tags = set(original_ds.keys())
    scrub_tags = set(scrubbed_ds.keys())

    # Removed tags
    for tag in orig_tags - scrub_tags:
        elem = original_ds[tag]
        if elem.tag.is_private:
            private_removed += 1
        else:
            removed += 1

    # Modified tags
    for tag in orig_tags & scrub_tags:
        orig_val = str(original_ds[tag].value)
        scrub_val = str(scrubbed_ds[tag].value)
        if orig_val != scrub_val:
            if scrub_val.startswith("TOKEN_"):
                dummied += 1
            elif scrub_val in ("", "0", "00000000", "b''"):
                zeroed += 1

    return {
        "num_tags_removed": removed,
        "num_tags_dummied": dummied,
        "num_tags_zeroed": zeroed,
        "private_tags_removed": private_removed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single-file benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_single_file(
    file_path: str,
    pipeline: Any,
    config: Dict[str, Any],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> FileMetrics:
    """Run the pipeline on a single file and collect metrics."""
    metrics = FileMetrics(file_path=file_path, mode="single")

    # Pre-load original DICOM for metadata comparison
    try:
        original_ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        metrics.study_id = getattr(original_ds, "StudyInstanceUID", "")[:16]
        metrics.series_id = getattr(original_ds, "SeriesInstanceUID", "")[:16]
        metrics.input_size_bytes = os.path.getsize(file_path)
    except Exception:
        metrics.study_id = "unknown"

    # Run pipeline
    rss_before = psutil.Process().memory_info().rss if _HAS_PSUTIL else 0
    t0 = time.perf_counter()

    try:
        result = pipeline({"image": file_path})
        metrics.runtime_seconds = round(time.perf_counter() - t0, 4)
    except Exception as e:
        metrics.runtime_seconds = round(time.perf_counter() - t0, 4)
        metrics.status = "error"
        metrics.error_message = str(e)
        return metrics

    if _HAS_PSUTIL:
        rss_after = psutil.Process().memory_info().rss
        metrics.peak_rss_mb = round((rss_after - rss_before) / (1024 * 1024), 2)

    # Extract redaction stats
    stats = result.get("image_redaction_stats", {})
    metrics.total_ocr_detections = stats.get("total_detections", 0)
    metrics.low_confidence_count = stats.get("low_confidence_count", 0)
    metrics.safelisted_count = stats.get("safelisted_count", 0)
    metrics.ner_classified_count = stats.get("ner_classified_count", 0)
    metrics.redacted_count = stats.get("redacted_count", 0)

    if metrics.low_confidence_count > 0:
        metrics.status = "low_confidence"

    # Extract tokenization info
    scrubbed = result.get("image_scrubbed_ds")
    if scrubbed is not None:
        metrics.patient_id_token = getattr(scrubbed, "PatientID", "")
        metrics.patient_name_token = str(getattr(scrubbed, "PatientName", ""))

        # Metadata diff
        try:
            tag_changes = count_metadata_changes(original_ds, scrubbed)
            metrics.num_tags_removed = tag_changes["num_tags_removed"]
            metrics.num_tags_dummied = tag_changes["num_tags_dummied"]
            metrics.num_tags_zeroed = tag_changes["num_tags_zeroed"]
            metrics.private_tags_removed = tag_changes["private_tags_removed"]
        except Exception:
            pass

    # Output file size
    saved_path = result.get("image_saved_path", "")
    if saved_path and os.path.exists(saved_path):
        metrics.output_size_bytes = os.path.getsize(saved_path)

    # PHI quality scoring
    norm_path = os.path.normpath(file_path)
    if norm_path in ground_truth:
        quality = compute_phi_quality(ground_truth[norm_path], stats)
        metrics.gt_phi_count = quality["gt_phi_count"]
        metrics.detected_phi_count = quality["detected_phi_count"]
        metrics.true_positives = quality["true_positives"]
        metrics.false_positives = quality["false_positives"]
        metrics.false_negatives = quality["false_negatives"]
        metrics.precision = quality["precision"]
        metrics.recall = quality["recall"]
        metrics.f1 = quality["f1"]

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Series benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_series(
    uris: List[str],
    label: str,
    study_uid: str,
    series_uid: str,
    pipeline: Any,
    config: Dict[str, Any],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> FileMetrics:
    """Run the series pipeline on a group of DICOM slices and collect metrics."""
    metrics = FileMetrics(
        file_path=uris[0] if uris else "",
        mode="series",
        study_id=study_uid[:16],
        series_id=series_uid[:16],
        num_slices=len(uris),
    )

    # Total input size
    metrics.input_size_bytes = sum(
        os.path.getsize(u) for u in uris if os.path.exists(u)
    )

    rss_before = psutil.Process().memory_info().rss if _HAS_PSUTIL else 0
    t0 = time.perf_counter()

    try:
        result = pipeline({"image": uris, "label": label, "num_slices": len(uris)})
        metrics.runtime_seconds = round(time.perf_counter() - t0, 4)
    except Exception as e:
        metrics.runtime_seconds = round(time.perf_counter() - t0, 4)
        metrics.status = "error"
        metrics.error_message = str(e)
        return metrics

    if _HAS_PSUTIL:
        rss_after = psutil.Process().memory_info().rss
        metrics.peak_rss_mb = round((rss_after - rss_before) / (1024 * 1024), 2)

    # Extract redaction stats
    stats = result.get("image_redaction_stats", {})
    metrics.total_ocr_detections = stats.get("total_detections", 0)
    metrics.low_confidence_count = stats.get("low_confidence_count", 0)
    metrics.safelisted_count = stats.get("safelisted_count", 0)
    metrics.ner_classified_count = stats.get("ner_classified_count", 0)
    metrics.redacted_count = stats.get("redacted_count", 0)

    if metrics.low_confidence_count > 0:
        metrics.status = "low_confidence"

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Tokenization collision check
# ═══════════════════════════════════════════════════════════════════════════

def check_token_collisions(all_metrics: List[FileMetrics]) -> Dict[str, Any]:
    """Verify that no two distinct patient IDs produce the same token.

    Returns a summary dict with collision count and details.
    """
    token_to_originals: Dict[str, set] = {}
    for m in all_metrics:
        if m.patient_id_token:
            token_to_originals.setdefault(m.patient_id_token, set()).add(m.study_id)

    collisions = {
        tok: list(origins)
        for tok, origins in token_to_originals.items()
        if len(origins) > 1
    }

    return {
        "total_unique_tokens": len(token_to_originals),
        "collision_count": len(collisions),
        "collisions": collisions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CSV writer
# ═══════════════════════════════════════════════════════════════════════════

def write_results_csv(metrics_list: List[FileMetrics], output_path: str) -> None:
    """Write all per-file metrics to a CSV file."""
    if not metrics_list:
        logger.warning("No metrics to write.")
        return

    fieldnames = list(asdict(metrics_list[0]).keys())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(asdict(m))

    logger.info("Wrote %d rows to %s", len(metrics_list), output_path)


def write_summary_json(
    metrics_list: List[FileMetrics],
    collision_report: Dict[str, Any],
    output_path: str,
) -> None:
    """Write an aggregate summary JSON file."""
    successful = [m for m in metrics_list if m.status == "success"]
    errors = [m for m in metrics_list if m.status == "error"]
    low_conf = [m for m in metrics_list if m.status == "low_confidence"]

    runtimes = [m.runtime_seconds for m in successful]
    f1_scores = [m.f1 for m in successful if m.gt_phi_count > 0]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(metrics_list),
        "successful": len(successful),
        "errors": len(errors),
        "low_confidence": len(low_conf),
        "runtime": {
            "mean_seconds": round(np.mean(runtimes), 4) if runtimes else 0,
            "std_seconds": round(np.std(runtimes), 4) if runtimes else 0,
            "min_seconds": round(min(runtimes), 4) if runtimes else 0,
            "max_seconds": round(max(runtimes), 4) if runtimes else 0,
            "total_seconds": round(sum(runtimes), 4) if runtimes else 0,
        },
        "phi_detection": {
            "mean_f1": round(np.mean(f1_scores), 4) if f1_scores else None,
            "std_f1": round(np.std(f1_scores), 4) if f1_scores else None,
            "files_with_ground_truth": len(f1_scores),
        },
        "tokenization": collision_report,
        "storage": {
            "total_input_bytes": sum(m.input_size_bytes for m in metrics_list),
            "total_output_bytes": sum(m.output_size_bytes for m in successful),
            "overhead_ratio": round(
                sum(m.output_size_bytes for m in successful)
                / max(sum(m.input_size_bytes for m in metrics_list), 1),
                4,
            ),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote summary to %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AEGIS Benchmarking — measure de-identification performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory containing DICOM files to benchmark.",
    )
    parser.add_argument(
        "--config", default="src/monai_aegis/config/config.yaml",
        help="Path to config.yaml (default: src/monai_aegis/config/config.yaml).",
    )
    parser.add_argument(
        "--output", default="tests/benchmark/results/",
        help="Directory for benchmark result CSVs and summary JSON.",
    )
    parser.add_argument(
        "--ground-truth", default=None, dest="ground_truth",
        help="Path to ground-truth annotations CSV (optional).",
    )
    parser.add_argument(
        "--mode", choices=["single", "series"], default="single",
        help="Pipeline mode: 'single' (per-file) or 'series' (volume-aware).",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config = load_config(args.config)
    fs = AegisFileSystem.from_config(config)

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.output, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # ---- Ground truth ----
    ground_truth: Dict[str, List[Dict[str, Any]]] = {}
    if args.ground_truth:
        ground_truth = load_ground_truth(args.ground_truth)

    # ---- Discover DICOM files ----
    logger.info("Discovering DICOMs in %s ...", args.input)
    slices = discover_dicoms(args.input, fs=fs)

    if not slices:
        logger.error("No DICOM files found in %s", args.input)
        sys.exit(1)

    logger.info("Found %d DICOM slices", len(slices))

    all_metrics: List[FileMetrics] = []

    if args.mode == "single":
        # ---- Single-file mode ----
        pipeline = build_pipeline(
            config_path=os.path.abspath(args.config),
            output_dir=os.path.join(results_dir, "output"),
            input_dir=args.input,
        )

        for i, s in enumerate(slices):
            logger.info("[%d/%d] Processing %s", i + 1, len(slices), os.path.basename(s.uri))
            m = benchmark_single_file(
                file_path=s.uri,
                pipeline=pipeline,
                config=config,
                ground_truth=ground_truth,
                output_dir=os.path.join(results_dir, "output"),
            )
            all_metrics.append(m)

            if m.status == "error":
                logger.error("  ERROR: %s", m.error_message)
            else:
                logger.info(
                    "  OK: OCR=%d, redacted=%d, runtime=%.2fs",
                    m.total_ocr_detections, m.redacted_count, m.runtime_seconds,
                )

    else:
        # ---- Series mode ----
        series_groups = group_into_series(slices)

        pipeline = build_series_pipeline(
            config_path=os.path.abspath(args.config),
            output_dir=os.path.join(results_dir, "output"),
            input_dir=args.input,
        )
        pipeline_single = build_pipeline(
            config_path=os.path.abspath(args.config),
            output_dir=os.path.join(results_dir, "output"),
            input_dir=args.input,
        )

        idx = 0
        total = len(series_groups)
        for (study_uid, series_uid), series_slices in series_groups.items():
            idx += 1
            sub_series_list = validate_series(series_slices)
            for sub_idx, sub_series in enumerate(sub_series_list):
                sorted_series = sort_slices(sub_series)
                uris = [s.uri for s in sorted_series]
                label = series_uid if len(sub_series_list) == 1 else f"{series_uid}_sub{sub_idx}"

                if len(uris) == 1:
                    logger.info("[%d/%d] Single-slice series %s", idx, total, label[:16])
                    m = benchmark_single_file(
                        file_path=uris[0],
                        pipeline=pipeline_single,
                        config=config,
                        ground_truth=ground_truth,
                        output_dir=os.path.join(results_dir, "output"),
                    )
                else:
                    logger.info("[%d/%d] Series %s (%d slices)", idx, total, label[:16], len(uris))
                    m = benchmark_series(
                        uris=uris,
                        label=label,
                        study_uid=study_uid,
                        series_uid=series_uid,
                        pipeline=pipeline,
                        config=config,
                        ground_truth=ground_truth,
                        output_dir=os.path.join(results_dir, "output"),
                    )

                all_metrics.append(m)

                if m.status == "error":
                    logger.error("  ERROR: %s", m.error_message)
                else:
                    logger.info(
                        "  OK: slices=%d, OCR=%d, redacted=%d, runtime=%.2fs",
                        m.num_slices, m.total_ocr_detections,
                        m.redacted_count, m.runtime_seconds,
                    )

    # ---- Collision check ----
    collision_report = check_token_collisions(all_metrics)
    if collision_report["collision_count"] > 0:
        logger.warning(
            "TOKEN COLLISION DETECTED: %d collisions across %d unique tokens",
            collision_report["collision_count"],
            collision_report["total_unique_tokens"],
        )
    else:
        logger.info(
            "No token collisions detected (%d unique tokens)",
            collision_report["total_unique_tokens"],
        )

    # ---- Write results ----
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    json_path = os.path.join(results_dir, "benchmark_summary.json")

    write_results_csv(all_metrics, csv_path)
    write_summary_json(all_metrics, collision_report, json_path)

    # ---- Print summary to console ----
    successful = [m for m in all_metrics if m.status == "success"]
    print("\n" + "=" * 72)
    print("AEGIS BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"  Total files processed : {len(all_metrics)}")
    print(f"  Successful            : {len(successful)}")
    print(f"  Errors                : {len([m for m in all_metrics if m.status == 'error'])}")
    print(f"  Low confidence        : {len([m for m in all_metrics if m.status == 'low_confidence'])}")

    if successful:
        runtimes = [m.runtime_seconds for m in successful]
        print(f"  Mean runtime          : {np.mean(runtimes):.3f}s ± {np.std(runtimes):.3f}s")
        print(f"  Total runtime         : {sum(runtimes):.1f}s")

    f1_scores = [m.f1 for m in successful if m.gt_phi_count > 0]
    if f1_scores:
        print(f"  Mean PHI F1           : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    print(f"  Token collisions      : {collision_report['collision_count']}")
    print(f"  Results               : {results_dir}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
