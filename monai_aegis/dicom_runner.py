"""
Package-local DICOM pipeline runner.
"""
import argparse
import logging
import os
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from monai.data import DataLoader, Dataset

from monai_aegis import reporting
from monai_aegis.config.config_loader import load_config
from monai_aegis.config.storage import AegisFileSystem
from monai_aegis.transforms import context_keys as ckeys
from monai_aegis.transforms.context_keys import ck
from monai_aegis.transforms.discovery import (
    discover_dicoms,
    group_into_series,
    sort_slices,
    validate_series,
)
from monai_aegis.transforms.pipeline import build_pipeline, build_series_pipeline

logger = logging.getLogger(__name__)


@dataclass
class RunnerPaths:
    config_path: str
    input_dir: str
    output_dir: str
    not_processed_dir: str
    pipeline_input_dir: str
    dataloader_num_workers: int


@dataclass
class RunSummary:
    processed: int = 0
    not_processed: int = 0
    errors: int = 0
    series: int = 0
    slices: int = 0


def _require_config_value(config: dict[str, Any], section: str, key: str) -> str:
    section_data = config.get(section)
    if not isinstance(section_data, dict) or key not in section_data:
        raise KeyError(f"Missing required config value: {section}.{key}")
    value = section_data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config value must be a non-empty string: {section}.{key}")
    return value


def _get_num_workers(config: dict[str, Any]) -> int:
    runtime = config.get("runtime", {})
    value = runtime.get("dataloader_num_workers", 0)
    if not isinstance(value, int) or value < 0:
        raise ValueError("Config value must be a non-negative integer: runtime.dataloader_num_workers")
    return value


def simple_collate(batch):
    return batch


class SafeTransformPipeline:
    """Wrap the transform pipeline so worker failures are returned as data."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, data):
        try:
            return self.pipeline(data)
        except Exception as exc:
            logger.error(
                "Error in transform pipeline: %s",
                getattr(exc, "filepath", data.get("image")),
                exc_info=True,
            )
            data["error"] = str(exc)
            data["error_trace"] = traceback.format_exc()
            return data


def _build_runner_paths(config_path: str, folder_key: str) -> tuple[dict[str, Any], AegisFileSystem, RunnerPaths]:
    config = load_config(config_path)
    input_dir = _require_config_value(config, "paths", "input_dir")
    base_output_dir = _require_config_value(config, "paths", "output_dir")
    not_processed_dir = _require_config_value(config, "paths", "not_processed_dir")
    folder_name = _require_config_value(config, "paths", folder_key)
    timestamp_format = _require_config_value(config, "paths", "timestamp_format")
    dataloader_num_workers = _get_num_workers(config)
    timestamp_str = datetime.now().strftime(timestamp_format)
    output_dir = os.path.join(base_output_dir, folder_name, timestamp_str)

    normalized_config_path = os.path.abspath(config_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(not_processed_dir, exist_ok=True)
    fs = AegisFileSystem.from_config(config)

    return config, fs, RunnerPaths(
        config_path=normalized_config_path,
        input_dir=input_dir,
        output_dir=output_dir,
        not_processed_dir=not_processed_dir,
        pipeline_input_dir=os.path.join(input_dir, folder_name),
        dataloader_num_workers=dataloader_num_workers,
    )


def _run_dataloader(data_list: list[dict[str, Any]], pipeline: Any, num_workers: int) -> Iterable[dict[str, Any]]:
    dataset = Dataset(data=data_list, transform=SafeTransformPipeline(pipeline))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=simple_collate,
    )
    for batch in dataloader:
        yield from batch


def _resolve_file_path(data_dict: dict[str, Any]) -> str:
    file_path = data_dict.get("file_path", data_dict.get("image", "unknown"))
    if not isinstance(file_path, str) and hasattr(file_path, "meta"):
        file_path = file_path.meta.get("filename_or_obj", "unknown")
    return str(file_path)


def _quarantine_file(file_path: str, paths: RunnerPaths, reason: str) -> None:
    if not isinstance(file_path, str) or file_path == "unknown":
        return
    rel_path = os.path.relpath(file_path, paths.pipeline_input_dir)
    dest = os.path.join(paths.not_processed_dir, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        shutil.copy2(file_path, dest)
        logger.warning("%s: %s -> %s", reason, rel_path, dest)
    except Exception:
        logger.warning("%s: %s", reason, rel_path)


def _quarantine_many(files: Iterable[str], paths: RunnerPaths, reason: str) -> None:
    for file_path in files:
        if isinstance(file_path, str):
            _quarantine_file(file_path, paths, reason)


def _cleanup_output(data_dict: dict[str, Any], paths: RunnerPaths, file_path: str) -> None:
    rel_path = os.path.relpath(file_path, paths.pipeline_input_dir)
    filename = os.path.basename(file_path)
    saved_key = ck("image", ckeys.SAVED_PATH)
    out_path = data_dict.get(saved_key, os.path.join(paths.output_dir, filename))
    if out_path == os.path.join(paths.output_dir, filename):
        out_path = data_dict.get(saved_key, os.path.join(paths.output_dir, rel_path))
    if os.path.exists(out_path):
        os.remove(out_path)


def _handle_error_result(data_dict: dict[str, Any], paths: RunnerPaths, summary: RunSummary, reason: str) -> bool:
    if "error" not in data_dict:
        return False
    summary.errors += 1
    image_value = data_dict.get("image", [])
    if isinstance(image_value, list):
        _quarantine_many(image_value, paths, reason)
    else:
        _quarantine_file(_resolve_file_path(data_dict), paths, reason)
    return True


def _handle_single_result(
    data_dict: dict[str, Any],
    paths: RunnerPaths,
    summary: RunSummary,
    count_as_processed: str,
) -> None:
    file_path = _resolve_file_path(data_dict)
    if file_path == "unknown":
        return

    stats_dict = data_dict.get(ck("image", ckeys.REDACTION_STATS), {})
    low_conf = stats_dict.get("low_confidence_count", 0)

    if low_conf > 0:
        _quarantine_file(file_path, paths, "NOT PROCESSED")
        _cleanup_output(data_dict, paths, file_path)
        summary.not_processed += 1
        return

    out_path = data_dict.get(ck("image", ckeys.SAVED_PATH), os.path.join(paths.output_dir, os.path.basename(file_path)))
    if os.path.exists(out_path):
        logger.info("Saved DICOM -> %s", out_path)
        if count_as_processed == "processed":
            summary.processed += 1
        else:
            summary.slices += 1
    else:
        logger.warning("Output DICOM not found for %s", os.path.basename(file_path))


def _log_single_summary(summary: RunSummary) -> None:
    logger.info(
        "Single-file DICOM complete. Processed: %d | Not processed: %d | Errors: %d",
        summary.processed,
        summary.not_processed,
        summary.errors,
    )


def _log_series_summary(summary: RunSummary) -> None:
    logger.info(
        "DICOM series processing complete. Series: %d | Slices: %d | Errors: %d",
        summary.series,
        summary.slices,
        summary.errors,
    )


def run_single(config_path: str) -> None:
    config, fs, paths = _build_runner_paths(config_path, "dicom_folder")
    logger.info("Building single-file DICOM pipeline -> %s", paths.output_dir)
    pipeline = build_pipeline(
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        input_dir=paths.pipeline_input_dir,
    )

    summary = RunSummary()
    report = reporting.GroundTruthAccumulator(config)

    slices = discover_dicoms(paths.pipeline_input_dir, fs=fs)
    data_list = [{"image": s.uri} for s in slices]

    if not data_list:
        logger.warning("No DICOM files found in %s", paths.input_dir)
        return

    for data_dict in _run_dataloader(data_list, pipeline, num_workers=0):
        report.collect(data_dict)
        if _handle_error_result(data_dict, paths, summary, "NOT PROCESSED (ERROR)"):
            continue
        _handle_single_result(data_dict, paths, summary, count_as_processed="processed")

    report.flush(paths.output_dir)
    _log_single_summary(summary)


def run_series(config_path: str) -> None:
    config, fs, paths = _build_runner_paths(config_path, "dicom_folder")

    logger.info("Discovering DICOM files in %s...", paths.pipeline_input_dir)
    slices = discover_dicoms(paths.pipeline_input_dir, fs=fs)

    if not slices:
        logger.warning("No DICOM files found. Falling back to single-file mode.")
        run_single(paths.config_path)
        return

    series_groups = group_into_series(slices)

    logger.info("Building series pipeline -> %s", paths.output_dir)
    pipeline = build_series_pipeline(
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        input_dir=paths.pipeline_input_dir,
    )
    pipeline_single = build_pipeline(
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        input_dir=paths.pipeline_input_dir,
    )

    singletons_data = []
    series_data = []

    for (study_uid, series_uid), series_slices in series_groups.items():
        sub_series_list = validate_series(series_slices)
        for sub_idx, sub_series in enumerate(sub_series_list):
            sorted_series = sort_slices(sub_series)
            uris = [s.uri for s in sorted_series]

            label = series_uid
            if len(sub_series_list) > 1:
                label += f" (sub-{sub_idx})"

            if len(uris) == 1:
                singletons_data.append(
                    {"image": uris[0], "label": label, "filename": os.path.basename(uris[0])}
                )
            else:
                series_data.append(
                    {
                        "image": uris,
                        "label": label,
                        "num_slices": len(uris),
                        "study_uid": study_uid[:8],
                    }
                )

    summary = RunSummary()
    report = reporting.GroundTruthAccumulator(config)

    if singletons_data:
        for data_dict in _run_dataloader(
            singletons_data,
            pipeline_single,
            num_workers=paths.dataloader_num_workers,
        ):
            report.collect(data_dict)
            if _handle_error_result(data_dict, paths, summary, "NOT PROCESSED (ERROR)"):
                continue
            _handle_single_result(data_dict, paths, summary, count_as_processed="slices")

    if series_data:
        for data_dict in _run_dataloader(
            series_data,
            pipeline,
            num_workers=paths.dataloader_num_workers,
        ):
            report.collect(data_dict)
            label = data_dict.get("label", "unknown")
            if _handle_error_result(data_dict, paths, summary, f"NOT PROCESSED (ERROR): Series {label}"):
                continue

            stats_dict = data_dict.get(ck("image", ckeys.REDACTION_STATS), {})
            strategy = stats_dict.get("volume_strategy", "unknown")
            target_token = data_dict.get(ck("image", ckeys.TARGET_TOKEN))
            out_msg = f"Token: {target_token}" if target_token else "Original structure"
            num_slices = data_dict.get("num_slices", 0)

            logger.info(
                "Series %s complete: strategy=%s, redacted=%d | Output: %s",
                label,
                strategy,
                stats_dict.get("redacted_count", 0),
                out_msg,
            )

            summary.series += 1
            summary.slices += num_slices

    report.flush(paths.output_dir)
    _log_series_summary(summary)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Aegis DICOM De-identification Pipeline")
    parser.add_argument(
        "--config",
        default="monai_aegis/config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "series"],
        default="series",
        help="'single' (per-file) or 'series' (volume-aware, default).",
    )
    args = parser.parse_args()

    if args.mode == "series":
        run_series(args.config)
    else:
        run_single(args.config)


__all__ = ["main", "run_series", "run_single"]


if __name__ == "__main__":
    main()
