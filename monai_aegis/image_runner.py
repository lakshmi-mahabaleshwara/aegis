"""
Package-local standard image pipeline runner.
"""
import argparse
import logging
import os
import shutil
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from monai.data import DataLoader, Dataset

from monai_aegis.config.config_loader import load_config
from monai_aegis.config.storage import AegisFileSystem
from monai_aegis.transforms.discovery import discover_images
from monai_aegis.transforms.pipeline import build_image_pipeline, build_image_series_pipeline

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


def _resolve_rel_path(data_dict: dict[str, Any], paths: RunnerPaths, file_path: str) -> str:
    rel_path = data_dict.get("rel_path")
    if rel_path:
        return rel_path
    return os.path.relpath(file_path, paths.pipeline_input_dir)


def _quarantine_file(file_path: str, rel_path: str, paths: RunnerPaths, reason: str) -> None:
    if not isinstance(file_path, str) or file_path == "unknown":
        return
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
            rel_path = os.path.relpath(file_path, paths.pipeline_input_dir)
            _quarantine_file(file_path, rel_path, paths, reason)


def _cleanup_output(data_dict: dict[str, Any], paths: RunnerPaths, rel_path: str) -> None:
    out_path = data_dict.get("image_saved_path")
    if not out_path:
        filename = os.path.basename(rel_path)
        out_path = os.path.join(paths.output_dir, filename.rsplit(".", 1)[0] + ".png")
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
        file_path = _resolve_file_path(data_dict)
        rel_path = _resolve_rel_path(data_dict, paths, file_path)
        _quarantine_file(file_path, rel_path, paths, reason)
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
    rel_path = _resolve_rel_path(data_dict, paths, file_path)
    stats_dict = data_dict.get("image_redaction_stats", {})
    low_conf = stats_dict.get("low_confidence_count", 0)

    if low_conf > 0:
        _quarantine_file(file_path, rel_path, paths, "NOT PROCESSED")
        _cleanup_output(data_dict, paths, rel_path)
        summary.not_processed += 1
        return

    logger.info("Processed: %s", rel_path)
    if count_as_processed == "processed":
        summary.processed += 1
    else:
        summary.slices += 1


def _log_single_summary(summary: RunSummary) -> None:
    logger.info(
        "Single-file image processing complete. Processed: %d | Not processed: %d | Errors: %d",
        summary.processed,
        summary.not_processed,
        summary.errors,
    )


def _log_series_summary(summary: RunSummary) -> None:
    logger.info(
        "Image series processing complete. Folders: %d | Images: %d | Errors: %d",
        summary.series,
        summary.slices,
        summary.errors,
    )


def run_single(config_path: str) -> None:
    _config, fs, paths = _build_runner_paths(config_path, "image_folder")

    image_files = []
    walker = fs.walk(paths.pipeline_input_dir) if fs.protocol != "file" else os.walk(paths.pipeline_input_dir)
    for root, _dirs, filenames in walker:
        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = fs.join(root, filename) if fs.protocol != "file" else os.path.join(root, filename)
                image_files.append(file_path)
    image_files.sort()

    if not image_files:
        logger.warning("No image files found in %s", paths.pipeline_input_dir)
        return

    pipeline = build_image_pipeline(
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        input_dir=paths.pipeline_input_dir,
    )

    summary = RunSummary()
    data_list = []
    for file_path in image_files:
        rel_path = os.path.relpath(file_path, paths.input_dir)
        data_list.append({"image": file_path, "file_path": file_path, "rel_path": rel_path})

    for data_dict in _run_dataloader(
        data_list,
        pipeline,
        num_workers=paths.dataloader_num_workers,
    ):
        if _handle_error_result(data_dict, paths, summary, "NOT PROCESSED (ERROR)"):
            continue
        _handle_single_result(data_dict, paths, summary, count_as_processed="processed")

    _log_single_summary(summary)


def run_series(config_path: str) -> None:
    _config, fs, paths = _build_runner_paths(config_path, "image_folder")

    image_files = discover_images(paths.pipeline_input_dir, fs=fs)
    if not image_files:
        logger.warning("No image files found. Falling back to single-file mode.")
        run_single(paths.config_path)
        return

    folder_groups = defaultdict(list)
    
    # Use absolute paths for comparison since fsspec LocalFileSystem yields absolute paths
    abs_input_dir = os.path.abspath(paths.pipeline_input_dir) if fs.protocol == "file" else paths.pipeline_input_dir.rstrip("/")

    for file_path in image_files:
        parent_dir = fs.dirname(file_path) if fs.protocol != "file" else os.path.dirname(file_path)
        
        if fs.protocol == "file":
            abs_parent = os.path.abspath(parent_dir)
            is_root_file = (abs_parent == abs_input_dir)
        else:
            is_root_file = (parent_dir.rstrip("/") == abs_input_dir)
        
        if is_root_file:
            # Files directly in the root input directory are treated as individual singletons
            # Group them by their own file path so len(folder_images) == 1
            folder_groups[file_path].append(file_path)
        else:
            folder_groups[parent_dir].append(file_path)

    pipeline = build_image_series_pipeline(
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        input_dir=paths.pipeline_input_dir,
        output_ext=".png",
    )
    pipeline_single = build_image_pipeline(
        config_path=paths.config_path,
        output_dir=paths.output_dir,
        input_dir=paths.pipeline_input_dir,
        output_ext=".png",
    )

    singletons_data = []
    series_data = []
    for parent_dir, folder_images in folder_groups.items():
        label = os.path.basename(parent_dir) if fs.protocol == "file" else fs.basename(parent_dir)
        if len(folder_images) == 1:
            rel_path = (
                folder_images[0][len(paths.pipeline_input_dir):].lstrip("/")
                if fs.protocol != "file"
                else os.path.relpath(folder_images[0], paths.pipeline_input_dir)
            )
            singletons_data.append(
                {
                    "image": folder_images[0],
                    "file_path": folder_images[0],
                    "label": label,
                    "rel_path": rel_path,
                }
            )
        else:
            series_data.append({"image": folder_images, "label": label, "num_slices": len(folder_images)})

    summary = RunSummary()

    if singletons_data:
        for data_dict in _run_dataloader(
            singletons_data,
            pipeline_single,
            num_workers=paths.dataloader_num_workers,
        ):
            if _handle_error_result(data_dict, paths, summary, "NOT PROCESSED (ERROR)"):
                continue
            _handle_single_result(data_dict, paths, summary, count_as_processed="slices")

    if series_data:
        for data_dict in _run_dataloader(
            series_data,
            pipeline,
            num_workers=paths.dataloader_num_workers,
        ):
            label = data_dict.get("label", "unknown")
            if _handle_error_result(data_dict, paths, summary, f"NOT PROCESSED (ERROR): Series {label}"):
                continue

            stats_dict = data_dict.get("image_redaction_stats", {})
            strategy = stats_dict.get("volume_strategy", "unknown")
            num_slices = data_dict.get("num_slices", 0)

            logger.info(
                "Folder %s complete: strategy=%s, redacted=%d",
                label,
                strategy,
                stats_dict.get("redacted_count", 0),
            )

            summary.series += 1
            summary.slices += num_slices

    _log_series_summary(summary)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Aegis Image De-identification Pipeline")
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
