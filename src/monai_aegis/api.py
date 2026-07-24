"""Invocation facade — the single-call skill surface of Aegis.

One function, :func:`deidentify`, takes an input path (file or directory),
an output directory, and a config, and returns the versioned PHI-free
envelope defined by :mod:`monai_aegis.envelope`. Every skill wrapper — the
``aegis-deidentify`` CLI, MCP tools, catalog manifests — is a thin adapter
over this module; it contains no behavior of its own beyond orchestration.

Contract (what wrappers may rely on):
  - Output is written directly into ``output_dir`` — no timestamped
    subdirectories. The caller owns the layout.
  - Files are discovered and processed in sorted order, and the ground-truth
    reports are written once per invocation, so identical input + identical
    config + identical ``AEGIS_TOKEN_SALT`` reproduce identical results.
  - Per-file failures never abort the run; they are recorded in the
    envelope (status ``partial``, or ``error`` when nothing processed).
  - Behavior is configured exclusively through the YAML config (base +
    overlay + env interpolation) — this module adds no override knobs.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Heavy pipeline imports (torch / easyocr via monai_aegis.transforms) are
# deferred into function bodies so importing this module stays cheap — the
# MCP server and CLI import it at startup.
from monai_aegis import envelope

logger = logging.getLogger(__name__)

#: File extensions treated as standard images; everything else is probed as
#: DICOM. Canonical home for this constant — other modules import it.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

#: Discovery modes, mirroring the unified CLI: 'auto' covers both kinds.
MODES = ("auto", "dicom", "image")


class InputError(ValueError):
    """The input path, mode, or output directory is invalid (exit code 2)."""


def default_config_path() -> str:
    """Path of the packaged base ``config.yaml``."""
    return str(Path(__file__).resolve().parent / "config" / "config.yaml")


def skill_overlay_path() -> str:
    """Path of the packaged skill-mode overlay ``config.skill.yaml``."""
    return str(Path(__file__).resolve().parent / "config" / "config.skill.yaml")


def classify_input(path: str) -> str:
    """Return the pipeline kind ('image' or 'dicom') for a single file."""
    return "image" if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS else "dicom"


def discover_dicom_files(input_dir: str) -> List[str]:
    """Content-driven DICOM discovery (DICM magic), sorted."""
    from monai_aegis.transforms.discovery import discover_dicoms

    return sorted(s.uri for s in discover_dicoms(input_dir))


def discover_image_files(input_dir: str) -> List[str]:
    """Extension-driven image discovery (.jpg/.jpeg/.png), sorted."""
    from monai_aegis.transforms.discovery import discover_images

    return sorted(discover_images(input_dir))


def discover_inputs(input_dir: str, mode: str = "auto") -> List[Tuple[str, str]]:
    """Discover files to process as sorted ``(path, kind)`` pairs.

    'dicom' is content-driven (DICM magic), 'image' extension-driven,
    'auto' both — DICOM first, matching the unified CLI's run order.

    Raises:
        InputError: If *mode* is not one of :data:`MODES`.
    """
    if mode not in MODES:
        raise InputError(f"unknown mode {mode!r}. Use one of: {', '.join(MODES)}.")
    files: List[Tuple[str, str]] = []
    if mode in ("auto", "dicom"):
        files.extend((path, "dicom") for path in discover_dicom_files(input_dir))
    if mode in ("auto", "image"):
        files.extend((path, "image") for path in discover_image_files(input_dir))
    return files


def resolve_overlay_path(overlay_path: Optional[str] = None) -> Optional[str]:
    """Pick the effective overlay for a skill invocation.

    An explicit *overlay_path* wins; otherwise the ``AEGIS_CONFIG_OVERRIDE``
    env var is left to the config loader; otherwise the packaged skill
    overlay applies — deployments customize by supplying their own YAML,
    not code.
    """
    if overlay_path is None and not os.environ.get("AEGIS_CONFIG_OVERRIDE"):
        return skill_overlay_path()
    return overlay_path


def load_run_config(
    config_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the resolved config for a skill invocation (base + overlay + env)."""
    from monai_aegis.config.config_loader import load_config

    base = config_path or default_config_path()
    return load_config(base, overlay_path=resolve_overlay_path(overlay_path))


def _build_pipeline(
    kind: str,
    config_path: str,
    input_dir: str,
    output_dir: str,
    overlay_path: Optional[str] = None,
) -> Any:
    from monai_aegis.transforms.pipeline import build_image_pipeline, build_pipeline

    builder = build_image_pipeline if kind == "image" else build_pipeline
    return builder(
        config_path=config_path,
        output_dir=output_dir,
        input_dir=input_dir,
        overlay_path=overlay_path,
    )


def deidentify(
    input_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
    mode: str = "auto",
) -> Dict[str, Any]:
    """De-identify one file or every file in a directory; return the envelope.

    Args:
        input_path: A DICOM/JPEG/PNG file, or a directory to scan.
        output_dir: Destination for de-identified artifacts and the
            ground-truth reports. Created if missing; the caller owns the
            layout (no timestamped subdirectories are added).
        config_path: Base config YAML (default: the packaged config).
        overlay_path: Overlay YAML deep-merged on top (default: the packaged
            skill overlay, unless ``AEGIS_CONFIG_OVERRIDE`` is set).
        mode: Directory discovery mode — 'auto', 'dicom', or 'image'.
            Ignored for single-file input (the file's own kind wins).

    Returns:
        The PHI-free run envelope (see :mod:`monai_aegis.envelope`).

    Raises:
        InputError: If the input path does not exist or *mode* is invalid.
    """
    from monai_aegis import reporting

    started = time.monotonic()
    source = Path(input_path).expanduser().resolve()
    if mode not in MODES:
        raise InputError(f"unknown mode {mode!r}. Use one of: {', '.join(MODES)}.")
    if not source.exists():
        raise InputError(f"no such file or directory: {source}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        targets = [(str(source), classify_input(str(source)))]
        input_dir = str(source.parent)
        message = ""
    else:
        targets = discover_inputs(str(source), mode)
        input_dir = str(source)
        wanted = {"auto": "DICOM or image", "dicom": "DICOM", "image": "image (.jpg/.jpeg/.png)"}
        message = f"No {wanted[mode]} files found in {source}." if not targets else ""

    config = load_run_config(config_path, overlay_path)
    resolved_config_path = config_path or default_config_path()
    resolved_overlay = resolve_overlay_path(overlay_path)

    has_dicom = any(kind == "dicom" for _path, kind in targets)
    accumulator = reporting.GroundTruthAccumulator(config, include_tag_actions=has_dicom)

    pipelines: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    for path, kind in targets:
        name = os.path.basename(path)
        try:
            if kind not in pipelines:
                pipelines[kind] = _build_pipeline(kind, resolved_config_path, input_dir, str(out_dir), resolved_overlay)
            result = pipelines[kind]({"image": path})
            accumulator.collect(result)
            entries.append(
                envelope.file_entry(
                    name,
                    kind,
                    envelope.summarize_result(result),
                    envelope.collect_artifacts(result),
                )
            )
            logger.info("Processed %s (%s)", name, kind)
        except Exception as exc:  # per-file: record and continue (P4: message only)
            logger.exception("Failed on %s", name)
            entries.append(envelope.failed_file_entry(name, kind, f"{type(exc).__name__}: {exc}"))

    report_paths: List[str] = []
    if targets and reporting.is_enabled(config):
        accumulator.flush(str(out_dir))
        for filename in (reporting.PIXEL_DETECTIONS_FILE, reporting.TAG_ACTIONS_FILE):
            candidate = out_dir / filename
            if candidate.is_file():
                report_paths.append(str(candidate))

    return envelope.build_envelope(
        input_path=str(source),
        output_dir=str(out_dir),
        mode=mode,
        files=entries,
        reports=report_paths,
        elapsed_seconds=time.monotonic() - started,
        message=message,
    )


__all__ = [
    "IMAGE_EXTENSIONS",
    "MODES",
    "InputError",
    "default_config_path",
    "skill_overlay_path",
    "resolve_overlay_path",
    "load_run_config",
    "classify_input",
    "discover_dicom_files",
    "discover_image_files",
    "discover_inputs",
    "deidentify",
]
