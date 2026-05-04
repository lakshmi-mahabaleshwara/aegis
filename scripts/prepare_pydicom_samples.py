#!/usr/bin/env python3
"""
Prepare the pydicom built-in samples used by the AEGIS benchmark.

Walks pydicom's installed test-data tree, applies the filters declared in
``tests/benchmark/public_dataset/datasets.yaml`` (entry: ``pydicom_samples``),
and copies matching ``.dcm`` files to the target directory. Re-runnable —
existing files are overwritten.

Usage
-----
::

    # Default — uses the registry shipped with the repo.
    python scripts/prepare_pydicom_samples.py

    # Different registry / different entry / different target.
    python scripts/prepare_pydicom_samples.py \\
        --registry tests/benchmark/public_dataset/datasets.yaml \\
        --dataset  pydicom_samples \\
        --target   /tmp/pydicom_samples

The registry entry must use ``source.type: pydicom_builtin`` and may
declare these filter fields::

    filters:
      accepted_modalities: [CT, MR, US, CR, DX, MG, PT, XA, RF, OT]
      require_pixel_data: true     # skip files without a Rows tag > 0
      file_suffix: .dcm
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("prepare_pydicom_samples")

# ---------------------------------------------------------------------------
# Defaults — used if the registry file is missing or the entry omits a field.
# Keeps the script runnable even on a stripped-down checkout.
# ---------------------------------------------------------------------------
DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / (
    "tests/benchmark/public_dataset/datasets.yaml"
)
DEFAULT_DATASET = "pydicom_samples"
DEFAULT_TARGET = "tests/benchmark/public_dataset/pydicom_samples"
DEFAULT_FILTERS: Dict[str, Any] = {
    "accepted_modalities": ["CT", "MR", "US", "CR", "DX", "MG", "PT", "XA", "RF", "OT"],
    "require_pixel_data": True,
    "file_suffix": ".dcm",
}


# ═══════════════════════════════════════════════════════════════════════════
# Config resolution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrepConfig:
    """Resolved configuration for one preparation run."""

    target_dir: Path
    accepted_modalities: set
    require_pixel_data: bool
    file_suffix: str

    @classmethod
    def from_registry_entry(cls, entry: Dict[str, Any], target_override: Optional[str]) -> "PrepConfig":
        filters = {**DEFAULT_FILTERS, **(entry.get("filters") or {})}
        target = target_override or entry.get("target_dir") or DEFAULT_TARGET
        return cls(
            target_dir=Path(target).resolve(),
            accepted_modalities={m.upper() for m in filters["accepted_modalities"]},
            require_pixel_data=bool(filters["require_pixel_data"]),
            file_suffix=str(filters["file_suffix"]),
        )


def load_registry_entry(registry_path: Path, dataset_key: str) -> Dict[str, Any]:
    """Read a single dataset entry from the YAML registry.

    Returns an empty dict when the registry file is missing — callers fall
    back to ``DEFAULT_*`` constants in that case.
    """
    if not registry_path.is_file():
        logger.warning("Registry %s not found — using built-in defaults.", registry_path)
        return {}

    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}

    datasets = registry.get("datasets") or {}
    if dataset_key not in datasets:
        raise SystemExit(
            f"Dataset '{dataset_key}' not found in registry {registry_path}. "
            f"Available: {sorted(datasets.keys())}"
        )

    entry = datasets[dataset_key]
    source_type = (entry.get("source") or {}).get("type")
    if source_type != "pydicom_builtin":
        raise SystemExit(
            f"Dataset '{dataset_key}' has source.type='{source_type}'. "
            f"This script only handles 'pydicom_builtin'."
        )
    return entry


# ═══════════════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════════════

def iter_pydicom_test_files(data_root: Path, suffix: str):
    """Yield every file under pydicom's data root that ends with ``suffix``."""
    for root, _, files in os.walk(data_root):
        for name in files:
            if name.endswith(suffix):
                yield Path(root) / name


def should_include(path: Path, cfg: PrepConfig) -> Tuple[bool, str]:
    """Return ``(include, reason)`` for a candidate file.

    Reason is the modality or a short code (``no_pixel``, ``unreadable``)
    used only for logging.
    """
    import pydicom  # local import — keeps --help fast and avoids hard dep at import time

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception as exc:  # pydicom raises a wide variety of errors here
        return False, f"unreadable ({exc.__class__.__name__})"

    modality = str(getattr(ds, "Modality", "")).upper()
    if modality not in cfg.accepted_modalities:
        return False, f"skip_modality={modality or 'NONE'}"

    if cfg.require_pixel_data and int(getattr(ds, "Rows", 0) or 0) <= 0:
        return False, "no_pixel"

    return True, modality


# ═══════════════════════════════════════════════════════════════════════════
# Copy step
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PrepReport:
    target_dir: Path
    copied: int = 0
    skipped: int = 0
    by_modality: Dict[str, int] = field(default_factory=dict)
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def record_copy(self, modality: str) -> None:
        self.copied += 1
        self.by_modality[modality] = self.by_modality.get(modality, 0) + 1

    def record_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def prepare(cfg: PrepConfig, dry_run: bool = False) -> PrepReport:
    import pydicom.data  # local import for the same reason as above

    data_root = Path(pydicom.data.DATA_ROOT)
    logger.info("pydicom data root: %s", data_root)
    logger.info("Target directory : %s", cfg.target_dir)

    if not dry_run:
        cfg.target_dir.mkdir(parents=True, exist_ok=True)

    report = PrepReport(target_dir=cfg.target_dir)

    for src in iter_pydicom_test_files(data_root, cfg.file_suffix):
        ok, reason = should_include(src, cfg)
        if not ok:
            report.record_skip(reason)
            continue

        dst = cfg.target_dir / src.name
        if not dry_run:
            shutil.copy2(src, dst)
        report.record_copy(reason)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pydicom built-in test data into the AEGIS benchmark fixture directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY),
        help="Path to dataset registry YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="Registry key for the dataset entry to prepare (default: %(default)s).",
    )
    parser.add_argument(
        "--target", default=None,
        help="Override the registry's target_dir.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Walk and filter without copying anything.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce logging to WARNING.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    entry = load_registry_entry(Path(args.registry), args.dataset)
    cfg = PrepConfig.from_registry_entry(entry, args.target)
    report = prepare(cfg, dry_run=args.dry_run)

    print("\n" + "=" * 72)
    print(f"PYDICOM SAMPLES — {'dry run' if args.dry_run else 'prepared'}")
    print("=" * 72)
    print(f"  Target        : {report.target_dir}")
    print(f"  Copied        : {report.copied}")
    print(f"  Skipped       : {report.skipped}")
    if report.by_modality:
        print("  By modality   : " + ", ".join(
            f"{m}={n}" for m, n in sorted(report.by_modality.items())
        ))
    if report.skip_reasons and not args.quiet:
        top = sorted(report.skip_reasons.items(), key=lambda kv: -kv[1])[:5]
        print("  Top skips     : " + ", ".join(f"{r}={n}" for r, n in top))
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
