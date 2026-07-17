#!/usr/bin/env python
"""Prefetch all Aegis model weights for offline / air-gapped deployment.

Downloads, at their **pinned revisions**, everything the pipeline needs at
runtime so it can then run with zero network access:

  1. The Stanford de-identifier NER model (HuggingFace) at the exact
     ``ner.model_revision`` commit from config.yaml → local HF cache
     (or ``HF_HOME`` if set).
  2. The EasyOCR detection + recognition weights for the configured
     languages → ``ocr.model_storage_directory`` (or EasyOCR's default).
  3. Optional TrOCR handwriting weights when ``ocr.handwriting.enabled``.
  4. Optional Florence-2 VLM weights when ``ocr.vlm.enabled``.

Typical uses::

    # Populate caches on a connected machine / in a Docker build stage
    python scripts/prefetch_models.py --config monai_aegis/config/config.yaml

    # Then run fully offline:
    export AEGIS_HF_OFFLINE=true AEGIS_MODEL_DOWNLOADS=false
    aegis-pipeline --config monai_aegis/config/config.yaml

For a portable bundle, set HF_HOME and --easyocr-dir to a directory you
ship to the air-gapped host, and point AEGIS_OCR_MODEL_DIR/HF_HOME at it
there.
"""
import argparse
import logging
import sys

from monai_aegis.config.config_loader import as_bool, load_config
from monai_aegis.transforms.language_presets import resolve_ocr_languages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prefetch_models")


def prefetch_ner(config: dict) -> None:
    ner_cfg = config.get("ner", {})
    if not as_bool(ner_cfg.get("enabled"), default=False):
        logger.info("NER disabled in config — skipping HuggingFace prefetch")
        return

    from huggingface_hub import snapshot_download

    # Resolve model + revision exactly as the runtime classifier does, so
    # the prefetched snapshot is the one the pipeline will actually load.
    from monai_aegis.transforms.ner_classifier import PHIClassifier

    classifier = PHIClassifier(config)
    model_name, revision = classifier.model_name, classifier.model_revision
    logger.info("Prefetching NER model %s (revision: %s)...", model_name, revision or "default")
    path = snapshot_download(repo_id=model_name, revision=revision)
    logger.info("NER model cached at: %s", path)


def prefetch_easyocr(config: dict, easyocr_dir: str | None) -> None:
    import easyocr

    ocr_cfg = config.get("ocr", {})
    languages = resolve_ocr_languages(ocr_cfg)
    target_dir = easyocr_dir or ocr_cfg.get("model_storage_directory") or None
    logger.info(
        "Prefetching EasyOCR weights for %s into %s...",
        languages, target_dir or "default cache (~/.EasyOCR)",
    )
    # Instantiating the Reader with downloads enabled fetches the weights.
    easyocr.Reader(
        languages,
        gpu=False,
        model_storage_directory=target_dir,
        download_enabled=True,
        verbose=False,
    )
    logger.info("EasyOCR weights ready in: %s", target_dir or "~/.EasyOCR/model")


def prefetch_handwriting(config: dict) -> None:
    hw = config.get("ocr", {}).get("handwriting", {})
    if not as_bool(hw.get("enabled"), default=False):
        logger.info("Handwriting disabled in config — skipping TrOCR prefetch")
        return

    from huggingface_hub import snapshot_download

    model_name = hw.get("model_name", "microsoft/trocr-base-handwritten")
    revision = hw.get("model_revision") or None
    logger.info("Prefetching handwriting model %s (revision: %s)...", model_name, revision or "default")
    path = snapshot_download(repo_id=model_name, revision=revision)
    logger.info("Handwriting model cached at: %s", path)


def prefetch_vlm(config: dict) -> None:
    vlm = config.get("ocr", {}).get("vlm", {})
    if not as_bool(vlm.get("enabled"), default=False):
        logger.info("VLM disabled in config — skipping Florence-2 prefetch")
        return

    from huggingface_hub import snapshot_download

    model_name = vlm.get("model_name", "microsoft/Florence-2-base")
    revision = vlm.get("model_revision") or None
    logger.info("Prefetching VLM %s (revision: %s)...", model_name, revision or "default")
    path = snapshot_download(repo_id=model_name, revision=revision)
    logger.info("VLM cached at: %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="monai_aegis/config/config.yaml",
        help="Path to config.yaml (source of model name, revision, languages)",
    )
    parser.add_argument(
        "--easyocr-dir",
        default=None,
        help="Override target directory for EasyOCR weights "
             "(default: ocr.model_storage_directory from config, else ~/.EasyOCR)",
    )
    parser.add_argument("--skip-ner", action="store_true", help="Skip the HuggingFace NER model")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip the EasyOCR weights")
    parser.add_argument("--skip-handwriting", action="store_true", help="Skip TrOCR weights")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip Florence-2 VLM weights")
    parser.add_argument(
        "--force-optional",
        action="store_true",
        help="Prefetch handwriting + VLM even when disabled in config",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.skip_ner:
        prefetch_ner(config)
    if not args.skip_ocr:
        prefetch_easyocr(config, args.easyocr_dir)

    if args.force_optional:
        # Temporarily enable so the prefetch helpers run.
        config.setdefault("ocr", {}).setdefault("handwriting", {})["enabled"] = True
        config.setdefault("ocr", {}).setdefault("vlm", {})["enabled"] = True

    if not args.skip_handwriting:
        prefetch_handwriting(config)
    if not args.skip_vlm:
        prefetch_vlm(config)

    logger.info(
        "Done. For air-gapped runs set: AEGIS_HF_OFFLINE=true AEGIS_MODEL_DOWNLOADS=false"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
