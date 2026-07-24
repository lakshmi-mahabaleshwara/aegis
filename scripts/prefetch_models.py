#!/usr/bin/env python
"""Prefetch all Aegis model weights for offline / air-gapped deployment.

Downloads, at their **pinned revisions**, everything the pipeline needs at
runtime so it can then run with zero network access:

  1. The Stanford de-identifier NER model (HuggingFace) at the exact
     ``ner.model_revision`` commit from config.yaml → local HF cache
     (or ``HF_HOME`` if set).
  2. The EasyOCR detection + recognition weights for the configured
     languages → ``ocr.model_storage_directory`` (or EasyOCR's default).

Typical uses::

    # Populate caches on a connected machine / in a Docker build stage
    python scripts/prefetch_models.py --config src/monai_aegis/config/config.yaml

    # Then run fully offline:
    export AEGIS_HF_OFFLINE=true AEGIS_MODEL_DOWNLOADS=false
    aegis-pipeline --config src/monai_aegis/config/config.yaml

For a portable bundle, set HF_HOME and --easyocr-dir to a directory you
ship to the air-gapped host, and point AEGIS_OCR_MODEL_DIR/HF_HOME at it
there.
"""
import argparse
import logging
import sys

from monai_aegis.config.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prefetch_models")


def prefetch_ner(config: dict) -> None:
    ner_cfg = config.get("ner", {})
    if not ner_cfg.get("enabled", False):
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
    languages = ocr_cfg.get("languages", ["en"])
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="src/monai_aegis/config/config.yaml",
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
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.skip_ner:
        prefetch_ner(config)
    if not args.skip_ocr:
        prefetch_easyocr(config, args.easyocr_dir)

    logger.info(
        "Done. For air-gapped runs set: AEGIS_HF_OFFLINE=true AEGIS_MODEL_DOWNLOADS=false"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
