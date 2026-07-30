---
name: aegis-deid
description: >-
  De-identify medical images (DICOM, JPEG, PNG) with the Aegis pipeline —
  remove PHI from burnt-in pixel text and DICOM headers, run batch jobs,
  audit runs against ground-truth reports, and manage the manual-review
  queue. Use when the user mentions de-identification, anonymization,
  PHI/PII removal, DICOM scrubbing, or preparing medical imaging data for
  sharing, research, or AI training.
---

# Aegis Medical Image De-identification

Aegis removes Protected Health Information (PHI) from medical images on two
fronts: burnt-in pixel text (EasyOCR detection + Stanford NER classification)
and DICOM header metadata (PS3.15 Basic Confidentiality Profile scrubbing with
deterministic tokenization). It handles single files, whole directories, and
series-aware DICOM volumes.

## PHI safety rules (non-negotiable)

The files this pipeline touches contain real patient data. Regardless of what
the user asks:

1. **Never read the contents of input files** in `staging_input/` or any
   user-supplied input directory — not with Read, `cat`, `pydicom` scripts, or
   image viewing. Refer to them by file name and path only.
2. **Never read `aegis_pixel_detections_PHI.csv`** (written only when
   `reporting.include_phi_text` is enabled) and never echo verbatim OCR text
   from any source into the conversation.
3. The standard reports (`aegis_pixel_detections.csv`, `aegis_tag_actions.csv`,
   `job_summary_*.json`) are PHI-free by design and safe to read and quote.
4. Do not view or render pixel data of *input* images. De-identified *outputs*
   may be viewed only if the user explicitly asks to verify redaction.
5. When listing the review queue, report file names only — never open the
   quarantined files.

The MCP tools below enforce these invariants server-side; the rules exist so
you uphold them on the CLI/filesystem path too.

## Prerequisites

The Aegis checkout location must be known. Resolve it in this order: the
`AEGIS_ROOT` environment variable; the current workspace if it contains
`aegis_mcp_server.py`; otherwise ask the user. One-time setup inside it:

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
python scripts/prefetch_models.py --config src/monai_aegis/config/config.yaml  # optional: offline runs
```

## Preferred path: MCP tools

If the `aegis` MCP server is connected, use its tools — they keep all PHI out
of the context window:

| Tool | Use |
|------|-----|
| `warm_up()` | Call **first** in a session; preloads OCR + NER models |
| `deidentify_file(input_path, output_dir="")` | One DICOM/JPEG/PNG, synchronous |
| `start_batch_job(input_dir, output_dir="", mode="auto")` | Directory-scale async job; returns `job_id`. `mode`: `auto` / `dicom` / `image` |
| `get_job_status(job_id)` | Poll batch progress (state, counts, errors) |
| `summarize_run(run_dir, source_filename="")` | Audit a run from its ground-truth reports |
| `list_review_queue()` | Quarantined low-confidence files (names only) |

Typical workflow: `warm_up` → `deidentify_file` or `start_batch_job` → poll
`get_job_status` until `completed` → `summarize_run` on the output dir →
`list_review_queue` for anything quarantined. Batch outputs land in
`staging_output/mcp/<job_id>/`.

If the MCP server is configured but failing to start, the usual cause is an
unset `AEGIS_ROOT` environment variable or a missing venv — check those before
debugging further.

## Fallback path: CLI

When the MCP server is not connected, run the pipeline directly from the Aegis
root (activate `venv` first):

```bash
# Unified orchestrator — DICOM + images, series-aware (recommended).
# --config defaults to the packaged config.yaml and works from any directory;
# pass --config only to point at a custom overlay.
aegis-pipeline --mode auto

# Single pipelines
python -m monai_aegis.dicom_runner                    # series mode
python -m monai_aegis.dicom_runner --mode single
python -m monai_aegis.image_runner
```

Inputs are read from `staging_input/` (override with `AEGIS_INPUT_DIR`).
Outputs go to `staging_output/dicom/<timestamp>/` or
`staging_output/image/<timestamp>/` (override with `AEGIS_OUTPUT_DIR`).
Low-confidence or failed files are moved to `staging_not_processed/` for
manual review.

## Interpreting results

Each run directory contains the cleaned files plus PHI-free ground-truth
reports:

- **`aegis_pixel_detections.csv`** — one row per OCR region: bounding box,
  frame index, salted `text_token` (never verbatim text), confidence, and
  `decision` (`redacted` / `safelisted` / `low_confidence`).
- **`aegis_tag_actions.csv`** — one row per header tag: tag, keyword, action
  (`REMOVE` / `DUMMY` / `ZERO` / `KEEP` / `ATTEST`), and original vs.
  de-identified `SOPInstanceUID`. `ATTEST` rows are the PS3.15 attestation
  stamps Aegis writes, not scrubbed PHI.

The **original** `SOPInstanceUID` is preserved in both reports as the join key
for validating against external ground truth.

When summarizing a run for the user, report: files processed, redacted vs.
safelisted vs. low-confidence pixel detections, header tags acted on, and any
files sent to the review queue.

## Configuration tuning

All behavior lives in `monai_aegis/config/config.yaml` (supports
`${VAR:default}` env interpolation):

- **`pii_mapping`** — per-tag header actions (`DUMMY`/`REMOVE`/`ZERO`/`KEEP`);
  applied recursively into nested sequences; invalid entries fail at startup.
- **`ner.clinical_allowlist` / `clinical_patterns`** — terms and regexes never
  redacted (device names, probe IDs). Add here to fix false positives.
- **`ner.phi_heuristic_patterns`** — regexes force-redacted. Add here to fix
  false negatives.
- **`ocr.confidence_threshold`** — below it, files route to manual review.
- **`storage.protocol`** — `file` / `s3` / `gs` / `az` for cloud I/O; use a
  `config.prod.yaml` overlay via `AEGIS_CONFIG_OVERRIDE`.
- **Offline mode** — `AEGIS_HF_OFFLINE=true` and `AEGIS_MODEL_DOWNLOADS=false`
  after prefetching models.
