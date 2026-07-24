---
name: aegis-deidentify
description: >-
  Used for removing PHI from medical images (DICOM, JPEG, PNG) — burnt-in
  pixel text (OCR + NER) and DICOM header tags — producing de-identified
  files, PHI-free ground-truth reports, and a JSON summary. Not for clinical
  decisions, diagnosis, or regulatory certification of de-identification.
license: Apache-2.0
allowed-tools: Bash
metadata:
  tags: [medtech, dicom, de-identification, phi, hipaa]
---

<!--
  Canonical agent-skill card for wrapping Aegis in any catalog. Copy this
  file into your catalog's skill directory and adjust paths/commands to your
  install. The scope and limitations text is authored to be reused verbatim;
  see docs/skills/DISCLAIMER.md. Keep the guardrails intact.
-->

# Aegis De-identify

## Purpose
- Remove PHI from a medical image file (DICOM/JPEG/PNG) or a directory of
  them: burnt-in pixel text via OCR + Stanford NER, and DICOM header tags
  via PS3.15 metadata scrubbing with deterministic tokenization.
- Emit a PHI-free JSON envelope plus ground-truth CSV reports for auditing.
- Run the documented command; do not hand-write an inference loop or edit
  the pipeline internals.

## Instructions
- Install: `pip install monai-aegis` (add `[mcp]` or `[cloud]` extras as
  needed). Fix `AEGIS_TOKEN_SALT` for reproducible, re-linkable output.
- Run `aegis-deidentify <input> --output-dir <out>`. Read the JSON envelope
  from stdout; logs go to stderr. If a host exposes `run_script`, use
  `run_script("aegis-deidentify", args=[input, "--output-dir", out])`.
- Then verify the result: `aegis-verify <out>` returns a PHI-free pass/fail
  report. Treat a run as reviewed only after verification passes.
- Never print detected PHI text, and never read the opt-in
  `aegis_pixel_detections_PHI.csv`. Files routed to manual review
  (exit code 4) must be surfaced, not silently accepted.

## Available Scripts
| Script | Purpose | Arguments |
|---|---|---|
| `aegis-deidentify` | De-identify a file or directory; emit an envelope. | `INPUT --output-dir OUT [--overlay YAML] [--mode auto\|dicom\|image]` |
| `aegis-verify` | Verify a de-identified run against a checklist. | `RUN_DIR [--checklist YAML]` |
| `aegis-fixture` | Generate a synthetic fixture (no real data). | `OUT.dcm [--text LINE ...] [--with-private-tag]` |

## Prerequisites
- Python ≥ 3.9. Models: pinned Stanford NER (~500 MB) + EasyOCR weights,
  downloaded on first run unless run offline.
- Environment: `AEGIS_TOKEN_SALT` (reproducibility), `AEGIS_DEVICE`
  (`cpu`/`cuda`/`mps`), and for air-gapped runs `AEGIS_HF_OFFLINE=true` +
  `AEGIS_MODEL_DOWNLOADS=false` after prefetching weights.
- Customize behavior with `--overlay your.yaml` (or `AEGIS_CONFIG_OVERRIDE`),
  never with new flags.

## Limitations
<!-- Reuse docs/skills/DISCLAIMER.md verbatim. Short form: -->
- Engineering de-identification, not a clinical or regulatory certification.
- OCR recall is not 100%; low-confidence detections are routed to manual
  review rather than silently passed.
- Header scrubbing covers the configured `pii_mapping`; a site with
  different requirements must supply its own mapping and matching checklist.
- Not for clinical decisions, diagnosis, triage, or patient-facing use.

## Troubleshooting
| Error | Cause | Fix |
|---|---|---|
| Non-zero exit, envelope `status: error` | Bad input path, or models missing while downloads disabled. | Check the path; prefetch models or allow downloads. |
| Exit code 3 (partial) | Some files failed; others succeeded. | Read `errors` and per-file `status` in the envelope. |
| Exit code 4 | Output produced, but files were routed to manual review. | Inspect the review queue; do not treat as fully clean. |
| `aegis-verify` fails | Output violates a checklist assertion (e.g. a PHI tag survived). | Read `findings`; fix config/mapping and re-run. |
