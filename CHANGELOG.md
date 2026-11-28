# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Manifests and skill wrappers pin against these releases, so behavior changes
that affect output are called out explicitly under **Changed**.

## [Unreleased]

### Fixed
- **Burnt-in PHI was not redacted in multi-frame DICOMs (cine loops).**
  A multi-frame file is its own one-file series, so it runs through the
  single-file pipeline, whose loader (`LoadDicomRaw`) only reshaped 2D and
  single-frame RGB arrays. RGB cine loops `(frames, H, W, 3)` were then
  read with the wrong axes: keyframe OCR scanned image rows instead of
  frames, never saw the overlay text, and the file was written with 0
  pixels changed, not sent to review, and still stamped as pixel-redacted.
  Grayscale cine loops crashed EasyOCR and went to review. The loader now
  converts both to `(C, frames, H, W)` volumes using `SamplesPerPixel`
  (as pydicom does), the header-scrub step writes them back as
  multi-frame, and any other layout fails closed. `RedactPixelPHI` also
  rejects a 4-D array whose channel axis is not 1 or 3, so a mis-shaped
  volume is quarantined instead of passed through. Affects the runners
  (both modes), `aegis-deidentify` and the MCP server.
- **Partial names and times were left visible.** OCR returns a text line
  as separate word groups ("Robinson," / "Brandy S.", "08/01/2021" /
  "23:16"), and each was classified alone, without the context that makes
  it PHI, so a first name or a time beside a redacted surname or date was
  kept. Fragments on the same line are now also classified together
  (`ner.line_context`, default on): PHI spans are located in the joined
  line and every fragment they cover is redacted. It only adds redaction;
  fragments matching the clinical allowlist/patterns stay kept, and
  separate columns (e.g. scanner settings) are not joined. OCR also drops
  very short tokens next to longer ones (the "S." in "Brandy S."), so a
  redaction box now also grows sideways over adjacent pixels of its own
  text colour — up to `ocr.redaction_padding` text heights (default 1.5),
  stopping at a gap wider than a word space and never into kept text.
  Background and image content beside the text (e.g. an ultrasound fan)
  are not touched. The pixel report still records the OCR box.
- **Low-confidence text in a multi-file series no longer slips through.**
  Single files and single images with low-confidence OCR regions went to
  review, but multi-file DICOM series and image folders were released
  regardless. They now follow the same rule: the whole series/folder is
  copied to the review folder, and any output already written for it is
  removed. The series summary log now reports `Not processed`.
- **Burnt-in names were not redacted under transformers 5.** transformers 5
  loads the pinned Stanford NER model's (uncased) tokenizer with
  `do_lower_case=False`, so every capitalized word became `[UNK]` and the
  model stopped detecting patient names, doctors and hospitals; only dates
  and the regex heuristics still fired. transformers is now held to
  `>=4.57.0,<4.58` (pyproject, requirements.txt, Docker constraints), and a
  regression test checks the tokenizer keeps capitalized names.
- **NER label mismatch.** `ner.phi_labels` listed labels the model never
  emits (`DOCTOR`, `IDNUM`, …) while omitting ones it does (`HCW` for
  doctors/staff, `ID` for identifiers), so NER-detected doctors and IDs were
  kept. The list now matches the model (`PATIENT`, `HCW`, `HOSPITAL`, `ID`,
  `DATE`, `PHONE`; `VENDOR` deliberately excluded), and the classifier checks
  it against the loaded model at startup — warning on dead labels and failing
  closed (treating text as PHI) when nothing overlaps.
- **Ground-truth reports are crash-safe.** Report rows now stream to disk per
  file instead of being written once at the end, so an interrupted run keeps
  the rows of every completed file. Applies to the DICOM and image runners,
  `api.deidentify`, and MCP batch jobs.

### Changed
- **Ages in burnt-in text are now redacted** via new `ner.age_patterns`
  (`Age 57Y`, `57 y/o`, `F 57Y`, DICOM `006M`, `M/57`, …). The model has no
  AGE label, so ages were previously always kept. Age patterns run before the
  clinical rules, which would otherwise preserve sex+age overlays such as
  `F 57Y` as an ultrasound frequency setting.
- Pixel redaction output changes as a result of the three items above: more
  regions (names, doctors, hospitals, IDs, ages) are redacted.
- Multi-frame DICOMs now get real OCR, so they take longer and redact
  more: up to 25 actual frames per file go through OCR + NER (about 20 s
  per cine loop on CPU, vs. ~2 s when the frames were misread), and
  burnt-in PHI is removed from every frame. Grayscale cine loops that used
  to land in the review queue with an OCR error are now processed.
- The vendor string written by `notebooks/synthetic_phi_generation.ipynb`
  (`SYNTHETIC-DEID-TUTORIAL`) is no longer redacted from burnt-in text. NER
  tagged its all-caps words as a person (`HCW`); a fully anchored
  `ner.clinical_patterns` entry now keeps exactly that string.

## [0.2.0] - 2026-07-24

### Added
- **Agent-skill surface.** A single-invocation entry point,
  `monai_aegis.api.deidentify(input, output_dir, ...)`, and the
  `aegis-deidentify` console command. One input, one output directory, one
  PHI-free JSON envelope on stdout; behavior stays config-driven.
- **Result envelope contract.** `monai_aegis.envelope` builds a versioned,
  PHI-free run summary validated by the packaged
  `monai_aegis/schemas/envelope.schema.json`, shared by the CLI, the MCP
  server, and any wrapper.
- **Declarative verification.** `monai_aegis.verify` and the `aegis-verify`
  command run an independent second pass over de-identified output, driven
  by a YAML checklist (`monai_aegis/checklists/ps315.yaml` by default) and
  reporting against `monai_aegis/schemas/verification.schema.json`.
  Verification depends only on pydicom + PyYAML. Includes an
  `instance_uids_remapped` check that flags any patient-linkable UID left
  under an original institutional root, closing the loop on the UID-graph
  remap below.
- **Synthetic fixtures.** `monai_aegis.fixtures` and the `aegis-fixture`
  command generate deterministic fake DICOM/PNG/JPEG inputs (burnt-in text
  plus fake header identifiers) for exercising the pipeline without real
  data.
- **Skill config overlay.** `monai_aegis/config/config.skill.yaml`, applied
  by default on the skill surface, pins the behaviors a harness relies on
  (flat caller-owned output, deterministic ordering, ground-truth reports).
- **Packaging extras.** `pip install 'monai-aegis[cloud]'` (S3/GCS/Azure
  fsspec backends) and `'monai-aegis[mcp]'` (the aegis-mcp server).
- **`aegis-mcp` console command.** The MCP server now ships inside the
  package (`monai_aegis.mcp_server`) and is runnable as `aegis-mcp`.

### Changed
- **Deterministic DICOM UID regeneration.** Regenerated SOP Instance UIDs
  are now derived from `(tokenization salt, original UID)` instead of being
  random, so identical input + config + `AEGIS_TOKEN_SALT` reproduce
  byte-identical output. The single-file path, which previously left the
  source SOP Instance UID unchanged, now regenerates it like the series
  path, and `file_meta.MediaStorageSOPInstanceUID` is kept in sync with the
  dataset UID. Downstream consumers that joined on an output file's UID from
  a prior release will see different (now stable) values; the ground-truth
  CSVs are unaffected — they always keyed on the original UID.
- **Full UID-graph remap (PS3.15 action U).** Scrubbing now replaces *every*
  patient-linkable UID — Study, Series, FrameOfReference, and any referenced
  instance UID in nested sequences — not just SOP Instance UID. Previously
  these passed through verbatim, leaving the original institution-issued
  identifiers in the output and re-linkable to the source study. The remap is
  deterministic and consistent: a UID reused across the object (e.g. a
  ReferencedSOPInstanceUID) is rewritten to the same value everywhere, so
  cross-references survive; slices sharing a Study/Series UID receive the same
  replacement. Class- and implementation-identifying UIDs (SOP Class,
  Transfer Syntax, Implementation Class) and anything under the DICOM standard
  root are preserved so the object stays valid. Each remap is recorded in the
  ground-truth tag report as an `action=REMAP` row.
- The MCP server's result summarization now delegates to the shared
  `monai_aegis.envelope` implementation. Tool response shapes are unchanged.
- Behavioral dependencies now carry upper version bounds (pydicom `<4`,
  monai `<2`, easyocr `<2`, transformers `<6`). Minimum supported Python is
  now 3.10.
- **Repository restructure.** Sources moved to a `src/monai_aegis/` layout
  with `pyproject.toml` at the repo root (subpackages are now auto-discovered
  via `packages.find`). Docker files moved to `docker/`, the demo notebook to
  `notebooks/`. Import paths (`monai_aegis.*`) and console commands are
  unchanged. The pipeline runners' `--config` default now resolves the
  packaged `config.yaml` from the package location, so it works from any
  working directory.
- The MCP server implementation now lives in the package
  (`monai_aegis.mcp_server`, runnable as `aegis-mcp`); the root
  `aegis_mcp_server.py` remains as a thin backward-compatible launcher.

### Removed
- Stale `monai_aegis/setup.py`, which had drifted from `pyproject.toml`
  (missing packages and console scripts). `pyproject.toml` is the single
  packaging source of truth.

## [0.1.0]

### Added
- DICOM and image de-identification pipelines: PS3.15 metadata scrubbing
  with deterministic tokenization, OCR + Stanford NER burnt-in PHI
  redaction, series-aware volume processing, US fan-geometry scoped OCR.
- Ground-truth reporting (`aegis_pixel_detections.csv`,
  `aegis_tag_actions.csv`) and PS3.15 attestation stamping.
- fsspec-backed storage (local/S3/GCS/Azure), offline/air-gapped model
  resolution, and the `aegis-pipeline` unified CLI.
- `aegis-mcp` server exposing the pipeline as Model Context Protocol tools.

[0.2.0]: https://github.com/lakshmi-mahabaleshwara/aegis/releases/tag/v0.2.0
[0.1.0]: https://github.com/lakshmi-mahabaleshwara/aegis/releases/tag/v0.1.0
