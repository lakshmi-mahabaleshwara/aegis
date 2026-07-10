# Aegis

**Medical Image De-identification Pipeline with OCR, NER, and Series-Aware Volume Processing**

Aegis is a production-ready pipeline for de-identifying medical images (DICOM series, individual DICOMs, JPEG, PNG) by removing Protected Health Information (PHI) while preserving critical clinical markers, image quality, and geometric metadata.

### Use Cases

- Preparing anonymized medical imaging datasets for AI training
- Batch de-identification of DICOM studies
- Reproducible preprocessing pipelines for research datasets
- Extensible transforms for modality-specific workflows (e.g., ultrasound)
- Cloud-native processing with S3, GCS, or Azure storage backends

---

## 🌟 Features

### DICOM De-identification
- **PS3.15 Basic Application Level Confidentiality Profile** — metadata scrubbing follows the DICOM standard with REMOVE / DUMMY / ZERO actions, configurable via `pii_mapping` in `config.yaml`
- **Recursive sequence scrubbing** — PII actions are applied to nested DICOM sequences (not just top-level tags)
- **Deterministic tokenization** — patient metadata (PatientName, PatientID) is replaced with reproducible hash tokens via `AegisIdentityManager`, enabling re-linkage when the original salt is available
- **Private tag removal** — all vendor-private tags are stripped for enhanced privacy
- **Content-driven discovery** — DICOM files are detected by reading the 132-byte preamble (`DICM` magic), not by file extension, so extensionless hospital exports are found automatically

### Pixel Redaction
- **Stanford NER-based PHI detection** — semantic classification via `stanford-deidentifier-base` (F1 ≥ 97.9%)
- **3-layer classification pipeline**: clinical allowlist → PHI heuristics → NER model
- **EasyOCR-based text detection** with configurable confidence thresholds
- **US fan-geometry scoped OCR** — reads `SequenceOfUltrasoundRegions` (0018,6011) from DICOM metadata to restrict OCR to non-diagnostic zones, eliminating false positives in the clinical fan area
- **Pixel-level union masking** — for multi-frame volumes, PHI detected on any sampled keyframe is propagated and zeroed across all frames, ensuring no PHI survives between slices
- **Adaptive keyframe sampling** — instead of scanning every frame, samples a configurable percentage (default 10%, minimum 3, maximum 25) and always includes the first and last frames
- **Low-confidence routing** — images with uncertain OCR are flagged for manual review
- **Regex safelist fallback** — preserved for environments without NER model
- **Fully configurable** — PHI labels, clinical allowlist, clinical patterns, and PHI heuristics all defined in `config.yaml`

### Pipeline Architecture
- **DICOM pipeline** (`monai_aegis.dicom_runner`) — DICOM single-file or series-aware volume mode
- **Image pipeline** (`monai_aegis.image_runner`) — JPEG/PNG single-file or series-aware folder processing
- **Unified orchestrator** (`monai_aegis.cli`) — `--mode auto|dicom|image`
- **Independent scaling** — each pipeline can run on separate infrastructure

### Ground-Truth Reporting & Validation
- **Per-run "ground truth" CSVs** — when `reporting.save_ground_truth: true` (the default), each run records exactly what Aegis detected and redacted, so you can validate output against a known answer key
- **`aegis_pixel_detections.csv`** — one row per OCR region: original/de-identified `SOPInstanceUID`, bounding box (`x, y, w, h`), frame index, OCR text, confidence, and decision (`redacted` / `safelisted` / `low_confidence`)
- **`aegis_tag_actions.csv`** — one row per scrubbed DICOM header tag: tag, keyword, action (`REMOVE` / `DUMMY` / `ZERO`), and the original/de-identified `SOPInstanceUID`
- **Stable join key** — the *original* `SOPInstanceUID` is preserved in both files (even though Aegis regenerates it on output), so reports join directly to external ground truth that keys on the source UID
- **Bring-your-own-database mode** — set `reporting.save_ground_truth: false` to skip CSV writing and instead pull the same per-file records straight from the pipeline data dict via `monai_aegis.reporting.extract_records(...)`

### Cloud Storage (fsspec)
- **Pluggable backends** — local filesystem, S3, GCS, Azure via `fsspec`
- **Byte-stream I/O** — no temp files for cloud reads/writes
- **Environment-aware config** — `${VAR_NAME:default}` interpolation with overlay support

---

## ⚖️ Standards Compliance

Aegis implements the **DICOM PS3.15 Basic Application Level Confidentiality
Profile** (Annex E) for DICOM metadata de-identification, with the following
options enabled:

- **Basic Application Level Confidentiality Profile** (mandatory baseline)
- **Clean Pixel Data Option** — OCR + NER-based burnt-in PHI removal
- **Retain Longitudinal Temporal Information with Modified Dates Option**
  (dates zeroed, not removed, to preserve relative temporal relationships)

All DICOM UIDs (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID) are
replaced with deterministically generated tokens, enabling re-linkage when the original salt is available.

HIPAA Safe Harbor de-identification is satisfied as a consequence of
PS3.15 Basic Profile compliance, since the 18 HIPAA identifiers are a
subset of the tags addressed by the Basic Profile.

---

## 🔧 Architecture

### Pipeline Flow

![Architecture Diagram](docs/architecture_diagram.png)

The pipeline processes files through two distinct paths:

**DICOM Path** — Load DICOM → (if ultrasound: US Region Masking to scope OCR to annotation areas) → Pixel Redaction (OCR + NER on full image or scoped to non-diagnostic zones) → Metadata Scrub (tokenize patient IDs, remove private tags) → Save

**Image Path** — Load Image → Pixel Redaction (OCR + NER on full image) → Save. No metadata scrub since JPEG/PNG files have no DICOM tags.

Low-confidence OCR detections on either path are routed to `staging_not_processed/` for manual review.

### Component Overview

![Component Diagram](docs/component_diagram.png)

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Virtual environment recommended

### Install Package
```bash
git clone <repo-url>
cd aegis

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e monai_aegis/
```

---

## 🚀 Quick Start

### Command-Line Runners
```bash
# DICOM pipeline — series mode (default)
python -m monai_aegis.dicom_runner --config monai_aegis/config/config.yaml

# DICOM pipeline — single-file mode
python -m monai_aegis.dicom_runner --config monai_aegis/config/config.yaml --mode single

# Image pipeline — series mode (default)
python -m monai_aegis.image_runner --config monai_aegis/config/config.yaml

# Image pipeline — single-file mode
python -m monai_aegis.image_runner --config monai_aegis/config/config.yaml --mode single

# Unified orchestrator — runs both DICOM and Image pipelines
aegis-pipeline --config monai_aegis/config/config.yaml --mode auto

# Output:
#   Processed files → staging_output/dicom/<timestamp>/ or staging_output/image/<timestamp>/
#   Rejected/flagged files → staging_not_processed/ (manual review, geometry mismatch, or transform errors)
```

---

## ⚙️ Configuration

Edit `monai_aegis/config/config.yaml`. Supports `${VAR_NAME:default}` environment variable interpolation:

```yaml
paths:
  input_dir: '${AEGIS_INPUT_DIR:staging_input}'
  output_dir: '${AEGIS_OUTPUT_DIR:staging_output}'
  not_processed_dir: '${AEGIS_REVIEW_DIR:staging_not_processed}'
  dicom_folder: 'dicom'
  image_folder: 'image'
  timestamp_format: '%Y-%m-%d_%H-%M'

runtime:
  dataloader_num_workers: 0

reporting:
  save_ground_truth: true   # true → write aegis_pixel_detections.csv + aegis_tag_actions.csv
                            #        to the run output directory
                            # false → no CSVs; read records from the data dict yourself

storage:
  protocol: '${AEGIS_STORAGE_PROTOCOL:file}'   # file, s3, gs, az
  options: {}                                   # fsspec options (credentials, etc.)

tokenization:
  salt: '${AEGIS_TOKEN_SALT:}'

series:
  enabled: true
  accepted_modalities: ['CT', 'MR', 'US', 'CR', 'DX', 'MG', 'PT', 'XA', 'RF', 'OT']
  keyframe_sampling:
    sample_pct: 0.10    # Sample 10% of frames as keyframes
    floor: 3             # Minimum keyframes (below this, scan all)
    ceiling: 25          # Maximum keyframes for very large volumes
  output_structure: 'hierarchical'

us_regions:
  enabled: true                   # Enable US fan-geometry scoped OCR
  zero_outside_regions: true      # Zero diagnostic pixels before OCR

ocr:
  languages: ['en']
  gpu_usage: false
  confidence_threshold: 0.4

ner:
  enabled: true                   # Set false to fall back to regex safelist
  model_name: 'StanfordAIMI/stanford-deidentifier-base'
  device: '${AEGIS_DEVICE:cpu}'   # 'cpu', 'cuda', or 'mps'
  phi_labels:                     # NER entity types treated as PHI
    - PATIENT
    - DOCTOR
    - HOSPITAL
    - DATE
    - IDNUM
    # ... (22 labels total)
  clinical_allowlist:             # Terms never redacted
    - mindray
    - ge
    - adult abd n
    # ... (device manufacturers, imaging modes)
  clinical_patterns:              # Regex for clinical/technical text (preserved)
    - '^(F\s*H?\d|D\s+\d|G\s+\d|FR\s+\d|DR\s+\d)'
    - '^(AP|MI|TIS|SSI)\s+\d'
    # ... (probe IDs, measurements, exam types)
  phi_heuristic_patterns:         # Regex for PHI in short OCR fragments (redacted)
    - '^\d{8}[-/]\d{6}[-/]?\w*$'  # Date-ID combos
    - '(?i)(hospital|clinic|diagnostic|medical)'
    # ... (dates, doctor names, patient IDs)

pii_mapping:
  '(0010, 0010)': 'DUMMY'   # PatientName → TOKEN_xxxxx
  '(0010, 0020)': 'DUMMY'   # PatientID → TOKEN_xxxxx
  '(0010, 0030)': 'REMOVE'  # PatientBirthDate → removed
  '(0008, 0020)': 'ZERO'    # StudyDate → 00000000
  '(0008, 0050)': 'REMOVE'  # AccessionNumber → removed
```


### Cloud Storage (S3 / GCS / Azure)

```yaml
# config.prod.yaml — overlay for cloud deployment
storage:
  protocol: 's3'
  options:
    key: '${AWS_ACCESS_KEY_ID}'
    secret: '${AWS_SECRET_ACCESS_KEY}'

paths:
  input_dir: 's3://my-bucket/incoming'
  output_dir: 's3://my-bucket/deidentified'
```

```bash
# Apply overlay via environment variable
export AEGIS_CONFIG_OVERRIDE=config.prod.yaml
pip install s3fs  # one-time
python -m monai_aegis.dicom_runner --config monai_aegis/config/config.yaml
```

| Backend | Protocol | Extra Package |
|---------|----------|---------------|
| Local   | `file`   | (built-in)    |
| AWS S3  | `s3`     | `s3fs`        |
| GCS     | `gs`     | `gcsfs`       |
| Azure   | `az`     | `adlfs`       |

### Offline / Air-Gapped Deployment

The NER model is **pinned to an exact revision** in `config.yaml`
(`ner.model_revision`), so the model cannot silently change under a
deployment. Both model sources can be forced fully offline — no network
calls at runtime, not even update checks.

```bash
# 1. On a connected machine (or Docker build stage): prefetch pinned weights
python scripts/prefetch_models.py --config monai_aegis/config/config.yaml

# 2. Run fully offline
export AEGIS_HF_OFFLINE=true        # NER: resolve from local HF cache only
export AEGIS_MODEL_DOWNLOADS=false  # EasyOCR: never attempt a download
aegis-pipeline --config monai_aegis/config/config.yaml
```

| Env var | Config key | Effect |
|---------|-----------|--------|
| — | `ner.model_revision` | Exact NER model commit (pinned in config.yaml) |
| `AEGIS_HF_OFFLINE` | `ner.local_files_only` | `true` → zero HuggingFace network calls |
| `AEGIS_OCR_MODEL_DIR` | `ocr.model_storage_directory` | Pre-bundled EasyOCR weights dir |
| `AEGIS_MODEL_DOWNLOADS` | `ocr.download_enabled` | `false` → EasyOCR never downloads |

The Docker image bakes all weights in at build time and sets the offline
flags, so containers run air-gapped by default.

### NER Classification Pipeline

When `ner.enabled: true`, each OCR-detected text goes through a 3-layer pipeline:

| Layer | Source | Action | Example |
|-------|--------|--------|---------|
| 1. Clinical allowlist | `ner.clinical_allowlist` | **Preserve** | `mindray`, `adult abd n` |
| 2. Clinical patterns | `ner.clinical_patterns` | **Preserve** | `FH5.0`, `D 13.0`, `SC5-1N` |
| 3. PHI heuristics | `ner.phi_heuristic_patterns` | **Redact** | `20260117-091825-6AAA` |
| 4. Stanford NER | `ner.phi_labels` | **Redact if PHI** | Names, remaining identifiers |

### PII Mapping Actions
| Action | Behavior | Example |
|--------|----------|---------|
| `DUMMY` | Replace with deterministic hash token | `John Doe` → `TOKEN_a1b2c3d4e5f6` |
| `REMOVE` | Delete the DICOM tag entirely | Tag removed from dataset |
| `ZERO` | Set to zero / empty value | `20250216` → `00000000` |

---

## 📊 Ground-Truth Reporting & Validation

Aegis attacks PHI on two fronts — **burnt-in pixel text** (`RedactPixelPHId`,
OCR + NER) and **header metadata** (`ScrubDicomMetadatad`, `pii_mapping`). When
`reporting.save_ground_truth: true` (default), each run writes one CSV per
stage to the run output directory, capturing exactly what was detected and
redacted. These are Aegis's *own* ground truth and can be diffed against a
synthetic/manual answer key to compute precision/recall.

### Output location
Reports are written alongside the cleaned files in the timestamped run
directory:

```
staging_output/dicom/<timestamp>/
├── <cleaned DICOM files…>
├── aegis_pixel_detections.csv      # burnt-in pixel PHI (RedactPixelPHId)
└── aegis_tag_actions.csv           # header tag scrubbing (ScrubDicomMetadatad)
```
(Image runs write to `staging_output/image/<timestamp>/`. PNG/JPEG inputs have
no DICOM header, so image runs emit only `aegis_pixel_detections.csv` —
`aegis_tag_actions.csv` is not created, and SOP UID columns are blank.)

### `aegis_pixel_detections.csv` — one row per OCR region
| Column | Meaning |
|--------|---------|
| `source_path` | Source file path |
| `original_sop_uid` | SOPInstanceUID of the **source** DICOM (join key) |
| `deid_sop_uid` | Regenerated SOPInstanceUID of the de-identified output |
| `frame_index` | Frame the text was found on (blank for single-frame) |
| `bbox_x, bbox_y, bbox_w, bbox_h` | Detection box (top-left + size) |
| `ocr_text` | Text EasyOCR read |
| `confidence` | OCR confidence |
| `decision` | `redacted`, `safelisted`, or `low_confidence` |

### `aegis_tag_actions.csv` — one row per scrubbed header tag
| Column | Meaning |
|--------|---------|
| `source_path` | Source file path |
| `original_sop_uid` / `deid_sop_uid` | Source vs. output SOPInstanceUID |
| `tag` | DICOM tag, e.g. `(0010,0010)` |
| `keyword` | Tag keyword, e.g. `PatientName` |
| `action` | `REMOVE` / `DUMMY` / `ZERO` |
| `redacted` | `True` when the action de-identified the element |

### Validating against external ground truth
The *original* `SOPInstanceUID` is preserved in both reports (Aegis regenerates
the UID on output, but the source UID is captured before that), so the files
join directly to ground truth keyed on the source UID:

- **Pixel PHI** — join on `original_sop_uid`, match boxes by IoU, then compare
  `decision == "redacted"` against the expected `is_phi`.
- **Header PHI** — join on `original_sop_uid` + `tag` to confirm each expected
  tag was acted on.

### Bring-your-own-database mode
Set `reporting.save_ground_truth: false` to skip CSV writing entirely and pull
the same records from the pipeline data dict yourself:

```python
from monai_aegis import reporting
from monai_aegis.transforms.pipeline import build_pipeline

pipeline = build_pipeline(config_path="monai_aegis/config/config.yaml",
                          output_dir="out", input_dir="in")
result = pipeline({"image": "/path/to/file.dcm"})

pixel_rows, tag_rows = reporting.extract_records(result)   # CSV-ready dict rows
# → persist pixel_rows / tag_rows in your own datastore
```

---

## 🧪 Testing

```bash
# Run unit tests
python -m unittest discover tests/unit -v

# Run integration tests
python -m unittest discover tests/integration -v
```


---

## 📁 Project Structure

```
aegis/
├── monai_aegis/                      # Installable package
│   ├── pyproject.toml                # PEP 621 package config
│   ├── cli.py                        # aegis-pipeline entry point
│   ├── dicom_runner.py               # Packaged DICOM orchestration
│   ├── image_runner.py               # Packaged image orchestration
│   ├── reporting.py                  # Ground-truth CSV reporting (detections + tag actions)
│   ├── config/
│   │   ├── config.yaml               # De-identification + NER + storage + series settings
│   │   ├── config_loader.py          # Env var interpolation + overlay merging
│   │   └── storage.py                # AegisFileSystem (fsspec wrapper)
│   └── transforms/
│       ├── __init__.py                # Public API exports
│       ├── io.py                      # LoadDicomRaw/d, SaveDicom/d, LoadImage/d, SaveImage/d
│       ├── series_io.py               # LoadDicomSeries/d, SaveDicomSeries/d
│       ├── discovery.py               # discover_dicoms, is_dicom_file, group/validate/sort
│       ├── pixel.py                   # RedactPixelPHI/d (OCR + NER + volume keyframe + US mask scoping)
│       ├── us_regions.py              # RedactByUSRegions/d (SequenceOfUltrasoundRegions fan-geometry masking)
│       ├── ner_classifier.py          # PHIClassifier (Stanford NER wrapper)
│       ├── metadata.py                # ScrubDicomMetadata/d (single + series scrub)
│       ├── exceptions.py              # Custom exception hierarchy
│       ├── utility.py                 # AegisIdentityManager + tokenized output path helpers
│       └── pipeline.py                # build_pipeline / build_series_pipeline / build_image_pipeline / build_image_series_pipeline
├── tests/
│   ├── unit/                          # Unit tests
│   │   ├── test_config_loader.py      # Config loading, env vars, overlays
│   │   ├── test_storage.py            # AegisFileSystem, fsspec, byte-stream round-trip
│   │   ├── test_raw_loader.py         # DICOM + image loading transforms
│   │   ├── test_discovery.py          # Discovery, grouping, validation, sorting
│   │   ├── test_series_io.py          # Series load/save transforms
│   │   ├── test_image_series_io.py    # Image series load/save + tokenization
│   │   ├── test_volume_redaction.py   # Volume keyframe OCR
│   │   ├── test_us_regions.py         # US region mask building + pipeline integration
│   │   └── ...                        # Additional test files
│   └── integration/                   # End-to-end tests
├── run_dicom_pipeline.py              # Compatibility wrapper for monai_aegis.dicom_runner
├── run_image_pipeline.py              # Compatibility wrapper for monai_aegis.image_runner
├── run_pipeline.py                    # Compatibility wrapper for monai_aegis.cli
├── aegis_demo.ipynb                   # Interactive walkthrough notebook
├── Dockerfile                         # Container build
├── staging_input/                     # Input files (not tracked)
├── staging_output/                    # Processed output (not tracked)
└── staging_not_processed/             # Low-confidence files for review (not tracked)
```

---


## ⚠️ Error Handling

All transforms use a custom exception hierarchy (defined in `exceptions.py`) that carries **filepath** and **transform name** for diagnostic context:

```
AegisTransformError          ← base (catch-all)
├── DicomLoadError           ← corrupted / unreadable DICOM
├── ImageLoadError           ← unreadable JPEG / PNG
├── PixelRedactionError      ← OCR / NER failure
├── MetadataScrubError       ← tag parsing / pixel injection
├── DicomSaveError           ← write failure
├── SeriesLoadError          ← series assembly / geometry mismatch (NEW)
└── SeriesSaveError          ← series write failure (NEW)
```

**Usage in calling code:**

```python
from monai_aegis.transforms.exceptions import AegisTransformError, DicomLoadError

try:
    result = pipeline({'image': filepath})
except DicomLoadError as e:
    print(f"Bad DICOM: {e.filepath} in {e.transform}")
except AegisTransformError as e:
    print(f"Pipeline error: {e}")  # catches any transform failure
```

---

## 🐳 Docker

```bash
# Build
docker build -t aegis:latest .

# Run DICOM pipeline
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  aegis:latest aegis-pipeline --mode dicom

# Run Image pipeline
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  aegis:latest aegis-pipeline --mode image

# Run tests
docker run --rm aegis:latest \
  python -m unittest discover /app/tests/unit -v
```

See [DOCKER_TESTING.md](DOCKER_TESTING.md) for full Docker guide.

---

## 🤝 Contributing

```bash
# Install dev dependencies
pip install -e "monai_aegis/[dev]"

# Format
black monai_aegis/transforms/ tests/

# Lint
flake8 monai_aegis/transforms/ tests/

# Type check
mypy monai_aegis/transforms/
```

---

## 📄 License

Apache 2.0

---

## 🙏 Acknowledgments

- [MONAI](https://docs.monai.io/) — Medical Open Network for AI
- [StanfordAIMI](https://huggingface.co/StanfordAIMI/stanford-deidentifier-base) — Stanford De-identifier NER model
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — OCR detection library
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — NER pipeline
- [PyDICOM](https://pydicom.github.io/) — DICOM file handling
- [HIPAA De-identification Guidelines](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
