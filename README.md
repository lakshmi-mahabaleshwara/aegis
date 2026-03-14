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

### Visual De-identification (Pixel Redaction)
- **Stanford NER-based PHI detection** — semantic classification via `stanford-deidentifier-base` (F1 ≥ 97.9%)
- **3-layer classification pipeline**: clinical allowlist → PHI heuristics → NER model
- **EasyOCR-based text detection** with configurable confidence thresholds
- **Regex safelist fallback** — preserved for environments without NER model
- **Low-confidence routing** — images with uncertain OCR are flagged for manual review
- **Redaction mask output** — redacted regions are exposed as side-channel metadata for downstream review
- **Fully configurable** — PHI labels, clinical allowlist, clinical patterns, and PHI heuristics all defined in `config.yaml`

### Separate Pipeline Architecture
- **DICOM pipeline** (`monai_aegis.dicom_runner`) — DICOM single-file or series-aware volume mode
- **Image pipeline** (`monai_aegis.image_runner`) — JPEG/PNG single-file or series-aware folder processing
- **Unified orchestrator** (`monai_aegis.cli`) — `--mode auto|dicom|image`
- **Independent scaling** — each pipeline can run on separate infrastructure

### Cloud Storage (fsspec)
- **Pluggable backends** — local filesystem, S3, GCS, Azure via `fsspec`
- **Byte-stream I/O** — no temp files for cloud reads/writes
- **Environment-aware config** — `${VAR_NAME:default}` interpolation with overlay support

### Metadata De-identification (DICOM)
- **Configurable PII mapping** with REMOVE / DUMMY / ZERO actions
- **Deterministic tokenization** for patient identifiers (enables re-identification if needed)
- **Private tag removal** for enhanced privacy

### Pipeline Architecture
- **MONAI-compliant transforms** — array + dictionary variants, `MetaTensor` propagation
- **Four pipeline modes**: DICOM single (`build_pipeline`), DICOM series (`build_series_pipeline`), Image single (`build_image_pipeline`), and Image series (`build_image_series_pipeline`)
- **Shared pixel redaction** — `RedactPixelPHId` is format-agnostic, used across all pipelines
- **Destructive redaction** — `RedactPixelPHId` permanently zeros PHI pixels and exposes redaction masks/stats as side outputs
- **Safety lock** — warns when prior spatial transforms may compromise OCR accuracy
- **Thread-safe by design** — only save transforms are `ThreadUnsafe`
- **Non-destructive** — input files remain unchanged

---

## 🔧 Architecture

### Architecture Diagram

```mermaid
flowchart TD
    classDef input fill:#e1fcff,stroke:#01579b,stroke-width:2px,color:#000;
    classDef process fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef subProcess fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5,color:#000;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000;
    classDef config fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
    classDef ioZone fill:#eceff1,stroke:#607d8b,stroke-width:2px,stroke-dasharray: 5 5,color:#000;

    subgraph Inputs ["Inputs"]
        dicom["DICOM Files (.dcm)"]:::input
        img["Image Files (.jpg, .png)"]:::input
    end

    config["config.yaml\n(OCR, NER, PII Mapping)"]:::config

    subgraph Pipeline ["Aegis Transform Pipeline"]
        
        subgraph Ingestion ["INGESTION ZONE (Single Disk Read)"]
            load["1. Load Transforms\n(LoadDicomRawd / LoadDicomSeriesd / LoadImaged / LoadImageSeriesd)\n(Raw Files → MetaTensor(s))\n<i>Thread-safe</i>"]:::process
        end

        subgraph Logic ["LOGIC ZONE (Purely In-Memory)"]
            subgraph step2 ["2. RedactPixelPHId <i>(Thread-safe via threading.local)</i>"]
                ocr["EasyOCR Engine\n(Text Detection)"]:::subProcess
                
                subgraph NER ["PHIClassifier (3-Layer Pipeline)"]
                    direction TB
                    L1{"1. Clinical\nAllowlist / Patterns?"}:::decision
                    L2{"2. Heuristic\nPHI Patterns?"}:::decision
                    L3["3. Stanford NER\nModel"]:::subProcess
                end
                
                action_redact["Apply Black-Box\nRedaction Mask"]:::process
            end

            scrub["3. ScrubDicomMetadatad\n(Processes cached pydicom.Dataset)\n<i>Thread-safe</i>"]:::process
            id_manager["AegisIdentityManager\n(Deterministic Tokenization / Hashing)"]:::subProcess
        end
        
        subgraph Persistence ["PERSISTENCE ZONE (Single Disk Write)"]
            save["4. Save Transforms\n(SaveDicomd / SaveDicomSeriesd / SaveImaged / SaveImageSeriesd)\n(Write to Disk tokens)\n<i>ThreadUnsafe</i>"]:::process
        end
    end

    subgraph Outputs ["Outputs"]
        out_not_proc["staging_not_processed/\n(Low-confidence / Manual Review)"]:::output
        out_proc["staging_output/\n(De-identified Files)"]:::output
    end

    %% Connections
    config -. "Configures" .-> Pipeline
    
    dicom -- "staging_input/" --> load
    img -- "staging_input/" --> load
    
    load --> ocr
    
    ocr -- "Confidence < Threshold" --> out_not_proc
    ocr -- "Confidence ≥ Threshold" --> L1
    
    L1 -- "Yes (Preserve)" --> scrub
    L1 -- "No" --> L2
    
    L2 -- "Yes" --> action_redact
    L2 -- "No" --> L3
    
    L3 -- "Classified as PHI" --> action_redact
    L3 -- "Classified as Clinical" --> scrub
    
    action_redact --> scrub
    
    scrub <--> |"Tokenize/Hash Patient PII"| id_manager
    scrub --> save
    save --> out_proc
```

An architectural priority is **strict I/O separation**, allowing the pipeline to scale efficiently across workers and cloud storage. All I/O flows through `AegisFileSystem` (fsspec) for cloud-native reads/writes. The pipeline is split into three explicit zones:

### 1. Ingestion Zone (Single Read)
* **DICOM Load (`LoadDicomRawd`)**: Reads DICOM files into a MONAI `MetaTensor`, caches the `pydicom.Dataset` in memory, and enriches `MetaTensor.meta` with `modality`, `patient_id`, `study_date`.
* **Image Load (`LoadImaged`)**: Reads JPEG/PNG files into a channel-first `MetaTensor`. No DICOM metadata.
* Both use `AegisFileSystem` byte streams for cloud storage compatibility.

### 2. Logic Zone (Purely In-Memory)
* **Redact (`RedactPixelPHId`)**: A destructive `MapTransform` that uses EasyOCR to detect text, then classifies each detection through a **3-layer pipeline**:
   * **(1) Clinical allowlist**: preserves known device/parameter terms.
   * **(2) PHI heuristics**: catch obvious date-IDs and institution names.
   * **(3) Stanford NER**: the `stanford-deidentifier-base` model provides semantic classification for remaining text.
   
   A binary **redaction mask** (matched to the MetaTensor's `spatial_shape`) is generated from PHI detections and stored alongside the output image for downstream review. Images with low OCR confidence bypass this step and are routed to `staging_not_processed/` for manual review. A **safety lock** warns if prior spatial transforms may compromise OCR accuracy. When `ner.enabled: false`, the system falls back to regex safelist matching.
* **Scrub (`ScrubDicomMetadatad`)**: A pure in-memory metadata scrubber (primarily for DICOMs). It works entirely on the cached `pydicom.Dataset` from the Ingestion Zone without hitting the disk again. It calls `AegisIdentityManager` to perform deterministic tokenization, dummy replacement, or removal of designated DICOM tags.

### 3. Persistence Zone (Single Write)
* **DICOM Save (`SaveDicomd` / `SaveDicomSeriesd`)**: Writes de-identified DICOM datasets. `ThreadUnsafe`.
* **Image Save (`SaveImaged`)**: Converts channel-first tensors back to HWC, normalizes to uint8, and writes PNG/JPEG. `ThreadUnsafe`.

### Component Interaction & Thread Safety Model

```mermaid
classDiagram
    class RunPipeline {
        +run_pipeline(config_path)
        -check_confidence(stats)
        -route_file()
    }

    class PipelineBuilder {
        +build_pipeline(config, output_dir)
    }

    class Transforms {
        <<Module>>
    }

    class LoadDicomRawd {
        +__call__(data)
        <<MapTransform>>
    }

    class LoadDicomSeriesd {
        +__call__(data)
        <<MapTransform>>
    }

    class LoadImaged {
        +__call__(data)
        <<MapTransform>>
    }

    class LoadImageSeriesd {
        +__call__(data)
        <<MapTransform>>
    }

    class RedactPixelPHId {
        +__call__(data)
        -transform: RedactPixelPHI
        <<MapTransform>>
    }

    class RedactPixelPHI {
        +reader (thread_local)
        +ner_classifier (thread_local)
        +detect_text()
        +apply_redaction()
    }

    class PHIClassifier {
        +classify_texts(texts)
        -pipeline (thread_local)
        -phi_labels
        -clinical_allowlist
        -clinical_patterns
        -phi_heuristic_patterns
    }

    class ScrubDicomMetadatad {
        +__call__(data)
        -transform: ScrubDicomMetadata
        <<MapTransform>>
    }

    class SaveDicomd {
        +__call__(data)
        <<MapTransform, ThreadUnsafe>>
    }

    class SaveDicomSeriesd {
        +__call__(data)
        <<MapTransform, ThreadUnsafe>>
    }

    class SaveImaged {
        +__call__(data)
        <<MapTransform, ThreadUnsafe>>
    }

    class SaveImageSeriesd {
        +__call__(data)
        <<MapTransform, ThreadUnsafe>>
    }

    RunPipeline --> PipelineBuilder
    PipelineBuilder ..> Transforms
    RunPipeline --> RedactPixelPHI : reads stats

    LoadDicomRawd --|> MapTransform
    LoadDicomSeriesd --|> MapTransform
    LoadImaged --|> MapTransform
    LoadImageSeriesd --|> MapTransform
    RedactPixelPHId *-- RedactPixelPHI
    RedactPixelPHI *-- PHIClassifier
    RedactPixelPHId --|> MapTransform
    ScrubDicomMetadatad *-- ScrubDicomMetadata
    ScrubDicomMetadatad --|> MapTransform
    SaveDicomd --|> MapTransform
    SaveDicomd --|> ThreadUnsafe
    SaveDicomSeriesd --|> MapTransform
    SaveDicomSeriesd --|> ThreadUnsafe
    SaveImaged --|> MapTransform
    SaveImaged --|> ThreadUnsafe
    SaveImageSeriesd --|> MapTransform
    SaveImageSeriesd --|> ThreadUnsafe
```

The pipeline is intentionally designed around multithreading bottlenecks and file I/O safety when operating within a PyTorch `DataLoader(num_workers > 0)`.

* **Concurrent Processing**: Steps 1–3 (`Load`, `Redact`, `Scrub`) are entirely thread-safe and operate purely in-memory. By pushing the heavy OCR computation to parallel background workers, the pipeline scales across CPU cores.
* **Thread-Local Isolation**: To prevent locking issues and race conditions with stateful AI models, `RedactPixelPHId` utilizes Python's `threading.local()`. This guarantees every worker thread spins up its own isolated EasyOCR reader and NER classifier instances.
* **I/O Isolation for Safety**: The pipeline intentionally marks **Step 4 (`SaveDicomd`)** as `ThreadUnsafe`. MONAI detects this and routes all dataset saving sequential actions to the main thread. This prevents file locking contentions, HDD thrashing, and ensures files uniquely derived from their source are written securely without race conditions.

| Transform | MONAI Bases | Strategy | Invertible | `num_workers > 0` |
|-----------|-------------|----------|------------|-------------------|
| `LoadDicomRawd` | `MapTransform` | Stateless, enriched `MetaTensor.meta` | — | ✅ Safe |
| `LoadDicomSeriesd` | `MapTransform` | Stateless volume loading | — | ✅ Safe |
| `LoadImaged` | `MapTransform` | Stateless image loading | — | ✅ Safe |
| `LoadImageSeriesd` | `MapTransform` | Stateless image series loading | — | ✅ Safe |
| `RedactPixelPHId` | `MapTransform` | `threading.local()` for EasyOCR + NER; keyframe OCR for volumes; emits redaction mask/stats side outputs | — | ✅ Safe |
| `PHIClassifier` | — | `threading.local()` for NER pipeline | — | ✅ Safe |
| `ScrubDicomMetadatad` | `MapTransform` | Pure in-memory; geometry-preserving series scrub | — | ✅ Safe |
| `SaveDicomd` | `MapTransform`, `ThreadUnsafe` | File I/O | — | ⚠️ Main thread sequential write |
| `SaveDicomSeriesd` | `MapTransform`, `ThreadUnsafe` | Series file I/O, path-preserving | — | ⚠️ Main thread sequential write |
| `SaveImaged` | `MapTransform`, `ThreadUnsafe` | Image file I/O | — | ⚠️ Main thread sequential write |
| `SaveImageSeriesd` | `MapTransform`, `ThreadUnsafe` | Image series file I/O | — | ⚠️ Main thread sequential write |

---

## 📦 Installation

### Prerequisites
- Python 3.8+
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
#   Processed files → staging_output/dicom/YYYY-MM-DD/ or staging_output/image/YYYY-MM-DD/
#   Low-confidence files → staging_not_processed/ (manual review)
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

storage:
  protocol: '${AEGIS_STORAGE_PROTOCOL:file}'   # file, s3, gs, az
  options: {}                                   # fsspec options (credentials, etc.)

tokenization:
  salt: '${AEGIS_TOKEN_SALT:default_dev_salt_string}'

series:
  enabled: true
  accepted_modalities: ['CT', 'MR', 'US', 'CR', 'DX', 'MG', 'PT', 'XA', 'RF', 'OT']
  keyframe_count: 3
  output_structure: 'hierarchical'

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
python -m monai_aegis.dicom_runner
```

| Backend | Protocol | Extra Package |
|---------|----------|---------------|
| Local   | `file`   | (built-in)    |
| AWS S3  | `s3`     | `s3fs`        |
| GCS     | `gs`     | `gcsfs`       |
| Azure   | `az`     | `adlfs`       |

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

## 🧪 Testing

```bash
# Run all 97 unit tests
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
│   ├── config/
│   │   ├── config.yaml               # De-identification + NER + storage + series settings
│   │   ├── config_loader.py          # Env var interpolation + overlay merging
│   │   └── storage.py                # AegisFileSystem (fsspec wrapper)
│   └── transforms/
│       ├── __init__.py                # Public API exports
│       ├── io.py                      # LoadDicomRaw/d, SaveDicom/d, LoadImage/d, SaveImage/d
│       ├── series_io.py               # LoadDicomSeries/d, SaveDicomSeries/d
│       ├── discovery.py               # discover_dicoms, group/validate/sort
│       ├── pixel.py                   # RedactPixelPHI/d (OCR + NER + volume keyframe)
│       ├── ner_classifier.py          # PHIClassifier (Stanford NER wrapper)
│       ├── metadata.py                # ScrubDicomMetadata/d (single + series scrub)
│       ├── exceptions.py              # Custom exception hierarchy
│       ├── utility.py                 # AegisIdentityManager (tokenization)
│       └── pipeline.py                # build_pipeline / build_series_pipeline / build_image_pipeline
├── tests/
│   ├── unit/                          # 97 unit tests (13 test files)
│   │   ├── test_config_loader.py      # Config loading, env vars, overlays
│   │   ├── test_storage.py            # AegisFileSystem, fsspec, byte-stream round-trip
│   │   ├── test_raw_loader.py         # DICOM + image loading transforms
│   │   ├── test_discovery.py          # Discovery, grouping, validation, sorting
│   │   ├── test_series_io.py          # Series load/save transforms
│   │   ├── test_volume_redaction.py   # Volume keyframe OCR
│   │   └── ...                        # Additional test files
│   └── integration/                   # End-to-end tests
├── run_dicom_pipeline.py              # Compatibility wrapper for monai_aegis.dicom_runner
├── run_image_pipeline.py              # Compatibility wrapper for monai_aegis.image_runner
├── run_pipeline.py                    # Compatibility wrapper for monai_aegis.cli
├── de-identification.ipynb            # Interactive walkthrough notebook
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

Apache 2.0 (pending)

---

## 🙏 Acknowledgments

- [MONAI](https://docs.monai.io/) — Medical Open Network for AI
- [StanfordAIMI](https://huggingface.co/StanfordAIMI/stanford-deidentifier-base) — Stanford De-identifier NER model
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — OCR detection library
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — NER pipeline
- [PyDICOM](https://pydicom.github.io/) — DICOM file handling
- [HIPAA De-identification Guidelines](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
