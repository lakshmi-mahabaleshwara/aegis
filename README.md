# MONAI Aegis

**Medical Image De-identification Pipeline with OCR and Metadata Scrubbing**

MONAI Aegis is a production-ready pipeline for de-identifying medical images (DICOM, JPEG, PNG) by removing Protected Health Information (PHI) while preserving critical clinical markers and image quality.

---

## 🌟 Features

### Visual De-identification (Pixel Redaction)
- **EasyOCR-based text detection** with configurable confidence thresholds
- **Thread-safe** OCR via `threading.local()` — safe for `DataLoader(num_workers > 0)`
- **Safelist support** for preserving clinical markers (ultrasound parameters, measurements)
- **Low-confidence routing** — images with uncertain OCR are flagged for manual review

### Metadata De-identification (DICOM)
- **Configurable PII mapping** with REMOVE / DUMMY / ZERO actions
- **Deterministic tokenization** for patient identifiers (enables re-identification if needed)
- **Private tag removal** for enhanced privacy
- **Pure in-memory scrubbing** — no file I/O during transformation

### Pipeline Architecture
- **MONAI-compliant transforms** — array + dictionary variants, `MetaTensor` propagation
- **Thread-safe by design** — only `SaveDicomd` (file I/O) is `ThreadUnsafe`
- **Four-step pipeline**: Load → Redact → Scrub → Save
- **Non-destructive** — input files remain unchanged

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

### Python API
```python
from transforms.pipeline import build_pipeline

# Build the de-identification pipeline
pipeline = build_pipeline(
    config_path='monai_aegis/config/config.yaml',
    output_dir='staging_output'
)

# Process a DICOM file
result = pipeline({'image': 'staging_input/scan.dcm'})

# Process a JPEG file
result = pipeline({'image': 'staging_input/photo.jpg'})
```

### Command-Line Runner
```bash
# Process all files in staging_input/
python run_pipeline.py --config monai_aegis/config/config.yaml

# Output:
#   Processed files → staging_output/
#   Low-confidence files → staging_not_processed/ (manual review)
```

---

## ⚙️ Configuration

Edit `monai_aegis/config/config.yaml`:

```yaml
ocr:
  languages: ['en']
  gpu_usage: false
  confidence_threshold: 0.4

safelist:
  - '^B$'                        # B-mode indicator
  - '^FH?\s*\d+\.?\d*$'         # Frequency: FH5.0
  - '^D\s*\d+\.?\d*$'           # Depth: D 13.0
  - '\d+\.?\d*\s*(cm|mm|MHz)$'  # Measurements: 2.35 cm

pii_mapping:
  '(0010, 0010)': 'DUMMY'   # PatientName → TOKEN_xxxxx
  '(0010, 0020)': 'DUMMY'   # PatientID → TOKEN_xxxxx
  '(0008, 0020)': 'ZERO'    # StudyDate → 00000000
  '(0008, 0050)': 'REMOVE'  # AccessionNumber → removed
```

### PII Mapping Actions
| Action | Behavior | Example |
|--------|----------|---------|
| `DUMMY` | Replace with deterministic hash token | `John Doe` → `TOKEN_a1b2c3d4e5f6` |
| `REMOVE` | Delete the DICOM tag entirely | Tag removed from dataset |
| `ZERO` | Set to zero / empty value | `20250216` → `00000000` |

---

## 🧪 Testing

```bash
# Run all 18 unit tests
PYTHONPATH=monai_aegis python -m unittest discover tests/unit -v

# Run integration tests
PYTHONPATH=monai_aegis python -m unittest discover tests/integration -v
```

### Test Coverage
| Transform | Tests |
|-----------|-------|
| `RedactPixelPHI` / `RedactPixelPHId` | Visual redaction, safelist, shape preservation |
| `ScrubDicomMetadata` / `ScrubDicomMetadatad` | Tag scrubbing, no-I/O assertion, orientation |
| `LoadDicomRawd` | DICOM and JPEG loading |
| `detect_text` / `apply_redaction` | OCR detection, confidence filtering, redaction |

---

## 📁 Project Structure

```
aegis/
├── monai_aegis/                      # Installable package
│   ├── pyproject.toml                # PEP 621 package config
│   ├── config/
│   │   └── config.yaml               # De-identification settings
│   └── transforms/
│       ├── __init__.py                # Public API exports
│       ├── io.py                      # LoadDicomRaw/d, SaveDicom/d
│       ├── pixel.py                   # RedactPixelPHI/d (thread-safe OCR)
│       ├── metadata.py                # ScrubDicomMetadata/d (pure in-memory)
│       ├── utility.py                 # AegisIdentityManager (tokenization)
│       └── pipeline.py                # build_pipeline() composer
├── tests/
│   ├── unit/                          # 18 unit tests
│   └── integration/                   # End-to-end tests
├── run_pipeline.py                    # CLI entry point
├── Dockerfile                         # Container build
├── test_docker.sh                     # Docker test script
├── staging_input/                     # Input files (not tracked)
├── staging_output/                    # Processed output (not tracked)
└── staging_not_processed/             # Low-confidence files for review (not tracked)
```

---

## 🔧 Architecture

### Pipeline Flow

```
Input File (.dcm / .jpg / .png)
    │
    ▼
LoadDicomRawd           ← Thread-safe: raw DICOM/image → MetaTensor
    │
    ▼
RedactPixelPHId         ← Thread-safe: EasyOCR + safelist → redacted pixels
    │
    ▼
ScrubDicomMetadatad     ← Thread-safe: in-memory metadata scrub (DICOM only)
    │
    ▼
SaveDicomd              ← ThreadUnsafe: write scrubbed DICOM to disk
    │
    ▼
Output File (de-identified)
```

### Thread Safety Model
| Transform | Strategy | `num_workers > 0` |
|-----------|----------|-------------------|
| `LoadDicomRawd` | Stateless | ✅ Safe |
| `RedactPixelPHId` | `threading.local()` for EasyOCR | ✅ Safe |
| `ScrubDicomMetadatad` | Pure in-memory, no I/O | ✅ Safe |
| `SaveDicomd` | `ThreadUnsafe` mixin | ⚠️ Single-worker |

---

## 🐳 Docker

```bash
# Build
docker build -t monai_aegis:latest .

# Run pipeline
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  monai_aegis:latest

# Run tests
docker run --rm monai_aegis:latest \
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
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — OCR detection library
- [PyDICOM](https://pydicom.github.io/) — DICOM file handling
- [HIPAA De-identification Guidelines](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
