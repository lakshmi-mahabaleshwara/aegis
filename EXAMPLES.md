# MONAI Aegis — Examples & How-To Guide

Step-by-step instructions for running the de-identification pipeline.

---

## 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd aegis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install the package (editable mode)
pip install -e monai_aegis/
```

### Verify Installation

```bash
# Quick sanity check — should print without errors
python -c "from monai_aegis.transforms.pipeline import build_pipeline; print('Import OK')"
```

---

## 2. Prepare Input Data

Place your medical image files in the `staging_input/` directory:

```bash
# Supported formats: .dcm, .jpg, .jpeg, .png
cp /path/to/your/files/*.dcm  staging_input/
cp /path/to/your/files/*.jpg  staging_input/
```

**Current staging_input/ contains:**
| File | Type |
|------|------|
| `05.dcm` — `10.dcm` | DICOM ultrasound images |
| `U0000001.dcm` | DICOM file |
| `1.png`, `2.png`, `3.png` | PNG images |
| `202601170918250004ABD.JPG` — `202601170929370009ABD.JPG` | JPEG ultrasound images |

---

## 3. Run the Pipeline

### Option A: Command-Line Runner (Recommended)

```bash
# From the project root directory (aegis/)
aegis-pipeline --config monai_aegis/config/config.yaml
```

**What happens:**
- All files in `staging_input/` are processed
- De-identified files → `staging_output/`
- Low-confidence files → `staging_not_processed/` (manual review required)
- Console shows per-file progress and final summary

**Expected output:**
```
2026-02-25 20:09:30 [INFO] __main__ — Building pipeline...
2026-02-25 20:09:35 [INFO] __main__ — Processing: 05.dcm
...
2026-02-25 20:10:04 [INFO] __main__ — Processing complete. Processed: 13 | Not processed: 3 | Errors: 0
2026-02-25 20:10:04 [WARNING] __main__ — 3 file(s) in staging_not_processed/ need manual review.
```

### Option B: Python API

```python
from monai_aegis.transforms.pipeline import build_pipeline

# Build the pipeline (loads config once)
pipeline = build_pipeline(
    config_path='monai_aegis/config/config.yaml',
    output_dir='staging_output'
)

# Process a single DICOM file
result = pipeline({'image': 'staging_input/U0000001.dcm'})

# Process a single JPEG file
result = pipeline({'image': 'staging_input/202601170918250004ABD.JPG'})

# Inspect the result dictionary
print(result.keys())
# dict_keys(['image', 'image_meta_dict', 'image_dicom_dataset', 'image_redaction_stats', 'image_scrubbed_ds'])
```

### Option C: Process a Single File (Quick Test)

```bash
# Process just one file to verify setup
python -c "
from monai_aegis.transforms.pipeline import build_pipeline
pipeline = build_pipeline(config_path='monai_aegis/config/config.yaml', output_dir='staging_output')
result = pipeline({'image': 'staging_input/202601170918250004ABD.JPG'})
print('Processed successfully')
print('Redaction stats:', result.get('image_redaction_stats', {}))
"
```

---

## 4. Check Output

```bash
# List processed files
ls -la staging_output/

# List files flagged for manual review
ls -la staging_not_processed/

# Compare input vs output (visual check)
open staging_input/202601170918250004ABD.JPG     # Original with PHI
open staging_output/202601170918250004ABD.JPG    # PHI redacted
```

---

## 5. Run Tests

### All Unit Tests (30 tests)

```bash
python -m unittest discover tests/unit -v
```

### Run Specific Test Modules

```bash
# NER classifier tests
python -m unittest tests.unit.test_ner_classifier -v

# Pixel scrubber tests (OCR + redaction)
python -m unittest tests.unit.test_pixel_scrubber -v

# Redact pixel PHI tests (InvertibleTransform)
python -m unittest tests.unit.test_redact_pixel_phi -v

# DICOM metadata scrubber tests
python -m unittest tests.unit.test_scrub_dicom_metadata -v

# I/O transform tests (load/save)
python -m unittest tests.unit.test_io_transforms -v
```

### Integration Tests

```bash
python -m unittest discover tests/integration -v
```

---

## 6. Configuration Tuning

Edit `monai_aegis/config/config.yaml` to customize behavior:

### Adjust OCR Confidence Threshold

```yaml
ocr:
  confidence_threshold: 0.4   # Lower = more text detected (may include noise)
                                # Higher = stricter filtering (may miss faint text)
```

### Enable / Disable NER

```yaml
ner:
  enabled: true    # true = Stanford NER model (recommended)
                   # false = falls back to regex safelist matching
```

### Change NER Device

```yaml
ner:
  device: 'cpu'    # Options: 'cpu', 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon)
```

### Add Clinical Terms to Allowlist

```yaml
ner:
  clinical_allowlist:
    - mindray
    - ge
    - your_custom_term    # ← Add terms that should NEVER be redacted
```

### Add PHI Heuristic Patterns

```yaml
ner:
  phi_heuristic_patterns:
    - '(?i)(your_hospital_name)'   # ← Catch specific institution names
    - '^\d{6,}$'                   # ← Catch long numeric IDs
```

### Configure PII Tag Actions

```yaml
pii_mapping:
  '(0010, 0010)': 'DUMMY'    # PatientName → hashed token
  '(0010, 0020)': 'DUMMY'    # PatientID → hashed token
  '(0008, 0020)': 'ZERO'     # StudyDate → 00000000
  '(0008, 0050)': 'REMOVE'   # AccessionNumber → deleted entirely
```

---

## 7. Docker Usage

```bash
# Build the Docker image
docker build -t monai_aegis:latest .

# Run the pipeline inside Docker
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  monai_aegis:latest

# Run tests inside Docker
docker run --rm monai_aegis:latest \
  python -m unittest discover /app/tests/unit -v
```

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'monai_aegis'` | Install the package with `pip install -e monai_aegis/` from the repo root |
| `No module named 'easyocr'` | Run `pip install -e monai_aegis/` to install dependencies |
| NER model download is slow | First run downloads ~400MB model; subsequent runs use cache |
| All files going to `staging_not_processed/` | Lower `ocr.confidence_threshold` in config.yaml (e.g., `0.3`) |
| CUDA out of memory | Set `ner.device: 'cpu'` in config.yaml |
| `staging_output/` is empty | Check that `staging_input/` contains supported file formats (.dcm, .jpg, .png) |
