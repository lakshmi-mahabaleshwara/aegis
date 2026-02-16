# MONAI Aegis

**Medical Image De-identification Pipeline with OCR and Metadata Scrubbing**

MONAI Aegis is a production-ready pipeline for de-identifying medical images (DICOM and JPEG) by removing Protected Health Information (PHI) while preserving critical clinical markers and image quality.

---

## 🌟 Features

### Visual De-identification (Pixel Scrubbing)
- **EasyOCR-based text detection** with configurable confidence thresholds
- **Safelist support** for preserving clinical markers (ultrasound parameters, measurements, etc.)
- **Grayscale and RGB image support** with proper orientation handling
- **Redaction via black boxes** on detected PHI text regions

### Metadata De-identification (DICOM)
- **Configurable PII mapping** with REMOVE/DUMMY/ZERO actions
- **Deterministic tokenization** for patient identifiers (enables re-identification if needed)
- **Private tag removal** for enhanced privacy
- **Pixel data injection** maintaining original bit depth and quality

### Pipeline Architecture
- **MONAI transforms-based** for composability and extensibility
- **MONAI LoadImaged** with PydicomReader for orientation-safe DICOM loading
- **MetaTensor propagation** throughout the entire pipeline
- **Separate processing** for DICOM vs JPEG/PNG formats
- **Non-destructive** - input files remain unchanged

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Install Package
```bash
# Clone repository
git clone <repo-url>
cd aegis/monai_aegis

# Create virtual environment (optional but recommended)
python -m venv ../venv
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate

# Install in editable mode
pip install -e .
```

This will install all dependencies:
- MONAI >= 1.0.0
- PyTorch >= 1.13.0
- pydicom >= 2.3.0
- easyocr >= 1.6.0
- opencv-python >= 4.7.0
- PyYAML, Pillow, NumPy

---

## 🚀 Quick Start

### Basic Usage

```python
from scrubbers.basescrubber import build_pipeline

# Build the de-identification pipeline
pipeline = build_pipeline(
    config_path='config/config.yaml',
    output_dir='./output'
)

# Process a DICOM file
data = {'image': '/path/to/file.dcm'}
result = pipeline(data)
# De-identified DICOM saved to ./output/file.dcm

# Process a JPEG file
data = {'image': '/path/to/image.jpg'}
result = pipeline(data)
# De-identified JPEG saved to ./output/image.jpg
```

### Command-Line Pipeline Runner

```bash
# Process files in staging_input/ directory
python run_pipeline.py

# Outputs saved to staging_output/
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize behavior:

```yaml
ocr:
  languages: ['en']
  gpu_usage: false
  confidence_threshold: 0.4

safelist:
  # Preserve ultrasound clinical markers
  - '^B$'           # B-mode indicator
  - '^FH?\s*\d+\.?\d*$'  # Frequency: FH5.0, F H5.0
  - '^D\s*\d+\.?\d*$'    # Depth: D 13.0
  - '^G\s*\d+$'          # Gain: G 25, G25
  - '^6\s*\d+$'          # OCR error: "6 25" (G→6)
  # ... more patterns

pii_mapping:
  '(0010, 0010)': 'DUMMY'   # PatientName → TOKEN_xxxxx
  '(0010, 0020)': 'DUMMY'   # PatientID → TOKEN_xxxxx
  '(0008, 0020)': 'ZERO'    # StudyDate → 00000000
  '(0008, 0050)': 'REMOVE'  # AccessionNumber → removed
```

### Safelist Patterns
- **Regex-based**: Use Python regex to match clinical text
- **OCR error handling**: Account for common OCR misreads (e.g., G→6)
- **Examples**: See full list in `config/config.yaml`

### PII Mapping Actions
- **DUMMY**: Replace with deterministic token (TOKEN_hash)
- **REMOVE**: Delete the DICOM tag entirely
- **ZERO**: Set to zero/empty value

---

## 🧪 Testing

### Run All Tests
```bash
# Unit tests (17 tests)
python -m unittest discover tests/unit -v

# Integration tests (3 tests)
python -m unittest discover tests/integration -v

# All tests (20 tests)
python -m unittest discover tests -v
```

### Test Coverage
- ✅ **AegisPixelScrubber** - Visual redaction logic
- ✅ **AegisMetadataScrubber** - DICOM metadata scrubbing
- ✅ **pixel_scrubber** - OCR detection & redaction
- ✅ **metadata_scrubber** - Tag manipulation
- ✅ **Full pipeline** - End-to-end integration

---

## 📁 Project Structure

```
monai_aegis/
├── setup.py                    # Package configuration
├── config/
│   └── config.yaml             # De-identification configuration
├── scrubbers/
│   ├── basescrubber.py         # Pipeline builder (LoadImaged + EnsureChannelFirstd)
│   ├── AegisPixelScrubber.py   # Visual redaction transform
│   ├── AegisMetadataScrubber.py # Metadata scrubbing transform
│   ├── pixel_scrubber.py       # OCR detection/redaction functions
│   ├── metadata_scrubber.py    # DICOM tag manipulation
│   └── identity_manager.py      # Tokenization utility
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
└── run_pipeline.py             # CLI runner script
```

---

## 🔧 Architecture

### MONAI Pipeline Flow

```
Input File (.dcm / .jpg / .png)
    ↓
LoadImaged              # MONAI loader (PydicomReader, no reorientation)
    ↓
EnsureChannelFirstd     # Normalize to (C, H, W) → MetaTensor
    ↓
AegisPixelScrubber     # EasyOCR + Visual redaction (preserves MetaTensor)
    ↓
AegisMetadataScrubber  # DICOM metadata scrubbing (DICOM only)
    ↓
Output File (saved by MetadataScrubber or run_pipeline.py)
```

### Key Design Decisions

1. **MONAI LoadImaged**: Uses `PydicomReader(affine_lps_to_ras=False)` to preserve original DICOM orientation
2. **MetaTensor Throughout**: All transforms preserve MONAI MetaTensor with metadata attached
3. **Separate Metadata Scrubber**: Only processes DICOM files, skips JPEG/PNG
4. **Safelist First**: Clinical markers checked against safelist before redaction
5. **Deterministic Tokens**: Same input always generates same token (enables re-identification)
6. **Absolute Imports**: Modern package structure via `setup.py` (no path hacks)

---

## 🐛 Known Issues & Limitations

- **OCR Accuracy**: EasyOCR may misread characters (e.g., G→6, Z→2). Use safelist to compensate.
- **GPU Support**: EasyOCR runs on CPU by default. Enable GPU in config for better performance.
- **Deprecated PyDICOM warnings**: Expected for legacy DICOM attributes (non-critical)

---

## 🤝 Contributing

### Code Style
```bash
# Format code
pip install black
black scrubbers/ tests/

# Lint code
pip install flake8
flake8 scrubbers/ tests/
```

### Adding Tests
- Unit tests: `tests/unit/test_<module>.py`
- Integration tests: `tests/integration/test_<feature>.py`
- Run tests before committing: `python -m unittest discover tests -v`

---

## 📄 License

[Add your license here]

---

## 📧 Contact

[Add contact information]

---

## 🙏 Acknowledgments

- **MONAI**: Medical Open Network for AI
- **EasyOCR**: OCR detection library
- **PyDICOM**: DICOM file handling

---

## 📚 Additional Resources

- [MONAI Documentation](https://docs.monai.io/)
- [DICOM Standard](https://www.dicomstandard.org/)
- [HIPAA De-identification Guidelines](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html)
