# Docker Build and Test Guide for MONAI Aegis

## Quick Start

### 1. Build the Docker Image
```bash
docker build -t monai_aegis:latest .
```

### 2. Run the Pipeline
```bash
# Process files in staging_input/ directory
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  monai_aegis:latest
```

## Detailed Usage

### Build Options

**Standard build (CPU only):**
```bash
docker build -t monai_aegis:latest .
```

**Build with specific tag:**
```bash
docker build -t monai_aegis:v1.0 .
```

**Build with no cache (clean build):**
```bash
docker build --no-cache -t monai_aegis:latest .
```

### Run Modes

#### 1. Process Files from Host Directory
```bash
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  monai_aegis:latest
```

#### 2. Run with Custom Config
```bash
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  -v "$(pwd)/custom_config.yaml:/app/monai_aegis/config/config.yaml:ro" \
  monai_aegis:latest
```

#### 3. Run Unit Tests
```bash
docker run --rm \
  monai_aegis:latest \
  python -m unittest discover /app/tests/unit -v
```

#### 4. Run Integration Tests
```bash
docker run --rm \
  monai_aegis:latest \
  python -m unittest discover /app/tests/integration -v
```

#### 5. Interactive Shell (for debugging)
```bash
docker run --rm -it \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  monai_aegis:latest \
  /bin/bash
```

### Volume Mappings Explained

- **`-v "$(pwd)/staging_input:/app/staging_input:ro"`**
  - Maps your local `staging_input/` to container's `/app/staging_input`
  - `:ro` = read-only (container can't modify original files)

- **`-v "$(pwd)/staging_output:/app/staging_output"`**
  - Maps your local `staging_output/` to container's `/app/staging_output`
  - Container writes de-identified files here

## Testing Workflow

### Step 1: Prepare Test Data
```bash
# Place test files in staging_input/
cp /path/to/test.dcm staging_input/
cp /path/to/test.jpg staging_input/
```

### Step 2: Build Image
```bash
docker build -t monai_aegis:test .
```

### Step 3: Run Pipeline
```bash
docker run --rm \
  -v "$(pwd)/staging_input:/app/staging_input:ro" \
  -v "$(pwd)/staging_output:/app/staging_output" \
  monai_aegis:test
```

### Step 4: Verify Output
```bash
# Check output files
ls -lh staging_output/

# View DICOM metadata
python -c "import pydicom; ds=pydicom.dcmread('staging_output/test.dcm'); print(ds)"
```

## Troubleshooting

### Issue: "Permission denied" errors
**Solution:** Ensure output directory has write permissions
```bash
chmod -R 777 staging_output/
```

### Issue: EasyOCR model download fails
**Solution:** Pre-download models or use volume mount
```bash
# Pre-download models
docker run --rm -v ~/.EasyOCR:/root/.EasyOCR monai_aegis:latest python -c "import easyocr; easyocr.Reader(['en'])"
```

### Issue: Out of memory
**Solution:** Increase Docker memory limit
```bash
# In Docker Desktop: Settings → Resources → Memory
# Or use --memory flag:
docker run --rm --memory=4g -v ... monai_aegis:latest
```

## Performance Notes

- **First run**: EasyOCR downloads models (~1GB, takes time)
- **Subsequent runs**: Models cached, faster execution
- **CPU mode**: Slower OCR processing (expected without GPU)
- **GPU support**: Requires NVIDIA Docker runtime (not covered here)

## Clean Up

```bash
# Remove Docker image
docker rmi monai_aegis:latest

# Remove all stopped containers
docker container prune

# Remove dangling images
docker image prune
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Docker Test

on: [push]

jobs:
  docker-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t monai_aegis:test .
      - name: Run tests
        run: docker run --rm monai_aegis:test python -m unittest discover /app/tests -v
```
