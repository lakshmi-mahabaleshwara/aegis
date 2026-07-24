FROM python:3.10-slim

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Build from the repository root:  docker build -f docker/Dockerfile -t aegis .
# COPY paths below are relative to that root (the build context).
COPY pyproject.toml MANIFEST.in README.md /app/
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY scripts/ /app/scripts/

# Install the package in editable mode (src/ layout, pyproject at root)
RUN pip install --no-cache-dir -e /app/

# Set environment variable to avoid EasyOCR download errors
ENV EASYOCR_MODULE_PATH=/root/.EasyOCR

# Bake model weights into the image at their pinned revisions so the
# container never needs network access at runtime (air-gapped ready).
RUN python /app/scripts/prefetch_models.py --config /app/src/monai_aegis/config/config.yaml

# Run fully offline: NER resolves from the baked-in HF cache, EasyOCR
# never attempts a download.
ENV AEGIS_HF_OFFLINE=true \
    AEGIS_MODEL_DOWNLOADS=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Default command: run the pipeline (the --config default resolves to the
# packaged config.yaml, so no path argument is needed).
CMD ["aegis-pipeline"]
