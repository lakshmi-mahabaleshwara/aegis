FROM python:3.10-slim

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package and project files
COPY monai_aegis/ /app/monai_aegis/
COPY tests/ /app/tests/
COPY run_pipeline.py /app/

# Install the package in editable mode
RUN pip install --no-cache-dir -e /app/monai_aegis/

# Add monai_aegis to PYTHONPATH so transforms module is importable
ENV PYTHONPATH="/app/monai_aegis:${PYTHONPATH}"

# Set environment variable to avoid EasyOCR download errors
ENV EASYOCR_MODULE_PATH=/root/.EasyOCR

# Default command: run the pipeline (paths read from config.yaml)
CMD ["python", "run_pipeline.py", "--config", "monai_aegis/config/config.yaml"]
