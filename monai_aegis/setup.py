"""
MONAI Aegis - Medical Image De-identification Pipeline
"""
from setuptools import setup

setup(
    name="monai_aegis",
    version="0.1.0",
    description="MONAI-based medical image de-identification pipeline with OCR and metadata scrubbing",
    author="Aegis Team",
    python_requires=">=3.8",
    packages=["monai_aegis", "monai_aegis.transforms", "monai_aegis.config"],
    package_dir={"monai_aegis": "."},
    include_package_data=True,
    install_requires=[
        "monai>=1.0.0",
        "torch>=1.13.0",
        "pydicom>=2.3.0",
        "easyocr>=1.6.0",
        "opencv-python>=4.7.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "pillow>=9.0.0",
        "transformers>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aegis-pipeline=monai_aegis.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
