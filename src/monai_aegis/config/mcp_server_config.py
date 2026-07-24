import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

AEGIS_ROOT = Path(
    os.environ.get("AEGIS_ROOT") or Path(__file__).resolve().parent.parent.parent
).expanduser().resolve()
CONFIG_PATH = AEGIS_ROOT / "monai_aegis" / "config" / "config.yaml"
REVIEW_DIR = AEGIS_ROOT / "staging_not_processed"
DEFAULT_OUTPUT_DIR = AEGIS_ROOT / "staging_output" / "mcp"

# Canonical values live with the shared skill surface; re-exported here so
# the server keeps a single import site for its constants.
from monai_aegis.api import IMAGE_EXTENSIONS  # noqa: F401
from monai_aegis.envelope import PIXEL_DECISIONS  # noqa: F401
# Batch discovery modes — aligned with the unified CLI (monai_aegis.cli):
# 'auto' covers both DICOM and standard images, and is the default there too.
BATCH_MODES = ("auto", "dicom", "image")
REVIEW_QUEUE_LIMIT = 50
HEADER_AUDIT_LIMIT = 30
JOB_ERRORS_LIMIT = 10

mcp = FastMCP("aegis-mcp")
