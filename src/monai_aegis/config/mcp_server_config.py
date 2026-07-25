import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from monai_aegis.api import IMAGE_EXTENSIONS, default_config_path  # noqa: F401
from monai_aegis.envelope import PIXEL_DECISIONS  # noqa: F401

# The packaged base config, resolved from the installed package location.
# This works identically for an editable ``src/`` checkout and a
# ``site-packages`` install because it is anchored on the package module, not
# on a computed repo root — so it never depends on where the process runs.
CONFIG_PATH = Path(default_config_path())

# Runtime output/review directories resolve against the current working
# directory, never the package location. A package-relative default would put
# writes inside ``src/`` (polluting the source tree) or inside site-packages
# (usually read-only), which is exactly the failure mode to avoid. The
# defaults and env overrides mirror config.yaml's ``paths`` section, so the
# MCP server and the CLI agree on where files land.
_OUTPUT_BASE = Path(os.environ.get("AEGIS_OUTPUT_DIR") or "staging_output").expanduser()
DEFAULT_OUTPUT_DIR = (_OUTPUT_BASE / "mcp").resolve()
REVIEW_DIR = Path(os.environ.get("AEGIS_REVIEW_DIR") or "staging_not_processed").expanduser().resolve()

# Batch discovery modes — aligned with the unified CLI (monai_aegis.cli):
# 'auto' covers both DICOM and standard images, and is the default there too.
BATCH_MODES = ("auto", "dicom", "image")
REVIEW_QUEUE_LIMIT = 50
HEADER_AUDIT_LIMIT = 30
JOB_ERRORS_LIMIT = 10

mcp = FastMCP("aegis-mcp")
