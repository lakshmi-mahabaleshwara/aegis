"""
MONAI Aegis package root.
"""
import warnings

# Suppress expected upstream deprecation warnings (e.g. from numba / torch)
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*The cuda.cudart module is deprecated.*"
)

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("monai-aegis")
except Exception:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

__all__ = ["config", "transforms", "__version__"]
