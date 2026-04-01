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

__all__ = ["config", "transforms"]
