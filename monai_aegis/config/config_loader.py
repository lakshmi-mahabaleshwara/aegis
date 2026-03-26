"""
MONAI Aegis Config Loader — Environment-aware YAML configuration.

Resolves ``${VAR_NAME}`` and ``${VAR_NAME:default}`` placeholders in all
string values after loading, enabling production deployments to inject
paths, credentials, and settings via environment variables.

Supports an optional **overlay** file for environment-specific overrides
(e.g. ``config.prod.yaml``).  The overlay is deep-merged on top of the
base config before env-var resolution.

Usage::

    from monai_aegis.config.config_loader import load_config

    # Minimal — uses defaults baked into config.yaml
    config = load_config('monai_aegis/config/config.yaml')

    # With overlay (staging / prod)
    config = load_config(
        'monai_aegis/config/config.yaml',
        overlay_path='config/config.prod.yaml',
    )

    # Or set AEGIS_CONFIG_OVERRIDE=/path/to/override.yaml in the environment

Environment variable syntax in YAML values::

    paths:
      input_dir: '${AEGIS_INPUT_DIR:staging_input}'   # uses default if unset
      output_dir: '${AEGIS_OUTPUT_DIR}'                # raises if unset
"""
from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any, Dict, Optional

import yaml

__all__ = ["load_config", "resolve_env_vars", "deep_merge", "get_keyframe_sampling"]

logger = logging.getLogger(__name__)

# Matches ${VAR} or ${VAR:default_value}
_ENV_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')


def get_keyframe_sampling(config: dict) -> dict:
    """Read keyframe_sampling block from config with safe defaults.

    Falls back gracefully if the legacy ``keyframe_count`` key is present,
    using it as the ``floor`` value so existing deployments are unaffected.

    Args:
        config: Fully resolved config dict (output of :func:`load_config`).

    Returns:
        Dict with keys ``sample_pct``, ``floor``, and ``ceiling``.
    """
    series_cfg = config.get("series", {})
    # Backwards compatibility: if old key exists, derive floor from it
    legacy_count = series_cfg.get("keyframe_count")
    sampling = series_cfg.get("keyframe_sampling", {})
    return {
        "sample_pct": sampling.get("sample_pct", 0.10),
        "floor":      sampling.get("floor", legacy_count or 3),
        "ceiling":    sampling.get("ceiling", 25),
    }


def resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${VAR_NAME:default}`` in all string values.

    Args:
        obj: Any Python object (str, dict, list, int, etc.).

    Returns:
        A copy with all ``${...}`` placeholders replaced by their
        environment-variable values.

    Raises:
        KeyError: If a ``${VAR}`` (no default) references an unset variable.
    """
    if isinstance(obj, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)          # None when no default given
            value = os.environ.get(var_name)
            if value is not None:
                return value
            if default is not None:
                return default
            raise KeyError(
                f"Environment variable '{var_name}' is not set and no "
                f"default was provided (use ${{VAR:default}} syntax)."
            )
        return _ENV_PATTERN.sub(_replacer, obj)

    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]

    # int, float, bool, None — return as-is
    return obj


def deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep-merge *overlay* into a copy of *base*.

    - Dict values are merged recursively.
    - All other types are replaced by the overlay value.

    Args:
        base: Base configuration dictionary.
        overlay: Override dictionary (takes precedence).

    Returns:
        A new merged dictionary (neither input is modified).
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str,
    overlay_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load YAML config with env-var interpolation and optional overlay.

    Resolution order:

    1. Load *config_path* (base config).
    2. If *overlay_path* is given **or** ``AEGIS_CONFIG_OVERRIDE`` is set,
       load the overlay YAML and deep-merge it on top of the base.
    3. Resolve all ``${VAR_NAME:default}`` placeholders from env vars.

    Args:
        config_path: Path to the base ``config.yaml``.
        overlay_path: Optional path to an override YAML file.
            If ``None``, falls back to the ``AEGIS_CONFIG_OVERRIDE``
            environment variable.

    Returns:
        Fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        KeyError: If a ``${VAR}`` without a default references an unset
            environment variable.
    """
    # --- 1. Load base config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Loaded base config from %s", config_path)

    # --- 2. Overlay (explicit path or env var) ---
    overlay = overlay_path or os.environ.get('AEGIS_CONFIG_OVERRIDE')
    if overlay and os.path.isfile(overlay):
        with open(overlay, 'r') as f:
            overlay_data = yaml.safe_load(f) or {}
        config = deep_merge(config, overlay_data)
        logger.info("Merged overlay config from %s", overlay)

    # --- 3. Resolve env vars ---
    config = resolve_env_vars(config)

    return config
