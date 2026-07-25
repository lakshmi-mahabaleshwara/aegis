"""Path-resolution contract for the MCP server config.

Regression guard for the ``src/`` layout: the server's config path must be
resolved from the installed package (works in an editable checkout and in
site-packages alike), and its runtime output/review directories must resolve
relative to the working directory — never inside the package tree, which is
often read-only when pip-installed.
"""
import importlib
from pathlib import Path

import monai_aegis
import monai_aegis.config.mcp_server_config as cfg
from monai_aegis.api import default_config_path

_PACKAGE_DIR = Path(monai_aegis.__file__).resolve().parent   # .../src/monai_aegis
_PACKAGE_PARENT = _PACKAGE_DIR.parent                        # .../src


def _reload_with_env(monkeypatch, **env):
    """Reload the config module with a patched environment, restoring after."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(cfg)


def test_config_path_uses_canonical_resolver():
    assert cfg.CONFIG_PATH == Path(default_config_path())
    assert cfg.CONFIG_PATH.is_file()
    # Anchored on the package module, so it lives inside the package.
    assert cfg.CONFIG_PATH.is_relative_to(_PACKAGE_DIR)


def test_runtime_dirs_are_not_inside_the_package():
    # The core regression: writes must never default into src/ or site-packages.
    for path in (cfg.DEFAULT_OUTPUT_DIR, cfg.REVIEW_DIR):
        assert not path.is_relative_to(_PACKAGE_DIR), path
        assert not path.is_relative_to(_PACKAGE_PARENT), path


def test_runtime_dirs_default_relative_to_cwd(monkeypatch):
    reloaded = _reload_with_env(
        monkeypatch, AEGIS_OUTPUT_DIR=None, AEGIS_REVIEW_DIR=None
    )
    try:
        cwd = Path.cwd().resolve()
        assert reloaded.DEFAULT_OUTPUT_DIR == cwd / "staging_output" / "mcp"
        assert reloaded.REVIEW_DIR == cwd / "staging_not_processed"
    finally:
        importlib.reload(cfg)


def test_env_overrides_are_honored(monkeypatch, tmp_path):
    out = tmp_path / "custom_out"
    review = tmp_path / "custom_review"
    reloaded = _reload_with_env(
        monkeypatch,
        AEGIS_OUTPUT_DIR=str(out),
        AEGIS_REVIEW_DIR=str(review),
    )
    try:
        assert reloaded.DEFAULT_OUTPUT_DIR == (out / "mcp").resolve()
        assert reloaded.REVIEW_DIR == review.resolve()
    finally:
        importlib.reload(cfg)


def test_no_aegis_root_symbol_remains():
    # AEGIS_ROOT was the footgun; it must be gone so nothing re-introduces
    # package-relative path math through it.
    assert not hasattr(cfg, "AEGIS_ROOT")
