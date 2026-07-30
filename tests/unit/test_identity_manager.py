"""Salt-resolution precedence for AegisIdentityManager.from_config().

Explicit config salt is authoritative over the ambient AEGIS_TOKEN_SALT so a
library/API caller's configured pseudonym linkage can't be silently overridden
by the host environment. The env var still applies when config pins no salt.
"""
from monai_aegis.transforms.utility import AegisIdentityManager


def test_explicit_config_salt_wins_over_env(monkeypatch):
    monkeypatch.setenv("AEGIS_TOKEN_SALT", "env-salt")
    mgr = AegisIdentityManager.from_config({"tokenization": {"salt": "config-salt"}})
    assert mgr.salt == "config-salt"


def test_env_salt_used_when_config_has_no_salt(monkeypatch):
    monkeypatch.setenv("AEGIS_TOKEN_SALT", "env-salt")
    assert AegisIdentityManager.from_config({}).salt == "env-salt"
    # An empty/blank config salt is not "explicit" — env still applies.
    assert AegisIdentityManager.from_config(
        {"tokenization": {"salt": ""}}
    ).salt == "env-salt"


def test_default_when_neither_config_nor_env(monkeypatch):
    monkeypatch.delenv("AEGIS_TOKEN_SALT", raising=False)
    mgr = AegisIdentityManager.from_config({})
    assert mgr.salt == AegisIdentityManager._DEFAULT_SALT


def test_config_salt_wins_even_without_env(monkeypatch):
    monkeypatch.delenv("AEGIS_TOKEN_SALT", raising=False)
    mgr = AegisIdentityManager.from_config({"tokenization": {"salt": "config-salt"}})
    assert mgr.salt == "config-salt"
