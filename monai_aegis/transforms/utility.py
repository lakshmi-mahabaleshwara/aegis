"""
Aegis Utility — AegisIdentityManager

Deterministic identity tokenization for de-identification.
Generates reproducible SHA-256 tokens for PII values, enabling
re-identification when the salt is preserved.
"""
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class AegisIdentityManager:
    """Manages identity tokenization for de-identification.

    Generates deterministic, consistent tokens for PII values using
    SHA-256 hashing with a configurable salt. The same input always
    produces the same token, enabling re-identification if the salt
    is preserved.

    Salt resolution order:
        1. Explicit ``salt`` argument (highest priority)
        2. ``AEGIS_TOKEN_SALT`` environment variable
        3. ``tokenization.salt`` in the config dict
        4. Built-in default (lowest priority, **not recommended for production**)

    Example::

        # From environment variable (recommended for production)
        os.environ['AEGIS_TOKEN_SALT'] = 'my-secret-salt'
        mgr = AegisIdentityManager()

        # From config overlay
        mgr = AegisIdentityManager.from_config(config)

        # Explicit salt
        mgr = AegisIdentityManager(salt='my-secret-salt')
    """

    _DEFAULT_SALT = "aegis"

    def __init__(self, salt: Optional[str] = None) -> None:
        """Initialize the identity manager.

        Args:
            salt: Secret salt for deterministic hashing. If not provided,
                falls back to ``AEGIS_TOKEN_SALT`` env var, then to a
                built-in default. **Keep this value secret and consistent
                across runs** — re-identification requires the same salt.
        """
        if salt is not None:
            self.salt = salt
        else:
            self.salt = os.environ.get("AEGIS_TOKEN_SALT", self._DEFAULT_SALT)

        if self.salt == self._DEFAULT_SALT:
            logger.warning(
                "Using default token salt. Set AEGIS_TOKEN_SALT env var "
                "or tokenization.salt in config for production use."
            )

        self._cache: dict[str, str] = {}
        logger.info("AegisIdentityManager initialized (salt configured: %s)",
                     self.salt != self._DEFAULT_SALT)

    @classmethod
    def from_config(cls, config: dict) -> "AegisIdentityManager":
        """Create an AegisIdentityManager from the pipeline config.

        Resolution order:
            1. ``AEGIS_TOKEN_SALT`` environment variable
            2. ``config['tokenization']['salt']``
            3. Built-in default

        Args:
            config: The loaded pipeline configuration dictionary.

        Returns:
            Configured AegisIdentityManager instance.
        """
        # Env var takes priority over config
        env_salt = os.environ.get("AEGIS_TOKEN_SALT")
        if env_salt:
            return cls(salt=env_salt)

        # Check config
        token_config = config.get("tokenization", {})
        config_salt = token_config.get("salt")
        if config_salt:
            return cls(salt=config_salt)

        # Fall back to default
        return cls()

    def get_token(self, value: str) -> str:
        """Generate a consistent token for a given value using SHA-256.

        Results are cached — repeated calls with the same value return
        the same token without recomputing the hash.

        Args:
            value: The input string to tokenize (e.g., PatientID).

        Returns:
            A shortened hash token string prefixed with ``TOKEN_``.
        """
        if not value:
            return ""

        if value in self._cache:
            return self._cache[value]

        data = f"{value}{self.salt}".encode("utf-8")
        token = f"TOKEN_{hashlib.sha256(data).hexdigest()[:16]}"
        self._cache[value] = token
        return token

    @property
    def cache_size(self) -> int:
        """Number of unique values currently cached."""
        return len(self._cache)
