"""
Aegis Utility — AegisIdentityManager

Deterministic identity tokenization for de-identification.
Generates reproducible SHA-256 tokens for PII values or raw bytes, enabling
re-identification when the salt is preserved.
"""
import hashlib
import logging
import os
from typing import Optional, Union

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
        self._bytes_cache: dict[bytes, str] = {}
        logger.info("AegisIdentityManager initialized (salt configured: %s)",
                     self.salt != self._DEFAULT_SALT)

    @classmethod
    def from_config(cls, config: dict) -> "AegisIdentityManager":
        """Create an AegisIdentityManager from the pipeline config.

        Resolution order:
            1. ``config['tokenization']['salt']`` — an explicit, non-empty salt
            2. ``AEGIS_TOKEN_SALT`` environment variable
            3. Built-in default

        Explicit config wins over the environment on purpose: a library or
        API caller who passes a salt in *config* must be able to rely on it,
        rather than have an ambient host ``AEGIS_TOKEN_SALT`` silently
        override the pseudonym linkage they configured. The env var remains
        the ergonomic deployment override — the shipped ``config.yaml`` sets
        ``salt: '${AEGIS_TOKEN_SALT:}'``, so an env-driven salt still flows in
        through interpolation when config doesn't pin an explicit value.

        Args:
            config: The loaded pipeline configuration dictionary.

        Returns:
            Configured AegisIdentityManager instance.
        """
        # Explicit config salt wins — a caller-supplied value is authoritative.
        token_config = config.get("tokenization", {})
        config_salt = token_config.get("salt")
        if config_salt:
            return cls(salt=config_salt)

        # Otherwise the environment provides the salt (deployment override).
        env_salt = os.environ.get("AEGIS_TOKEN_SALT")
        if env_salt:
            return cls(salt=env_salt)

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

    def get_token_from_bytes(self, data: bytes) -> str:
        """Generate a consistent token from raw file bytes.

        Used for the 'Loose Flow' to fingerprint non-DICOM standard images.

        Args:
            data: Raw binary bytes of the file.

        Returns:
            A shortened hash token string prefixed with ``TOKEN_``.
        """
        if not data:
            return ""

        if data in self._bytes_cache:
            return self._bytes_cache[data]

        salt_bytes = self.salt.encode("utf-8")
        token = f"TOKEN_{hashlib.sha256(data + salt_bytes).hexdigest()[:16]}"
        self._bytes_cache[data] = token
        return token

    @property
    def cache_size(self) -> int:
        """Number of unique values currently cached."""
        return len(self._cache)

import posixpath


def resolve_target_token(
    uri: str,
    identity_manager: Optional[AegisIdentityManager],
    input_dir: str = "",
    fs: Optional[object] = None,
) -> Optional[str]:
    """Resolve a token from the top-level folder relative to ``input_dir``.

    Contract used across all loaders:
      - If ``input_dir`` is empty, do not tokenize.
      - If the file sits directly under ``input_dir``, do not tokenize.
      - Otherwise tokenize the first relative path segment.
    """
    if identity_manager is None or not input_dir:
        return None

    if fs is not None:
        rel_path = fs.relpath(uri, input_dir)
        sep = "/"
    else:
        rel_path = os.path.relpath(uri, input_dir)
        sep = os.sep

    parts = [part for part in rel_path.split(sep) if part not in ("", ".")]
    if len(parts) <= 1:
        return None
    return identity_manager.get_token(parts[0])

def build_output_path(
    uri: str,
    output_dir: str,
    input_dir: str = "",
    target_token: Optional[str] = None,
    is_cloud: bool = False,
    output_ext: Optional[str] = None,
) -> str:
    """Build output path preserving relative structure but overriding parent block with token."""
    if is_cloud:
        basename = posixpath.basename(uri)
    else:
        basename = os.path.basename(uri)

    if output_ext is not None:
        name, _ = os.path.splitext(basename)
        basename = f"{name}{output_ext}"

    if not input_dir:
        # No reference structure, just put it in target_token/basename if token exists
        if target_token:
            rel_path = f"{target_token}/{basename}" if is_cloud else os.path.join(target_token, basename)
        else:
            rel_path = basename
    else:
        # Prevent trailing slashes from breaking relpath
        input_dir = input_dir.rstrip('/\\')
        
        if is_cloud:
            # Strip protocol prefix if treating as posix block
            rel_path = posixpath.relpath(uri, input_dir)
            sep = '/'
        else:
            rel_path = os.path.relpath(uri, input_dir)
            sep = os.sep

        if output_ext is not None:
            dirname = posixpath.dirname(rel_path) if is_cloud else os.path.dirname(rel_path)
            if is_cloud:
                rel_path = f"{dirname}/{basename}" if dirname else basename
            else:
                rel_path = os.path.join(dirname, basename) if dirname else basename

        if target_token:
            parts = rel_path.split(sep)
            if len(parts) > 1:
                # Replace the top-level directory (parent folder) with the token
                parts[0] = target_token
                rel_path = sep.join(parts)

    if is_cloud:
        return f"{output_dir}/{rel_path}" if not output_dir.endswith('/') else f"{output_dir}{rel_path}"
    else:
        return os.path.join(output_dir, rel_path)
