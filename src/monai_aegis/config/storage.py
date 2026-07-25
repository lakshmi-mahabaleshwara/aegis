"""
MONAI Aegis Storage Layer — fsspec-based file system abstraction.

Provides :class:`AegisFileSystem`, a thin wrapper around `fsspec` that
gives the pipeline a unified API for local, S3, GCS, and Azure storage.

All I/O transforms and discovery functions accept an optional ``fs``
parameter.  When ``None`` (default), local filesystem is used — existing
behaviour is completely unchanged.

Usage::

    # Local (default)
    fs = AegisFileSystem()

    # S3
    fs = AegisFileSystem(protocol='s3', key='...', secret='...')

    # From config.yaml
    fs = AegisFileSystem.from_config(config)

    # Read / write
    with fs.open_read('s3://bucket/scan.dcm') as f:
        ds = pydicom.dcmread(f)

    with fs.open_write('s3://bucket/out.dcm') as f:
        ds.save_as(f)
"""
from __future__ import annotations

import io
import logging
import os
import posixpath
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fsspec

__all__ = ["AegisFileSystem"]

logger = logging.getLogger(__name__)

# Protocols that resolve to the local filesystem — everything else is remote.
_LOCAL_PROTOCOLS = frozenset({"file", "local"})

# Remembers which remote protocols have already been warned about, so the
# experimental notice is emitted once per process rather than on every
# pipeline build or dataloader worker.
_experimental_warned: set = set()

_EXPERIMENTAL_STORAGE_NOTICE = (
    "Storage protocol '%s' selected — remote/cloud storage is EXPERIMENTAL. "
    "DICOM/image reads and writes use this backend, but ground-truth reports, "
    "review-queue quarantine, the verification pass, and the public API's "
    "input/output path handling still operate on local paths. A run may "
    "therefore split its outputs between remote and local storage; validate "
    "results before relying on this beyond experimentation."
)


class AegisFileSystem:
    """Unified file system abstraction using fsspec.

    Supports URIs with any protocol registered in fsspec:
    ``file://``, ``s3://``, ``gs://``, ``az://``, ``memory://``, or
    plain paths (treated as local filesystem).

    Args:
        protocol: fsspec protocol string (``'file'``, ``'s3'``, ``'gs'``,
            ``'az'``, ``'memory'``).  Defaults to ``'file'`` (local).
        **storage_options: Additional keyword arguments passed to
            ``fsspec.filesystem()`` (e.g. ``key``, ``secret``,
            ``endpoint_url`` for S3).
    """

    def __init__(self, protocol: str = 'file', **storage_options: Any):
        self.protocol = protocol
        self.storage_options = storage_options
        self.fs = fsspec.filesystem(protocol, **storage_options)
        logger.info("AegisFileSystem initialized: protocol=%s", protocol)
        if protocol not in _LOCAL_PROTOCOLS and protocol not in _experimental_warned:
            _experimental_warned.add(protocol)
            logger.warning(_EXPERIMENTAL_STORAGE_NOTICE, protocol)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AegisFileSystem':
        """Create an AegisFileSystem from the ``storage`` section of config.

        Expected config structure::

            storage:
              protocol: 's3'
              options:
                key: '...'
                secret: '...'

        If ``storage`` is absent, returns a local filesystem instance.

        Args:
            config: Full configuration dictionary.

        Returns:
            Configured :class:`AegisFileSystem`.
        """
        storage_cfg = config.get('storage', {})
        protocol = storage_cfg.get('protocol', 'file')
        options = storage_cfg.get('options', {})
        return cls(protocol=protocol, **options)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def open_read(self, path: str) -> io.IOBase:
        """Open *path* for binary reading.

        Args:
            path: File path or URI.

        Returns:
            A file-like object opened in ``'rb'`` mode.
        """
        return self.fs.open(path, 'rb')

    def open_write(self, path: str) -> io.IOBase:
        """Open *path* for binary writing, creating parent directories.

        Args:
            path: File path or URI.

        Returns:
            A file-like object opened in ``'wb'`` mode.
        """
        parent = self._parent(path)
        if parent:
            self.makedirs(parent)
        return self.fs.open(path, 'wb')

    # ------------------------------------------------------------------
    # Directory operations
    # ------------------------------------------------------------------

    def walk(self, path: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        """Recursively walk a directory tree.

        Yields ``(dirpath, dirnames, filenames)`` tuples, matching the
        interface of :func:`os.walk`.

        Args:
            path: Root directory to walk.
        """
        for dirpath, dirnames, filenames in self.fs.walk(path):
            yield dirpath, dirnames, filenames

    def exists(self, path: str) -> bool:
        """Check whether *path* exists."""
        return self.fs.exists(path)

    def isdir(self, path: str) -> bool:
        """Check whether *path* is a directory."""
        return self.fs.isdir(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Create directory and parents."""
        self.fs.makedirs(path, exist_ok=exist_ok)

    def relpath(self, path: str, start: str) -> str:
        """Compute relative path from *start* to *path*.

        Uses ``posixpath`` for cloud URIs, ``os.path`` for local.
        """
        if self.protocol == 'file':
            return os.path.relpath(path, start)
        # Cloud URIs use POSIX paths
        # Strip protocol prefix for relative computation
        return posixpath.relpath(path, start)

    def join(self, *parts: str) -> str:
        """Join path components.

        Uses ``os.path.join`` for local, ``posixpath.join`` for cloud.
        """
        if self.protocol == 'file':
            return os.path.join(*parts)
        return posixpath.join(*parts)

    def basename(self, path: str) -> str:
        """Return the final component of a path."""
        if self.protocol == 'file':
            return os.path.basename(path)
        return posixpath.basename(path)

    def dirname(self, path: str) -> str:
        """Return the directory component of a path."""
        if self.protocol == 'file':
            return os.path.dirname(path)
        return posixpath.dirname(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parent(self, path: str) -> str:
        """Get parent directory of *path*."""
        return self.dirname(path)

    def __repr__(self) -> str:
        return f"AegisFileSystem(protocol='{self.protocol}')"
