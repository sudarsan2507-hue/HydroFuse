"""Loading and access helpers for ``config.yaml``.

Every tunable value in this project (coordinates, dates, thresholds, file
paths) lives in ``config.yaml`` at the project root.  Modules should never
hardcode those values; they should call :func:`load_config` and read what they
need from the returned :class:`Config`.

Typical use::

    from backend.config import load_config

    cfg = load_config()
    lat = cfg.get("area.center_lat")
    db_path = cfg.resolve_path(cfg.get("backend.database_url"))
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# The project root is the directory that contains this package's parent, i.e.
# .../Patent_Sand/backend/config.py -> .../Patent_Sand
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

# Sentinel used so that `get()` can tell "no default supplied" apart from
# "default is None".
_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


class Config:
    """A thin, read-only wrapper around the parsed ``config.yaml`` mapping.

    The wrapper exists for two reasons:

    1. Dotted-key lookup (``cfg.get("area.center_lat")``) is easier to read at
       call sites than repeated dictionary indexing.
    2. A missing key raises a clear :class:`ConfigError` naming the key,
       instead of a bare ``KeyError`` somewhere deep in the pipeline.
    """

    def __init__(self, data: dict[str, Any], source_path: Path | None = None) -> None:
        """Wrap an already-parsed configuration mapping.

        Args:
            data: The parsed YAML document.
            source_path: Where the document came from, used in error messages.
        """
        self._data = data
        self.source_path = source_path

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Return the value stored at ``dotted_key``.

        Args:
            dotted_key: Path through the nested mapping, e.g. ``"backend.port"``.
            default: Value to return when the key is absent.  If omitted, a
                missing key raises :class:`ConfigError`.

        Returns:
            The configured value, or ``default`` when the key is absent.

        Raises:
            ConfigError: If the key is absent and no default was supplied.
        """
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(
                        f"Missing configuration key {dotted_key!r} "
                        f"(loaded from {self.source_path})"
                    )
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return a whole top-level section as a plain dictionary.

        Args:
            name: Top-level section name, e.g. ``"synthetic"``.

        Returns:
            A shallow copy of the section so callers cannot mutate the config.
        """
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section {name!r} is not a mapping")
        return dict(value)

    def resolve_path(self, path_like: str) -> Path:
        """Turn a configured path into an absolute path under the project root.

        Relative paths in ``config.yaml`` (such as ``data/satellite.parquet``)
        are interpreted relative to the project root, so the pipeline behaves
        the same no matter which directory it is launched from.  Absolute paths
        are returned unchanged.

        Args:
            path_like: A path string from the configuration file.

        Returns:
            An absolute :class:`~pathlib.Path`.
        """
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    def database_path(self) -> Path:
        """Return the absolute path of the SQLite database file.

        Returns:
            The filesystem path taken from ``backend.database_url``.

        Raises:
            ConfigError: If the configured URL is not a SQLite URL.
        """
        url = str(self.get("backend.database_url"))
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            raise ConfigError(
                f"Only sqlite:/// URLs are supported by database_path(), got {url!r}"
            )
        return self.resolve_path(url[len(prefix):])

    def database_url(self) -> str:
        """Return the SQLAlchemy URL with any relative path made absolute.

        SQLAlchemy resolves relative SQLite paths against the current working
        directory, which would put the database in a different place depending
        on where the process was started.  Expanding it here avoids that.

        Returns:
            A SQLAlchemy connection URL.
        """
        url = str(self.get("backend.database_url"))
        if url.startswith("sqlite:///"):
            return f"sqlite:///{self.database_path().as_posix()}"
        return url

    def node_ids(self) -> list[str]:
        """Return the list of configured sensor node IDs.

        Returns:
            Node identifiers in the order they appear in ``config.yaml``.
        """
        return [str(node["node_id"]) for node in self.get("nodes")]

    def as_dict(self) -> dict[str, Any]:
        """Return the whole configuration as a plain dictionary."""
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Config(source_path={self.source_path!r})"


def load_config(path: Path | str | None = None) -> Config:
    """Read ``config.yaml`` from disk and return it as a :class:`Config`.

    Args:
        path: Optional override for the configuration file location.  Tests use
            this to load a temporary configuration.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: If the file does not exist or does not contain a mapping.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {config_path} did not contain a mapping")

    return Config(data, source_path=config_path)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the default configuration, loaded once and cached.

    Use this from long-running code (such as the FastAPI app) so that the YAML
    file is not re-read on every request.  Tests that need a different
    configuration should call :func:`load_config` directly.

    Returns:
        The cached default configuration.
    """
    return load_config()
