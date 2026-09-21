"""Database engine and session management.

The backend uses SQLite through SQLAlchemy 2.0.  The engine is created lazily
from ``backend.database_url`` in ``config.yaml`` so that tests can point the
application at a temporary database before the first request is handled.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import Config, get_config
from backend.models import Base

# Module-level singletons.  They stay None until the first call to
# `get_engine()` or an explicit `configure_engine()` from a test fixture.
_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def configure_engine(database_url: str, echo: bool = False) -> Engine:
    """Create (or replace) the global engine and session factory.

    Args:
        database_url: A SQLAlchemy connection URL.
        echo: If True, SQLAlchemy logs every statement it runs.

    Returns:
        The newly created engine.
    """
    global _engine, _SessionFactory

    connect_args = {}
    if database_url.startswith("sqlite"):
        # FastAPI may serve a request on a different thread than the one that
        # opened the connection; SQLite refuses that by default.
        connect_args["check_same_thread"] = False
        _ensure_parent_directory(database_url)

    _engine = create_engine(database_url, echo=echo, future=True, connect_args=connect_args)
    _SessionFactory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def _ensure_parent_directory(database_url: str) -> None:
    """Create the directory that will hold a SQLite file, if it is missing.

    Args:
        database_url: A ``sqlite:///`` URL.  Anything else is ignored.
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return
    file_part = database_url[len(prefix):]
    if not file_part or file_part == ":memory:":
        return
    Path(file_part).parent.mkdir(parents=True, exist_ok=True)


def get_engine(config: Config | None = None) -> Engine:
    """Return the global engine, creating it from configuration if needed.

    Args:
        config: Optional configuration override.  Defaults to the cached
            project configuration.

    Returns:
        The SQLAlchemy engine.
    """
    global _engine
    if _engine is None:
        cfg = config if config is not None else get_config()
        configure_engine(cfg.database_url())
    assert _engine is not None  # for type checkers; configure_engine sets it
    return _engine


def init_db(config: Config | None = None) -> None:
    """Create any tables that do not exist yet.

    This is enough for a research prototype; a production system would use a
    migration tool such as Alembic instead.

    Args:
        config: Optional configuration override.
    """
    engine = get_engine(config)
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """Yield a database session, closing it when the caller is done.

    This is used as a FastAPI dependency (``Depends(get_session)``) and can
    also be used directly with ``next(get_session())`` in scripts.

    Yields:
        An open SQLAlchemy session bound to the global engine.
    """
    get_engine()  # make sure the factory exists
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine and session factory.

    Tests call this between cases so that each one starts from a clean
    database connection.
    """
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
