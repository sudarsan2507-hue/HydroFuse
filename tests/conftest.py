"""Shared pytest fixtures.

Every test that touches the database gets its own temporary SQLite file, so
tests never see each other's rows and never write into ``data/readings.db``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Make the project root importable when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config, load_config  # noqa: E402
from backend.db import configure_engine, init_db, reset_engine  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture()
def config() -> Config:
    """Return the project configuration loaded from ``config.yaml``."""
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture()
def temp_database(tmp_path: Path) -> Iterator[Path]:
    """Point the backend at a fresh SQLite file for the duration of one test.

    Yields:
        The path of the temporary database file.
    """
    db_path = tmp_path / "test_readings.db"
    configure_engine("sqlite:///{}".format(db_path.as_posix()))
    init_db()
    try:
        yield db_path
    finally:
        reset_engine()


@pytest.fixture()
def client(temp_database: Path) -> Iterator[TestClient]:
    """Return a FastAPI test client backed by the temporary database.

    Yields:
        A client whose requests hit the real application and real database
        code, just against a throwaway file.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def session(temp_database: Path) -> Iterator[Session]:
    """Return a database session bound to the temporary database.

    Yields:
        An open session, closed when the test finishes.
    """
    from backend.db import get_session

    generator = get_session()
    db_session = next(generator)
    try:
        yield db_session
    finally:
        generator.close()
