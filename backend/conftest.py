"""Test-wide setup: importable package, and a database and storage of its own.

Two things happen here, both before ``app`` is imported by any test.

``backend/`` goes on the path so tests can ``from app.services... import ...``.

The database URL and storage root are redirected into a temporary directory. This is
not tidiness: without it the suite would read and write the developer's real
``mlbenchx.db`` and ``storage/``, so tests would see each other's rows, a failing test
would leave debris that changed the next run's results, and running the suite would
quietly mutate whatever the developer had been looking at. Config reads these from the
environment at import, which is why they are set here rather than in a fixture.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_SANDBOX = Path(tempfile.mkdtemp(prefix="mlbenchx-tests-"))

import os  # noqa: E402 - must follow the sandbox, precede any app import

# The suite runs against SQLite by default and against PostgreSQL when asked to, because
# the two are not interchangeable in the ways that bite: server-side defaults, JSON
# column types, timezone-aware timestamps and transactional DDL all differ. Stage 4's
# experiment history is specified to live on PostgreSQL, so "it passes on SQLite" stopped
# being enough.
#
# A separate variable rather than reusing MLBENCHX_DATABASE_URL on purpose: the suite
# drops and recreates every table between tests, and a developer who had exported the
# normal variable to point at their working database would lose it.
os.environ["MLBENCHX_DATABASE_URL"] = os.environ.get(
    "MLBENCHX_TEST_DATABASE_URL"
) or f"sqlite:///{_SANDBOX / 'test.db'}"
os.environ["MLBENCHX_STORAGE_ROOT"] = str(_SANDBOX / "storage")

from app.db import Base, engine, init_db  # noqa: E402


def pytest_sessionstart(session) -> None:
    """Create the schema once, in the sandbox database."""
    init_db()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Remove the sandbox. Nothing in it is worth keeping between runs."""
    engine.dispose()
    shutil.rmtree(_SANDBOX, ignore_errors=True)


@pytest.fixture
def clean_db():
    """An empty database for a test that counts rows or lists datasets."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
