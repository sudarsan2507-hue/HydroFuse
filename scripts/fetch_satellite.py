r"""Fetch (or synthesize) satellite features and save them to data/satellite.parquet.

Run it with::

    .venv\Scripts\python scripts\fetch_satellite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import load_config  # noqa: E402
from satellite.pipeline import get_satellite_dataframe, save_satellite_dataframe  # noqa: E402


def main() -> int:
    """Fetch the satellite frame and save it, printing a short summary."""
    config = load_config()
    df = get_satellite_dataframe(config)
    path = save_satellite_dataframe(df, config)
    print(f"Saved {len(df)} daily rows to {path}")
    print(df.describe().T[["mean", "min", "max"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
