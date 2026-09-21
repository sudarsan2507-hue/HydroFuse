"""A small synthetic satellite frame, used when Google Earth Engine is not available.

Produces the exact same columns as the real GEE pull
(:mod:`satellite.gee_client`), one row per day, so downstream code never
needs to know which source it got its data from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import Config

SATELLITE_COLUMNS = [
    "s1_vv_db",
    "s1_vh_db",
    "ndvi",
    "ndwi",
    "smap_soil_moisture",
    "elevation_m",
    "slope_deg",
    "lst_c",
]


def generate_synthetic_satellite_frame(config: Config, seed: int | None = None) -> pd.DataFrame:
    """Build a plausible daily satellite DataFrame for the configured date range.

    Args:
        config: Project configuration (uses ``date_range`` and
            ``project.random_seed``).
        seed: Optional seed override; defaults to ``project.random_seed``.

    Returns:
        A DataFrame indexed by date (one row per day in the configured
        range) with the same columns real GEE data would have.
    """
    if seed is None:
        seed = int(config.get("project.random_seed"))
    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        start=config.get("date_range.start"),
        end=config.get("date_range.end"),
        freq="D",
    )
    n = len(dates)

    # Terrain (elevation/slope) is static per area, not per day, but is
    # reported once per row for a simple tidy-frame join downstream.
    elevation = float(rng.uniform(700.0, 720.0))
    slope = float(rng.uniform(1.0, 4.0))

    df = pd.DataFrame(
        {
            "s1_vv_db": -10.0 + rng.normal(0.0, 1.0, n),
            "s1_vh_db": -17.0 + rng.normal(0.0, 1.0, n),
            "ndvi": np.clip(0.4 + rng.normal(0.0, 0.05, n), -1.0, 1.0),
            "ndwi": np.clip(0.1 + rng.normal(0.0, 0.05, n), -1.0, 1.0),
            "smap_soil_moisture": np.clip(0.25 + rng.normal(0.0, 0.03, n), 0.0, 1.0),
            "elevation_m": elevation,
            "slope_deg": slope,
            "lst_c": 28.0 + rng.normal(0.0, 2.0, n),
        },
        index=pd.Index(dates, name="date"),
    )
    return df
