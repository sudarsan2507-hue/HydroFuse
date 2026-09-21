"""Load LeakDB scenarios into tidy pandas DataFrames.

Each LeakDB scenario is a simulated year (30-minute steps) of a real water
distribution network (Hanoi), with one pressure CSV per node, a
``Labels.csv`` marking every timestamp a leak was active anywhere in the
network, and (if the scenario has one) a ``Leaks/Leak_<node>_info.csv`` with
the leak's node, size and start/end time.

This is real, published data - but pressure at network junctions, not
accelerometer/soil-moisture readings at a pipe. It is kept in its own module
and its own model (see :mod:`model.train_leakdb`) rather than merged into the
project's own sensor pipeline, which measures something physically
different.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_timeseries_csv(path: Path) -> pd.Series:
    """Read one LeakDB two-column timestamped CSV into a Series indexed by time.

    Node pressure files use the header ``Timestamp,Value``; ``Labels.csv``
    uses ``Timestamp,Label``. Both are read the same way, keyed off
    whichever second column is actually present.
    """
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    value_column = [c for c in df.columns if c != "Timestamp"][0]
    return df.set_index("Timestamp")[value_column]


def load_scenario(scenario_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load one scenario's node pressures, leak label, and metadata.

    Args:
        scenario_dir: Path to a ``Scenario-N`` directory (as extracted by
            ``scripts/fetch_leakdb.py``).

    Returns:
        A tuple ``(df, meta)``:

        - ``df``: indexed by timestamp, one column per node
          (``pressure_node_<id>``) plus ``label`` (1.0 while any leak is
          active in the network, 0.0 otherwise).
        - ``meta``: scenario metadata - ``network_name``, ``duration``,
          and, if the scenario has a leak, ``leak_node``/``leak_start``/
          ``leak_end``/``leak_diameter_m``.
    """
    scenario_dir = Path(scenario_dir)

    pressure_dir = scenario_dir / "Pressures"
    node_series = {}
    for csv_path in sorted(pressure_dir.glob("Node_*.csv")):
        node_id = csv_path.stem.replace("Node_", "")
        node_series[f"pressure_node_{node_id}"] = _read_timeseries_csv(csv_path)

    df = pd.DataFrame(node_series)
    df["label"] = _read_timeseries_csv(scenario_dir / "Labels.csv").reindex(df.index)
    df = df.rename_axis("timestamp").reset_index()

    meta: dict = {"scenario": scenario_dir.name, "leaks": []}
    leaks_dir = scenario_dir / "Leaks"
    for leak_info_path in sorted(leaks_dir.glob("Leak_*_info.csv")) if leaks_dir.exists() else []:
        info = pd.read_csv(leak_info_path)
        info.columns = [c.strip() for c in info.columns]
        values = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in zip(info["Description"], info["Value"])}
        meta["leaks"].append(
            {
                "node": values.get("Leak Node"),
                "start": values.get("Leak Start"),
                "end": values.get("Leak End"),
                "diameter_m": values.get("Leak Diameter"),
            }
        )

    return df, meta


def load_scenarios(extract_dir: Path, scenario_ids: list[int]) -> pd.DataFrame:
    """Load and concatenate several scenarios, tagged with a scenario_id column.

    Args:
        extract_dir: Directory containing ``Scenario-N`` folders.
        scenario_ids: Which scenario numbers to load.

    Returns:
        A concatenated DataFrame with an added ``scenario_id`` column,
        one row per (scenario, timestamp).
    """
    extract_dir = Path(extract_dir)
    frames = []
    for scenario_id in scenario_ids:
        df, _meta = load_scenario(extract_dir / f"Scenario-{scenario_id}")
        df["scenario_id"] = scenario_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
