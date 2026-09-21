r"""Phase 9: single entry point for the whole pipeline.

    .venv\Scripts\python run_pipeline.py --source synthetic
    .venv\Scripts\python run_pipeline.py --source db

Both sources produce readings in the exact same shape (the ESP32 payload),
so everything downstream - features, fusion, the map - runs unmodified
regardless of ``--source``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from backend.config import load_config
from features.extract import build_feature_frame
from features.moisture import calibrate_moisture_pct
from fusion.fusion import build_grid, compute_node_bias, fuse_leak_probability
from mapping.map import build_leak_map, save_leak_map
from model.train import FEATURE_COLUMNS
from satellite.pipeline import get_satellite_dataframe
from synthetic.generator import generate_scenario


def load_readings_synthetic(config) -> dict[str, pd.DataFrame]:
    """Generate a synthetic reading history for every configured node.

    Nodes cycle through the configured scenarios so the demo map shows a mix
    of conditions rather than three identical nodes.
    """
    scenarios = list(config.get("synthetic.scenarios"))
    seed = int(config.get("project.random_seed"))
    result = {}
    for i, node in enumerate(config.get("nodes")):
        scenario = scenarios[i % len(scenarios)]
        df = generate_scenario(scenario, config, seed=seed + i)
        df = df.copy()
        df["node_id"] = node["node_id"]
        result[node["node_id"]] = df
    return result


def load_readings_db(config) -> dict[str, pd.DataFrame]:
    """Load reading history for every node from the SQLite database."""
    from backend.db import get_session, init_db
    from backend.models import Reading

    init_db()
    session = next(get_session())
    rows = session.query(Reading).order_by(Reading.timestamp).all()
    session.close()

    by_node: dict[str, list[dict]] = {}
    for row in rows:
        by_node.setdefault(row.node_id, []).append(row.to_dict())

    return {node_id: pd.DataFrame(records) for node_id, records in by_node.items() if records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "db"], default="synthetic")
    args = parser.parse_args()

    config = load_config()
    print(f"[pipeline] source={args.source}")

    readings_by_node = load_readings_synthetic(config) if args.source == "synthetic" else load_readings_db(config)
    if not readings_by_node:
        print("[pipeline] No readings available for the requested source; nothing to do.")
        return 1

    satellite_df = get_satellite_dataframe(config)
    satellite_moisture_fraction = float(satellite_df["smap_soil_moisture"].iloc[-1])
    normal_moisture_pct = calibrate_moisture_pct(
        pd.Series([float(config.get("synthetic.soil.no_leak_baseline_raw"))]),
        raw_dry=float(config.get("sensor.soil_raw_dry")),
        raw_wet=float(config.get("sensor.soil_raw_wet")),
    ).iloc[0]

    try:
        bundle = joblib.load(config.resolve_path(config.get("model.model_path")))
        model = bundle["model"]
    except FileNotFoundError:
        print("[pipeline] No trained model found; run scripts/train_model.py first. Using probability=0.5 for all nodes.")
        model = None

    node_ids, biases, probs, latest_readings = [], [], [], {}
    for node_id, df in readings_by_node.items():
        features = build_feature_frame(df, config, satellite_df=satellite_df)
        if features.empty:
            continue
        last = features.iloc[-1]

        bias = compute_node_bias(last["moisture_pct"], satellite_moisture_fraction)

        if model is not None:
            X = last[FEATURE_COLUMNS].to_frame().T
            X["rain_likelihood_flag"] = X["rain_likelihood_flag"].astype(int)
            prob = float(model.predict_proba(X)[:, 1][0])
        else:
            prob = 0.5

        node_ids.append(node_id)
        biases.append(bias)
        probs.append(prob)
        clean_df = df[~df["dropout"]] if "dropout" in df.columns else df
        latest_readings[node_id] = clean_df.iloc[-1].to_dict()

    node_locations = pd.DataFrame(
        [{"node_id": n["node_id"], "lat": n["lat"], "lon": n["lon"]} for n in config.get("nodes") if n["node_id"] in node_ids]
    ).set_index("node_id").loc[node_ids].reset_index()

    grid = build_grid(config)
    fused = fuse_leak_probability(
        grid,
        node_locations,
        pd.Series(biases, index=node_ids),
        satellite_moisture_fraction,
        normal_moisture_pct,
        pd.Series(probs, index=node_ids),
        config,
    )

    leak_map = build_leak_map(fused, node_locations, latest_readings, config)
    path = save_leak_map(leak_map, config)

    print(f"[pipeline] {len(node_ids)} node(s) processed.")
    print(f"[pipeline] Leak probability range: {fused['leak_probability'].min():.2f} - {fused['leak_probability'].max():.2f}")
    print(f"[pipeline] Map saved to {path}")

    # A small summary the dashboard UI reads to show per-node risk without
    # having to re-run the whole fusion pipeline on every page load.
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "leak_probability_min": float(fused["leak_probability"].min()),
        "leak_probability_max": float(fused["leak_probability"].max()),
        "nodes": [
            {
                "node_id": node_id,
                "leak_probability": prob,
                "moisture_bias_pct": bias,
                "latest_reading": {
                    k: v
                    for k, v in latest_readings.get(node_id, {}).items()
                    if k in ("timestamp", "soil_raw", "temp_c", "humidity_pct", "pressure_hpa")
                },
            }
            for node_id, prob, bias in zip(node_ids, probs, biases)
        ],
    }
    summary_path = config.resolve_path(config.get("mapping.output_path")).parent / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[pipeline] Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
