r"""Phase 6: build features from synthetic data and train the leak classifier.

Run: .venv\Scripts\python scripts\train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backend.config import load_config  # noqa: E402
from features.extract import build_feature_frame  # noqa: E402
from model.train import train_and_compare  # noqa: E402
from synthetic.generator import generate_all  # noqa: E402


def main() -> int:
    config = load_config()
    seed = int(config.get("project.random_seed"))

    print("Generating synthetic readings and building features...")
    # Each scenario is its own continuous time series (they all share a
    # node_id and base_time by design - see config.yaml), so features are
    # built per scenario and then stacked, never by concatenating raw
    # readings across scenarios first (that would interleave three
    # unrelated series under one timeline).
    scenario_readings = generate_all(config, seed=seed)
    features = pd.concat(
        [build_feature_frame(df, config) for df in scenario_readings.values()],
        ignore_index=True,
    )
    print(f"Built {len(features)} feature rows.")

    print("Training RandomForest and XGBoost (time-based split)...")
    results = train_and_compare(features, config)

    for name in ("random_forest", "xgboost"):
        m = results[name]
        print(f"\n{name}:")
        print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f}")
        print(f"  confusion_matrix={m['confusion_matrix']}")
        if "rain_false_positive_rate" in m:
            print(f"  rain_not_leak false-positive rate: {m['rain_false_positive_rate']:.3%}")

    print(f"\nBest model: {results['best_model']}")
    print(f"Saved to {config.resolve_path(config.get('model.model_path'))}")
    print(f"Plots saved to {config.resolve_path(config.get('model.output_dir'))}")

    metrics_path = config.resolve_path(config.get("model.output_dir")) / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Metrics saved to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
