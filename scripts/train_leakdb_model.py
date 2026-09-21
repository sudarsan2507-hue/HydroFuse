r"""Train and evaluate a leak classifier on the real LeakDB dataset.

Run scripts\fetch_leakdb.py first. Then:

    .venv\Scripts\python scripts\train_leakdb_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import load_config  # noqa: E402
from external_data.leakdb import load_scenarios  # noqa: E402
from model.train_leakdb import build_pressure_features, train_and_evaluate  # noqa: E402


def main() -> int:
    config = load_config()
    extract_dir = config.resolve_path(config.get("leakdb.extract_dir"))
    if not extract_dir.exists():
        print(f"No LeakDB data at {extract_dir}. Run scripts/fetch_leakdb.py first.")
        return 1

    all_scenarios = sorted(set(config.get("leakdb.train_scenarios")) | set(config.get("leakdb.test_scenarios")))
    print(f"Loading LeakDB scenarios {all_scenarios} from {extract_dir} ...")
    raw = load_scenarios(extract_dir, all_scenarios)
    print(f"Loaded {len(raw)} timestamped rows across {len(all_scenarios)} scenarios.")

    features = build_pressure_features(raw, window=int(config.get("leakdb.pressure_change_window_readings")))
    print(f"Built {len(features)} feature rows (after dropping warm-up rows).")

    metrics, model = train_and_evaluate(features, config)

    print(f"\nTrain rows: {metrics['n_train']} (scenarios {metrics['train_scenarios']})")
    print(f"Test rows:  {metrics['n_test']} (scenarios {metrics['test_scenarios']})")
    print(f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}")
    print(f"confusion_matrix={metrics['confusion_matrix']}")

    model_path = config.resolve_path(config.get("leakdb.model_path"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": ["mean_pressure", "min_pressure", "std_pressure", "pressure_change"]}, model_path)

    metrics_path = config.resolve_path(config.get("leakdb.metrics_path"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
