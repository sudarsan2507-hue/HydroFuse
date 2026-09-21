"""Train a leak classifier on LeakDB's real (pressure-based) data.

This is intentionally a separate, simpler model from
:mod:`model.train` - LeakDB measures network pressure at junctions, not
ground vibration/soil moisture, so its features are different and its
results say nothing about the accelerometer-based approach directly. It
exists as an independent sanity check: "does a simple classifier detect
leaks in real, published water-network data at all?" - using a
leave-scenarios-out split (whole simulated years held out, never split
mid-year) so nothing about a specific scenario's leak leaks into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from backend.config import Config

FEATURE_COLUMNS = ["mean_pressure", "min_pressure", "std_pressure", "pressure_change"]


def build_pressure_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Turn LeakDB's per-node pressure columns into a compact feature set.

    Args:
        df: Output of :func:`external_data.leakdb.load_scenario` or
            :func:`external_data.leakdb.load_scenarios` (one column per
            node, plus ``label`` and ``timestamp``).
        window: Readings to look back for the pressure-change feature (30
            minutes each).

    Returns:
        A DataFrame with :data:`FEATURE_COLUMNS` plus ``label`` (and
        ``scenario_id``/``timestamp`` if present in the input).
    """
    node_columns = [c for c in df.columns if c.startswith("pressure_node_")]
    pressures = df[node_columns]

    features = pd.DataFrame(index=df.index)
    features["mean_pressure"] = pressures.mean(axis=1)
    features["min_pressure"] = pressures.min(axis=1)
    features["std_pressure"] = pressures.std(axis=1)

    # Group by scenario (if present) so the lagged diff never crosses from
    # one simulated year into another.
    group_key = df["scenario_id"] if "scenario_id" in df.columns else pd.Series(0, index=df.index)
    features["pressure_change"] = features["mean_pressure"] - features.groupby(group_key)["mean_pressure"].shift(window)

    features["label"] = df["label"].astype(int)
    if "scenario_id" in df.columns:
        features["scenario_id"] = df["scenario_id"]
    if "timestamp" in df.columns:
        features["timestamp"] = df["timestamp"]

    return features.dropna(subset=FEATURE_COLUMNS)


def train_and_evaluate(features: pd.DataFrame, config: Config) -> dict:
    """Train a RandomForest on LeakDB features, split by scenario.

    Args:
        features: Output of :func:`build_pressure_features`, with a
            ``scenario_id`` column.
        config: Project configuration (``leakdb.train_scenarios`` /
            ``leakdb.test_scenarios``, ``model.random_forest``).

    Returns:
        A dict of metrics: precision, recall, f1, roc_auc, confusion_matrix,
        plus ``n_train``/``n_test``.
    """
    train_ids = set(config.get("leakdb.train_scenarios"))
    test_ids = set(config.get("leakdb.test_scenarios"))

    train_df = features[features["scenario_id"].isin(train_ids)]
    test_df = features[features["scenario_id"].isin(test_ids)]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    rf_cfg = config.section("model")["random_forest"]
    model = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        random_state=int(config.get("project.random_seed")),
        class_weight="balanced",
    ).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "LeakDB (Hanoi_CMH, real published network-pressure data)",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_scenarios": sorted(train_ids),
        "test_scenarios": sorted(test_ids),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, model
