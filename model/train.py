"""Phase 6: train and compare RandomForest vs XGBoost leak classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from backend.config import Config

FEATURE_COLUMNS = [
    "moisture_pct",
    "moisture_deriv_pct_per_hr",
    "moisture_rolling_slope_pct_per_hr",
    "time_since_sharp_rise_hr",
    "humidity_change_pct",
    "pressure_change_hpa",
    "rain_likelihood_flag",
    "dominant_freq_hz",
    "leak_band_power",
    "spectral_entropy",
    "vibration_rms",
]


def time_based_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted feature frame into train/test without shuffling.

    Args:
        df: Feature frame with a ``window_end`` column, any row order.
        train_fraction: Fraction of rows (earliest first) used for training.

    Returns:
        ``(train_df, test_df)``, both sorted by ``window_end``.
    """
    sorted_df = df.sort_values("window_end").reset_index(drop=True)
    split_at = int(len(sorted_df) * train_fraction)
    return sorted_df.iloc[:split_at], sorted_df.iloc[split_at:]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the model's X/y from a feature frame, filling NaN warm-up rows.

    Args:
        df: Feature frame with :data:`FEATURE_COLUMNS` and ``label``.

    Returns:
        ``(X, y)``.
    """
    X = df[FEATURE_COLUMNS].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X["rain_likelihood_flag"] = X["rain_likelihood_flag"].astype(int)
    y = df["label"].astype(int)
    return X, y


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, df_test: pd.DataFrame) -> dict:
    """Compute standard metrics plus the rain-scenario false-positive rate.

    Args:
        model: A fitted classifier with ``predict``/``predict_proba``.
        X_test: Test features.
        y_test: Test labels.
        df_test: The corresponding test rows (needs a ``scenario`` column)
            for the rain-specific false-positive check.

    Returns:
        A dict of metrics, including ``rain_false_positive_rate``: the key
        metric for this project, since a false alarm on rain (not a leak)
        is the failure mode that matters most.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if "scenario" in df_test.columns:
        rain_mask = df_test["scenario"] == "rain_not_leak"
        if rain_mask.any():
            rain_pred = y_pred[rain_mask.to_numpy()]
            metrics["rain_false_positive_rate"] = float(np.mean(rain_pred == 1))

    return metrics


def train_and_compare(features: pd.DataFrame, config: Config) -> dict:
    """Train RandomForest and XGBoost, compare them, save plots and the best model.

    Args:
        features: Combined feature frame across all scenarios (needs
            :data:`FEATURE_COLUMNS`, ``label``, ``scenario``, ``window_end``).
        config: Project configuration.

    Returns:
        A dict with each model's metrics and which one was saved as "best"
        (lowest rain_not_leak false-positive rate, tie-broken by ROC-AUC).
    """
    train_df, test_df = time_based_split(features, float(config.get("model.train_fraction")))
    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    rf_cfg = config.section("model")["random_forest"]
    xgb_cfg = config.section("model")["xgboost"]

    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        random_state=int(config.get("project.random_seed")),
    ).fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        random_state=int(config.get("project.random_seed")),
        eval_metric="logloss",
    ).fit(X_train, y_train)

    results = {
        "random_forest": evaluate(rf, X_test, y_test, test_df),
        "xgboost": evaluate(xgb, X_test, y_test, test_df),
    }

    models = {"random_forest": rf, "xgboost": xgb}
    best_name = min(
        models,
        key=lambda name: (
            results[name].get("rain_false_positive_rate", 1.0),
            -results[name]["roc_auc"] if not np.isnan(results[name]["roc_auc"]) else 0.0,
        ),
    )
    best_model = models[best_name]
    results["best_model"] = best_name

    output_dir = config.resolve_path(config.get("model.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_feature_importance_plot(best_model, X_train.columns, output_dir / "feature_importance.png")
    _save_shap_summary_plot(best_model, X_test, output_dir / "shap_summary.png")

    model_path = config.resolve_path(config.get("model.model_path"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "feature_columns": FEATURE_COLUMNS, "model_name": best_name}, model_path)

    return results


def _save_feature_importance_plot(model, feature_names, path: Path) -> None:
    """Save a horizontal bar chart of feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.barh(importances.index, importances.values, color="#2a78d6")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_shap_summary_plot(model, X_sample: pd.DataFrame, path: Path) -> None:
    """Save a SHAP summary (beeswarm) plot for the trained model."""
    sample = X_sample.sample(min(200, len(X_sample)), random_state=0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class for a binary classifier

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
