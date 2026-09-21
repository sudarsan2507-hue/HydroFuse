"""Phase 7: fuse node-corrected satellite moisture with the classifier's leak probability.

**Method (the project's core novelty):**

1. **Local bias correction.** At each sensor node, compare its own calibrated
   soil moisture (Phase 5) to the coarse satellite (SMAP) estimate at the
   same time/place. The difference is that node's *local bias* - how far off
   the satellite pixel is at this specific spot (vegetation, soil type,
   micro-topography all shift SMAP away from ground truth).
2. **Inverse-distance spreading.** Each node's bias is spread across the
   whole grid, weighted by ``1 / distance^power`` and cut off beyond
   ``fusion.idw_cutoff_m`` - nearby grid cells trust a node's correction
   almost fully; distant cells barely feel it. Where multiple nodes'
   influence overlaps, their corrections are averaged, weighted the same way.
3. **Corrected moisture anomaly.** Adding the spread bias to the satellite's
   raw estimate gives a corrected moisture value per grid cell; its
   deviation from the area's normal moisture is the "moisture anomaly".
4. **Final fusion.** The per-cell leak probability is a weighted blend of the
   trained classifier's output (from the nearest node' features) and the
   corrected moisture anomaly, combined via ``fusion.classifier_weight`` /
   ``fusion.moisture_anomaly_weight``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import Config

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres between one point and an array of points.

    Args:
        lat1, lon1: The reference point, in decimal degrees.
        lat2, lon2: Arrays of target points, in decimal degrees.

    Returns:
        Distances in metres, same shape as ``lat2``/``lon2``.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_node_bias(node_moisture_pct: float, satellite_moisture_fraction: float) -> float:
    """Local bias between a node's calibrated moisture and the satellite estimate.

    Args:
        node_moisture_pct: Node's calibrated soil moisture, 0-100%.
        satellite_moisture_fraction: SMAP soil moisture, 0-1 fraction.

    Returns:
        Bias in percentage points (node minus satellite, both on a 0-100 scale).
    """
    return node_moisture_pct - satellite_moisture_fraction * 100.0


def build_grid(config: Config) -> pd.DataFrame:
    """Build a regular lat/lon grid covering the configured area.

    Args:
        config: Project configuration (``area.*``, ``area.grid_resolution_m``).

    Returns:
        A DataFrame with one row per grid cell: ``lat``, ``lon``.
    """
    min_lat, max_lat = config.get("area.min_lat"), config.get("area.max_lat")
    min_lon, max_lon = config.get("area.min_lon"), config.get("area.max_lon")
    resolution_m = float(config.get("area.grid_resolution_m"))

    lat_step_deg = resolution_m / 111_320.0  # ~metres per degree latitude
    center_lat = (min_lat + max_lat) / 2.0
    lon_step_deg = resolution_m / (111_320.0 * np.cos(np.radians(center_lat)))

    lats = np.arange(min_lat, max_lat, lat_step_deg)
    lons = np.arange(min_lon, max_lon, lon_step_deg)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    return pd.DataFrame({"lat": grid_lat.ravel(), "lon": grid_lon.ravel()})


def spread_bias_idw(
    grid: pd.DataFrame,
    node_locations: pd.DataFrame,
    node_biases: pd.Series,
    power: float,
    cutoff_m: float,
) -> np.ndarray:
    """Spread per-node bias corrections across a grid by inverse-distance weighting.

    Args:
        grid: Grid cells, with ``lat``/``lon`` columns.
        node_locations: One row per node, with ``lat``/``lon`` columns, same
            order as ``node_biases``.
        node_biases: Each node's local bias (see :func:`compute_node_bias`).
        power: IDW exponent; higher keeps the correction more local.
        cutoff_m: Nodes farther than this from a cell do not influence it.

    Returns:
        One spread bias value per grid cell (0.0 where no node is in range).
    """
    n_cells = len(grid)
    n_nodes = len(node_locations)
    weights = np.zeros((n_cells, n_nodes))
    distances = np.zeros((n_cells, n_nodes))

    for j in range(n_nodes):
        node = node_locations.iloc[j]
        distances[:, j] = haversine_distance_m(node["lat"], node["lon"], grid["lat"].to_numpy(), grid["lon"].to_numpy())

    within_cutoff = distances <= cutoff_m
    # Avoid division by zero when a grid cell sits exactly on a node.
    safe_distances = np.where(distances < 1.0, 1.0, distances)
    weights = np.where(within_cutoff, 1.0 / (safe_distances**power), 0.0)

    weight_sums = weights.sum(axis=1)
    spread = np.divide(
        weights @ node_biases.to_numpy(),
        weight_sums,
        out=np.zeros(n_cells),
        where=weight_sums > 0,
    )
    return spread


def fuse_leak_probability(
    grid: pd.DataFrame,
    node_locations: pd.DataFrame,
    node_biases: pd.Series,
    satellite_moisture_fraction: float,
    normal_moisture_pct: float,
    classifier_probability_by_node: pd.Series,
    config: Config,
) -> pd.DataFrame:
    """Combine corrected moisture anomaly with classifier output into a per-cell probability.

    Args:
        grid: Grid cells (``lat``/``lon``).
        node_locations: One row per node (``lat``/``lon``), same order as
            ``node_biases`` and ``classifier_probability_by_node``.
        node_biases: Per-node local bias (see :func:`compute_node_bias`).
        satellite_moisture_fraction: Area-wide SMAP estimate (0-1), applied
            uniformly across the grid before correction (a real deployment
            would use a per-cell satellite pixel value instead).
        normal_moisture_pct: The area's normal moisture level, for computing
            the anomaly (e.g. the no_leak baseline, calibrated to %).
        classifier_probability_by_node: Each node's classifier leak
            probability, spread across the grid the same way as the bias
            (nearest-node influence falls off with distance).
        config: Project configuration (``fusion.*``).

    Returns:
        ``grid`` with added columns ``corrected_moisture_pct``,
        ``moisture_anomaly_pct``, ``classifier_probability`` and
        ``leak_probability`` (the final fused value, clipped to [0, 1]).
    """
    power = float(config.get("fusion.idw_power"))
    cutoff_m = float(config.get("fusion.idw_cutoff_m"))

    spread_bias = spread_bias_idw(grid, node_locations, node_biases, power, cutoff_m)
    corrected_moisture_pct = satellite_moisture_fraction * 100.0 + spread_bias
    moisture_anomaly_pct = corrected_moisture_pct - normal_moisture_pct

    spread_classifier_prob = spread_bias_idw(
        grid, node_locations, classifier_probability_by_node, power, cutoff_m
    )

    # Normalise the anomaly to a 0-1 "moisture-driven probability": a
    # anomaly of 0 or negative (drier/normal) contributes ~0; each
    # percentage point wetter than normal raises it, saturating at 1.
    moisture_component = np.clip(moisture_anomaly_pct / 20.0, 0.0, 1.0)

    classifier_weight = float(config.get("fusion.classifier_weight"))
    moisture_weight = float(config.get("fusion.moisture_anomaly_weight"))

    result = grid.copy()
    result["corrected_moisture_pct"] = corrected_moisture_pct
    result["moisture_anomaly_pct"] = moisture_anomaly_pct
    result["classifier_probability"] = spread_classifier_prob
    result["leak_probability"] = np.clip(
        classifier_weight * spread_classifier_prob + moisture_weight * moisture_component, 0.0, 1.0
    )
    return result
