"""Phase 8: render the fused leak-probability grid as an interactive Folium map."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap

from backend.config import Config


def build_leak_map(
    fused_grid: pd.DataFrame,
    node_locations: pd.DataFrame,
    latest_readings: dict[str, dict],
    config: Config,
) -> folium.Map:
    """Build the leak-probability map: heatmap + node markers + alert markers.

    Args:
        fused_grid: Output of :func:`fusion.fusion.fuse_leak_probability`
            (``lat``, ``lon``, ``leak_probability``, ...).
        node_locations: One row per node (``node_id``, ``lat``, ``lon``).
        latest_readings: Mapping from node_id to its latest reading dict
            (payload-shaped), shown in each marker's popup.
        config: Project configuration (``area.*``, ``mapping.*``).

    Returns:
        The assembled Folium map (not yet saved).
    """
    m = folium.Map(
        location=[config.get("area.center_lat"), config.get("area.center_lon")],
        zoom_start=17,
        tiles=config.get("mapping.tile_layer"),
    )

    heat_data = fused_grid[["lat", "lon", "leak_probability"]].values.tolist()
    HeatMap(heat_data, radius=int(config.get("mapping.heatmap_radius_px")), min_opacity=0.3).add_to(m)

    threshold = float(config.get("mapping.alert_probability_threshold"))
    alert_cells = fused_grid[fused_grid["leak_probability"] >= threshold]
    for _, cell in alert_cells.iterrows():
        folium.CircleMarker(
            location=[cell["lat"], cell["lon"]],
            radius=4,
            color="#e34948",
            fill=True,
            fill_color="#e34948",
            fill_opacity=0.9,
            popup=f"Leak probability: {cell['leak_probability']:.0%}",
        ).add_to(m)

    for _, node in node_locations.iterrows():
        reading = latest_readings.get(node["node_id"], {})
        popup_lines = [f"<b>{node['node_id']}</b>"]
        for key in ("timestamp", "soil_raw", "temp_c", "humidity_pct", "pressure_hpa"):
            if key in reading:
                popup_lines.append(f"{key}: {reading[key]}")
        folium.Marker(
            location=[node["lat"], node["lon"]],
            popup="<br>".join(popup_lines),
            icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def save_leak_map(fused_map: folium.Map, config: Config) -> Path:
    """Save the map to the configured output path.

    Args:
        fused_map: A map built by :func:`build_leak_map`.
        config: Project configuration (``mapping.output_path``).

    Returns:
        The absolute path the file was written to.
    """
    path = config.resolve_path(config.get("mapping.output_path"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fused_map.save(str(path))
    return path
