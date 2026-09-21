"""Pull satellite features for the configured area/date range from Google Earth Engine.

Requires a Google account with a registered Earth Engine cloud project (free).
Authentication is interactive the first time (``ee.Authenticate()`` opens a
browser); after that, credentials are cached locally and ``ee.Initialize()``
is silent.

If ``satellite.gee_project_id`` is not set, or authentication/initialization
fails for any reason (no credentials yet, no network, wrong project), this
module raises :class:`SatelliteUnavailable` instead of crashing the pipeline.
:mod:`satellite.pipeline` catches that and falls back to a synthetic frame.
"""

from __future__ import annotations

import pandas as pd

from backend.config import Config
from satellite.synthetic_fallback import SATELLITE_COLUMNS


class SatelliteUnavailable(RuntimeError):
    """Raised when Google Earth Engine cannot be reached or is not configured."""


def _initialize_ee(config: Config):
    """Authenticate and initialize the Earth Engine API.

    Args:
        config: Project configuration; needs ``satellite.gee_project_id``.

    Returns:
        The imported, initialized ``ee`` module.

    Raises:
        SatelliteUnavailable: If no project ID is configured, the
            ``earthengine-api`` package is missing, or initialization fails.
    """
    project_id = config.get("satellite.gee_project_id", None)
    if not project_id:
        raise SatelliteUnavailable(
            "satellite.gee_project_id is not set in config.yaml - "
            "falling back to synthetic satellite data"
        )

    try:
        import ee
    except ImportError as exc:
        raise SatelliteUnavailable("earthengine-api is not installed") from exc

    try:
        ee.Initialize(project=project_id)
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
        except Exception as exc:
            raise SatelliteUnavailable(
                f"Could not authenticate/initialize Earth Engine for project "
                f"{project_id!r}: {exc}"
            ) from exc

    return ee


def _area_geometry(ee_module, config: Config):
    """Build an Earth Engine rectangle geometry from the configured area.

    Args:
        ee_module: The initialized ``ee`` module.
        config: Project configuration.

    Returns:
        An ``ee.Geometry.Rectangle`` covering ``area.min_lon/min_lat`` to
        ``area.max_lon/max_lat``.
    """
    return ee_module.Geometry.Rectangle(
        [
            config.get("area.min_lon"),
            config.get("area.min_lat"),
            config.get("area.max_lon"),
            config.get("area.max_lat"),
        ]
    )


def _mask_s2_clouds(image, max_cloud_probability: float):
    """Mask cloudy pixels out of a Sentinel-2 SR image using its QA60 band.

    Args:
        image: A Sentinel-2 SR image.
        max_cloud_probability: Unused placeholder kept for signature symmetry
            with the config value; QA60 is a bitmask, not a probability, so
            the threshold is applied via the cloud/cirrus bits instead.

    Returns:
        The image with cloudy/cirrus pixels masked out.
    """
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)


def fetch_satellite_dataframe(config: Config) -> pd.DataFrame:
    """Pull Sentinel-1, Sentinel-2, SMAP, DEM and MODIS LST for the study area.

    Args:
        config: Project configuration (area, date range, collection IDs).

    Returns:
        A DataFrame indexed by date with columns matching
        :data:`satellite.synthetic_fallback.SATELLITE_COLUMNS`. A date with no
        satellite pass that day is simply absent (Sentinel revisit is every
        few days) - see :func:`satellite.pipeline.align_to_sensor_timestamps`
        for how the gap is handled downstream.

    Raises:
        SatelliteUnavailable: If Earth Engine cannot be reached or is not
            configured; see module docstring.
    """
    ee = _initialize_ee(config)
    geometry = _area_geometry(ee, config)
    start = config.get("date_range.start")
    end = config.get("date_range.end")
    cloud_max = float(config.get("satellite.s2_cloud_probability_max"))

    # --- Sentinel-1 GRD: VV/VH backscatter, averaged over the area per date.
    s1 = (
        ee.ImageCollection(config.get("satellite.collections.sentinel1"))
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )

    # --- Sentinel-2 SR: cloud-masked NDVI/NDWI.
    s2 = (
        ee.ImageCollection(config.get("satellite.collections.sentinel2"))
        .filterBounds(geometry)
        .filterDate(start, end)
        .map(lambda img: _mask_s2_clouds(img, cloud_max))
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi").addBands(
            img.normalizedDifference(["B3", "B8"]).rename("ndwi")
        ))
    )

    # --- SMAP soil moisture.
    smap = (
        ee.ImageCollection(config.get("satellite.collections.smap"))
        .filterBounds(geometry)
        .filterDate(start, end)
        .select(["sm_surface"])
        .map(lambda img: img.rename("smap_soil_moisture"))
    )

    # --- MODIS LST (Kelvin, scaled by 0.02 per the product spec -> Celsius).
    lst = (
        ee.ImageCollection(config.get("satellite.collections.modis_lst"))
        .filterBounds(geometry)
        .filterDate(start, end)
        .select(["LST_Day_1km"])
        .map(lambda img: img.multiply(0.02).subtract(273.15).rename("lst_c").copyProperties(img, ["system:time_start"]))
    )

    # --- DEM: static, one value for the whole run.
    dem = ee.Image(config.get("satellite.collections.dem")).select("DEM")
    slope = ee.Terrain.slope(dem)
    terrain_stats = dem.addBands(slope).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=30, maxPixels=1e9
    ).getInfo()
    elevation_m = terrain_stats.get("DEM")
    slope_deg = terrain_stats.get("slope")

    def _collection_to_rows(collection, band_names: list[str], reducer_scale: int) -> list[dict]:
        """Reduce every image in a collection to one row of area-mean values."""

        def _reduce_one(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geometry, scale=reducer_scale, maxPixels=1e9
            )
            return ee.Feature(None, stats.set("date", image.date().format("YYYY-MM-dd")))

        features = collection.map(_reduce_one).getInfo()["features"]
        return [feature["properties"] for feature in features]

    s1_rows = _collection_to_rows(s1, ["VV", "VH"], 10)
    s2_rows = _collection_to_rows(s2, ["ndvi", "ndwi"], 10)
    smap_rows = _collection_to_rows(smap, ["smap_soil_moisture"], 1000)
    lst_rows = _collection_to_rows(lst, ["lst_c"], 1000)

    def _rows_to_frame(rows: list[dict], rename: dict[str, str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(rename.values())).set_index(
                pd.DatetimeIndex([], name="date")
            )
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").rename(columns=rename)
        # Two passes on the same day (e.g. overlapping S1 orbits) average out.
        return frame.groupby(level=0).mean()

    s1_df = _rows_to_frame(s1_rows, {"VV": "s1_vv_db", "VH": "s1_vh_db"})
    s2_df = _rows_to_frame(s2_rows, {"ndvi": "ndvi", "ndwi": "ndwi"})
    smap_df = _rows_to_frame(smap_rows, {"smap_soil_moisture": "smap_soil_moisture"})
    lst_df = _rows_to_frame(lst_rows, {"lst_c": "lst_c"})

    combined = s1_df.join(s2_df, how="outer").join(smap_df, how="outer").join(lst_df, how="outer")
    combined["elevation_m"] = elevation_m
    combined["slope_deg"] = slope_deg
    combined = combined.reindex(columns=SATELLITE_COLUMNS)
    combined = combined.sort_index()
    return combined
