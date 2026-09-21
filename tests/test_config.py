"""Phase 1 tests: the configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import PROJECT_ROOT, Config, ConfigError, load_config


def test_config_file_exists() -> None:
    """The project ships a config.yaml at the repository root."""
    assert (PROJECT_ROOT / "config.yaml").exists()


def test_expected_sections_present(config: Config) -> None:
    """Every phase's configuration section is present."""
    expected = [
        "project",
        "area",
        "date_range",
        "nodes",
        "sensor",
        "backend",
        "synthetic",
        "satellite",
        "features",
        "model",
        "fusion",
        "mapping",
    ]
    for section in expected:
        assert config.get(section) is not None, "missing section: {}".format(section)


def test_dotted_lookup(config: Config) -> None:
    """Nested values are reachable with a dotted key."""
    assert isinstance(config.get("area.center_lat"), float)
    assert isinstance(config.get("backend.port"), int)


def test_missing_key_raises(config: Config) -> None:
    """A missing key without a default raises a clear error naming the key."""
    with pytest.raises(ConfigError) as excinfo:
        config.get("area.does_not_exist")
    assert "area.does_not_exist" in str(excinfo.value)


def test_missing_key_returns_default(config: Config) -> None:
    """A missing key with a default returns that default, including None."""
    assert config.get("area.does_not_exist", 7) == 7
    assert config.get("area.does_not_exist", None) is None


def test_relative_paths_resolve_against_project_root(config: Config) -> None:
    """Relative configured paths become absolute paths under the project root."""
    resolved = config.resolve_path("data/satellite.parquet")
    assert resolved.is_absolute()
    assert resolved.parent == (PROJECT_ROOT / "data").resolve()


def test_database_url_is_absolute(config: Config) -> None:
    """The SQLAlchemy URL does not depend on the current working directory."""
    url = config.database_url()
    assert url.startswith("sqlite:///")
    assert config.database_path().is_absolute()


def test_node_ids_match_nodes_section(config: Config) -> None:
    """node_ids() lists every configured node, in order."""
    node_ids = config.node_ids()
    assert node_ids == [node["node_id"] for node in config.get("nodes")]
    assert len(node_ids) == len(set(node_ids)), "node IDs must be unique"


def test_nodes_lie_inside_the_configured_area(config: Config) -> None:
    """Each node sits inside the study-area bounding box."""
    for node in config.get("nodes"):
        assert config.get("area.min_lat") <= node["lat"] <= config.get("area.max_lat")
        assert config.get("area.min_lon") <= node["lon"] <= config.get("area.max_lon")


def test_soil_calibration_is_consistent(config: Config) -> None:
    """Lower ADC counts mean wetter soil, so the wet value is below the dry one."""
    assert config.get("sensor.soil_raw_wet") < config.get("sensor.soil_raw_dry")


def test_missing_file_raises(tmp_path: Path) -> None:
    """Loading a configuration file that does not exist raises ConfigError."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_malformed_file_raises(tmp_path: Path) -> None:
    """A YAML file that is not a mapping raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_section_returns_a_copy(config: Config) -> None:
    """Mutating a returned section does not change the loaded configuration."""
    section = config.section("backend")
    section["port"] = 1
    assert config.get("backend.port") != 1
