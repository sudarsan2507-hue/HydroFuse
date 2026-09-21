r"""Post fake ESP32 payloads to the running backend, so it can be tested
without hardware.

This generator is deliberately simple: plausible values with a bit of noise,
just enough to exercise ingestion, validation and querying.  The physically
realistic, labelled scenario generator (no_leak / leak / rain_not_leak) is
Phase 3 and lives in ``synthetic/``; this script will keep working afterwards
as a quick smoke test.

Examples::

    # 1. start the API in one terminal
    .venv\Scripts\python -m uvicorn backend.main:app --reload

    # 2. post 10 readings from every configured node in another
    .venv\Scripts\python scripts/post_fake_payloads.py --count 10

    # print payloads without sending them
    .venv\Scripts\python scripts/post_fake_payloads.py --count 2 --dry-run

    # send one deliberately invalid payload to check the API rejects it
    .venv\Scripts\python scripts/post_fake_payloads.py --invalid
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

# Allow `python scripts/post_fake_payloads.py` to import the project packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Config, load_config  # noqa: E402


def make_payload(
    node_id: str,
    timestamp: datetime,
    config: Config,
    rng: random.Random,
) -> dict[str, Any]:
    """Build one plausible sensor payload.

    Args:
        node_id: The node the reading claims to come from.
        timestamp: Measurement time; used to drive a daily temperature cycle.
        config: Project configuration, for sample rate and burst length.
        rng: Seeded random generator, so runs are reproducible.

    Returns:
        A dictionary matching the ESP32 JSON payload schema.
    """
    sample_rate = float(config.get("sensor.accel_sample_rate_hz"))
    burst_samples = int(config.get("sensor.accel_burst_samples"))
    soil_dry = int(config.get("sensor.soil_raw_dry"))
    soil_wet = int(config.get("sensor.soil_raw_wet"))
    soil_min = int(config.get("sensor.soil_raw_min"))
    soil_max = int(config.get("sensor.soil_raw_max"))

    # Time of day as a fraction, used for the daily temperature/humidity cycle.
    seconds_into_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    daily = math.sin(2 * math.pi * (seconds_into_day / 86400.0 - 0.25))

    # Soil sits near the middle of the calibrated range, drifting a little.
    soil_mid = (soil_dry + soil_wet) / 2.0
    soil_raw = int(round(soil_mid + rng.gauss(0.0, 60.0)))
    soil_raw = max(soil_min, min(soil_max, soil_raw))

    # Accelerometer burst: gravity on Z plus low-amplitude background noise.
    accel: list[list[float]] = []
    for index in range(burst_samples):
        t = index / sample_rate
        hum = 0.004 * math.sin(2 * math.pi * 50.0 * t)  # mains-coupled hum
        accel.append(
            [
                round(rng.gauss(0.0, 0.01) + hum, 5),
                round(rng.gauss(0.0, 0.01), 5),
                round(1.0 + rng.gauss(0.0, 0.01), 5),
            ]
        )

    return {
        "node_id": node_id,
        "timestamp": timestamp.replace(tzinfo=timezone.utc).isoformat(),
        "soil_raw": soil_raw,
        "accel": accel,
        "mag": [
            round(rng.gauss(22.0, 0.5), 3),
            round(rng.gauss(-8.0, 0.5), 3),
            round(rng.gauss(38.0, 0.5), 3),
        ],
        "temp_c": round(24.0 + 5.0 * daily + rng.gauss(0.0, 0.3), 2),
        "humidity_pct": round(min(100.0, max(0.0, 60.0 - 10.0 * daily + rng.gauss(0.0, 1.5))), 2),
        "pressure_hpa": round(1010.0 + rng.gauss(0.0, 0.8), 2),
    }


def make_invalid_payload() -> dict[str, Any]:
    """Build a payload that must be rejected, for checking validation.

    It breaks three rules at once: the ADC value is above the configured
    maximum, the accelerometer burst is empty, and the magnetometer vector has
    only two components.

    Returns:
        A deliberately invalid payload dictionary.
    """
    return {
        "node_id": "node_01",
        "timestamp": "2025-01-01T00:00:00Z",
        "soil_raw": 99999,
        "accel": [],
        "mag": [1.0, 2.0],
        "temp_c": 25.0,
        "humidity_pct": 55.0,
        "pressure_hpa": 1010.0,
    }


def post_payload(client: httpx.Client, url: str, payload: dict[str, Any]) -> httpx.Response:
    """POST a single payload to the ingest endpoint.

    Args:
        client: An open HTTP client.
        url: Full URL of the ingest endpoint.
        payload: The payload to send.

    Returns:
        The HTTP response.
    """
    return client.post(url, json=payload, timeout=30.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    config = load_config()
    host = config.get("backend.host")
    port = config.get("backend.port")
    default_url = "http://{}:{}".format(host, port)

    parser = argparse.ArgumentParser(
        description="Post fake ESP32 payloads to the backend.",
    )
    parser.add_argument("--url", default=default_url, help="Base URL of the API")
    parser.add_argument("--count", type=int, default=10, help="Readings to send per node")
    parser.add_argument(
        "--nodes",
        nargs="*",
        default=None,
        help="Node IDs to simulate (default: every node in config.yaml)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(config.get("sensor.reading_interval_s")),
        help="Simulated seconds between consecutive readings",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Real seconds to wait between posts (default: 0, send as fast as possible)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config.get("project.random_seed")),
        help="Random seed",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print payloads instead of sending")
    parser.add_argument("--invalid", action="store_true", help="Send one invalid payload and exit")
    return parser.parse_args(argv)


def _send_invalid(ingest_url: str, dry_run: bool) -> int:
    """Send (or print) the deliberately invalid payload.

    Args:
        ingest_url: Full URL of the ingest endpoint.
        dry_run: If True, print the payload instead of sending it.

    Returns:
        Process exit code.
    """
    payload = make_invalid_payload()
    if dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    with httpx.Client() as client:
        response = post_payload(client, ingest_url, payload)

    print("Invalid payload -> HTTP {} (422 is the expected result)".format(response.status_code))
    print(json.dumps(response.json(), indent=2)[:1200])
    return 0 if response.status_code == 422 else 1


def main(argv: list[str] | None = None) -> int:
    """Generate payloads and post them to the backend.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if any request failed.
    """
    args = parse_args(argv)
    config = load_config()
    ingest_url = args.url.rstrip("/") + "/ingest"

    if args.invalid:
        return _send_invalid(ingest_url, args.dry_run)

    node_ids = args.nodes if args.nodes else config.node_ids()
    rng = random.Random(args.seed)

    # Space the readings out backwards from now, so the newest is "just taken".
    end_time = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    start_time = end_time - timedelta(seconds=args.interval * max(args.count - 1, 0))

    sent = 0
    failed = 0
    with httpx.Client() as client:
        for step in range(args.count):
            timestamp = start_time + timedelta(seconds=args.interval * step)

            for node_id in node_ids:
                payload = make_payload(node_id, timestamp, config, rng)

                if args.dry_run:
                    # Trim the burst so the terminal output stays readable.
                    preview = dict(payload)
                    preview["accel"] = payload["accel"][:3] + [["...truncated..."]]
                    print(json.dumps(preview, indent=2))
                    sent += 1
                    continue

                response = post_payload(client, ingest_url, payload)
                if response.status_code == 201:
                    sent += 1
                else:
                    failed += 1
                    print(
                        "  {} {} -> HTTP {}: {}".format(
                            node_id, timestamp, response.status_code, response.text[:200]
                        )
                    )

            if args.delay > 0 and not args.dry_run:
                time.sleep(args.delay)

    action = "generated" if args.dry_run else "posted to {}".format(ingest_url)
    print("{} payloads {}, {} failed.".format(sent, action, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
