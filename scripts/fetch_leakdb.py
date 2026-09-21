r"""Download and extract the LeakDB sample scenarios (Hanoi network, 10 scenarios).

LeakDB (KIOS Research Center) is a real, published benchmark for leak
detection in water distribution networks - not accelerometer data (nobody
publishes that; utilities don't instrument pipes with vibration sensors the
way this project does), but real simulated pressure/flow data with labelled
leak episodes, used here as an independent, real-data sanity check.

Run: .venv\Scripts\python scripts\fetch_leakdb.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import load_config  # noqa: E402


def download(url: str, dest: Path) -> None:
    """Stream a URL to a local file, skipping if it already exists.

    Args:
        url: The file to download.
        dest: Local destination path.
    """
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Downloaded {dest.stat().st_size / 1e6:.1f} MB")


def extract(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file, skipping if the target directory already has content.

    Args:
        zip_path: The downloaded zip file.
        extract_dir: Directory to extract into.
    """
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Already extracted: {extract_dir}")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print("Done.")


def main() -> int:
    config = load_config()
    zip_path = config.resolve_path(config.get("leakdb.zip_path"))
    extract_dir = config.resolve_path(config.get("leakdb.extract_dir"))

    download(config.get("leakdb.source_url"), zip_path)
    extract(zip_path, extract_dir)

    scenarios = sorted(p.name for p in extract_dir.iterdir() if p.is_dir())
    print(f"Available scenarios: {scenarios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
