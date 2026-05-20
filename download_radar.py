from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin
import re

import requests


def create_archive_folders(root_folder: Path, days: int = 7) -> None:
    """Create one folder per day for the radar archive."""
    for day_offset in range(days):
        day = date.today() - timedelta(days=day_offset)
        folder = root_folder / day.isoformat()
        folder.mkdir(parents=True, exist_ok=True)


def parse_latest_radar_image_url(page_url: str, site_code: str, fallback_url: str = None) -> str:
    """Parse the radar page and return the latest available radar PNG URL."""
    response = requests.get(page_url, timeout=20)
    response.raise_for_status()
    page_text = response.text

    pattern = re.compile(r'["\'](/radar/%s\.T\.(\d+)\.png)["\']' % re.escape(site_code))
    matches = pattern.findall(page_text)
    if matches:
        latest_path = max(matches, key=lambda item: item[1])[0]
        return urljoin(page_url, latest_path)

    if fallback_url:
        return fallback_url

    raise ValueError(f"Unable to parse live radar URL from {page_url}")


def fetch_live_radar_image(image_url: str, archive_root: Path, days: int = 7) -> Path:
    """Download the current radar image and store it in today's archive folder."""
    archive_root = Path(archive_root)
    create_archive_folders(archive_root, days)

    today_folder = archive_root / date.today().isoformat()
    today_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_path = today_folder / f"radar_{timestamp}.png"

    response = requests.get(image_url, timeout=20)
    response.raise_for_status()

    save_path.write_bytes(response.content)
    return save_path


def fetch_live_radar_from_page(page_url: str, site_code: str, archive_root: Path, days: int = 7) -> Path:
    """Download the latest live radar image by parsing the radar page."""
    image_url = parse_latest_radar_image_url(page_url, site_code)
    return fetch_live_radar_image(image_url, archive_root, days)


def list_archive_images(archive_root: Path):
    """List available radar images in the archive."""
    archive_root = Path(archive_root)
    if not archive_root.exists():
        return []
    return sorted(archive_root.rglob("*.png"))
