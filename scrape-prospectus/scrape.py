from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import mimetypes
import re


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-")


def get_extension_from_url(url: str) -> str:
    guess = Path(url).suffix
    if guess:
        return guess

    mime = mimetypes.guess_type(url)[0]
    if mime:
        ext = mimetypes.guess_extension(mime)
        if ext:
            return ext

    return ".dat"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    dest.write_bytes(response.content)


def process_entry(entry: Dict[str, Any], base_dir: Path) -> None:
    title = entry.get("title", "unknown")
    folder = base_dir / slugify(title)
    folder.mkdir(parents=True, exist_ok=True)

    for field in ("brochure", "prospectus"):
        url: Optional[str] = entry.get(field)
        if not url:
            continue

        ext = get_extension_from_url(url)
        filename = f"{field}{ext}"
        dest = folder / filename

        download_file(url, dest)


def main() -> None:
    data_path = Path("data/links.json")
    raw_dir = Path("data/raw")

    entries: list[Dict[str, Any]] = json.loads(data_path.read_text())

    for entry in entries:
        process_entry(entry, raw_dir)


if __name__ == "__main__":
    main()
