from __future__ import annotations

import json
import re
import mimetypes
import pdfplumber
import requests
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List


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
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            print(f"Skipping (status {response.status_code}): {url}")
            return
        dest.write_bytes(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def detect_file_type(path: Path) -> Optional[str]:
    name = path.stem.lower()
    if "prospectus" in name:
        return "prospectus"
    return None


def to_slug_capital_case(slug: str) -> str:
    parts = slug.split("-")
    capped = [p.capitalize() for p in parts]
    return "-".join(capped)


PREFIX_MAP = {
    "b-tech": "BTech",
    "b-sc": "BSC",
    "b-a": "BA",
    "ba": "BA",
    "bachelor-in-design": "BDES",
    "bachelors-in-design": "BDES",
    "bachelor-of-management-studies": "BMS",
}


def detect_prefix(dirname: str):
    parts = dirname.split("-")
    first_two = "-".join(parts[:2])
    if first_two in PREFIX_MAP:
        return PREFIX_MAP[first_two], "-".join(parts[2:])
    if parts[0] in PREFIX_MAP:
        return PREFIX_MAP[parts[0]], "-".join(parts[1:])
    return None, dirname


def extract_text(pdf_path: Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def rename_and_flatten(raw_root: Path) -> None:
    for program_dir in raw_root.iterdir():
        if not program_dir.is_dir():
            continue

        prefix, program_slug = detect_prefix(program_dir.name)
        if prefix is None:
            continue

        program_name = to_slug_capital_case(program_slug)

        for file in program_dir.iterdir():
            if not file.is_file() or file.suffix.lower() != ".pdf":
                continue

            file_type = detect_file_type(file)
            if file_type is None:
                continue

            new_name = f"{prefix}_{program_name}_{file_type}.pdf"
            dest = raw_root / new_name
            shutil.move(str(file), str(dest))

        try:
            program_dir.rmdir()
        except OSError:
            pass


def process_all_pdfs(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        text = extract_text(pdf_path)
        out_path = out_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved → {out_path}")


def main() -> None:
    json_path = Path("data/prospectus-links.json")
    raw_base = Path("data/raw/prospectus")
    extract_base = Path("data/extracted/academics")

    entries: list[Dict[str, Any]] = json.loads(json_path.read_text())

    for entry in entries:
        title = entry.get("title", "unknown")
        folder = raw_base / slugify(title)
        folder.mkdir(parents=True, exist_ok=True)

        prospectus_url: Optional[str] = entry.get("prospectus")
        if not prospectus_url:
            continue

        ext = get_extension_from_url(prospectus_url)
        filename = f"prospectus{ext}"
        dest = folder / filename

        print(f"Downloading {prospectus_url}")
        download_file(prospectus_url, dest)

    rename_and_flatten(raw_base)
    process_all_pdfs(raw_base, extract_base)


if __name__ == "__main__":
    main()
