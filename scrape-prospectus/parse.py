from pathlib import Path
from typing import List

import pdfplumber
from rich import print


def extract_text(pdf_path: Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def process_all_pdfs(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/raw")

    for pdf_path in pdf_files:
        print(f"[bold cyan]Extracting:[/bold cyan] {pdf_path.name}")
        text = extract_text(pdf_path)

        out_path = out_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")

        print(f"[green]Saved →[/green] {out_path}")


def main() -> None:
    raw_dir = Path("data/raw")
    out_dir = Path("data/extracted")
    process_all_pdfs(raw_dir, out_dir)


if __name__ == "__main__":
    main()
