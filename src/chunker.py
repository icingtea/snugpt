from pathlib import Path
from typing import List


class Chunker:
    @staticmethod
    def chunk_from_file(path: Path) -> List[str]:
        text = path.read_text(encoding="utf-8")
        return Chunker.chunk_from_text(text)

    @staticmethod
    def chunk_from_text(text: str, chunk_size: int = 16384) -> List[str]:
        overlap = chunk_size // 3

        chunks: List[str] = []

        start = 0
        while start < len(text):
            left = max(0, start - overlap)
            right = min(len(text), start + chunk_size + overlap)

            chunk = text[left:right]
            chunks.append(chunk)

            start += chunk_size

        return chunks
