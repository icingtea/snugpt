from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class Chunker(ABC):
    @staticmethod
    @abstractmethod
    def chunk_from_file(path_to_file: Path) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def chunk_from_text(text: str) -> List[str]:
        pass
