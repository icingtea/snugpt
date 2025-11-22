from typing import List
from pymongo import MongoClient
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.models.chunks import BaseChunk
from src.chunker import Chunker
from src.seeder import Seeder
from src.config import ENV_CONFIG
from src.models.chunks import FacultyChunk, SchoolEnum, DepartmentEnum

SCHOOL_MAP = {
    "School of Humanities and Social Sciences": SchoolEnum.SHSS,
    "School of Engineering": SchoolEnum.SOE,
    "School of Natural Sciences": SchoolEnum.SNS,
    "School of Management & Entrepreneurship": SchoolEnum.SME,
    "Management & Entrepreneurship": SchoolEnum.SME,
}


class FacultySeeder(Seeder):
    def __init__(
        self,
        client: MongoClient,
        path_to_read_from=Path("./data/extracted/faculty"),
        collection_name="Menu",
    ):
        super().__init__(client, path_to_read_from, collection_name)
        self.model = SentenceTransformer(ENV_CONFIG.EMBEDDING_MODEL)

    def get_chunks(self) -> List[FacultyChunk]:
        file_path = self.path_to_read_from / "menu.txt"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        embedding = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

        return [FacultyChunk(embedding=embedding.tolist(), document=text)]


if __name__ == "__main__":
    client = MongoClient(ENV_CONFIG.MONGODB_CONNECTION_STRING)

    ms = MenuSeeder(client)

    chunks = ms.get_chunks()

    ms.seed(chunks)
