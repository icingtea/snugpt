from typing import List
from pymongo import MongoClient
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.models.chunks import BaseChunk
from src.chunker import Chunker
from src.seeder import Seeder
from src.config import ENV_CONFIG
from src.models.chunks import WeeklyMenuChunk


class MenuSeeder(Seeder):
    def __init__(
        self,
        client: MongoClient,
        path_to_read_from=Path("./data/extracted/menu"),
        collection_name="Menu",
    ):
        super().__init__(client, path_to_read_from, collection_name)

    def get_chunks(self) -> List[WeeklyMenuChunk]:
        file_path = self.path_to_read_from / "menu.txt"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        embedding = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

        return [WeeklyMenuChunk(embedding=embedding.tolist(), document=text)]


if __name__ == "__main__":
    client = MongoClient(ENV_CONFIG.MONGODB_CONNECTION_STRING)
    client["snugpt"]["Menu"].delete_many({})

    ms = MenuSeeder(client)

    chunks = ms.get_chunks()

    ms.seed(chunks)
