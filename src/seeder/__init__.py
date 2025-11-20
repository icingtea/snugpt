from abc import ABC, abstractmethod
from pathlib import Path
from pymongo import MongoClient
from typing import List

from models.chunks import BaseChunk


class Seeder(ABC):
    def __init__(
        self, client: MongoClient, path_to_read_from: Path, collection_name: str
    ):
        self.path_to_read_from = path_to_read_from
        self.client = client
        self.collection_name = collection_name

    @abstractmethod
    def get_chunks(self) -> List[BaseChunk]:
        pass

    def seed(self, chunks: List[BaseChunk]):
        db = self.client["snugpt"]
        collection = db[self.collection_name]

        docs = [c.model_dump() for c in chunks]
        collection.insert_many(docs)
