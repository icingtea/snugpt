from typing import List
from pymongo import MongoClient
from pathlib import Path

from src.chunker import Chunker
from src.seeder import Seeder
from src.config import ENV_CONFIG
from src.models.chunks import StudentDocumentChunk, SourceEnum


class StudentsSeeder(Seeder):
    def __init__(
        self,
        client: MongoClient,
        path_to_read_from=Path("./data/extracted/students"),
        collection_name="Students",
    ):
        super().__init__(client, path_to_read_from, collection_name)

    def get_chunks(self) -> List[StudentDocumentChunk]:
        chunks: List[StudentDocumentChunk] = []

        files = list(self.path_to_read_from.glob("*.txt"))

        for file_path in files:
            if not file_path.is_file():
                continue

            print(f"Processing {file_path}...")

            file_name_lower = file_path.stem.lower()

            if "email" in file_name_lower:
                source = SourceEnum.EMAIL
            else:
                source = SourceEnum.DOCUMENT

            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            sub_chunks = Chunker.chunk_from_text(original_content)

            for sub in sub_chunks:
                embedding = self.model.encode(
                    sub, convert_to_numpy=True, normalize_embeddings=True
                )

                chunk = StudentDocumentChunk(
                    embedding=embedding.tolist(),
                    document=sub,
                    source=source,
                )
                chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    client = MongoClient(ENV_CONFIG.MONGODB_CONNECTION_STRING)
    # client["snugpt"]["Faculty"].delete_many({})

    ss = StudentsSeeder(client)

    chunks = ss.get_chunks()

    print(chunks[-1])

    ss.seed(chunks)
