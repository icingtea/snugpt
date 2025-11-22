from typing import List
from pymongo import MongoClient
from pathlib import Path

from src.chunker import Chunker
from src.seeder import Seeder
from src.config import ENV_CONFIG
from src.models.chunks import AcademicDocumentChunk, SchoolEnum, DepartmentEnum


class UGHandBookSeeder(Seeder):
    def __init__(
        self,
        client: MongoClient,
        path_to_read_from=Path("./data/extracted/academics"),
        collection_name="Academics",
    ):
        super().__init__(client, path_to_read_from, collection_name)

    def get_chunks(self) -> List[AcademicDocumentChunk]:
        chunks: List[AcademicDocumentChunk] = []

        file_path = self.path_to_read_from / "UG_HANDBOOK.txt"

        print(f"Processing {file_path}...")

        schools = list(SchoolEnum)
        departments = list(DepartmentEnum)

        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        header_lines = [
            "Schools: " + ", ".join(s.value for s in schools),
            "Departments: " + ", ".join(d.value for d in departments),
            "",
        ]
        header = "\n".join(header_lines) + "\n"

        sub_chunks = Chunker.chunk_from_text(original_content)

        for sub in sub_chunks:
            chunk_text = header + sub

            embedding = self.model.encode(
                chunk_text, convert_to_numpy=True, normalize_embeddings=True
            )

            chunk = AcademicDocumentChunk(
                embedding=embedding.tolist(),
                document=chunk_text,
                schools=schools,
                departments=departments,
            )
            chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    client = MongoClient(ENV_CONFIG.MONGODB_CONNECTION_STRING)
    # client["snugpt"]["Faculty"].delete_many({})

    ugs = UGHandBookSeeder(client)

    chunks = ugs.get_chunks()

    print(chunks[-1])

    ugs.seed(chunks)
