from typing import List
from pymongo import MongoClient
from pathlib import Path

from src.chunker import Chunker
from src.seeder import Seeder
from src.config import ENV_CONFIG
from src.models.chunks import AcademicDocumentChunk, SchoolEnum, DepartmentEnum


class ProspectusSeeder(Seeder):
    def __init__(
        self,
        client: MongoClient,
        path_to_read_from=Path("./data/extracted/academics"),
        collection_name="Academics",
    ):
        super().__init__(client, path_to_read_from, collection_name)

    def get_chunks(self) -> List[AcademicDocumentChunk]:
        chunks: List[AcademicDocumentChunk] = []

        files = list(self.path_to_read_from.glob("*-PROSPECTUS.txt"))

        for file_path in files:
            if not file_path.is_file():
                continue

            print(f"Processing {file_path}...")

            name_parts = file_path.stem.split("-")
            school_part = name_parts[0]
            department_part = name_parts[1]

            schools = [SchoolEnum(school_part)]
            departments = [DepartmentEnum(department_part)]

            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            header = f"School: {school_part}\nDepartment: {department_part}\n\n"

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

    ps = ProspectusSeeder(client)

    chunks = ps.get_chunks()

    print(chunks[-1])

    ps.seed(chunks)
