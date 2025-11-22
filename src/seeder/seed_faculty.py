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

DEPARTMENT_MAP = {
    "Department of English": DepartmentEnum.ENG,
    "Department of Life Sciences": DepartmentEnum.BIOTECH,
    "Department of Physics": DepartmentEnum.PHY,
    "Department of Chemistry": DepartmentEnum.CHEM,
    "Department of Mathematics": DepartmentEnum.MATH,
    "Department of Economics": DepartmentEnum.ECO,
    "Department of Computer Science and Engineering": DepartmentEnum.CSE,
    "Department of Electrical Engineering": DepartmentEnum.ECE,
    "Department of Mechanical Engineering": DepartmentEnum.MECH,
    "Department of Chemical Engineering": DepartmentEnum.CHEM_ENG,
    "Department of Civil Engineering": DepartmentEnum.CIVIL,
}


class FacultySeeder(Seeder):
    def __init__(
        self,
        client: MongoClient,
        path_to_read_from=Path("./data/extracted/faculty"),
        collection_name="Faculty",
    ):
        super().__init__(client, path_to_read_from, collection_name)

    def get_chunks(self) -> List[FacultyChunk]:
        chunks: List[FacultyChunk] = []

        files = list(self.path_to_read_from.glob("*"))

        for file_path in files:
            if not file_path.is_file():
                continue

            print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            school_raw = lines[1].strip()
            dept_raw = lines[2].strip()

            school_enum = SCHOOL_MAP.get(school_raw)
            dept_enum = DEPARTMENT_MAP.get(dept_raw)

            schools = []
            departments = []

            if school_enum:
                schools.append(school_enum)

            if dept_enum:
                departments.append(dept_enum)

            embedding = self.model.encode(
                content, convert_to_numpy=True, normalize_embeddings=True
            )

            chunk = FacultyChunk(
                embedding=embedding.tolist(),
                document=content,
                schools=schools,
                departments=departments,
            )
            chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    client = MongoClient(ENV_CONFIG.MONGODB_CONNECTION_STRING)
    # client["snugpt"]["Faculty"].delete_many({})

    fs = FacultySeeder(client)

    chunks = fs.get_chunks()

    print(chunks[-1])

    fs.seed(chunks)
