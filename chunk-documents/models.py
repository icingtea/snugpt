import nanoid
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import List


class SchoolEnum(StrEnum):
    SHSS = "SHSS"
    SOE = "SOE"
    SNS = "SNS"
    SME = "SME"


class DepartmentEnum(StrEnum):
    ENG = "ENGLISH"
    BIOTECH = "BIOTECHNOLOGY"
    CHEM = "CHEMISTRY"
    ECO = "ECONOMICS"
    MATH = "MATHEMATICS"


class SourceEnum(StrEnum):
    STUDENT_HANDBOOK = "STUDENT_HANDBOOK"
    SC_CONSTITUTION = "STUDENT_COUNCIL_CONSTITUTION"
    EMAIL = "EMAIL"


class BaseChunk(BaseModel):
    chunk_id: str = Field(default_factory=nanoid.generate)
    embedding: List[float] = Field(default_factory=list)
    document: str


class FacultyChunk(BaseChunk):
    schools: List[SchoolEnum] = Field(default_factory=list)
    departments: List[DepartmentEnum] = Field(default_factory=list)


class StudentDocumentChunk(BaseChunk):
    source: SourceEnum


class WeeklyMenuChunk(BaseChunk):
    start_date: str


class AcademicDocumentChunk(BaseChunk):
    schools: List[SchoolEnum] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
