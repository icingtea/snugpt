import nanoid
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import List


class CollectionEnum(StrEnum):
    ACADEMICS = "Academics"
    FACULTY = "Faculty"
    MENU = "Menu"
    STUDENTS = "Students"


class SchoolEnum(StrEnum):
    SHSS = "SHSS"
    SOE = "SOE"
    SNS = "SNS"
    SME = "SME"


class DepartmentEnum(StrEnum):
    ENG = "ENG"
    BIOTECH = "BIOTECH"
    PHY = "PHY"
    CHEM = "CHEM"
    MATH = "MATH"
    ECO = "ECO"
    CSE = "CSE"
    ECE = "ECE"
    MECH = "MECH"
    CHEM_ENG = "CHEM_ENG"
    CIVIL = "CIVIL"


class SourceEnum(StrEnum):
    DOCUMENT = "DOCUMENT"
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


class WeeklyMenuChunk(BaseChunk): ...


class AcademicDocumentChunk(BaseChunk):
    schools: List[SchoolEnum] = Field(default_factory=list)
    departments: List[DepartmentEnum] = Field(default_factory=list)
