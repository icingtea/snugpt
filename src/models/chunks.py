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
    ENG = "ENGLISH"
    BIOTECH = "BIOTECHNOLOGY"
    PHY = "PHYSICS"
    CHEM = "CHEMISTRY"
    MATH = "MATHEMATICS"
    ECO = "ECONOMICS"
    CSE = "COMPUTER SCIENCE AND ENGINEERING"
    ECE = "ELECTRICAL AND COMPUTER ENGINEERING"
    MECH = "MECHANICAL ENGINEERING"
    CHEM_ENG = "CHEMICAL ENGINEERING"
    CIVIL = "CIVIL ENGINEERING"


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


class WeeklyMenuChunk(BaseChunk): ...


class AcademicDocumentChunk(BaseChunk):
    schools: List[SchoolEnum] = Field(default_factory=list)
    departments: List[str] = Field(default_factory=list)
