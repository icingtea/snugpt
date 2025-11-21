from pydantic import BaseModel
from typing import Optional
from src.models.chunks import CollectionEnum

class GraphState(BaseModel):
    collection: Optional[CollectionEnum]
    prompt: Optional[str]
    