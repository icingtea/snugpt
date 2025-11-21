from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from src.models.chunks import CollectionEnum

class GraphState(BaseModel):
    collection: Optional[CollectionEnum] = None
    prompt: Optional[str] = None

    memory: Annotated[List[Any], add_messages] = Field(default_factory=list)
    
    filters: Dict[str, Any] = Field(default_factory=dict)
    context: List[str] = Field(default_factory=list)

    response: Optional[str] = None
    error: Optional[str] = None
