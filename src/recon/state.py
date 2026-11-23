from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Annotated, Sequence
from recon.models import Article
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import operator

def merge_unique_strings(current: list[str], update: list[str]) -> list[str]:
    """Appends new competitors without duplicates."""
    seen = set(current)
    return current + [item for item in update if item not in seen]

class AgentState(BaseModel):
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)
    competitors: Annotated[list[str], merge_unique_strings] = Field(default_factory=list)
    articles: Annotated[list[Article], operator.add] = Field(default_factory=list)
    retrieved_docs: Annotated[list[Document], operator.add] = Field(default_factory=list) 
    next_agent: str = ""                                                                                        
    brief: str = ""         
    #Loop safety guardrails
    iteration_count: int = 0
    error_message: str | None = None
