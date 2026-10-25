from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from recon.models import Article


class AgentState(BaseModel):
    history: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    articles: list[Article] = Field(default_factory=list)
    retrieved_docs: list[Document] = Field(default_factory=list)
    brief: str = ""
    next_agent: str = ""
