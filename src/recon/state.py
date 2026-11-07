from langchain_core.documents import Document
from pydantic import BaseModel, Field

from recon.models import Article


class AgentState(BaseModel):
    competitors: list[str] = Field(default_factory=list)
    articles: list[Article] = Field(default_factory=list)
    retrieved_docs: list[Document] = Field(default_factory=list)
    brief: str = ""
    next_agent: str = ""
