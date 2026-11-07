from datetime import datetime
from typing import Annotated
from pydantic import BaseModel,Field
from .company import Company


class Article(BaseModel):
    competitor: Annotated[Company,Field(description="Competing Company")]
    title: str
    url: str
    summary: str = ""
    published: datetime | None = None
