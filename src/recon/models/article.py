from datetime import datetime

from pydantic import BaseModel


class Article(BaseModel):
    competitor: str
    title: str
    url: str
    summary: str = ""
    published: datetime | None = None
