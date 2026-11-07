from typing import Annotated

from pydantic import BaseModel, Field

from recon.models.article import Article
from recon.models.company import Company


class ReconOut(BaseModel):
    company: Annotated[Company, Field(description="Competitor company")]
    articles: Annotated[
        list[Article],
        Field(max_length=3, description="Articles related to the company"),
    ]
    summary: Annotated[
        str, Field(max_length=200, description="Summary of the relevant articles")
    ]
    confidence_score: Annotated[
        float, Field(ge=0, le=1, description="Confidence score of the decision")
    ]
