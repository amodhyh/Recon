from typing import Annotated

from pydantic import BaseModel, Field


class Company(BaseModel):
    company_name: Annotated[
        str, Field(min_length=1, max_length=60, description="Company name")
    ]
    domains: Annotated[
        list[str], Field(min_length=1, max_length=30, description="Company domains")
    ]
