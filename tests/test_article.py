from datetime import datetime

import pytest
from pydantic import ValidationError

from recon.models import Article, Company

ACME = Company(company_name="Acme", domains=["acme.test"])


def test_article_requires_core_fields():
    with pytest.raises(ValidationError):
        Article.model_validate({"title": "Launch"})


def test_article_optional_fields_default():
    article = Article(competitor=ACME, title="Launch", url="https://acme.test/launch")
    assert article.summary == ""
    assert article.published is None


def test_article_parses_iso_published_date():
    article = Article.model_validate(
        {
            "competitor": {"company_name": "Acme", "domains": ["acme.test"]},
            "title": "Launch",
            "url": "https://acme.test/launch",
            "published": "2026-10-20T09:30:00",
        }
    )
    assert article.competitor == ACME
    assert article.published == datetime(2026, 10, 20, 9, 30)
