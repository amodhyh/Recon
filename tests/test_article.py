from datetime import datetime

import pytest
from pydantic import ValidationError

from recon.models import Article


def test_article_requires_core_fields():
    with pytest.raises(ValidationError):
        Article(title="Launch")


def test_article_optional_fields_default():
    article = Article(competitor="Acme", title="Launch", url="https://acme.test/launch")
    assert article.summary == ""
    assert article.published is None


def test_article_parses_iso_published_date():
    article = Article(
        competitor="Acme",
        title="Launch",
        url="https://acme.test/launch",
        published="2026-10-20T09:30:00",
    )
    assert article.published == datetime(2026, 10, 20, 9, 30)
