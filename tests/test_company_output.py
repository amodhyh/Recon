import pytest
from pydantic import ValidationError

from recon.models import Article, Company, ReconOut

ACME = Company(company_name="Acme", domains=["acme.test"])
ARTICLE = Article(competitor=ACME, title="Launch", url="https://acme.test/launch")


def test_company_requires_name_and_a_domain():
    with pytest.raises(ValidationError):
        Company(company_name="", domains=["acme.test"])
    with pytest.raises(ValidationError):
        Company(company_name="Acme", domains=[])


def test_recon_out_accepts_valid_data():
    out = ReconOut(company=ACME, articles=[ARTICLE], summary="Short", confidence_score=0.8)
    assert out.confidence_score == 0.8


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_recon_out_rejects_out_of_range_confidence(score):
    with pytest.raises(ValidationError):
        ReconOut(company=ACME, articles=[], summary="x", confidence_score=score)


def test_recon_out_limits_articles_to_three():
    with pytest.raises(ValidationError):
        ReconOut(company=ACME, articles=[ARTICLE] * 4, summary="x", confidence_score=0.5)
