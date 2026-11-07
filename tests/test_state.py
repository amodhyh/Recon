from recon.models import Article, Company
from recon.state import AgentState

ACME = Company(company_name="Acme", domains=["acme.test"])


def test_state_defaults_are_empty():
    state = AgentState()
    assert state.competitors == []
    assert state.articles == []
    assert state.retrieved_docs == []
    assert state.brief == ""
    assert state.next_agent == ""


def test_state_defaults_are_not_shared_between_instances():
    a, b = AgentState(), AgentState()
    a.competitors.append("Acme")
    assert b.competitors == []


def test_state_accepts_articles():
    article = Article(competitor=ACME, title="Launch", url="https://acme.test")
    state = AgentState(articles=[article])
    assert state.articles[0].competitor.company_name == "Acme"
